"""Source-family components, fixed folds and guarded native sample selection."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .io import save_csv, save_json, save_jsonl
from .raw_lineage import sha256_file


class Components:
    def __init__(self, ids):
        self.parent = {name: name for name in ids}

    def find(self, name):
        if self.parent[name] != name:
            self.parent[name] = self.find(self.parent[name])
        return self.parent[name]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_sources(rows, exposed_ids, known_edges=()):
    """All known source-equivalent members and overlap relations are transitive."""
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate source sample IDs")
    if not set(exposed_ids) <= by_id.keys():
        raise ValueError("Previously exposed training IDs are absent from inventory")
    graph = Components(by_id)
    evidence = []
    missing_parents = set()

    def edge(a, b, kind, detail=None):
        if a not in by_id or b not in by_id:
            raise ValueError("Verified grouping edge references absent sample")
        graph.join(a, b)
        evidence.append(dict(first=a, second=b, kind=kind, detail=detail))

    for row in rows:
        if row["optical"]["width"] == 256:
            parents = row.get("exact_pixel_crop_parents", [])
            if not parents:
                missing_parents.add(row["id"])
            for parent in parents:
                if parent["parent_id"] not in by_id:
                    missing_parents.add(row["id"])
                    continue
                edge(
                    row["id"],
                    parent["parent_id"],
                    "joint_optical_SAR_exact_quadrant",
                    dict(x=parent["x"], y=parent["y"], width=256, height=256),
                )
    for modality in ["joint", "optical", "sar"]:
        lookup = {}
        for row in rows:
            rec = row["optical"]
            if modality == "joint":
                key = (
                    rec["width"],
                    rec["height"],
                    rec["channels"],
                    row["sar"]["channels"],
                    row.get("joint_pixel_sha256"),
                )
            else:
                rec = row[modality]
                # A uniform single-modality image alone is not source evidence.
                if max(rec["maximum"]) <= min(rec["minimum"]):
                    continue
                key = (
                    rec["width"],
                    rec["height"],
                    rec["channels"],
                    tuple(rec["dtype"]),
                    rec["pixel_sha256"],
                )
            if key[-1] is None:
                continue
            if key in lookup:
                edge(
                    lookup[key],
                    row["id"],
                    "exact_" + modality + "_pixels",
                    "conservative content-equivalence grouping, not geolocation proof",
                )
            else:
                lookup[key] = row["id"]
    for item in known_edges:
        if not item.get("evidence"):
            raise ValueError("A larger-scene/overlap link requires explicit evidence")
        edge(
            item["first"],
            item["second"],
            "verified_larger_scene_or_overlap",
            item["evidence"],
        )
    groups = defaultdict(list)
    for name in sorted(by_id):
        groups[graph.find(name)].append(name)
    edge_kinds = defaultdict(set)
    for item in evidence:
        edge_kinds[graph.find(item["first"])].add(item["kind"])
    assignments, families = [], []
    for root, members in sorted(groups.items()):
        family_id = "OSF-" + hashlib.sha256("|".join(members).encode()).hexdigest()[:16]
        folds = {by_id[name]["official_split"] for name in members} - {"unspecified"}
        anchored = sorted(set(members) & set(exposed_ids))
        incomplete = any(
            name in missing_parents or not by_id[name]["eligible_by_frozen_file_rules"]
            for name in members
        )
        if anchored:
            split, reason = (
                "train",
                "previously_fitted32_and_all_source_relatives_forced_to_training",
            )
            if incomplete:
                # Keep exposure quarantined from held-out folds and reject incomplete members at selection.
                reason += "; incomplete_source_members_not_selected"
        elif incomplete:
            split, reason = "quarantine", "incomplete_or_unresolved_parent_relationship"
        elif len(folds) == 1:
            split, reason = (
                next(iter(folds)),
                "whole_family_preserves_single_official_sample_fold",
            )
        elif len(folds) > 1:
            split, reason = (
                "quarantine",
                "family_crosses_official_folds; do_not_resolve_by_random_or_majority_assignment",
            )
        else:
            split, reason = (
                "quarantine",
                "no_official_fold_and_not_part_of_frozen_native256_protocol",
            )
        parents_512 = sorted(
            name for name in members if by_id[name]["optical"]["width"] == 512
        )
        families.append(
            dict(
                family_id=family_id,
                split=split,
                members=members,
                member_count=len(members),
                parent_512_ids=parents_512,
                official_folds=sorted(folds),
                exposure_anchor_ids=anchored,
                grouping_evidence=sorted(edge_kinds[root])
                or ["native512_self_root_no_cross_file_link_identified"],
                reason=reason,
                geographic_independence="unknown_remaining_spatial_relations",
                complete_parent_scene_id=None,
            )
        )
        for name in members:
            row = by_id[name]
            size = row["optical"]["width"]
            candidates = (
                [name]
                if size == 512
                else sorted(
                    {
                        p["parent_id"]
                        for p in row.get("exact_pixel_crop_parents", [])
                        if p["parent_id"] in by_id
                    }
                )
            )
            assignments.append(
                dict(
                    sample_id=name,
                    native_size=size,
                    release_subset=row["subset"],
                    release_version="OSDataset2.0_V1_including_legacy_OS_256_and_512_and_13_region_extension",
                    parent_512_id=candidates[0] if len(candidates) == 1 else None,
                    parent_512_ids=candidates,
                    family_id=family_id,
                    split=split,
                    grouping_evidence=sorted(edge_kinds[root])
                    or ["native512_self_root"],
                    official_split=row["official_split"],
                    split_reason=reason,
                    previously_fitted32=name in exposed_ids,
                    source_eligible=row["eligible_by_frozen_file_rules"],
                    geographic_independence="not_claimed",
                )
            )
    assignments.sort(key=lambda row: row["sample_id"])
    return assignments, families, evidence


def choose_native_representatives(rows, assignments):
    mapping = {row["sample_id"]: row for row in assignments}
    selected = []
    seen = set()
    for row in sorted(rows, key=lambda row: row["id"]):
        assigned = mapping[row["id"]]
        if (
            assigned["split"] not in ["train", "val", "test"]
            or row["optical"]["width"] != 256
        ):
            continue
        if not row["eligible_by_frozen_file_rules"]:
            continue
        if not all(
            max(row[mod]["maximum"]) > min(row[mod]["minimum"])
            for mod in ["optical", "sar"]
        ):
            continue
        key = row["joint_pixel_sha256"]
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            dict(
                sample_id=row["id"],
                family_id=assigned["family_id"],
                split=assigned["split"],
                native_size=256,
                optical_path=row["optical"]["path"],
                sar_path=row["sar"]["path"],
                optical_sha256=row["optical"]["sha256"],
                sar_sha256=row["sar"]["sha256"],
                joint_pixel_sha256=key,
            )
        )
    return selected


def freeze_families(config):
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError("Use a new output for a distinct family/split version")
    rows = [
        json.loads(line)
        for line in Path(config["audit_manifest"]).read_text().splitlines()
    ]
    exposed = json.loads(Path(config["exposed32_manifest"]).read_text())
    exposed_ids = {row["id"] for row in exposed["samples"]}
    overlap = json.loads(Path(config["known_overlap_evidence"]).read_text())
    assignments, families, evidence = group_sources(
        rows, exposed_ids, overlap["verified_edges"]
    )
    representatives = choose_native_representatives(rows, assignments)
    split_groups = {
        split: {row["family_id"] for row in representatives if row["split"] == split}
        for split in ["train", "val", "test"]
    }
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if split_groups[a] & split_groups[b]:
            raise AssertionError("Source family crosses splits")
    for row in assignments:
        if row["previously_fitted32"] and row["split"] != "train":
            raise AssertionError("Exposed sample escaped training")
    output.mkdir(parents=True)
    save_json(output / "preregistered_protocol.json", config)
    save_jsonl(output / "sample_manifest.jsonl", assignments)
    save_jsonl(output / "families.jsonl", families)
    save_jsonl(output / "grouping_edges.jsonl", evidence)
    flat = [
        {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
            for k, v in row.items()
        }
        for row in assignments
    ]
    save_csv(output / "sample_assignments.csv", flat)
    save_csv(
        output / "family_splits.csv",
        [
            dict(
                family_id=row["family_id"],
                split=row["split"],
                member_count=row["member_count"],
                parent_512_ids="|".join(row["parent_512_ids"]),
                exposure_anchor_count=len(row["exposure_anchor_ids"]),
                reason=row["reason"],
            )
            for row in families
        ],
    )
    for split in ["train", "val", "test"]:
        save_jsonl(
            output / f"native256_{split}.jsonl",
            [row for row in representatives if row["split"] == split],
        )
    source_files = {
        name: sha256_file(config[key])
        for name, key in [
            ("audit_manifest", "audit_manifest"),
            ("exposed32_manifest", "exposed32_manifest"),
            ("known_overlap_evidence", "known_overlap_evidence"),
        ]
    }
    frozen_files = {
        name: sha256_file(output / name)
        for name in [
            "preregistered_protocol.json",
            "sample_manifest.jsonl",
            "families.jsonl",
            "grouping_edges.jsonl",
            "sample_assignments.csv",
            "family_splits.csv",
            "native256_train.jsonl",
            "native256_val.jsonl",
            "native256_test.jsonl",
        ]
    }
    summary = dict(
        total_release_pairs=len(rows),
        release_subset_counts=dict(Counter(row["subset"] for row in rows)),
        native_size_counts=dict(Counter(str(row["optical"]["width"]) for row in rows)),
        family_count=len(families),
        family_split_counts=dict(Counter(row["split"] for row in families)),
        sample_split_counts=dict(Counter(row["split"] for row in assignments)),
        unique_native256_split_counts=dict(
            Counter(row["split"] for row in representatives)
        ),
        exposed32_ids=len(exposed_ids),
        exposed32_family_count=len(
            {row["family_id"] for row in assignments if row["previously_fitted32"]}
        ),
        exposed32_all_training=True,
        source_family_overlap_across_splits=False,
        split_basis="source-family isolation; official sample folds retained at component level",
        exact_parent_file_ambiguous=sum(
            row["native_size"] == 256 and row["parent_512_id"] is None
            for row in assignments
        ),
        larger_scene_or_geographic_independence="unknown_not_claimed",
        native512_and_RGB_extension_not_used_for_native256_training=True,
        test_sealed=True,
        samples_deleted_to_match_paper_counts=False,
    )
    save_json(output / "grouping_summary.json", summary)
    save_json(
        output / "freeze_receipt.json",
        dict(
            source_files=source_files,
            frozen_files=frozen_files,
            summary=summary,
            validation_model_results_seen_before_freeze=False,
            future_augmentations_inherit_family_id=True,
        ),
    )
    return summary
