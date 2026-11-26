"""Matched fixed diagnostics and verified optimizer continuation, without model changes."""

from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .family_evaluation import quantiles
from .io import save_csv, save_json, save_jsonl
from .raw_lineage import sha256_file


def state_digest(value):
    digest = hashlib.sha256()

    def add(obj):
        if isinstance(obj, torch.Tensor):
            a = obj.detach().cpu().contiguous()
            digest.update(str((str(a.dtype), tuple(a.shape))).encode())
            digest.update(a.numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=lambda x: str(x)):
                digest.update(str(key).encode())
                add(obj[key])
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                add(item)
        else:
            digest.update(repr(obj).encode())

    add(value)
    return digest.hexdigest()


def replay_sampler(seed, length, batch_size, epochs):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(range(length)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for _ in range(epochs):
        for _ in loader:
            pass
    return generator


def select_monitor(
    train_rows, validation_rows, selection_seed, families, pairs_per_family
):
    groups = defaultdict(list)
    for index, row in enumerate(train_rows):
        groups[row["family_id"]].append(index)
    eligible = sorted(
        key for key, ids in groups.items() if len(ids) >= pairs_per_family
    )
    if families * pairs_per_family != len(validation_rows) or families > len(eligible):
        raise ValueError(
            "Monitor and validation sizes or family availability do not match"
        )
    rng = np.random.default_rng(selection_seed)
    chosen_families = sorted(rng.choice(eligible, families, replace=False).tolist())
    indices = []
    for family in chosen_families:
        indices.extend(
            sorted(rng.choice(groups[family], pairs_per_family, replace=False).tolist())
        )
    if {train_rows[i]["family_id"] for i in indices} & {
        r["family_id"] for r in validation_rows
    }:
        raise ValueError("Monitor overlaps held-out source families")
    return indices


def freeze_monitor(config, parent, train_rows, val_rows):
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError(
            "Use a new diagnostic output; frozen monitor cannot be replaced"
        )
    parent_root = Path(config["parent_root"])
    source_plans = [
        json.loads(line)
        for line in (parent_root / "frozen_validation.jsonl").read_text().splitlines()
    ]
    if [p["sample_id"] for p in source_plans] != [r["sample_id"] for r in val_rows]:
        raise ValueError("Validation plans and native records differ")
    indices = select_monitor(
        train_rows,
        val_rows,
        config["monitor_selection_seed"],
        config["monitor_families"],
        config["monitor_pairs_per_family"],
    )
    plans = []
    for index, reference in zip(indices, source_plans):
        row = train_rows[index]
        plan = copy.deepcopy(reference)
        plan.update(
            sample_id=row["sample_id"],
            family_id=row["family_id"],
            split="train",
            paired_validation_sample_id=reference["sample_id"],
            paired_validation_family_id=reference["family_id"],
            plan_source="identical affine assigned from existing validation plan; native image content differs",
        )
        plans.append(plan)
    root.mkdir(parents=True)
    save_json(root / "preregistered_diagnostic.json", config)
    save_jsonl(root / "monitor_samples.jsonl", [train_rows[i] for i in indices])
    save_jsonl(root / "monitor_transforms.jsonl", plans)
    save_jsonl(root / "validation_samples.jsonl", val_rows)
    # Keep validation plan bytes, not just numeric equivalence.
    (root / "validation_transforms.jsonl").write_bytes(
        (parent_root / "frozen_validation.jsonl").read_bytes()
    )
    failures = json.loads(Path(config["failure_pair_source"]).read_text())
    save_json(root / "fixed_failure_pairs.json", failures)
    hashes = {
        name: sha256_file(root / name)
        for name in [
            "preregistered_diagnostic.json",
            "monitor_samples.jsonl",
            "monitor_transforms.jsonl",
            "validation_samples.jsonl",
            "validation_transforms.jsonl",
            "fixed_failure_pairs.json",
        ]
    }
    exposed = {
        r["id"]
        for r in json.loads(Path(parent["exposed32_manifest"]).read_text())["samples"]
    }
    train_lookup = {r["sample_id"]: r for r in train_rows}
    save_json(
        root / "monitor_freeze.json",
        dict(
            hashes=hashes,
            source_checkpoint_sha256=sha256_file(config["resume_checkpoint"]),
            parent_family_freeze_sha256=sha256_file(
                parent_root / "freeze_receipt.json"
            ),
            parent_validation_plan_sha256=sha256_file(
                parent_root / "frozen_validation.jsonl"
            ),
            train_monitor_pairs=len(plans),
            train_monitor_families=len({r["family_id"] for r in plans}),
            validation_pairs=len(val_rows),
            validation_families=len({r["family_id"] for r in val_rows}),
            paired_affine_multiset_exact=True,
            source_family_disjoint=True,
            previous32_sample_ids_in_monitor=sum(
                p["sample_id"] in exposed for p in plans
            ),
            sample_selection="fixed RNG over eligible source families then native pairs; no model scores",
            model_results_seen_before_monitor_freeze=False,
            architecture_changed=False,
            training_config_changed=False,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    return indices


def verify_monitor(root):
    root = Path(root)
    receipt = json.loads((root / "monitor_freeze.json").read_text())
    for name, expected in receipt["hashes"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Frozen diagnostic file changed: {name}")
    return receipt


class FrozenDiagnosticView(Dataset):
    def __init__(self, base, indices, plans):
        self.base = base
        self.indices = list(indices)
        self.plans = plans
        self.records = [base.records[i] for i in self.indices]
        self.split = base.split
        if [r["sample_id"] for r in self.records] != [p["sample_id"] for p in plans]:
            raise ValueError("Fixed diagnostic plans do not match samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        optical, sar_base = self.base.arrays[self.indices[index]]
        plan = self.plans[index]
        matrix = np.asarray(plan["optical_input_to_sar_input"])
        sar = cv2.warpAffine(
            sar_base,
            matrix[:2],
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = (
            cv2.warpAffine(
                np.ones((256, 256), "float32"),
                matrix[:2],
                (256, 256),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            >= 1 - 1e-6
        )
        return dict(
            id=plan["sample_id"],
            family_id=plan["family_id"],
            optical=torch.from_numpy(optical.copy())[None].float() / 127.5 - 1,
            sar=torch.from_numpy(sar.copy())[None].float() / 127.5 - 1,
            matrix=torch.tensor(matrix, dtype=torch.float32),
            source_support=torch.ones(1, 256, 256, dtype=torch.bool),
            target_support=torch.from_numpy(valid)[None],
        )


def aggregate_families(output, tag):
    output = Path(output)
    import gzip

    rows = [
        json.loads(line)
        for line in gzip.open(output / f"{tag}_per_pair.jsonl.gz", "rt")
    ]
    data = np.load(output / f"{tag}_distances.npz")
    if list(data["sample_id"]) != [r["sample_id"] for r in rows]:
        raise ValueError("Pair ordering differs from stored distance rows")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["family_id"]].append(i)
    result = []
    for family, indices in sorted(groups.items()):
        members = [rows[i] for i in indices]
        n = sum(r["count"] for r in members)
        distances = data["nearest_distance_px"][indices][data["supervised"][indices]]
        result.append(
            dict(
                family_id=family,
                pairs=len(indices),
                effective_source_points=n,
                recall_at_1=(
                    sum(r["recall1_correct"] for r in members) / n if n else None
                ),
                recall_at_4=(
                    sum(r["recall4_correct"] for r in members) / n if n else None
                ),
                median_distance_px=float(np.median(distances)) if n else None,
                p95_distance_px=float(np.percentile(distances, 95)) if n else None,
                macro_pair_recall_at_1=(
                    float(np.mean([r["recall_at_1"] for r in members if r["count"]]))
                    if n
                    else None
                ),
                macro_pair_recall_at_4=(
                    float(np.mean([r["recall_at_4"] for r in members if r["count"]]))
                    if n
                    else None
                ),
            )
        )
    save_csv(output / f"{tag}_per_family.csv", result)
    summary = dict(
        families=len(result),
        family_distributions={
            key: quantiles([r[key] for r in result if r[key] is not None])
            for key in [
                "recall_at_1",
                "recall_at_4",
                "median_distance_px",
                "p95_distance_px",
            ]
        },
    )
    save_json(output / f"{tag}_family_summary.json", summary)
    return summary
