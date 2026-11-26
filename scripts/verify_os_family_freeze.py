"""Independent consistency checks over frozen family assignments and known links."""

import argparse
import json
from pathlib import Path

from sdgm.io import save_json
from sdgm.os_family_data import verify_freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_family_baseline.json")
    args = parser.parse_args()
    root = Path(json.loads(Path(args.config).read_text())["output"])
    verify_freeze(root)
    samples = [
        json.loads(line)
        for line in (root / "sample_manifest.jsonl").read_text().splitlines()
    ]
    by_id = {row["sample_id"]: row for row in samples}
    edges = [
        json.loads(line)
        for line in (root / "grouping_edges.jsonl").read_text().splitlines()
    ]
    assert len(samples) == len(by_id) == 17168
    families = {}
    for row in samples:
        families.setdefault(row["family_id"], set()).add(row["split"])
        assert all(
            by_id[p]["family_id"] == row["family_id"] for p in row["parent_512_ids"]
        )
        if row["previously_fitted32"]:
            assert row["split"] == "train"
    assert all(len(splits) == 1 for splits in families.values())
    assert all(
        by_id[e["first"]]["family_id"] == by_id[e["second"]]["family_id"] for e in edges
    )
    sets = {}
    for split in ["train", "val", "test"]:
        rows = [
            json.loads(line)
            for line in (root / f"native256_{split}.jsonl").read_text().splitlines()
        ]
        assert all(by_id[row["sample_id"]]["split"] == split for row in rows)
        assert len({row["joint_pixel_sha256"] for row in rows}) == len(rows)
        sets[split] = {row["family_id"] for row in rows}
    assert not (
        sets["train"] & sets["val"]
        or sets["train"] & sets["test"]
        or sets["val"] & sets["test"]
    )
    assert all(
        row["split"] == "quarantine"
        for row in samples
        if not row["release_subset"].startswith("OSdataset/")
    )
    root.joinpath("verification").mkdir(exist_ok=True)
    save_json(
        root / "verification/family_freeze_checks.json",
        dict(
            all_release_records_retained=True,
            release_samples=17168,
            every_known_edge_in_same_family=True,
            all_parent_aliases_in_same_family=True,
            all_exposed32_training=True,
            every_family_single_split=True,
            native_representatives_exact_content_unique=True,
            source_family_disjoint=True,
            unassigned_RGB512_quarantined=True,
            all_frozen_hashes_match=True,
            test_model_evaluation=False,
            residual_geographic_relations="unknown_not_claimed",
        ),
    )
    print("Family freeze verified")


if __name__ == "__main__":
    main()
