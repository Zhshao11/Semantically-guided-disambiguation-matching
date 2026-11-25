"""Freeze raw identity and geographic split before inspecting training crops."""

import argparse
import json
from pathlib import Path

import cv2

from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.raw_lineage import (
    corroborate_coverage,
    group_scenes,
    inventory_raw,
    sha256_file,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/raw_rebuild.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    catalog = json.loads(Path(config["catalog"]).read_text())
    rows = inventory_raw(catalog, config["raw_root"])
    save_jsonl(output / "raw_manifest.jsonl", rows)
    optical = {
        row["parent_image_id"]: row["path"]
        for row in rows
        if row["modality"] == "optical"
    }
    coverage = corroborate_coverage(
        optical, {row["parent_image_id"] for row in catalog}, config
    )
    save_json(output / "coverage_evidence.json", coverage)
    scene_rows, splits, summary = group_scenes(rows, coverage, config)
    save_csv(output / "scene_groups.csv", scene_rows)
    save_csv(output / "splits.csv", splits)
    if not (output / "tile_manifest.jsonl").exists():
        save_jsonl(output / "tile_manifest.jsonl", [])
    summary.update(
        raw_files=len(rows),
        parents=len(scene_rows),
        catalog_parents=len({row["parent_image_id"] for row in catalog}),
        available_modalities={
            m: sum(row["modality"] == m for row in rows)
            for m in ["optical", "sar", "label"]
        },
        complete_release_download_claimed=False,
        scope="downloaded pilot originals; catalog separately lists all published files",
        manifest_hashes={
            name: sha256_file(output / name)
            for name in ["raw_manifest.jsonl", "scene_groups.csv", "splits.csv"]
        },
        stages_completed=[
            "identify_coverage_and_related_scenes",
            "geographic_grouping",
            "split_assignment",
        ],
        training_tiles_generated=0,
        next_stage="independent_base_alignment_audit",
    )
    save_json(output / "lineage_summary.json", summary)
    save_json(output / "frozen_raw_config.json", config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
