"""Record pilot availability, exclusions and stage gates without accuracy claims."""

import importlib.metadata
import json
import platform
from pathlib import Path

from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.raw_lineage import sha256_file


def main():
    config = json.loads(Path("configs/raw_rebuild.json").read_text())
    output = Path(config["output"])
    raw = [
        json.loads(line)
        for line in (output / "raw_manifest.jsonl").read_text().splitlines()
    ]
    catalog = json.loads(Path(config["catalog"]).read_text())
    present = {(r["parent_image_id"], r["modality"]) for r in raw}
    availability = []
    for parent in sorted({r["parent_image_id"] for r in catalog}):
        row = dict(parent_image_id=parent)
        row.update(
            {
                m + "_locally_present": (parent, m) in present
                for m in ["optical", "sar", "label"]
            }
        )
        row["optical_sar_pair_complete"] = (
            row["optical_locally_present"] and row["sar_locally_present"]
        )
        availability.append(row)
    save_csv(output / "catalog_availability.csv", availability)
    audit = json.loads((output / "base_audit/audit_summary.json").read_text())
    lineage = json.loads((output / "lineage_summary.json").read_text())
    if (output / "tile_manifest.jsonl").read_text().strip() or audit[
        "coarse_admitted_regions"
    ]:
        raise ValueError("This report summarizes the non-admitted pilot only")
    issues = [
        dict(
            code="INCOMPLETE_PILOT_ACQUISITION",
            detail=f"{len(raw)} local files out of {len(catalog)} catalog entries; this is a pilot, not a complete release",
        ),
        dict(
            code="MISSING_INTRINSIC_GEOREFERENCE",
            detail="Inspected TIFFs have no CRS/geotransform/GCP/RPC; external map footprints are grouping evidence only",
        ),
        dict(
            code="CONDITIONAL_GEOGRAPHY",
            detail="Publisher figure matching supports approximate coverage; map datum unknown, 3km allowance is an assumption, not a surveyed error bound",
        ),
        dict(code="NATIVE_BASE_ACCURACY_NOT_CERTIFIED", detail=audit),
        dict(
            code="MISSING_ACQUISITION_PRODUCT_IDS",
            detail="Publisher full-size TIFF IDs known; original GF-1/GF-3 product IDs unknown",
        ),
    ]
    for row in raw:
        if not row["paired_local_present"]:
            issues.append(
                dict(
                    code="LOCAL_COUNTERPART_MISSING",
                    raw_file_id=row["raw_file_id"],
                    detail=row["pairing_status"],
                )
            )
    coverage = json.loads((output / "coverage_evidence.json").read_text())
    for parent in sorted({r["parent_image_id"] for r in raw}):
        if coverage.get(parent, {}).get("status") != "publisher_figure_supported":
            issues.append(
                dict(
                    code="GEOGRAPHY_UNCONFIRMED",
                    parent_image_id=parent,
                    split="unassigned",
                )
            )
    save_jsonl(output / "exceptions.jsonl", issues)
    sources = dict(
        publisher_repository="https://github.com/AmberHen/WHU-OPT-SAR-dataset",
        publisher_paper_doi="10.1016/j.jag.2021.102638",
        official_drive_root="https://drive.google.com/drive/folders/1W3iMpkehng7ADXmhz9pPvmgFQBayq22t",
        drive_modality_folders=dict(
            optical="1FEjzFEpIN8WDwNpY0JnU3CwOrW6uGnN5",
            sar="1Zb_vh-dBGQaE2RCMdgdQflBG5NzC16Pp",
            label="1F_Rvv6VwLBkKBCgQxCYBDTz29qexqJHC",
        ),
        catalog_collection="public embeddedfolderview file listings; 100 IDs per modality; catalog source IDs and URLs retained",
        release_form="individual full-size TIFF files; no original-release archive acquired in this round",
        publisher_checksums_available=False,
        local_hash_status="SHA256 measured after block decoding; not authenticated against a publisher checksum",
        original_meaning="publisher full-size release, not raw satellite acquisition products",
        original_product_ids_recovered=False,
        source_hashes={
            str(path): sha256_file(path)
            for path in [
                Path(config["catalog"]),
                output / "source/publisher_README.md",
                Path(config["publisher_figure"]),
            ]
        },
    )
    save_json(output / "source_evidence.json", sources)
    save_json(
        output / "environment.json",
        dict(
            python=platform.python_version(),
            platform=platform.platform(),
            packages={
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "opencv-python",
                    "rasterio",
                    "pyproj",
                    "shapely",
                    "torch",
                    "pytest",
                    "gdown",
                ]
            },
        ),
    )
    save_json(
        output / "round_status.json",
        dict(
            item1=dict(
                status="pilot_lineage_created",
                raw_files=lineage["raw_files"],
                parents=lineage["parents"],
                groups=lineage["groups"],
                unresolved_parents=lineage["unresolved_parents"],
                geography="provisional_not_certified",
            ),
            item2=dict(
                status="visual_screening_completed_not_accuracy_certification",
                tentative_checkpoints=audit["tentative_checkpoints"],
                coarse_admitted_regions=0,
                fine_admitted_regions=0,
                independent_human_review=False,
            ),
            item3=dict(
                status="generator_implemented_fixture_tested_real_generation_blocked",
                actual_tiles_generated=0,
                tile_manifest="empty by admission design",
            ),
            item4=dict(
                status="not_run_data_admission_unsatisfied",
                model_changed=False,
                fixed32_training_run=False,
                geographic_validation_metrics=None,
            ),
            prealignment_performed=False,
            tolerance_widened=False,
            network_upgrade=False,
            manifests={
                name: sha256_file(output / name)
                for name in [
                    "raw_manifest.jsonl",
                    "scene_groups.csv",
                    "splits.csv",
                    "tile_manifest.jsonl",
                ]
            },
        ),
    )


if __name__ == "__main__":
    main()
