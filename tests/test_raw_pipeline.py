"""Native-data provenance and reproducible synthetic geometry, without real admission."""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio

from sdgm.geometry import transform_points
from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.raw_alignment import checkpoint_measurement, region_admission
from sdgm.raw_lineage import (
    group_scenes,
    inventory_raw,
    nominal_sheet_bounds,
    sha256_file,
)
from sdgm.raw_tiles import generate_dataset, generate_one, sample_seed, write_raster


def test_grouping_merges_transitive_overlap_and_leaves_unknown_unassigned():
    # Distinct parent footprints form A--B--C even though A and C do not touch.
    bounds = {
        "A": [111, 30, 111.01, 30.01],
        "B": [111.06, 30, 111.07, 30.01],
        "C": [111.12, 30, 111.13, 30.01],
        "D": [113, 30, 113.01, 30.01],
    }
    raw = [
        dict(
            parent_image_id=p, modality=m, width=600, height=600, raw_file_id=f"{p}/{m}"
        )
        for p in ["A", "B", "C", "D", "UNKNOWN"]
        for m in ["optical", "sar"]
    ]
    coverage = {
        p: dict(status="publisher_figure_supported", nominal_bounds_wgs84=b)
        for p, b in bounds.items()
    }
    config = dict(
        coverage_uncertainty_m=3000, geographic_split_longitudes=[112.5, 114.75]
    )
    scenes, splits, summary = group_scenes(raw, coverage, config)
    lookup = {r["parent_image_id"]: r for r in scenes}
    assert (
        lookup["A"]["geo_group_id"]
        == lookup["B"]["geo_group_id"]
        == lookup["C"]["geo_group_id"]
    )
    assert lookup["D"]["split"] == "validation"
    assert lookup["A"]["split"] == "training"
    assert lookup["UNKNOWN"]["split"] == "unassigned"
    assert summary["minimum_cross_split_buffered_separation_m"] > 0
    assert len([r for r in splits if r["split"] != "unassigned"]) == 2


def test_checkpoint_uncertainty_cannot_certify_precision_or_a_region():
    roi = dict(
        width=512,
        height=512,
        x=100,
        y=200,
        parent_image_id="P",
        geo_group_id="G",
        split="training",
    )
    point = dict(
        id="P1",
        optical_xy=[10, 20],
        sar_xy=[13, 24],
        optical_uncertainty_px=5,
        sar_uncertainty_px=6,
    )
    row = checkpoint_measurement(point, roi)
    assert row["nominal_residual_px"] == 5
    assert row["conservative_residual_plus_uncertainty_px"] == 16
    assert row["optical_original_xy"] == [110, 220]
    assert row["sar_original_xy"] == [113, 224]
    assert not region_admission([row])["coarse_allowed"]
    # Even a perfect sparse point is not a region certificate.
    row["conservative_residual_plus_uncertainty_px"] = 0
    assert not region_admission([row])["fine_allowed"]
    assert not region_admission([row])["coarse_allowed"]
    with pytest.raises(ValueError, match="outside"):
        checkpoint_measurement(dict(point, sar_xy=[512, 10]), roi)


@pytest.fixture
def pilot(tmp_path):
    raw_root = tmp_path / "raw"
    catalog = []
    images = {}
    for index, parent in enumerate(["TRAIN", "VAL"]):
        yy, xx = np.indices((600, 600))
        optical = np.stack(
            [
                (xx % 256).astype("uint8"),
                (yy % 256).astype("uint8"),
                ((xx + yy) % 256).astype("uint8"),
                np.full_like(xx, 73 + index, dtype="uint8"),
            ]
        )
        sar = np.zeros((600, 600), "uint8")
        sar[98, 103] = 255  # native optical (100,100) + B(3,-2)
        images[parent] = optical
        for modality, array in [("optical", optical), ("sar", sar)]:
            path = raw_root / modality / f"{parent}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_raster(path, array)
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
                with rasterio.open(path, "r+") as image:
                    mask = np.full((600, 600), 255, "uint8")
                    mask[60, 60] = 0
                    image.write_mask(mask)
            catalog.append(
                dict(
                    parent_image_id=parent,
                    modality=modality,
                    source_id=f"public-{parent}-{modality}",
                    url="fixture://source",
                )
            )
    # Incomplete parent: source counterpart catalog exists, local file does not.
    path = raw_root / "sar" / "INCOMPLETE.tif"
    write_raster(path, np.zeros((16, 16), "uint8"))
    catalog.extend(
        dict(
            parent_image_id="INCOMPLETE",
            modality=m,
            source_id=f"missing-{m}",
            url="fixture://missing",
        )
        for m in ["optical", "sar"]
    )
    raw = inventory_raw(catalog, raw_root)
    by_parent = {
        p: {r["modality"]: r for r in raw if r["parent_image_id"] == p}
        for p in ["TRAIN", "VAL"]
    }
    output = tmp_path / "reports"
    output.mkdir()
    save_jsonl(output / "raw_manifest.jsonl", raw)
    scenes = [
        dict(parent_image_id=p, geo_group_id=f"G-{p}", split=s)
        for p, s in [("TRAIN", "training"), ("VAL", "validation")]
    ]
    save_csv(output / "scene_groups.csv", scenes)
    save_csv(
        output / "splits.csv",
        [dict(geo_group_id=r["geo_group_id"], split=r["split"]) for r in scenes],
    )
    save_jsonl(output / "tile_manifest.jsonl", [])
    rois = [
        dict(
            roi_id=f"{r['parent_image_id']}/roi0",
            parent_image_id=r["parent_image_id"],
            x=16,
            y=24,
            width=512,
            height=512,
        )
        for r in scenes
    ]
    save_json(output / "base_audit/inspection_rois.json", rois)
    config = dict(
        seed=11,
        tile_size=512,
        training_tiles=2,
        validation_tiles=2,
        perturbation=dict(
            rotation_degrees=[0, 0],
            scale=[1, 1],
            translation_pixels=[8, 8],
            interpolation="bilinear",
            fill_value=0,
        ),
    )
    save_json(output / "frozen_raw_config.json", config)
    eligibility = {
        r["roi_id"]: dict(
            coarse_allowed=True,
            fine_allowed=False,
            checkpoint_ids=["fixture-not-real-audit"],
            base_optical_to_sar_matrix=[[1, 0, 3], [0, 1, -2], [0, 0, 1]],
            base_relation_status="synthetic_test_fixture_only",
        )
        for r in rois
    }
    audit = dict(
        prealignment_performed=False,
        geographic_independence_admitted=True,  # synthetic fixture groups, never a real-data certificate
        region_admissions=eligibility,
        lineage_hashes={
            n: sha256_file(output / n)
            for n in ["raw_manifest.jsonl", "scene_groups.csv", "splits.csv"]
        },
    )
    return dict(
        output=output,
        config=config,
        audit=audit,
        raw=raw,
        by_parent=by_parent,
        rois=rois,
        images=images,
    )


def test_raw_inventory_distinguishes_catalog_and_local_pairing(pilot):
    incomplete = next(r for r in pilot["raw"] if r["parent_image_id"] == "INCOMPLETE")
    assert incomplete["paired_source_file_id"] == "missing-optical"
    assert incomplete["paired_local_present"] is False
    for row in pilot["raw"]:
        assert row["crs"] is None and row["transform"] is None
        assert row["nodata"] is None
        assert len(row["sha256"]) == 64
    assert nominal_sheet_bounds("NH49E001014") == pytest.approx(
        [111.25, 31 + 5 / 6, 111.5, 32]
    )
    assert nominal_sheet_bounds("not-a-sheet") is None


def test_reproducible_chain_preserves_rgb_nir_and_zero_support(pilot, tmp_path):
    summary = generate_dataset(
        pilot["config"], pilot["output"], tmp_path / "generated", pilot["audit"]
    )
    assert summary["samples"] == 4 and summary["unique_raw_windows"] == 2
    rows = [
        json.loads(line)
        for line in (pilot["output"] / "tile_manifest.jsonl").read_text().splitlines()
    ]
    for row in rows:
        with rasterio.open(row["paths"]["optical_rgbnir.tif"]) as image:
            np.testing.assert_array_equal(
                image.read(), pilot["images"][row["parent_image_id"]][:, 24:536, 16:528]
            )
        # A_s @ C_s @ B @ inv(C_o): (84,76) -> (95,82).
        mapped = transform_points(
            row["transform_chain"]["optical_crop_to_perturbed_sar"], [[84, 76]]
        )[0]
        np.testing.assert_allclose(mapped, [95, 82])
        with rasterio.open(row["paths"]["sar_perturbed.tif"]) as image:
            sar = image.read(1)
        y, x = np.unravel_index(sar.argmax(), sar.shape)
        assert [x, y] == [95, 82]
        mask = cv2.imread(row["paths"]["valid_sar.png"], 0)
        assert sar[100, 100] == 0 and mask[100, 100] == 255
        assert (mask[:8] == 0).all() and (mask[:, :8] == 0).all()
        assert mask[60 - 24 + 8, 60 - 16 + 8] == 0
        assert row["fine_supervision_admitted"] is False
    # Regenerate the same validation sample elsewhere: pixels, seed and matrices fixed.
    validation = next(r for r in rows if r["split"] == "validation")
    regenerated = generate_one(
        pilot["by_parent"]["VAL"],
        pilot["rois"][1],
        pilot["audit"]["region_admissions"]["VAL/roi0"],
        "G-VAL",
        "validation",
        pilot["config"],
        tmp_path / "reproduced",
        repeat=validation["repeat"],
    )
    assert regenerated["output_sha256"] == validation["output_sha256"]
    assert regenerated["transform_chain"] == validation["transform_chain"]
    assert regenerated["random_seed"] == validation["random_seed"]
    assert sample_seed(11, "training", "VAL/roi0", 0) != sample_seed(
        11, "validation", "VAL/roi0", 0
    )
    with pytest.raises(FileExistsError):
        generate_dataset(
            pilot["config"], pilot["output"], tmp_path / "generated", pilot["audit"]
        )


def test_validation_admission_preflight_writes_nothing(pilot, tmp_path):
    audit = copy.deepcopy(pilot["audit"])
    audit["region_admissions"]["VAL/roi0"]["coarse_allowed"] = False
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="validation"):
        generate_dataset(pilot["config"], pilot["output"], destination, audit)
    assert not destination.exists()
    assert not (pilot["output"] / "tile_manifest.jsonl").read_text()


def test_source_tamper_is_rejected_before_writes(pilot, tmp_path):
    path = Path(pilot["by_parent"]["TRAIN"]["sar"]["path"])
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash changed"):
        generate_dataset(
            pilot["config"], pilot["output"], tmp_path / "reject", pilot["audit"]
        )
    assert not (tmp_path / "reject").exists()


def test_stale_audit_is_rejected(pilot, tmp_path):
    with (pilot["output"] / "scene_groups.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="stale"):
        generate_dataset(
            pilot["config"], pilot["output"], tmp_path / "reject", pilot["audit"]
        )
    assert not (tmp_path / "reject").exists()


def test_raw_audit_export_handles_existing_status_and_zero_points(
    pilot, tmp_path, monkeypatch
):
    from scripts.audit_raw_alignment import main

    rois = json.loads((pilot["output"] / "base_audit/inspection_rois.json").read_text())
    for roi, split in zip(rois, ["training", "validation"]):
        roi.update(
            split=split,
            geo_group_id=f"G-{roi['parent_image_id']}",
            checkpoint_status="awaiting_manual_review",
        )
    save_json(pilot["output"] / "base_audit/inspection_rois.json", rois)
    config = dict(
        pilot["config"],
        output=str(pilot["output"]),
        coarse_tolerance_px=8,
        fine_target_error_px=2,
    )
    save_json(tmp_path / "audit_config.json", config)
    save_json(
        tmp_path / "observations.json",
        dict(
            checkpoints=[],
            roi_notes={r["roi_id"]: "unobservable fixture" for r in rois},
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_raw_alignment.py",
            "--config",
            str(tmp_path / "audit_config.json"),
            "--observations",
            str(tmp_path / "observations.json"),
        ],
    )
    main()
    admission = json.loads((pilot["output"] / "base_audit/admission.json").read_text())
    assert not any(
        row["coarse_allowed"] for row in admission["region_admissions"].values()
    )
    reviews = [
        json.loads(line)
        for line in (pilot["output"] / "base_audit/region_reviews.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all(row["checkpoint_status"] == "no_point_selected" for row in reviews)
    assert (
        json.loads((pilot["output"] / "base_audit/audit_summary.json").read_text())[
            "nominal_residual_range_px"
        ]
        is None
    )


def test_external_mask_tamper_is_rejected(pilot, tmp_path):
    row = pilot["by_parent"]["TRAIN"]["sar"]
    assert row["supporting_files"]
    with Path(row["supporting_files"][0]["path"]).open("ab") as stream:
        stream.write(b"modified-mask")
    with pytest.raises(ValueError, match="supporting file hash changed"):
        generate_dataset(
            pilot["config"], pilot["output"], tmp_path / "reject", pilot["audit"]
        )
    assert not (tmp_path / "reject").exists()


def test_fractional_warp_requires_full_valid_support(pilot, tmp_path):
    config = copy.deepcopy(pilot["config"])
    config["perturbation"]["translation_pixels"] = [0.5, 0.5]
    row = generate_one(
        pilot["by_parent"]["TRAIN"],
        pilot["rois"][0],
        pilot["audit"]["region_admissions"]["TRAIN/roi0"],
        "G-TRAIN",
        "training",
        config,
        tmp_path / "fractional",
    )
    mask = cv2.imread(row["paths"]["valid_sar.png"], 0)
    assert (mask[0] == 0).all() and (mask[:, 0] == 0).all()
    # One invalid raw pixel contributes to four output interpolation footprints.
    assert (mask[36:38, 44:46] == 0).all()
    assert mask[100, 100] == 255


def test_provisional_geography_cannot_be_used_as_independent_validation(
    pilot, tmp_path
):
    audit = copy.deepcopy(pilot["audit"])
    audit["geographic_independence_admitted"] = False
    with pytest.raises(ValueError, match="Geographic independence"):
        generate_dataset(pilot["config"], pilot["output"], tmp_path / "reject", audit)
    assert not (tmp_path / "reject").exists()
