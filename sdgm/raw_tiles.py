"""Frozen native crops and synthetic affine perturbations, guarded by admission."""

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from .geometry import homogeneous
from .io import save_json, save_jsonl
from .raw_lineage import sha256_file


def sample_seed(master_seed, split, roi_id, repeat):
    value = f"{master_seed}|{split}|{roi_id}|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def perturbation_matrix(seed, size, settings):
    random = np.random.default_rng(seed)
    angle = float(random.uniform(*settings["rotation_degrees"]))
    scale = float(random.uniform(*settings["scale"]))
    shift = random.uniform(*settings["translation_pixels"], size=2)
    matrix = homogeneous(
        cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), angle, scale)
    )
    matrix[:2, 2] += shift
    return matrix, dict(
        rotation_degrees=angle, scale=scale, translation_xy=shift.tolist()
    )


def write_raster(path, array):
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[2],
        height=array.shape[1],
        count=array.shape[0],
        dtype=str(array.dtype),
        compress="deflate",
    ) as output:
        output.write(array)
        if array.shape[0] == 4:
            output.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
                rasterio.enums.ColorInterp.undefined,
            )


def generate_one(raw_rows, roi, admission, group, split, config, destination, repeat=0):
    """Base relation accuracy is separate from exact synthetic transformation."""
    if (
        split not in ["training", "validation", "testing"]
        or group == "GEOGRAPHY_UNCONFIRMED"
    ):
        raise ValueError("Geographic grouping and split must precede cropping")
    if not admission.get("coarse_allowed", False):
        raise ValueError(
            "This raw region has not passed the coarse base-alignment audit"
        )
    size = config["tile_size"]
    x, y = int(roi["x"]), int(roi["y"])
    if (roi["width"], roi["height"]) != (size, size):
        raise ValueError("Native crop size mismatch")
    seed = sample_seed(config["seed"], split, roi["roi_id"], repeat)
    synthetic, parameters = perturbation_matrix(seed, size, config["perturbation"])
    base = homogeneous(admission["base_optical_to_sar_matrix"])
    optical_crop = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], float)
    sar_crop = optical_crop.copy()
    total = synthetic @ sar_crop @ base @ np.linalg.inv(optical_crop)
    sample_id = hashlib.sha256(f"{roi['roi_id']}|{repeat}|{seed}".encode()).hexdigest()[
        :20
    ]
    folder = Path(destination) / split / sample_id
    if folder.exists():
        raise FileExistsError(f"A frozen sample already exists: {folder}")
    optical_record, sar_record = raw_rows["optical"], raw_rows["sar"]
    window = Window(x, y, size, size)
    with rasterio.open(optical_record["path"]) as image:
        if image.count != 4 or image.dtypes != ("uint8",) * 4:
            raise ValueError("Original uint8 RGB and NIR must be retained")
        if x < 0 or y < 0 or x + size > image.width or y + size > image.height:
            raise ValueError("Crop outside original optical extent")
        optical = image.read(window=window)
        optical_valid = (image.read_masks([1, 2, 3], window=window) > 0).all(
            0
        ) & np.isfinite(optical[:3]).all(0)
    with rasterio.open(sar_record["path"]) as image:
        if image.count != 1 or image.dtypes != ("uint8",):
            raise ValueError("Expected native uint8 single-band SAR")
        if x < 0 or y < 0 or x + size > image.width or y + size > image.height:
            raise ValueError("Crop outside original SAR extent")
        sar = image.read(1, window=window)
        sar_valid = (image.read_masks(1, window=window) > 0) & np.isfinite(sar)
    warped = cv2.warpAffine(
        sar,
        synthetic[:2],
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=config["perturbation"]["fill_value"],
    )
    valid_warped = (
        cv2.warpAffine(
            sar_valid.astype("float32"),
            synthetic[:2],
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        >= 1 - 1e-6
    )
    folder.mkdir(parents=True)
    write_raster(folder / "optical_rgbnir.tif", optical)
    write_raster(folder / "sar_base.tif", sar)
    write_raster(folder / "sar_perturbed.tif", warped)
    for name, mask in [
        ("valid_optical.png", optical_valid),
        ("valid_sar.png", valid_warped),
    ]:
        if not cv2.imwrite(str(folder / name), mask.astype("uint8") * 255):
            raise OSError(f"Could not write {name}")
    paths = {
        name: str(folder / name)
        for name in [
            "optical_rgbnir.tif",
            "sar_base.tif",
            "sar_perturbed.tif",
            "valid_optical.png",
            "valid_sar.png",
        ]
    }
    metadata = dict(
        tile_id=sample_id,
        parent_image_id=roi["parent_image_id"],
        geo_group_id=group,
        split=split,
        roi_id=roi["roi_id"],
        raw_window_id=f"{roi['parent_image_id']}:{x},{y},{size},{size}",
        repeat=repeat,
        overlap_status="repeats share an identical native window; distinct ROIs may overlap; never independent sample counts",
        crop_window=dict(x=x, y=y, width=size, height=size),
        original_files={
            key: {
                field: record[field]
                for field in [
                    "raw_file_id",
                    "source_file_id",
                    "sha256",
                    "path",
                    "supporting_files",
                ]
            }
            for key, record in raw_rows.items()
            if key in ["optical", "sar"]
        },
        preprocessing=dict(
            spatial_resize=False,
            optical_bands_preserved=[1, 2, 3, 4],
            model_rgb_bands=[1, 2, 3],
            nir_used_by_model=False,
            grayscale_conversion=False,
            prealignment=False,
            planned_model_normalization="RGB and SAR uint8 / 127.5 - 1; not DINO normalization",
        ),
        base_coordinate_relation=dict(
            matrix=base.tolist(),
            status=admission["base_relation_status"],
            evidence_checkpoint_ids=admission["checkpoint_ids"],
        ),
        transform_chain=dict(
            optical_original_to_crop=optical_crop.tolist(),
            sar_original_to_crop=sar_crop.tolist(),
            base_optical_original_to_sar_original=base.tolist(),
            synthetic_sar_crop_to_perturbed=synthetic.tolist(),
            optical_crop_to_perturbed_sar=total.tolist(),
            composition="A_s @ C_s @ B_base @ inverse(C_o)",
        ),
        coordinate_convention="zero-based (x=column,y=row) pixel centers, column vectors; optical crop to perturbed SAR",
        perturbation_parameters=parameters,
        random_seed=seed,
        master_seed=config["seed"],
        interpolation="OpenCV INTER_LINEAR",
        interpolation_version=cv2.__version__,
        fill_value=config["perturbation"]["fill_value"],
        validity="GDAL masks and finite values; full bilinear SAR support required; zero intensity alone is not NoData",
        source_accuracy_status="coarse_audit_admitted_only",
        fine_supervision_admitted=bool(admission.get("fine_allowed", False)),
        supervision_source="audited nominal base relationship plus known synthetic affine; synthetic exactness does not certify native registration",
        validation_frozen=True,
        paths=paths,
        output_sha256={name: sha256_file(path) for name, path in paths.items()},
    )
    save_json(folder / "metadata.json", metadata)
    return metadata


def generate_dataset(config, output, destination, audit):
    """Preflight every split and source before writes; commit one frozen dataset."""
    output, destination = Path(output), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"Frozen destination already exists: {destination}")
    if audit.get("geographic_independence_admitted") is not True:
        raise ValueError(
            "Geographic independence is not certified; provisional splits cannot generate an independent validation dataset"
        )
    if audit.get("prealignment_performed", False):
        raise ValueError("No prealignment correction is implemented in this run")
    if config["perturbation"]["interpolation"] != "bilinear":
        raise ValueError("Only declared bilinear interpolation is implemented")
    if json.loads((output / "frozen_raw_config.json").read_text()) != config:
        raise ValueError("Configuration differs from frozen lineage configuration")
    for name in ["raw_manifest.jsonl", "scene_groups.csv", "splits.csv"]:
        if audit.get("lineage_hashes", {}).get(name) != sha256_file(output / name):
            raise ValueError(f"Audit is absent or stale for {name}")
    if (output / "tile_manifest.jsonl").read_text().strip():
        raise FileExistsError("Lineage already has a frozen nonempty tile manifest")
    raw = [
        json.loads(line)
        for line in (output / "raw_manifest.jsonl").read_text().splitlines()
    ]
    by_parent = {}
    for row in raw:
        by_parent.setdefault(row["parent_image_id"], {})[row["modality"]] = row
    with (output / "scene_groups.csv").open() as stream:
        scenes = {row["parent_image_id"]: row for row in csv.DictReader(stream)}
    with (output / "splits.csv").open() as stream:
        splits = {row["geo_group_id"]: row["split"] for row in csv.DictReader(stream)}
    rois = {
        row["roi_id"]: row
        for row in json.loads((output / "base_audit/inspection_rois.json").read_text())
    }
    eligibility = audit["region_admissions"]
    plan, group_sets = [], {}
    for split, count in [
        ("training", config["training_tiles"]),
        ("validation", config["validation_tiles"]),
    ]:
        if not isinstance(count, int) or count < 1:
            raise ValueError(
                "Both training and validation require positive sample counts"
            )
        admitted = sorted(
            key
            for key, value in eligibility.items()
            if value["coarse_allowed"]
            and scenes[rois[key]["parent_image_id"]]["split"] == split
        )
        if not admitted:
            raise ValueError(
                f"No audited and geographically assigned regions for {split}; no files written"
            )
        group_sets[split] = set()
        for index in range(count):
            roi = rois[admitted[index % len(admitted)]]
            scene = scenes[roi["parent_image_id"]]
            group = scene["geo_group_id"]
            if group == "GEOGRAPHY_UNCONFIRMED" or splits.get(group) != split:
                raise ValueError("Region lacks consistent geographic split")
            region = eligibility[roi["roi_id"]]
            base = homogeneous(region["base_optical_to_sar_matrix"])
            if (
                not region.get("checkpoint_ids")
                or not np.isfinite(base).all()
                or abs(np.linalg.det(base)) < 1e-8
            ):
                raise ValueError(
                    "Invalid admitted base relationship or absent audit evidence"
                )
            files = by_parent[roi["parent_image_id"]]
            if not {"optical", "sar"} <= files.keys():
                raise ValueError("Incomplete source pair")
            for modality in ["optical", "sar"]:
                record = files[modality]
                if not (
                    0 <= roi["x"] <= record["width"] - config["tile_size"]
                    and 0 <= roi["y"] <= record["height"] - config["tile_size"]
                ):
                    raise ValueError("Crop outside frozen source dimensions")
            group_sets[split].add(group)
            plan.append((files, roi, region, group, split, index // len(admitted)))
    if group_sets["training"] & group_sets["validation"]:
        raise ValueError("Geographic leakage across generated splits")
    checked = set()
    for files, *_ in plan:
        for modality in ["optical", "sar"]:
            record = files[modality]
            if record["path"] not in checked:
                if sha256_file(record["path"]) != record["sha256"]:
                    raise ValueError(
                        f"Raw source hash changed: {record['raw_file_id']}"
                    )
                with rasterio.open(record["path"]) as image:
                    current_support = {
                        str(Path(name).absolute())
                        for name in image.files
                        if Path(name).absolute() != Path(record["path"]).absolute()
                    }
                expected_support = {item["path"] for item in record["supporting_files"]}
                if current_support != expected_support or any(
                    sha256_file(item["path"]) != item["sha256"]
                    for item in record["supporting_files"]
                ):
                    raise ValueError(
                        f"Raw supporting file hash changed: {record['raw_file_id']}"
                    )
                checked.add(record["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    manifest = []
    try:
        for files, roi, region, group, split, repeat in plan:
            row = generate_one(files, roi, region, group, split, config, stage, repeat)
            stage_folder = Path(row["paths"]["optical_rgbnir.tif"]).parent
            row["paths"] = {
                name: str(destination / Path(path).relative_to(stage))
                for name, path in row["paths"].items()
            }
            row["lineage_hashes"] = audit["lineage_hashes"]
            row["audit_sha256"] = hashlib.sha256(
                json.dumps(audit, sort_keys=True).encode()
            ).hexdigest()
            save_json(stage_folder / "metadata.json", row)
            manifest.append(row)
        save_jsonl(stage / "tile_manifest.jsonl", manifest)
        save_json(stage / "generation_config.json", config)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    save_jsonl(output / "tile_manifest.jsonl", manifest)
    unique_windows = {row["raw_window_id"] for row in manifest}
    summary = dict(
        samples=len(manifest),
        unique_raw_windows=len(unique_windows),
        repeated_window_samples=len(manifest) - len(unique_windows),
        splits={
            split: dict(
                samples=sum(row["split"] == split for row in manifest),
                unique_raw_windows=len(
                    {row["raw_window_id"] for row in manifest if row["split"] == split}
                ),
                geographic_groups=sorted(groups),
            )
            for split, groups in group_sets.items()
        },
        geographic_independence="conditional on publisher coverage-map identification and declared buffer assumptions",
        transform_seed_frozen=True,
        prealignment_performed=False,
        original_rgb_nir_sar_preserved=True,
        hashes={
            name: sha256_file(output / name)
            for name in [
                "raw_manifest.jsonl",
                "scene_groups.csv",
                "splits.csv",
                "tile_manifest.jsonl",
            ]
        },
    )
    save_json(output / "generation_summary.json", summary)
    return summary
