"""Decode native OS/UBCv2 releases and freeze model-independent eligibility."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import warnings
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import box
from shapely.strtree import STRtree

from sdgm.audit import raster_record
from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.raw_lineage import sha256_file

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def pixel_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def joint_hash(a, b):
    return hashlib.sha256((a + b).encode()).hexdigest()


def inspect_image(path):
    path = Path(path)
    if path.suffix.lower() == ".png":
        with Image.open(path) as image:
            image.load()
            array = np.array(image)
            mode = image.mode
        array = array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)
        mask = np.ones_like(array, dtype=bool)
        record = dict(
            width=array.shape[2],
            height=array.shape[1],
            channels=array.shape[0],
            mode=mode,
            dtype=[str(array.dtype)] * array.shape[0],
            crs=None,
            transform=None,
            bounds=None,
            nodata=None,
            gcps_count=0,
            rpcs_present=False,
            spectral_mapping="stored PNG channels; no replication or conversion",
        )
    else:
        with rasterio.open(path) as image:
            array = image.read()
            mask = image.read_masks() > 0
            record = dict(
                width=image.width,
                height=image.height,
                channels=image.count,
                dtype=list(image.dtypes),
                crs=str(image.crs) if image.crs else None,
                transform=list(image.transform)[:6] if image.crs else None,
                bounds=list(image.bounds) if image.crs else None,
                nodata=None if image.nodata is None else str(image.nodata),
                gcps_count=len(image.gcps[0]),
                rpcs_present=image.rpcs is not None,
                band_color_interpretation=[v.name for v in image.colorinterp],
                band_descriptions=list(image.descriptions),
                tags=image.tags(),
            )
    finite = np.isfinite(array)
    record.update(
        path=str(path),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        pixel_sha256=pixel_hash(array),
        minimum=[
            float(b[v & m].min()) if (v & m).any() else None
            for b, v, m in zip(array, finite, mask)
        ],
        maximum=[
            float(b[v & m].max()) if (v & m).any() else None
            for b, v, m in zip(array, finite, mask)
        ],
        nonfinite_count=int((~finite).sum()),
        valid_pixels_per_band=(mask & finite).sum((1, 2)).tolist(),
        masked_fraction=(~mask).mean((1, 2)).tolist(),
        zero_fraction=(array == 0).mean((1, 2)).tolist(),
    )
    return record, array


def inspect_pair(job):
    row = dict(job)
    errors = []
    arrays = {}
    for modality in ["optical", "sar"]:
        path = row.pop(modality + "_path")
        if path is None:
            row[modality] = None
            errors.append("missing_counterpart")
            continue
        try:
            record, array = inspect_image(path)
            row[modality] = record
            arrays[modality] = array
            if record["nonfinite_count"]:
                errors.append("nonfinite_values")
            if not min(record["valid_pixels_per_band"]):
                errors.append("no_valid_pixels")
        except Exception as error:
            row[modality] = dict(path=str(path), decode_error=str(error))
            errors.append("decode_failure")
    if len(arrays) == 2:
        a, b = arrays["optical"], arrays["sar"]
        if a.shape[-2:] != b.shape[-2:]:
            errors.append("spatial_size_mismatch")
        row["joint_pixel_sha256"] = joint_hash(
            row["optical"]["pixel_sha256"], row["sar"]["pixel_sha256"]
        )
        if (
            row["dataset"] == "OS"
            and row["subset"].startswith("OSdataset/512")
            and a.shape[-2:] == b.shape[-2:] == (512, 512)
        ):
            row["quadrant_hashes"] = [
                dict(
                    x=x,
                    y=y,
                    width=256,
                    height=256,
                    joint_pixel_sha256=joint_hash(
                        pixel_hash(a[:, y : y + 256, x : x + 256]),
                        pixel_hash(b[:, y : y + 256, x : x + 256]),
                    ),
                )
                for y in [0, 256]
                for x in [0, 256]
            ]
        if row["dataset"] == "UBCv2":
            first, second = row["optical"], row["sar"]
            row["same_geographic_grid"] = bool(
                first["crs"]
                and first["crs"] == second["crs"]
                and first["transform"] == second["transform"]
                and a.shape[-2:] == b.shape[-2:]
            )
            row["grid_equality_is_registration_precision"] = False
    row["hard_exclusion_reasons"] = sorted(set(errors))
    row["eligible_by_frozen_file_rules"] = not errors
    row["cloud_status"] = "unassessed_no_cloud_exclusion"
    return row


def os_jobs(root):
    patch = root / "Patch-level Subset"
    folders = sorted({p.parent for p in patch.rglob("opt*.png")})
    for folder in folders:
        optical = {p.name[3:]: p for p in folder.glob("opt*.png")}
        sar = {p.name[3:]: p for p in folder.glob("sar*.png")}
        subset = folder.relative_to(patch).as_posix()
        for key in sorted(optical.keys() | sar.keys()):
            yield dict(
                dataset="OS",
                record_kind="patch_pair",
                id=f"{subset}/{Path(key).stem}",
                subset=subset,
                official_split=(
                    folder.name if subset.startswith("OSdataset/") else "unspecified"
                ),
                region_id=folder.name if not subset.startswith("OSdataset/") else None,
                parent_scene_id=None,
                geographic_independence="unknown",
                supervision="publisher_coregistered_base_no_per_pair_gt",
                optical_path=optical.get(key),
                sar_path=sar.get(key),
            )


def ubc_jobs(root):
    for split, sub in [
        ("train", "train_extracted/train"),
        ("val", "train_extracted/val"),
        ("test", "raw_extracted"),
    ]:
        folder = root / sub
        optical = {p.name: p for p in (folder / "rgb").glob("*.tif")}
        sar = {p.name: p for p in (folder / "sar").glob("*.tif")}
        for key in sorted(optical.keys() | sar.keys()):
            match = re.fullmatch(
                r"([^_]+)_(.+)_(-?\d+\.\d+)_(-?\d+\.\d+)", Path(key).stem
            )
            sensor, city, lat, lon = (
                match.groups() if match else (None, None, None, None)
            )
            yield dict(
                dataset="UBCv2",
                record_kind="patch_pair",
                id=f"{split}/{Path(key).stem}",
                subset=split,
                official_split=split,
                region_id=city,
                filename_sensor_token=sensor,
                filename_latitude=float(lat) if lat else None,
                filename_longitude=float(lon) if lon else None,
                parent_scene_id=None,
                geographic_independence="not_assumed_from_official_split",
                supervision="publisher_finely_aligned_base_no_registration_gt",
                optical_path=optical.get(key),
                sar_path=sar.get(key),
            )


def summarize(rows):
    counts = Counter()
    errors = Counter()
    profiles = Counter()
    ranges = {}
    for row in rows:
        counts[row["subset"]] += 1
        errors.update(row["hard_exclusion_reasons"])
        for mod in ["optical", "sar"]:
            rec = row.get(mod)
            if not rec or "decode_error" in rec:
                continue
            key = f"{row['subset']}|{mod}|{rec['height']}x{rec['width']}|C{rec['channels']}|{','.join(rec['dtype'])}"
            profiles[key] += 1
            interval = ranges.setdefault(key, [float("inf"), float("-inf")])
            valid_min = [v for v in rec["minimum"] if v is not None]
            valid_max = [v for v in rec["maximum"] if v is not None]
            if valid_min:
                interval[0] = min(interval[0], min(valid_min))
                interval[1] = max(interval[1], max(valid_max))
    return dict(
        actual_pairs=len(rows),
        subset_counts=dict(counts),
        hard_exclusion_counts=dict(errors),
        missing_or_damaged_pairs=sum(bool(r["hard_exclusion_reasons"]) for r in rows),
        native_profiles=[
            dict(profile=key, images=count, observed_range=ranges[key])
            for key, count in sorted(profiles.items())
        ],
        images_resized=False,
        pixel_zero_automatically_invalid=False,
    )


def previews(rows, destination):
    groups = defaultdict(list)
    for row in rows:
        if row["eligible_by_frozen_file_rules"]:
            key = (
                row["subset"]
                if row["dataset"] == "OS"
                else f"{row['official_split']}/{row['region_id']}"
            )
            groups[key].append(row)
    chosen = []
    for key, members in sorted(groups.items()):
        chosen.append(sorted(members, key=lambda row: row["id"])[len(members) // 2])
    for index, row in enumerate(chosen):
        panels = []
        for mod in ["optical", "sar"]:
            rec, a = inspect_image(row[mod]["path"])
            a = a[:3]
            if row["dataset"] == "UBCv2" and mod == "sar":
                a = np.log1p(np.maximum(a, 0))
                lo, hi = np.percentile(a[np.isfinite(a)], [2, 98])
                a = np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1) * 255
            a = np.clip(a, 0, 255).astype("uint8")
            a = np.repeat(a, 3, 0) if len(a) == 1 else a
            panels.append(Image.fromarray(np.moveaxis(a, 0, -1)))
        h = max(im.height for im in panels)
        w = sum(im.width for im in panels)
        canvas = Image.new("RGB", (w, h + 44), "white")
        for offset, image in [(0, panels[0]), (panels[0].width, panels[1])]:
            canvas.paste(image, (offset, 44))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 3), row["id"], fill="black")
        draw.text(
            (4, 20),
            "Native pixels | optical left, SAR right | UBC SAR display log1p+p2/p98 only",
            fill="black",
        )
        name = f"preview_{index:03d}.png"
        destination.mkdir(parents=True, exist_ok=True)
        canvas.save(destination / name)
        chosen[index] = dict(
            id=row["id"],
            file=name,
            native_hw=[panels[0].height, panels[0].width],
            subset=row["subset"],
        )
    save_json(destination / "index.json", chosen)


def relationships(rows, output):
    quadrants = defaultdict(list)
    for row in rows:
        for q in row.get("quadrant_hashes", []):
            quadrants[q["joint_pixel_sha256"]].append(
                dict(
                    parent_id=row["id"], split=row["official_split"], x=q["x"], y=q["y"]
                )
            )
    mapping = []
    for row in rows:
        if row["subset"].startswith("OSdataset/256"):
            matches = quadrants.get(row.get("joint_pixel_sha256"), [])
            row["exact_pixel_crop_parents"] = matches
            mapping.append(
                dict(
                    tile_id=row["id"],
                    official_split=row["official_split"],
                    parent_candidates=json.dumps(matches),
                    exact_match_count=len(matches),
                    crosses_official_split=any(
                        m["split"] != row["official_split"] for m in matches
                    ),
                    geographic_parent_scene_recovered=False,
                )
            )
    save_csv(output / "cross_resolution_mapping.csv", mapping)
    return dict(
        tiles_256=len(mapping),
        matched_exactly=sum(r["exact_match_count"] > 0 for r in mapping),
        ambiguous_matches=sum(r["exact_match_count"] > 1 for r in mapping),
        cross_split_matches=sum(r["crosses_official_split"] for r in mapping),
        geographic_scene_mapping="unknown; 512 crop parent identity is not a full acquisition scene",
    )


def spatial_split_check(rows):
    by_crs = defaultdict(list)
    city_splits = defaultdict(Counter)
    for row in rows:
        city_splits[row["region_id"] or "unknown"][row["official_split"]] += 1
        rec = row.get("optical") or {}
        if rec.get("crs") and rec.get("bounds"):
            by_crs[rec["crs"]].append(row)
    count = 0
    examples = []
    for crs, group in by_crs.items():
        polygons = [box(*r["optical"]["bounds"]) for r in group]
        tree = STRtree(polygons)
        for i, polygon in enumerate(polygons):
            for j in tree.query(polygon):
                j = int(j)
                if j <= i or group[i]["official_split"] == group[j]["official_split"]:
                    continue
                area = polygon.intersection(polygons[j]).area
                if area > 1e-6:
                    count += 1
                    if len(examples) < 30:
                        examples.append(
                            dict(
                                first=group[i]["id"],
                                second=group[j]["id"],
                                crs=crs,
                                overlap_area_crs_units_squared=area,
                            )
                        )
    return dict(
        city_split_counts={k: dict(v) for k, v in city_splits.items()},
        cross_official_split_overlapping_optical_footprints=count,
        overlap_examples=examples,
        no_overlap_would_not_prove_source_scene_independence=True,
        cross_dataset_OS_UBC_geographic_disjointness="unknown: OS scene locations unavailable",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_ubc_audit.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ubcv2/freeze_receipt.json").exists():
        raise FileExistsError(
            "Frozen audit already exists; use a fresh output configuration to preserve the external evaluation pool"
        )
    # This policy is frozen before reading any new model result.
    policy_path = output / "ubcv2/frozen_policy.json"
    if (
        policy_path.exists()
        and json.loads(policy_path.read_text()) != config["ubc_policy"]
    ):
        raise ValueError("Refuse to silently alter a frozen UBC policy")
    save_json(policy_path, config["ubc_policy"])
    os_root = Path(config["os_root"])
    readme = os_root / "Patch-level Subset/OSdataset/ReadMe.docx"
    with zipfile.ZipFile(readme) as z:
        xml = ET.fromstring(z.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text = "\n".join("".join(n.itertext()) for n in xml.findall(".//w:t", ns))
    (output / "source").mkdir(exist_ok=True)
    (output / "source/OS_ReadMe_extracted.txt").write_text(text)
    save_json(
        output / "source/OS_ReadMe_identity.json",
        dict(path=str(readme), sha256=sha256_file(readme)),
    )
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        os_rows = list(pool.map(inspect_pair, os_jobs(os_root)))
    mapping = relationships(os_rows, output / "os")
    save_jsonl(output / "os/manifest.jsonl", os_rows)
    save_json(
        output / "os/summary.json", dict(**summarize(os_rows), cross_resolution=mapping)
    )
    previews(os_rows, output / "os/previews")
    print("OS_PATCH_AUDIT_COMPLETE", len(os_rows), flush=True)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        ubc_rows = list(pool.map(inspect_pair, ubc_jobs(Path(config["ubc_root"]))))
    save_jsonl(output / "ubcv2/manifest.jsonl", ubc_rows)
    save_json(output / "ubcv2/summary.json", summarize(ubc_rows))
    save_json(output / "ubcv2/spatial_split_check.json", spatial_split_check(ubc_rows))
    exclusions = [
        dict(
            id=r["id"],
            split=r["official_split"],
            included=r["eligible_by_frozen_file_rules"],
            reasons="|".join(r["hard_exclusion_reasons"]),
            cloud_status=r["cloud_status"],
            optical_sha256=(r.get("optical") or {}).get("sha256"),
            sar_sha256=(r.get("sar") or {}).get("sha256"),
        )
        for r in ubc_rows
    ]
    save_csv(output / "ubcv2/frozen_exclusions.csv", exclusions)
    pool_rows = [
        r
        for r in exclusions
        if r["split"] == config["ubc_policy"]["evaluation_split"] and r["included"]
    ]
    save_csv(output / "ubcv2/frozen_evaluation_pool.csv", pool_rows)
    save_json(
        output / "ubcv2/freeze_receipt.json",
        dict(
            policy_sha256=sha256_file(policy_path),
            manifest_sha256=sha256_file(output / "ubcv2/manifest.jsonl"),
            exclusions_sha256=sha256_file(output / "ubcv2/frozen_exclusions.csv"),
            pool_sha256=sha256_file(output / "ubcv2/frozen_evaluation_pool.csv"),
            eligible_test_pairs=len(pool_rows),
            model_results_used=False,
            prior_analysis_exposed=True,
            blind_test=False,
        ),
    )
    previews(ubc_rows, output / "ubcv2/previews")
    print("UBCV2_AUDIT_COMPLETE", len(ubc_rows), flush=True)
    # The scene has no patch-to-scene map. Its GCPs never enter patch training.
    scene = os_root / "Scene-level Subset"
    scene_rows = {
        mod: raster_record(scene / name)
        for mod, name in [("optical", "Optical.tif"), ("sar", "sar.tif")]
    }
    gt = np.loadtxt(scene / "GT.txt")
    checks = {}
    for mod, col in [("optical", 0), ("sar", 2)]:
        rec = scene_rows[mod]
        xy = gt[:, col : col + 2]
        checks[mod] = dict(
            in_frame=int(
                (
                    (xy >= 0).all(1)
                    & (xy[:, 0] < rec["width"])
                    & (xy[:, 1] < rec["height"])
                ).sum()
            ),
            points=len(gt),
        )
    save_json(
        output / "os/scene_metadata.json",
        dict(
            images=scene_rows,
            gt_shape=list(gt.shape),
            gt_finite=bool(np.isfinite(gt).all()),
            gt_sha256=sha256_file(scene / "GT.txt"),
            gt_coordinates="x_optical y_optical x_sar y_sar, pixel units; Evaluation.py documents zero-based XY with configurable offsets",
            checks=checks,
            zero_vs_one_based_independently_certified=False,
            evaluation_script_sha256=sha256_file(scene / "code/Evaluation.py"),
            patch_scene_mapping=None,
            used_for_patch_training=False,
        ),
    )
    for name in ["GT.txt", "code/Evaluation.py"]:
        (output / "source" / Path(name).name).write_bytes((scene / name).read_bytes())
    print("SCENE_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
