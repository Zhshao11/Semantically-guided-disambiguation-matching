"""Full-decoding audits for the three actual datasets available on zzvr."""

import csv
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform_bounds

from .geometry import flow_to_map, homogeneous, solve_affine, transform_points
from .io import load_datum, read_flo, save_json, save_jsonl

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def issue(issues, dataset, sample, severity, code, detail):
    issues.append(
        dict(
            dataset=dataset, sample=sample, severity=severity, code=code, detail=detail
        )
    )


def raster_record(path):
    """Decode every block, retaining per-band validity and range statistics."""
    path = Path(path)
    with rasterio.open(path) as image:
        n = image.count
        finite = np.zeros(n, dtype=np.int64)
        zero = finite.copy()
        masked = finite.copy()
        lower, upper = np.full(n, np.inf), np.full(n, -np.inf)
        for _, window in image.block_windows(1):
            array = image.read(window=window)
            valid = np.isfinite(array)
            mask = image.read_masks(window=window) > 0
            finite += valid.sum((1, 2))
            zero += (array == 0).sum((1, 2))
            masked += (~mask).sum((1, 2))
            for band in range(n):
                values = array[band][valid[band] & mask[band]]
                if values.size:
                    lower[band] = min(lower[band], float(values.min()))
                    upper[band] = max(upper[band], float(values.max()))
        count = image.height * image.width
        return dict(
            path=str(path),
            bytes=path.stat().st_size,
            width=image.width,
            height=image.height,
            bands=n,
            dtype=list(image.dtypes),
            nodata=None if image.nodata is None else str(image.nodata),
            crs=None if image.crs is None else image.crs.to_string(),
            transform=list(image.transform)[:6],
            bounds=list(image.bounds),
            georeferenced=image.crs is not None
            and image.transform != rasterio.Affine.identity(),
            finite_pixels=finite.tolist(),
            nonfinite_pixels=(count - finite).tolist(),
            zero_fraction=(zero / count).tolist(),
            masked_fraction=(masked / count).tolist(),
            minimum=[float(x) if np.isfinite(x) else None for x in lower],
            maximum=[float(x) if np.isfinite(x) else None for x in upper],
        )


def coverage(first, second):
    if not first["georeferenced"] or not second["georeferenced"]:
        return dict(status="unavailable_missing_georeference", overlap_fraction=None)
    a, b = first["bounds"], second["bounds"]
    if first["crs"] != second["crs"]:
        b = transform_bounds(second["crs"], first["crs"], *b, densify_pts=21)
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    denominator = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return dict(
        status="ok",
        overlap_fraction=intersection / denominator if denominator > 0 else 0,
        same_crs=first["crs"] == second["crs"],
        same_transform=np.allclose(first["transform"], second["transform"]),
    )


def raster_issues(record, dataset, sample, issues):
    if any(record["nonfinite_pixels"]):
        issue(issues, dataset, sample, "error", "nonfinite_pixels", record["path"])
    if any(value is None for value in record["minimum"]):
        issue(issues, dataset, sample, "error", "empty_valid_band", record["path"])
    if any(a == b for a, b in zip(record["minimum"], record["maximum"])):
        issue(issues, dataset, sample, "warning", "constant_band", record["path"])


def whu_sample(job):
    split, sample, paths = job
    issues = []
    record = dict(
        id=f"{split}/{sample}",
        split=split,
        sample=sample,
        paths={k: str(v) for k, v in paths.items()},
        status="ok",
    )
    try:
        optical, sar = raster_record(paths["optical"]), raster_record(paths["sar"])
        record.update(optical=optical, sar=sar, coverage=coverage(optical, sar))
        for raster in [optical, sar]:
            raster_issues(raster, "WHU", record["id"], issues)
        if (optical["width"], optical["height"], optical["bands"]) != (512, 512, 1):
            issue(
                issues,
                "WHU",
                record["id"],
                "error",
                "unexpected_optical_layout",
                str(optical),
            )
        if (sar["width"], sar["height"], sar["bands"]) != (512, 512, 1):
            issue(
                issues, "WHU", record["id"], "error", "unexpected_sar_layout", str(sar)
            )
        datum = load_datum(paths["datum"])
        flow = read_flo(paths["flow"])
        opt = cv2.imread(str(paths["optical"]), cv2.IMREAD_UNCHANGED)
        warped = cv2.imread(str(paths["sar"]), cv2.IMREAD_UNCHANGED)
        if not (datum[0].shape == (512, 512, 2) and datum[4].shape == (512, 512)):
            raise ValueError("Unexpected datum embedded image layout")
        if not (
            np.array_equal(datum[0][..., 0], opt)
            and np.array_equal(datum[0][..., 1], warped)
        ):
            issue(
                issues,
                "WHU",
                record["id"],
                "error",
                "datum_image_mismatch",
                "Embedded channels differ from TIFF",
            )
        if not np.isfinite(datum[4]).all():
            raise ValueError("Nonfinite base SAR")
        if flow.shape != (512, 512, 2) or not np.isfinite(flow).all():
            raise ValueError("Invalid flow dimensions or values")
        mapping = flow_to_map(flow)
        yy, xx = np.indices((512, 512))
        points = np.stack([xx, yy], -1)[::32, ::32].reshape(-1, 2)
        targets = mapping[::32, ::32].reshape(-1, 2)
        metadata_matrix = homogeneous(datum[2])
        metadata_error = float(
            np.max(
                np.linalg.norm(
                    transform_points(metadata_matrix, points) - targets, axis=1
                )
            )
        )
        if metadata_error > 0.001:
            issue(
                issues,
                "WHU",
                record["id"],
                "error",
                "flow_metadata_direction_mismatch",
                str(metadata_error),
            )
        fitted = solve_affine(points, targets)
        if not fitted.valid:
            raise ValueError(f"Flow affine fit failed: {fitted.reason}")
        record["flow"] = dict(
            shape=list(flow.shape),
            dtype=str(flow.dtype),
            direction="base_grid_to_warped_SAR_xy_displacement",
            affine=fitted.matrix.tolist(),
            affine_fit_rmse_px=fitted.rmse,
        )
        record["flow"]["metadata_forward_max_error_px"] = metadata_error
        if fitted.rmse > 0.01:
            issue(
                issues,
                "WHU",
                record["id"],
                "error",
                "nonaffine_flow",
                f"fit RMSE {fitted.rmse}",
            )
        valid = (
            (mapping[..., 0] >= 1)
            & (mapping[..., 0] < 510)
            & (mapping[..., 1] >= 1)
            & (mapping[..., 1] < 510)
        )
        recovered = cv2.remap(
            warped,
            mapping[..., 0].astype("float32"),
            mapping[..., 1].astype("float32"),
            cv2.INTER_LINEAR,
        )
        mae = (
            float(np.abs(recovered.astype(float) - datum[4])[valid].mean())
            if valid.any()
            else None
        )
        record["flow"]["valid_overlap_fraction"] = float(valid.mean())
        record["flow"]["base_reconstruction_mae_uint8"] = mae
        record["datum"] = dict(
            shapes=[list(np.shape(a)) for a in datum],
            dtypes=[str(np.asarray(a).dtype) for a in datum],
            base_sar_min=int(datum[4].min()),
            base_sar_max=int(datum[4].max()),
        )
        if not valid.any():
            issue(
                issues,
                "WHU",
                record["id"],
                "error",
                "no_flow_overlap",
                "No valid target pixels",
            )
    except Exception as error:
        record["status"] = "error"
        issue(issues, "WHU", record["id"], "error", "read_or_schema_error", str(error))
    if any(x["severity"] == "error" for x in issues):
        record["status"] = "error"
    return record, issues


def audit_whu(root, output, workers):
    root, output = Path(root), Path(output)
    issues, jobs, split_counts = [], [], {}
    layouts = {
        "optical": ("opt", "_opt.tif"),
        "sar": ("sar_warped", "_sar_warped.tif"),
        "datum": ("datum", ".npy"),
        "flow": ("truth_flow", ".flo"),
    }
    bases = sorted(root.rglob("opt"))
    if not bases:
        issue(issues, "WHU", "dataset", "error", "missing_layout", str(root))
    for optical_dir in bases:
        base = optical_dir.parent
        split = "training" if "training" in base.relative_to(root).parts else "testing"
        indexes = {
            key: {p.name[: -len(suffix)]: p for p in (base / folder).glob("*" + suffix)}
            for key, (folder, suffix) in layouts.items()
        }
        all_ids = set.union(*(set(x) for x in indexes.values()))
        split_counts[split] = {key: len(value) for key, value in indexes.items()}
        for sample in sorted(all_ids):
            missing = [key for key, index in indexes.items() if sample not in index]
            if missing:
                issue(
                    issues,
                    "WHU",
                    split + "/" + sample,
                    "error",
                    "missing_corresponding_files",
                    str(missing),
                )
            else:
                jobs.append((split, sample, {k: v[sample] for k, v in indexes.items()}))
    issue(
        issues,
        "WHU",
        "dataset",
        "warning",
        "derived_grayscale_patch_release",
        "Available release is a warped 512x512 grayscale derivative. Original 100-scene RGBNIR/SAR/land-use labels and geographic coverage cannot be certified from it.",
    )
    issue(
        issues,
        "WHU",
        "dataset",
        "warning",
        "missing_parent_scene_identity",
        "Patch IDs do not identify source scenes. Source-scene independence and spatial train/test leakage remain unverified.",
    )
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (record, found) in enumerate(pool.map(whu_sample, jobs), 1):
            records.append(record)
            issues.extend(found)
            if index % 500 == 0:
                print(f"WHU decoded {index}/{len(jobs)}", flush=True)
    missing_geo = sum(
        not r.get("optical", {}).get("georeferenced", False) for r in records
    )
    if missing_geo:
        issue(
            issues,
            "WHU",
            "dataset",
            "warning",
            "missing_georeference",
            f"{missing_geo}/{len(records)} pairs lack CRS/geotransform. Coverage limited to pixel extent and synthetic-flow overlap.",
        )
    return finish(
        "WHU",
        records,
        issues,
        output,
        dict(
            root=str(root),
            split_counts=split_counts,
            original_release_verified=False,
            full_decoding=True,
        ),
    )


def audit_srif(root, output):
    root = Path(root)
    issues = []
    records = []
    expected = {f"gt_{i}.txt" for i in range(1, 201)} | {
        f"pair{i}_{j}.jpg" for i in range(1, 201) for j in [1, 2]
    }
    actual = {p.name for p in root.iterdir() if p.is_file()} if root.is_dir() else set()
    for name in sorted(expected - actual):
        issue(issues, "SRIF", name, "error", "missing_file", name)
    for name in sorted(actual - expected):
        issue(issues, "SRIF", name, "warning", "unexpected_file", name)
    for index in range(1, 201):
        record = dict(id=str(index), status="ok")
        try:
            images = []
            for side in [1, 2]:
                path = root / f"pair{index}_{side}.jpg"
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"Cannot decode {path}")
                images.append(
                    dict(
                        path=str(path),
                        shape=list(image.shape),
                        dtype=str(image.dtype),
                        minimum=int(image.min()),
                        maximum=int(image.max()),
                    )
                )
            matrix = homogeneous(np.loadtxt(root / f"gt_{index}.txt"))
            determinant = float(np.linalg.det(matrix[:2, :2]))
            condition = float(np.linalg.cond(matrix[:2, :2]))
            if abs(determinant) < 1e-10 or condition > 1e8:
                raise ValueError("Degenerate affine label")
            record.update(
                images=images,
                gt_path=str(root / f"gt_{index}.txt"),
                matrix=matrix[:2].tolist(),
                determinant=determinant,
                condition=condition,
                direction="image_1_to_image_2_xy; assign optical/SAR only after modality review",
            )
        except Exception as error:
            record["status"] = "error"
            issue(
                issues, "SRIF", str(index), "error", "read_or_matrix_error", str(error)
            )
        records.append(record)
    return finish(
        "SRIF",
        records,
        issues,
        output,
        dict(
            root=str(root),
            expected_pairs=200,
            expected_files=600,
            actual_files=len(actual),
            full_decoding=True,
        ),
    )


def audit_spacenet(root, output):
    root = Path(root)
    records = []
    issues = []
    optics = sorted(root.rglob("*optical*.tif"))
    expected = {
        "02_optical_train_01.tif",
        "02_optical_train_02.tif",
        "03_optical_train_01.tif",
        "02_optical_publictest.tif",
        "03_optical_publictest.tif",
    }
    for name in sorted(expected - {p.name for p in optics}):
        issue(issues, "SpaceNet9", name, "error", "missing_public_image", name)
    for path in optics:
        sample = path.stem.replace("optical_", "")
        record = dict(id=sample, status="ok")
        try:
            sar_path = path.with_name(path.name.replace("optical", "sar"))
            csv_path = path.with_name(
                path.name.replace("optical", "tiepoints")
            ).with_suffix(".csv")
            optical, sar = raster_record(path), raster_record(sar_path)
            record.update(
                optical=optical,
                sar=sar,
                coverage=coverage(optical, sar),
                tiepoint_path=str(csv_path),
            )
            for image in [optical, sar]:
                raster_issues(image, "SpaceNet9", sample, issues)
            if optical["bands"] != 3 or sar["bands"] != 1:
                issue(
                    issues,
                    "SpaceNet9",
                    sample,
                    "error",
                    "unexpected_bands",
                    f"Optical {optical['bands']}, SAR {sar['bands']}",
                )
            if not optical["georeferenced"] or not sar["georeferenced"]:
                issue(
                    issues,
                    "SpaceNet9",
                    sample,
                    "error",
                    "missing_georeference",
                    "Required by official scoring",
                )
            if csv_path.exists():
                with csv_path.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = ["sar_row", "sar_col", "optical_row", "optical_col"]
                    if reader.fieldnames != fields:
                        raise ValueError(f"Unexpected CSV fields {reader.fieldnames}")
                    values = np.array(
                        [[float(row[k]) for k in fields] for row in reader], dtype=float
                    ).reshape(-1, 4)
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError("Empty or nonfinite tie points")
                inside = (
                    (values >= 0).all(1)
                    & (values[:, 0] < sar["height"])
                    & (values[:, 1] < sar["width"])
                    & (values[:, 2] < optical["height"])
                    & (values[:, 3] < optical["width"])
                )
                record["tiepoints"] = dict(
                    count=len(values),
                    outside_count=int((~inside).sum()),
                    duplicate_count=len(values) - len(np.unique(values, axis=0)),
                    fractional_coordinate_count=int(
                        np.any(values != np.floor(values), axis=1).sum()
                    ),
                    fields=fields,
                    coordinate_order="row,col; convert to x=col,y=row",
                )
                unique_values, counts = np.unique(values, axis=0, return_counts=True)
                duplicate_groups = [
                    (np.flatnonzero(np.all(values == value, axis=1)) + 2).tolist()
                    for value in unique_values[counts > 1]
                ]
                record["tiepoints"]["duplicate_csv_line_groups"] = duplicate_groups
                if duplicate_groups:
                    issue(
                        issues,
                        "SpaceNet9",
                        sample,
                        "warning",
                        "duplicate_tiepoints",
                        f"Identical point rows at CSV lines {duplicate_groups}. Original CSV retained.",
                    )
                if not inside.all():
                    issue(
                        issues,
                        "SpaceNet9",
                        sample,
                        "error",
                        "out_of_bounds_tiepoints",
                        str(np.where(~inside)[0].tolist()),
                    )
            elif "publictest" in path.name:
                record["tiepoints"] = dict(status="officially_withheld")
                issue(
                    issues,
                    "SpaceNet9",
                    sample,
                    "info",
                    "officially_withheld_test_labels",
                    "Public test images have no released CSV",
                )
            else:
                raise ValueError("Missing training CSV")
        except Exception as error:
            record["status"] = "error"
            issue(
                issues, "SpaceNet9", sample, "error", "read_or_schema_error", str(error)
            )
        if any(
            row["sample"] == sample and row["severity"] == "error" for row in issues
        ):
            record["status"] = "error"
        records.append(record)
    issue(
        issues,
        "SpaceNet9",
        "dataset",
        "info",
        "private_split_unavailable",
        "Final private AOI images and labels are not in public release",
    )
    return finish(
        "SpaceNet9",
        records,
        issues,
        output,
        dict(
            root=str(root),
            full_decoding=True,
            complete_public_image_release=len(expected & {p.name for p in optics}) == 5,
            complete_competition_release=False,
        ),
    )


def finish(dataset, records, issues, output, extra):
    output = Path(output) / dataset.lower()
    save_jsonl(output / "manifest.jsonl", records)
    save_jsonl(output / "anomalies.jsonl", issues)
    summary = dict(
        dataset=dataset,
        pairs=len(records),
        decoded_ok=sum(r["status"] == "ok" for r in records),
        issue_severity=dict(Counter(i["severity"] for i in issues)),
        issue_codes=dict(Counter(i["code"] for i in issues)),
        **extra,
    )
    save_json(output / "summary.json", summary)
    return summary
