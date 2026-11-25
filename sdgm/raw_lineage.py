"""Raw-file identity, corroborated coverage and geographic grouping.

Map-sheet names propose coverage. A distinct publisher figure independently
corroborates each admitted file through image-content retrieval. Missing
corroboration remains unresolved and cannot be assigned to a data split.
"""

import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as transform_geometry, unary_union

from .audit import raster_record


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nominal_sheet_bounds(parent_id):
    """Nominal 1:50,000 sheet proposal, never an intrinsic raster transform."""
    match = re.fullmatch(r"N([A-Z])(\d{2})E(\d{3})(\d{3})", parent_id)
    if match is None:
        return None
    band, zone, row, column = match.groups()
    row, column = int(row), int(column)
    if not (1 <= row <= 24 and 1 <= column <= 24):
        return None
    west = (int(zone) - 31) * 6 + (column - 1) / 4
    north = (ord(band) - ord("A") + 1) * 4 - (row - 1) / 6
    return [west, north - 1 / 6, west + 1 / 4, north]


def figure_window(bounds, calibration):
    west, south, east, north = bounds
    sx = (calibration["longitude_114_x"] - calibration["longitude_111_x"]) / 3
    sy = (calibration["latitude_30_y"] - calibration["latitude_33_y"]) / 3
    x = round(calibration["longitude_111_x"] + (west - 111) * sx)
    y = round(calibration["latitude_33_y"] + (33 - north) * sy)
    return [x, y, round((east - west) * sx), round((north - south) * sy)]


def normalized_template(image):
    image = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1.2)[3:-3, 3:-3]
    image = image.ravel().astype(np.float64)
    image -= image.mean()
    norm = np.linalg.norm(image)
    return image / norm if norm > 1e-9 else np.zeros_like(image)


def corroborate_coverage(optical_paths, parent_ids, config):
    """Compare content with all visible publisher-map cells, with negatives."""
    figure = cv2.imread(config["publisher_figure"], cv2.IMREAD_GRAYSCALE)
    if figure is None:
        raise FileNotFoundError(config["publisher_figure"])
    templates = []
    for parent in sorted(parent_ids):
        bounds = nominal_sheet_bounds(parent)
        if bounds is None:
            continue
        x, y, width, height = figure_window(bounds, config["figure_calibration"])
        patch = figure[y : y + height, x : x + width]
        if patch.shape != (height, width) or (patch < 215).mean() <= 0.9:
            continue
        # The figure uses 51x34 pixel cells after digitizing the degree ticks.
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell = figure[y + dy : y + dy + 34, x + dx : x + dx + 51]
                if cell.shape == (34, 51):
                    templates.append((parent, dx, dy, normalized_template(cell)))
    matrix = (
        np.stack([row[3] for row in templates]) if templates else np.empty((0, 1260))
    )
    result = {}
    for parent, path in sorted(optical_paths.items()):
        with rasterio.open(path) as image:
            bands = image.read(
                [1, 2, 3],
                out_shape=(3, 34, 51),
                resampling=rasterio.enums.Resampling.average,
            )
        gray = 0.299 * bands[0] + 0.587 * bands[1] + 0.114 * bands[2]
        query = normalized_template(gray)
        all_scores = matrix @ query if len(matrix) else []
        per_parent = {}
        for (candidate, dx, dy, _), score in zip(templates, all_scores):
            if candidate not in per_parent or score > per_parent[candidate][0]:
                per_parent[candidate] = (float(score), dx, dy)
        ranked = sorted(
            per_parent, key=lambda candidate: per_parent[candidate][0], reverse=True
        )
        expected = per_parent.get(parent)
        best_other = max(
            (per_parent[candidate][0] for candidate in ranked if candidate != parent),
            default=-1,
        )
        supported = bool(
            expected
            and ranked[0] == parent
            and expected[0] >= config["coverage_match_minimum_ncc"]
            and expected[0] - best_other >= config["coverage_match_minimum_margin"]
        )
        result[parent] = dict(
            status=(
                "publisher_figure_supported"
                if supported
                else "geographic_independence_unconfirmed"
            ),
            nominal_bounds_wgs84=nominal_sheet_bounds(parent),
            expected_cell_ncc=expected[0] if expected else None,
            nearest_other_cell_ncc=best_other,
            expected_is_top1=bool(ranked and ranked[0] == parent),
            expected_offset_pixels=list(expected[1:]) if expected else None,
            content_reference=config["publisher_figure_url"],
            content_reference_sha256=sha256_file(config["publisher_figure"]),
            coverage_uncertainty_m=(
                config["coverage_uncertainty_m"] if supported else None
            ),
            coverage_basis="map-sheet proposal corroborated by raw RGB content in independently published geographic coverage figure",
            map_datum_status="unspecified_by_figure; WGS84 is an approximate grouping assumption",
            buffer_status="declared conservative digitization allowance, not a measured geolocation confidence bound",
            uncertainties="TIFF has no intrinsic CRS. Figure-based coverage supports kilometre-scale grouping, not pixel-level georeferencing or optical-SAR alignment accuracy. Map datum is unspecified; 3 km is an assumption, not measured error. Geographic disjointness is conditional on these coverage assumptions.",
        )
    return result


def inventory_raw(catalog, root):
    root = Path(root)
    rows = []
    lookup = {(row["parent_image_id"], row["modality"]): row for row in catalog}
    for modality in ["optical", "sar", "label"]:
        for path in sorted((root / modality).glob("*.tif")):
            source = lookup.get((path.stem, modality))
            if source is None:
                raise ValueError(f"File is absent from source catalog: {path}")
            record = raster_record(path)
            if record["crs"] is None:
                record["transform"] = None
                record["bounds"] = None
            record.update(
                raw_file_id=f"WHU-original/{modality}/{path.stem}",
                source_file_id=source["source_id"],
                source_url=source["url"],
                parent_image_id=path.stem,
                modality=modality,
                sha256=sha256_file(path),
            )
            with rasterio.open(path) as image:
                record["band_color_interpretation"] = [
                    value.name for value in image.colorinterp
                ]
                record["band_descriptions"] = list(image.descriptions)
                record["gcps_count"] = len(image.gcps[0])
                record["rpcs_present"] = image.rpcs is not None
                record["pixel_extent"] = [0, 0, image.width, image.height]
                record["spectral_mapping"] = (
                    {
                        "rgb_one_based": [1, 2, 3],
                        "rgb_basis": "TIFF red/green/blue color interpretation; verify before use",
                        "nir_one_based": [4],
                        "nir_basis": "publisher RGB+NIR description, fourth TIFF band undefined; preserved unused",
                    }
                    if modality == "optical"
                    else None
                )
                record["supporting_files"] = [
                    {"path": str(Path(name).absolute()), "sha256": sha256_file(name)}
                    for name in image.files
                    if Path(name).absolute() != path.absolute()
                ]
            paired_modality = "sar" if modality == "optical" else "optical"
            paired = lookup.get((path.stem, paired_modality))
            record["paired_source_file_id"] = paired["source_id"] if paired else None
            counterpart = root / paired_modality / path.name
            record["paired_local_present"] = counterpart.is_file()
            record["paired_local_path"] = (
                str(counterpart) if counterpart.is_file() else None
            )
            record["pairing_status"] = (
                "local_counterpart_present"
                if counterpart.is_file()
                else "catalog_only_counterpart_missing_locally"
            )
            record["pairing_basis"] = (
                "same release parent ID, dimensions checked separately; alignment requires checkpoint audit"
            )
            record["original_satellite_product_id"] = None
            rows.append(record)
    return rows


def group_scenes(raw_rows, coverage, config):
    by_parent = {}
    for row in raw_rows:
        by_parent.setdefault(row["parent_image_id"], {})[row["modality"]] = row
    local_crs = "+proj=aeqd +lat_0=31 +lon_0=112.5 +datum=WGS84 +units=m +no_defs"
    project = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    polygons = {}
    for parent, files in by_parent.items():
        evidence = coverage.get(parent, {})
        if (
            evidence.get("status") != "publisher_figure_supported"
            or not {"optical", "sar"} <= files.keys()
        ):
            continue
        if (files["optical"]["width"], files["optical"]["height"]) != (
            files["sar"]["width"],
            files["sar"]["height"],
        ):
            continue
        polygons[parent] = transform_geometry(
            project, box(*evidence["nominal_bounds_wgs84"])
        )
    pending = set(polygons)
    components = []
    buffered = {
        parent: polygon.buffer(config["coverage_uncertainty_m"])
        for parent, polygon in polygons.items()
    }
    while pending:
        first = min(pending)
        pending.remove(first)
        component = [first]
        frontier = [first]
        while frontier:
            current = frontier.pop()
            linked = sorted(
                parent
                for parent in pending
                if buffered[current].intersects(buffered[parent])
            )
            for parent in linked:
                pending.remove(parent)
                component.append(parent)
                frontier.append(parent)
        components.append(sorted(component))
    assignments = {}
    split_rows = []
    first_limit, second_limit = config["geographic_split_longitudes"]
    for component in sorted(components):
        group = "G-" + hashlib.sha256("|".join(component).encode()).hexdigest()[:12]
        longitude = float(
            np.mean(
                [
                    (
                        coverage[parent]["nominal_bounds_wgs84"][0]
                        + coverage[parent]["nominal_bounds_wgs84"][2]
                    )
                    / 2
                    for parent in component
                ]
            )
        )
        split = (
            "training"
            if longitude < first_limit
            else "validation" if longitude < second_limit else "testing"
        )
        for parent in component:
            assignments[parent] = (group, split)
        split_rows.append(
            dict(
                geo_group_id=group,
                split=split,
                parent_image_ids="|".join(component),
                basis="whole connected component of corroborated geographic footprints with 3km positional uncertainty buffers",
                independence_status="not_certified_conditional_publisher_coverage",
                split_status="provisional_assignment_only",
                split_rule="predeclared longitude bands; no performance-based reassignment",
            )
        )
    unresolved = sorted(set(by_parent) - set(assignments))
    if unresolved:
        split_rows.append(
            dict(
                geo_group_id="GEOGRAPHY_UNCONFIRMED",
                split="unassigned",
                parent_image_ids="|".join(unresolved),
                basis="missing coverage corroboration or incomplete pair",
                independence_status="unconfirmed",
                split_status="unassigned",
                split_rule="no random fallback",
            )
        )
    scene_rows = []
    for parent, files in sorted(by_parent.items()):
        evidence = coverage.get(parent, {})
        group, split = assignments.get(parent, ("GEOGRAPHY_UNCONFIRMED", "unassigned"))
        scene_rows.append(
            dict(
                parent_image_id=parent,
                geo_group_id=group,
                split=split,
                split_status=(
                    "provisional_assignment_only"
                    if parent in assignments
                    else "unassigned"
                ),
                geographic_independence_status="not_certified",
                optical_raw_file_id=files.get("optical", {}).get("raw_file_id"),
                sar_raw_file_id=files.get("sar", {}).get("raw_file_id"),
                coverage_bounds_wgs84=json.dumps(evidence.get("nominal_bounds_wgs84")),
                coverage_basis=evidence.get("coverage_basis", "unavailable"),
                geographic_evidence_status=evidence.get("status", "unconfirmed"),
                uncertainty_m=evidence.get("coverage_uncertainty_m"),
                grouping_reason=(
                    "intersection of buffered corroborated footprints; different parent IDs can share one group"
                    if parent in assignments
                    else "unconfirmed; pooled without a split"
                ),
                uncertainties=evidence.get("uncertainties", "coverage unavailable"),
            )
        )
    separations = []
    for i, first in enumerate(components):
        for second in components[i + 1 :]:
            if assignments[first[0]][1] != assignments[second[0]][1]:
                distance = unary_union([buffered[p] for p in first]).distance(
                    unary_union([buffered[p] for p in second])
                )
                if distance <= 0:
                    raise AssertionError("Overlapping geographic groups cross splits")
                separations.append(float(distance))
    return (
        scene_rows,
        split_rows,
        dict(
            projected_grouping_crs=local_crs,
            coverage_buffer_m=config["coverage_uncertainty_m"],
            minimum_cross_split_buffered_separation_m=(
                min(separations) if separations else None
            ),
            groups=len(components),
            unresolved_parents=len(unresolved),
            intrinsic_raster_georeferencing_recovered=False,
        ),
    )
