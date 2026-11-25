"""Read spatial and land-cover strata for independent raw alignment inspection."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from sdgm.io import save_json, save_csv
from sdgm.preview import montage, panel, rgb, stretch, phase_shift


def main():
    config = json.loads(Path("configs/raw_rebuild.json").read_text())
    output = Path(config["output"])
    scenes = list(csv.DictReader((output / "scene_groups.csv").open()))
    raw = [
        json.loads(line)
        for line in (output / "raw_manifest.jsonl").read_text().splitlines()
    ]
    paths = {(row["parent_image_id"], row["modality"]): row["path"] for row in raw}
    audit = output / "base_audit"
    audit.mkdir(parents=True, exist_ok=True)
    all_rois = []
    overview = []
    for scene in scenes:
        if scene["split"] == "unassigned":
            continue
        parent = scene["parent_image_id"]
        with (
            rasterio.open(paths[parent, "optical"]) as optical,
            rasterio.open(paths[parent, "sar"]) as sar,
        ):
            full_rgb = np.moveaxis(
                optical.read(
                    [1, 2, 3],
                    out_shape=(3, 370, 556),
                    resampling=rasterio.enums.Resampling.average,
                ),
                0,
                -1,
            ).copy()
            full_sar = sar.read(
                1, out_shape=(370, 556), resampling=rasterio.enums.Resampling.average
            )
            selected = []
            label_path = paths.get((parent, "label"))
            if label_path:
                with rasterio.open(label_path) as image:
                    label = image.read(1)
                candidate = []
                for y in range(0, optical.height - 511, 512):
                    for x in range(0, optical.width - 511, 512):
                        counts = {
                            str(k): float((label[y : y + 512, x : x + 512] == k).mean())
                            for k in [10, 20, 30, 40, 50, 60, 70]
                        }
                        candidate.append((x, y, counts))
                classes = sorted(
                    [10, 20, 30, 40, 50, 60, 70],
                    key=lambda k: sum(v[str(k)] for _, _, v in candidate),
                    reverse=True,
                )
                for key in classes:
                    ranked = sorted(
                        candidate, key=lambda c: c[2][str(key)], reverse=True
                    )
                    for x, y, counts in ranked:
                        if counts[str(key)] < 0.10:
                            break
                        if all(np.hypot(x - a[0], y - a[1]) >= 1100 for a in selected):
                            selected.append((x, y, f"label_stratum_{key}", counts))
                            break
                    if len(selected) >= 3:
                        break
            fractions = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)]
            for fx, fy in fractions:
                if len(selected) >= 3:
                    break
                x = round((optical.width - 512) * fx)
                y = round((optical.height - 512) * fy)
                if all(np.hypot(x - a[0], y - a[1]) >= 800 for a in selected):
                    selected.append((x, y, "spatial_stratum_unknown_landcover", {}))
            panels = []
            for index, (x, y, stratum, counts) in enumerate(selected):
                window = Window(x, y, 512, 512)
                a = np.moveaxis(optical.read([1, 2, 3], window=window), 0, -1)
                b = sar.read(1, window=window)
                # Contrast stretch is for inspection PNGs only. Raw files stay intact.
                a = stretch(a)
                b = stretch(b)
                file = f"{parent}_roi{index}.png"
                montage(
                    [
                        panel(a, f"{parent} ROI{index} RGB x={x}, y={y}"),
                        panel(b, "SAR, same raw pixel window"),
                    ],
                    2,
                    audit / file,
                )
                gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
                proxy = phase_shift(gray, b)
                all_rois.append(
                    dict(
                        roi_id=f"{parent}/roi{index}",
                        parent_image_id=parent,
                        geo_group_id=scene["geo_group_id"],
                        split=scene["split"],
                        x=x,
                        y=y,
                        width=512,
                        height=512,
                        stratum=stratum,
                        label_fractions=counts,
                        file=file,
                        phase_proxy=proxy,
                        proxy_is_ground_truth=False,
                        checkpoint_status="awaiting_manual_review",
                    )
                )
                panels.extend(
                    [
                        panel(a, f"ROI{index} ({x},{y}) RGB", (320, 320)),
                        panel(b, "SAR", (320, 320)),
                    ]
                )
                cv2.rectangle(
                    full_rgb,
                    (round(x / 10), round(y / 10)),
                    (round((x + 512) / 10), round((y + 512) / 10)),
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    full_rgb,
                    str(index),
                    (round(x / 10) + 2, round(y / 10) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                )
            montage(panels, 2, audit / f"{parent}_strata.png")
            overview.extend(
                [
                    panel(full_rgb, parent + " " + scene["split"] + " RGB", (556, 370)),
                    panel(stretch(full_sar), "SAR (preview only)", (556, 370)),
                ]
            )
    montage(overview, 2, audit / "parent_overview.png")
    save_json(audit / "inspection_rois.json", all_rois)
    save_json(
        audit / "sampling_protocol.json",
        dict(
            parent_count=len(overview) // 2,
            rois=len(all_rois),
            fit_points=0,
            rule="three separated land-cover/spatial strata per geographically supported parent",
            classification_labels_used_for="stratification only; not correspondence truth",
            prealignment_performed=False,
            checkpoints="independent of any alignment fit; no forced point for unobservable terrain",
        ),
    )


if __name__ == "__main__":
    main()
