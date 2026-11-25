"""Render the explicit preliminary visual picks and their uncertainty circles."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sdgm.io import load_datum, save_csv, save_json
from sdgm.preview import montage, panel, rgb, stretch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zzvr.json")
    parser.add_argument("--points", default="configs/whu_screening_points.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    evidence = json.loads(Path(args.points).read_text())
    output = Path(config["output"]) / "whu" / "diagnostics"
    rows = []
    crops = []
    vectors = []
    for point in evidence["points"]:
        split, sample = point["sample"].split("/")
        paths = sorted(Path(config["whu"]).glob(f"{split}/**/datum/{sample}.npy"))
        if len(paths) != 1:
            raise ValueError(f"Ambiguous datum for {point['sample']}")
        datum = load_datum(paths[0])
        images = [datum[0][..., 0], datum[4]]
        coords = [point["optical_xy"], point["sar_xy"]]
        pair = []
        detail = []
        accepted = point.get("accepted_for_screening", True)
        suffix = "" if accepted else " REJECTED"
        for name, image, xy in zip(["Optical", "Base SAR"], images, coords):
            picture = rgb(stretch(image)).copy()
            x, y = xy
            radius = point["uncertainty_px"]
            cv2.circle(picture, (x, y), radius, (255, 255, 0), 1)
            cv2.drawMarker(picture, (x, y), (255, 0, 0), cv2.MARKER_CROSS, 14, 1)
            pair.append(
                panel(picture, f"{point['id']}{suffix} {name} ({x},{y}) +/- {radius}px")
            )
            crop = picture[
                max(0, y - 48) : min(512, y + 49), max(0, x - 48) : min(512, x + 49)
            ]
            detail.append(panel(crop, point["id"] + suffix + " " + name, (256, 256)))
        montage(pair, 2, output / f"checkpoint_{point['id']}.png")
        crops.extend(detail)
        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        if accepted:
            vectors.append([dx, dy])
        row = {k: v for k, v in point.items() if k not in ["optical_xy", "sar_xy"]}
        row.update(
            optical_x=coords[0][0],
            optical_y=coords[0][1],
            sar_x=coords[1][0],
            sar_y=coords[1][1],
            dx_px=dx,
            dy_px=dy,
            distance_px=float(np.hypot(dx, dy)),
            status=(
                "approximate_visual_pick"
                if accepted
                else "rejected_ambiguous_candidate"
            ),
            accepted_for_screening=accepted,
            usable_as_accuracy_ground_truth=False,
        )
        rows.append(row)
    montage(crops, 4, output / "checkpoint_details.png")
    save_csv(output / "preliminary_checkpoints.csv", rows)
    save_json(output / "visual_review.json", evidence)
    accepted_rows = [row for row in rows if row["accepted_for_screening"]]
    save_json(
        output / "checkpoint_summary.json",
        dict(
            count=len(rows),
            accepted_screening_points=len(accepted_rows),
            rejected_candidates=len(rows) - len(accepted_rows),
            patches_with_accepted_points=len({r["sample"] for r in accepted_rows}),
            median_approximate_dx_dy=np.median(vectors, axis=0).tolist(),
            maximum_approximate_distance_px=max(
                r["distance_px"] for r in accepted_rows
            ),
            picking_radius_px=[
                min(r["uncertainty_px"] for r in accepted_rows),
                max(r["uncertainty_px"] for r in accepted_rows),
            ],
            interpretation="Observed offsets are within the estimated picking uncertainty. No subpixel accuracy claim is supported. Mountain local misalignment is unresolved.",
            source_scene_independence_verified=False,
        ),
    )


if __name__ == "__main__":
    main()
