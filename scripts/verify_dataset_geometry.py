"""Check transform direction against real WHU payloads and independent raster warps."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sdgm.geometry import homogeneous, transform_points
from sdgm.io import load_datum, read_flo, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zzvr.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    selected = json.loads(
        (output / "whu/diagnostics/selected_samples.json").read_text()
    )
    results = []
    for row in selected:
        split, sample = row["id"].split("/")
        path = next(Path(config["whu"]).glob(f"{split}/**/datum/{sample}.npy"))
        datum = load_datum(path)
        flow = read_flo(path.parent.parent / "truth_flow" / f"{sample}.flo")
        yy, xx = np.indices((512, 512))
        points = np.stack([xx, yy], -1).astype(float)
        prediction = transform_points(datum[2], points)
        flow_error = float(
            np.max(np.linalg.norm(prediction - (points + flow), axis=-1))
        )
        displaced = transform_points(datum[2], datum[1]) - datum[1]
        corner_error = float(np.max(np.abs(displaced - datum[5])))
        rebuilt = cv2.warpAffine(datum[4], np.array(datum[2]), (512, 512))
        rebuild_mae = float(np.abs(rebuilt.astype(float) - datum[0][..., 1]).mean())
        forward = points + flow
        inverse = transform_points(np.linalg.inv(homogeneous(datum[2])), points)
        common = np.ones((512, 512), bool)
        for mapping in [forward, inverse]:
            common &= (
                (mapping[..., 0] >= 1)
                & (mapping[..., 0] < 510)
                & (mapping[..., 1] >= 1)
                & (mapping[..., 1] < 510)
            )
        errors = []
        for mapping in [forward, inverse]:
            image = cv2.remap(
                datum[0][..., 1],
                mapping[..., 0].astype("float32"),
                mapping[..., 1].astype("float32"),
                cv2.INTER_LINEAR,
            )
            errors.append(float(np.abs(image.astype(float) - datum[4])[common].mean()))
        valid = (
            flow_error < 0.001
            and corner_error < 0.001
            and rebuild_mae < 2
            and errors[0] < errors[1]
        )
        results.append(
            dict(
                id=row["id"],
                forward_flow_max_error_px=flow_error,
                metadata_corner_displacement_max_error_px=corner_error,
                base_to_warped_raster_mae_uint8=rebuild_mae,
                correct_forward_sampling_mae_uint8=errors[0],
                wrong_inverse_sampling_mae_uint8=errors[1],
                common_valid_pixels=int(common.sum()),
                passed=valid,
            )
        )
    report = dict(
        passed=all(r["passed"] for r in results),
        samples=len(results),
        results=results,
        distinction="Checks synthetic base-SAR-to-warped-SAR geometry. Native optical-to-base-SAR alignment needs independent visual checkpoints.",
    )
    save_json(output / "geometry/dataset_geometry.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
