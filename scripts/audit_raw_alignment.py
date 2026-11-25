"""Publish native-base observations without converting screening into labels."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.raw_alignment import checkpoint_measurement, region_admission
from sdgm.raw_lineage import sha256_file


def render(points, roi, audit):
    image = Image.open(audit / roi["file"]).convert("RGB")
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for p in points:
        for k, key, uncertainty in [
            (0, "optical_xy", "optical_uncertainty_px"),
            (1, "sar_xy", "sar_uncertainty_px"),
        ]:
            x, y = p[key]
            x += k * 512
            y += 36
            r = p[uncertainty]
            draw.ellipse((x - r, y - r, x + r, y + r), outline="cyan", width=1)
            draw.line((x - 5, y, x + 5, y), fill="red", width=1)
            draw.line((x, y - 5, x, y + 5), fill="red", width=1)
            draw.text((x + 5, y + 3), p["id"], fill="cyan")
        # Center each modality's chip separately. Coordinates stay explicit.
        chips = Image.new("RGB", (768, 430), "white")
        cd = ImageDraw.Draw(chips)
        for k, key in [(0, "optical_xy"), (1, "sar_xy")]:
            x, y = p[key]
            crop = marked.crop(
                (x + k * 512 - 48, y + 36 - 48, x + k * 512 + 48, y + 36 + 48)
            ).resize((384, 384), Image.Resampling.NEAREST)
            chips.paste(crop, (384 * k, 46))
            cd.text(
                (384 * k + 4, 4),
                f"{p['id']} {'OPT' if k==0 else 'SAR'} native XY={p[key]}",
                fill="black",
            )
            cd.text(
                (384 * k + 4, 20),
                "Tentative visual check; cyan=uncertainty, red=estimate",
                fill="black",
            )
        chips.save(audit / f"{p['id']}_review.png")
    marked.save(audit / (roi["roi_id"].replace("/", "_") + "_checkpoints.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/raw_rebuild.json")
    parser.add_argument(
        "--observations", default="configs/raw_alignment_observations.json"
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    audit = output / "base_audit"
    observations = json.loads(Path(args.observations).read_text())
    rois = json.loads((audit / "inspection_rois.json").read_text())
    by_id = {r["roi_id"]: r for r in rois}
    points = [
        checkpoint_measurement(p, by_id[p["roi_id"]])
        for p in observations["checkpoints"]
    ]
    if len({p["id"] for p in points}) != len(points):
        raise ValueError("Duplicate checkpoint IDs")
    regions, admissions = [], {}
    for roi in rois:
        matches = [p for p in points if p["roi_id"] == roi["roi_id"]]
        admission = region_admission(
            matches, config["coarse_tolerance_px"], config["fine_target_error_px"]
        )
        admissions[roi["roi_id"]] = admission
        regions.append(
            {
                **roi,
                "checkpoint_status": (
                    "tentative_points" if matches else "no_point_selected"
                ),
                "visual_review_note": observations["roi_notes"][roi["roi_id"]],
                "coarse_allowed": False,
                "fine_allowed": False,
                "checkpoint_ids": [p["id"] for p in matches],
            }
        )
        if matches:
            render(matches, roi, audit)
    lineage_hashes = {
        name: sha256_file(output / name)
        for name in ["raw_manifest.jsonl", "scene_groups.csv", "splits.csv"]
    }
    admission = dict(
        prealignment_performed=False,
        fit_points=[],
        region_admissions=admissions,
        lineage_hashes=lineage_hashes,
        observations_sha256=sha256_file(args.observations),
        independent_human_review=False,
        geographic_independence_admitted=False,
        geographic_status="publisher-map evidence supports provisional groups; datum and coverage-error bound are not independently verified",
        coarse_status="not_admitted_to_8px_supervision",
        fine_status="1_to_2px_precision_not_certified",
        decision="Do not crop real training samples or migrate/retrain until native-base and geographic admission are adequate",
    )
    save_json(audit / "admission.json", admission)
    save_json(audit / "observations.json", observations)
    save_jsonl(audit / "checkpoint_records.jsonl", points)
    flat = [
        {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            for k, v in row.items()
        }
        for row in points
    ]
    save_csv(audit / "checkpoint_records.csv", flat)
    save_jsonl(audit / "region_reviews.jsonl", regions)
    summary = dict(
        parents_sampled=len({r["parent_image_id"] for r in rois}),
        regions_previewed=len(rois),
        reserved_testing_regions=sum(r["split"] == "testing" for r in rois),
        tentative_checkpoints=len(points),
        checkpoint_parent_count=len({p["parent_image_id"] for p in points}),
        nominal_residual_range_px=(
            [
                min(p["nominal_residual_px"] for p in points),
                max(p["nominal_residual_px"] for p in points),
            ]
            if points
            else None
        ),
        residual_plus_uncertainty_range_px=(
            [
                min(p["conservative_residual_plus_uncertainty_px"] for p in points),
                max(p["conservative_residual_plus_uncertainty_px"] for p in points),
            ]
            if points
            else None
        ),
        coarse_admitted_regions=0,
        fine_admitted_regions=0,
        independent_human_review=False,
        systematic_bias="not assessed: sparse object-dependent tentative points do not support a global correction",
        purpose="screening only; no true registration accuracy or spatial generalization certification",
        stages_3_4="generator can be tested on fixtures; real data generation and network migration deferred by admission gate",
    )
    save_json(audit / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
