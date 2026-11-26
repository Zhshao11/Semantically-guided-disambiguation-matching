"""Freeze epoch20 failure queries, then track those exact cases without reselection."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .coarse import spatial_candidates
from .io import save_json
from scripts.train_structure_smoke import prepare, geometry_for


@torch.no_grad()
def failure_diagnostics(model, validation, config, root, epoch):
    root = Path(root)
    output = root / "fixed_failures"
    output.mkdir(exist_ok=True)
    model.eval()
    frozen_path = root / "fixed_failure_queries.json"
    existing = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    if epoch != 20 and existing is None:
        raise ValueError("Failure query IDs must be frozen at epoch20")
    pairs = json.loads((root / "fixed_failure_pairs.json").read_text())["samples"]
    indices = {r["sample_id"]: i for i, r in enumerate(validation.records)}
    frozen = []
    observations = []
    for pair in pairs:
        sample_id = pair["sample_id"]
        item = validation[indices[sample_id]]
        batch = prepare(
            {
                key: value[None] if isinstance(value, torch.Tensor) else [value]
                for key, value in item.items()
            },
            next(model.parameters()).device,
        )
        pred = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, config)
        cand = spatial_candidates(
            pred["scores"],
            geom["target_valid"],
            model.target_pixel_xy,
            count=config["candidate_count"],
            nms_radius_px=config["nms_radius_px"],
        )
        query_xy = model.source_pixel_xy.cpu().numpy()
        gt = geom["gt_pixel_xy"][0].cpu().numpy()
        candidate_xy = cand["pixel_xy"][0].cpu().numpy()
        distances = np.linalg.norm(candidate_xy - gt[:, None], axis=-1)
        distances[~cand["valid"][0].cpu().numpy()] = np.inf
        nearest = distances.min(-1)
        valid = geom["supervised"][0].cpu().numpy()
        support = item["target_support"][0].numpy().astype("uint8")
        # Padded mask exposes both interpolation and native image extent boundaries.
        margin = cv2.distanceTransform(np.pad(support, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        rounded = np.rint(gt).astype(int)
        rounded = np.clip(rounded, 0, 255)
        margins = margin[rounded[:, 1], rounded[:, 0]]
        if existing is None:
            specs = [
                (
                    "near_gt_miss",
                    valid & (nearest > 8) & (nearest <= 16),
                    "smallest_distance",
                ),
                (
                    "similar_structure_candidate",
                    valid & (nearest > 16) & (nearest <= 64),
                    "median_distance",
                ),
                ("large_range_error", valid & (nearest > 64), "largest_distance"),
                (
                    "support_or_label_review",
                    valid & (nearest > 8),
                    "smallest_support_margin",
                ),
            ]
            used = set()
            choices = []
            for category, mask, rule in specs:
                candidates = np.array(
                    [int(i) for i in np.flatnonzero(mask) if int(i) not in used]
                )
                if not len(candidates):
                    continue
                if rule == "smallest_distance":
                    q = int(candidates[np.argmin(nearest[candidates])])
                elif rule == "largest_distance":
                    q = int(candidates[np.argmax(nearest[candidates])])
                elif rule == "smallest_support_margin":
                    q = int(candidates[np.argmin(margins[candidates])])
                else:
                    q = int(
                        candidates[
                            np.argsort(nearest[candidates])[len(candidates) // 2]
                        ]
                    )
                used.add(q)
                case = dict(
                    case_id=sample_id + "/Q" + str(q),
                    sample_id=sample_id,
                    family_id=pair["family_id"],
                    query_index=q,
                    proposed_category=category,
                    selection_rule=rule,
                    source_pixel_xy=query_xy[q].tolist(),
                    gt_pixel_xy=gt[q].tolist(),
                    selection_epoch=20,
                    selection_is_descriptive_not_data_filtering=True,
                )
                choices.append(case)
                frozen.append(case)
        else:
            choices = [r for r in existing["cases"] if r["sample_id"] == sample_id]
        rgb = ((item["optical"][0].numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
        sar = ((item["sar"][0].numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
        sheet = Image.new("RGB", (768, max(1, len(choices)) * 240), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, case in enumerate(choices):
            q = case["query_index"]
            if existing is not None:
                np.testing.assert_allclose(gt[q], case["gt_pixel_xy"], atol=0, rtol=0)
            record = dict(
                **case,
                epoch=epoch,
                supervised=bool(valid[q]),
                actual_support_margin_px=float(margins[q]),
                candidate_xy=candidate_xy[q].tolist(),
                candidate_valid=cand["valid"][0, q].cpu().tolist(),
                candidate_distances_px=distances[q].tolist(),
                top1_distance_px=float(distances[q, 0]),
                nearest_distance_px=float(nearest[q]),
                recall1_hit=bool(distances[q, 0] <= 8),
                recall4_hit=bool(nearest[q] <= 8),
                gt_outside=bool(geom["gt_outside"][0, q]),
                gt_invalid_support=bool(geom["gt_invalid_support"][0, q]),
                invalid_source=bool(geom["invalid_source"][0, q]),
                label_precision="base residual unknown; no independent landmark accuracy certificate",
            )
            observations.append(record)
            y = row_index * 240
            draw.text(
                (4, y + 2),
                f"Q{q} {case['proposed_category']} | nearest={nearest[q]:.2f}px top1={distances[q,0]:.2f}px margin={margins[q]:.1f}px",
                fill="black",
            )
            for col, (image, xy, label, color) in enumerate(
                [
                    (rgb, query_xy[q], "Optical source", "cyan"),
                    (sar, gt[q], "SAR synthetic label", "green"),
                    (sar, candidate_xy[q, 0], "SAR top1 candidate", "red"),
                ]
            ):
                x0, y0 = np.rint(xy).astype(int) - 48
                crop = (
                    Image.fromarray(image)
                    .convert("RGB")
                    .crop((x0, y0, x0 + 96, y0 + 96))
                    .resize((192, 192), Image.Resampling.NEAREST)
                )
                d = ImageDraw.Draw(crop)
                d.line((88, 96, 104, 96), fill=color, width=2)
                d.line((96, 88, 96, 104), fill=color, width=2)
                sheet.paste(crop, (col * 256, y + 40))
                draw.text(
                    (col * 256 + 4, y + 22),
                    label + f" XY={np.round(xy,1)}",
                    fill="black",
                )
        sheet.save(output / f'epoch{epoch:02d}_{sample_id.rsplit("/",1)[-1]}.png')
    if existing is None:
        save_json(
            frozen_path,
            dict(
                selection_epoch=20,
                pair_source="fixed six failures inherited from previous report",
                selection="descriptive source query screening only; same IDs used at epochs25/30",
                proposed_categories_are_not_all_visually_confirmed=True,
                all_samples_retained=True,
                cases=frozen,
            ),
        )
    save_json(output / f"epoch{epoch:02d}_query_observations.json", observations)
    return observations
