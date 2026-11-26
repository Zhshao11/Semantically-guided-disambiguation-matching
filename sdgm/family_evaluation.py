"""Native candidate metrics and per-pair distributions for frozen family folds."""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .coarse import neighborhood_mass_loss, spatial_candidates
from .io import save_csv, save_json
from scripts.train_structure_smoke import (
    prepare,
    geometry_for,
    candidate_metrics,
    visualize_batch,
)


def quantiles(values):
    return (
        {
            str(q): float(np.percentile(values, q))
            for q in [0, 5, 25, 50, 75, 90, 95, 99, 100]
        }
        if len(values)
        else {}
    )


@torch.no_grad()
def evaluate_family(
    model, dataset, config, device, output, tag, details=False, failures=False
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    all_nearest = []
    total_loss = 0.0
    total_queries = 0
    mask_counts = dict(
        gt_outside=0, empty_positive=0, invalid_source=0, gt_invalid_support=0
    )
    first_storage = []
    nearest_storage = []
    support_storage = []
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    for batch in loader:
        batch = prepare(batch, device)
        prediction = model(batch["optical"], batch["sar"])
        if not all(torch.isfinite(value).all() for value in prediction.values()):
            raise FloatingPointError("Nonfinite evaluation prediction")
        geometry = geometry_for(model, batch, config)
        candidates = spatial_candidates(
            prediction["scores"],
            geometry["target_valid"],
            model.target_pixel_xy,
            count=config["candidate_count"],
            nms_radius_px=config["nms_radius_px"],
        )
        coordinates = candidates["pixel_xy"][candidates["valid"]]
        if not ((coordinates >= 0).all() and (coordinates <= 255).all()):
            raise AssertionError("Candidate outside native extent")
        actual_support = geometry["target_valid"][:, None, :].expand(
            -1, candidates["indices"].shape[1], -1
        )
        if not actual_support.gather(-1, candidates["indices"].clamp_min(0))[
            candidates["valid"]
        ].all():
            raise AssertionError("Candidate outside actual valid SAR support")
        for key in mask_counts:
            mask_counts[key] += int(geometry[key].sum())
        for index, sample_id in enumerate(batch["id"]):
            one = {key: value[index : index + 1] for key, value in geometry.items()}
            chosen = {
                key: value[index : index + 1] for key, value in candidates.items()
            }
            metric = candidate_metrics(chosen, one, config["recall_radius_px"])
            first = np.asarray(metric.pop("first_distance"))
            nearest = np.asarray(metric.pop("closest_distance"))
            if not np.isfinite(first).all() or not np.isfinite(nearest).all():
                raise FloatingPointError("Nonfinite supervised candidate distance")
            n = metric["count"]
            loss = float(
                neighborhood_mass_loss(prediction["scores"][index : index + 1], one)
            )
            total_queries += n
            total_loss += loss * n
            all_nearest.extend(nearest.tolist())
            rows.append(
                dict(
                    sample_id=sample_id,
                    family_id=batch["family_id"][index],
                    count=n,
                    recall1_correct=metric["recall1_correct"],
                    recall4_correct=metric["recall4_correct"],
                    recall_at_1=metric["recall1_correct"] / n if n else None,
                    recall_at_4=metric["recall4_correct"] / n if n else None,
                    nearest_median_px=float(np.median(nearest)) if n else None,
                    nearest_p95_px=float(np.percentile(nearest, 95)) if n else None,
                    nearest_max_px=float(nearest.max()) if n else None,
                    first_median_px=float(np.median(first)) if n else None,
                    loss=loss,
                    gt_outside=int(one["gt_outside"].sum()),
                )
            )
            if details:
                keep = one["supervised"][0].cpu().numpy()
                first_row = np.zeros(keep.shape, "float32")
                nearest_row = first_row.copy()
                first_row[keep] = first
                nearest_row[keep] = nearest
                first_storage.append(first_row)
                nearest_storage.append(nearest_row)
                support_storage.append(keep)
    active = [row for row in rows if row["count"]]
    summary = dict(
        tag=tag,
        split=dataset.split,
        pairs=len(rows),
        families=len({row["family_id"] for row in rows}),
        supervised_queries=total_queries,
        loss=total_loss / max(total_queries, 1),
        recall_at_1=sum(row["recall1_correct"] for row in rows) / max(total_queries, 1),
        recall_at_4=sum(row["recall4_correct"] for row in rows) / max(total_queries, 1),
        native_recall_radius_px=config["recall_radius_px"],
        candidates_per_source=config["candidate_count"],
        nearest_candidate_distance_px=quantiles(all_nearest),
        pair_distributions={
            key: quantiles([row[key] for row in active])
            for key in [
                "recall_at_1",
                "recall_at_4",
                "nearest_median_px",
                "nearest_p95_px",
            ]
        },
        pair_macro_recall_at_1=(
            float(np.mean([row["recall_at_1"] for row in active])) if active else None
        ),
        pair_macro_recall_at_4=(
            float(np.mean([row["recall_at_4"] for row in active])) if active else None
        ),
        pairs_recall4_below_half=sum(row["recall_at_4"] < 0.5 for row in active),
        pairs_without_supervised_queries=len(rows) - len(active),
        masked=mask_counts,
        all_supervised_distances_finite=True,
        source_family_isolation=True,
        geographic_independence="not_claimed",
        test_evaluated=False,
        UBC_evaluated=False,
        label_interpretation="relative to exact synthetic affine over publisher-claimed co-registered base; native residual unknown",
    )
    save_json(output / f"{tag}_metrics.json", summary)
    save_csv(output / f"{tag}_per_pair.csv", rows)
    with gzip.open(output / f"{tag}_per_pair.jsonl.gz", "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    if details:
        np.savez_compressed(
            output / f"{tag}_distances.npz",
            sample_id=np.array([row["sample_id"] for row in rows]),
            family_id=np.array([row["family_id"] for row in rows]),
            first_distance_px=np.stack(first_storage),
            nearest_distance_px=np.stack(nearest_storage),
            supervised=np.stack(support_storage),
        )
    if failures:
        ranked = sorted(
            active,
            key=lambda row: (
                row["recall_at_4"],
                -row["nearest_median_px"],
                row["sample_id"],
            ),
        )
        selected = []
        seen = set()
        for row in ranked:
            if row["family_id"] not in seen:
                selected.append(row)
                seen.add(row["family_id"])
            if len(selected) == 6:
                break
        indices = {row["sample_id"]: i for i, row in enumerate(dataset.records)}
        for rank, row in enumerate(selected):
            item = dataset[indices[row["sample_id"]]]
            batch = prepare(
                {
                    key: value[None] if isinstance(value, torch.Tensor) else [value]
                    for key, value in item.items()
                },
                device,
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
            path = output / "failures" / f"{tag}_{rank:02d}.png"
            visualize_batch(batch, model, cand, geom, path)
            image = cv2.imread(str(path))
            image[:40] = 255
            cv2.putText(
                image,
                f"R@1 {row['recall_at_1']:.3f} R@4 {row['recall_at_4']:.3f} med {row['nearest_median_px']:.1f}px",
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
            cv2.putText(
                image,
                "Cyan source | green label | red top1 | orange other",
                (5, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
            cv2.imwrite(str(path), image)
        save_json(
            output / f"{tag}_failures.json",
            dict(
                selection="lowest pair recall@4, then highest median distance; one pair per family; descriptive only",
                samples=selected,
            ),
        )
    return summary
