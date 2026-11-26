"""Two fixed structural peaks plus two full-grid semantic peaks, no structural pool."""

import numpy as np
import torch


@torch.no_grad()
def supplement_from_full_grid(
    prefix, ranking_scores, target_valid, target_xy, radius=24.0
):
    """Reserve prefix's first two, suppress in that order, fill two by ranking_scores.

    ranking_scores must contain every target grid location, not selected structural
    candidates. Only target_valid and the same sequential spatial NMS restrict it.
    The API accepts no geometric labels or structural scores for supplementation.
    """
    if ranking_scores.ndim != 3 or ranking_scores.shape[-1] != len(target_xy):
        raise ValueError("Supplement score must cover the complete target grid")
    if target_valid.shape != (ranking_scores.shape[0], len(target_xy)):
        raise ValueError("Target validity must cover the complete target grid")
    if (
        not torch.isfinite(ranking_scores).all()
        or radius < 0
        or not np.isfinite(radius)
    ):
        raise ValueError("Nonfinite full-grid scores or invalid NMS radius")
    if (
        prefix["indices"].shape[:2] != ranking_scores.shape[:2]
        or prefix["indices"].shape[-1] < 2
    ):
        raise ValueError("Need two fixed structural prefix slots per source")
    available = target_valid[:, None].expand_as(ranking_scores).clone()
    indices = [prefix["indices"][..., i] for i in range(2)]
    validity = [prefix["valid"][..., i] for i in range(2)]
    values = [prefix["scores"][..., i] for i in range(2)]
    for index, valid in zip(indices, validity):
        center = target_xy[index.clamp_min(0)]
        suppressed = ((target_xy[None, None] - center[:, :, None]) ** 2).sum(
            -1
        ) <= radius**2
        available &= ~(suppressed & valid[..., None])
    for _ in range(2):
        index = ranking_scores.masked_fill(
            ~available, torch.finfo(ranking_scores.dtype).min
        ).argmax(-1)
        valid = available.any(-1)
        value = ranking_scores.gather(-1, index[..., None]).squeeze(-1)
        center = target_xy[index]
        suppressed = ((target_xy[None, None] - center[:, :, None]) ** 2).sum(
            -1
        ) <= radius**2
        available &= ~(suppressed & valid[..., None])
        indices.append(torch.where(valid, index, -torch.ones_like(index)))
        validity.append(valid)
        values.append(torch.where(valid, value, torch.zeros_like(value)))
    indices = torch.stack(indices, -1)
    valid = torch.stack(validity, -1)
    xy = target_xy[indices.clamp_min(0)]
    return dict(
        indices=indices,
        valid=valid,
        pixel_xy=torch.where(valid[..., None], xy, torch.zeros_like(xy)),
        scores=torch.stack(values, -1),
    )


def paired_counts(original_distance, method_distance, keep, radius=8.0):
    original = (original_distance <= radius) & keep
    method = (method_distance <= radius) & keep
    n = int(keep.sum())
    original_hits = int(original.sum())
    new = int((~original & method & keep).sum())
    lost = int((original & ~method & keep).sum())
    misses = n - original_hits
    return dict(
        effective_queries=n,
        original_hits=original_hits,
        original_misses=misses,
        hits=int(method.sum()),
        new_hits=new,
        lost_hits=lost,
        net_hits=new - lost,
        recall_at_4=int(method.sum()) / n if n else None,
        delta_recall4=(new - lost) / n if n else None,
        rescue_rate=new / misses if misses else None,
        loss_rate=lost / original_hits if original_hits else None,
    )


def paired_family_intervals(rows, seed=2026112612, replicates=2000):
    columns = [
        "effective_queries",
        "structure_hits",
        "fusion_hits",
        "semantic_hits",
        "fusion_new_hits",
        "fusion_lost_hits",
        "semantic_new_hits",
        "semantic_lost_hits",
    ]
    values = np.array([[r[k] for k in columns] for r in rows], dtype="float64")
    rng = np.random.default_rng(seed)
    results = {
        k: []
        for k in [
            "fusion_minus_structure_recall4",
            "semantic_minus_structure_recall4",
            "semantic_minus_fusion_recall4",
            "fusion_rescue_rate",
            "semantic_rescue_rate",
            "fusion_loss_rate",
            "semantic_loss_rate",
        ]
    }
    for _ in range(replicates):
        n, a, f, s, fn, fl, sn, sl = values[
            rng.integers(0, len(values), len(values))
        ].sum(0)
        row = [
            (f - a) / n,
            (s - a) / n,
            (s - f) / n,
            fn / (n - a) if n > a else np.nan,
            sn / (n - a) if n > a else np.nan,
            fl / a if a else np.nan,
            sl / a if a else np.nan,
        ]
        for key, value in zip(results, row):
            results[key].append(value)
    return dict(
        seed=seed,
        replicates=replicates,
        unit="paired whole source-family resampling",
        interval95={
            k: (
                np.nanpercentile(v, [2.5, 97.5]).tolist()
                if np.isfinite(v).any()
                else None
            )
            for k, v in results.items()
        },
        interpretation="descriptive sample uncertainty conditional on this frozen trained seed, family split and labels; no geographic-independence or training-seed uncertainty claim",
    )
