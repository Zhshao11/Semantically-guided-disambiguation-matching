"""Ranks and candidate budgets for structure baseline and GT-free descriptor heads."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .coarse import spatial_candidates
from .context_descriptor_data import frozen_inputs, group_maps
from .context_descriptors import full_grid_pair_mass_loss
from .semantic_bottleneck import full_grid_rank
from .semantic_evaluation import candidate_arrays
from .semantic_routing import sample_native_map
from .semantic_supplement import supplement_from_full_grid, paired_counts
from .family_evaluation import quantiles
from .io import save_json, save_csv
from scripts.train_structure_smoke import prepare, geometry_for


def descriptor_spatial_stats(descriptor, mask):
    # B,N,D. Quantify spatial variation independent of descriptor unit norm.
    result = []
    for d, m in zip(descriptor, mask):
        value = d[m]
        if len(value):
            mean = value.mean(0)
            result.append(
                dict(
                    mean_vector_norm=float(mean.norm()),
                    variance=float((value - mean).square().sum(-1).mean()),
                    average_offdiagonal_cosine=(
                        float(
                            (
                                (value.sum(0).square().sum() - value.square().sum())
                                / (len(value) * (len(value) - 1))
                            ).clamp(-1, 1)
                        )
                        if len(value) > 1
                        else None
                    ),
                )
            )
    return result


def paired_interval(family_rows, cfg):
    x = np.array(
        [
            [
                r[k]
                for k in [
                    "effective_queries",
                    "structure_hits",
                    "merged_hits",
                    "new_hits",
                    "lost_hits",
                ]
            ]
            for r in family_rows
        ],
        dtype=float,
    )
    if not len(x):
        return {}
    rng = np.random.default_rng(cfg["seed"])
    d = []
    res = []
    for _ in range(cfg["replicates"]):
        n, a, b, added, lost = x[rng.integers(0, len(x), len(x))].sum(0)
        d.append((b - a) / n)
        res.append(added / (n - a) if n > a else np.nan)
    return dict(
        delta_recall4_interval95=np.percentile(d, [2.5, 97.5]).tolist(),
        rescue_rate_interval95=(
            np.nanpercentile(res, [2.5, 97.5]).tolist()
            if np.isfinite(res).any()
            else None
        ),
        seed=cfg["seed"],
        replicates=cfg["replicates"],
        scope="paired family sample uncertainty only; smoke fits are not independent generalization",
    )


@torch.no_grad()
def evaluate_context(
    model,
    semantic_head,
    heads,
    dataset,
    parent,
    cfg,
    projection,
    device,
    output,
    tag,
    cache=None,
    baseline_only=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    semantic_head.eval()
    for head in heads.values():
        head.eval()
    methods = ["structure"] + (list(heads) if not baseline_only else [])
    storage = {m: {} for m in methods}
    masks = []
    ids = []
    families = []
    fractions = []
    target_fraction = []
    collapse = {m: dict(optical=[], sar=[]) for m in heads}
    loss_rows = {m: [] for m in heads}

    def add(store, key, value):
        store.setdefault(key, []).extend(value.cpu().numpy())

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in loader:
        batch = prepare(batch, device)
        inp = frozen_inputs(model, semantic_head, batch)
        g = geometry_for(model, batch, parent)
        source_valid = (
            inp["optical_M"]
            .flatten(1)
            .gather(
                -1,
                (model.source_feature_xy[:, 1] * 32 + model.source_feature_xy[:, 0])[
                    None
                ].expand(len(batch["id"]), -1),
            )
        )
        original = spatial_candidates(
            inp["structural_scores"],
            g["target_valid"],
            model.target_pixel_xy,
            count=4,
            nms_radius_px=24,
        )
        first, near = candidate_arrays(original, g)
        st = storage["structure"]
        add(st, "first", first)
        add(st, "nearest", near)
        add(st, "merged_nearest", near)
        add(st, "indices", original["indices"])
        add(st, "valid", original["valid"])
        add(st, "merged_indices", original["indices"])
        rank = full_grid_rank(
            inp["structural_scores"],
            g["positive"],
            g["target_valid"],
            cfg["score_tie_atol"],
        )
        for k, v in rank.items():
            add(st, "rank_" + k, v)
        for m, head in heads.items():
            om, sm, of, sf = group_maps(inp, projection, m)
            result = head(om, sm, model.source_pixel_xy, model.target_pixel_xy)
            loss, details = full_grid_pair_mass_loss(
                result["scores"], g, cfg["temperature"]
            )
            loss_rows[m].extend(
                details["per_pair"][details["valid_pair"]].cpu().tolist()
            )
            independent = spatial_candidates(
                result["scores"],
                g["target_valid"],
                model.target_pixel_xy,
                count=4,
                nms_radius_px=24,
            )
            merged = supplement_from_full_grid(
                original, result["scores"], g["target_valid"], model.target_pixel_xy
            )
            f, n = candidate_arrays(independent, g)
            _, mn = candidate_arrays(merged, g)
            st = storage[m]
            add(st, "first", f)
            add(st, "nearest", n)
            add(st, "merged_nearest", mn)
            add(st, "indices", independent["indices"])
            add(st, "valid", independent["valid"])
            add(st, "merged_indices", merged["indices"])
            ranks = full_grid_rank(
                result["scores"],
                g["positive"],
                g["target_valid"],
                cfg["score_tie_atol"],
            )
            for k, v in ranks.items():
                add(st, "rank_" + k, v)
            collapse[m]["optical"].extend(
                descriptor_spatial_stats(result["optical"], g["supervised"])
            )
            collapse[m]["sar"].extend(
                descriptor_spatial_stats(result["sar"], g["target_valid"])
            )
            if m == list(heads)[0]:
                fractions.extend(
                    sample_native_map(of, model.source_pixel_xy)
                    .transpose(1, 2)
                    .cpu()
                    .numpy()
                )
                target_fraction.extend(
                    sample_native_map(sf, g["gt_pixel_xy"])
                    .transpose(1, 2)
                    .cpu()
                    .numpy()
                )
        masks.extend(g["supervised"].cpu().numpy())
        ids.extend(batch["id"])
        families.extend(batch["family_id"])
    arrays = dict(
        sample_id=np.array(ids),
        family_id=np.array(families),
        supervised=np.stack(masks),
    )
    keep = arrays["supervised"]
    summaries = {}
    pair_rows = []
    family_rows = []
    for m in methods:
        st = {k: np.stack(v) for k, v in storage[m].items()}
        arrays.update({m + "_" + k: v for k, v in st.items()})
    structure_hit = arrays["structure_nearest"] <= 8
    for m in methods:
        st = {k: arrays[m + "_" + k] for k in storage[m]}
        merged = st["merged_nearest"]
        groups = {}
        for name, k in [("all", keep), ("structure_miss", keep & ~structure_hit)]:
            groups[name] = dict(
                queries=int(k.sum()),
                best_positive_rank_min=quantiles(st["rank_rank_min"][k]),
                best_positive_rank_max=quantiles(st["rank_rank_max"][k]),
                top1_coverage=float((st["first"][k] <= 8).mean()) if k.any() else None,
                independent_NMS4_coverage=(
                    float((st["nearest"][k] <= 8).mean()) if k.any() else None
                ),
                raw_top4_guaranteed=(
                    float(st["rank_top4_guaranteed"][k].mean()) if k.any() else None
                ),
                raw_top4_possible=(
                    float(st["rank_top4_possible"][k].mean()) if k.any() else None
                ),
            )
        count = paired_counts(arrays["structure_nearest"], merged, keep)
        summary = dict(
            method=m,
            tag=tag,
            pairs=len(ids),
            families=len(set(families)),
            **count,
            ranking=groups,
            merged_distance_px=quantiles(merged[keep]),
            independent_distance_px=quantiles(st["nearest"][keep]),
            loss_pair_mean=float(np.mean(loss_rows[m])) if m in heads else None,
            interpretation=(
                "native synthetic-label-relative; fixed small-fit subset is training diagnosis only"
                if dataset.split == "train" and len(dataset) <= 32
                else "fixed source-family diagnostic, not geographic independence"
            ),
            OS_test_evaluated=False,
            UBC_evaluated=False,
        )
        if m in heads:
            summary["descriptor_spatial_diagnostics"] = collapse[m]
        local = []
        for fam in sorted(set(families)):
            idx = np.flatnonzero(np.array(families) == fam)
            k = keep[idx]
            c = paired_counts(arrays["structure_nearest"][idx], merged[idx], k)
            row = dict(
                method=m,
                family_id=fam,
                pairs=len(idx),
                effective_queries=int(k.sum()),
                structure_hits=c["original_hits"],
                merged_hits=c["hits"],
                **{
                    z: c[z]
                    for z in [
                        "new_hits",
                        "lost_hits",
                        "net_hits",
                        "recall_at_4",
                        "rescue_rate",
                    ]
                },
                rank_median=float(np.median(st["rank_rank_min"][idx][k])),
                rank_p75=float(np.percentile(st["rank_rank_min"][idx][k], 75)),
                rank_p95=float(np.percentile(st["rank_rank_min"][idx][k], 95)),
                top1_coverage=float((st["first"][idx][k] <= 8).mean()),
                independent_NMS4_coverage=float((st["nearest"][idx][k] <= 8).mean()),
            )
            family_rows.append(row)
            local.append(row)
        if m != "structure":
            summary["family_intervals"] = paired_interval(local, cfg["bootstrap"])
        summaries[m] = summary
        for i, sample in enumerate(ids):
            k = keep[i]
            c = paired_counts(arrays["structure_nearest"][i], merged[i], k)
            pair_rows.append(
                dict(
                    method=m,
                    sample_id=sample,
                    family_id=families[i],
                    **c,
                    rank_median=float(np.median(st["rank_rank_min"][i, k])),
                    rank_p95=float(np.percentile(st["rank_rank_min"][i, k], 95)),
                )
            )
    if len(fractions):
        arrays["source_window_valid_fraction"] = np.stack(fractions)
        arrays["target_GT_window_valid_fraction"] = np.stack(target_fraction)
    save_json(output / (tag + "_summary.json"), summaries)
    save_csv(output / (tag + "_per_pair.csv"), pair_rows)
    save_csv(output / (tag + "_per_family.csv"), family_rows)
    np.savez_compressed(output / (tag + "_queries.npz"), **arrays)
    return summaries
