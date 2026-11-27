"""Same fixed native evaluation for equal-budget semantic-head objectives."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .coarse import spatial_candidates
from .correspondence_objective import fixed_competition, correspondence_loss
from .family_evaluation import quantiles
from .io import save_json, save_csv
from .semantic_bottleneck import comparison_stats, full_grid_rank
from .semantic_evaluation import candidate_arrays, DistributionStats
from .semantic_routing import distillation_loss
from .semantic_score_control import probability_scores
from .semantic_supplement import supplement_from_full_grid, paired_counts
from scripts.train_structure_smoke import prepare, geometry_for


@torch.no_grad()
def evaluate_branches(
    model, heads, dataset, parent, cfg, locked, device, output, split, step
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for head in heads.values():
        head.eval()
    model.eval()
    tau = locked["locked_tau"]
    alpha = locked["locked_alpha"]
    storage = {n: {} for n in heads}
    stats = {n: DistributionStats() for n in heads}
    base = {}
    ids = []
    families = []

    def append(store, key, value):
        store.setdefault(key, []).extend(value.cpu().numpy())

    for batch in DataLoader(
        dataset, batch_size=cfg["evaluation_batch_size"], shuffle=False
    ):
        batch = prepare(batch, device)
        feature = model(batch["optical"], batch["sar"])
        g = geometry_for(model, batch, parent)
        comp = fixed_competition(feature["scores"], g, model.target_pixel_xy)
        original = spatial_candidates(
            feature["scores"],
            g["target_valid"],
            model.target_pixel_xy,
            count=4,
            nms_radius_px=24,
        )
        f, d = candidate_arrays(original, g)
        append(base, "structure_first", f)
        append(base, "structure_nearest", d)
        append(base, "structure_indices", original["indices"])
        append(base, "supervised", g["supervised"])
        append(base, "positive", comp["positive"])
        append(base, "negative_indices", comp["negative_indices"])
        append(base, "negative_valid", comp["negative_valid"])
        ids.extend(batch["id"])
        families.extend(batch["family_id"])
        for name, head in heads.items():
            st = storage[name]
            p = head(feature["sar_descriptors"])
            kl, perkl, sampled, teacher = distillation_loss(
                batch["teacher"], p, model.source_pixel_xy, g
            )
            scores = probability_scores(teacher.detach(), p.flatten(2))["dist"]
            corr, cstatus = correspondence_loss(scores, comp, tau)
            corrquery = torch.zeros_like(perkl)
            corrquery[cstatus["eligible"]] = cstatus["per_eligible"]
            comparison = comparison_stats(scores, comp["positive"], comp["negative"])
            ranking = full_grid_rank(scores, comp["positive"], g["target_valid"])
            actual = supplement_from_full_grid(
                original, scores, g["target_valid"], model.target_pixel_xy
            )
            sem4 = spatial_candidates(
                scores,
                g["target_valid"],
                model.target_pixel_xy,
                count=4,
                nms_radius_px=24,
            )
            first, near = candidate_arrays(actual, g)
            _, snear = candidate_arrays(sem4, g)
            append(st, "first", first)
            append(st, "nearest", near)
            append(st, "indices", actual["indices"])
            append(st, "valid", actual["valid"])
            append(st, "semantic4_nearest", snear)
            append(st, "semantic4_indices", sem4["indices"])
            append(st, "semantic4_valid", sem4["valid"])
            append(st, "KL", perkl)
            append(st, "corr", corrquery)
            append(st, "corr_eligible", cstatus["eligible"])
            for key in ["win", "loss", "tie", "delta", "rank_min", "rank_max", "valid"]:
                append(st, "PN_" + key, comparison[key])
            for key in ["rank_min", "rank_max", "top4_guaranteed", "top4_possible"]:
                append(st, "full_" + key, ranking[key])
            stats[name].add(sampled, g["supervised"])
    arrays = {k: np.stack(v) for k, v in base.items()}
    arrays.update(sample_id=np.array(ids), family_id=np.array(families))
    keep = arrays["supervised"]
    results = {}
    rows = []
    family_rows = []
    previous = np.load(Path(cfg["score_control_root"]) / f"{split}_score_queries.npz")
    np.testing.assert_array_equal(arrays["sample_id"], previous["sample_id"])
    np.testing.assert_array_equal(keep, previous["supervised"])
    np.testing.assert_array_equal(
        arrays["structure_indices"], previous["structure_indices"]
    )
    for name in heads:
        st = {k: np.stack(v) for k, v in storage[name].items()}
        arrays.update({name + "_" + k: v for k, v in st.items()})
        if step == 0:
            for key, past in [
                ("nearest", "dist_supplement_nearest_distance"),
                ("indices", "dist_supplement_indices"),
            ]:
                np.testing.assert_array_equal(st[key], previous[past])
            np.testing.assert_array_equal(st["PN_delta"], previous["C_dist_delta"])
        count = paired_counts(arrays["structure_nearest"], st["nearest"], keep)
        v = keep & st["PN_valid"]
        ce = st["corr_eligible"]
        summary = dict(
            branch=name,
            additional_step=step,
            total_step=cfg["start_step"] + step,
            split=split,
            pairs=len(ids),
            families=len(set(families)),
            **count,
            recall_at_1=float((st["first"][keep] <= 8).mean()),
            nearest_distance_px=quantiles(st["nearest"][keep]),
            KL=float(st["KL"][keep].mean()),
            corr=float(st["corr"][ce].mean()) if ce.any() else None,
            corr_eligible_queries=int(ce.sum()),
            PN_win_rate=float(st["PN_win"][v].mean()),
            PN_loss_rate=float(st["PN_loss"][v].mean()),
            PN_tie_rate=float(st["PN_tie"][v].mean()),
            PN_delta=quantiles(st["PN_delta"][v]),
            full_best_positive_rank_min=quantiles(st["full_rank_min"][keep]),
            full_best_positive_rank_max=quantiles(st["full_rank_max"][keep]),
            semantic_independent4_coverage=float(
                (st["semantic4_nearest"][keep] <= 8).mean()
            ),
            SAR_distribution=stats[name].summary(),
            tau=tau,
            alpha=alpha if name == "KL_corr" else 0,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        )
        if step == 0:
            summary["starting_previous_distance_score_results_exact"] = True
        results[name] = summary
        for i, sample in enumerate(ids):
            mask = keep[i]
            n = int(mask.sum())
            c = paired_counts(arrays["structure_nearest"][i], st["nearest"][i], mask)
            cv = ce[i]
            pv = mask & st["PN_valid"][i]
            rows.append(
                dict(
                    branch=name,
                    split=split,
                    additional_step=step,
                    sample_id=sample,
                    family_id=families[i],
                    **c,
                    KL=float(st["KL"][i, mask].mean()) if n else None,
                    corr=float(st["corr"][i, cv].mean()) if cv.any() else None,
                    PN_win_rate=float(st["PN_win"][i, pv].mean()) if pv.any() else None,
                    median_distance_px=(
                        float(np.median(st["nearest"][i, mask])) if n else None
                    ),
                    p95_distance_px=(
                        float(np.percentile(st["nearest"][i, mask], 95)) if n else None
                    ),
                )
            )
        for family in sorted(set(families)):
            idx = np.flatnonzero(np.array(families) == family)
            k = keep[idx]
            pv = k & st["PN_valid"][idx]
            cv = ce[idx]
            c = paired_counts(arrays["structure_nearest"][idx], st["nearest"][idx], k)
            family_rows.append(
                dict(
                    branch=name,
                    split=split,
                    additional_step=step,
                    family_id=family,
                    pairs=len(idx),
                    **c,
                    KL=float(st["KL"][idx][k].mean()),
                    corr=float(st["corr"][idx][cv].mean()) if cv.any() else None,
                    PN_win=int(st["PN_win"][idx][pv].sum()),
                    PN_valid=int(pv.sum()),
                    PN_win_rate=float(st["PN_win"][idx][pv].mean()),
                    median_distance_px=float(np.median(st["nearest"][idx][k])),
                    p95_distance_px=float(np.percentile(st["nearest"][idx][k], 95)),
                )
            )
    if step == 0:
        for key in storage["KL_control"]:
            np.testing.assert_array_equal(
                arrays["KL_control_" + key], arrays["KL_corr_" + key]
            )
    tag = f"step{step:05d}_{split}"
    save_json(output / (tag + "_summary.json"), results)
    save_csv(output / (tag + "_per_pair.csv"), rows)
    save_csv(output / (tag + "_per_family.csv"), family_rows)
    np.savez_compressed(output / (tag + "_queries.npz"), **arrays)
    print(
        str(
            {
                n: dict(
                    step=step,
                    split=split,
                    R4=x["recall_at_4"],
                    KL=x["KL"],
                    corr=x["corr"],
                    PN_win=x["PN_win_rate"],
                )
                for n, x in results.items()
            }
        ),
        flush=True,
    )
    return results
