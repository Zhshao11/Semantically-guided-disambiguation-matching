"""Rank-first and net-coverage summaries for the fixed probability-score control."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.family_evaluation import quantiles
from sdgm.io import save_json, save_csv
from sdgm.semantic_supplement import paired_counts


def q(v):
    return quantiles(np.asarray(v)[np.isfinite(v)])


def rank_summary(a, layer, method, mask):
    prefix = layer + "_" + method + "_"
    v = mask & a[prefix + "valid"]
    return dict(
        queries=int(mask.sum()),
        valid=int(v.sum()),
        win=int((a[prefix + "win"] & v).sum()),
        loss=int((a[prefix + "loss"] & v).sum()),
        tie=int((a[prefix + "tie"] & v).sum()),
        win_rate=float(a[prefix + "win"][v].mean()) if v.any() else None,
        loss_rate=float(a[prefix + "loss"][v].mean()) if v.any() else None,
        tie_rate=float(a[prefix + "tie"][v].mean()) if v.any() else None,
        exact_ties=int((a[prefix + "exact_tie"] & v).sum()),
        delta=q(a[prefix + "delta"][v]),
        rank_min=q(a[prefix + "rank_min"][v]),
        rank_max=q(a[prefix + "rank_max"][v]),
    )


def coverage(a, mask):
    methods = {
        "structure": a["structure_nearest"],
        "dot": a["dot_supplement_nearest_distance"],
        "dist": a["dist_supplement_nearest_distance"],
    }
    return {
        m: dict(
            **paired_counts(methods["structure"], d, mask),
            nearest_distance_px=q(d[mask]),
        )
        for m, d in methods.items()
    }


def bootstrap(rows, spec):
    rows = [r for r in rows if r["queries"]]
    if not rows:
        return {}
    names = [
        "queries",
        "structure_hits",
        "dot_hits",
        "dist_hits",
        "dot_new_hits",
        "dot_lost_hits",
        "dist_new_hits",
        "dist_lost_hits",
        "B_dot_win",
        "B_dist_win",
        "C_dot_win",
        "C_dist_win",
        "valid",
        "C_dot_NMS4_hits",
        "C_dist_NMS4_hits",
    ]
    data = np.array([[r[k] for k in names] for r in rows], dtype=float)
    metric_names = [
        "dot_minus_structure_recall4",
        "dist_minus_structure_recall4",
        "dist_minus_dot_recall4",
        "dot_rescue_rate",
        "dist_rescue_rate",
        "dot_loss_rate",
        "dist_loss_rate",
        "B_dist_minus_dot_win",
        "C_dist_minus_dot_win",
        "C_dist_minus_dot_NMS4",
    ]
    values = []
    rng = np.random.default_rng(spec["seed"])
    for _ in range(spec["replicates"]):
        n, st, dot, dis, dn, dl, en, el, bw, be, cw, ce, v, nh, ne = data[
            rng.integers(0, len(data), len(data))
        ].sum(0)
        values.append(
            [
                (dot - st) / n,
                (dis - st) / n,
                (dis - dot) / n,
                dn / (n - st) if n > st else np.nan,
                en / (n - st) if n > st else np.nan,
                dl / st if st else np.nan,
                el / st if st else np.nan,
                (be - bw) / v if v else np.nan,
                (ce - cw) / v if v else np.nan,
                (ne - nh) / n,
            ]
        )
    values = np.array(values)
    return dict(
        families=len(rows),
        seed=spec["seed"],
        replicates=spec["replicates"],
        interval95={
            k: (
                np.nanpercentile(values[:, i], [2.5, 97.5]).tolist()
                if np.isfinite(values[:, i]).any()
                else None
            )
            for i, k in enumerate(metric_names)
        },
        scope="conditional on frozen model/seed and current source families; paired query-weighted family sampling, not geography or training-seed uncertainty",
    )


def family_row(a, mask):
    r = dict(
        queries=int(mask.sum()),
        valid=int((mask & a["B_dot_valid"] & a["C_dot_valid"]).sum()),
    )
    for m, dist in [
        ("structure", a["structure_nearest"]),
        ("dot", a["dot_supplement_nearest_distance"]),
        ("dist", a["dist_supplement_nearest_distance"]),
    ]:
        counts = paired_counts(a["structure_nearest"], dist, mask)
        for key in [
            "hits",
            "new_hits",
            "lost_hits",
            "net_hits",
            "recall_at_4",
            "rescue_rate",
            "loss_rate",
        ]:
            r[m + "_" + key] = counts[key]
    for l in ["B", "C"]:
        for m in ["dot", "dist"]:
            r[l + "_" + m + "_win"] = int((mask & a[l + "_" + m + "_win"]).sum())
            r[l + "_" + m + "_tie"] = int((mask & a[l + "_" + m + "_tie"]).sum())
    for m in ["dot", "dist"]:
        r["C_" + m + "_NMS4_hits"] = int(
            (mask & (a[m + "_semantic4_nearest_distance"] <= 8)).sum()
        )
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_semantic_score_control.json")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    all_summary = {}
    layer_rows = []
    budget_rows = []
    family_rows = []
    pair_rows = []
    for split in config["evaluation_splits"]:
        z = np.load(root / f"{split}_score_queries.npz")
        a = {k: z[k] for k in z.files}
        summary = {}
        for group in config["groups"]:
            mask = a["group_" + group]
            n = int(mask.sum())
            s = dict(
                queries=n,
                layers={},
                C_full={},
                coverage=coverage(a, mask),
                sharpness={},
            )
            fr = []
            for fam in sorted(set(a["family_id"])):
                fm = mask & (a["family_id"] == fam)[:, None]
                row = dict(split=split, group=group, family_id=fam, **family_row(a, fm))
                fr.append(row)
                family_rows.append(row)
            s["family_intervals"] = bootstrap(fr, config["bootstrap"])
            for layer in ["B", "C"]:
                s["layers"][layer] = {
                    m: rank_summary(a, layer, m, mask) for m in ["dot", "dist"]
                }
                v = mask & a[layer + "_dot_valid"] & a[layer + "_dist_valid"]
                s["layers"][layer]["paired_transition"] = dict(
                    dot_loss_to_dist_win=int(
                        (a[layer + "_dot_loss"] & a[layer + "_dist_win"] & v).sum()
                    ),
                    dot_win_to_dist_loss=int(
                        (a[layer + "_dot_win"] & a[layer + "_dist_loss"] & v).sum()
                    ),
                    dot_tie_to_dist_win=int(
                        (a[layer + "_dot_tie"] & a[layer + "_dist_win"] & v).sum()
                    ),
                    delta_win_rate=(
                        float(
                            (
                                a[layer + "_dist_win"][v].astype(float)
                                - a[layer + "_dot_win"][v]
                            ).mean()
                        )
                        if v.any()
                        else None
                    ),
                )
                pn = a[layer + "_dot_bestP_target_norm2"]
                nn = a[layer + "_dot_bestN_target_norm2"]
                fixed = a[layer + "_distance_gap_at_dot_extrema"]
                actual = a[layer + "_dist_delta"]
                s["sharpness"][layer] = dict(
                    dot_bestP_target_norm2=q(pn[v]),
                    dot_bestN_target_norm2=q(nn[v]),
                    NminusP_norm2=q((nn - pn)[v]),
                    fraction_dot_N_sharper=(
                        float((nn[v] > pn[v] + config["score_tie_atol"]).mean())
                        if v.any()
                        else None
                    ),
                    dot_extrema_distance_gap=q(fixed[v]),
                    actual_reselected_distance_gap=q(actual[v]),
                    fixed_pair_flip_to_positive_but_actual_not_win=int(
                        (
                            (fixed > config["score_tie_atol"])
                            & ~a[layer + "_dist_win"]
                            & a[layer + "_dot_loss"]
                            & v
                        ).sum()
                    ),
                    note="fixed dot extrema explain norm correction only; actual distance winners are independently reselected, so pair flip alone does not imply rank recovery",
                )
                for method in ["dot", "dist"]:
                    t = s["layers"][layer][method]
                    layer_rows.append(
                        dict(
                            split=split,
                            group=group,
                            layer=layer,
                            score=method,
                            queries=n,
                            valid=t["valid"],
                            win=t["win"],
                            loss=t["loss"],
                            tie=t["tie"],
                            win_rate=t["win_rate"],
                            loss_rate=t["loss_rate"],
                            tie_rate=t["tie_rate"],
                            delta_median=t["delta"].get("50"),
                            delta_p5=t["delta"].get("5"),
                            delta_p95=t["delta"].get("95"),
                            rank_min_median=t["rank_min"].get("50"),
                            rank_max_median=t["rank_max"].get("50"),
                        )
                    )
            for method in ["dot", "dist"]:
                pre = "C_" + method + "_full_"
                v = mask & a[pre + "valid"]
                s["C_full"][method] = dict(
                    best_positive_rank_min=q(a[pre + "rank_min"][v]),
                    best_positive_rank_max=q(a[pre + "rank_max"][v]),
                    raw_top4_coverage_guaranteed=(
                        float(a[pre + "top4_guaranteed"][v].mean()) if v.any() else None
                    ),
                    raw_top4_coverage_possible=(
                        float(a[pre + "top4_possible"][v].mean()) if v.any() else None
                    ),
                    semantic_independent4_NMS_coverage=(
                        float(
                            (
                                a[method + "_semantic4_nearest_distance"][mask] <= 8
                            ).mean()
                        )
                        if n
                        else None
                    ),
                    semantic_peak_nearest_distance_px=q(
                        a[method + "_semantic4_nearest_distance"][mask]
                    ),
                )
            s["family_change"] = {
                m: dict(
                    improved=sum(r[m + "_net_hits"] > 0 for r in fr),
                    degraded=sum(r[m + "_net_hits"] < 0 for r in fr),
                    unchanged=sum(
                        r[m + "_net_hits"] == 0 and r["queries"] > 0 for r in fr
                    ),
                )
                for m in ["dot", "dist"]
            }
            s["family_dist_vs_dot"] = dict(
                improved=sum(r["dist_hits"] > r["dot_hits"] for r in fr),
                degraded=sum(r["dist_hits"] < r["dot_hits"] for r in fr),
                unchanged=sum(
                    r["dist_hits"] == r["dot_hits"] and r["queries"] > 0 for r in fr
                ),
            )
            summary[group] = s
        for i, sample in enumerate(a["sample_id"]):
            mask = np.zeros_like(a["supervised"])
            mask[i] = a["supervised"][i]
            pair_rows.append(
                dict(
                    split=split,
                    sample_id=sample,
                    family_id=a["family_id"][i],
                    **family_row(a, mask),
                )
            )
        for m, t in summary["all_supervised"]["coverage"].items():
            budget_rows.append(
                dict(
                    split=split,
                    method=m,
                    **{k: v for k, v in t.items() if k != "nearest_distance_px"},
                    median_px=t["nearest_distance_px"]["50"],
                    p95_px=t["nearest_distance_px"]["95"],
                )
            )
        save_json(root / f"{split}_group_summary.json", summary)
        all_summary[split] = summary
        # Compact complete per-query CSV includes original fixed group membership.
        with gzip.open(root / f"{split}_per_query.csv.gz", "wt", newline="") as stream:
            fields = (
                ["sample_id", "family_id", "query_id", "supervised", "fixed_groups"]
                + [
                    l + "_" + m + "_" + k
                    for l in ["B", "C"]
                    for m in ["dot", "dist"]
                    for k in ["win", "loss", "tie", "delta", "rank_min", "rank_max"]
                ]
                + [m + "_supplement_nearest_distance" for m in ["dot", "dist"]]
            )
            w = csv.DictWriter(stream, fieldnames=fields)
            w.writeheader()
            for i, sample in enumerate(a["sample_id"]):
                for j in range(144):
                    row = dict(
                        sample_id=sample,
                        family_id=a["family_id"][i],
                        query_id=j,
                        supervised=bool(a["supervised"][i, j]),
                        fixed_groups=";".join(
                            g for g in config["groups"] if a["group_" + g][i, j]
                        ),
                    )
                    for k in fields[5:]:
                        row[k] = a[k][i, j].item()
                    w.writerow(row)
    save_csv(root / "PN_score_comparison.csv", layer_rows)
    save_csv(root / "candidate_budget_comparison.csv", budget_rows)
    save_csv(root / "family_paired_score_control.csv", family_rows)
    save_csv(root / "per_pair_score_control.csv", pair_rows)
    val = all_summary["validation"]
    groups = ["all_supervised", "structure_miss", "semantic_new", "semantic_lost"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for l, c in [("B", "tab:blue"), ("C", "tab:orange")]:
        for m, style in [("dot", "--"), ("dist", "-")]:
            axes[0].plot(
                range(4),
                [100 * val[g]["layers"][l][m]["win_rate"] for g in groups],
                style + "o",
                color=c,
                label=l + " " + m,
            )
    axes[0].set_xticks(range(4), ["All", "Struct miss", "Old new277", "Old lost5645"])
    axes[0].set_ylabel("Best P beats max N (%)")
    axes[0].legend(fontsize=8)
    for m, c in [("dot", "tab:blue"), ("dist", "tab:orange")]:
        axes[1].plot(
            range(4),
            [val[g]["C_full"][m]["best_positive_rank_min"]["50"] for g in groups],
            "o-",
            color=c,
            label=m,
        )
    axes[1].set_xticks(range(4), ["All", "Struct miss", "Old new277", "Old lost5645"])
    axes[1].set_ylabel("C best positive full-grid median rank")
    axes[1].legend()
    for split, c in [("training_monitor", "tab:blue"), ("validation", "tab:orange")]:
        t = all_summary[split]["all_supervised"]["coverage"]
        axes[2].plot(
            range(3),
            [100 * t[m]["recall_at_4"] for m in ["structure", "dot", "dist"]],
            "o-",
            color=c,
            label=split,
        )
    axes[2].set_xticks(range(3), ["4struct", "2struct+2dot", "2struct+2dist"])
    axes[2].set_ylabel("Actual fixed4-candidate recall@4 (%)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(root / "score_ranking_and_coverage.png", dpi=180)
    plt.close(fig)
    print(
        "Ranking, sharpness correction, fixed-budget coverage and family intervals summarized",
        flush=True,
    )


if __name__ == "__main__":
    main()
