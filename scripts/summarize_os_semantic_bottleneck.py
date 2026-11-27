"""Group and family paired summaries of identical-position three-layer scores."""

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


def dist(values):
    return quantiles(np.asarray(values)[np.isfinite(values)])


def rate(mask, keep):
    return float(mask[keep].mean()) if keep.any() else None


def layer_summary(a, layer, mask):
    valid = mask & a[layer + "_valid"]
    rank = mask & a[layer + "_ranking_valid"]
    pairs = int(a[layer + "_pairwise_count"][valid].sum())
    result = dict(
        queries=int(mask.sum()),
        valid_comparisons=int(valid.sum()),
        valid_rank_queries=int(rank.sum()),
        invalid_comparisons=int((mask & ~a[layer + "_valid"]).sum()),
        win=int((a[layer + "_win"] & valid).sum()),
        loss=int((a[layer + "_loss"] & valid).sum()),
        tie=int((a[layer + "_tie"] & valid).sum()),
        win_rate=rate(a[layer + "_win"], valid),
        loss_rate=rate(a[layer + "_loss"], valid),
        tie_rate=rate(a[layer + "_tie"], valid),
        exact_ties=int((a[layer + "_exact_tie"] & valid).sum()),
        margin=dist(a[layer + "_delta"][valid]),
        best_positive_rank_min=dist(a[layer + "_rank_min"][rank]),
        best_positive_rank_max=dist(a[layer + "_rank_max"][rank]),
        positive_rank1_guaranteed=rate(a[layer + "_rank1_guaranteed"], rank),
        positive_rank1_possible=rate(a[layer + "_rank1_possible"], rank),
        pairwise_count=pairs,
        pairwise_win_rate=(
            int(a[layer + "_pairwise_win"][valid].sum()) / pairs if pairs else None
        ),
        pairwise_loss_rate=(
            int(a[layer + "_pairwise_loss"][valid].sum()) / pairs if pairs else None
        ),
        pairwise_tie_rate=(
            int(a[layer + "_pairwise_tie"][valid].sum()) / pairs if pairs else None
        ),
        continuous_GT_score_sanity=dist(
            a[layer + "_continuous_score"][mask & a["continuous_inverse_valid"]]
        ),
        continuous_minus_best_grid_positive_score=dist(
            (a[layer + "_continuous_score"] - a[layer + "_positive_max"])[
                valid & a["continuous_inverse_valid"]
            ]
        ),
    )
    return result


def distance_summary(values, mask):
    v = values[mask]
    finite = v[np.isfinite(v)]
    return dict(
        queries=int(mask.sum()),
        finite=int(len(finite)),
        quantiles_px=dist(v),
        counts=dict(
            hit_le8=int((finite <= 8).sum()),
            near_gt_8to16=int(((finite > 8) & (finite <= 16)).sum()),
            intermediate_16to32=int(((finite > 16) & (finite <= 32)).sum()),
            far_gt32=int((finite > 32).sum()),
        ),
        interpretation="distance-only descriptive bins; near or far cannot independently certify localization or instance-error cause",
    )


def group_summary(a, mask):
    n = int(mask.sum())
    full = mask & a["C_full_valid"]
    r = dict(
        queries=n,
        layers={l: layer_summary(a, l, mask) for l in ["A", "B", "C"]},
        no_positive=int(((a["positive_original_count"] == 0) & mask).sum()),
        negative_lt16=int((a["negative_insufficient"] & mask).sum()),
        inverse_incomplete=int((~a["inverse_complete"] & mask).sum()),
        continuous_inverse_outside=int((~a["continuous_inverse_valid"] & mask).sum()),
        complete_inverse_queries=int((a["inverse_complete"] & mask).sum()),
        complete_inverse_win_rates={
            l: rate(a[l + "_win"], mask & a["inverse_complete"] & a[l + "_valid"])
            for l in ["A", "B", "C"]
        },
        C_full_best_positive_rank_min=dist(a["C_full_rank_min"][full]),
        C_full_best_positive_rank_max=dist(a["C_full_rank_max"][full]),
        C_before_NMS_top4_coverage_index_tiebreak=rate(
            a["C_raw_top4_nearest"] <= 8, mask
        ),
        C_before_NMS_top4_coverage_guaranteed=rate(a["C_full_top4_guaranteed"], full),
        C_before_NMS_top4_coverage_possible=rate(a["C_full_top4_possible"], full),
        C_after_semantic_only4_NMS_coverage=rate(
            a["C_semantic4_nearest_distance"] <= 8, mask
        ),
        C_existing_2struct2semantic_coverage=rate(a["semantic_nearest"] <= 8, mask),
        C_after_NMS_gains_raw_top4=int(
            (
                (a["C_raw_top4_nearest"] > 8)
                & (a["C_semantic4_nearest_distance"] <= 8)
                & mask
            ).sum()
        ),
        C_after_NMS_loses_raw_top4=int(
            (
                (a["C_raw_top4_nearest"] <= 8)
                & (a["C_semantic4_nearest_distance"] > 8)
                & mask
            ).sum()
        ),
        all_positive_suppressed_by_struct_prefix=int(
            ((a["positive_remaining_after_prefix"] == 0) & mask).sum()
        ),
        structural_first2_hits=int(
            ((a["structure_first2_distance"] <= 8) & mask).sum()
        ),
        C_peak_top1=distance_summary(a["C_semantic4_first_distance"], mask),
        C_nearest_semantic4_peak=distance_summary(
            a["C_semantic4_nearest_distance"], mask
        ),
        C_nearest_supplemental2_peak=distance_summary(
            a["supplemental2_nearest_distance"], mask
        ),
        actual_2struct2semantic_nearest=distance_summary(a["semantic_nearest"], mask),
        inverse_continuous_roundtrip_error_px=dist(
            a["continuous_inverse_to_source_error"][
                mask & a["continuous_inverse_valid"]
            ]
        ),
    )
    for left, right in [("A", "B"), ("B", "C"), ("A", "C")]:
        valid = mask & a[left + "_valid"] & a[right + "_valid"]
        r[f"{left}_to_{right}"] = dict(
            valid=int(valid.sum()),
            win_to_loss=int((a[left + "_win"] & a[right + "_loss"] & valid).sum()),
            win_to_tie=int((a[left + "_win"] & a[right + "_tie"] & valid).sum()),
            nonwin_to_win=int((~a[left + "_win"] & a[right + "_win"] & valid).sum()),
            paired_win_rate_change=(
                float(
                    (
                        a[right + "_win"][valid].astype(float) - a[left + "_win"][valid]
                    ).mean()
                )
                if valid.any()
                else None
            ),
        )
    return r


def family_row(a, family, mask):
    r = dict(
        family_id=family,
        queries=int(mask.sum()),
        inverse_incomplete=int((mask & ~a["inverse_complete"]).sum()),
    )
    for l in ["A", "B", "C"]:
        v = mask & a[l + "_valid"]
        rv = mask & a[l + "_ranking_valid"]
        r.update(
            {
                l + "_valid": int(v.sum()),
                l + "_win": int((v & a[l + "_win"]).sum()),
                l + "_loss": int((v & a[l + "_loss"]).sum()),
                l + "_tie": int((v & a[l + "_tie"]).sum()),
                l
                + "_margin_median": (
                    float(np.median(a[l + "_delta"][v])) if v.any() else None
                ),
                l
                + "_rank_min_median": (
                    float(np.median(a[l + "_rank_min"][rv])) if rv.any() else None
                ),
                l
                + "_rank_max_median": (
                    float(np.median(a[l + "_rank_max"][rv])) if rv.any() else None
                ),
            }
        )
    for k, hit in [
        ("C_raw4", a["C_raw_top4_nearest"] <= 8),
        ("C_NMS4", a["C_semantic4_nearest_distance"] <= 8),
        ("C_supplement", a["semantic_nearest"] <= 8),
        ("structure", a["structure_nearest"] <= 8),
    ]:
        r[k + "_hits"] = int((hit & mask).sum())
    for key in ["C_full_rank_min", "positive_remaining_after_prefix"]:
        r[key + "_median"] = float(np.median(a[key][mask])) if mask.any() else None
    return r


def bootstrap(rows, config):
    rows = [r for r in rows if r["queries"]]
    if not rows:
        return {}
    rng = np.random.default_rng(config["seed"])
    keys = [
        "queries",
        "A_valid",
        "A_win",
        "B_valid",
        "B_win",
        "C_valid",
        "C_win",
        "C_raw4_hits",
        "C_NMS4_hits",
        "C_supplement_hits",
        "structure_hits",
    ]
    data = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    metrics = {
        k: []
        for k in [
            "A_win",
            "B_win",
            "C_win",
            "B_minus_A_win",
            "C_minus_B_win",
            "C_minus_A_win",
            "C_raw4_coverage",
            "C_NMS4_coverage",
            "C_supplement_minus_structure",
        ]
    }
    for _ in range(config["replicates"]):
        n, av, aw, bv, bw, cv, cw, raw, nms, supp, st = data[
            rng.integers(0, len(data), len(data))
        ].sum(0)
        wa = aw / av if av else np.nan
        wb = bw / bv if bv else np.nan
        wc = cw / cv if cv else np.nan
        vals = [
            wa,
            wb,
            wc,
            wb - wa,
            wc - wb,
            wc - wa,
            raw / n,
            nms / n,
            (supp - st) / n,
        ]
        for k, v in zip(metrics, vals):
            metrics[k].append(v)
    return dict(
        families=len(rows),
        replicates=config["replicates"],
        seed=config["seed"],
        interval95={
            k: (
                np.nanpercentile(v, [2.5, 97.5]).tolist()
                if np.isfinite(v).any()
                else None
            )
            for k, v in metrics.items()
        },
        scope="paired family sampling conditional on frozen model/labels; not causal attribution, geographic independence or repeat-seed uncertainty",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_semantic_bottleneck.json")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    all_summaries = {}
    layer_rows = []
    family_rows = []
    for split in config["evaluation_splits"]:
        z = np.load(root / f"{split}_query_diagnostics.npz")
        a = {k: z[k] for k in z.files}
        summaries = {}
        for group in config["groups"]:
            mask = a["group_" + group]
            summary = group_summary(a, mask)
            summaries[group] = summary
            fr = []
            for family in sorted(set(a["family_id"])):
                fm = mask & (a["family_id"] == family)[:, None]
                row = family_row(a, family, fm)
                row.update(split=split, group=group)
                fr.append(row)
                family_rows.append(row)
            summary["family_paired_intervals"] = bootstrap(fr, config["bootstrap"])
            for layer, ls in summary["layers"].items():
                layer_rows.append(
                    dict(
                        split=split,
                        group=group,
                        layer=layer,
                        queries=ls["queries"],
                        valid_comparisons=ls["valid_comparisons"],
                        win_rate=ls["win_rate"],
                        loss_rate=ls["loss_rate"],
                        tie_rate=ls["tie_rate"],
                        delta_median=ls["margin"].get("50"),
                        delta_p5=ls["margin"].get("5"),
                        delta_p95=ls["margin"].get("95"),
                        rank_min_median=ls["best_positive_rank_min"].get("50"),
                        rank_max_median=ls["best_positive_rank_max"].get("50"),
                        guaranteed_rank1_rate=ls["positive_rank1_guaranteed"],
                        possible_rank1_rate=ls["positive_rank1_possible"],
                        pairwise_win_rate=ls["pairwise_win_rate"],
                        pairwise_loss_rate=ls["pairwise_loss_rate"],
                        pairwise_tie_rate=ls["pairwise_tie_rate"],
                    )
                )
        all_summaries[split] = summaries
        save_json(root / f"{split}_group_summary.json", summaries)
        # Include every original query and its exclusion flags, not only successes.
        fields = [
            "sample_id",
            "family_id",
            "query_id",
            "supervised",
            "groups",
            "P_count",
            "N_count",
            "inverse_complete",
        ]
        fields += [
            l + "_" + k
            for l in ["A", "B", "C"]
            for k in ["valid", "rank_min", "rank_max", "delta", "win", "loss", "tie"]
        ]
        fields += [
            "C_full_rank_min",
            "C_full_rank_max",
            "C_raw_top4_nearest",
            "C_semantic4_first_distance",
            "C_semantic4_nearest_distance",
            "supplemental2_nearest_distance",
            "positive_remaining_after_prefix",
        ]
        with gzip.open(root / f"{split}_per_query.csv.gz", "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, sample in enumerate(a["sample_id"]):
                for q in range(a["supervised"].shape[1]):
                    row = dict(
                        sample_id=sample,
                        family_id=a["family_id"][i],
                        query_id=q,
                        supervised=bool(a["supervised"][i, q]),
                        groups=";".join(
                            g for g in config["groups"] if a["group_" + g][i, q]
                        ),
                        P_count=int(a["positive_original_count"][i, q]),
                        N_count=int(a["negative_original_count"][i, q]),
                        inverse_complete=bool(a["inverse_complete"][i, q]),
                    )
                    for key in fields[8:]:
                        row[key] = a[key][i, q].item()
                    writer.writerow(row)
    save_csv(root / "layer_group_comparison.csv", layer_rows)
    save_csv(root / "family_paired_diagnostics.csv", family_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    groups = ["all_supervised", "structure_miss", "semantic_new", "semantic_lost"]
    val = all_summaries["validation"]
    for l, color in [("A", "tab:blue"), ("B", "tab:orange"), ("C", "tab:green")]:
        axes[0].plot(
            np.arange(4),
            [100 * val[g]["layers"][l]["win_rate"] for g in groups],
            "o-",
            label=l,
            color=color,
        )
        axes[1].plot(
            np.arange(4),
            [val[g]["layers"][l]["best_positive_rank_min"]["50"] for g in groups],
            "o-",
            label=l,
            color=color,
        )
    for ax in axes[:2]:
        ax.set_xticks(np.arange(4), ["All", "Struct miss", "New277", "Lost5645"])
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Best P beats max hard N (%)")
    axes[1].set_ylabel("Median best-P optimistic rank (P + up to16 N)")
    for i, g in enumerate(["semantic_new", "semantic_lost"]):
        counts = val[g]["C_nearest_supplemental2_peak"]["counts"]
        axes[2].bar(
            np.arange(4) + i * 0.35,
            [
                counts[k]
                for k in ["hit_le8", "near_gt_8to16", "intermediate_16to32", "far_gt32"]
            ],
            width=0.35,
            label=g,
        )
    axes[2].set_xticks(np.arange(4) + 0.175, ["<=8", "8..16", "16..32", ">32"])
    axes[2].set_ylabel("Queries, nearest added semantic peak")
    axes[2].set_xlabel("GT distance (native px)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "bottleneck_group_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g, color in [
        ("all_supervised", "black"),
        ("structure_miss", "tab:blue"),
        ("semantic_new", "tab:green"),
        ("semantic_lost", "tab:red"),
    ]:
        v = val[g]
        axes[0].plot(
            [0, 1, 2],
            [
                100 * v[k]
                for k in [
                    "C_before_NMS_top4_coverage_index_tiebreak",
                    "C_after_semantic_only4_NMS_coverage",
                    "C_existing_2struct2semantic_coverage",
                ]
            ],
            "o-",
            label=g,
            color=color,
        )
        z = np.load(root / "validation_query_diagnostics.npz")
        mask = z["group_" + g]
        values = np.sort(z["C_full_rank_min"][mask])
        axes[1].plot(
            values, np.arange(1, len(values) + 1) / len(values), label=g, color=color
        )
    axes[0].set_xticks([0, 1, 2], ["Raw semantic top4", "Semantic NMS4", "Existing2+2"])
    axes[0].set_ylabel("Coverage (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("C full-grid best-P optimistic rank")
    axes[1].set_ylabel("CDF")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "C_grid_rank_and_NMS.png", dpi=180)
    plt.close(fig)
    print(
        "Group, family paired intervals and all-query diagnostics summarized",
        flush=True,
    )


if __name__ == "__main__":
    main()
