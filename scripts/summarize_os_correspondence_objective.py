"""Final controlled branch comparison, fixed subgroups and paired family intervals."""

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


def family_bootstrap(rows, cfg):
    cols = [
        "n",
        "control_hits",
        "corr_hits",
        "control_new",
        "control_lost",
        "corr_new",
        "corr_lost",
        "structure_hits",
        "control_PNwin",
        "corr_PNwin",
        "PNvalid",
        "control_NMS4",
        "corr_NMS4",
    ]
    data = np.array([[r[k] for k in cols] for r in rows if r["n"]], dtype=float)
    keys = [
        "corr_minus_control_recall4",
        "control_minus_structure_recall4",
        "corr_minus_structure_recall4",
        "control_rescue_rate",
        "corr_rescue_rate",
        "corr_minus_control_PNwin",
        "corr_minus_control_semantic4",
    ]
    result = {k: [] for k in keys}
    rng = np.random.default_rng(cfg["seed"])
    for _ in range(cfg["replicates"]):
        n, c, e, cn, cl, en, el, st, cw, ew, pv, cs, es = data[
            rng.integers(0, len(data), len(data))
        ].sum(0)
        vals = [
            (e - c) / n,
            (c - st) / n,
            (e - st) / n,
            cn / (n - st) if n > st else np.nan,
            en / (n - st) if n > st else np.nan,
            (ew - cw) / pv if pv else np.nan,
            (es - cs) / n,
        ]
        for k, v in zip(keys, vals):
            result[k].append(v)
    return dict(
        families=len(data),
        replicates=cfg["replicates"],
        seed=cfg["seed"],
        interval95={
            k: (
                np.nanpercentile(v, [2.5, 97.5]).tolist()
                if np.isfinite(v).any()
                else None
            )
            for k, v in result.items()
        },
        scope="paired family sampling conditional on one training seed and source-family split, not repeat-seed or geographic uncertainty",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_correspondence_objective.json")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output"])
    rows = []
    family_rows = []
    summaries = {}
    groups_all = {}
    constant_reference = {}
    for split in ["training_monitor", "validation"]:
        prev = np.load(Path(cfg["score_control_root"]) / f"{split}_score_queries.npz")
        final = np.load(
            root
            / "evaluation"
            / f"step{cfg['additional_steps']:05d}_{split}_queries.npz"
        )
        keep = final["supervised"]
        pc = final["positive"].sum(-1)
        nc = final["negative_valid"].sum(-1)
        eligible = keep & (pc > 0) & (nc > 0)
        constant_reference[split] = dict(
            constant_equal_target_scores_corr_mean=float(
                np.log1p(nc[eligible] / pc[eligible]).mean()
            ),
            definition="log((|P|+|N|)/|P|), same target score for each query; no training or model inference",
            eligible_queries=int(eligible.sum()),
            positive_count_distribution={
                str(int(k)): int(n)
                for k, n in zip(*np.unique(pc[eligible], return_counts=True))
            },
            interpretation="analytic diagnostic reference, not a trained third arm or a model selection criterion",
        )
        names = final["family_id"]
        start = np.load(root / "evaluation" / f"step00000_{split}_queries.npz")
        for step in cfg["evaluation_additional_steps"]:
            result = json.loads(
                (
                    root / "evaluation" / f"step{step:05d}_{split}_summary.json"
                ).read_text()
            )
            for branch, r in result.items():
                rows.append(
                    dict(
                        split=split,
                        branch=branch,
                        additional_step=step,
                        total_step=cfg["start_step"] + step,
                        KL=r["KL"],
                        corr=r["corr"],
                        PN_win_rate=r["PN_win_rate"],
                        PN_delta_median=r["PN_delta"]["50"],
                        full_rank_median=r["full_best_positive_rank_min"]["50"],
                        full_rank_p95=r["full_best_positive_rank_min"]["95"],
                        independent_semantic4=r["semantic_independent4_coverage"],
                        recall_at_1=r["recall_at_1"],
                        recall_at_4=r["recall_at_4"],
                        new_hits=r["new_hits"],
                        lost_hits=r["lost_hits"],
                        net_hits=r["net_hits"],
                        rescue_rate=r["rescue_rate"],
                        median_distance_px=r["nearest_distance_px"]["50"],
                        p95_distance_px=r["nearest_distance_px"]["95"],
                        entropy=r["SAR_distribution"]["mean_query_entropy_nats"],
                        spatial_rms=r["SAR_distribution"][
                            "per_image_spatial_probability_rms"
                        ]["50"],
                    )
                )
        complete = json.loads(
            (
                root
                / "evaluation"
                / f"step{cfg['additional_steps']:05d}_{split}_summary.json"
            ).read_text()
        )
        local_families = []
        for fam in sorted(set(names)):
            idx = np.flatnonzero(names == fam)
            k = keep[idx]
            n = int(k.sum())
            row = dict(
                split=split,
                family_id=fam,
                n=n,
                structure_hits=int(((final["structure_nearest"][idx] <= 8) & k).sum()),
            )
            for branch, short in [("KL_control", "control"), ("KL_corr", "corr")]:
                counts = paired_counts(
                    final["structure_nearest"][idx], final[branch + "_nearest"][idx], k
                )
                for key in [
                    "hits",
                    "new_hits",
                    "lost_hits",
                    "net_hits",
                    "recall_at_4",
                    "rescue_rate",
                ]:
                    row[
                        (
                            short + "_" + key.replace("_hits", "")
                            if key in ["new_hits", "lost_hits"]
                            else short + "_" + key
                        )
                    ] = counts[key]
                row[short + "_PNwin"] = int((final[branch + "_PN_win"][idx] & k).sum())
                row[short + "_NMS4"] = int(
                    ((final[branch + "_semantic4_nearest"][idx] <= 8) & k).sum()
                )
                row[short + "_KL"] = float(final[branch + "_KL"][idx][k].mean())
            row["PNvalid"] = int((final["KL_control_PN_valid"][idx] & k).sum())
            row["net_corr_minus_control"] = row["corr_hits"] - row["control_hits"]
            row["delta_recall4"] = row["net_corr_minus_control"] / n
            family_rows.append(row)
            local_families.append(row)
        c = (final["KL_control_nearest"] <= 8) & keep
        e = (final["KL_corr_nearest"] <= 8) & keep
        comparative = dict(
            split=split,
            effective_queries=int(keep.sum()),
            branches=complete,
            experiment_new_vs_control=int((~c & e & keep).sum()),
            experiment_lost_vs_control=int((c & ~e & keep).sum()),
            experiment_net_vs_control=int(e.sum()) - int(c.sum()),
            family_change=dict(
                improved=sum(r["net_corr_minus_control"] > 0 for r in local_families),
                degraded=sum(r["net_corr_minus_control"] < 0 for r in local_families),
                unchanged=sum(r["net_corr_minus_control"] == 0 for r in local_families),
            ),
            family_intervals=family_bootstrap(local_families, cfg["bootstrap"]),
        )
        # Intermediate evidence: ordering and goal conflict, not substituted for coverage.
        pv = keep & final["KL_control_PN_valid"] & final["KL_corr_PN_valid"]
        comparative["paired_PN_change"] = dict(
            control_loss_to_experiment_win=int(
                (final["KL_control_PN_loss"] & final["KL_corr_PN_win"] & pv).sum()
            ),
            control_win_to_experiment_loss=int(
                (final["KL_control_PN_win"] & final["KL_corr_PN_loss"] & pv).sum()
            ),
        )
        groups = {}
        for group in [
            "all_supervised",
            "structure_miss",
            "semantic_new",
            "semantic_lost",
            "retained_hit",
            "both_miss",
        ]:
            k = prev["group_" + group]
            group_row = dict(queries=int(k.sum()), branches={})
            for branch in cfg["branches"]:
                v = k & final[branch + "_PN_valid"]
                ce = k & final[branch + "_corr_eligible"]
                group_row["branches"][branch] = dict(
                    **paired_counts(
                        final["structure_nearest"], final[branch + "_nearest"], k
                    ),
                    KL=float(final[branch + "_KL"][k].mean()) if k.any() else None,
                    corr=(
                        float(final[branch + "_corr"][ce].mean()) if ce.any() else None
                    ),
                    PN_win_rate=(
                        float(final[branch + "_PN_win"][v].mean()) if v.any() else None
                    ),
                    PN_delta=quantiles(final[branch + "_PN_delta"][v]),
                    full_rank=quantiles(final[branch + "_full_rank_min"][k]),
                    independent_semantic4=(
                        float((final[branch + "_semantic4_nearest"][k] <= 8).mean())
                        if k.any()
                        else None
                    ),
                )
            groups[group] = group_row
        groups_all[split] = groups
        summaries[split] = comparative
        save_json(root / f"{split}_final_comparison.json", comparative)
        save_json(root / f"{split}_fixed_groups.json", groups)
    save_csv(root / "objective_continuation_history.csv", rows)
    save_json(root / "constant_score_reference.json", constant_reference)
    save_csv(root / "paired_family_final.csv", family_rows)
    gradients = list(
        csv.DictReader((root / "component_gradient_diagnostics.csv").open())
    )
    save_json(
        root / "loss_conflict_summary.json",
        dict(
            gradient_checks=[
                {
                    k: (
                        v
                        if k == "branch"
                        else (
                            float(v)
                            if k
                            not in [
                                "conflicting",
                                "all_head_parameters_reached_by_both",
                            ]
                            else v == "True"
                        )
                    )
                    for k, v in r.items()
                }
                for r in gradients
            ],
            note="each row is one fixed training batch at prespecified updates, descriptive sparse checks only; cosine<0 indicates local objective conflict, not entire trajectory frequency",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for split, color in [
        ("training_monitor", "tab:blue"),
        ("validation", "tab:orange"),
    ]:
        for branch, style in [("KL_control", "--"), ("KL_corr", "-")]:
            selected = [
                r for r in rows if r["split"] == split and r["branch"] == branch
            ]
            for ax, key in zip(axes, ["KL", "PN_win_rate", "recall_at_4"]):
                ax.plot(
                    [r["additional_step"] for r in selected],
                    [r[key] * (100 if key != "KL" else 1) for r in selected],
                    style + "o",
                    color=color,
                    label=split + " " + branch,
                )
    for ax, title in zip(
        axes,
        [
            "Original KL on fixed view",
            "Distance PN win rate (%)",
            "Actual2+2distance recall@4 (%)",
        ],
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Additional updates")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "paired_objective_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, split in enumerate(["training_monitor", "validation"]):
        r = summaries[split]
        a = r["branches"]
        delta = 100 * (a["KL_corr"]["recall_at_4"] - a["KL_control"]["recall_at_4"])
        lo, hi = 100 * np.array(
            r["family_intervals"]["interval95"]["corr_minus_control_recall4"]
        )
        axes[0].errorbar(
            i, delta, yerr=[[delta - lo], [hi - delta]], fmt="o", capsize=5
        )
        axes[1].bar(
            np.arange(2) + i * 0.3,
            [r["experiment_new_vs_control"], r["experiment_lost_vs_control"]],
            width=0.3,
            label=split,
        )
    axes[0].axhline(0, color="gray")
    axes[0].set_xticks([0, 1], ["Train monitor", "Validation"])
    axes[0].set_ylabel("Experiment minus KL control recall@4 (pp)")
    axes[0].set_title("Paired95%family-sampling interval")
    axes[1].set_xticks([0.15, 1.15], ["New hits vs control", "Lost vs control"])
    axes[1].set_ylabel("Source queries")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "paired_objective_net_coverage.png", dpi=180)
    plt.close(fig)
    if (root / "negative_boundary_query_audit.json").exists():
        boundary = json.loads((root / "negative_boundary_query_audit.json").read_text())
        slot = json.loads((root / "negative_distance_training.json").read_text())
        exp = [r for r in gradients if r["branch"] == "KL_corr"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        x = [int(r["additional_step"]) for r in exp]
        axes[0].plot(x, [float(r["KL_gradient_norm"]) for r in exp], "o-", label="KL")
        axes[0].plot(
            x,
            [float(r["weighted_corr_gradient_norm"]) for r in exp],
            "o-",
            label="alpha * corr",
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Additional update (prespecified checks)")
        axes[0].set_ylabel("SAR head gradient norm")
        axes[0].legend()
        axes[1].plot(x, [float(r["gradient_cosine"]) for r in exp], "o-")
        axes[1].axhline(0, color="gray")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Additional update")
        axes[1].set_ylabel("KL / corr gradient cosine")
        near10 = [
            slot["counts"][0] / slot["valid_negative_slots"],
            boundary["closest_negative_per_query"]["fraction_8to10"],
            boundary["first_highest_structure_negative_per_query"]["fraction_8to10"],
        ]
        near16 = [
            slot["fraction_8to16"],
            boundary["closest_negative_per_query"]["fraction_8to16"],
            boundary["first_highest_structure_negative_per_query"]["fraction_8to16"],
        ]
        xx = np.arange(3)
        axes[2].bar(xx - 0.18, np.array(near10) * 100, 0.36, label="8..10px")
        axes[2].bar(
            xx + 0.18, np.array(near16) * 100, 0.36, label="8..16px (includes8..10)"
        )
        axes[2].set_xticks(
            xx,
            ["All negative slots", "Closest / query", "First structure / query"],
            fontsize=8,
        )
        axes[2].set_ylabel("Fraction (%)")
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / "gradient_conflict_and_negative_boundary.png", dpi=180)
        plt.close(fig)
    print(
        "Equal-budget branch comparison and fixed-family paired intervals exported",
        flush=True,
    )


if __name__ == "__main__":
    main()
