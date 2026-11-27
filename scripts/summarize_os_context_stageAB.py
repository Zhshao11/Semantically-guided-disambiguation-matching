"""First-delivery summaries: mandatory structural ranks and equal-capacity fixed fits."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.io import save_json, save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_context_descriptor.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output"])
    history = json.loads((root / "smoke_evaluation_history.json").read_text())
    rows = []
    baseline = []
    for split in ["training_monitor", "validation"]:
        s = json.loads((root / "baseline" / (split + "_summary.json")).read_text())[
            "structure"
        ]
        for group, metric in s["ranking"].items():
            baseline.append(
                dict(
                    split=split,
                    group=group,
                    queries=metric["queries"],
                    rank_median=metric["best_positive_rank_min"]["50"],
                    rank_p75=metric["best_positive_rank_min"]["75"],
                    rank_p95=metric["best_positive_rank_min"]["95"],
                    pessimistic_rank_median=metric["best_positive_rank_max"]["50"],
                    pessimistic_rank_p75=metric["best_positive_rank_max"]["75"],
                    pessimistic_rank_p95=metric["best_positive_rank_max"]["95"],
                    recall_at_1=metric["top1_coverage"],
                    recall_at_4=metric["independent_NMS4_coverage"],
                )
            )
    save_csv(root / "pure_structure_baseline_ranks.csv", baseline)
    for h in history:
        s = h["summary"]
        a = s["ranking"]["all"]
        miss = s["ranking"]["structure_miss"]
        group = h["method"]
        row = dict(
            step=h["step"],
            group=group,
            loss=s["loss_pair_mean"],
            rank_median=a["best_positive_rank_min"]["50"],
            rank_p75=a["best_positive_rank_min"]["75"],
            rank_p95=a["best_positive_rank_min"]["95"],
            structure_miss_rank_median=miss["best_positive_rank_min"].get("50"),
            structure_miss_rank_p75=miss["best_positive_rank_min"].get("75"),
            structure_miss_rank_p95=miss["best_positive_rank_min"].get("95"),
            top1=a["top1_coverage"],
            independent_NMS4=a["independent_NMS4_coverage"],
            merged_recall4=s["recall_at_4"],
            new_hits=s["new_hits"],
            lost_hits=s["lost_hits"],
            net_hits=s["net_hits"],
            rescue_rate=s["rescue_rate"],
        )
        if "descriptor_spatial_diagnostics" in s:
            for side in ["optical", "sar"]:
                stats = s["descriptor_spatial_diagnostics"][side]
                row[side + "_mean_spatial_variance"] = float(
                    np.mean([z["variance"] for z in stats])
                )
                row[side + "_mean_offdiagonal_cosine"] = float(
                    np.mean([z["average_offdiagonal_cosine"] for z in stats])
                )
        rows.append(row)
    save_csv(root / "four_group_small_fit_history.csv", rows)
    final = {r["group"]: r for r in rows if r["step"] == cfg["smoke"]["steps"]}
    start = {r["group"]: r for r in rows if r["step"] == 0}
    response = {}
    for name in cfg["groups"]:
        a = start[name]
        b = final[name]
        response[name] = dict(
            initial_loss=a["loss"],
            final_loss=b["loss"],
            loss_ratio=b["loss"] / a["loss"],
            initial_rank=a["rank_median"],
            final_rank=b["rank_median"],
            initial_independent4=a["independent_NMS4"],
            final_independent4=b["independent_NMS4"],
            optical_spatial_variance=b["optical_mean_spatial_variance"],
            SAR_spatial_variance=b["sar_mean_spatial_variance"],
            engineering_response=b["loss"] < a["loss"]
            and b["independent_NMS4"] > a["independent_NMS4"]
            and min(b["optical_mean_spatial_variance"], b["sar_mean_spatial_variance"])
            > 1e-5,
            interpretation="numeric and fit response only, no arbitrary90% admission threshold or validation-based selection",
        )
    # Full fit can still be capped by a fixed structural prefix's suppression.
    data = np.load(root / "smoke" / f"step{cfg['smoke']['steps']:05d}_queries.npz")
    source_idx = np.rint(np.linspace(0, 31, 12)).astype(int)
    yy, xx = np.meshgrid(source_idx, source_idx, indexing="ij")
    source = np.stack([xx, yy], -1).reshape(-1, 2) * 8 + 3.5
    grid = np.stack(
        np.meshgrid(np.arange(32) * 8 + 3.5, np.arange(32) * 8 + 3.5), -1
    ).reshape(-1, 2)
    prefix = grid[data["structure_indices"][..., :2]]
    available = np.ones((len(data["sample_id"]), 144, 1024), dtype=bool)
    for k in range(2):
        available &= ((grid[None, None] - prefix[:, :, k, None]) ** 2).sum(-1) > 24**2
    positive = ((grid[None, None] - source[None, :, None]) ** 2).sum(-1) <= 8**2
    blocked = ~(positive & available).any(-1)
    sh = data["structure_nearest"] <= 8
    kh = data["supervised"]
    ceiling = (
        (np.linalg.norm(prefix - source[None, :, None], axis=-1).min(-1) <= 8)
        | ~blocked
    ) & kh
    save_json(
        root / "small_fit_prefix_geometry.json",
        dict(
            supervised_queries=int(kh.sum()),
            P_fully_suppressed_by_prefix=int((blocked & kh).sum()),
            structure_miss_P_fully_suppressed=int((blocked & ~sh & kh).sum()),
            two_plus_two_geometric_upper_coverage=float(ceiling[kh].mean()),
            note="GT used only to diagnose fixed-prefix reachability, never to select candidates or alter windows; identity32pair training subset only",
        ),
    )
    save_json(
        root / "stageAB_response.json",
        dict(
            groups=response,
            all_groups_responded=all(
                r["engineering_response"] for r in response.values()
            ),
            formal_budget_run=False,
            stageC_preregistered_seeds=cfg["formal_preregistration"]["paired_seeds"],
            first_delivery_only=True,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for group in cfg["groups"]:
        selected = [r for r in rows if r["group"] == group]
        for ax, key in zip(axes, ["loss", "independent_NMS4", "merged_recall4"]):
            ax.plot(
                [r["step"] for r in selected],
                [r[key] * (100 if key != "loss" else 1) for r in selected],
                "o-",
                label=group,
            )
    axes[0].set_ylabel("Per-pair full-grid positive-mass loss")
    axes[1].set_ylabel("Independent NMS4 coverage (%)")
    axes[2].set_ylabel("Fixed2structure+2learned coverage (%)")
    for ax in axes:
        ax.set_xlabel("Fixed32-pair fitting updates")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Engineering fit only: same initialization, batches,800updates per group"
    )
    fig.tight_layout()
    fig.savefig(root / "four_group_small_fit.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row, split in enumerate(["training_monitor", "validation"]):
        selected = [r for r in baseline if r["split"] == split]
        for i, r in enumerate(selected):
            axes[row].plot(
                [50, 75, 95],
                [r["rank_median"], r["rank_p75"], r["rank_p95"]],
                "o-",
                label=r["group"],
            )
        axes[row].set_title(split + " frozen structural score")
        axes[row].set_xlabel("Best-positive rank percentile")
        axes[row].set_ylabel("Native1024-grid rank")
        axes[row].legend()
        axes[row].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "pure_structure_rank_baseline.png", dpi=180)
    plt.close(fig)
    print(
        "First-delivery structural ranks and four small-fit responses summarized",
        flush=True,
    )


if __name__ == "__main__":
    main()
