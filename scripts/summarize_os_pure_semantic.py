"""Summarize the frozen pure-semantic supplementation control without tuning."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.io import save_csv, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_pure_semantic_supplement.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    rows = []
    all_data = {}
    overlap = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for s, color in [("training_monitor", "tab:blue"), ("validation", "tab:orange")]:
        summary = json.loads((root / f"{s}_summary.json").read_text())
        all_data[s] = summary
        for method in ["structure", "fusion", "semantic"]:
            rows.append(
                dict(
                    split=s,
                    method=method,
                    effective_queries=summary["effective_queries"],
                    original_misses=summary["original_misses"],
                    **{
                        k: v
                        for k, v in summary["methods"][method].items()
                        if k not in ["nearest_distance_px"]
                    },
                )
            )
        methods = summary["methods"]
        axes[0].plot(
            ["4 structure", "2 structure +2 fusion", "2 structure +2 semantic"],
            [
                100 * methods[k]["recall_at_4"]
                for k in ["structure", "fusion", "semantic"]
            ],
            "o-",
            label=s,
            color=color,
        )
        x = np.arange(2) + (0 if s == "training_monitor" else 0.25)
        axes[1].bar(
            x,
            [100 * methods[k]["rescue_rate"] for k in ["fusion", "semantic"]],
            width=0.25,
            label=s,
            color=color,
        )
        axes[2].bar(
            x,
            [methods[k]["net_hits"] for k in ["fusion", "semantic"]],
            width=0.25,
            label=s,
            color=color,
        )
        q = np.load(root / f"{s}_queries.npz")
        keep = q["supervised"]
        miss = (q["structure_nearest"] > 8) & keep
        rf = miss & (q["fusion_nearest"] <= 8)
        rs = miss & (q["semantic_nearest"] <= 8)
        overlap[s] = dict(
            original_misses=int(miss.sum()),
            rescued_by_both=int((rf & rs).sum()),
            only_fusion=int((rf & ~rs).sum()),
            only_semantic=int((rs & ~rf).sum()),
            rescued_by_neither=int((miss & ~rf & ~rs).sum()),
            interpretation="descriptive rescue-set overlap only; no combined candidate budget evaluated",
        )
    axes[0].set_ylabel("Recall@4 (%)")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].set_xticks([0.125, 1.125], ["Fusion", "Pure semantic"])
    axes[1].set_ylabel("Rescued / original misses (%)")
    axes[2].set_xticks([0.125, 1.125], ["Fusion", "Pure semantic"])
    axes[2].set_ylabel("New hits minus lost hits")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "three_way_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, split in enumerate(["training_monitor", "validation"]):
        summary = all_data[split]
        for j, method in enumerate(["fusion", "semantic"]):
            d = 100 * summary["methods"][method]["delta_recall4"]
            interval = 100 * np.array(
                summary["family_intervals"]["interval95"][
                    f"{method}_minus_structure_recall4"
                ]
            )
            axes[i].errorbar(
                j, d, yerr=[[d - interval[0]], [interval[1] - d]], fmt="o", capsize=6
            )
        axes[i].axhline(0, color="gray", lw=1)
        axes[i].set_xticks(
            [0, 1], ["Fusion minus structure", "Semantic minus structure"]
        )
        axes[i].set_title(split)
        axes[i].set_ylabel("Delta recall@4 (percentage points)")
        axes[i].grid(axis="y", alpha=0.2)
    fig.suptitle("95% paired family-resampling intervals (fixed trained seed)")
    fig.tight_layout()
    fig.savefig(root / "paired_family_intervals.png", dpi=180)
    plt.close(fig)
    save_csv(root / "three_way_comparison.csv", rows)
    save_json(root / "rescue_overlap.json", overlap)
    print(
        "Three-way comparison, paired intervals and rescue overlap exported", flush=True
    )


if __name__ == "__main__":
    main()
