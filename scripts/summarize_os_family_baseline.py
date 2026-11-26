"""Static scientific plots and compact results for source-family validation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.io import save_json, save_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_family_baseline.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    run = root / "run"
    history = json.loads((run / "evaluation_history.json").read_text())
    final = history[-1]
    small = json.loads(
        (root / "small32_diagnostic/small32_validation_metrics.json").read_text()
    )
    save_json(
        root / "result_summary.json",
        dict(
            grouping=json.loads((root / "grouping_summary.json").read_text()),
            small32_validation=small,
            formal_final_training=final["training"],
            formal_final_validation=final["validation"],
            completion=json.loads((run / "completion.json").read_text()),
            UBC_risk=json.loads((root / "ubcv2_risk/risk_summary.json").read_text()),
            direct_comparison_to_512_for_method_improvement=False,
        ),
    )
    rows = []
    for point in history:
        for split in ["training", "validation"]:
            m = point[split]
            q = m["nearest_candidate_distance_px"]
            rows.append(
                dict(
                    epoch=point["epoch"],
                    split=split,
                    loss=m["loss"],
                    recall1=m["recall_at_1"],
                    recall4=m["recall_at_4"],
                    median_px=q["50"],
                    p95_px=q["95"],
                    p99_px=q["99"],
                )
            )
    save_csv(run / "evaluation_curve.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for split, color in [("training", "#2465a6"), ("validation", "#c04d2b")]:
        points = [r for r in rows if r["split"] == split]
        x = [r["epoch"] for r in points]
        axes[0].plot(
            x, [r["loss"] for r in points], marker="o", color=color, label=split
        )
        axes[1].plot(
            x,
            [r["recall1"] for r in points],
            marker="o",
            color=color,
            label=split + " @1",
        )
        axes[1].plot(
            x,
            [r["recall4"] for r in points],
            marker="s",
            linestyle="--",
            color=color,
            label=split + " @4",
        )
        axes[2].plot(
            x,
            [r["median_px"] for r in points],
            marker="o",
            color=color,
            label=split + " median",
        )
        axes[2].plot(
            x,
            [r["p95_px"] for r in points],
            linestyle="--",
            color=color,
            label=split + " P95",
        )
    for axis, title in zip(
        axes,
        [
            "Fixed-perturbation loss",
            "Native recall at 8 px",
            "Nearest candidate to synthetic label",
        ],
    ):
        axis.set(xlabel="Complete training epoch", title=title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylabel("Native pixels")
    fig.savefig(root / "training_validation_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for path, label, color in [
        (
            root / "small32_diagnostic/small32_validation_distances.npz",
            "32-pair model / validation",
            "#777777",
        ),
        (
            run / f"epoch{config['epochs']:02d}_training_distances.npz",
            "formal / training",
            "#2465a6",
        ),
        (
            run / f"epoch{config['epochs']:02d}_validation_distances.npz",
            "formal / validation",
            "#c04d2b",
        ),
    ]:
        with np.load(path) as data:
            values = np.sort(data["nearest_distance_px"][data["supervised"]])
            axes[0].plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                label=label,
                color=color,
            )
            pair_recall = ((data["nearest_distance_px"] <= 8) & data["supervised"]).sum(
                1
            ) / data["supervised"].sum(1)
            axes[1].hist(
                pair_recall,
                bins=np.linspace(0, 1, 21),
                histtype="step",
                density=True,
                label=label,
                color=color,
            )
            pair_medians = np.array(
                [
                    np.median(d[m])
                    for d, m in zip(data["nearest_distance_px"], data["supervised"])
                    if m.any()
                ]
            )
            axes[2].hist(
                pair_medians,
                bins=np.arange(0, 66, 3),
                histtype="step",
                density=True,
                label=label,
                color=color,
            )
    axes[0].axvline(8, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlim=(0, 150),
        ylim=(0, 1),
        xlabel="Native pixels to synthetic label",
        ylabel="Empirical CDF",
        title="Point-weighted distance",
    )
    axes[1].set(
        xlabel="Pair recall@4",
        ylabel="Density",
        title="Distribution across image pairs",
    )
    axes[2].set(
        xlabel="Pair median nearest distance (px)",
        ylabel="Density",
        title="Pair distance medians",
    )
    for axis in axes:
        axis.legend(fontsize=6)
        axis.grid(alpha=0.2)
    fig.savefig(root / "pair_distance_distributions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
