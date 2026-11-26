"""Epoch20/25/30 and every-epoch matched diagnostics, without convergence claims."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.io import save_json, save_csv
from sdgm.raw_lineage import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_continuation_diagnostic.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    run = root / "run"
    history = json.loads((run / "evaluation_history.json").read_text())
    rows = []
    for point in history:
        for split in ["training_monitor", "validation"]:
            m = point[split]
            d = m["nearest_candidate_distance_px"]
            rows.append(
                dict(
                    epoch=point["epoch"],
                    split=split,
                    pairs=m["pairs"],
                    families=m["families"],
                    effective_source_points=m["supervised_queries"],
                    loss=m["loss"],
                    recall_at_1=m["recall_at_1"],
                    recall_at_4=m["recall_at_4"],
                    median_px=d["50"],
                    p95_px=d["95"],
                    p99_px=d["99"],
                    macro_pair_recall1=m["pair_macro_recall_at_1"],
                    macro_pair_recall4=m["pair_macro_recall_at_4"],
                    family_median_recall4=m["family_aggregation"][
                        "family_distributions"
                    ]["recall_at_4"]["50"],
                )
            )
    save_csv(root / "every_epoch_comparison.csv", rows)
    family_rows = []
    for point in history:
        for split in ["training_monitor", "validation"]:
            with (
                run / f"epoch{point['epoch']:02d}_{split}_per_family.csv"
            ).open() as stream:
                families = list(csv.DictReader(stream))
            family_rows.append(
                dict(
                    epoch=point["epoch"],
                    split=split,
                    families=len(families),
                    family_macro_recall1=float(
                        np.mean([float(r["recall_at_1"]) for r in families])
                    ),
                    family_macro_recall4=float(
                        np.mean([float(r["recall_at_4"]) for r in families])
                    ),
                    median_family_nearest_distance=float(
                        np.median([float(r["median_distance_px"]) for r in families])
                    ),
                    family_recall4_p05=float(
                        np.percentile([float(r["recall_at_4"]) for r in families], 5)
                    ),
                    family_recall4_p95=float(
                        np.percentile([float(r["recall_at_4"]) for r in families], 95)
                    ),
                )
            )
    save_csv(root / "family_epoch_comparison.csv", family_rows)
    save_csv(
        root / "epoch20_25_30_comparison.csv",
        [r for r in rows if r["epoch"] in config["comparison_epochs"]],
    )
    summaries = []
    baseline = history[0]
    for point in history:
        if point["epoch"] in config["comparison_epochs"]:
            summaries.append(
                dict(
                    epoch=point["epoch"],
                    training_monitor=point["training_monitor"],
                    validation=point["validation"],
                )
            )
    save_json(
        root / "comparison_summary.json",
        dict(
            comparison_epochs=summaries,
            monitor_freeze=json.loads((root / "monitor_freeze.json").read_text()),
            resume_audit=json.loads((root / "resume_audit.json").read_text()),
            convergence_guaranteed=False,
            final_vs_epoch20_validation_recall4_delta=history[-1]["validation"][
                "recall_at_4"
            ]
            - baseline["validation"]["recall_at_4"],
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for split, color in [("training_monitor", "#2465a6"), ("validation", "#c04d2b")]:
        data = [r for r in rows if r["split"] == split]
        x = [r["epoch"] for r in data]
        axes[0].plot(
            x,
            [r["recall_at_1"] for r in data],
            marker="o",
            color=color,
            label=split + " @1",
        )
        axes[0].plot(
            x,
            [r["recall_at_4"] for r in data],
            marker="s",
            linestyle="--",
            color=color,
            label=split + " @4",
        )
        axes[1].plot(
            x,
            [r["median_px"] for r in data],
            marker="o",
            color=color,
            label=split + " median",
        )
        axes[1].plot(
            x,
            [r["p95_px"] for r in data],
            linestyle="--",
            color=color,
            label=split + " P95",
        )
        axes[2].plot(x, [r["loss"] for r in data], marker="o", color=color, label=split)
    axes[0].set(
        ylabel="Recall at 8 native px", ylim=(0, 1), title="Matched fixed evaluation"
    )
    axes[1].set(ylabel="Native px to synthetic label", title="Median and tail")
    axes[2].set(
        ylabel="Neighborhood probability mass loss", title="Ten-epoch diagnostic window"
    )
    for axis in axes:
        axis.set_xlabel("Completed epoch")
        axis.legend(fontsize=6)
        axis.grid(alpha=0.2)
    fig.savefig(root / "matched_diagnostic_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for epoch, color in [(20, "#777777"), (25, "#2465a6"), (30, "#c04d2b")]:
        path = run / f"epoch{epoch:02d}_validation_distances.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            d = np.sort(data["nearest_distance_px"][data["supervised"]])
            axes[0].plot(
                d, np.arange(1, len(d) + 1) / len(d), label=f"epoch{epoch}", color=color
            )
            recall = ((data["nearest_distance_px"] <= 8) & data["supervised"]).sum(
                1
            ) / data["supervised"].sum(1)
            axes[1].hist(
                recall,
                bins=np.linspace(0, 1, 21),
                density=True,
                histtype="step",
                label=f"epoch{epoch}",
                color=color,
            )
    axes[0].axvline(8, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Native px to synthetic label", ylabel="CDF", xlim=(0, 100), ylim=(0, 1)
    )
    axes[1].set(xlabel="Validation pair recall@4", ylabel="Density")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.savefig(root / "validation_pair_progress.png", dpi=180)
    plt.close(fig)
    observations = []
    fixed = json.loads((root / "fixed_failure_queries.json").read_text())["cases"]
    ids = [r["case_id"] for r in fixed]
    for epoch in config["comparison_epochs"]:
        path = root / f"fixed_failures/epoch{epoch:02d}_query_observations.json"
        if not path.exists():
            continue
        obs = json.loads(path.read_text())
        assert [r["case_id"] for r in obs] == ids
        observations.extend(obs)
    flat = [
        {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()}
        for r in observations
    ]
    save_csv(root / "fixed_failure_tracking.csv", flat)
    progress = []
    for category in sorted({r["proposed_category"] for r in fixed}):
        for epoch in config["comparison_epochs"]:
            members = [
                r
                for r in observations
                if r["proposed_category"] == category and r["epoch"] == epoch
            ]
            if members:
                progress.append(
                    dict(
                        category=category,
                        epoch=epoch,
                        cases=len(members),
                        recall4_hits=sum(r["recall4_hit"] for r in members),
                        nearest_distance_median_px=float(
                            np.median([r["nearest_distance_px"] for r in members])
                        ),
                        nearest_distance_max_px=max(
                            r["nearest_distance_px"] for r in members
                        ),
                    )
                )
    save_csv(root / "fixed_failure_category_progress.csv", progress)
    print("Every-epoch pair/family comparisons and fixed-query tracking summarized")


if __name__ == "__main__":
    main()
