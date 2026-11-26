"""Describe frozen routing changes and constant-distribution diagnostics; no tuning."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.io import save_json, save_csv
from sdgm.os_family_data import OSFamilyDataset
from sdgm.semantic_evaluation import frozen_views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_routing.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    parent = json.loads(Path(config["parent_config"]).read_text())
    history = json.loads((root / "training/distillation_history.json").read_text())
    final = {
        row["split"]: row
        for row in history
        if row["step"] == config["training"]["steps"]
    }
    constant = np.array(final["training_monitor"]["teacher"]["mean_soft_occupancy"])
    result = {}
    rows = []
    examples = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for si, split in enumerate(["training_monitor", "validation"]):
        summary = json.loads(
            (root / "evaluation" / f"{split}_routing_summary.json").read_text()
        )
        q = np.load(root / "evaluation" / f"{split}_routing_queries.npz")
        keep = q["supervised"]
        a = q["A_nearest"] <= 8
        c = q["C_nearest"] <= 8
        changed_order = (q["A_indices"] != q["C_indices"]).any(-1) & keep
        changed_set = (
            np.sort(q["A_indices"], axis=-1) != np.sort(q["C_indices"], axis=-1)
        ).any(-1) & keep
        teacher = final[split]["teacher"]
        constant_kl = float(
            -np.dot(
                np.array(teacher["mean_soft_occupancy"]),
                np.log(np.maximum(constant, 1e-12)),
            )
            - teacher["mean_query_entropy_nats"]
        )
        result[split] = dict(
            image_pairs=len(q["sample_id"]),
            effective_queries=int(keep.sum()),
            changed_ordered_candidate_queries=int(changed_order.sum()),
            changed_candidate_set_queries=int(changed_set.sum()),
            order_only_changes=int((changed_order & ~changed_set).sum()),
            new_hits=int((~a & c & keep).sum()),
            lost_hits=int((a & ~c & keep).sum()),
            KL_final_head=final[split]["KL"],
            KL_constant_training_teacher_marginal=constant_kl,
            teacher_constant_definition="one frozen32-vector from fixed training-monitor teacher marginal, no validation fit; diagnostic only",
            teacher_within_image_best_constant_KL_median=teacher[
                "per_image_JS_to_image_mean_nats"
            ]["50"],
            image_constant_caveat="per-image teacher mean is an oracle diagnostic using teacher image content, not a deployable SAR predictor or an extra trained baseline",
            ordered_query_change_caveat="ordered candidate-array differences include swapping ranks3/4; set-change count is the appropriate replacement denominator",
        )
        assert result[split]["new_hits"] == summary["replacement"]["new_hits_C"]
        assert result[split]["lost_hits"] == summary["replacement"]["lost_hits_C"]
        for fi, family in enumerate(sorted(set(q["family_id"]))):
            ids = np.flatnonzero(q["family_id"] == family)
            k = keep[ids]
            n = int(k.sum())
            new = int((~a[ids] & c[ids] & k).sum())
            lost = int((a[ids] & ~c[ids] & k).sum())
            rows.append(
                dict(
                    split=split,
                    family_id=family,
                    effective_queries=n,
                    new_hits=new,
                    lost_hits=lost,
                    net_hits=new - lost,
                    delta_recall4_pp=100 * (new - lost) / n,
                    candidate_set_changed_queries=int(changed_set[ids].sum()),
                    order_only_queries=int(
                        (changed_order[ids] & ~changed_set[ids]).sum()
                    ),
                )
            )
        for state, mask in [("new_hit", ~a & c & keep), ("lost_hit", a & ~c & keep)]:
            found = np.argwhere(mask)
            for index, query in found[:2]:
                examples.append(
                    dict(
                        split=split,
                        state=state,
                        index=int(index),
                        query=int(query),
                        sample_id=str(q["sample_id"][index]),
                        family_id=str(q["family_id"][index]),
                    )
                )
        x = np.arange(2) + si * 0.25
        axes[0].bar(
            x,
            [result[split]["new_hits"], result[split]["lost_hits"]],
            width=0.25,
            label=split,
        )
        delta = 100 * (
            summary["methods"]["C"]["recall_at_4"]
            - summary["methods"]["A"]["recall_at_4"]
        )
        interval = 100 * np.array(
            summary["family_bootstrap"]["C_minus_A_recall4_interval95"]
        )
        axes[1].errorbar(
            si,
            delta,
            yerr=[[delta - interval[0]], [interval[1] - delta]],
            fmt="o",
            capsize=5,
        )
        axes[2].hist(
            [r["delta_recall4_pp"] for r in rows if r["split"] == split],
            bins=np.linspace(-1, 1, 21),
            alpha=0.55,
            label=split,
        )
    axes[0].set_xticks([0.125, 1.125], ["New hits", "Lost hits"])
    axes[0].set_ylabel("Source-query count")
    axes[0].set_ylim(0, 190)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_xticks([0, 1], ["Train monitor", "Validation"])
    axes[1].set_ylabel("C minus A recall@4 (percentage points)")
    axes[1].set_title("95% family-resampling intervals", fontsize=10)
    axes[2].axvline(0, color="gray", lw=1)
    axes[2].set_xlabel("Per-family C minus A recall@4 (pp)")
    axes[2].set_ylabel("Families")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "paired_routing_changes.png", dpi=180)
    plt.close(fig)
    save_json(root / "additional_diagnostics.json", result)
    save_csv(root / "paired_family_changes.csv", rows)
    # Illustrative examples are explicitly selected after evaluation; these do not
    # define the performance sample or replace the pre-existing failure batch.
    train = OSFamilyDataset(config["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(config["family_root"], "val", parent)
    views = dict(
        zip(["training_monitor", "validation"], frozen_views(config, train, val))
    )
    target = np.stack(
        np.meshgrid(np.arange(32) * 8 + 3.5, np.arange(32) * 8 + 3.5), -1
    ).reshape(-1, 2)
    source_axis = np.linspace(0, 31, 12).round().astype(int) * 8 + 3.5
    source = np.stack(np.meshgrid(source_axis, source_axis), -1).reshape(-1, 2)
    for ex in examples:
        if ex["split"] != "validation":
            continue
        item = views[ex["split"]][ex["index"]]
        query = ex["query"]
        xy = source[query]
        gt = (item["matrix"].numpy() @ np.array([*xy, 1]))[:2]
        q = np.load(root / "evaluation" / f"{ex['split']}_routing_queries.npz")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
        axes[0].imshow(item["optical"][0], cmap="gray", vmin=-1, vmax=1)
        axes[0].scatter(*xy, c="cyan", s=70, marker="+")
        axes[0].set_title("Optical source")
        for ax, method in zip(axes[1:], ["A", "C"]):
            ax.imshow(item["sar"][0], cmap="gray", vmin=-1, vmax=1)
            ax.scatter(*gt, c="lime", s=80, marker="+")
            for rank, idx in enumerate(q[f"{method}_indices"][ex["index"], query]):
                if idx < 0:
                    continue
                p = target[idx]
                ax.scatter(
                    *p,
                    facecolors="none",
                    edgecolors="red" if rank < 2 else "orange",
                    s=90,
                )
                ax.text(p[0] + 3, p[1] + 3, str(rank + 1), color="yellow", fontsize=8)
            dist = q[f"{method}_nearest"][ex["index"], query]
            ax.set_title(f"{method}: nearest {dist:.2f}px")
        for ax in axes:
            ax.set_xlim(-0.5, 255.5)
            ax.set_ylim(255.5, -0.5)
            ax.axis("off")
        fig.suptitle(
            f"{ex['state']} | {ex['sample_id']} | query{query} | synthetic-label-relative",
            fontsize=10,
        )
        fig.tight_layout()
        name = f"{ex['state']}_{ex['index']}_{query}.png"
        (root / "replacement_examples").mkdir(exist_ok=True)
        fig.savefig(root / "replacement_examples" / name, dpi=180)
        plt.close(fig)
        ex["file"] = "replacement_examples/" + name
    save_json(
        root / "replacement_example_selection.json",
        dict(
            selection="first2 queries in fixed sample/query order for new/lost state after final evaluation; descriptive examples, not representative failure sampling",
            examples=examples,
        ),
    )
    print(
        "Paired candidate-set changes and constant-distribution diagnostics exported",
        flush=True,
    )


if __name__ == "__main__":
    main()
