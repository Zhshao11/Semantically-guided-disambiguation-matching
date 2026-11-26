"""Export latent-prototype spatial maps, paired changes and diagnostic figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever
from sdgm.io import save_csv, save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.semantic_evaluation import frozen_views
from sdgm.semantic_routing import SARPrototypeHead, distillation_loss
from sdgm.semantic_teacher import TeacherView
from scripts.train_structure_smoke import prepare, geometry_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_routing.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    parent = json.loads(Path(config["parent_config"]).read_text())
    history = json.loads((root / "training/distillation_history.json").read_text())
    metrics = []
    for row in history:
        p = row["SAR"]
        metrics.append(
            dict(
                step=row["step"],
                split=row["split"],
                KL=row["KL"],
                SAR_entropy=p["mean_query_entropy_nats"],
                SAR_active=p["hard_active_prototypes"],
                SAR_active_above_1pct=p["hard_active_above_1pct"],
                SAR_max_occupancy=p["largest_hard_fraction"],
                SAR_spatial_rms_median=p["per_image_spatial_probability_rms"]["50"],
                SAR_within_image_JS_median=p["per_image_JS_to_image_mean_nats"]["50"],
                SAR_marginal_minus_conditional_entropy=p[
                    "marginal_minus_conditional_entropy_nats"
                ],
                augmentation_JS=row["augmentation_consistency_JS_nats"],
            )
        )
    save_csv(root / "distillation_comparison.csv", metrics)
    tables = {
        s: json.loads((root / "evaluation" / f"{s}_routing_summary.json").read_text())
        for s in ["training_monitor", "validation"]
    }
    rows = []
    for split, table in tables.items():
        for method, m in table["methods"].items():
            rows.append(
                dict(
                    split=split,
                    method=method,
                    recall_at_1=m["recall_at_1"],
                    recall_at_4=m["recall_at_4"],
                    median_distance_px=m["nearest_candidate_distance_px"]["50"],
                    p95_distance_px=m["nearest_candidate_distance_px"]["95"],
                    effective_queries=table["effective_queries"],
                )
            )
    save_csv(root / "ABC_comparison.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for split, color in [
        ("training_monitor", "tab:blue"),
        ("validation", "tab:orange"),
    ]:
        selected = [r for r in metrics if r["split"] == split]
        axes[0].plot(
            [r["step"] for r in selected],
            [r["KL"] for r in selected],
            "o-",
            label=split,
            color=color,
        )
        axes[1].plot(
            [r["step"] for r in selected],
            [r["SAR_entropy"] for r in selected],
            "o-",
            label=split,
            color=color,
        )
        axes[2].plot(
            [r["step"] for r in selected],
            [r["SAR_spatial_rms_median"] for r in selected],
            "o-",
            label=split,
            color=color,
        )
    for ax, title in zip(
        axes,
        [
            "Fixed-view KL (teacher || SAR)",
            "SAR mean entropy (nats)",
            "Median within-image spatial RMS",
        ],
    ):
        ax.set_title(title)
        ax.set_xlabel("Head updates")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(root / "distillation_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, split in enumerate(["training_monitor", "validation"]):
        final = next(
            r
            for r in history
            if r["split"] == split and r["step"] == config["training"]["steps"]
        )
        for col, kind in enumerate(["teacher", "SAR"]):
            x = np.arange(32)
            data = final[kind]
            axes[row, col].bar(
                x - 0.2, data["mean_soft_occupancy"], 0.4, label="soft mass"
            )
            axes[row, col].bar(
                x + 0.2, data["hard_occupancy"], 0.4, label="hard argmax"
            )
            axes[row, col].set_title(f"{split}: {kind} latent occupancy")
            axes[row, col].set_xlabel("Prototype ID (no class names)")
            axes[row, col].set_ylabel("Fraction")
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "prototype_occupancy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, color in [
        ("training_monitor", "tab:blue"),
        ("validation", "tab:orange"),
    ]:
        table = tables[split]
        m = table["methods"]
        axes[0].plot(
            ["A", "B: lambda0", "C", "C shuffled"],
            [100 * m[k]["recall_at_4"] for k in ["A", "B", "C", "C_shuffle"]],
            "o-",
            label=split,
            color=color,
        )
        q = np.load(root / "evaluation" / f"{split}_routing_queries.npz")
        for method, style in [("A", "--"), ("C", "-")]:
            d = np.sort(q[f"{method}_nearest"][q["supervised"]])
            axes[1].plot(
                d,
                np.arange(1, len(d) + 1) / len(d),
                style,
                color=color,
                label=f"{split} {method}",
            )
    axes[0].set_ylabel("Recall@4 (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_xlim(0, 60)
    axes[1].axvline(8, color="gray", lw=1)
    axes[1].set_xlabel("Nearest candidate distance (native px)")
    axes[1].set_ylabel("CDF")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "routing_comparison.png", dpi=180)
    plt.close(fig)
    # Samples fixed by source ordering and previous failure list, not final gains.
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(
            config["structural_checkpoint"], map_location="cpu", weights_only=False
        )["model"]
    )
    model.requires_grad_(False).eval()
    head = SARPrototypeHead().to(device)
    head.load_state_dict(
        torch.load(
            root / "training/semantic_head.pt", map_location="cpu", weights_only=False
        )["head"]
    )
    head.eval()
    train = OSFamilyDataset(config["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(config["family_root"], "val", parent)
    monitor, val_view = frozen_views(config, train, val)
    plan = []
    for split, view in [("training_monitor", monitor), ("validation", val_view)]:
        data = TeacherView(view, root / "teacher")
        indices = [0, len(data) // 2]
        if split == "validation":
            lookup = {r["sample_id"]: i for i, r in enumerate(data.records)}
            for suffix in ["796", "81"]:
                indices.extend(
                    i
                    for sample_id, i in lookup.items()
                    if sample_id.endswith("/" + suffix)
                )
        for index in sorted(set(indices)):
            item = data[index]
            batch = prepare(
                {
                    k: v[None] if isinstance(v, torch.Tensor) else [v]
                    for k, v in item.items()
                },
                device,
            )
            with torch.no_grad():
                out = model(batch["optical"], batch["sar"])
                p = head(out["sar_descriptors"])
                g = geometry_for(model, batch, parent)
                _, kl, warped, t = distillation_loss(
                    batch["teacher"], p, model.source_pixel_xy, g
                )
            target_valid = g["target_valid"][0].reshape(32, 32).cpu().numpy()
            sar_ids = p[0].argmax(0).float().cpu().numpy()
            sar_ids[~target_valid] = np.nan
            sar_entropy = -(p[0] * p[0].clamp_min(1e-8).log()).sum(0).cpu().numpy()
            sar_entropy[~target_valid] = np.nan
            fig, ax = plt.subplots(2, 4, figsize=(13, 6.7))
            for a, img, title, cmap, vmin, vmax in [
                (
                    ax[0, 0],
                    batch["optical"][0, 0].cpu(),
                    "Native optical (gray replicated to DINO)",
                    "gray",
                    -1,
                    1,
                ),
                (ax[1, 0], batch["sar"][0, 0].cpu(), "Augmented SAR", "gray", -1, 1),
                (
                    ax[0, 1],
                    batch["teacher"][0].argmax(0).cpu(),
                    "Optical prototype argmax",
                    "turbo",
                    0,
                    31,
                ),
                (
                    ax[1, 1],
                    sar_ids,
                    "SAR prototype argmax (valid support)",
                    "turbo",
                    0,
                    31,
                ),
                (
                    ax[0, 2],
                    -(batch["teacher"][0] * batch["teacher"][0].clamp_min(1e-8).log())
                    .sum(0)
                    .cpu(),
                    "Teacher entropy (nats)",
                    "viridis",
                    0,
                    np.log(32),
                ),
                (
                    ax[1, 2],
                    sar_entropy,
                    "SAR entropy (valid support)",
                    "viridis",
                    0,
                    np.log(32),
                ),
            ]:
                im = a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                a.set_title(title, fontsize=9)
                a.axis("off")
                if cmap != "gray":
                    fig.colorbar(im, ax=a, fraction=0.045)
            k = g["supervised"][0].cpu().numpy()
            qmap = np.full(144, np.nan)
            qmap[k] = warped[0, :, g["supervised"][0]].argmax(0).cpu().numpy()
            im = ax[0, 3].imshow(qmap.reshape(12, 12), cmap="turbo", vmin=0, vmax=31)
            ax[0, 3].set_title("SAR sampled at GT -> source grid", fontsize=9)
            ax[0, 3].axis("off")
            fig.colorbar(im, ax=ax[0, 3], fraction=0.045)
            errors = np.full(144, np.nan)
            errors[k] = kl[0, g["supervised"][0]].cpu().numpy()
            im = ax[1, 3].imshow(errors.reshape(12, 12), cmap="magma", vmin=0, vmax=4)
            ax[1, 3].set_title("KL at existing supervised queries", fontsize=9)
            ax[1, 3].axis("off")
            fig.colorbar(im, ax=ax[1, 3], fraction=0.045)
            fig.suptitle(
                f"{split} | {item['id']} | latent IDs, not land-cover classes",
                fontsize=11,
            )
            fig.tight_layout()
            name = f"{split}_{item['id'].split('/')[-1]}_semantics.png"
            (root / "spatial_maps").mkdir(exist_ok=True)
            fig.savefig(root / "spatial_maps" / name, dpi=170)
            plt.close(fig)
            plan.append(
                dict(
                    split=split,
                    sample_id=item["id"],
                    family_id=item["family_id"],
                    file="spatial_maps/" + name,
                    selection="fixed first/middle sample plus previous frozen failure81/796; not selected by semantic gains",
                )
            )
    save_json(root / "spatial_map_samples.json", plan)
    print(
        "Distillation, occupancy, routing figures and fixed spatial maps exported",
        flush=True,
    )


if __name__ == "__main__":
    main()
