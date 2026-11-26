"""Summarize frozen native inventories and the OS-only engineering fit."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdgm.io import save_json
from sdgm.raw_lineage import sha256_file


def main():
    root = Path("reports/os_ubc_20261126")
    os_rows = [
        json.loads(line)
        for line in (root / "os/manifest.jsonl").read_text().splitlines()
    ]
    by_hash = defaultdict(list)
    for row in os_rows:
        by_hash[row.get("joint_pixel_sha256", row["id"])].append(row)
    duplicates = [group for group in by_hash.values() if len(group) > 1]
    save_json(
        root / "os/exact_duplicates.json",
        dict(
            exact_joint_pixel_duplicate_groups=len(duplicates),
            redundant_pairs=sum(len(group) - 1 for group in duplicates),
            cross_official_split_duplicate_groups=sum(
                len({row["official_split"] for row in group}) > 1
                for group in duplicates
            ),
            subset_unique_pairs={
                subset: len(
                    {
                        row["joint_pixel_sha256"]
                        for row in os_rows
                        if row["subset"] == subset
                    }
                )
                for subset in sorted({row["subset"] for row in os_rows})
            },
            duplicate_group_examples=[
                [row["id"] for row in group] for group in duplicates[:30]
            ],
            meaning="Repeated pixel content is established; absence of detected duplicates never establishes geographic independence",
        ),
    )
    ubc = [
        json.loads(line)
        for line in (root / "ubcv2/manifest.jsonl").read_text().splitlines()
    ]
    save_json(
        root / "ubcv2/geographic_metadata_summary.json",
        dict(
            pairs=len(ubc),
            same_native_optical_sar_grid=sum(
                row.get("same_geographic_grid", False) for row in ubc
            ),
            crs_counts=dict(Counter(row["optical"]["crs"] for row in ubc)),
            bounds_available=sum(row["optical"]["bounds"] is not None for row in ubc),
            no_grid_equality_precision_claim=True,
            parent_acquisition_scene_ids_recovered=False,
            raw_SAR_units="physical amplitude/intensity interpretation not established; retain float32 values, not the older local generated uint8 conversion",
        ),
    )
    scene = json.loads((root / "os/scene_metadata.json").read_text())
    save_json(
        root / "os/scene_file_identities.json",
        {
            mod: dict(
                path=record["path"],
                bytes=record["bytes"],
                sha256=sha256_file(record["path"]),
            )
            for mod, record in scene["images"].items()
        },
    )
    fit = Path("reports/structure_os_native_20261126")
    initial = json.loads((fit / "initial_metrics.json").read_text())
    final = json.loads((fit / "final_metrics.json").read_text())
    fixed = json.loads((fit / "fixed_training_subset.json").read_text())
    save_json(
        root / "decisions.json",
        dict(
            OS=dict(
                status="usable_for_publisher_aligned_base_plus_synthetic_affine_coarse_training_with_explicit_assumptions",
                current_experiment="engineering_only",
                independent_geographic_validation="waiting_for_scene_information_or_explicit_protocol",
                fine_precision_certified=False,
                per_patch_native_registration_gt=False,
                full_scene_GCPs=72,
                full_scene_GCPs_used_for_patch_training=False,
            ),
            UBCv2=dict(
                status="raw_images_and_frozen_file_pool_ready_for_future_cross_domain_protocol; no_current_model_evaluation",
                prior_analysis_exposure=True,
                blind_test=False,
                hyperparameter_selection_allowed=False,
                fine_precision_certified=False,
                building_annotations_are_registration_gt=False,
                exclusion_policy="complete/decodable/finite/nonempty/support-compatible; no cloud exclusion this round",
                future_preprocessing_and_synthetic_or_point_evaluation_protocol="must be fixed before running the frozen model; do not tune on UBC results",
            ),
            WHU=dict(reconstruction_paused=True, existing_engineering_preserved=True),
            other_external_candidates=["SRIF", "SpaceNet9"],
            native_OS_smoke=dict(
                training_pairs=32,
                unique_512_crop_parents=len(
                    {
                        p["parent_id"]
                        for row in fixed["samples"]
                        for p in row["raw_window_mapping"]
                    }
                ),
                initial=initial,
                final=final,
                independent_validation=False,
                UBC_evaluated=False,
            ),
        ),
    )
    history = [
        json.loads(line) for line in (fit / "training.jsonl").read_text().splitlines()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    loss = np.array([row["loss"] for row in history])
    steps = np.array([row["step"] for row in history])
    axes[0].plot(
        steps, loss, color="silver", alpha=0.7, linewidth=0.7, label="minibatch"
    )
    axes[0].plot(
        steps[24:],
        np.convolve(loss, np.ones(25) / 25, mode="valid"),
        label="25-step mean",
    )
    axes[0].set(
        xlabel="Training step",
        ylabel="Neighborhood mass loss",
        title="32 fixed native 256 pairs",
    )
    axes[0].legend()
    for tag in ["initial", "final"]:
        distance = np.sort(
            json.loads((fit / f"{tag}_nearest_distances.json").read_text())
        )
        axes[1].plot(
            distance, np.arange(1, len(distance) + 1) / len(distance), label=tag
        )
    axes[1].axvline(
        8, color="black", linestyle="--", linewidth=0.8, label="8 px recall tolerance"
    )
    axes[1].set(
        xlabel="Nearest native candidate to synthetic label (px)",
        ylabel="Empirical CDF",
        xlim=(0, 160),
        ylim=(0, 1),
        title="Training-set diagnostic only",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(fit / "loss_distance.png", dpi=180)
    plt.close(fig)
    save_json(
        root / "delivery_identities.json",
        dict(
            files={
                str(path.relative_to(root)): sha256_file(path)
                for path in [
                    root / "os/manifest.jsonl",
                    root / "ubcv2/manifest.jsonl",
                    root / "ubcv2/frozen_policy.json",
                    root / "ubcv2/frozen_exclusions.csv",
                    root / "ubcv2/frozen_evaluation_pool.csv",
                ]
            },
            model_data_scope="OS official train only; no validation or UBC model run",
            resized=False,
        ),
    )


if __name__ == "__main__":
    main()
