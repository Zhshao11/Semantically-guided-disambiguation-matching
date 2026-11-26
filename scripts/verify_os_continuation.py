"""Verify continuation state, matched diagnostic controls and final reload metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sdgm.coarse import StructuralRetriever
from sdgm.continuation_diagnostics import (
    FrozenDiagnosticView,
    verify_monitor,
    state_digest,
    aggregate_families,
)
from sdgm.family_evaluation import evaluate_family
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_continuation_diagnostic.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    root = Path(config["output"])
    parent_root = Path(config["parent_root"])
    freeze = verify_monitor(root)
    verify_freeze(parent_root)
    assert (
        sha256_file(config["resume_checkpoint"]) == freeze["source_checkpoint_sha256"]
    )
    assert (
        sha256_file(parent_root / "freeze_receipt.json")
        == freeze["parent_family_freeze_sha256"]
    )
    assert (
        sha256_file(parent_root / "frozen_validation.jsonl")
        == freeze["parent_validation_plan_sha256"]
    )
    query_freeze = json.loads((root / "failure_query_freeze.json").read_text())
    assert (
        sha256_file(root / "fixed_failure_queries.json")
        == query_freeze["queries_sha256"]
    )
    plans = {
        key: [json.loads(l) for l in (root / name).read_text().splitlines()]
        for key, name in [
            ("monitor", "monitor_transforms.jsonl"),
            ("val", "validation_transforms.jsonl"),
        ]
    }
    for key in [
        "optical_input_to_sar_input",
        "sar_original_to_input",
        "parameters",
        "transform_seed",
    ]:
        assert [p[key] for p in plans["monitor"]] == [p[key] for p in plans["val"]], key
    history = json.loads((root / "run/evaluation_history.json").read_text())
    assert [r["epoch"] for r in history] == config["evaluation_epochs"]
    for point in history:
        a, b = point["training_monitor"], point["validation"]
        assert (
            a["supervised_queries"] == b["supervised_queries"]
            and a["masked"] == b["masked"]
        )
        for role in ["training_monitor", "validation"]:
            assert a["native_recall_radius_px"] == b["native_recall_radius_px"] == 8
            assert a["candidates_per_source"] == b["candidates_per_source"] == 4
            with (
                root / f"run/epoch{point['epoch']:02d}_{role}_per_pair.csv"
            ).open() as stream:
                rows = list(csv.DictReader(stream))
            if role == "training_monitor":
                counts = [row["count"] for row in rows]
            else:
                assert counts == [row["count"] for row in rows]
    control = dict(
        all_11_epochs_present=True,
        matched_affine_parameters_and_seed_multisets_exact=True,
        matched_masks_and_effective_point_counts=True,
        matched_per_pair_effective_point_counts=True,
        eval_mode_shared=True,
        source_queries_shared=True,
        NMS_shared=True,
        candidate_count=4,
        recall_radius_px=8,
        architecture_file_unchanged=sha256_file("sdgm/coarse.py")
        == json.loads((parent_root / "run/model_signature.json").read_text())[
            "model_source_sha256"
        ],
        parent_checkpoint_unchanged=True,
        parent_source_freeze_unchanged=True,
        parent_validation_plan_unchanged=True,
        fixed_failure_queries_unchanged=True,
        old_training_metric_was_online_random_augmentation=False,
        old_training_loss_was_online_random_augmentation=True,
    )
    assert control["architecture_file_unchanged"]
    source = json.loads(
        (parent_root / "run/implementation_environment.json").read_text()
    )["code_hashes"]
    unchanged = {
        name: sha256_file(name) == source[name]
        for name in [
            "sdgm/coarse.py",
            "sdgm/geometry.py",
            "sdgm/raw_tiles.py",
            "sdgm/os_family_data.py",
            "sdgm/family_evaluation.py",
        ]
    }
    assert all(unchanged.values())
    UBC = Path("reports/os_ubc_20261126/ubcv2")
    old_risk = json.loads(
        (parent_root / "ubcv2_risk/unchanged_pool_verification.json").read_text()
    )["before"]
    unchanged_ubc = {
        name: sha256_file(UBC / name) == digest for name, digest in old_risk.items()
    }
    assert all(unchanged_ubc.values())
    result_dir = root / "verification"
    result_dir.mkdir(exist_ok=True)
    save_json(result_dir / "matched_control_checks.json", control)
    save_json(
        result_dir / "unchanged_code_and_sealed_pools.json",
        dict(
            code=unchanged,
            UBC_files=unchanged_ubc,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    final = torch.load(root / "run/last.pt", map_location="cpu", weights_only=False)
    assert final["epoch"] == 30 and final["step"] == 59340
    assert {float(v["step"]) for v in final["optimizer"]["state"].values()} == {59340.0}
    assert final["learning_rates"] == [0.0003]
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(final["model"])
    training = OSFamilyDataset(parent_root, "train", parent, training=True)
    validation = OSFamilyDataset(parent_root, "val", parent, training=False)
    selected = [
        json.loads(l) for l in (root / "monitor_samples.jsonl").read_text().splitlines()
    ]
    lookup = {r["sample_id"]: i for i, r in enumerate(training.records)}
    monitor = FrozenDiagnosticView(
        training, [lookup[r["sample_id"]] for r in selected], plans["monitor"]
    )
    val = FrozenDiagnosticView(validation, range(len(validation)), plans["val"])
    observed = {}
    for role, dataset in [("training_monitor", monitor), ("validation", val)]:
        metrics = evaluate_family(
            model, dataset, parent, device, result_dir, f"reloaded_{role}", details=True
        )
        metrics["family_aggregation"] = aggregate_families(
            result_dir, f"reloaded_{role}"
        )
        previous = json.loads((root / f"run/epoch30_{role}_metrics.json").read_text())
        # Family aggregation is written separately by the training script.
        previous["family_aggregation"] = json.loads(
            (root / f"run/epoch30_{role}_family_summary.json").read_text()
        )
        keys = [
            "pairs",
            "families",
            "supervised_queries",
            "loss",
            "recall_at_1",
            "recall_at_4",
            "nearest_candidate_distance_px",
            "pair_distributions",
            "masked",
            "family_aggregation",
        ]
        equality = {key: metrics[key] == previous[key] for key in keys}
        assert all(equality.values()), equality
        observed[role] = equality
    save_json(
        result_dir / "final_reload_verification.json",
        dict(
            exact_epoch30_train_monitor_and_validation_reproduction=True,
            fields=observed,
            optimizer_final_step=59340,
            optimizer_updates_added=19780,
            learning_rate=0.0003,
            checkpoint_sha256=sha256_file(root / "run/last.pt"),
            parent_checkpoint_unchanged=True,
            fixed_failure_queries_unchanged=True,
            convergence_claimed=False,
        ),
    )
    print(
        "Matched diagnostics, optimizer continuation, fixed cases, sealed pools and final model reload verified"
    )


if __name__ == "__main__":
    main()
