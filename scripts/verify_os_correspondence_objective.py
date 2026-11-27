"""Verify equal budgets, restored moments, unchanged sources and final reload."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdgm.coarse import StructuralRetriever
from sdgm.continuation_diagnostics import state_digest
from sdgm.correspondence_objective import continued_batch_schedule
from sdgm.correspondence_evaluation import evaluate_branches
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views
from sdgm.semantic_routing import SARPrototypeHead
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_correspondence_objective.json")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output"])
    semantic = json.loads(Path(cfg["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    locked = json.loads((root / "locked_protocol.json").read_text())
    assert all(locked[k] == v for k, v in cfg.items())
    assert locked["validation_results_seen_before_lock"] is False
    for file, key in [
        ("train_only_calibration.json", "calibration_sha256"),
        ("shared_batch_schedule.jsonl", "schedule_sha256"),
        ("smoke_samples.jsonl", "smoke_samples_sha256"),
    ]:
        assert sha256_file(root / file) == locked[key]
    before = json.loads((root / "frozen_provenance_before.json").read_text())
    assert before == json.loads((root / "frozen_provenance_after.json").read_text())
    for name, digest in before.items():
        assert sha256_file(name) == digest, name
    verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
    original = torch.load(
        cfg["resume_checkpoint"], map_location="cpu", weights_only=False
    )
    restore = json.loads((root / "branch_start_restore_audit.json").read_text())
    assert restore["KL_control"] == restore["KL_corr"]
    for value in restore.values():
        assert value["model"] == state_digest(original["head"]) and value[
            "optimizer"
        ] == state_digest(original["optimizer"])
    rows = [
        json.loads(l)
        for l in (root / "training/training.jsonl").read_text().splitlines()
    ]
    assert len(rows) == cfg["additional_steps"] == 5000
    for i, row in enumerate(rows):
        assert row["additional_step"] == i + 1 and row["total_step"] == 5001 + i
        assert all(not r["nonfinite"] for r in row["branches"].values())
        c = row["branches"]["KL_control"]
        e = row["branches"]["KL_corr"]
        assert c["objective"] == c["KL"]
        assert (
            abs(e["objective"] - (e["KL"] + locked["locked_alpha"] * e["corr"])) < 2e-6
        )
        for key in [
            "effective_KL_queries",
            "effective_corr_queries",
            "empty_P",
            "empty_N",
            "N_lt16",
            "existing_invalid_queries",
        ]:
            assert c[key] == e[key]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{cfg['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(
            semantic["structural_checkpoint"], map_location="cpu", weights_only=False
        )["model"]
    )
    model.requires_grad_(False).eval()
    heads = {}
    completion = json.loads((root / "completion.json").read_text())
    for name in cfg["branches"]:
        path = root / "training" / (name + "_last.pt")
        assert sha256_file(path) == completion["final_checkpoints"][name]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["total_step"] == 10000 and saved["additional_steps"] == 5000
        assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {10000}
        assert saved["structural_digest"] == state_digest(model.state_dict())
        h = SARPrototypeHead().to(device)
        h.load_state_dict(saved["head"])
        h.requires_grad_(False).eval()
        heads[name] = h
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent)
    schedule, audit = continued_batch_schedule(
        len(train),
        cfg["training_batch_size"],
        semantic["training"]["sampler_seed"],
        5000,
        5000,
        original["sampler_state"],
    )
    planned = [
        json.loads(l)
        for l in (root / "shared_batch_schedule.jsonl").read_text().splitlines()
    ]
    for expected, row in zip(schedule, planned):
        assert all(row[k] == v for k, v in expected.items())
        train.set_epoch(row["augmentation_epoch"])
        plans = [train.transform_record(i) for i in row["indices"]]
        assert row["sample_ids"] == [p["sample_id"] for p in plans]
        assert row["transform_seeds"] == [p["transform_seed"] for p in plans]
    torch.testing.assert_close(
        audit["final_sampler_state"],
        torch.load(root / "final_planned_sampler_state.pt", weights_only=True),
    )
    output = root / "verification"
    output.mkdir(exist_ok=True)
    checks = {}
    for split, view in zip(
        ["training_monitor", "validation"], frozen_views(semantic, train, val)
    ):
        result = evaluate_branches(
            model,
            heads,
            TeacherView(view, Path(semantic["output"]) / "teacher"),
            parent,
            cfg,
            locked,
            device,
            output,
            split,
            5000,
        )
        previous = json.loads(
            (root / "evaluation" / f"step05000_{split}_summary.json").read_text()
        )
        assert result == previous
        a = np.load(root / "evaluation" / f"step05000_{split}_queries.npz")
        b = np.load(output / f"step05000_{split}_queries.npz")
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        initial = np.load(root / "evaluation" / f"step00000_{split}_queries.npz")
        np.testing.assert_array_equal(
            initial["negative_indices"], a["negative_indices"]
        )
        np.testing.assert_array_equal(initial["positive"], a["positive"])
        checks[split] = dict(
            final_head_reload_all_arrays_exact=True,
            evaluation_PN_fixed_across_branches_and_checkpoints=True,
            supervised_queries=int(a["supervised"].sum()),
        )
    negatives = json.loads((root / "negative_distance_training.json").read_text())
    assert sum(negatives["counts"]) == negatives["valid_negative_slots"]
    save_json(
        output / "verification_summary.json",
        dict(
            passed=True,
            initial_heads_and_AdamW_exact=True,
            optimizer_reset=False,
            schedule_sampler_partial_epoch_and_all_augmentation_seeds_verified=True,
            additional_updates_each=5000,
            original_KL_unchanged=True,
            teacher_and_structure_unchanged=True,
            checks=checks,
            same_negative_sets_for_both_branches=True,
            validation_not_used_for_alpha_tau=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    print(
        "Restored optimizer, identical partial-epoch schedule, frozen sources and exact final reload verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
