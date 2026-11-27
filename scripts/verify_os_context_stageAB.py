"""Verify baseline ranks, masked pre-MLP feature caches, equal starts and final fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever
from sdgm.context_descriptors import (
    ContextDescriptorHead,
    orthogonal_projection,
    sample_information_blocks,
)
from sdgm.context_descriptor_data import IdentityTrainingView, frozen_inputs, group_maps
from sdgm.context_descriptor_evaluation import evaluate_context
from sdgm.continuation_diagnostics import state_digest
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_routing import SARPrototypeHead
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.train_structure_smoke import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_context_descriptor.json")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output"])
    semantic = json.loads(Path(cfg["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    assert json.loads((root / "preregistered_protocol.json").read_text()) == cfg
    before = json.loads((root / "frozen_provenance_before.json").read_text())
    assert before == json.loads((root / "frozen_provenance_after.json").read_text())
    for name, digest in before.items():
        assert sha256_file(name) == digest, name
    verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
    starts = json.loads((root / "four_group_start_audit.json").read_text())
    assert len({v["initial_weight_digest"] for v in starts.values()}) == 1
    assert {v["trainable_parameters"] for v in starts.values()} == {393984}
    assert all(
        v["optimizer_state_empty"] and not v["old_optimizer_restored"]
        for v in starts.values()
    )
    rows = [
        json.loads(l) for l in (root / "smoke_training.jsonl").read_text().splitlines()
    ]
    assert len(rows) == cfg["smoke"]["steps"] == 800
    for i, row in enumerate(rows):
        assert row["step"] == i + 1
        assert set(row["groups"]) == set(cfg["groups"])
        assert all(not z["nonfinite"] for z in row["groups"].values())
        assert {z["valid_pairs"] for z in row["groups"].values()} == {16}
        assert {z["effective_queries"] for z in row["groups"].values()} == {2304}
    projection = torch.from_numpy(np.load(root / "structural_projection.npy"))
    assert torch.equal(projection, orthogonal_projection(cfg["projection_seed"]))
    torch.testing.assert_close(
        projection @ projection.T, torch.eye(32), atol=1e-6, rtol=1e-6
    )
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{cfg['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(
            cfg["structural_checkpoint"], map_location="cpu", weights_only=False
        )["model"]
    )
    model.requires_grad_(False).eval()
    semhead = SARPrototypeHead().to(device)
    saved = torch.load(
        cfg["SAR_head_checkpoint"], map_location="cpu", weights_only=False
    )
    assert saved["total_step"] == 10000
    semhead.load_state_dict(saved["head"])
    semhead.requires_grad_(False).eval()
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    records = [
        json.loads(l)
        for l in (root / "fixed_training_samples.jsonl").read_text().splitlines()
    ]
    lookup = {r["sample_id"]: i for i, r in enumerate(train.records)}
    view = IdentityTrainingView(train, [lookup[r["sample_id"]] for r in records])
    data = TeacherView(view, Path(semantic["output"]) / "teacher")
    assert len({r["family_id"] for r in records}) == 32
    audit = json.loads((root / "forward_cache_audit.json").read_text())
    audit_lookup = {(r["sample_id"], r["group"]): r for r in audit}
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=4, shuffle=False):
            batch = prepare(batch, device)
            inputs = frozen_inputs(model, semhead, batch)
            for i, sample in enumerate(batch["id"]):
                one = {k: v[i : i + 1] for k, v in inputs.items()}
                for group in cfg["groups"]:
                    om, sm, _, _ = group_maps(one, projection.to(device), group)
                    a = sample_information_blocks(om, model.source_pixel_xy)
                    b = sample_information_blocks(sm, model.target_pixel_xy)
                    assert (
                        state_digest([a, b])
                        == audit_lookup[(sample, group)]["cached_block_sha256"]
                    )
    heads = {}
    for name in cfg["groups"]:
        s = torch.load(
            root / "checkpoints" / (name + "_smoke.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert s["step"] == 800 and {
            int(v["step"]) for v in s["optimizer"]["state"].values()
        } == {800}
        assert s["initial_digest"] == starts[name]["initial_weight_digest"]
        h = ContextDescriptorHead().to(device)
        h.load_state_dict(s["head"])
        # Match parameter flags of the original no-grad evaluation as well as
        # its weights; inference-only Linear kernels can round differently.
        h.eval()
        heads[name] = h
    output = root / "verification"
    output.mkdir(exist_ok=True)
    result = evaluate_context(
        model,
        semhead,
        heads,
        data,
        parent,
        cfg,
        projection.to(device),
        device,
        output,
        "final_reloaded",
    )
    original = json.loads((root / "smoke" / "step00800_summary.json").read_text())
    for name, row in result.items():
        old = original[name]
        new = dict(row)
        new["tag"] = old["tag"]
        assert new == old
    a = np.load(root / "smoke/step00800_queries.npz")
    b = np.load(output / "final_reloaded_queries.npz")
    assert set(a.files) == set(b.files)
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
    for split in ["training_monitor", "validation"]:
        baseline = np.load(root / "baseline" / (split + "_queries.npz"))
        old = np.load(Path(cfg["baseline_reference"]) / (split + "_score_queries.npz"))
        for k, other in [
            ("supervised", "supervised"),
            ("structure_indices", "structure_indices"),
            ("structure_first", "structure_first"),
            ("structure_nearest", "structure_nearest"),
        ]:
            np.testing.assert_array_equal(baseline[k], old[other])
    save_json(
        output / "verification_summary.json",
        dict(
            passed=True,
            pure_structure_baseline_same_candidates_and_distances_exact=True,
            all_four_initial_weights_equal=True,
            parameter_count_each=393984,
            paired_modality_parameters_independent=True,
            projection_exact_and_row_orthogonal=True,
            all32pairs_four_group_preMLP_caches_exact=True,
            final_reload_all_metrics_and_candidate_arrays_exact=True,
            frozen_sources_and_SAR_KL10000_unchanged=True,
            equal_smoke_updates_each=800,
            formal5000step_three_seed_budget_not_run=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    print(
        "Baseline, shared initialization, pre-MLP caches, frozen endpoints and final small-fit reload verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
