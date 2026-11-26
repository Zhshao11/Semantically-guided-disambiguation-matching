"""Smoke, reset, fixed-budget KL-only semantic head, then one frozen A/B/C comparison."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sdgm.coarse import StructuralRetriever
from sdgm.continuation_diagnostics import state_digest
from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import (
    frozen_views,
    distillation_diagnostics,
    calibrate_lambda,
    evaluate_routing,
)
from sdgm.semantic_routing import SARPrototypeHead, distillation_loss
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.train_structure_smoke import prepare, geometry_for


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def step_head(model, head, batch, parent, optimizer, config):
    with torch.no_grad():
        out = model(batch["optical"], batch["sar"])
        geometry = geometry_for(model, batch, parent)
    optimizer.zero_grad(set_to_none=True)
    p = head(out["sar_descriptors"].detach())
    loss, _, _, _ = distillation_loss(
        batch["teacher"], p, model.source_pixel_xy, geometry
    )
    if not geometry["supervised"].any():
        raise ValueError("No valid supervision in minibatch")
    loss.backward()
    grads = [p.grad for p in head.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("Missing/nonfinite SAR head gradient")
    norm = torch.nn.utils.clip_grad_norm_(
        head.parameters(), config["gradient_clip"], error_if_nonfinite=True
    )
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("Frozen structure received gradient")
    optimizer.step()
    return dict(
        loss=float(loss.detach()),
        gradient_norm=float(norm),
        effective_queries=int(geometry["supervised"].sum()),
        nonfinite=False,
        head_all_parameter_gradients_finite=True,
        structure_has_no_gradient=True,
    )


def smoke(model, training, config, parent, device, output):
    spec = config["smoke"]
    if len(training) < spec["pairs"]:
        raise ValueError("Too few smoke pairs")
    # Multi-family first: one RNG-selected pair from each of32 RNG-selected families.
    groups = {}
    for i, r in enumerate(training.records):
        groups.setdefault(r["family_id"], []).append(i)
    rng = np.random.default_rng(spec["seed"])
    chosen = sorted(
        int(rng.choice(groups[f]))
        for f in rng.choice(sorted(groups), spec["pairs"], replace=False)
    )
    save_jsonl(output / "smoke_samples.jsonl", [training.records[i] for i in chosen])
    subset = Subset(training, chosen)
    seed_all(spec["seed"])
    head = SARPrototypeHead().to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loader = DataLoader(
        subset,
        batch_size=spec["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(spec["seed"]),
    )
    before = state_digest(model.state_dict())
    rows = []
    batches = iter(loader)
    # Fixed training transforms in the smoke; formal run reinitializes the head.
    training.view.set_epoch(1)
    for step in range(1, spec["steps"] + 1):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = prepare(batch, device)
        rows.append(
            dict(
                step=step,
                **step_head(model, head, batch, parent, optimizer, config["training"]),
            )
        )
    assert state_digest(model.state_dict()) == before
    save_csv(output / "smoke_training.csv", rows)
    # Deterministic fixed-subset mean KL before/after: restore seed for initial head.
    seed_all(spec["seed"])
    initial = SARPrototypeHead().to(device).eval()
    means = []
    with torch.no_grad():
        for candidate in [initial, head.eval()]:
            total = 0.0
            n = 0
            for batch in DataLoader(
                subset, batch_size=spec["batch_size"], shuffle=False
            ):
                batch = prepare(batch, device)
                out = model(batch["optical"], batch["sar"])
                g = geometry_for(model, batch, parent)
                _, kl, _, _ = distillation_loss(
                    batch["teacher"],
                    candidate(out["sar_descriptors"]),
                    model.source_pixel_xy,
                    g,
                )
                total += float(kl[g["supervised"]].sum())
                n += int(g["supervised"].sum())
            means.append(total / n)
    if not means[1] < means[0]:
        raise AssertionError("Smoke failed to lower fixed-subset KL")
    result = dict(
        passed=True,
        pairs=len(chosen),
        families=len({training.records[i]["family_id"] for i in chosen}),
        steps=spec["steps"],
        fixed_subset_initial_KL=means[0],
        fixed_subset_final_KL=means[1],
        structure_unchanged=True,
        gradients_finite=True,
        formal_head_reset=True,
        validation_used=False,
        scope="engineering check, not semantic effectiveness",
    )
    save_json(output / "smoke_summary.json", result)
    print(json.dumps(dict(stage="smoke_passed", **result)), flush=True)


def frozen_provenance(config):
    hashes = {
        p: sha256_file(p)
        for p in [
            "sdgm/coarse.py",
            "sdgm/geometry.py",
            "sdgm/raw_tiles.py",
            "sdgm/os_family_data.py",
            "sdgm/family_evaluation.py",
            config["structural_checkpoint"],
            str(Path(config["family_root"]) / "freeze_receipt.json"),
            str(Path(config["diagnostic_root"]) / "monitor_freeze.json"),
            str(Path(config["diagnostic_root"]) / "validation_transforms.jsonl"),
        ]
    }
    ubc = Path("reports/os_ubc_20261126/ubc")
    if not ubc.exists():
        ubc = Path("reports/os_ubc_20261126/ubcv2")
    for name in [
        "manifest.jsonl",
        "frozen_policy.json",
        "frozen_exclusions.csv",
        "frozen_evaluation_pool.csv",
        "freeze_receipt.json",
    ]:
        p = ubc / name
        if not p.exists():
            raise FileNotFoundError(p)
        hashes[str(p)] = sha256_file(p)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_routing.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    root = Path(config["output"])
    if json.loads((root / "preregistered_protocol.json").read_text()) != config:
        raise ValueError("Protocol changed after teacher build")
    if (root / "training").exists():
        raise FileExistsError("Do not replace a trained run; use a new version")
    verify_freeze(config["family_root"])
    if (
        sha256_file(config["structural_checkpoint"])
        != config["structural_checkpoint_sha256"]
    ):
        raise ValueError("Wrong structural endpoint")
    cache = verify_teacher_cache(root / "teacher", config)
    initial_provenance = frozen_provenance(config)
    save_json(root / "frozen_provenance_before.json", initial_provenance)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    checkpoint = torch.load(
        config["structural_checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 30
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    model_digest = state_digest(model.state_dict())
    del checkpoint
    train = OSFamilyDataset(config["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(config["family_root"], "val", parent, training=False)
    monitor, val_view = frozen_views(config, train, val)
    training = TeacherView(train, root / "teacher")
    monitors = [
        ("training_monitor", TeacherView(monitor, root / "teacher")),
        ("validation", TeacherView(val_view, root / "teacher")),
    ]
    output = root / "training"
    output.mkdir()
    smoke(model, training, config, parent, device, output)
    seed_all(config["training"]["seed"])
    head = SARPrototypeHead().to(device)
    initial_head_digest = state_digest(head.state_dict())
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    generator = torch.Generator().manual_seed(config["training"]["sampler_seed"])
    loader = DataLoader(
        training,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history = []
    epoch = 0
    iterator = None
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    stream = (output / "training.jsonl").open("w")
    for step in range(config["training"]["steps"] + 1):
        if step:
            head.train()
            if iterator is None:
                epoch += 1
                train.set_epoch(epoch)
                iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                train.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            row = dict(
                step=step,
                augmentation_epoch=epoch,
                learning_rate=config["training"]["learning_rate"],
                **step_head(
                    model,
                    head,
                    prepare(batch, device),
                    parent,
                    optimizer,
                    config["training"],
                ),
            )
            stream.write(json.dumps(row) + "\n")
            if step % 100 == 0:
                stream.flush()
                save_json(output / "progress.json", row)
                print(json.dumps(dict(stage="train", **row)), flush=True)
        if step in config["training"]["diagnostic_steps"]:
            for split, data in monitors:
                summary = distillation_diagnostics(
                    model,
                    head,
                    data,
                    parent,
                    device,
                    output,
                    f"step{step:05d}_{split}",
                    consistency=step in [0, config["training"]["steps"]],
                )
                history.append(dict(step=step, split=split, **summary))
                print(
                    json.dumps(
                        dict(
                            stage="fixed_distillation",
                            step=step,
                            split=split,
                            KL=summary["KL"],
                        )
                    ),
                    flush=True,
                )
    stream.close()
    assert state_digest(model.state_dict()) == model_digest
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    # One final head; no validation-best checkpoint or lambda search.
    torch.save(
        dict(
            head=head.state_dict(),
            optimizer=optimizer.state_dict(),
            step=config["training"]["steps"],
            initial_head_digest=initial_head_digest,
            structural_model_digest=model_digest,
            config=config,
            teacher_cache_version=cache["cache_version"],
            sampler_state=generator.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        ),
        output / "semantic_head.pt",
    )
    save_json(output / "distillation_history.json", history)
    calibration = calibrate_lambda(model, head, monitors[0][1], parent, config, device)
    save_json(root / "routing_calibration.json", calibration)
    results = {}
    for split, data in monitors:
        results[split] = evaluate_routing(
            model,
            head,
            data,
            parent,
            config,
            device,
            root / "evaluation",
            split,
            calibration["lambda_value"],
        )
    verify_freeze(config["family_root"])
    final_provenance = frozen_provenance(config)
    assert initial_provenance == final_provenance
    save_json(root / "frozen_provenance_after.json", final_provenance)
    final_model_digest = state_digest(model.state_dict())
    assert final_model_digest == model_digest
    save_json(
        root / "completion.json",
        dict(
            completed=True,
            fixed_budget_steps=config["training"]["steps"],
            smoke_reset=True,
            trained_only_SAR_head=True,
            KL_only=True,
            structure_digest_before=model_digest,
            structure_digest_after=final_model_digest,
            structure_parameters_frozen=True,
            teacher_frozen=True,
            prototypes_frozen=True,
            lambda_train_calibration=calibration["lambda_value"],
            A_B_exact=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
            source_hashes_unchanged=True,
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            head_checkpoint_sha256=sha256_file(output / "semantic_head.pt"),
            label_scope="relative to synthetic affine over nominal publisher-aligned base; prototype IDs are not named classes",
        ),
    )
    print(
        json.dumps(dict(stage="complete", validation=results["validation"])), flush=True
    )


if __name__ == "__main__":
    main()
