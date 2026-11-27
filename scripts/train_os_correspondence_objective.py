"""Train-only calibration/smoke then paired equal-budget objective continuation."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sdgm.coarse import StructuralRetriever
from sdgm.continuation_diagnostics import state_digest
from sdgm.correspondence_objective import (
    fixed_competition,
    correspondence_loss,
    loss_gradients,
    continued_batch_schedule,
    restored_head_and_optimizer,
)
from sdgm.io import save_json, save_jsonl, save_csv
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_routing import SARPrototypeHead, distillation_loss
from sdgm.semantic_score_control import probability_scores
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from sdgm.semantic_evaluation import frozen_views
from scripts.train_structure_smoke import prepare, geometry_for
from scripts.train_os_semantic_routing import frozen_provenance


def loss_bundle(model, head, batch, parent, tau, features=None):
    with torch.no_grad():
        if features is None:
            features = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        comp = fixed_competition(features["scores"], geom, model.target_pixel_xy)
    p = head(features["sar_descriptors"].detach())
    kl, per_kl, _, teacher = distillation_loss(
        batch["teacher"], p, model.source_pixel_xy, geom
    )
    scores = probability_scores(teacher.detach(), p.flatten(2))["dist"]
    corr, status = correspondence_loss(scores, comp, tau)
    return kl, corr, geom, comp, status, scores, features


@torch.no_grad()
def assess_smoke(model, head, subset, parent, device, tau):
    total_kl = 0.0
    nkl = 0
    total_corr = 0.0
    ncorr = 0
    wins = 0
    deltas = []
    for batch in DataLoader(subset, batch_size=16, shuffle=False):
        batch = prepare(batch, device)
        kl, corr, g, c, status, scores, _ = loss_bundle(model, head, batch, parent, tau)
        nk = int(g["supervised"].sum())
        nc = int(status["eligible"].sum())
        nkl += nk
        ncorr += nc
        total_kl += float(kl) * nk
        total_corr += float(corr) * nc
        from sdgm.semantic_bottleneck import comparison_stats

        stats = comparison_stats(scores, c["positive"], c["negative"])
        valid = status["eligible"]
        wins += int((stats["win"] & valid).sum())
        deltas.extend(stats["delta"][valid].cpu().tolist())
    return dict(
        KL=total_kl / max(nkl, 1),
        corr=total_corr / max(ncorr, 1),
        eligible_queries=ncorr,
        PN_win_rate=wins / max(ncorr, 1),
        PN_delta_median=float(np.median(deltas)),
        empty_corr=ncorr == 0,
    )


def make_smoke_subset(train, cfg, root):
    groups = {}
    for i, r in enumerate(train.records):
        groups.setdefault(r["family_id"], []).append(i)
    rng = np.random.default_rng(cfg["smoke"]["seed"])
    indices = sorted(
        int(rng.choice(groups[f]))
        for f in rng.choice(sorted(groups), cfg["smoke"]["families"], replace=False)
    )
    save_jsonl(root / "smoke_samples.jsonl", [train.records[i] for i in indices])
    return indices


def prepare_protocol(cfg, semantic, parent, model, train, teacher_train, saved, device):
    root = Path(cfg["output"])
    root.mkdir(parents=True)
    save_json(root / "proposed_protocol.json", cfg)
    indices = make_smoke_subset(train, cfg, root)
    train.set_epoch(cfg["smoke"]["augmentation_epoch"])
    subset = Subset(teacher_train, indices)
    head, opt = restored_head_and_optimizer(saved, SARPrototypeHead, device)
    assert state_digest(opt.state_dict()) == state_digest(saved["optimizer"])
    stdevs = []
    # Evaluate original restored head, training data only. No validation object
    # or scores exist in this preparation phase.
    with torch.no_grad():
        for batch in DataLoader(
            subset, batch_size=cfg["smoke"]["batch_size"], shuffle=False
        ):
            batch = prepare(batch, device)
            _, _, g, c, status, scores, _ = loss_bundle(model, head, batch, parent, 1.0)
            keep = c["positive"] | c["negative"]
            counts = keep.sum(-1).clamp_min(1)
            means = (scores * keep).sum(-1) / counts
            variance = ((scores - means[..., None]).square() * keep).sum(-1) / counts
            stdevs.extend(variance.sqrt()[status["eligible"]].cpu().tolist())
    if not stdevs:
        raise RuntimeError("No train calibration queries")
    raw_tau = float(np.median(stdevs))
    tau = float(np.clip(raw_tau, *cfg["smoke"]["tau_bounds"]))
    gradients = []
    for batch in DataLoader(
        subset, batch_size=cfg["smoke"]["batch_size"], shuffle=False
    ):
        batch = prepare(batch, device)
        kl, corr, _, _, _, _, _ = loss_bundle(model, head, batch, parent, tau)
        gradients.append(loss_gradients(kl, corr, head))
    kl_rms = float(np.sqrt(np.mean([g["KL_gradient_norm"] ** 2 for g in gradients])))
    corr_rms = float(
        np.sqrt(np.mean([g["corr_gradient_norm"] ** 2 for g in gradients]))
    )
    if min(kl_rms, corr_rms) <= 0:
        raise RuntimeError("Zero calibration component gradient")
    raw_alpha = (
        cfg["smoke"]["target_weighted_corr_to_KL_gradient_ratio"] * kl_rms / corr_rms
    )
    alpha = float(np.clip(raw_alpha, *cfg["smoke"]["alpha_bounds"]))
    calibration = dict(
        tau=tau,
        alpha=alpha,
        unclamped_tau=raw_tau,
        unclamped_alpha=raw_alpha,
        KL_gradient_RMS=kl_rms,
        corr_gradient_RMS=corr_rms,
        weighted_corr_to_KL_gradient_ratio=alpha * corr_rms / kl_rms,
        gradient_batches=gradients,
        selection="single formula on fixed32 training families, no hyperparameter search",
        validation_used=False,
    )
    save_json(root / "train_only_calibration.json", calibration)
    before = assess_smoke(model, head, subset, parent, device, tau)
    rows = []
    generator = torch.Generator().manual_seed(cfg["smoke"]["seed"])
    loader = DataLoader(
        subset, batch_size=cfg["smoke"]["batch_size"], shuffle=True, generator=generator
    )
    it = iter(loader)
    for step in range(1, cfg["smoke"]["steps"] + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = prepare(batch, device)
        opt.zero_grad(set_to_none=True)
        kl, corr, g, c, status, scores, _ = loss_bundle(model, head, batch, parent, tau)
        diagnostic = (
            loss_gradients(kl, corr, head) if step in [1, cfg["smoke"]["steps"]] else {}
        )
        total = kl + alpha * corr
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            head.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
        )
        opt.step()
        assert all(p.grad is None for p in model.parameters())
        rows.append(
            dict(
                step=step,
                KL=float(kl.detach()),
                corr=float(corr.detach()),
                total=float(total.detach()),
                gradient_norm=float(norm),
                **diagnostic,
            )
        )
    after = assess_smoke(model, head, subset, parent, device, tau)
    if after["empty_corr"] or not np.isfinite([after["KL"], after["corr"]]).all():
        raise RuntimeError("Smoke invalid")
    # No performance threshold used to choose a different alpha/tau. Responses
    # are diagnostics, not validation-driven acceptance or an early-stop rule.
    save_csv(root / "smoke_training.csv", rows)
    save_json(
        root / "smoke_response.json",
        dict(
            before=before,
            after=after,
            valid_gradients=True,
            alpha_tau_retuned=False,
            formal_model_and_optimizer_will_reload=True,
            validation_used=False,
            parameters_at_preparation_start=state_digest(saved["head"]),
        ),
    )
    schedule, audit = continued_batch_schedule(
        len(train),
        cfg["training_batch_size"],
        semantic["training"]["sampler_seed"],
        cfg["start_step"],
        cfg["additional_steps"],
        saved["sampler_state"],
    )
    state = audit.pop("final_sampler_state")
    torch.save(state, root / "final_planned_sampler_state.pt")
    # Explicit schedule proves identical pair/epoch augmentations. Seeds are
    # computed by original transform_record and kept for every scheduled pair.
    for row in schedule:
        train.set_epoch(row["augmentation_epoch"])
        plans = [train.transform_record(i) for i in row["indices"]]
        row["sample_ids"] = [p["sample_id"] for p in plans]
        row["transform_seeds"] = [p["transform_seed"] for p in plans]
    save_jsonl(root / "shared_batch_schedule.jsonl", schedule)
    save_json(root / "sampler_resume_audit.json", audit)
    locked = copy.deepcopy(cfg)
    locked["locked_tau"] = tau
    locked["locked_alpha"] = alpha
    locked["calibration_sha256"] = sha256_file(root / "train_only_calibration.json")
    locked["schedule_sha256"] = sha256_file(root / "shared_batch_schedule.jsonl")
    locked["smoke_samples_sha256"] = sha256_file(root / "smoke_samples.jsonl")
    locked["validation_results_seen_before_lock"] = False
    save_json(root / "locked_protocol.json", locked)
    print(
        json.dumps(
            dict(
                stage="train_only_smoke_and_lock",
                calibration=calibration,
                before=before,
                after=after,
                sampler=audit,
            )
        ),
        flush=True,
    )


def train_protocol(cfg, semantic, parent, model, train, teacher_train, saved, device):
    root = Path(cfg["output"])
    locked = json.loads((root / "locked_protocol.json").read_text())
    assert all(locked[k] == v for k, v in cfg.items())
    assert (
        sha256_file(root / "shared_batch_schedule.jsonl") == locked["schedule_sha256"]
    )
    assert (
        sha256_file(root / "train_only_calibration.json")
        == locked["calibration_sha256"]
    )
    if (root / "training").exists():
        raise FileExistsError("Formal training cannot overwrite")
    output = root / "training"
    output.mkdir()
    tau = locked["locked_tau"]
    alpha = locked["locked_alpha"]
    heads = {}
    optimizers = {}
    digests = {}
    for name in cfg["branches"]:
        h, o = restored_head_and_optimizer(saved, SARPrototypeHead, device)
        heads[name] = h
        optimizers[name] = o
        assert state_digest(h.state_dict()) == state_digest(saved["head"])
        assert state_digest(o.state_dict()) == state_digest(saved["optimizer"])
        digests[name] = dict(
            model=state_digest(h.state_dict()),
            optimizer=state_digest(o.state_dict()),
            restored_steps=sorted({int(v["step"]) for v in o.state.values()}),
            lr=[g["lr"] for g in o.param_groups],
        )
    assert digests["KL_control"] == digests["KL_corr"]
    save_json(root / "branch_start_restore_audit.json", digests)
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    val = OSFamilyDataset(semantic["family_root"], "val", parent)
    monitor, valview = frozen_views(semantic, train, val)
    evalsets = [
        (
            "training_monitor",
            TeacherView(monitor, Path(semantic["output"]) / "teacher"),
        ),
        ("validation", TeacherView(valview, Path(semantic["output"]) / "teacher")),
    ]
    from sdgm.correspondence_evaluation import evaluate_branches

    structural_before = state_digest(model.state_dict())
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for split, view in evalsets:
        evaluate_branches(
            model,
            heads,
            view,
            parent,
            cfg,
            locked,
            device,
            root / "evaluation",
            split,
            0,
        )
    schedule = [
        json.loads(l)
        for l in (root / "shared_batch_schedule.jsonl").read_text().splitlines()
    ]
    log = (output / "training.jsonl").open("w")
    gradient_rows = []
    negative_hist = np.zeros(9, dtype="int64")
    bins = np.array([8, 10, 12, 16, 24, 32, 64, 128, 256, np.inf])
    nvalid_total = 0
    digest = hashlib.sha256()
    negative_digest = hashlib.sha256()
    grad_model_before = structural_before
    for item in schedule:
        step = item["additional_step"]
        train.set_epoch(item["augmentation_epoch"])
        batch = next(
            iter(
                DataLoader(
                    Subset(teacher_train, item["indices"]),
                    batch_size=len(item["indices"]),
                    shuffle=False,
                )
            )
        )
        assert batch["id"] == item["sample_ids"]
        actual_seeds = [
            train.transform_record(i)["transform_seed"] for i in item["indices"]
        ]
        assert actual_seeds == item["transform_seeds"]
        batch = prepare(batch, device)
        with torch.no_grad():
            features = model(batch["optical"], batch["sar"])
            g = geometry_for(model, batch, parent)
            fixed = fixed_competition(features["scores"], g, model.target_pixel_xy)
            negative_digest.update(fixed["negative_indices"].cpu().numpy().tobytes())
            distances = torch.linalg.vector_norm(
                model.target_pixel_xy[fixed["negative_indices"].clamp_min(0)]
                - g["gt_pixel_xy"][:, :, None],
                dim=-1,
            )
            validneg = fixed["negative_valid"] & g["supervised"][..., None]
            negative_hist += np.histogram(distances[validneg].cpu().numpy(), bins=bins)[
                0
            ]
            nvalid_total += int(validneg.sum())
            digest.update(json.dumps(item, sort_keys=True).encode())
            digest.update(batch["matrix"].cpu().numpy().tobytes())
        row = dict(
            additional_step=step,
            total_step=cfg["start_step"] + step,
            augmentation_epoch=item["augmentation_epoch"],
            branches={},
        )
        for branch in cfg["branches"]:
            head = heads[branch]
            head.train()
            opt = optimizers[branch]
            opt.zero_grad(set_to_none=True)
            # Use precisely the shared structure feature/mask/negative object.
            p = head(features["sar_descriptors"].detach())
            kl, _, _, teacher = distillation_loss(
                batch["teacher"], p, model.source_pixel_xy, g
            )
            scores = probability_scores(teacher.detach(), p.flatten(2))["dist"]
            corr, status = correspondence_loss(scores, fixed, tau)
            component = (
                loss_gradients(kl, corr, head)
                if step in cfg["gradient_diagnostic_steps"]
                else {}
            )
            coefficient = alpha if branch == "KL_corr" else 0.0
            total = kl + coefficient * corr
            total.backward()
            grads = [v.grad for v in head.parameters()]
            assert all(v is not None and torch.isfinite(v).all() for v in grads)
            norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            assert [z["lr"] for z in opt.param_groups] == [cfg["learning_rate"]]
            opt.step()
            row["branches"][branch] = dict(
                KL=float(kl.detach()),
                corr=float(corr.detach()),
                objective=float(total.detach()),
                effective_KL_queries=int(g["supervised"].sum()),
                effective_corr_queries=int(status["eligible"].sum()),
                empty_P=int(status["empty_positive"].sum()),
                empty_N=int(status["empty_negative"].sum()),
                N_lt16=int(status["negative_insufficient"].sum()),
                existing_invalid_queries=int((~g["supervised"]).sum()),
                gradient_norm=float(norm),
                nonfinite=False,
            )
            if component:
                component.update(
                    branch=branch,
                    additional_step=step,
                    alpha=alpha,
                    weighted_corr_gradient_norm=alpha * component["corr_gradient_norm"],
                    weighted_corr_to_KL_ratio=alpha
                    * component["corr_gradient_norm"]
                    / max(component["KL_gradient_norm"], 1e-30),
                )
                gradient_rows.append(component)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        log.write(json.dumps(row) + "\n")
        if step % 100 == 0:
            log.flush()
            save_json(output / "progress.json", row)
            print(json.dumps(row), flush=True)
        if step in cfg["evaluation_additional_steps"]:
            for split, view in evalsets:
                evaluate_branches(
                    model,
                    heads,
                    view,
                    parent,
                    cfg,
                    locked,
                    device,
                    root / "evaluation",
                    split,
                    step,
                )
    log.close()
    assert state_digest(model.state_dict()) == grad_model_before
    save_csv(root / "component_gradient_diagnostics.csv", gradient_rows)
    for name in cfg["branches"]:
        assert {int(v["step"]) for v in optimizers[name].state.values()} == {
            cfg["start_step"] + cfg["additional_steps"]
        }
        torch.save(
            dict(
                head=heads[name].state_dict(),
                optimizer=optimizers[name].state_dict(),
                total_step=cfg["start_step"] + cfg["additional_steps"],
                additional_steps=cfg["additional_steps"],
                locked_protocol=locked,
                structural_digest=structural_before,
                final_sampler_state=torch.load(
                    root / "final_planned_sampler_state.pt", weights_only=True
                ),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
            ),
            output / (name + "_last.pt"),
        )
    save_json(
        root / "negative_distance_training.json",
        dict(
            valid_negative_slots=nvalid_total,
            bins_px=[
                "8..10",
                "10..12",
                "12..16",
                "16..24",
                "24..32",
                "32..64",
                "64..128",
                "128..256",
                ">256",
            ],
            counts=negative_hist.tolist(),
            fraction_8to16=float(negative_hist[:3].sum() / max(nvalid_total, 1)),
            strictly_outside8=True,
            note="same fixed-structure negatives reused for both heads; augmentation-specific, no student online mining; not necessarily semantic class negatives",
        ),
    )
    save_json(
        root / "completion.json",
        dict(
            completed=True,
            additional_steps_each=cfg["additional_steps"],
            optimizer_total_steps_each=10000,
            both_initial_model_and_optimizer_exact=True,
            same_batches_augmentation_and_structural_negative_sets=True,
            shared_batch_input_digest=digest.hexdigest(),
            shared_negative_index_digest=negative_digest.hexdigest(),
            structural_digest_before=structural_before,
            structural_digest_after=state_digest(model.state_dict()),
            only_SAR_head_updated=True,
            tau=tau,
            alpha=alpha,
            KL_definition_unchanged=True,
            no_distance_structural_fusion=True,
            lambda_unchanged=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            final_checkpoints={
                n: sha256_file(output / (n + "_last.pt")) for n in cfg["branches"]
            },
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_correspondence_objective.json")
    parser.add_argument("--phase", choices=["prepare", "train"], required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    semantic = json.loads(Path(cfg["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    root = Path(cfg["output"])
    if args.phase == "prepare" and root.exists():
        raise FileExistsError("Preparation requires a new output")
    assert cfg["additional_steps"] == 5000 and cfg["start_step"] == 5000
    assert (
        cfg["training_batch_size"] == semantic["training"]["batch_size"]
        and cfg["learning_rate"] == semantic["training"]["learning_rate"]
    )
    if sha256_file(cfg["resume_checkpoint"]) != cfg["resume_checkpoint_sha256"]:
        raise ValueError("Wrong formal initial checkpoint")
    verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
    before = frozen_provenance(semantic)
    before[cfg["resume_checkpoint"]] = sha256_file(cfg["resume_checkpoint"])
    for p in [
        Path(semantic["output"]) / "teacher/cache_receipt.json",
        Path(semantic["output"]) / "routing_calibration.json",
    ]:
        before[str(p)] = sha256_file(p)
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
    saved = torch.load(cfg["resume_checkpoint"], map_location="cpu", weights_only=False)
    assert saved["step"] == 5000
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    teacher_train = TeacherView(train, Path(semantic["output"]) / "teacher")
    if args.phase == "prepare":
        prepare_protocol(
            cfg, semantic, parent, model, train, teacher_train, saved, device
        )
        save_json(root / "frozen_provenance_before.json", before)
    else:
        assert before == json.loads(
            (root / "frozen_provenance_before.json").read_text()
        )
        train_protocol(
            cfg, semantic, parent, model, train, teacher_train, saved, device
        )
        after = {p: sha256_file(p) for p in before}
        assert before == after
        save_json(root / "frozen_provenance_after.json", after)


if __name__ == "__main__":
    main()
