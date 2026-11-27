"""First delivery: baseline ranks, four equal-capacity forward checks and fixed fits."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sdgm.coarse import StructuralRetriever
from sdgm.context_descriptors import (
    ContextDescriptorHead,
    orthogonal_projection,
    full_grid_pair_mass_loss,
    sample_information_blocks,
)
from sdgm.context_descriptor_data import IdentityTrainingView, frozen_inputs, group_maps
from sdgm.context_descriptor_evaluation import evaluate_context
from sdgm.continuation_diagnostics import state_digest
from sdgm.io import save_json, save_jsonl, save_csv
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views
from sdgm.semantic_routing import SARPrototypeHead
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.train_structure_smoke import prepare, geometry_for
from scripts.train_os_semantic_routing import frozen_provenance


def gradient_check(head):
    result = {}
    for name, part in [("optical", head.optical_head), ("sar", head.sar_head)]:
        gradients = [p.grad for p in part.parameters()]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing/nonfinite descriptor-head gradient")
        norm = float(
            torch.sqrt(sum(g.detach().double().square().sum() for g in gradients))
        )
        if norm <= 0:
            raise RuntimeError("No gradient on one modality head")
        result[name] = dict(gradient_norm=norm, all_parameter_gradients_finite=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_context_descriptor.json")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    semantic = json.loads(Path(cfg["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    root = Path(cfg["output"])
    if root.exists():
        raise FileExistsError("New stageA/B output required, never replace fits")
    assert cfg["structural_checkpoint"] == semantic["structural_checkpoint"]
    for file, key in [
        ("structural_checkpoint", "structural_checkpoint_sha256"),
        ("SAR_head_checkpoint", "SAR_head_checkpoint_sha256"),
    ]:
        if sha256_file(cfg[file]) != cfg[key]:
            raise ValueError("Wrong frozen endpoint " + file)
    verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
    before = frozen_provenance(semantic)
    for name in [
        cfg["SAR_head_checkpoint"],
        str(Path(semantic["output"]) / "teacher/cache_receipt.json"),
    ]:
        before[name] = sha256_file(name)
    root.mkdir(parents=True)
    save_json(root / "preregistered_protocol.json", cfg)
    save_json(root / "frozen_provenance_before.json", before)
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
    semsaved = torch.load(
        cfg["SAR_head_checkpoint"], map_location="cpu", weights_only=False
    )
    assert semsaved["total_step"] == 10000
    semhead = SARPrototypeHead().to(device)
    semhead.load_state_dict(semsaved["head"])
    semhead.requires_grad_(False).eval()
    frozen_digests = dict(
        structural=state_digest(model.state_dict()),
        SAR_semantic=state_digest(semhead.state_dict()),
    )
    projection = orthogonal_projection(cfg["projection_seed"]).to(device)
    np.save(root / "structural_projection.npy", projection.cpu().numpy())
    save_json(
        root / "projection_receipt.json",
        dict(
            seed=cfg["projection_seed"],
            shape=list(projection.shape),
            sha256=sha256_file(root / "structural_projection.npy"),
            row_orthogonality_max_error=float(
                (projection @ projection.T - torch.eye(32, device=device)).abs().max()
            ),
            shared_modalities=True,
            frozen=True,
        ),
    )
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent)
    monitor, valview = frozen_views(semantic, train, val)
    # Mandatory structure ranking precedes descriptor fitting or optimization.
    for split, view in [("training_monitor", monitor), ("validation", valview)]:
        result = evaluate_context(
            model,
            semhead,
            {},
            TeacherView(view, Path(semantic["output"]) / "teacher"),
            parent,
            cfg,
            projection,
            device,
            root / "baseline",
            split,
            baseline_only=True,
        )
        old = np.load(Path(cfg["baseline_reference"]) / f"{split}_score_queries.npz")
        new = np.load(root / "baseline" / f"{split}_queries.npz")
        np.testing.assert_array_equal(
            new["structure_indices"], old["structure_indices"]
        )
        np.testing.assert_array_equal(
            new["structure_nearest"], old["structure_nearest"]
        )
        np.testing.assert_array_equal(new["supervised"], old["supervised"])
        print(
            json.dumps(
                dict(
                    stage="mandatory_structure_ranking",
                    split=split,
                    summary=result["structure"],
                )
            ),
            flush=True,
        )
    groups = {}
    for i, row in enumerate(train.records):
        groups.setdefault(row["family_id"], []).append(i)
    rng = np.random.default_rng(cfg["smoke"]["selection_seed"])
    indices = sorted(
        int(rng.choice(groups[f]))
        for f in rng.choice(sorted(groups), cfg["smoke"]["families"], replace=False)
    )
    small = IdentityTrainingView(train, indices)
    dataset = TeacherView(small, Path(semantic["output"]) / "teacher")
    save_jsonl(root / "fixed_training_samples.jsonl", dataset.records)
    save_json(
        root / "small_geometry_protocol.json",
        dict(
            sample_ids=[r["sample_id"] for r in dataset.records],
            matrix="identity for all optical andSAR, native base pairs, no image warp or extra augmentation",
            source_support="full256 original extent",
            target_support="full256 original extent",
            label_scope="publisher nominal co-registration, residual unknown",
            validation_used=False,
        ),
    )
    torch.manual_seed(cfg["smoke"]["head_seed"])
    initial = ContextDescriptorHead()
    initial_state = copy.deepcopy(initial.state_dict())
    initial_digest = state_digest(initial_state)
    heads = {}
    optimizers = {}
    audit = {}
    for name in cfg["groups"]:
        h = ContextDescriptorHead().to(device)
        h.load_state_dict(copy.deepcopy(initial_state))
        heads[name] = h
        o = torch.optim.AdamW(
            h.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"],
            betas=tuple(cfg["optimizer"]["betas"]),
        )
        optimizers[name] = o
        audit[name] = dict(
            initial_weight_digest=state_digest(h.state_dict()),
            trainable_parameters=sum(p.numel() for p in h.parameters()),
            optimizer_steps_start=0,
            old_optimizer_restored=False,
            optimizer_state_empty=len(o.state) == 0,
            optical_SAR_parameters_independent=h.optical_head[0].weight.data_ptr()
            != h.sar_head[0].weight.data_ptr(),
        )
    assert {r["initial_weight_digest"] for r in audit.values()} == {initial_digest}
    assert len({r["trainable_parameters"] for r in audit.values()}) == 1
    save_json(root / "four_group_start_audit.json", audit)
    # Freeze raw inputs and geometric labels separately; pool/forward accepts no
    # matrix or GT. Cached scale-map values never depend on geometric labels.
    cached = []
    cache_audit = []
    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
        batch = prepare(batch, device)
        inp = frozen_inputs(model, semhead, batch)
        g = geometry_for(model, batch, parent)
        for i, sample in enumerate(batch["id"]):
            entry = dict(
                id=sample,
                family_id=batch["family_id"][i],
                geometry={k: v[i : i + 1] for k, v in g.items()},
                groups={},
            )
            for name in cfg["groups"]:
                single = {k: v[i : i + 1] for k, v in inp.items()}
                om, sm, of, sf = group_maps(single, projection, name)
                # This cache contains sampled PRE-MLP information blocks. It is
                # exactly the specified sample-before-nonlinearity interface.
                uo = sample_information_blocks(om, model.source_pixel_xy)
                us = sample_information_blocks(sm, model.target_pixel_xy)
                entry["groups"][name] = (uo.detach(), us.detach())
                cache_audit.append(
                    dict(
                        sample_id=sample,
                        group=name,
                        optical_shape=list(uo.shape),
                        SAR_shape=list(us.shape),
                        optical_valid_fraction=[
                            float(of[0, k].mean()) for k in range(4)
                        ],
                        SAR_valid_fraction=[float(sf[0, k].mean()) for k in range(4)],
                        cached_block_sha256=state_digest([uo, us]),
                        GT_in_forward=False,
                    )
                )
            cached.append(entry)
    save_json(root / "forward_cache_audit.json", cache_audit)
    histories = []

    def evaluate(step):
        res = evaluate_context(
            model,
            semhead,
            heads,
            dataset,
            parent,
            cfg,
            projection,
            device,
            root / "smoke",
            f"step{step:05d}",
        )
        for name, summary in res.items():
            histories.append(dict(step=step, method=name, summary=summary))
        print(
            json.dumps(
                dict(
                    stage="small_fixed_evaluation",
                    step=step,
                    groups={
                        k: dict(
                            loss=v["loss_pair_mean"],
                            rank=v["ranking"]["all"]["best_positive_rank_min"]["50"],
                            top1=v["ranking"]["all"]["top1_coverage"],
                            independent4=v["ranking"]["all"][
                                "independent_NMS4_coverage"
                            ],
                            merged4=v["recall_at_4"],
                        )
                        for k, v in res.items()
                    },
                )
            ),
            flush=True,
        )

    evaluate(0)
    log = (root / "smoke_training.jsonl").open("w")
    gradients = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(cfg["smoke"]["sampler_seed"])
    loader = DataLoader(
        list(range(len(cached))),
        batch_size=cfg["optimizer"]["batch_size"],
        shuffle=True,
        generator=generator,
    )
    iterator = iter(loader)
    schedule = []
    for step in range(1, cfg["smoke"]["steps"] + 1):
        try:
            index = next(iterator).tolist()
        except StopIteration:
            iterator = iter(loader)
            index = next(iterator).tolist()
        schedule.append(
            dict(step=step, indices=index, sample_ids=[cached[i]["id"] for i in index])
        )
        geom = {
            k: torch.cat([cached[i]["geometry"][k] for i in index])
            for k in cached[0]["geometry"]
        }
        row = dict(step=step, groups={})
        for name in cfg["groups"]:
            h = heads[name]
            h.train()
            o = optimizers[name]
            o.zero_grad(set_to_none=True)
            uo = torch.cat([cached[i]["groups"][name][0] for i in index])
            us = torch.cat([cached[i]["groups"][name][1] for i in index])
            first = torch.nn.functional.normalize(h.optical_head(uo), dim=-1, eps=1e-6)
            second = torch.nn.functional.normalize(h.sar_head(us), dim=-1, eps=1e-6)
            scores = torch.bmm(first, second.transpose(1, 2))
            loss, details = full_grid_pair_mass_loss(scores, geom, cfg["temperature"])
            if not details["valid_pair"].any():
                row["groups"][name] = dict(skipped=True, no_valid_pairs=True)
                continue
            loss.backward()
            check = (
                gradient_check(h) if step in [1, 100, cfg["smoke"]["steps"]] else None
            )
            norm = torch.nn.utils.clip_grad_norm_(
                h.parameters(),
                cfg["optimizer"]["gradient_clip"],
                error_if_nonfinite=True,
            )
            o.step()
            if check:
                gradients.append(dict(step=step, group=name, **check))
            row["groups"][name] = dict(
                loss=float(loss.detach()),
                gradient_norm=float(norm),
                valid_pairs=int(details["valid_pair"].sum()),
                effective_queries=int(details["supervised"].sum()),
                empty_positive=int(details["empty_positive"].sum()),
                nonfinite=False,
            )
        assert all(
            p.grad is None and not p.requires_grad for p in model.parameters()
        ) and all(p.grad is None and not p.requires_grad for p in semhead.parameters())
        log.write(json.dumps(row) + "\n")
        if step % 100 == 0:
            log.flush()
            save_json(root / "progress.json", row)
            print(json.dumps(dict(stage="fit", **row)), flush=True)
        if step in cfg["smoke"]["evaluation_steps"]:
            evaluate(step)
    log.close()
    save_jsonl(root / "shared_smoke_schedule.jsonl", schedule)
    save_json(root / "smoke_gradient_checks.json", gradients)
    save_json(root / "smoke_evaluation_history.json", histories)
    assert frozen_digests == dict(
        structural=state_digest(model.state_dict()),
        SAR_semantic=state_digest(semhead.state_dict()),
    )
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    for name in cfg["groups"]:
        torch.save(
            dict(
                head=heads[name].state_dict(),
                optimizer=optimizers[name].state_dict(),
                step=cfg["smoke"]["steps"],
                config=cfg,
                initial_digest=initial_digest,
                projection_sha256=sha256_file(root / "structural_projection.npy"),
                sampler_state=generator.get_state(),
            ),
            checkpoint_dir / (name + "_smoke.pt"),
        )
    save_json(
        root / "frozen_provenance_after.json", {p: sha256_file(p) for p in before}
    )
    assert before == json.loads((root / "frozen_provenance_after.json").read_text())
    save_json(
        root / "completion.json",
        dict(
            completed_first_delivery=True,
            stages=["pure_structure_ranks", "A_unit_forward", "B_fixed32fits"],
            formal_stageC_run=False,
            equal_smoke_updates_each=cfg["smoke"]["steps"],
            groups=cfg["groups"],
            new_descriptor_parameter_count_each=audit[cfg["groups"][0]][
                "trainable_parameters"
            ],
            frozen_digests_before_after=frozen_digests,
            teacher_SAR_KL10000_endpoint_fixed=True,
            old_optimizer_not_restored=True,
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            OS_test_evaluated=False,
            UBC_evaluated=False,
            formal_seeds_preregistered=cfg["formal_preregistration"]["paired_seeds"],
        ),
    )


if __name__ == "__main__":
    main()
