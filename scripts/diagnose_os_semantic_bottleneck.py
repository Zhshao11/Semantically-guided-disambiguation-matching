"""Three-layer same-position diagnostics; GT never enters deployed candidate APIs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, spatial_candidates, apply_affine
from sdgm.continuation_diagnostics import state_digest
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views, candidate_arrays
from sdgm.semantic_routing import (
    SARPrototypeHead,
    sample_native_map,
    probability_normalize,
)
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from sdgm.semantic_bottleneck import (
    hard_negative_peaks,
    comparison_stats,
    pairwise_comparisons,
    full_grid_rank,
)
from scripts.train_structure_smoke import prepare, geometry_for


@torch.no_grad()
def diagnose(model, head, data, semantic, config, parent, device, root, split):
    teacher_root = Path(semantic["output"]) / "teacher"
    tokens = np.load(teacher_root / "patch_tokens.npy", mmap_mode="r")
    records = [
        json.loads(l)
        for l in (teacher_root / "cache_samples.jsonl").read_text().splitlines()
    ]
    lookup = {r["sample_id"]: i for i, r in enumerate(records)}
    old = np.load(Path(config["supplement_root"]) / f"{split}_queries.npz")
    storage = {}
    sample_ids = []
    families = []
    offset = 0

    def add(key, value):
        storage.setdefault(key, []).append(value.detach().cpu().numpy())

    for batch in DataLoader(data, batch_size=config["batch_size"], shuffle=False):
        batch = prepare(batch, device)
        b = len(batch["id"])
        out = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        ps = head(out["sar_descriptors"])
        qo = probability_normalize(
            sample_native_map(batch["teacher"], model.source_pixel_xy)
        )
        score_C = torch.bmm(qo.transpose(1, 2), ps.flatten(2))
        inverse = torch.linalg.inv(batch["matrix"])
        back = apply_affine(inverse, model.target_pixel_xy)
        inverse_valid = (
            torch.isfinite(back).all(-1) & (back >= 0).all(-1) & (back <= 255).all(-1)
        )
        patch = torch.from_numpy(
            np.stack(
                [np.array(tokens[lookup[s]], dtype="float32") for s in batch["id"]]
            )
        ).to(device)
        dense = F.normalize(
            F.interpolate(
                patch.transpose(1, 2).reshape(b, 384, 18, 18),
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
            ),
            dim=1,
        )
        d_source = F.normalize(sample_native_map(dense, model.source_pixel_xy), dim=1)
        d_target = F.normalize(sample_native_map(dense, back), dim=1)
        score_A = torch.bmm(d_source.transpose(1, 2), d_target)
        q_target = probability_normalize(sample_native_map(batch["teacher"], back))
        score_B = torch.bmm(qo.transpose(1, 2), q_target)
        positive = geom["positive"]
        nidx, nvalid = hard_negative_peaks(
            out["scores"],
            positive,
            geom["target_valid"],
            model.target_pixel_xy,
            count=16,
            radius=24,
        )
        commonP = positive & inverse_valid[:, None]
        n_inverse_ok = (
            inverse_valid[:, None].expand(-1, 144, -1).gather(-1, nidx.clamp_min(0))
        )
        commonN_valid = nvalid & n_inverse_ok
        nmask = torch.zeros_like(positive, dtype=torch.long)
        nmask.scatter_add_(-1, nidx.clamp_min(0), commonN_valid.long())
        nmask = nmask > 0
        add("supervised", geom["supervised"])
        add("positive_original", positive)
        add("positive_common", commonP)
        add("negative_indices", nidx)
        add("negative_valid", nvalid)
        add("negative_common_valid", commonN_valid)
        add("inverse_target_valid", inverse_valid)
        add("target_valid", geom["target_valid"])
        add("gt_xy", geom["gt_pixel_xy"])
        add("inverse_target_xy", back)
        for k in [
            "gt_outside",
            "empty_positive",
            "invalid_source",
            "gt_invalid_support",
        ]:
            add(k, geom[k])
        add("positive_original_count", positive.sum(-1))
        add("negative_original_count", nvalid.sum(-1))
        add("negative_insufficient", nvalid.sum(-1) < 16)
        add(
            "positive_inverse_outside_count",
            (positive & ~inverse_valid[:, None]).sum(-1),
        )
        add("negative_inverse_outside_count", (nvalid & ~n_inverse_ok).sum(-1))
        add(
            "target_inverse_outside_count",
            ((~inverse_valid) & geom["target_valid"]).sum(-1),
        )
        complete = ~(positive & ~inverse_valid[:, None]).any(-1) & ~(
            nvalid & ~n_inverse_ok
        ).any(-1)
        add("inverse_complete", complete)
        # Store positive slots by index only for audit, never use this order for ranks.
        pidx = (
            torch.arange(1024, device=device)[None, None]
            .expand_as(positive)
            .masked_fill(~positive, 1024)
            .sort(-1)
            .values[..., :8]
        )
        assert (positive.sum(-1) <= 8).all()
        add("positive_indices", torch.where(pidx < 1024, pidx, -torch.ones_like(pidx)))
        add("positive_slot_valid", pidx < 1024)
        for layer, scores in [("A", score_A), ("B", score_B), ("C", score_C)]:
            assert torch.isfinite(scores).all()
            stats = comparison_stats(scores, commonP, nmask, config["score_tie_atol"])
            stats.update(
                pairwise_comparisons(
                    scores, commonP, nidx, commonN_valid, config["score_tie_atol"]
                )
            )
            for key, value in stats.items():
                add(layer + "_" + key, value)
            add(layer + "_positive_scores", scores.gather(-1, pidx.clamp_max(1023)))
            add(layer + "_negative_scores", scores.gather(-1, nidx.clamp_min(0)))
        for key, value in full_grid_rank(
            score_C, positive, geom["target_valid"], config["score_tie_atol"]
        ).items():
            add("C_full_" + key, value)
        # Continuous GT sanity reference kept separate from actual-grid ranks.
        gt = geom["gt_pixel_xy"]
        back_gt = torch.einsum("bij,bqj->bqi", inverse[:, :2, :2], gt) + inverse[
            :, :2, 2, None
        ].transpose(1, 2)
        cont_valid = (
            torch.isfinite(back_gt).all(-1)
            & (back_gt >= 0).all(-1)
            & (back_gt <= 255).all(-1)
        )
        cont_d = F.normalize(sample_native_map(dense, back_gt), dim=1)
        cont_q = probability_normalize(sample_native_map(batch["teacher"], back_gt))
        cont_s = probability_normalize(sample_native_map(ps, gt))
        add("continuous_inverse_valid", cont_valid)
        add(
            "continuous_inverse_to_source_error",
            torch.linalg.vector_norm(back_gt - model.source_pixel_xy[None], dim=-1),
        )
        for layer, value in [
            ("A", (d_source * cont_d).sum(1)),
            ("B", (qo * cont_q).sum(1)),
            ("C", (qo * cont_s).sum(1)),
        ]:
            add(layer + "_continuous_score", value)
        raw = score_C.masked_fill(
            ~geom["target_valid"][:, None], torch.finfo(score_C.dtype).min
        )
        rawidx = raw.argsort(dim=-1, descending=True, stable=True)[..., :4]
        rawvalid = geom["target_valid"][:, None].expand(-1, 144, -1).gather(-1, rawidx)
        rawxy = model.target_pixel_xy[rawidx]
        rawdist = torch.linalg.vector_norm(rawxy - gt[:, :, None], dim=-1).masked_fill(
            ~rawvalid, torch.inf
        )
        add("C_raw_top4_indices", rawidx)
        add("C_raw_top4_nearest", rawdist.min(-1).values)
        sem4 = spatial_candidates(
            score_C,
            geom["target_valid"],
            model.target_pixel_xy,
            count=4,
            nms_radius_px=24,
        )
        sfirst, snear = candidate_arrays(sem4, geom)
        add("C_semantic4_indices", sem4["indices"])
        add("C_semantic4_valid", sem4["valid"])
        add("C_semantic4_first_distance", sfirst)
        add("C_semantic4_nearest_distance", snear)
        add(
            "C_semantic4_all_distances",
            torch.linalg.vector_norm(
                sem4["pixel_xy"] - gt[:, :, None], dim=-1
            ).masked_fill(~sem4["valid"], torch.inf),
        )
        original = spatial_candidates(
            out["scores"],
            geom["target_valid"],
            model.target_pixel_xy,
            count=4,
            nms_radius_px=24,
        )
        np.testing.assert_array_equal(
            original["indices"].cpu().numpy(),
            old["structure_indices"][offset : offset + b],
        )
        # Recompute unchanged deployed supplementation through its GT-free API.
        from sdgm.semantic_supplement import supplement_from_full_grid

        supplement = supplement_from_full_grid(
            original, score_C, geom["target_valid"], model.target_pixel_xy
        )
        np.testing.assert_array_equal(
            supplement["indices"].cpu().numpy(),
            old["semantic_indices"][offset : offset + b],
        )
        _, near = candidate_arrays(supplement, geom)
        np.testing.assert_array_equal(
            near.cpu().numpy(), old["semantic_nearest"][offset : offset + b]
        )
        add(
            "structure_first2_distance",
            torch.linalg.vector_norm(
                original["pixel_xy"][..., :2, :] - gt[:, :, None], dim=-1
            )
            .masked_fill(~original["valid"][..., :2], torch.inf)
            .min(-1)
            .values,
        )
        supplemental_dist = torch.linalg.vector_norm(
            supplement["pixel_xy"][..., 2:, :] - gt[:, :, None], dim=-1
        ).masked_fill(~supplement["valid"][..., 2:], torch.inf)
        add("supplemental2_distances", supplemental_dist)
        add("supplemental2_nearest_distance", supplemental_dist.min(-1).values)
        available = geom["target_valid"][:, None].expand_as(positive).clone()
        for k in range(2):
            xy = original["pixel_xy"][..., k, :]
            suppressed = (
                (model.target_pixel_xy[None, None] - xy[:, :, None]) ** 2
            ).sum(-1) <= 24**2
            available &= ~(suppressed & original["valid"][..., k, None])
        add("positive_remaining_after_prefix", (positive & available).sum(-1))
        add(
            "C_best_positive_after_prefix",
            score_C.masked_fill(~(positive & available), -torch.inf).max(-1).values,
        )
        sample_ids.extend(batch["id"])
        families.extend(batch["family_id"])
        offset += b
    arrays = {k: np.concatenate(v, axis=0) for k, v in storage.items()}
    arrays.update(sample_id=np.array(sample_ids), family_id=np.array(families))
    np.testing.assert_array_equal(arrays["supervised"], old["supervised"])
    for key in [
        "structure_nearest",
        "semantic_nearest",
        "structure_indices",
        "semantic_indices",
    ]:
        arrays[key] = old[key]
    a = arrays["structure_nearest"] <= 8
    s = arrays["semantic_nearest"] <= 8
    keep = arrays["supervised"]
    for name, mask in dict(
        all_supervised=keep,
        structure_miss=keep & ~a,
        semantic_new=keep & ~a & s,
        semantic_lost=keep & a & ~s,
        retained_hit=keep & a & s,
        both_miss=keep & ~a & ~s,
    ).items():
        arrays["group_" + name] = mask
    if split == "validation":
        assert (
            arrays["group_semantic_new"].sum() == 277
            and arrays["group_semantic_lost"].sum() == 5645
        )
    np.savez_compressed(root / f"{split}_query_diagnostics.npz", **arrays)
    save_json(
        root / f"{split}_status.json",
        dict(
            pairs=len(sample_ids),
            families=len(set(families)),
            total_source_queries=int(keep.size),
            supervised_queries=int(keep.sum()),
            groups={
                name: int(arrays["group_" + name].sum()) for name in config["groups"]
            },
            raw_positive_empty=int((arrays["positive_original_count"] == 0).sum()),
            supervised_positive_empty=int(
                ((arrays["positive_original_count"] == 0) & keep).sum()
            ),
            negative_fewer_than16_all_queries=int(
                arrays["negative_insufficient"].sum()
            ),
            negative_fewer_than16_supervised=int(
                (arrays["negative_insufficient"] & keep).sum()
            ),
            inverse_outside_valid_target_slots=int(
                ((~arrays["inverse_target_valid"]) & arrays["target_valid"]).sum()
            ),
            supervised_incomplete_inverse_competition=int(
                (~arrays["inverse_complete"] & keep).sum()
            ),
            layers_use_identical_competition=True,
            deployed_candidates_exactly_unchanged=True,
            same_image_AB_GT_reference_not_cross_modal_upper_bound=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_bottleneck.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    semantic = json.loads(Path(config["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError("Use a new diagnostic output; no replacement")
    assert config["batch_size"] == parent["batch_size"] == 4
    assert config["positive_radius_px"] == parent["positive_radius_px"] == 8
    assert config["nms_radius_px"] == parent["nms_radius_px"] == 24
    receipt = verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
    previous = json.loads(
        (Path(config["supplement_root"]) / "frozen_provenance_after.json").read_text()
    )
    before = {p: sha256_file(p) for p in previous}
    assert before == previous
    for split in config["evaluation_splits"]:
        p = Path(config["supplement_root"]) / f"{split}_queries.npz"
        before[str(p)] = sha256_file(p)
    root.mkdir(parents=True)
    save_json(root / "preregistered_protocol.json", config)
    save_json(root / "frozen_provenance_before.json", before)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(
            semantic["structural_checkpoint"], map_location="cpu", weights_only=False
        )["model"]
    )
    model.requires_grad_(False).eval()
    head = SARPrototypeHead().to(device)
    head.load_state_dict(
        torch.load(
            Path(semantic["output"]) / "training/semantic_head.pt",
            map_location="cpu",
            weights_only=False,
        )["head"]
    )
    head.requires_grad_(False).eval()
    digest = (state_digest(model.state_dict()), state_digest(head.state_dict()))
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent, training=False)
    views = frozen_views(semantic, train, val)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for split, view in zip(config["evaluation_splits"], views):
        diagnose(
            model,
            head,
            TeacherView(view, Path(semantic["output"]) / "teacher"),
            semantic,
            config,
            parent,
            device,
            root,
            split,
        )
        print(json.dumps(dict(stage="diagnostics_complete", split=split)), flush=True)
    assert digest == (state_digest(model.state_dict()), state_digest(head.state_dict()))
    after = {p: sha256_file(p) for p in before}
    assert before == after
    save_json(root / "frozen_provenance_after.json", after)
    save_json(
        root / "completion.json",
        dict(
            completed=True,
            no_training=True,
            optimizer_steps_added=0,
            parameters_unchanged=True,
            previous_structure_and_semantic_candidates_exact=True,
            GT_only_diagnostic=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
            teacher_cache_version=receipt["cache_version"],
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        ),
    )


if __name__ == "__main__":
    main()
