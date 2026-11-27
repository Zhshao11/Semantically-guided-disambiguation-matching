"""Verify same PN scores, algebraic identity and independent full-grid distance NMS."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, apply_affine, spatial_candidates
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views
from sdgm.semantic_routing import (
    SARPrototypeHead,
    probability_normalize,
    sample_native_map,
)
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.verify_os_pure_semantic import numpy_full_grid_oracle
from scripts.train_structure_smoke import prepare, geometry_for


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_semantic_score_control.json")
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text())
    s = json.loads(Path(c["semantic_config"]).read_text())
    parent = json.loads(Path(s["parent_config"]).read_text())
    root = Path(c["output"])
    before = json.loads((root / "frozen_provenance_before.json").read_text())
    assert before == json.loads((root / "frozen_provenance_after.json").read_text())
    for name, digest in before.items():
        assert sha256_file(name) == digest, name
    verify_teacher_cache(Path(s["output"]) / "teacher", s)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{c['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(s["structural_checkpoint"], map_location="cpu", weights_only=False)[
            "model"
        ]
    )
    model.requires_grad_(False).eval()
    head = SARPrototypeHead().to(device)
    head.load_state_dict(
        torch.load(
            Path(s["output"]) / "training/semantic_head.pt",
            map_location="cpu",
            weights_only=False,
        )["head"]
    )
    head.requires_grad_(False).eval()
    train = OSFamilyDataset(s["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(s["family_root"], "val", parent)
    views = frozen_views(s, train, val)
    checks = {}
    for split, view in zip(c["evaluation_splits"], views):
        a = np.load(root / f"{split}_score_queries.npz")
        old = np.load(Path(c["diagnostic_root"]) / f"{split}_query_diagnostics.npz")
        tol = c["score_tie_atol"]
        for k in [
            "supervised",
            "positive_indices",
            "positive_slot_valid",
            "negative_indices",
            "negative_common_valid",
        ]:
            np.testing.assert_array_equal(a[k], old[k])
        pv = a["positive_slot_valid"]
        inv = old["inverse_target_valid"][:, None].repeat(144, axis=1)
        pv = pv & np.take_along_axis(inv, a["positive_indices"].clip(min=0), -1)
        nv = a["negative_common_valid"]
        for l in ["B", "C"]:
            for m in ["dot", "dist"]:
                pscore = a[l + "_" + m + "_positive_scores"]
                nscore = a[l + "_" + m + "_negative_scores"]
                pm = np.where(pv, pscore, -np.inf).max(-1)
                nm = np.where(nv, nscore, -np.inf).max(-1)
                v = pv.any(-1) & nv.any(-1)
                pm = np.where(pv.any(-1), pm, 0)
                delta = np.where(v, pm - nm, 0)
                score = np.concatenate([pscore, nscore], -1)
                support = np.concatenate([pv, nv], -1)
                rmin = 1 + ((score > pm[..., None] + tol) & support).sum(-1)
                rmax = 1 + ((nscore >= pm[..., None] - tol) & nv).sum(-1)
                rmin = np.where(pv.any(-1), rmin, 0)
                rmax = np.where(pv.any(-1), rmax, 0)
                for k, value in [
                    ("delta", delta),
                    ("rank_min", rmin),
                    ("rank_max", rmax),
                    ("win", v & (delta > tol)),
                    ("loss", v & (delta < -tol)),
                    ("tie", v & (np.abs(delta) <= tol)),
                ]:
                    np.testing.assert_array_equal(a[l + "_" + m + "_" + k], value)
                if m == "dot":
                    for k in ["delta", "rank_min", "rank_max", "win", "loss", "tie"]:
                        np.testing.assert_array_equal(
                            a[l + "_dot_" + k], old[l + "_" + k]
                        )
            # Full stored PN algebra from per-target norm arrays.
            norm = a[l + "_target_norm2"][:, None].repeat(144, axis=1)
            pn = a["query_norm2"][..., None]
            for k, idx, valid in [
                ("positive", a["positive_indices"], pv),
                ("negative", a["negative_indices"], nv),
            ]:
                expected = (
                    a[l + "_dot_" + k + "_scores"]
                    - 0.5 * np.take_along_axis(norm, idx.clip(min=0), -1)
                    - 0.5 * pn
                )
                np.testing.assert_array_equal(
                    expected[valid], a[l + "_dist_" + k + "_scores"][valid]
                )
        data = TeacherView(view, Path(s["output"]) / "teacher")
        batch = prepare(
            next(iter(DataLoader(data, batch_size=4, shuffle=False))), device
        )
        with torch.no_grad():
            out = model(batch["optical"], batch["sar"])
            g = geometry_for(model, batch, parent)
            ps = head(out["sar_descriptors"])
            qo = probability_normalize(
                sample_native_map(batch["teacher"], model.source_pixel_xy)
            )
            back = apply_affine(
                torch.linalg.inv(batch["matrix"]), model.target_pixel_xy
            )
            qt = probability_normalize(sample_native_map(batch["teacher"], back))
            original = spatial_candidates(
                out["scores"], g["target_valid"], model.target_pixel_xy
            )
            dot = torch.bmm(qo.transpose(1, 2), ps.flatten(2))
            distance = (
                dot
                - 0.5 * ps.square().sum(1).flatten(1)[:, None]
                - 0.5 * qo.square().sum(1)[:, :, None]
            )
            direct_errors = {}
            for l, target in [("B", qt), ("C", ps.flatten(2))]:
                # Independent direct squared-vector differences, chunk queries
                # to avoid materializing a large BxQx1024x32 tensor.
                direct = []
                for q in range(0, 144, 12):
                    direct.append(
                        -0.5
                        * (qo[:, :, q : q + 12, None] - target[:, :, None, :])
                        .square()
                        .sum(1)
                    )
                direct = torch.cat(direct, 1)
                score = (
                    torch.bmm(qo.transpose(1, 2), target)
                    - 0.5 * target.square().sum(1)[:, None]
                    - 0.5 * qo.square().sum(1)[:, :, None]
                )
                err = float((score - direct).abs().max())
                assert err < 1e-6, (l, err)
                direct_errors[l] = err
            for i in range(4):
                oracle = numpy_full_grid_oracle(
                    original["indices"][i].cpu().numpy(),
                    original["valid"][i].cpu().numpy(),
                    distance[i].cpu().numpy(),
                    g["target_valid"][i].cpu().numpy(),
                    model.target_pixel_xy.cpu().numpy(),
                )
                np.testing.assert_array_equal(oracle, a["dist_supplement_indices"][i])
        checks[split] = dict(
            all_PN_score_identity_and_stats_numpy_exact=True,
            previous_dot_exact=True,
            direct_squared_distance_max_error=direct_errors,
            full_grid_numpy_distance_NMS_queries=576,
            full_grid_NMS_indices_exact=True,
            effective_queries=int(a["supervised"].sum()),
        )
    save_json(
        root / "verification.json",
        dict(
            passed=True,
            checks=checks,
            no_training=True,
            no_new_normalization=True,
            fixed_weights_caches_masks_and_test_pool_hashes_unchanged=True,
            distance_not_fused_with_structure=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    print(
        "All stored score algebra/rank stats, direct vector distance and full-grid distance NMS verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
