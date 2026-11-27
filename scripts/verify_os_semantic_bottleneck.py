"""Independently reconstruct A/B/C competitor statistics and audit fixed N sets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, apply_affine
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views
from sdgm.semantic_routing import (
    SARPrototypeHead,
    sample_native_map,
    probability_normalize,
)
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.train_structure_smoke import prepare, geometry_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_bottleneck.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    semantic = json.loads(Path(config["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    before = json.loads((root / "frozen_provenance_before.json").read_text())
    assert before == json.loads((root / "frozen_provenance_after.json").read_text())
    for name, digest in before.items():
        assert sha256_file(name) == digest, name
    verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
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
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent)
    views = frozen_views(semantic, train, val)
    checks = {}
    atol = config["score_tie_atol"]
    for split, view in zip(config["evaluation_splits"], views):
        a = np.load(root / f"{split}_query_diagnostics.npz")
        shape = a["supervised"].shape
        assert shape == (952, 144) and a["supervised"].sum() == 116953
        previous = np.load(Path(config["supplement_root"]) / f"{split}_queries.npz")
        for key in [
            "supervised",
            "structure_nearest",
            "semantic_nearest",
            "structure_indices",
            "semantic_indices",
        ]:
            np.testing.assert_array_equal(a[key], previous[key])
        for layer in ["A", "B", "C"]:
            # Every stored query, not just the reported groups: reconstruct both
            # rank bounds and margin from independent NumPy competitor arrays.
            idx = a["positive_indices"]
            pv = a["positive_slot_valid"]
            inv = a["inverse_target_valid"][:, None].repeat(144, axis=1)
            pv = pv & np.take_along_axis(inv, idx.clip(min=0), -1)
            nv = a["negative_common_valid"]
            ps = a[layer + "_positive_scores"]
            ns = a[layer + "_negative_scores"]
            valid = pv.any(-1) & nv.any(-1)
            pm = np.where(pv, ps, -np.inf).max(-1)
            nm = np.where(nv, ns, -np.inf).max(-1)
            pm = np.where(pv.any(-1), pm, 0)
            score = np.concatenate([ps, ns], -1)
            support = np.concatenate([pv, nv], -1)
            minimum = 1 + ((score > pm[..., None] + atol) & support).sum(-1)
            maximum = 1 + ((ns >= pm[..., None] - atol) & nv).sum(-1)
            minimum = np.where(pv.any(-1), minimum, 0)
            maximum = np.where(pv.any(-1), maximum, 0)
            np.testing.assert_array_equal(minimum, a[layer + "_rank_min"])
            np.testing.assert_array_equal(maximum, a[layer + "_rank_max"])
            delta = np.where(valid, pm - nm, 0)
            np.testing.assert_array_equal(delta, a[layer + "_delta"])
            for key, value in [
                ("valid", valid),
                ("win", valid & (delta > atol)),
                ("loss", valid & (delta < -atol)),
                ("tie", valid & (np.abs(delta) <= atol)),
            ]:
                np.testing.assert_array_equal(value, a[layer + "_" + key])
        np.testing.assert_array_equal(
            a["positive_common"].sum(-1), a["A_positive_count"]
        )
        assert np.array_equal(a["A_valid"], a["B_valid"]) and np.array_equal(
            a["B_valid"], a["C_valid"]
        )
        qmask = a["supervised"]
        original = a["structure_nearest"] <= 8
        s = a["semantic_nearest"] <= 8
        np.testing.assert_array_equal(a["group_semantic_new"], qmask & ~original & s)
        np.testing.assert_array_equal(a["group_semantic_lost"], qmask & original & ~s)
        # Spotcheck hard negatives using CPU argmax over all1024 locations, with
        # exact same prechosen P and sequential24px exclusion, independent code.
        data = TeacherView(view, Path(semantic["output"]) / "teacher")
        batch = prepare(
            next(iter(DataLoader(data, batch_size=4, shuffle=False))), device
        )
        with torch.no_grad():
            out = model(batch["optical"], batch["sar"])
            geom = geometry_for(model, batch, parent)
            scores = out["scores"].cpu().numpy()
            positive = geom["positive"].cpu().numpy()
            target = model.target_pixel_xy.cpu().numpy()
            qs = head(out["sar_descriptors"])
            qo = probability_normalize(
                sample_native_map(batch["teacher"], model.source_pixel_xy)
            )
            full = torch.bmm(qo.transpose(1, 2), qs.flatten(2)).cpu().numpy()
        for b in range(4):
            for q in range(144):
                available = (
                    geom["target_valid"][b].cpu().numpy().copy() & ~positive[b, q]
                )
                chosen = []
                for _ in range(16):
                    if available.any():
                        idx = int(np.where(available, scores[b, q], -np.inf).argmax())
                        chosen.append(idx)
                        available &= ((target - target[idx]) ** 2).sum(-1) > 24**2
                    else:
                        chosen.append(-1)
                np.testing.assert_array_equal(chosen, a["negative_indices"][b, q])
            p = positive[b]
            support = geom["target_valid"][b].cpu().numpy()[None]
            best = np.where(p, full[b], -np.inf).max(-1)
            best = np.where(p.any(-1), best, 0)
            rmin = 1 + ((full[b] > best[:, None] + atol) & support).sum(-1)
            rmax = 1 + ((full[b] >= best[:, None] - atol) & support & ~p).sum(-1)
            rmin = np.where(p.any(-1), rmin, 0)
            rmax = np.where(p.any(-1), rmax, 0)
            np.testing.assert_array_equal(rmin, a["C_full_rank_min"][b])
            np.testing.assert_array_equal(rmax, a["C_full_rank_max"][b])
        if split == "validation":
            assert (
                a["group_semantic_new"].sum() == 277
                and a["group_semantic_lost"].sum() == 5645
            )
        checks[split] = dict(
            all137088_query_competitions_numpy_exact=True,
            all_layer_positions_identical=True,
            hard_negative_CPU_oracle_queries=576,
            C_full_grid_rank_CPU_oracle_queries=576,
            previous_candidate_outputs_exact=True,
            original_group_membership_exact=True,
            continuous_inverse_roundtrip_max_px=float(
                a["continuous_inverse_to_source_error"][qmask].max()
            ),
        )
    save_json(
        root / "verification.json",
        dict(
            passed=True,
            checks=checks,
            all_frozen_files_and_weights_unchanged=True,
            GT_only_diagnostic=True,
            no_training=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    print(
        "All three-layer competitor stats, fixed negative peaks, full-grid ranks and frozen groups verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
