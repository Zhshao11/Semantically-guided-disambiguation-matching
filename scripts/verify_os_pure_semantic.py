"""Verify frozen artifacts, exact controls and independent full-grid greedy oracle."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, spatial_candidates
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
from scripts.train_structure_smoke import prepare, geometry_for


def numpy_full_grid_oracle(prefix, prefix_valid, scores, valid, xy):
    result = []
    for indices, flags, ranking in zip(prefix, prefix_valid, scores):
        available = valid.copy()
        selected = list(indices[:2])
        for index, flag in zip(indices[:2], flags[:2]):
            if flag:
                available &= ((xy - xy[index]) ** 2).sum(-1) > 24**2
        for _ in range(2):
            if available.any():
                index = int(np.where(available, ranking, -np.inf).argmax())
                selected.append(index)
                available &= ((xy - xy[index]) ** 2).sum(-1) > 24**2
            else:
                selected.append(-1)
        result.append(selected)
    return np.array(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_pure_semantic_supplement.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    semantic = json.loads(Path(config["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    assert json.loads((root / "preregistered_protocol.json").read_text()) == config
    before = json.loads((root / "frozen_provenance_before.json").read_text())
    assert before == json.loads((root / "frozen_provenance_after.json").read_text())
    for path, digest in before.items():
        assert sha256_file(path) == digest, path
    verify_teacher_cache(Path(config["semantic_root"]) / "teacher", semantic)
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
            Path(config["semantic_root"]) / "training/semantic_head.pt",
            map_location="cpu",
            weights_only=False,
        )["head"]
    )
    head.requires_grad_(False).eval()
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent)
    views = frozen_views(semantic, train, val)
    checks = {}
    for split, view in zip(["training_monitor", "validation"], views):
        q = np.load(root / f"{split}_queries.npz")
        old = np.load(
            Path(config["semantic_root"])
            / "evaluation"
            / f"{split}_routing_queries.npz"
        )
        for key in ["sample_id", "family_id", "supervised"]:
            np.testing.assert_array_equal(q[key], old[key])
        for method, previous in [("structure", "A"), ("fusion", "C")]:
            for key in ["first", "nearest", "indices", "valid"]:
                np.testing.assert_array_equal(
                    q[f"{method}_{key}"], old[f"{previous}_{key}"]
                )
        keep = q["supervised"]
        a = q["structure_nearest"] <= 8
        summary = json.loads((root / f"{split}_summary.json").read_text())
        assert int(keep.sum()) == summary["effective_queries"] == 116953
        assert summary["original_misses"] == int((~a & keep).sum())
        for method in ["structure", "fusion", "semantic"]:
            d = q[f"{method}_nearest"]
            m = (d <= 8) & keep
            stored = summary["methods"][method]
            assert stored["new_hits"] == int((~a & m & keep).sum())
            assert stored["lost_hits"] == int((a & ~m & keep).sum())
            assert stored["net_hits"] == int(m.sum()) - int((a & keep).sum())
            if summary["original_misses"]:
                assert (
                    stored["rescue_rate"]
                    == stored["new_hits"] / summary["original_misses"]
                )
            indices = q[f"{method}_indices"]
            valid = q[f"{method}_valid"]
            assert (indices[~valid] == -1).all()
            assert q["target_valid"][
                np.arange(len(indices))[:, None, None], indices.clip(min=0)
            ][valid].all()
            np.testing.assert_array_equal(
                indices[..., :2], q["structure_indices"][..., :2]
            )
        families = list(csv.DictReader((root / f"{split}_per_family.csv").open()))
        for method in ["fusion", "semantic"]:
            for key in ["new_hits", "lost_hits", "hits"]:
                assert (
                    sum(int(r[f"{method}_{key}"]) for r in families)
                    == summary["methods"][method][key]
                )
        # Independently recompute every one of144 queries for the first4 samples
        # (fixed ordering, no performance filtering), using all1024 raw scores.
        data = TeacherView(view, Path(config["semantic_root"]) / "teacher")
        batch = prepare(
            next(iter(DataLoader(data, batch_size=4, shuffle=False))), device
        )
        with torch.no_grad():
            result = model(batch["optical"], batch["sar"])
            geom = geometry_for(model, batch, parent)
            p = head(result["sar_descriptors"])
            qo = probability_normalize(
                sample_native_map(batch["teacher"], model.source_pixel_xy)
            )
            score = torch.bmm(qo.transpose(1, 2), p.flatten(2)).cpu().numpy()
            original = spatial_candidates(
                result["scores"], geom["target_valid"], model.target_pixel_xy
            )
        for i in range(4):
            prefix = original["indices"][i].cpu().numpy()
            flags = original["valid"][i].cpu().numpy()
            valid = geom["target_valid"][i].cpu().numpy()
            xy = model.target_pixel_xy.cpu().numpy()
            predicted = numpy_full_grid_oracle(prefix, flags, score[i], valid, xy)
            np.testing.assert_array_equal(predicted, q["semantic_indices"][i])
        np.savez_compressed(
            root / f"{split}_full_grid_audit.npz",
            sample_id=np.array(batch["id"]),
            semantic_similarity=score,
            structure_prefix=original["indices"].cpu().numpy()[..., :2],
            target_valid=geom["target_valid"].cpu().numpy(),
            target_xy=xy,
            expected_semantic_indices=q["semantic_indices"][:4],
        )
        checks[split] = dict(
            exact_previous_structure_and_fusion=True,
            counts_and_rescue_denominators_verified=True,
            family_pairing_totals_verified=True,
            all_candidates_in_valid_target_grid=True,
            fixed_first_pairs=4,
            full_grid_numpy_oracle_queries=576,
            oracle_indices_exact=True,
        )
    save_json(
        root / "verification.json",
        dict(
            passed=True,
            no_training=True,
            source_and_weights_and_sealed_pool_hashes_unchanged=True,
            teacher_cache_verified=True,
            checks=checks,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    print(
        "Frozen controls, counts/rescue denominators and1152 full-grid NumPy oracle queries verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
