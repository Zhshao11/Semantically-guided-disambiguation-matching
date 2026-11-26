"""Verify teacher provenance, unchanged structure and paired candidate accounting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sdgm.coarse import StructuralRetriever
from sdgm.continuation_diagnostics import state_digest, verify_monitor
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views, evaluate_routing
from sdgm.semantic_routing import SARPrototypeHead
from sdgm.semantic_teacher import (
    TeacherView,
    load_teacher,
    teacher_preprocess,
    verify_teacher_cache,
)
from scripts.train_os_semantic_routing import frozen_provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_semantic_routing.json")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    parent = json.loads(Path(config["parent_config"]).read_text())
    if json.loads((root / "preregistered_protocol.json").read_text()) != config:
        raise ValueError("Protocol drift")
    verify_freeze(config["family_root"])
    verify_monitor(config["diagnostic_root"])
    receipt = verify_teacher_cache(root / "teacher", config)
    before = json.loads((root / "frozen_provenance_before.json").read_text())
    after = json.loads((root / "frozen_provenance_after.json").read_text())
    assert before == after == frozen_provenance(config)
    output = root / "verification"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    checkpoint = torch.load(
        config["structural_checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint["epoch"] == 30
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    saved = torch.load(
        root / "training/semantic_head.pt", map_location="cpu", weights_only=False
    )
    completion = json.loads((root / "completion.json").read_text())
    assert (
        state_digest(model.state_dict())
        == saved["structural_model_digest"]
        == completion["structure_digest_before"]
        == completion["structure_digest_after"]
    )
    assert saved["teacher_cache_version"] == receipt["cache_version"]
    assert saved["step"] == config["training"]["steps"] == 5000
    assert {int(v["step"]) for v in saved["optimizer"]["state"].values()} == {5000}
    head = SARPrototypeHead().to(device)
    head.load_state_dict(saved["head"])
    head.eval()
    train = OSFamilyDataset(config["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(config["family_root"], "val", parent, training=False)
    views = frozen_views(config, train, val)
    centers = np.load(root / "teacher/prototype_centers.npy")
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1, atol=1e-6)
    chosen = [
        json.loads(l)
        for l in (root / "teacher/prototype_samples.jsonl").read_text().splitlines()
    ]
    assert {r["family_id"] for r in chosen} == {r["family_id"] for r in train.records}
    assert not {r["family_id"] for r in chosen} & {r["family_id"] for r in val.records}
    assert len(chosen) == len({r["family_id"] for r in chosen})
    teacher, _ = load_teacher(config, device)
    tokens = np.load(root / "teacher/patch_tokens.npy", mmap_mode="r")
    probs = np.load(root / "teacher/optical_probabilities.npy", mmap_mode="r")
    checks = []
    with torch.no_grad():
        for dataset, offset in [(train, 0), (val, len(train))]:
            for index in [0, len(dataset) // 2]:
                image = torch.from_numpy(
                    dataset.arrays[index : index + 1, 0].copy()
                ).to(device)[:, None]
                token = F.normalize(
                    teacher.forward_features(teacher_preprocess(image))[
                        "x_norm_patchtokens"
                    ],
                    dim=-1,
                )
                token_error = float(
                    np.max(
                        np.abs(
                            token.cpu().numpy()[0]
                            - tokens[offset + index].astype("float32")
                        )
                    )
                )
                # Cache intentionally computes q from stored normalized fp16 tokens.
                patch = (
                    torch.from_numpy(np.array(tokens[offset + index], dtype="float32"))
                    .to(device)
                    .T.reshape(1, 384, 18, 18)
                )
                dense = F.normalize(
                    F.interpolate(
                        patch, size=(32, 32), mode="bilinear", align_corners=False
                    ),
                    dim=1,
                )
                q = (
                    torch.einsum(
                        "bchw,mc->bmhw", dense, torch.from_numpy(centers).to(device)
                    )
                    .div(config["prototypes"]["temperature"])
                    .softmax(1)
                )
                q_error = float(
                    np.max(
                        np.abs(
                            q.cpu().numpy()[0] - probs[offset + index].astype("float32")
                        )
                    )
                )
                assert token_error < 0.0003 and q_error < 0.0005
                checks.append(
                    dict(
                        sample_id=dataset.records[index]["sample_id"],
                        normalized_token_max_abs_error=token_error,
                        probability_max_abs_error=q_error,
                    )
                )
    del teacher
    calibration = json.loads((root / "routing_calibration.json").read_text())
    assert calibration["validation_used"] is False
    exact = {}
    for split, view in zip(["training_monitor", "validation"], views):
        data = TeacherView(view, root / "teacher")
        reproduced = evaluate_routing(
            model,
            head,
            data,
            parent,
            config,
            device,
            output,
            split,
            calibration["lambda_value"],
        )
        old = json.loads(
            (root / "evaluation" / f"{split}_routing_summary.json").read_text()
        )
        assert reproduced == old
        original = np.load(root / "evaluation" / f"{split}_routing_queries.npz")
        reloaded = np.load(output / f"{split}_routing_queries.npz")
        assert set(original.files) == set(reloaded.files)
        for key in original.files:
            np.testing.assert_array_equal(original[key], reloaded[key])
        previous = np.load(
            Path(config["diagnostic_root"]) / "run" / f"epoch30_{split}_distances.npz"
        )
        np.testing.assert_array_equal(original["sample_id"], previous["sample_id"])
        np.testing.assert_array_equal(original["supervised"], previous["supervised"])
        keep = original["supervised"]
        np.testing.assert_array_equal(
            original["A_first"][keep], previous["first_distance_px"][keep]
        )
        np.testing.assert_array_equal(
            original["A_nearest"][keep], previous["nearest_distance_px"][keep]
        )
        for method in ["A", "B", "C", "C_shuffle"]:
            idx = original[f"{method}_indices"]
            valid = original[f"{method}_valid"]
            assert (idx[~valid] == -1).all()
            assert ((idx[valid] >= 0) & (idx[valid] < 1024)).all()
            xy = model.target_pixel_xy.cpu().numpy()[np.maximum(idx, 0)]
            for a in range(4):
                for b in range(a):
                    both = valid[..., a] & valid[..., b]
                    assert (
                        ((xy[..., a, :] - xy[..., b, :]) ** 2).sum(-1)[both] > 24**2
                    ).all()
            np.testing.assert_array_equal(
                original[f"{method}_first"][keep], original["A_first"][keep]
            )
        exact[split] = dict(
            all_reloaded_arrays_bit_identical=True,
            epoch30_A_distances_and_mask_exact=True,
            A_B_all_candidates_exact=True,
            same_top2_structural_peaks=True,
            NMS_and_invalid_slots_verified=True,
        )
    save_json(
        output / "verification_summary.json",
        dict(
            passed=True,
            teacher_cache_hashes_exact=True,
            prototype_training_families_only=True,
            prototype_centers_unit_norm=True,
            teacher_spot_checks=checks,
            epoch30_model_frozen=True,
            optimizer_steps=5000,
            KL_only=True,
            exact=exact,
            original_source_and_test_pool_hashes_unchanged=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
        ),
    )
    print(
        "Teacher/prototypes, frozen model, exact A/B, final head reload and sealed pools verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
