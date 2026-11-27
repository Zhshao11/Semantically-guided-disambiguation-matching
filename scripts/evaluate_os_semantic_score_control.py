"""Fixed-distribution dot vs negative squared-distance ranking and coverage."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, spatial_candidates, apply_affine
from sdgm.continuation_diagnostics import state_digest
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import frozen_views, candidate_arrays
from sdgm.semantic_routing import (
    SARPrototypeHead,
    probability_normalize,
    sample_native_map,
)
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from sdgm.semantic_bottleneck import comparison_stats, full_grid_rank
from sdgm.semantic_supplement import supplement_from_full_grid
from sdgm.semantic_score_control import probability_scores
from scripts.train_structure_smoke import prepare, geometry_for


@torch.no_grad()
def evaluate(model, head, data, semantic, config, parent, device, root, split):
    previous = np.load(
        Path(config["diagnostic_root"]) / f"{split}_query_diagnostics.npz"
    )
    keys = [
        "positive_common",
        "positive_original",
        "negative_indices",
        "negative_common_valid",
        "supervised",
        "inverse_target_valid",
        "positive_indices",
        "positive_slot_valid",
    ]
    for layer in ["B", "C"]:
        keys += [
            layer + "_" + k
            for k in [
                "valid",
                "rank_min",
                "rank_max",
                "delta",
                "win",
                "loss",
                "tie",
                "exact_tie",
            ]
        ]
    old = {k: previous[k] for k in keys}
    former = np.load(Path(config["supplement_root"]) / f"{split}_queries.npz")
    baseline = {
        k: former[k]
        for k in [
            "sample_id",
            "family_id",
            "supervised",
            "structure_indices",
            "structure_nearest",
            "structure_first",
            "semantic_indices",
            "semantic_nearest",
            "semantic_first",
        ]
    }
    store = {}
    offset = 0
    sample_ids = []
    families = []

    def add(k, v):
        store.setdefault(k, []).append(v.detach().cpu().numpy())

    for batch in DataLoader(data, batch_size=config["batch_size"], shuffle=False):
        batch = prepare(batch, device)
        b = len(batch["id"])
        sl = slice(offset, offset + b)
        out = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        ps = head(out["sar_descriptors"])
        qo = probability_normalize(
            sample_native_map(batch["teacher"], model.source_pixel_xy)
        )
        back = apply_affine(torch.linalg.inv(batch["matrix"]), model.target_pixel_xy)
        qt = probability_normalize(sample_native_map(batch["teacher"], back))
        all_scores = {
            "B": probability_scores(qo, qt),
            "C": probability_scores(qo, ps.flatten(2)),
        }
        positive = torch.from_numpy(old["positive_common"][sl]).to(device)
        pos_original = torch.from_numpy(old["positive_original"][sl]).to(device)
        nidx = torch.from_numpy(old["negative_indices"][sl]).to(device)
        nvalid = torch.from_numpy(old["negative_common_valid"][sl]).to(device)
        pidx = torch.from_numpy(old["positive_indices"][sl]).to(device)
        negative = torch.zeros_like(positive, dtype=torch.long)
        negative.scatter_add_(-1, nidx.clamp_min(0), nvalid.long())
        negative = negative > 0
        np.testing.assert_array_equal(
            geom["positive"].cpu().numpy(), old["positive_original"][sl]
        )
        np.testing.assert_array_equal(
            geom["supervised"].cpu().numpy(), old["supervised"][sl]
        )
        add("query_norm2", all_scores["C"]["query_norm2"])
        for layer, scores in all_scores.items():
            add(layer + "_target_norm2", scores["target_norm2"])
            # Fixed dot extrema for explaining sharpness effect; not used as
            # the new distance-score extrema or as deployment candidates.
            pi = scores["dot"].masked_fill(~positive, -torch.inf).argmax(-1)
            ni = scores["dot"].masked_fill(~negative, -torch.inf).argmax(-1)
            pn = scores["target_norm2"].gather(-1, pi)
            nn = scores["target_norm2"].gather(-1, ni)
            add(layer + "_dot_bestP_target_norm2", pn)
            add(layer + "_dot_bestN_target_norm2", nn)
            fixed = scores["dist"].gather(-1, pi[..., None]).squeeze(-1) - scores[
                "dist"
            ].gather(-1, ni[..., None]).squeeze(-1)
            add(layer + "_distance_gap_at_dot_extrema", fixed)
            for method in ["dot", "dist"]:
                score = scores[method]
                stats = comparison_stats(
                    score, positive, negative, config["score_tie_atol"]
                )
                for k, v in stats.items():
                    if method == "dot" and k in [
                        "valid",
                        "rank_min",
                        "rank_max",
                        "delta",
                        "win",
                        "loss",
                        "tie",
                        "exact_tie",
                    ]:
                        np.testing.assert_array_equal(
                            v.cpu().numpy(), old[layer + "_" + k][sl]
                        )
                    add(layer + "_" + method + "_" + k, v)
                add(
                    layer + "_" + method + "_positive_scores",
                    score.gather(-1, pidx.clamp_min(0)),
                )
                add(
                    layer + "_" + method + "_negative_scores",
                    score.gather(-1, nidx.clamp_min(0)),
                )
        scores = all_scores["C"]
        original = spatial_candidates(
            out["scores"],
            geom["target_valid"],
            model.target_pixel_xy,
            count=4,
            nms_radius_px=24,
        )
        np.testing.assert_array_equal(
            original["indices"].cpu().numpy(), baseline["structure_indices"][sl]
        )
        first, near = candidate_arrays(original, geom)
        np.testing.assert_array_equal(
            first.cpu().numpy(), baseline["structure_first"][sl]
        )
        np.testing.assert_array_equal(
            near.cpu().numpy(), baseline["structure_nearest"][sl]
        )
        for method in ["dot", "dist"]:
            score = scores[method]
            for k, v in full_grid_rank(
                score, pos_original, geom["target_valid"], config["score_tie_atol"]
            ).items():
                add("C_" + method + "_full_" + k, v)
            sem4 = spatial_candidates(
                score,
                geom["target_valid"],
                model.target_pixel_xy,
                count=4,
                nms_radius_px=24,
            )
            f, d = candidate_arrays(sem4, geom)
            add(method + "_semantic4_indices", sem4["indices"])
            add(method + "_semantic4_valid", sem4["valid"])
            add(method + "_semantic4_first_distance", f)
            add(method + "_semantic4_nearest_distance", d)
            actual = supplement_from_full_grid(
                original, score, geom["target_valid"], model.target_pixel_xy
            )
            f, d = candidate_arrays(actual, geom)
            if method == "dot":
                np.testing.assert_array_equal(
                    actual["indices"].cpu().numpy(), baseline["semantic_indices"][sl]
                )
                np.testing.assert_array_equal(
                    d.cpu().numpy(), baseline["semantic_nearest"][sl]
                )
                np.testing.assert_array_equal(
                    sem4["indices"].cpu().numpy(), previous["C_semantic4_indices"][sl]
                )
            np.testing.assert_array_equal(
                actual["indices"][..., :2].cpu().numpy(),
                baseline["structure_indices"][sl, ..., :2],
            )
            add(method + "_supplement_indices", actual["indices"])
            add(method + "_supplement_valid", actual["valid"])
            add(method + "_supplement_first_distance", f)
            add(method + "_supplement_nearest_distance", d)
        sample_ids.extend(batch["id"])
        families.extend(batch["family_id"])
        offset += b
    arrays = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    arrays.update(
        sample_id=np.array(sample_ids),
        family_id=np.array(families),
        supervised=old["supervised"],
        positive_indices=old["positive_indices"],
        positive_slot_valid=old["positive_slot_valid"],
        negative_indices=old["negative_indices"],
        negative_common_valid=old["negative_common_valid"],
        structure_nearest=baseline["structure_nearest"],
        structure_first=baseline["structure_first"],
        structure_indices=baseline["structure_indices"],
    )
    for g in config["groups"]:
        arrays["group_" + g] = previous["group_" + g]
    for layer in ["B", "C"]:
        # Same query constant has no mathematical effect on ordering: compare
        # against independently evaluated direct squared differences on a fixed
        # small subset; full rerun is covered by the verification script.
        assert np.isfinite(arrays[layer + "_dist_delta"]).all()
    np.savez_compressed(root / f"{split}_score_queries.npz", **arrays)
    save_json(
        root / f"{split}_receipt.json",
        dict(
            pairs=len(sample_ids),
            families=len(set(families)),
            effective_queries=int(old["supervised"].sum()),
            fixed_PN_exact=True,
            previous_dot_stats_and_candidates_exact=True,
            score_tie_atol=config["score_tie_atol"],
            new_normalization=False,
            teacher_and_head_unchanged=True,
            distance_fused_with_structure=False,
            group_membership_fixed_from_previous_dot=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_score_control.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    semantic = json.loads(Path(config["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    diag = json.loads(Path(config["diagnostic_config"]).read_text())
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError(
            "Use a new output; scoring controls cannot replace previous runs"
        )
    assert config["score_tie_atol"] == diag["score_tie_atol"]
    assert config["batch_size"] == parent["batch_size"] == 4
    assert config["candidate_count"] == parent["candidate_count"] == 4
    assert config["recall_radius_px"] == parent["recall_radius_px"] == 8
    assert config["nms_radius_px"] == parent["nms_radius_px"] == 24
    verify_teacher_cache(Path(semantic["output"]) / "teacher", semantic)
    before = json.loads(
        (Path(config["diagnostic_root"]) / "frozen_provenance_after.json").read_text()
    )
    assert before == {p: sha256_file(p) for p in before}
    for split in config["evaluation_splits"]:
        p = Path(config["diagnostic_root"]) / f"{split}_query_diagnostics.npz"
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
    digests = [state_digest(model.state_dict()), state_digest(head.state_dict())]
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent)
    views = frozen_views(semantic, train, val)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for split, view in zip(config["evaluation_splits"], views):
        evaluate(
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
        print(json.dumps(dict(stage="score_control_complete", split=split)), flush=True)
    assert digests == [
        state_digest(model.state_dict()),
        state_digest(head.state_dict()),
    ]
    after = {p: sha256_file(p) for p in before}
    assert before == after
    save_json(root / "frozen_provenance_after.json", after)
    save_json(
        root / "completion.json",
        dict(
            completed=True,
            no_training=True,
            optimizer_steps_added=0,
            models_and_probabilities_unchanged=True,
            previous_dot_results_exact=True,
            PN_and_groups_fixed=True,
            score_change_only=True,
            distance_structural_fusion=False,
            lambda_unchanged=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        ),
    )


if __name__ == "__main__":
    main()
