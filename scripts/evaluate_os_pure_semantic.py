"""No-training comparison:4 structure vs2+2 fusion vs2+2 full-grid semantics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, spatial_candidates
from sdgm.continuation_diagnostics import state_digest
from sdgm.family_evaluation import quantiles
from sdgm.io import save_csv, save_json
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_evaluation import candidate_arrays, frozen_views
from sdgm.semantic_routing import (
    SARPrototypeHead,
    probability_normalize,
    sample_native_map,
    routed_candidates,
)
from sdgm.semantic_supplement import (
    supplement_from_full_grid,
    paired_counts,
    paired_family_intervals,
)
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache
from scripts.train_os_semantic_routing import frozen_provenance
from scripts.train_structure_smoke import prepare, geometry_for


def array_summary(arrays, ids, methods):
    keep = arrays["supervised"][ids]
    n = int(keep.sum())
    a = arrays["structure_nearest"][ids]
    row = dict(effective_queries=n)
    row["structure_hits"] = int(((a <= 8) & keep).sum())
    row["structure_misses"] = n - row["structure_hits"]
    for m in methods:
        near = arrays[f"{m}_nearest"][ids]
        first = arrays[f"{m}_first"][ids]
        counts = paired_counts(a, near, keep)
        for key, value in counts.items():
            if key not in ["effective_queries", "original_hits", "original_misses"]:
                row[f"{m}_{key}"] = value
        row[f"{m}_recall_at_1"] = float((first[keep] <= 8).mean()) if n else None
        row[f"{m}_median_px"] = float(np.median(near[keep])) if n else None
        row[f"{m}_p95_px"] = float(np.percentile(near[keep], 95)) if n else None
    return row


@torch.no_grad()
def evaluate(
    model, head, dataset, parent, semantic_config, config, device, root, split, weight
):
    methods = ["structure", "fusion", "semantic"]
    storage = {
        f"{m}_{k}": []
        for m in methods
        for k in ["first", "nearest", "indices", "valid"]
    }
    sample_ids = []
    families = []
    keeps = []
    mask_counts = defaultdict(int)
    valid_targets = []
    for batch in DataLoader(
        dataset, batch_size=config["evaluation_batch_size"], shuffle=False
    ):
        batch = prepare(batch, device)
        out = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        ps = head(out["sar_descriptors"])
        qo = probability_normalize(
            sample_native_map(batch["teacher"], model.source_pixel_xy)
        )
        similarity = torch.bmm(qo.transpose(1, 2), ps.flatten(2))
        if similarity.shape != (len(batch["id"]), 144, 1024):
            raise AssertionError("Need every target grid location")
        if not torch.isfinite(similarity).all():
            raise FloatingPointError("Nonfinite semantic score")
        original = spatial_candidates(
            out["scores"],
            geom["target_valid"],
            model.target_pixel_xy,
            count=4,
            nms_radius_px=24,
        )
        pred = dict(
            structure=original,
            fusion=routed_candidates(
                out["scores"],
                2 * similarity - 1,
                geom["target_valid"],
                model.target_pixel_xy,
                weight,
            ),
            semantic=supplement_from_full_grid(
                original, similarity, geom["target_valid"], model.target_pixel_xy
            ),
        )
        # Same helper must reproduce existing sequential2+2 fusion selection.
        reference = supplement_from_full_grid(
            original,
            out["scores"] + weight * (2 * similarity - 1),
            geom["target_valid"],
            model.target_pixel_xy,
        )
        for key in reference:
            assert torch.equal(reference[key], pred["fusion"][key]), key
        for method in methods:
            p = pred[method]
            assert torch.equal(p["indices"][..., :2], original["indices"][..., :2])
            available = geom["target_valid"][:, None].expand(-1, 144, -1)
            assert available.gather(-1, p["indices"].clamp_min(0))[p["valid"]].all()
            for i in range(4):
                for j in range(i):
                    both = p["valid"][..., i] & p["valid"][..., j]
                    distance = (
                        (p["pixel_xy"][..., i, :] - p["pixel_xy"][..., j, :]) ** 2
                    ).sum(-1)
                    assert (distance[both] > 24**2).all()
            first, near = candidate_arrays(p, geom)
            assert torch.isfinite(near[geom["supervised"]]).all()
            for key, value in [
                ("first", first),
                ("nearest", near),
                ("indices", p["indices"]),
                ("valid", p["valid"]),
            ]:
                storage[f"{method}_{key}"].extend(value.cpu().numpy())
        sample_ids.extend(batch["id"])
        families.extend(batch["family_id"])
        keeps.extend(geom["supervised"].cpu().numpy())
        valid_targets.extend(geom["target_valid"].cpu().numpy())
        for k in [
            "gt_outside",
            "empty_positive",
            "invalid_source",
            "gt_invalid_support",
        ]:
            mask_counts[k] += int(geom[k].sum())
    arrays = {k: np.stack(v) for k, v in storage.items()}
    arrays.update(
        sample_id=np.array(sample_ids),
        family_id=np.array(families),
        supervised=np.stack(keeps),
        target_valid=np.stack(valid_targets),
    )
    previous = np.load(
        Path(config["semantic_root"]) / "evaluation" / f"{split}_routing_queries.npz"
    )
    for k in ["sample_id", "family_id", "supervised"]:
        np.testing.assert_array_equal(arrays[k], previous[k])
    for method, old in [("structure", "A"), ("fusion", "C")]:
        for k in ["first", "nearest", "indices", "valid"]:
            np.testing.assert_array_equal(
                arrays[f"{method}_{k}"], previous[f"{old}_{k}"]
            )
    keep = arrays["supervised"]
    summary = dict(
        split=split,
        pairs=len(sample_ids),
        families=len(set(families)),
        effective_queries=int(keep.sum()),
        masked=dict(mask_counts),
        methods={},
        baseline_and_fusion_exactly_reproduced=True,
        semantic_search="all1024 positions, existing target_valid only then sequential24px union NMS; no structural prescreening",
        no_training=True,
        OS_test_evaluated=False,
        UBC_evaluated=False,
    )
    counts = array_summary(arrays, np.arange(len(sample_ids)), methods)
    for method in methods:
        summary["methods"][method] = {
            k[len(method) + 1 :]: v
            for k, v in counts.items()
            if k.startswith(method + "_")
        }
        summary["methods"][method]["nearest_distance_px"] = quantiles(
            arrays[f"{method}_nearest"][keep]
        )
        summary["methods"][method]["invalid_candidate_slots"] = int(
            (~arrays[f"{method}_valid"]).sum()
        )
    summary["original_hits"] = counts["structure_hits"]
    summary["original_misses"] = counts["structure_misses"]
    pair_rows = [
        dict(
            sample_id=sample,
            family_id=families[i],
            **array_summary(arrays, [i], methods),
        )
        for i, sample in enumerate(sample_ids)
    ]
    family_rows = []
    for family in sorted(set(families)):
        ids = np.flatnonzero(np.array(families) == family)
        row = dict(
            family_id=family, pairs=len(ids), **array_summary(arrays, ids, methods)
        )
        row["semantic_minus_fusion_recall4"] = (
            row["semantic_hits"] - row["fusion_hits"]
        ) / row["effective_queries"]
        family_rows.append(row)
    summary["family_changes"] = {
        method: dict(
            improved=sum(r[f"{method}_net_hits"] > 0 for r in family_rows),
            degraded=sum(r[f"{method}_net_hits"] < 0 for r in family_rows),
            unchanged=sum(r[f"{method}_net_hits"] == 0 for r in family_rows),
            delta_recall4=quantiles(
                [r[f"{method}_delta_recall4"] for r in family_rows]
            ),
        )
        for method in ["fusion", "semantic"]
    }
    summary["family_intervals"] = paired_family_intervals(
        family_rows, config["bootstrap"]["seed"], config["bootstrap"]["replicates"]
    )
    # Actual candidate set difference excludes permutations of ranks3/4.
    for method in ["fusion", "semantic"]:
        changed = (
            np.sort(arrays[f"{method}_indices"], axis=-1)
            != np.sort(arrays["structure_indices"], axis=-1)
        ).any(-1) & keep
        summary["methods"][method]["changed_candidate_set_queries"] = int(changed.sum())
        assert counts[f"{method}_net_hits"] == sum(
            r[f"{method}_hits"] - r["structure_hits"] for r in family_rows
        )
    save_json(root / f"{split}_summary.json", summary)
    save_csv(root / f"{split}_per_pair.csv", pair_rows)
    save_csv(root / f"{split}_per_family.csv", family_rows)
    np.savez_compressed(root / f"{split}_queries.npz", **arrays)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_pure_semantic_supplement.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    semantic = json.loads(Path(config["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError("New frozen comparison output required")
    assert config["evaluation_batch_size"] == parent["batch_size"] == 4
    assert config["candidate_count"] == parent["candidate_count"] == 4
    assert config["recall_radius_px"] == parent["recall_radius_px"] == 8
    assert config["nms_radius_px"] == parent["nms_radius_px"] == 24
    head_path = Path(config["semantic_root"]) / "training/semantic_head.pt"
    if sha256_file(head_path) != config["semantic_head_sha256"]:
        raise ValueError("Existing final semantic head required")
    if (
        sha256_file(semantic["structural_checkpoint"])
        != semantic["structural_checkpoint_sha256"]
    ):
        raise ValueError("Existing epoch30 structure required")
    receipt = verify_teacher_cache(Path(config["semantic_root"]) / "teacher", semantic)
    verify_freeze(semantic["family_root"])
    before = frozen_provenance(semantic)
    for path in [
        head_path,
        Path(config["semantic_root"]) / "routing_calibration.json",
        Path(config["semantic_root"]) / "teacher/cache_receipt.json",
    ]:
        before[str(path)] = sha256_file(path)
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
    saved = torch.load(head_path, map_location="cpu", weights_only=False)
    head.load_state_dict(saved["head"])
    head.requires_grad_(False).eval()
    assert (
        saved["teacher_cache_version"] == receipt["cache_version"]
        and saved["step"] == 5000
    )
    digests = dict(
        structure=state_digest(model.state_dict()),
        semantic_head=state_digest(head.state_dict()),
    )
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(semantic["family_root"], "val", parent, training=False)
    views = frozen_views(semantic, train, val)
    weight = json.loads(
        (Path(config["semantic_root"]) / "routing_calibration.json").read_text()
    )["lambda_value"]
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for split, view in zip(["training_monitor", "validation"], views):
        result = evaluate(
            model,
            head,
            TeacherView(view, Path(config["semantic_root"]) / "teacher"),
            parent,
            semantic,
            config,
            device,
            root,
            split,
            weight,
        )
        print(json.dumps(result), flush=True)
    assert digests == dict(
        structure=state_digest(model.state_dict()),
        semantic_head=state_digest(head.state_dict()),
    )
    after = {path: sha256_file(path) for path in before}
    assert before == after
    save_json(root / "frozen_provenance_after.json", after)
    save_json(
        root / "completion.json",
        dict(
            completed=True,
            no_training=True,
            optimizer_steps_added=0,
            frozen_model_digests=digests,
            all_source_and_pool_hashes_unchanged=True,
            full_valid_grid_search=True,
            existing_structure_and_fusion_outputs_exact=True,
            OS_test_evaluated=False,
            UBC_evaluated=False,
            semantic_cache_version=receipt["cache_version"],
            fusion_lambda_unchanged=weight,
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        ),
    )


if __name__ == "__main__":
    main()
