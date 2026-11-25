"""Fit 32 fixed training pairs and report engineering retrieval diagnostics."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import (
    StructuralRetriever,
    neighborhood_mass_loss,
    spatial_candidates,
    supervision_geometry,
)
from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.smoke_data import WHUSmokeDataset


def prepare(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def geometry_for(model, batch, config):
    return supervision_geometry(
        batch["matrix"],
        model.source_pixel_xy,
        model.target_pixel_xy,
        config["positive_radius_px"],
    )


def candidate_metrics(candidates, geometry, radius):
    distance = torch.linalg.vector_norm(
        candidates["pixel_xy"] - geometry["gt_pixel_xy"][:, :, None], dim=-1
    )
    distance = distance.masked_fill(~candidates["valid"], float("inf"))
    keep = geometry["supervised"]
    closest = distance.min(-1).values
    return dict(
        count=int(keep.sum()),
        recall1_correct=int(((distance[..., 0] <= radius) & keep).sum()),
        recall4_correct=int(((closest <= radius) & keep).sum()),
        closest_distance=closest[keep].cpu().tolist(),
        first_distance=distance[..., 0][keep].cpu().tolist(),
    )


def visualize_batch(batch, model, candidates, geometry, path):
    optical = (
        ((batch["optical"][0, 0].cpu().numpy() + 1) * 127.5)
        .clip(0, 255)
        .astype("uint8")
    )
    sar = ((batch["sar"][0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
    canvas = np.concatenate(
        [
            cv2.cvtColor(optical, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sar, cv2.COLOR_GRAY2BGR),
        ],
        axis=1,
    )
    valid = torch.where(geometry["supervised"][0])[0].cpu().numpy()
    picks = (
        valid[np.linspace(0, len(valid) - 1, min(18, len(valid))).round().astype(int)]
        if len(valid)
        else []
    )
    source = model.source_pixel_xy.cpu().numpy()
    target = geometry["gt_pixel_xy"][0].cpu().numpy()
    prediction = candidates["pixel_xy"][0].cpu().numpy()
    for index in picks:
        origin = tuple(np.round(source[index]).astype(int))
        gt = tuple(np.round(target[index] + [512, 0]).astype(int))
        cv2.circle(canvas, origin, 3, (255, 200, 0), -1)
        cv2.drawMarker(canvas, gt, (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        for rank in range(4):
            if not candidates["valid"][0, index, rank]:
                continue
            point = tuple(np.round(prediction[index, rank] + [512, 0]).astype(int))
            color = (0, 0, 255) if rank == 0 else (0, 150, 255)
            cv2.circle(canvas, point, 4 if rank == 0 else 2, color, 1)
            if rank == 0:
                cv2.line(canvas, origin, point, (160, 160, 160), 1)
    strip = np.full((40, 1024, 3), 255, np.uint8)
    cv2.putText(
        strip,
        "Engineering only | sources cyan | label green | top1 red | other candidates orange",
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate([strip, canvas]))


@torch.no_grad()
def evaluate(model, loader, config, device, output, tag):
    model.eval()
    rows = []
    loss_total = 0
    supervised_total = 0
    distances = []
    masked = dict(gt_outside=0, empty_positive=0)
    for index, batch in enumerate(loader):
        batch = prepare(batch, device)
        result = model(batch["optical"], batch["sar"])
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite evaluation output")
        geometry = geometry_for(model, batch, config)
        candidates = spatial_candidates(
            result["scores"],
            geometry["target_valid"],
            model.target_pixel_xy,
            nms_radius_px=config["nms_radius_px"],
        )
        count = int(geometry["supervised"].sum())
        loss = neighborhood_mass_loss(result["scores"], geometry)
        loss_total += float(loss) * count
        supervised_total += count
        for key in masked:
            masked[key] += int(geometry[key].sum())
        for item, sample in enumerate(batch["id"]):
            one_candidates = {
                key: value[item : item + 1] for key, value in candidates.items()
            }
            one_geometry = {
                key: value[item : item + 1] for key, value in geometry.items()
            }
            metrics = candidate_metrics(
                one_candidates, one_geometry, config["recall_radius_px"]
            )
            distances.extend(metrics.pop("closest_distance"))
            first_distance = metrics.pop("first_distance")
            rows.append(
                dict(
                    id=sample,
                    **metrics,
                    first_distance_median=(
                        float(np.median(first_distance)) if first_distance else None
                    ),
                )
            )
        if index < 4:
            visualize_batch(
                batch,
                model,
                candidates,
                geometry,
                output / "visualizations" / f"{tag}_{batch['id'][0]}.png",
            )
    total = sum(row["count"] for row in rows)
    summary = dict(
        tag=tag,
        supervised_queries=total,
        loss=loss_total / max(supervised_total, 1),
        recall_at_1=sum(row["recall1_correct"] for row in rows) / max(total, 1),
        recall_at_4=sum(row["recall4_correct"] for row in rows) / max(total, 1),
        recall_radius_px=config["recall_radius_px"],
        masked=masked,
        nearest_candidate_distance_px=(
            {
                str(q): float(np.percentile(distances, q))
                for q in [0, 25, 50, 75, 90, 95, 100]
            }
            if distances
            else {}
        ),
        all_reported_distances_finite=bool(np.isfinite(distances).all()),
        purpose="engineering_only_relative_to_derivative_labels",
    )
    save_json(output / f"{tag}_metrics.json", summary)
    save_jsonl(output / f"{tag}_per_sample.jsonl", rows)
    save_json(output / f"{tag}_nearest_distances.json", distances)
    return summary


def gradient_summary(model):
    groups = {}
    for name, module in [
        ("optical_adapter", model.optical_adapter),
        ("sar_adapter", model.sar_adapter),
        ("shared_encoder", model.encoder),
    ]:
        gradients = [
            p.grad
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        finite = all(torch.isfinite(g).all().item() for g in gradients)
        norm = math.sqrt(
            sum(float((g.detach().double() ** 2).sum()) for g in gradients)
        )
        groups[name] = dict(
            finite=finite,
            l2_norm=norm,
            parameters_with_grad=len(gradients),
            parameters_with_nonzero_grad=sum(
                bool(torch.count_nonzero(g)) for g in gradients
            ),
        )
    return groups


def benchmark(model, batch, geometry, optimizer, device):
    measurements = {}
    for mode in ["forward", "training"]:
        model.train(mode == "training")

        def operation():
            if mode == "forward":
                with torch.no_grad():
                    model(batch["optical"], batch["sar"])
            else:
                optimizer.zero_grad(set_to_none=True)
                result = model(batch["optical"], batch["sar"])
                neighborhood_mass_loss(result["scores"], geometry).backward()
                # Optimizer is intentionally excluded to leave benchmark weights fixed.

        for _ in range(3):
            operation()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(10):
            operation()
        torch.cuda.synchronize(device)
        measurements[mode] = dict(
            batch_wall_ms=(time.perf_counter() - start) * 1000 / 10,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            batch_size=batch["optical"].shape[0],
            iterations=10,
            includes_data_loading=False,
            includes_nms=False,
            includes_optimizer_step=False,
            operation="forward" if mode == "forward" else "forward_loss_backward",
        )
    optimizer.zero_grad(set_to_none=True)
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/structure_smoke.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("training.jsonl")):
        raise FileExistsError("Use a fresh output directory for a distinct smoke run")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the requested GPU profile")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    dataset = WHUSmokeDataset(
        config["data_root"],
        config["pairs"],
        seed,
        output / "fixed_training_subset.json",
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    evaluation = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    model = StructuralRetriever(config["temperature"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    save_json(output / "config.json", config)
    save_json(
        output / "environment.json",
        dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device),
            device=str(device),
            parameters=sum(p.numel() for p in model.parameters()),
            feature_shape=[config["batch_size"], 128, 64, 64],
            score_shape=[config["batch_size"], 576, 4096],
            extra_geometric_augmentation=False,
            test_split_used=False,
            scene_independence_verified=False,
            positive_neighborhood="Fixed Euclidean radius 8 original-image pixels, one coarse cell. No radius tuning.",
            benchmark_scope="GPU-resident input tensors; forward and forward/loss/backward; optimizer excluded from timed benchmark",
        ),
    )
    baseline = evaluate(model, evaluation, config, device, output, "initial")
    print(json.dumps(baseline), flush=True)
    sample = prepare(next(iter(evaluation)), device)
    geometry = geometry_for(model, sample, config)
    timing = benchmark(model, sample, geometry, optimizer, device)
    history = []
    gradient_records = []
    iterator = iter(loader)
    skipped = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    training_start = time.perf_counter()
    with (output / "training.jsonl").open("w") as stream:
        for step in range(1, config["steps"] + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            model.train()
            batch = prepare(batch, device)
            geometry = geometry_for(model, batch, config)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["optical"], batch["sar"])
            if not all(torch.isfinite(value).all() for value in result.values()):
                raise FloatingPointError(f"Nonfinite output at step {step}")
            loss = neighborhood_mass_loss(result["scores"], geometry)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            if not geometry["supervised"].any():
                skipped += 1
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            if step in [1, config["steps"]]:
                gradients = gradient_summary(model)
                gradient_records.append(dict(step=step, groups=gradients))
                if not all(
                    g["finite"] and g["l2_norm"] > 0 for g in gradients.values()
                ):
                    raise RuntimeError("Missing effective adapter/encoder gradient")
            optimizer.step()
            row = dict(
                step=step,
                loss=float(loss),
                gradient_norm_before_clipping=float(norm),
                supervised_queries=int(geometry["supervised"].sum()),
                nonfinite_detected=False,
            )
            history.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if step % 25 == 0:
                print(json.dumps(row), flush=True)
            if step % config["evaluate_every"] == 0 and step < config["steps"]:
                metrics = evaluate(
                    model, evaluation, config, device, output, f"step_{step:04d}"
                )
                print(json.dumps(metrics), flush=True)
    torch.cuda.synchronize(device)
    timing["actual_training_loop"] = dict(
        wall_seconds=time.perf_counter() - training_start,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
        includes_periodic_evaluation=True,
        includes_optimizer_step=True,
        skipped_unsupervised_batches=skipped,
    )
    final = evaluate(model, evaluation, config, device, output, "final")
    acceptance = dict(
        finite_forward_backward=True,
        effective_gradients=all(
            g["l2_norm"] > 0 and g["finite"]
            for r in gradient_records
            for g in r["groups"].values()
        ),
        loss_decreased_by_half=final["loss"] < baseline["loss"] * 0.5,
        recall4_improved=final["recall_at_4"] > baseline["recall_at_4"],
        median_distance_reduced=final["nearest_candidate_distance_px"]["50"]
        < baseline["nearest_candidate_distance_px"]["50"],
        engineering_only=True,
    )
    save_json(output / "acceptance.json", acceptance)
    save_json(output / "gradient_checks.json", gradient_records)
    save_json(output / "performance.json", timing)
    save_csv(output / "training.csv", history)
    torch.save(
        dict(
            model=model.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
            step=config["steps"],
            purpose="engineering_only_relative_to_derivative_labels",
        ),
        output / "last.pt",
    )
    print(
        json.dumps(dict(final=final, acceptance=acceptance, performance=timing)),
        flush=True,
    )
    return (
        0
        if all(value for key, value in acceptance.items() if key != "engineering_only")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
