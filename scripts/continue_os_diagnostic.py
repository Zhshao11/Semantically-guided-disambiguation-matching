"""Continue epoch20 model and AdamW for ten epochs under matched fixed diagnostics."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, neighborhood_mass_loss
from sdgm.continuation_diagnostics import (
    state_digest,
    replay_sampler,
    freeze_monitor,
    verify_monitor,
    FrozenDiagnosticView,
    aggregate_families,
)
from sdgm.family_evaluation import evaluate_family
from sdgm.fixed_failure_diagnostics import failure_diagnostics
from sdgm.io import save_json, save_jsonl, save_csv
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file
from scripts.train_structure_smoke import prepare, geometry_for, gradient_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_continuation_diagnostic.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    root = Path(config["output"])
    parent_root = Path(config["parent_root"])
    verify_freeze(parent_root)
    if config["end_epoch"] - config["start_epoch"] != 10:
        raise ValueError("This protocol has a ten-epoch budget")
    state = torch.load(
        config["resume_checkpoint"], map_location="cpu", weights_only=False
    )
    assert state["epoch"] == config["start_epoch"] == 20
    for key in [
        "temperature",
        "positive_radius_px",
        "recall_radius_px",
        "nms_radius_px",
        "candidate_count",
        "batch_size",
        "training_affine",
    ]:
        assert state["config"][key] == parent[key], key
    training = OSFamilyDataset(parent_root, "train", parent, training=True)
    validation_base = OSFamilyDataset(parent_root, "val", parent, training=False)
    indices = freeze_monitor(config, parent, training.records, validation_base.records)
    monitor_plans = [
        json.loads(l)
        for l in (root / "monitor_transforms.jsonl").read_text().splitlines()
    ]
    val_plans = [
        json.loads(l)
        for l in (root / "validation_transforms.jsonl").read_text().splitlines()
    ]
    monitor = FrozenDiagnosticView(training, indices, monitor_plans)
    validation = FrozenDiagnosticView(
        validation_base, range(len(validation_base)), val_plans
    )
    np.testing.assert_array_equal(
        [p["optical_input_to_sar_input"] for p in monitor_plans],
        [p["optical_input_to_sar_input"] for p in val_plans],
    )
    assert not {r["family_id"] for r in monitor.records} & {
        r["family_id"] for r in validation.records
    }
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parent["learning_rate"],
        weight_decay=parent["weight_decay"],
    )
    optimizer.load_state_dict(state["optimizer"])
    assert state_digest(model.state_dict()) == state_digest(state["model"])
    assert state_digest(optimizer.state_dict()) == state_digest(state["optimizer"])
    rates = [g["lr"] for g in optimizer.param_groups]
    assert rates == [0.0003]
    assert {float(v["step"]) for v in optimizer.state.values()} == {
        float(state["step"])
    }
    generator = replay_sampler(
        parent["seed"], len(training), parent["batch_size"], config["start_epoch"]
    )
    save_json(
        root / "resume_audit.json",
        dict(
            checkpoint_sha256=sha256_file(config["resume_checkpoint"]),
            epoch=state["epoch"],
            global_step=state["step"],
            model_digest=state_digest(state["model"]),
            optimizer_digest_before=state_digest(state["optimizer"]),
            optimizer_digest_after=state_digest(optimizer.state_dict()),
            model_and_optimizer_exactly_restored=True,
            restored_learning_rates=rates,
            learning_rate_schedule="original constant AdamW lr; no scheduler present or restarted",
            original_scheduler_state_present="scheduler" in state,
            optimizer_reinitialized=False,
            original_loader_rng_saved=False,
            loader_generator_recovery="replay same Torch DataLoader on index-only data for 20 full epochs",
            replay_generator_state_sha256=state_digest(generator.get_state()),
            global_rng_limitation="epoch20 checkpoint had no global RNG states; model has no dropout and augmentation uses per-sample hash seeds",
            training_seed=parent["seed"],
            source_folds_changed=False,
            architecture_changed=False,
            source_model_sha256=sha256_file("sdgm/coarse.py"),
            original_train_metric_audit="previous epoch20_training was full frozen training view in eval(), not online augmented training loss; separate realization from validation",
            previous_train_plan_sha256=sha256_file(
                parent_root / "frozen_train_evaluation.jsonl"
            ),
            validation_plan_sha256=sha256_file(parent_root / "frozen_validation.jsonl"),
        ),
    )
    run = root / "run"
    run.mkdir()
    eval_config = dict(parent, cuda_device=config["cuda_device"])
    history = []
    epoch_rows = []
    gradients = []
    step = state["step"]
    start = time.perf_counter()
    loader = DataLoader(
        training,
        batch_size=parent["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    torch.cuda.reset_peak_memory_stats(device)
    with (run / "training.jsonl").open("w") as stream:
        for epoch in range(config["start_epoch"], config["end_epoch"] + 1):
            if epoch > config["start_epoch"]:
                training.set_epoch(epoch)
                model.train()
                loss_sum = 0.0
                query_sum = 0
                epoch_start = time.perf_counter()
                for batch in loader:
                    batch = prepare(batch, device)
                    geometry = geometry_for(model, batch, parent)
                    optimizer.zero_grad(set_to_none=True)
                    result = model(batch["optical"], batch["sar"])
                    if not all(torch.isfinite(v).all() for v in result.values()):
                        raise FloatingPointError("Nonfinite model output")
                    loss = neighborhood_mass_loss(result["scores"], geometry)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    if not geometry["supervised"].any():
                        raise RuntimeError("No supervised query in minibatch")
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    if (
                        step == state["step"]
                        or step == state["step"] + 10 * len(loader) - 1
                    ):
                        groups = gradient_summary(model)
                        gradients.append(
                            dict(epoch=epoch, step=step + 1, groups=groups)
                        )
                        assert all(
                            g["finite"] and g["l2_norm"] > 0 for g in groups.values()
                        )
                    assert [g["lr"] for g in optimizer.param_groups] == rates
                    optimizer.step()
                    step += 1
                    n = int(geometry["supervised"].sum())
                    value = float(loss.detach())
                    loss_sum += value * n
                    query_sum += n
                    stream.write(
                        json.dumps(
                            dict(
                                epoch=epoch,
                                step=step,
                                loss=value,
                                supervised_queries=n,
                                gradient_norm=float(norm),
                                learning_rates=rates,
                                nonfinite=False,
                            )
                        )
                        + "\n"
                    )
                stream.flush()
                row = dict(
                    epoch=epoch,
                    steps=step,
                    augmented_training_loss=loss_sum / query_sum,
                    effective_source_points=query_sum,
                    wall_seconds=time.perf_counter() - epoch_start,
                    learning_rates=rates,
                )
                epoch_rows.append(row)
                save_json(run / "progress.json", row)
                print(json.dumps(row), flush=True)
            epoch_metrics = {}
            for label, dataset in [
                ("training_monitor", monitor),
                ("validation", validation),
            ]:
                tag = f"epoch{epoch:02d}_{label}"
                metrics = evaluate_family(
                    model,
                    dataset,
                    eval_config,
                    device,
                    run,
                    tag,
                    details=True,
                    failures=False,
                )
                metrics["family_aggregation"] = aggregate_families(run, tag)
                epoch_metrics[label] = metrics
            if epoch == 20:
                old = json.loads(
                    (parent_root / "run/epoch20_validation_metrics.json").read_text()
                )
                keys = [
                    "pairs",
                    "families",
                    "supervised_queries",
                    "loss",
                    "recall_at_1",
                    "recall_at_4",
                    "nearest_candidate_distance_px",
                    "pair_distributions",
                    "masked",
                ]
                equality = {k: epoch_metrics["validation"][k] == old[k] for k in keys}
                assert all(equality.values()), equality
                save_json(root / "epoch20_validation_reproduction.json", equality)
            if epoch in config["comparison_epochs"]:
                failure_diagnostics(model, validation, eval_config, root, epoch)
            history.append(dict(epoch=epoch, **epoch_metrics))
            save_json(run / "evaluation_history.json", history)
            save_csv(run / "epoch_training.csv", epoch_rows)
            print(
                json.dumps(
                    dict(
                        epoch=epoch,
                        monitor_recall4=epoch_metrics["training_monitor"][
                            "recall_at_4"
                        ],
                        validation_recall4=epoch_metrics["validation"]["recall_at_4"],
                        learning_rates=rates,
                    )
                ),
                flush=True,
            )
            if epoch in config["checkpoint_epochs"]:
                checkpoint = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=epoch,
                    step=step,
                    parent_config=parent,
                    diagnostic_config=config,
                    monitor_freeze_sha256=sha256_file(root / "monitor_freeze.json"),
                    loader_generator_state=generator.get_state(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all(),
                    numpy_rng_state=np.random.get_state(),
                    python_rng_state=random.getstate(),
                    learning_rates=rates,
                )
                torch.save(checkpoint, run / f"epoch{epoch:02d}.pt")
                torch.save(checkpoint, run / "last.pt")
            verify_monitor(root)
            verify_freeze(parent_root)
    torch.cuda.synchronize(device)
    save_json(run / "gradient_checks.json", gradients)
    save_json(
        run / "completion.json",
        dict(
            completed=True,
            resumed_epoch20=True,
            end_epoch=config["end_epoch"],
            diagnostic_epochs=10,
            new_optimizer_updates=step - state["step"],
            global_step=step,
            optimizer_final_steps=sorted(
                {float(v["step"]) for v in optimizer.state.values()}
            ),
            learning_rates=rates,
            constant_learning_rate_continuation=True,
            all_epoch_fixed_train_and_val_recorded=True,
            model_and_optimizer_restored=True,
            failure_set_not_reselected=True,
            test_evaluated=False,
            UBC_evaluated=False,
            convergence_claimed=False,
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
        ),
    )
    print("CONTINUATION_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
