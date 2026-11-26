"""Predeclared native256 structural baseline on frozen source-family folds."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever, neighborhood_mass_loss
from sdgm.family_evaluation import evaluate_family
from sdgm.io import save_csv, save_json, save_jsonl
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file
from scripts.train_structure_smoke import prepare, geometry_for, gradient_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_family_baseline.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    receipt = verify_freeze(root)
    if json.loads((root / "preregistered_protocol.json").read_text()) != config:
        raise ValueError("Run configuration differs from predeclared family freeze")
    run = root / "run"
    if run.exists():
        raise FileExistsError(
            "Use a distinct frozen protocol rather than overwrite a baseline"
        )
    run.mkdir()
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    training = OSFamilyDataset(root, "train", config, training=True)
    validation = OSFamilyDataset(root, "val", config, training=False)
    train_evaluation = training.fixed_view()
    groups_train = {r["family_id"] for r in training.records}
    groups_val = {r["family_id"] for r in validation.records}
    assert not groups_train & groups_val
    # Save actual fixed sample and perturbation records before any held-out model call.
    save_jsonl(
        root / "frozen_train_evaluation.jsonl",
        [train_evaluation.transform_record(i) for i in range(len(train_evaluation))],
    )
    save_jsonl(
        root / "frozen_validation.jsonl",
        [validation.transform_record(i) for i in range(len(validation))],
    )
    save_json(
        run / "runtime_context.json",
        dict(
            protocol_sha256=sha256_file(root / "preregistered_protocol.json"),
            freeze_receipt_sha256=sha256_file(root / "freeze_receipt.json"),
            validation_plan_sha256=sha256_file(root / "frozen_validation.jsonl"),
            train_plan_sha256=sha256_file(root / "frozen_train_evaluation.jsonl"),
            seed=seed,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(device),
            native_image_hw=[256, 256],
            feature_shape=[config["batch_size"], 128, 32, 32],
            score_shape=[config["batch_size"], 144, 1024],
            train_samples=len(training),
            validation_samples=len(validation),
            epochs=config["epochs"],
            source_family_disjoint=True,
            geographic_independence=False,
            test_evaluated=False,
            UBC_evaluated=False,
            training_from_scratch=True,
            training_augmentation="fixed hash(master,sample,epoch), not worker order",
            optimizer="AdamW",
            primary_checkpoint=config["primary_checkpoint"],
            checkpoint_selection="last predeclared epoch; validation best saved only as secondary",
        ),
    )
    small = StructuralRetriever(config["temperature"], image_hw=(256, 256)).to(device)
    small_state = torch.load(
        config["small_checkpoint"], map_location=device, weights_only=False
    )
    small.load_state_dict(small_state["model"])
    quick = evaluate_family(
        small,
        validation,
        config,
        device,
        root / "small32_diagnostic",
        "small32_validation",
        details=True,
        failures=True,
    )
    print(
        json.dumps(dict(stage="small32_isolated_validation", metrics=quick)), flush=True
    )
    del small, small_state
    torch.cuda.empty_cache()
    # Reset initialization after quick diagnostic; no small-checkpoint warm start.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = StructuralRetriever(config["temperature"], image_hw=(256, 256)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    save_json(
        run / "model_signature.json",
        dict(
            parameters=sum(p.numel() for p in model.parameters()),
            architecture="unchanged StructuralRetriever native256",
            candidates=config["candidate_count"],
            retrieval_radius_px=config["recall_radius_px"],
            model_source_sha256=sha256_file("sdgm/coarse.py"),
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        training,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    evaluations = []
    gradient_records = []
    epoch_rows = []
    best = -1.0
    best_epoch = None
    global_step = 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with (run / "training.jsonl").open("w") as stream:
        for epoch in range(0, config["epochs"] + 1):
            if epoch:
                training.set_epoch(epoch)
                model.train()
                loss_sum = 0.0
                query_sum = 0
                epoch_start = time.perf_counter()
                for batch in loader:
                    batch = prepare(batch, device)
                    geometry = geometry_for(model, batch, config)
                    optimizer.zero_grad(set_to_none=True)
                    result = model(batch["optical"], batch["sar"])
                    if not all(torch.isfinite(v).all() for v in result.values()):
                        raise FloatingPointError("Nonfinite forward")
                    loss = neighborhood_mass_loss(result["scores"], geometry)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    if not geometry["supervised"].any():
                        raise RuntimeError(
                            "Entire augmented minibatch has no supervision"
                        )
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    if (
                        global_step == 0
                        or global_step == len(loader) * config["epochs"] - 1
                    ):
                        gradients = gradient_summary(model)
                        gradient_records.append(
                            dict(epoch=epoch, step=global_step + 1, groups=gradients)
                        )
                        if not all(
                            g["finite"] and g["l2_norm"] > 0 for g in gradients.values()
                        ):
                            raise RuntimeError("Ineffective gradient")
                    optimizer.step()
                    global_step += 1
                    n = int(geometry["supervised"].sum())
                    loss_sum += float(loss) * n
                    query_sum += n
                    row = dict(
                        epoch=epoch,
                        step=global_step,
                        loss=float(loss),
                        supervised_queries=n,
                        gradient_norm_before_clipping=float(norm),
                        nonfinite=False,
                    )
                    stream.write(json.dumps(row) + "\n")
                stream.flush()
                epoch_row = dict(
                    epoch=epoch,
                    steps=global_step,
                    augmented_training_loss=loss_sum / max(query_sum, 1),
                    queries=query_sum,
                    wall_seconds=time.perf_counter() - epoch_start,
                )
                epoch_rows.append(epoch_row)
                save_json(run / "progress.json", epoch_row)
                print(json.dumps(epoch_row), flush=True)
            if epoch in config["evaluation_epochs"]:
                final = epoch == config["epochs"]
                val = evaluate_family(
                    model,
                    validation,
                    config,
                    device,
                    run,
                    f"epoch{epoch:02d}_validation",
                    details=final,
                    failures=final,
                )
                train = evaluate_family(
                    model,
                    train_evaluation,
                    config,
                    device,
                    run,
                    f"epoch{epoch:02d}_training",
                    details=final,
                    failures=final,
                )
                evaluations.append(dict(epoch=epoch, training=train, validation=val))
                save_json(run / "evaluation_history.json", evaluations)
                print(
                    json.dumps(
                        dict(
                            epoch=epoch,
                            train_recall4=train["recall_at_4"],
                            validation_recall4=val["recall_at_4"],
                        )
                    ),
                    flush=True,
                )
                if epoch and val["recall_at_4"] > best:
                    best = val["recall_at_4"]
                    best_epoch = epoch
                    torch.save(
                        dict(
                            model=model.state_dict(),
                            epoch=epoch,
                            config=config,
                            selection="secondary validation recall4",
                            family_freeze_sha256=sha256_file(
                                root / "freeze_receipt.json"
                            ),
                        ),
                        run / "best_validation.pt",
                    )
            if epoch:
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        epoch=epoch,
                        config=config,
                        step=global_step,
                        family_freeze_sha256=sha256_file(root / "freeze_receipt.json"),
                        selection="last_epoch_predeclared_primary",
                    ),
                    run / "last.pt",
                )
    torch.cuda.synchronize(device)
    save_json(
        run / "performance.json",
        dict(
            wall_seconds=time.perf_counter() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            includes_training_and_evaluations=True,
            steps=global_step,
            epochs=config["epochs"],
        ),
    )
    save_csv(run / "epoch_training.csv", epoch_rows)
    save_json(run / "gradient_checks.json", gradient_records)
    save_json(
        run / "completion.json",
        dict(
            training_complete=True,
            epochs=config["epochs"],
            steps=global_step,
            primary_checkpoint="last.pt",
            secondary_best_validation_epoch=best_epoch,
            secondary_best_validation_recall4=best,
            train_samples=len(training),
            validation_samples=len(validation),
            finite_forward_backward=True,
            effective_gradients=True,
            test_evaluated=False,
            UBC_evaluated=False,
            geographic_independence=False,
            interpretation="source-family-isolated validation relative to synthetic labels, not fine registration accuracy",
        ),
    )
    verify_freeze(root)
    print("BASELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
