"""Reload final baseline, verify held-out metrics and immutable source folds."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from sdgm.coarse import StructuralRetriever
from sdgm.family_evaluation import evaluate_family
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_family_baseline.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    run = root / "run"
    verify_freeze(root)
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == config["epochs"]
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(config["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(state["model"])
    dataset = OSFamilyDataset(root, "val", config, training=False)
    observed = evaluate_family(
        model,
        dataset,
        config,
        device,
        root / "verification",
        "reloaded_validation",
        details=True,
    )
    expected = json.loads(
        (run / f"epoch{config['epochs']:02d}_validation_metrics.json").read_text()
    )
    fields = [
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
    equality = {key: observed[key] == expected[key] for key in fields}
    assert all(equality.values()), equality
    plan_hash = sha256_file(root / "frozen_validation.jsonl")
    context = json.loads((run / "runtime_context.json").read_text())
    assert context["validation_plan_sha256"] == plan_hash
    training = OSFamilyDataset(root, "train", config, training=True)
    signatures = []
    for epoch in [0, 1, 2, config["epochs"]]:
        training.set_epoch(epoch)
        examples = [
            training.transform_record(index)
            for index in [0, len(training) // 2, len(training) - 1]
        ]
        signatures.append(
            dict(
                epoch=epoch,
                plan_sha256=hashlib.sha256(
                    json.dumps(examples, sort_keys=True).encode()
                ).hexdigest(),
                examples=examples,
            )
        )
    assert len({row["plan_sha256"] for row in signatures}) == len(signatures)
    save_json(root / "verification/training_augmentation_examples.json", signatures)
    save_json(
        root / "verification/reload_verification.json",
        dict(
            exact_validation_metric_reproduction=True,
            fields=equality,
            validation_plan_unchanged=True,
            source_freeze_unchanged=True,
            training_epoch_perturbations_differ=True,
            test_evaluated=False,
            UBC_evaluated=False,
            checkpoint_sha256=sha256_file(run / "last.pt"),
        ),
    )
    print(json.dumps(equality))


if __name__ == "__main__":
    main()
