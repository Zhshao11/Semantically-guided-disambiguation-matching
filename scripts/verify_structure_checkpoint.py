"""Reload an own smoke checkpoint and compare all reported training metrics."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sdgm.coarse import StructuralRetriever
from sdgm.io import save_json
from sdgm.os_data import OSNativeSmokeDataset
from sdgm.smoke_data import WHUSmokeDataset
from scripts.train_structure_smoke import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="reports/structure_os_native_20261126")
    args = parser.parse_args()
    output = Path(args.run)
    config = json.loads((output / "config.json").read_text())
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    checkpoint = torch.load(output / "last.pt", map_location=device, weights_only=False)
    model = StructuralRetriever(
        config["temperature"], image_hw=tuple(config.get("image_hw", [512, 512]))
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    if config.get("dataset_kind") == "os_native":
        dataset = OSNativeSmokeDataset(
            config["manifest"], config["pairs"], config["seed"]
        )
    else:
        dataset = WHUSmokeDataset(config["data_root"], config["pairs"], config["seed"])
    metrics = evaluate(
        model,
        DataLoader(dataset, batch_size=config["batch_size"], shuffle=False),
        config,
        device,
        output,
        "reload",
    )
    previous = json.loads((output / "final_metrics.json").read_text())
    fields = [
        "supervised_queries",
        "loss",
        "recall_at_1",
        "recall_at_4",
        "nearest_candidate_distance_px",
        "masked",
    ]
    comparison = {key: metrics[key] == previous[key] for key in fields}
    comparison["masked"] = all(
        metrics["masked"].get(key) == value for key, value in previous["masked"].items()
    ) and all(
        value == 0
        for key, value in metrics["masked"].items()
        if key not in previous["masked"]
    )
    save_json(
        output / "checkpoint_reload_verification.json",
        dict(
            exact_metric_reproduction=all(comparison.values()),
            fields=comparison,
            UBC_evaluated=False,
            independent_validation=False,
        ),
    )
    if not all(comparison.values()):
        raise AssertionError(comparison)
    print(comparison)


if __name__ == "__main__":
    main()
