"""Evaluate the one fixed final head with original baseline batch arithmetic."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from sdgm.coarse import StructuralRetriever
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.semantic_evaluation import frozen_views, evaluate_routing
from sdgm.semantic_routing import SARPrototypeHead
from sdgm.semantic_teacher import TeacherView, verify_teacher_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_routing.json")
    parser.add_argument("--record-batch-correction", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    parent = json.loads(Path(config["parent_config"]).read_text())
    verify_teacher_cache(root / "teacher", config)
    if args.record_batch_correction:
        backup = root / "engineering_batch16_arithmetic_audit"
        if backup.exists():
            raise FileExistsError("Correction already recorded")
        shutil.copytree(root / "evaluation", backup)
        save_json(
            root / "evaluation_batch_correction.json",
            dict(
                reason="Original semantic inference batch16 caused tiny candidate-ranking roundoff differences from original baseline batch4. Use original batch4 for final A/B/C exact baseline reproduction.",
                previous_batch_size=16,
                final_batch_size=parent["batch_size"],
                training_repeated=False,
                same_final_head=True,
                same_structural_checkpoint=True,
                same_lambda_from_training=True,
                previous_results_preserved=str(backup),
                validation_parameter_search=False,
            ),
        )
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(
            config["structural_checkpoint"], map_location="cpu", weights_only=False
        )["model"]
    )
    model.requires_grad_(False).eval()
    head = SARPrototypeHead().to(device)
    head.load_state_dict(
        torch.load(
            root / "training/semantic_head.pt", map_location="cpu", weights_only=False
        )["head"]
    )
    head.eval()
    train = OSFamilyDataset(config["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(config["family_root"], "val", parent, training=False)
    views = frozen_views(config, train, val)
    weight = json.loads((root / "routing_calibration.json").read_text())["lambda_value"]
    for split, view in zip(["training_monitor", "validation"], views):
        result = evaluate_routing(
            model,
            head,
            TeacherView(view, root / "teacher"),
            parent,
            config,
            device,
            root / "evaluation",
            split,
            weight,
        )
        print(json.dumps(dict(stage="fixed_head_eval", **result)), flush=True)
    completion = json.loads((root / "completion.json").read_text())
    completion["routing_evaluation_batch_size"] = parent["batch_size"]
    completion["arithmetic_correction_recorded"] = (
        root / "evaluation_batch_correction.json"
    ).exists()
    save_json(root / "completion.json", completion)


if __name__ == "__main__":
    main()
