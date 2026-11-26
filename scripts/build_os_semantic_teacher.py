"""Freeze the train-only optical prototype teacher before semantic training."""

import argparse
import json
from pathlib import Path

import torch

from sdgm.continuation_diagnostics import verify_monitor
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset, verify_freeze
from sdgm.raw_lineage import sha256_file
from sdgm.semantic_teacher import build_teacher_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_semantic_routing.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    if root.exists():
        raise FileExistsError("Semantic protocol already exists; do not replace it")
    verify_freeze(config["family_root"])
    verify_monitor(config["diagnostic_root"])
    if (
        sha256_file(config["structural_checkpoint"])
        != config["structural_checkpoint_sha256"]
    ):
        raise ValueError("Must use the fixed epoch30 checkpoint")
    root.mkdir(parents=True)
    save_json(root / "preregistered_protocol.json", config)
    parent = json.loads(Path(config["parent_config"]).read_text())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.cuda.set_device(device)
    train = OSFamilyDataset(config["family_root"], "train", parent, training=True)
    val = OSFamilyDataset(config["family_root"], "val", parent, training=False)
    result = build_teacher_cache(config, train, val, device)
    print(
        json.dumps(
            dict(stage="teacher_complete", cache_version=result["cache_version"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
