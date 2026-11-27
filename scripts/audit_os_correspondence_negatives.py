"""Replay frozen structural negatives to audit per-query boundary proximity."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sdgm.coarse import StructuralRetriever
from sdgm.correspondence_objective import fixed_competition
from sdgm.family_evaluation import quantiles
from sdgm.io import save_json
from sdgm.os_family_data import OSFamilyDataset
from sdgm.raw_lineage import sha256_file
from scripts.train_structure_smoke import prepare, geometry_for


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/os_correspondence_objective.json")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output"])
    semantic = json.loads(Path(cfg["semantic_config"]).read_text())
    parent = json.loads(Path(semantic["parent_config"]).read_text())
    locked = json.loads((root / "locked_protocol.json").read_text())
    assert (
        sha256_file(root / "shared_batch_schedule.jsonl") == locked["schedule_sha256"]
    )
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{cfg['cuda_device']}")
    torch.cuda.set_device(device)
    model = StructuralRetriever(parent["temperature"], image_hw=(256, 256)).to(device)
    model.load_state_dict(
        torch.load(
            semantic["structural_checkpoint"], map_location="cpu", weights_only=False
        )["model"]
    )
    model.requires_grad_(False).eval()
    train = OSFamilyDataset(semantic["family_root"], "train", parent, training=True)
    schedule = [
        json.loads(l)
        for l in (root / "shared_batch_schedule.jsonl").read_text().splitlines()
    ]
    digest = hashlib.sha256()
    closest = []
    first = []
    slot_hist = np.zeros(9, dtype="int64")
    near_bins = np.array([8, 10, 12, 16, 24, 32, 64, 128, 256, np.inf])
    for item in schedule:
        train.set_epoch(item["augmentation_epoch"])
        batch = prepare(
            next(
                iter(
                    DataLoader(
                        Subset(train, item["indices"]),
                        batch_size=len(item["indices"]),
                        shuffle=False,
                    )
                )
            ),
            device,
        )
        feature = model(batch["optical"], batch["sar"])
        g = geometry_for(model, batch, parent)
        c = fixed_competition(feature["scores"], g, model.target_pixel_xy)
        digest.update(c["negative_indices"].cpu().numpy().tobytes())
        d = torch.linalg.vector_norm(
            model.target_pixel_xy[c["negative_indices"].clamp_min(0)]
            - g["gt_pixel_xy"][:, :, None],
            dim=-1,
        )
        valid = c["negative_valid"] & g["supervised"][..., None]
        slot_hist += np.histogram(d[valid].cpu().numpy(), bins=near_bins)[0]
        keep = g["supervised"] & c["negative_valid"].any(-1)
        closest.extend(
            d.masked_fill(~c["negative_valid"], torch.inf)
            .min(-1)
            .values[keep]
            .cpu()
            .numpy()
        )
        first.extend(d[..., 0][keep & c["negative_valid"][..., 0]].cpu().numpy())
        if item["additional_step"] % 1000 == 0:
            print("Frozen-negative audit " + str(item["additional_step"]), flush=True)
    completion = json.loads((root / "completion.json").read_text())
    assert digest.hexdigest() == completion["shared_negative_index_digest"]
    existing = json.loads((root / "negative_distance_training.json").read_text())
    np.testing.assert_array_equal(slot_hist, existing["counts"])
    result = {}
    for key, values in [
        ("closest_negative_per_query", closest),
        ("first_highest_structure_negative_per_query", first),
    ]:
        values = np.asarray(values, dtype="float32")
        result[key] = dict(
            valid_query_occurrences=len(values),
            distance_px=quantiles(values),
            fraction_8to10=float((values <= 10).mean()),
            fraction_8to16=float((values <= 16).mean()),
            bins_px=existing["bins_px"],
            counts=np.histogram(values, bins=near_bins)[0].tolist(),
        )
    save_json(
        root / "negative_boundary_query_audit.json",
        dict(
            replayed_training_updates=5000,
            same_negative_index_digest_exact=True,
            index_digest=digest.hexdigest(),
            slot_histogram_exact=True,
            repeated_query_occurrences_not_unique_samples=True,
            no_training_or_head_forward=True,
            **result,
        ),
    )
    print(
        "All5000 frozen negative sets exact; per-query boundary distances exported",
        flush=True,
    )


if __name__ == "__main__":
    main()
