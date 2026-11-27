"""Frozen front adapters and context inputs, never supplied geometric labels."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .coarse import sample_support
from .context_descriptors import multiscale_maps


class IdentityTrainingView(Dataset):
    """Small-fit only: read native training pair, no geometric augmentation."""

    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)
        self.records = [base.records[i] for i in self.indices]
        self.split = "train"
        if base.split != "train":
            raise ValueError("Small fit training only")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.records[index]
        opt, sar = self.base.arrays[self.indices[index]]
        return dict(
            id=row["sample_id"],
            family_id=row["family_id"],
            optical=torch.from_numpy(opt.copy())[None].float() / 127.5 - 1,
            sar=torch.from_numpy(sar.copy())[None].float() / 127.5 - 1,
            matrix=torch.eye(3),
            source_support=torch.ones(1, 256, 256, dtype=torch.bool),
            target_support=torch.ones(1, 256, 256, dtype=torch.bool),
        )


@torch.no_grad()
def frozen_inputs(model, semantic_head, batch):
    pred = model(batch["optical"], batch["sar"])
    qs = semantic_head(pred["sar_descriptors"])
    mo = sample_support(
        batch["source_support"], model.target_pixel_xy, model.image_hw
    ).reshape(-1, 1, 32, 32)
    ms = sample_support(
        batch["target_support"], model.target_pixel_xy, model.image_hw
    ).reshape(-1, 1, 32, 32)
    return dict(
        optical_F=pred["optical_descriptors"],
        sar_F=pred["sar_descriptors"],
        optical_Q=batch["teacher"].detach(),
        sar_Q=qs.detach(),
        optical_M=mo,
        sar_M=ms,
        structural_scores=pred["scores"],
    )


@torch.no_grad()
def group_maps(inputs, projection, group):
    semantic = group.startswith("semantic")
    context = group.endswith("context")
    optical = inputs["optical_F"]
    sar = inputs["sar_F"]
    if semantic:
        eo = inputs["optical_Q"]
        es = inputs["sar_Q"]
    else:
        eo = torch.einsum("ec,bchw->behw", projection, optical)
        es = torch.einsum("ec,bchw->behw", projection, sar)
    om, of = multiscale_maps(optical, eo, inputs["optical_M"], context)
    sm, sf = multiscale_maps(sar, es, inputs["sar_M"], context)
    return om, sm, of, sf
