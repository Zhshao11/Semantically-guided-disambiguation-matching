"""Native256 family-aware inputs with epoch-seeded affine training augmentation."""

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .geometry import update_after_augmentation
from .raw_lineage import sha256_file
from .raw_tiles import perturbation_matrix, sample_seed


def verify_freeze(root):
    root = Path(root)
    receipt = json.loads((root / "freeze_receipt.json").read_text())
    for name, digest in receipt["frozen_files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Frozen family file changed: {name}")
    return receipt


def _load_native(row):
    arrays = []
    for modality in ["optical", "sar"]:
        path = row[modality + "_path"]
        if sha256_file(path) != row[modality + "_sha256"]:
            raise ValueError(f"Raw source changed: {row['sample_id']}")
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (256, 256):
                raise ValueError(
                    "Native256 grayscale is required; no resize or channel replication"
                )
            arrays.append(np.array(image))
    return np.stack(arrays)


class OSFamilyDataset(Dataset):
    def __init__(self, root, split, config, training=False, workers=8):
        if split not in ["train", "val"]:
            raise ValueError(
                "Final test is sealed; this training/validation reader cannot load it"
            )
        if training and split != "train":
            raise ValueError("Validation transformations must remain fixed")
        verify_freeze(root)
        self.root = Path(root)
        self.split = split
        self.config = config
        self.training = training
        self.epoch = 0
        self.records = [
            json.loads(line)
            for line in (self.root / f"native256_{split}.jsonl")
            .read_text()
            .splitlines()
        ]
        if not self.records:
            raise ValueError("Empty source-family split")
        if any(row["split"] != split for row in self.records):
            raise ValueError("Wrong source-family assignment")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            self.arrays = np.stack(list(pool.map(_load_native, self.records)))
        cv2.setNumThreads(1)

    def fixed_view(self):
        result = copy.copy(self)
        result.training = False
        result.epoch = 0
        return result

    def set_epoch(self, epoch):
        if not self.training:
            raise ValueError("Cannot alter frozen evaluation transformations")
        self.epoch = int(epoch)

    def transform_record(self, index):
        row = self.records[index]
        seed = sample_seed(
            self.config["seed"] if self.training else self.config["evaluation_seed"],
            "train_epoch" if self.training else self.split + "_fixed_evaluation",
            row["sample_id"],
            self.epoch if self.training else 0,
        )
        augmentation, parameters = perturbation_matrix(
            seed, 256, self.config["training_affine"]
        )
        matrix = update_after_augmentation(np.eye(3), np.eye(3), augmentation)
        return dict(
            sample_id=row["sample_id"],
            family_id=row["family_id"],
            split=self.split,
            transform_seed=seed,
            epoch=self.epoch if self.training else None,
            parameters=parameters,
            optical_original_to_input=np.eye(3).tolist(),
            sar_original_to_input=augmentation.tolist(),
            optical_input_to_sar_input=matrix.tolist(),
            base_relation="publisher-claimed co-registration; nominal identity; residual unknown",
            composition="A_s @ B_nominal @ inverse(A_o), B_nominal=I, A_o=I",
            interpolation="OpenCV INTER_LINEAR",
            fill_value=0,
            original_size=[256, 256],
            resize=False,
            normalization="uint8/127.5-1",
            fine_precision_certified=False,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        optical, base_sar = self.arrays[index]
        plan = self.transform_record(index)
        matrix = np.asarray(plan["optical_input_to_sar_input"])
        sar = cv2.warpAffine(
            base_sar,
            matrix[:2],
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = (
            cv2.warpAffine(
                np.ones((256, 256), "float32"),
                matrix[:2],
                (256, 256),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            >= 1 - 1e-6
        )
        return dict(
            id=plan["sample_id"],
            family_id=plan["family_id"],
            optical=torch.from_numpy(optical.copy())[None].float() / 127.5 - 1,
            sar=torch.from_numpy(sar.copy())[None].float() / 127.5 - 1,
            matrix=torch.tensor(matrix, dtype=torch.float32),
            source_support=torch.ones(1, 256, 256, dtype=torch.bool),
            target_support=torch.from_numpy(mask)[None],
        )
