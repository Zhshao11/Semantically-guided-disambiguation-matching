"""Fixed training-only subset of the existing WHU derivative release."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import homogeneous
from .io import load_datum, save_json


class WHUSmokeDataset(Dataset):
    def __init__(self, root, count=32, seed=20261125, manifest_path=None):
        self.root = Path(root)
        if not 16 <= count <= 32:
            raise ValueError("This engineering smoke requires 16 to 32 training pairs")
        files = sorted((self.root / "training").glob("**/datum/*.npy"))
        if len(files) != 5600:
            raise ValueError(
                f"Expected the audited 5600 training files, found {len(files)}"
            )
        indices = np.sort(
            np.random.default_rng(seed).choice(len(files), count, replace=False)
        )
        self.samples = []
        manifest = []
        for index in indices:
            path = files[int(index)]
            datum = load_datum(path)
            pair = np.asarray(datum[0])
            matrix = homogeneous(datum[2]).astype(np.float32)
            if pair.shape != (512, 512, 2) or not np.isfinite(pair).all():
                raise ValueError(f"Invalid pair {path}")
            # Only radiometric scaling. No random geometric augmentation or resize.
            optical = torch.from_numpy(pair[..., 0].copy()).float()[None] / 127.5 - 1
            sar = torch.from_numpy(pair[..., 1].copy()).float()[None] / 127.5 - 1
            self.samples.append(
                dict(
                    id=path.stem,
                    optical=optical,
                    sar=sar,
                    matrix=torch.from_numpy(matrix),
                )
            )
            manifest.append(
                dict(
                    id=path.stem,
                    split="training",
                    path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    matrix=matrix.tolist(),
                )
            )
        if manifest_path:
            save_json(
                manifest_path,
                dict(
                    seed=seed,
                    count=count,
                    samples=manifest,
                    scene_independence_verified=False,
                    extra_geometric_augmentation=False,
                    zero_pixels_are_nodata=False,
                    purpose="engineering_diagnosis_relative_to_derivative_labels",
                ),
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
