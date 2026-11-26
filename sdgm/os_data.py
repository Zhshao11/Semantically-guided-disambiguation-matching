"""Fixed native OS samples; exact synthetic perturbation over claimed alignment."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .io import save_json
from .raw_lineage import sha256_file
from .raw_tiles import perturbation_matrix, sample_seed


class OSNativeSmokeDataset(Dataset):
    def __init__(self, manifest, count=32, seed=20261126, manifest_path=None):
        if not 16 <= count <= 32:
            raise ValueError("Engineering smoke must use 16 to 32 fixed training pairs")
        rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        rows = sorted(
            [
                r
                for r in rows
                if r["record_kind"] == "patch_pair"
                and r["subset"] == "OSdataset/256/train"
                and r["official_split"] == "train"
                and r["eligible_by_frozen_file_rules"]
            ],
            key=lambda r: r["id"],
        )
        if len(rows) < count:
            raise ValueError("Insufficient audited native OS training pairs")
        # Freeze selection from file properties, before any model output exists.
        candidate_count = len(rows)
        rows = [
            r
            for r in rows
            if all(
                max(r[m]["maximum"]) > min(r[m]["minimum"]) for m in ["optical", "sar"]
            )
        ]
        informative_count = len(rows)
        unique = {}
        for row in rows:
            unique.setdefault(row["joint_pixel_sha256"], row)
        rows = list(unique.values())
        if len(rows) < count:
            raise ValueError("Insufficient nonconstant, distinct OS engineering pairs")
        selected = np.sort(
            np.random.default_rng(seed).choice(len(rows), count, replace=False)
        )
        self.samples = []
        metadata = []
        settings = dict(
            rotation_degrees=[-20.0, 20.0],
            scale=[0.9, 1.1],
            translation_pixels=[-16.0, 16.0],
        )
        self.image_hw = (256, 256)
        for index in selected:
            row = rows[int(index)]
            arrays = []
            for modality in ["optical", "sar"]:
                rec = row[modality]
                if sha256_file(rec["path"]) != rec["sha256"]:
                    raise ValueError("Audited OS file changed")
                with Image.open(rec["path"]) as image:
                    if image.mode != "L" or image.size != (256, 256):
                        raise ValueError(
                            "Expected native 256 single-band PNG; do not resize or replicate channels"
                        )
                    arrays.append(np.array(image))
            optical, base_sar = arrays
            transform_seed = sample_seed(seed, "engineering_training", row["id"], 0)
            matrix, parameters = perturbation_matrix(transform_seed, 256, settings)
            sar = cv2.warpAffine(
                base_sar,
                matrix[:2],
                (256, 256),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            support = (
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
            self.samples.append(
                dict(
                    id=row["id"],
                    optical=torch.from_numpy(optical.copy())[None].float() / 127.5 - 1,
                    sar=torch.from_numpy(sar.copy())[None].float() / 127.5 - 1,
                    matrix=torch.tensor(matrix, dtype=torch.float32),
                    source_support=torch.ones(1, 256, 256, dtype=torch.bool),
                    target_support=torch.from_numpy(support)[None],
                )
            )
            metadata.append(
                dict(
                    id=row["id"],
                    split="official_train_engineering_only",
                    optical=row["optical"],
                    sar=row["sar"],
                    native_hw=[256, 256],
                    resize=False,
                    channel_conversion=False,
                    normalization="uint8 / 127.5 - 1",
                    raw_window_mapping=row.get("exact_pixel_crop_parents", []),
                    parent_scene_id=None,
                    synthetic_seed=transform_seed,
                    synthetic_parameters=parameters,
                    synthetic_matrix=matrix.tolist(),
                    base_optical_to_sar="nominal identity from publisher coregistration claim; per-patch residual unknown",
                    transform_direction="optical native pixel -> perturbed SAR, conditional on nominal base identity",
                    transform_chain="A_s @ B_nominal; B_nominal=I, no additional crop/resize",
                    interpolation="OpenCV INTER_LINEAR",
                    fill=0,
                    masks="native extent and full interpolation support; zero values remain valid",
                    fine_supervision_certified=False,
                    generated_sar_pixel_sha256=__import__("hashlib")
                    .sha256(sar.tobytes())
                    .hexdigest(),
                )
            )
        if manifest_path:
            save_json(
                manifest_path,
                dict(
                    dataset="OS native 256",
                    count=count,
                    selection_policy="fixed RNG over audited official train; both modalities nonconstant; one representative per exact joint pixel hash; no model-based selection",
                    selection_counts=dict(
                        decoded_candidates=candidate_count,
                        nonconstant_pairs=informative_count,
                        distinct_pairs=len(rows),
                    ),
                    seed=seed,
                    source_manifest_sha256=sha256_file(manifest),
                    samples=metadata,
                    extra_geometric_augmentation=False,
                    transformations_frozen_across_epochs=True,
                    scene_independence_verified=False,
                    official_val_or_test_used=False,
                    UBC_used=False,
                    purpose="engineering_only_relative_to_known_synthetic_transform_and_publisher_claimed_base_alignment",
                ),
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
