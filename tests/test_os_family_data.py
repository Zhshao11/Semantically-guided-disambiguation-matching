import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from sdgm.io import save_json, save_jsonl
from sdgm.raw_lineage import sha256_file
from sdgm.os_family_data import OSFamilyDataset
from sdgm.geometry import transform_points


def fixture(tmp_path):
    rows = []
    y, x = np.indices((256, 256))
    image = ((x + 3 * y) % 256).astype("uint8")
    for split in ["train", "val"]:
        optical = tmp_path / f"{split}_opt.png"
        sar = tmp_path / f"{split}_sar.png"
        Image.fromarray(image).save(optical)
        Image.fromarray(image).save(sar)
        row = dict(
            sample_id=f"{split}/one",
            family_id=f"F-{split}",
            split=split,
            optical_path=str(optical),
            sar_path=str(sar),
            optical_sha256=sha256_file(optical),
            sar_sha256=sha256_file(sar),
        )
        save_jsonl(tmp_path / f"native256_{split}.jsonl", [row])
    save_json(
        tmp_path / "freeze_receipt.json",
        dict(
            frozen_files={
                f"native256_{s}.jsonl": sha256_file(tmp_path / f"native256_{s}.jsonl")
                for s in ["train", "val"]
            }
        ),
    )
    return dict(
        seed=2,
        evaluation_seed=3,
        training_affine=dict(
            rotation_degrees=[-20, 20], scale=[0.9, 1.1], translation_pixels=[-16, 16]
        ),
    )


def test_epoch_augmentation_reproducible_and_evaluation_fixed_with_family_inheritance(
    tmp_path,
):
    config = fixture(tmp_path)
    train = OSFamilyDataset(tmp_path, "train", config, training=True, workers=1)
    evaluation = train.fixed_view()
    val = OSFamilyDataset(tmp_path, "val", config, workers=1)
    a = train[0]
    train.set_epoch(1)
    b = train[0]
    train.set_epoch(0)
    c = train[0]
    assert a["family_id"] == b["family_id"] == c["family_id"] == "F-train"
    assert not torch.equal(a["matrix"], b["matrix"])
    torch.testing.assert_close(a["matrix"], c["matrix"])
    torch.testing.assert_close(a["sar"], c["sar"])
    eval_before = evaluation[0]
    train.set_epoch(8)
    eval_after = evaluation[0]
    torch.testing.assert_close(eval_before["matrix"], eval_after["matrix"])
    assert val.transform_record(0)["epoch"] is None
    with pytest.raises(ValueError, match="frozen"):
        val.set_epoch(1)
    with pytest.raises(ValueError, match="sealed"):
        OSFamilyDataset(tmp_path, "test", config)
    assert a["source_support"].all() and not b["target_support"].all()
    plan = train.transform_record(0)
    matrix = np.asarray(plan["optical_input_to_sar_input"])
    np.testing.assert_allclose(
        transform_points(matrix, [[7.0, 15.0]]),
        transform_points(plan["sar_original_to_input"], [[7.0, 15.0]]),
    )


def test_frozen_assignment_tamper_rejected(tmp_path):
    config = fixture(tmp_path)
    with (tmp_path / "native256_val.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="changed"):
        OSFamilyDataset(tmp_path, "val", config)
