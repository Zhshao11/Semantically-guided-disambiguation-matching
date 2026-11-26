import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from sdgm.io import save_jsonl
from sdgm.os_data import OSNativeSmokeDataset
from sdgm.raw_lineage import sha256_file


def test_os_loader_freezes_native_inputs_transforms_and_training_membership(tmp_path):
    rows = []
    yy, xx = np.indices((256, 256))
    for index in range(17):
        records = {}
        for modality in ["optical", "sar"]:
            path = tmp_path / f"{modality}{index}.png"
            Image.fromarray(((xx + 2 * yy + index) % 256).astype("uint8")).save(path)
            records[modality] = dict(
                path=str(path), sha256=sha256_file(path), minimum=[0], maximum=[255]
            )
        rows.append(
            dict(
                id=f"native/{index}",
                joint_pixel_sha256=f"joint-{index}",
                record_kind="patch_pair",
                subset="OSdataset/256/train",
                official_split="train",
                eligible_by_frozen_file_rules=True,
                **records,
            )
        )
    # A test row with a lexicographically first ID must never be selected.
    rows.append(
        dict(rows[0], id="000-test", subset="OSdataset/256/test", official_split="test")
    )
    save_jsonl(tmp_path / "manifest.jsonl", rows)
    first = OSNativeSmokeDataset(
        tmp_path / "manifest.jsonl",
        count=16,
        seed=7,
        manifest_path=tmp_path / "fixed.json",
    )
    second = OSNativeSmokeDataset(tmp_path / "manifest.jsonl", count=16, seed=7)
    assert len(first) == 16
    for a, b in zip(first.samples, second.samples):
        assert a["id"] != "000-test" and a["optical"].shape == (1, 256, 256)
        np.testing.assert_array_equal(a["sar"].numpy(), b["sar"].numpy())
        np.testing.assert_array_equal(a["matrix"].numpy(), b["matrix"].numpy())
        assert a["source_support"].all() and not a["target_support"].all()
    metadata = json.loads((tmp_path / "fixed.json").read_text())
    row = metadata["samples"][0]
    original = np.array(Image.open(row["optical"]["path"]))
    np.testing.assert_allclose(
        (first[0]["optical"][0].numpy() + 1) * 127.5, original, atol=1e-5
    )
    base = np.array(Image.open(row["sar"]["path"]))
    expected = cv2.warpAffine(
        base, np.array(row["synthetic_matrix"])[:2], (256, 256), flags=cv2.INTER_LINEAR
    )
    np.testing.assert_allclose(
        (first[0]["sar"][0].numpy() + 1) * 127.5, expected, atol=1e-5
    )
    assert not metadata["official_val_or_test_used"] and not metadata["UBC_used"]
    assert not row["fine_supervision_certified"]
    Path(rows[0]["optical"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        OSNativeSmokeDataset(tmp_path / "manifest.jsonl", count=17, seed=7)
