from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_os_ubc import inspect_pair, os_jobs, spatial_split_check


def test_native_inventory_pairs_by_id_and_detects_missing_corrupt_mismatched(tmp_path):
    folder = tmp_path / "Patch-level Subset/OSdataset/256/train"
    folder.mkdir(parents=True)
    Image.fromarray(np.zeros((24, 32), "uint8")).save(folder / "opt1.png")
    Image.fromarray(np.zeros((24, 32), "uint8")).save(folder / "sar1.png")
    Image.fromarray(np.ones((24, 32), "uint8")).save(folder / "opt2.png")
    (folder / "opt3.png").write_bytes(b"corrupt")
    Image.fromarray(np.zeros((24, 32), "uint8")).save(folder / "sar3.png")
    Image.fromarray(np.zeros((24, 32), "uint8")).save(folder / "opt4.png")
    Image.fromarray(np.zeros((32, 24), "uint8")).save(folder / "sar4.png")
    rows = [inspect_pair(job) for job in os_jobs(tmp_path)]
    by_id = {Path(row["id"]).name: row for row in rows}
    assert by_id["1"]["eligible_by_frozen_file_rules"]
    assert (
        by_id["1"]["optical"]["width"] == 32 and by_id["1"]["optical"]["height"] == 24
    )
    assert by_id["1"]["optical"]["valid_pixels_per_band"] == [24 * 32]
    assert by_id["1"]["optical"]["zero_fraction"] == [1]
    assert by_id["2"]["hard_exclusion_reasons"] == ["missing_counterpart"]
    assert by_id["3"]["hard_exclusion_reasons"] == ["decode_failure"]
    assert by_id["4"]["hard_exclusion_reasons"] == ["spatial_size_mismatch"]


def test_spatial_audit_counts_area_overlap_without_counting_touch_or_same_split():
    rows = [
        dict(
            id=name,
            official_split=split,
            region_id="same-city",
            optical=dict(crs="EPSG:32631", bounds=bounds),
        )
        for name, split, bounds in [
            ("a", "train", [0, 0, 10, 10]),
            ("b", "val", [5, 0, 15, 10]),
            ("c", "test", [15, 0, 25, 10]),
            ("d", "train", [0, 0, 1, 1]),
        ]
    ]
    result = spatial_split_check(rows)
    assert result["cross_official_split_overlapping_optical_footprints"] == 1
    assert result["overlap_examples"][0]["overlap_area_crs_units_squared"] == 50


def test_existing_external_pool_cannot_be_silently_rewritten(tmp_path, monkeypatch):
    import json
    import pytest
    from scripts.audit_os_ubc import main

    output = tmp_path / "frozen"
    (output / "ubcv2").mkdir(parents=True)
    receipt = output / "ubcv2/freeze_receipt.json"
    receipt.write_text('{"frozen":true}\n')
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(output=str(output))))
    monkeypatch.setattr("sys.argv", ["audit_os_ubc.py", "--config", str(config)])
    with pytest.raises(FileExistsError, match="Frozen audit already exists"):
        main()
    assert receipt.read_text() == '{"frozen":true}\n'
    assert not (output / "os").exists()
