import csv
import json

import numpy as np
import pytest
import rasterio

from sdgm.audit import audit_spacenet


@pytest.mark.parametrize("outside", [False, True])
def test_spacenet_duplicate_warning_and_coordinate_bounds(tmp_path, outside):
    root = tmp_path / "data"
    root.mkdir()
    names = [
        "02_train_01",
        "02_train_02",
        "03_train_01",
        "02_publictest",
        "03_publictest",
    ]
    for name in names:
        area, suffix = name.split("_", 1)
        for modality, bands in [("optical", 3), ("sar", 1)]:
            with rasterio.open(
                root / f"{area}_{modality}_{suffix}.tif",
                "w",
                driver="GTiff",
                width=10,
                height=8,
                count=bands,
                dtype="uint8",
                crs="EPSG:32637",
                transform=rasterio.Affine(1, 0, 500000, 0, -1, 4000000),
            ) as image:
                image.write(
                    np.arange(bands * 8 * 10, dtype=np.uint8).reshape(bands, 8, 10)
                )
        if "train" in name:
            with (root / f"{area}_tiepoints_{suffix}.csv").open(
                "w", newline=""
            ) as stream:
                writer = csv.writer(stream)
                writer.writerow(["sar_row", "sar_col", "optical_row", "optical_col"])
                writer.writerow([1, 2, 3, 4])
                writer.writerow([1, 2, 3, 4])
                writer.writerow([80 if outside else 2, 2, 3, 4])
    output = tmp_path / "reports"
    summary = audit_spacenet(root, output)
    assert summary["issue_codes"]["duplicate_tiepoints"] == 3
    assert summary["issue_codes"]["officially_withheld_test_labels"] == 2
    assert summary["issue_codes"].get("out_of_bounds_tiepoints", 0) == (
        3 if outside else 0
    )
    anomalies = [
        json.loads(line)
        for line in (output / "spacenet9/anomalies.jsonl").read_text().splitlines()
    ]
    assert all(
        row["severity"] == "warning"
        for row in anomalies
        if row["code"] == "duplicate_tiepoints"
    )
    assert summary["decoded_ok"] == (2 if outside else 5)
