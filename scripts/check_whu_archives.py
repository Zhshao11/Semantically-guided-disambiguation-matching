"""Compare extracted file names and sizes with the two supplied WHU ZIP indices."""

import argparse
import json
import zipfile
from pathlib import Path

from sdgm.io import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zzvr.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    extracted = Path(config["whu"])
    results = []
    for split in ["training", "testing"]:
        archive = extracted.parent / f"{split}.zip"
        if not archive.exists():
            results.append(dict(split=split, status="source_archive_unavailable"))
            continue
        with zipfile.ZipFile(archive) as z:
            files = [info for info in z.infolist() if not info.is_dir()]
            missing = []
            mismatch = []
            for info in files:
                relative = Path(info.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive index path")
                path = extracted / split / relative
                if not path.exists():
                    missing.append(info.filename)
                elif path.stat().st_size != info.file_size:
                    mismatch.append(info.filename)
            expected = {info.filename for info in files}
            actual = {
                str(path.relative_to(extracted / split))
                for path in (extracted / split).rglob("*")
                if path.is_file()
            }
            extra = sorted(actual - expected)
            results.append(
                dict(
                    split=split,
                    archive=str(archive),
                    archive_files=len(files),
                    extracted_files=len(actual),
                    missing=missing,
                    size_mismatch=mismatch,
                    extra=extra,
                    passed=not (missing or mismatch or extra),
                    crc_verification_performed=False,
                )
            )
    report = dict(
        results=results,
        scope="Archive index names and uncompressed sizes. Raster/flow/datum payloads are independently fully decoded by audit_datasets.py.",
    )
    save_json(Path(config["output"]) / "whu/archive_index.json", report)
    print(json.dumps(report, indent=2))
    return int(any(not row.get("passed", False) for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
