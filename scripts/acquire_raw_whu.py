"""Download original release files without changing bands or pixel values."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gdown
import rasterio


def acquire(job):
    row, root = job
    path = root / row["modality"] / row["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for attempt in range(3):
        try:
            if not path.exists():
                result = gdown.download(
                    id=row["source_id"],
                    output=str(path),
                    quiet=True,
                    use_cookies=False,
                    resume=True,
                )
                if result is None:
                    raise OSError("No downloaded file returned")
            # Decode all blocks before certifying a completed transfer.
            with rasterio.open(path) as image:
                for _, window in image.block_windows(1):
                    image.read(window=window)
                profile = dict(
                    width=image.width, height=image.height, bands=image.count
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return dict(
                **row,
                path=str(path),
                sha256=digest.hexdigest(),
                bytes=path.stat().st_size,
                status="complete",
                seconds=time.time() - started,
                **profile,
            )
        except Exception as error:
            if attempt == 2:
                return dict(**row, path=str(path), status="error", error=str(error))
    raise RuntimeError("Unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="configs/raw_whu_catalog.json")
    parser.add_argument(
        "--root", default="/mnt/sda/zhushihao/data/vis_sar/WHU_OPT_SAR_original"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--parents",
        required=True,
        help="Explicit comma-separated pilot parents; no implicit full-release download",
    )
    parser.add_argument("--modalities", default="optical,sar")
    args = parser.parse_args()
    root = Path(args.root)
    rows = json.loads(Path(args.catalog).read_text())
    parents = sorted({row["parent_image_id"] for row in rows})
    priority = {parents[i] for i in [0, 20, 40, 60, 80, 99]}
    selected_parents = set(args.parents.split(",")) if args.parents else set(parents)
    modalities = set(args.modalities.split(","))
    chosen = [
        row
        for row in rows
        if row["parent_image_id"] in selected_parents and row["modality"] in modalities
    ]
    chosen.sort(
        key=lambda row: (
            row["parent_image_id"] not in priority,
            row["parent_image_id"],
            row["modality"],
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / "acquisition.jsonl").open("a") as log,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        for row in pool.map(acquire, [(row, root) for row in chosen]):
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
