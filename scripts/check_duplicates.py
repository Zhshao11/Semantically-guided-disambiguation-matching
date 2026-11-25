"""Exact pixel hashes and bounded near-duplicate/overlap screening.

Non-detection never establishes geographic or parent-scene independence.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from sdgm.io import load_datum, save_json, save_jsonl


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(image):
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
    coefficients = cv2.dct(small)[:8, :8].ravel()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return int.from_bytes(np.packbits(bits).tobytes(), "little")


def describe(path):
    datum = load_datum(path)
    optical = np.ascontiguousarray(datum[0][..., 0])
    sar = np.ascontiguousarray(datum[4])
    first = hashlib.sha256(optical.tobytes()).hexdigest()
    second = hashlib.sha256(sar.tobytes()).hexdigest()
    return dict(
        id=f"{'training' if 'training' in path.parts else 'testing'}/{path.stem}",
        path=str(path),
        optical_sha256=first,
        base_sar_sha256=second,
        joint_sha256=hashlib.sha256(optical.tobytes() + sar.tobytes()).hexdigest(),
        optical_phash=perceptual_hash(optical),
        base_sar_phash=perceptual_hash(sar),
    )


@lru_cache(maxsize=128)
def image_features(path, modality):
    datum = load_datum(path)
    image = datum[0][..., 0] if modality == "optical" else datum[4]
    keypoints, descriptors = cv2.SIFT_create(
        nfeatures=700, contrastThreshold=0.015
    ).detectAndCompute(image, None)
    points = np.array([k.pt for k in keypoints], dtype="float32").reshape(-1, 2)
    return image, points, descriptors


def verify(first, second, modality):
    a, points_a, descriptors_a = image_features(first["path"], modality)
    b, points_b, descriptors_b = image_features(second["path"], modality)
    result = dict(verified_near_overlap=False, good_matches=0, inliers=0)
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        return result
    neighbors = cv2.BFMatcher().knnMatch(descriptors_a, descriptors_b, k=2)
    matches = [
        x
        for pair in neighbors
        if len(pair) == 2
        for x, y in [pair]
        if x.distance < 0.75 * y.distance
    ]
    result["good_matches"] = len(matches)
    if len(matches) < 12:
        return result
    source = np.array([points_a[m.queryIdx] for m in matches])
    target = np.array([points_b[m.trainIdx] for m in matches])
    matrix, inlier = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2,
        maxIters=2000,
        confidence=0.995,
    )
    if matrix is None:
        return result
    inlier = inlier.ravel().astype(bool)
    result["inliers"] = int(inlier.sum())
    if inlier.sum() < 12:
        return result
    warped = cv2.warpAffine(a, matrix, (512, 512))
    mask = (
        cv2.warpAffine(
            np.ones(a.shape, "uint8"), matrix, (512, 512), flags=cv2.INTER_NEAREST
        )
        > 0
    )
    if mask.sum() < 1000:
        return result
    x, y = warped[mask].astype(float), b[mask].astype(float)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    ncc = float(x @ y / denominator) if denominator > 0 else 0.0
    coverage = float(cv2.contourArea(cv2.convexHull(source[inlier])) / (512 * 512))
    prediction = source @ matrix[:, :2].T + matrix[:, 2]
    error = float(
        np.median(np.linalg.norm(prediction[inlier] - target[inlier], axis=1))
    )
    result.update(
        affine=matrix.tolist(),
        inlier_fraction=float(inlier.mean()),
        source_hull_fraction=coverage,
        overlapping_fraction=float(mask.mean()),
        ncc=ncc,
        median_reprojection_error_px=error,
    )
    result["verified_near_overlap"] = bool(
        inlier.sum() >= 20
        and inlier.mean() >= 0.5
        and coverage >= 0.10
        and mask.mean() >= 0.25
        and ncc >= 0.85
        and error <= 2
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/zzvr.json")
    parser.add_argument("--max-near-pairs", type=int, default=250)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    cv2.setNumThreads(1)
    cv2.setRNGSeed(config["seed"])
    root = Path(config["whu"])
    output = Path("reports/provenance_20261125")
    output.mkdir(parents=True, exist_ok=True)
    archive_rows = []
    for archive in sorted(root.parent.glob("*.zip")):
        print("Hashing", archive, flush=True)
        archive_rows.append(
            dict(
                path=str(archive),
                name=archive.name,
                bytes=archive.stat().st_size,
                sha256=hash_file(archive),
                publisher_digest_available=False,
            )
        )
    save_json(output / "archive_hashes.json", archive_rows)
    files = sorted(root.glob("**/datum/*.npy"))
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, row in enumerate(pool.map(describe, files), 1):
            records.append(row)
            if index % 1000 == 0:
                print("Pixel hashes", index, len(files), flush=True)
    save_jsonl(output / "pixel_hash_manifest.jsonl", records)
    exact = []
    for field in ["optical_sha256", "base_sar_sha256", "joint_sha256"]:
        groups = defaultdict(list)
        for row in records:
            groups[row[field]].append(row["id"])
        for digest, members in groups.items():
            if len(members) > 1:
                exact.append(
                    dict(
                        kind=field,
                        sha256=digest,
                        members=members,
                        crosses_split=len({m.split("/")[0] for m in members}) > 1,
                    )
                )
    save_json(output / "exact_duplicates.json", exact)
    optical = np.array([r["optical_phash"] for r in records], dtype="uint64")
    sar = np.array([r["base_sar_phash"] for r in records], dtype="uint64")
    lookup = np.array([i.bit_count() for i in range(256)], "uint8")

    def hamming(values, query):
        return lookup[(values ^ query).view("uint8").reshape(len(values), 8)].sum(1)

    candidates = {}
    training = np.array(
        [i for i, r in enumerate(records) if r["id"].startswith("training/")]
    )
    for i, row in enumerate(records):
        first = hamming(optical, optical[i])
        second = hamming(sar, sar[i])
        mean = (first + second) / 2
        # Tight all-pair screen plus cross-split nearest neighbors.
        shortlist = np.where(
            (first <= 6) & (second <= 6) & (np.arange(len(records)) > i)
        )[0].tolist()
        if row["id"].startswith("testing/"):
            shortlist.extend(
                training[np.argsort(mean[training], kind="stable")[:3]].tolist()
            )
        for j in shortlist:
            a, b = sorted([i, j])
            if mean[j] <= 18:
                candidates[a, b] = (float(mean[j]), int(first[j]), int(second[j]))
    ordered = sorted(
        candidates,
        key=lambda pair: (
            records[pair[0]]["id"].split("/")[0]
            == records[pair[1]]["id"].split("/")[0],
            candidates[pair][0],
            pair,
        ),
    )
    verification = []
    for count, (i, j) in enumerate(ordered[: args.max_near_pairs], 1):
        first, second = records[i], records[j]
        a, b = verify(first, second, "optical"), verify(first, second, "sar")
        row = dict(
            first=first["id"],
            second=second["id"],
            phash_mean_distance=candidates[i, j][0],
            crosses_split=first["id"].split("/")[0] != second["id"].split("/")[0],
            optical=a,
            base_sar=b,
            paired_near_overlap=a["verified_near_overlap"]
            and b["verified_near_overlap"],
        )
        verification.append(row)
        if count % 25 == 0:
            print(
                "Near verification",
                count,
                min(len(ordered), args.max_near_pairs),
                flush=True,
            )
    save_jsonl(output / "near_duplicate_verification.jsonl", verification)
    summary = dict(
        samples=len(records),
        exact_duplicate_groups=len(exact),
        cross_split_exact_groups=sum(row["crosses_split"] for row in exact),
        near_shortlisted_pairs=len(ordered),
        near_verified_pairs=len(verification),
        verification_limit=args.max_near_pairs,
        cross_split_paired_near_overlaps=sum(
            r["crosses_split"] and r["paired_near_overlap"] for r in verification
        ),
        cross_split_optical_near_overlaps=sum(
            r["crosses_split"] and r["optical"]["verified_near_overlap"]
            for r in verification
        ),
        cross_split_sar_near_overlaps=sum(
            r["crosses_split"] and r["base_sar"]["verified_near_overlap"]
            for r in verification
        ),
        geographic_independence_proven=False,
        interpretation="Detected exact/near overlaps establish particular repeated content. Non-detection cannot establish parent-scene independence; pHash shortlist and verification thresholds miss other forms of overlap.",
    )
    save_json(output / "duplicate_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
