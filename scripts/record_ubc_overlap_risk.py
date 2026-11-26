"""Read-only risk disclosure; never change the frozen UBC evaluation pool."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from shapely.geometry import box
from shapely.strtree import STRtree

from sdgm.io import save_csv, save_json
from sdgm.raw_lineage import sha256_file


def main():
    source = Path("reports/os_ubc_20261126/ubcv2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_family_baseline.json")
    args = parser.parse_args()
    output = Path(json.loads(Path(args.config).read_text())["output"]) / "ubcv2_risk"
    names = [
        "manifest.jsonl",
        "frozen_policy.json",
        "frozen_exclusions.csv",
        "frozen_evaluation_pool.csv",
        "freeze_receipt.json",
    ]
    before = {name: sha256_file(source / name) for name in names}
    receipt = json.loads((source / "freeze_receipt.json").read_text())
    assert before["frozen_evaluation_pool.csv"] == receipt["pool_sha256"]
    with (source / "frozen_evaluation_pool.csv").open() as stream:
        pool = {r["id"] for r in csv.DictReader(stream)}
    rows = [
        json.loads(line)
        for line in (source / "manifest.jsonl").read_text().splitlines()
    ]
    by_crs = {}
    for row in rows:
        by_crs.setdefault(row["optical"]["crs"], []).append(row)
    overlaps = []
    for crs, group in sorted(by_crs.items()):
        polygons = [box(*row["optical"]["bounds"]) for row in group]
        tree = STRtree(polygons)
        for i, p in enumerate(polygons):
            for j in tree.query(p):
                j = int(j)
                if j <= i or group[i]["official_split"] == group[j]["official_split"]:
                    continue
                area = p.intersection(polygons[j]).area
                if area <= 1e-6:
                    continue
                a, b = group[i], group[j]
                small = min(p.area, polygons[j].area)
                overlaps.append(
                    dict(
                        first=a["id"],
                        second=b["id"],
                        first_split=a["official_split"],
                        second_split=b["official_split"],
                        split_pair="|".join(
                            sorted([a["official_split"], b["official_split"]])
                        ),
                        crs=crs,
                        first_city=a["region_id"],
                        second_city=b["region_id"],
                        overlap_area_m2=area,
                        fraction_first=area / p.area,
                        fraction_second=area / polygons[j].area,
                        fraction_smaller=area / small,
                        intersection_over_union=area
                        / (p.area + polygons[j].area - area),
                        touches_frozen_test_pool=a["id"] in pool or b["id"] in pool,
                    )
                )
    overlaps.sort(key=lambda r: (-r["fraction_smaller"], r["first"], r["second"]))
    expected = json.loads((source / "spatial_split_check.json").read_text())[
        "cross_official_split_overlapping_optical_footprints"
    ]
    assert len(overlaps) == expected == 88
    test_ids = sorted(
        {
            row[key]
            for row in overlaps
            for key in ["first", "second"]
            if row[key] in pool
        }
    )
    import numpy as np

    def q(values):
        return {
            str(k): float(np.percentile(values, k))
            for k in [0, 25, 50, 75, 90, 95, 100]
        }

    summary = dict(
        overlap_edges=len(overlaps),
        split_pair_counts=dict(Counter(r["split_pair"] for r in overlaps)),
        test_touching_edges=sum(r["touches_frozen_test_pool"] for r in overlaps),
        unique_frozen_test_pairs_touched=len(test_ids),
        frozen_test_pool_size=len(pool),
        touched_test_ids=test_ids,
        overlap_area_m2=q([r["overlap_area_m2"] for r in overlaps]),
        fraction_smaller=q([r["fraction_smaller"] for r in overlaps]),
        intersection_over_union=q([r["intersection_over_union"] for r in overlaps]),
        fraction_smaller_bins=dict(
            under_10_percent=sum(r["fraction_smaller"] < 0.1 for r in overlaps),
            from_10_to_50_percent=sum(
                0.1 <= r["fraction_smaller"] < 0.5 for r in overlaps
            ),
            at_least_50_percent=sum(r["fraction_smaller"] >= 0.5 for r in overlaps),
        ),
        source_hashes=before,
        protocol="positive-area intersections of native optical GeoTIFF footprints in matching metre-based UTM CRS; boundary touch excluded",
        meaning="spatial overlap disclosure only; does not establish identical acquisition, temporal content or model leakage; no deletions",
        model_run=False,
        prior_analysis_exposure=True,
        blind_test=False,
        geographic_independence_claimed=False,
        pool_modified=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "all_88_overlap_edges.csv", overlaps)
    save_json(output / "risk_summary.json", summary)
    after = {name: sha256_file(source / name) for name in names}
    assert before == after
    save_json(
        output / "unchanged_pool_verification.json",
        dict(before=before, after=after, all_unchanged=True, new_model_run=False),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
