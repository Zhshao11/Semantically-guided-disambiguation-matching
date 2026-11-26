import json
import pytest
from sdgm.os_family import group_sources, choose_native_representatives


def row(
    name,
    size=512,
    fold="train",
    optical_hash=None,
    sar_hash=None,
    joint_hash=None,
    parents=(),
):
    def record(digest):
        return dict(
            width=size,
            height=size,
            channels=1,
            dtype=["uint8"],
            minimum=[0],
            maximum=[255],
            pixel_sha256=digest,
            path=f"/{name}/{digest}.png",
            sha256=digest,
        )

    return dict(
        id=name,
        subset=f"OSdataset/{size}/{fold}",
        official_split=fold,
        eligible_by_frozen_file_rules=True,
        optical=record(optical_hash or name + "-O"),
        sar=record(sar_hash or name + "-S"),
        joint_pixel_sha256=joint_hash or name + "-J",
        exact_pixel_crop_parents=[dict(parent_id=p, x=0, y=0) for p in parents],
    )


def test_transitive_parent_duplicate_and_overlap_links_merge_all_exposed_relatives():
    rows = [
        row("P"),
        row("P-copy", optical_hash="P-O", fold="val"),
        row("C", 256, parents=["P"]),
        row("farther", fold="test"),
    ]
    assignments, families, edges = group_sources(
        rows,
        {"C"},
        [
            dict(
                first="P-copy",
                second="farther",
                evidence="independent overlap proof fixture",
            )
        ],
    )
    assert len(families) == 1 and families[0]["split"] == "train"
    assert {r["split"] for r in assignments} == {"train"}
    assert "exact_optical_pixels" in families[0]["grouping_evidence"]
    assert "verified_larger_scene_or_overlap" in families[0]["grouping_evidence"]


def test_conflicting_folds_and_unresolved_parents_are_quarantined_not_randomly_split():
    rows = [
        row("A", fold="train"),
        row("B", fold="val", joint_hash="A-J"),
        row("orphan", 256, parents=[]),
        row("region", fold="unspecified"),
    ]
    assignments, families, edges = group_sources(rows, set())
    assert all(r["split"] == "quarantine" for r in assignments)
    assert choose_native_representatives(rows, assignments) == []
    with pytest.raises(ValueError, match="evidence"):
        group_sources(rows, set(), [dict(first="A", second="B")])


def test_ambiguous_parent_aliases_and_native_representative_dedup_remain_in_one_family():
    rows = [
        row("P"),
        row("Q"),
        row("child", 256, parents=["P", "Q"]),
        row("copy", 256, joint_hash="child-J", parents=["Q"]),
    ]
    assignments, families, edges = group_sources(rows, {"child"})
    child = next(r for r in assignments if r["sample_id"] == "child")
    assert child["parent_512_id"] is None and child["parent_512_ids"] == ["P", "Q"]
    assert (
        len(families) == 1
        and len(choose_native_representatives(rows, assignments)) == 1
    )
    assert {r["family_id"] for r in assignments} == {families[0]["family_id"]}
    # Augmented samples inherit this frozen raw-family ID; no new group is created.
    augmented = dict(child, sample_id="child/epoch17")
    assert (
        augmented["family_id"] == child["family_id"] and augmented["split"] == "train"
    )
