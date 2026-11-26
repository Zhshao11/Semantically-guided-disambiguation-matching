import json
import numpy as np
import torch
from sdgm.family_evaluation import evaluate_family


class KnownScores(torch.nn.Module):
    image_hw = (256, 256)
    source_pixel_xy = torch.tensor(
        [[20.0, 20.0], [60.0, 20.0], [100.0, 20.0], [140.0, 20.0]]
    )
    target_pixel_xy = source_pixel_xy.clone()

    def forward(self, optical, sar):
        return dict(
            scores=torch.tensor([4.0, 3.0, 2.0, 1.0])[None, None].expand(
                len(optical), 4, 4
            )
        )


class FixedPairs:
    split = "val"
    records = [dict(sample_id="one"), dict(sample_id="outside")]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        matrix = torch.eye(3)
        if index:
            matrix[0, 2] = 1000
        return dict(
            id=self.records[index]["sample_id"],
            family_id="F" + str(index),
            optical=torch.zeros(1, 256, 256),
            sar=torch.zeros(1, 256, 256),
            matrix=matrix,
            source_support=torch.ones(1, 256, 256, dtype=torch.bool),
            target_support=torch.ones(1, 256, 256, dtype=torch.bool),
        )


def test_per_pair_aggregate_retains_zero_supervision_pairs_and_masks_distances(
    tmp_path,
):
    config = dict(
        batch_size=2,
        positive_radius_px=8.0,
        recall_radius_px=8.0,
        candidate_count=4,
        nms_radius_px=24.0,
    )
    result = evaluate_family(
        KnownScores(),
        FixedPairs(),
        config,
        torch.device("cpu"),
        tmp_path,
        "fixture",
        details=True,
    )
    assert result["pairs"] == 2 and result["pairs_without_supervised_queries"] == 1
    assert (
        result["supervised_queries"] == 4
        and result["recall_at_1"] == 0.25
        and result["recall_at_4"] == 1
    )
    assert result["nearest_candidate_distance_px"]["100"] == 0
    data = np.load(tmp_path / "fixture_distances.npz")
    assert data["supervised"][0].all() and not data["supervised"][1].any()
    assert (data["nearest_distance_px"][data["supervised"]] == 0).all()
    assert (
        json.loads((tmp_path / "fixture_metrics.json").read_text())["masked"][
            "gt_outside"
        ]
        == 4
    )
