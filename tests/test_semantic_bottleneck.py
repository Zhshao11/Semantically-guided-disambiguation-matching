import torch

from sdgm.semantic_bottleneck import (
    hard_negative_peaks,
    comparison_stats,
    full_grid_rank,
    pairwise_comparisons,
)


def test_hard_negatives_exclude_positive_and_use_sequential_spatial_nms():
    xy = torch.stack([torch.arange(8) * 8, torch.zeros(8)], 1).float()
    scores = torch.tensor([[[100.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]]])
    positive = torch.zeros_like(scores, dtype=torch.bool)
    positive[..., 0] = True
    idx, valid = hard_negative_peaks(
        scores, positive, torch.ones(1, 8, dtype=torch.bool), xy, count=4
    )
    assert idx.tolist() == [[[1, 5, -1, -1]]]
    assert valid.sum() == 2


def test_rank_bounds_and_delta_ties_never_infer_win_from_index_order():
    scores = torch.tensor([[[0.5, 0.5, 0.2, 0.1], [0.4, 0.5, 0.2, 0.1]]])
    p = torch.tensor([[[True, False, False, False], [True, False, False, False]]])
    n = ~p
    result = comparison_stats(scores, p, n)
    assert result["rank_min"].tolist() == [[1, 2]] and result["rank_max"].tolist() == [
        [2, 2]
    ]
    assert result["tie"].tolist() == [[True, False]]
    assert result["loss"].tolist() == [[False, True]] and not result["win"].any()
    assert not result["rank1_guaranteed"].any()
    order = torch.tensor([1, 0, 2, 3])
    permuted = comparison_stats(scores[..., order], p[..., order], n[..., order])
    assert all(torch.equal(result[k], permuted[k]) for k in result)


def test_empty_P_and_insufficient_N_are_explicit():
    scores = torch.zeros(1, 2, 4)
    p = torch.zeros_like(scores, dtype=torch.bool)
    n = p.clone()
    p[:, 1, 0] = True
    r = comparison_stats(scores, p, n)
    assert not r["valid"].any() and r["ranking_valid"].tolist() == [[False, True]]
    assert r["rank_min"][0, 0] == 0


def test_full_grid_top4_tie_bounds_and_pairwise_ties():
    score = torch.zeros(1, 1, 8)
    p = torch.zeros_like(score, dtype=torch.bool)
    p[..., 7] = True
    r = full_grid_rank(score, p, torch.ones(1, 8, dtype=torch.bool))
    assert r["rank_min"].item() == 1 and r["rank_max"].item() == 8
    assert r["top4_possible"].item() and not r["top4_guaranteed"].item()
    pair = pairwise_comparisons(
        score, p, torch.tensor([[[0, 1]]]), torch.ones(1, 1, 2, dtype=torch.bool)
    )
    assert pair["pairwise_tie"].item() == 2 and pair["pairwise_count"].item() == 2
    # If several positions are positive, ties among them do not lower the
    # ranking of the best/first positive from that neighborhood.
    p[..., :7] = True
    r = full_grid_rank(score, p, torch.ones(1, 8, dtype=torch.bool))
    assert r["rank_max"].item() == 1 and r["top4_guaranteed"].item()
