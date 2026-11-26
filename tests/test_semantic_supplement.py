import numpy as np
import pytest
import torch

from sdgm.coarse import spatial_candidates
from sdgm.semantic_routing import routed_candidates
from sdgm.semantic_supplement import (
    supplement_from_full_grid,
    paired_counts,
    paired_family_intervals,
)


def test_full_grid_semantic_peaks_can_have_lowest_structural_scores():
    xy = torch.stack([torch.arange(8) * 32, torch.zeros(8)], 1).float()
    scores = torch.arange(8, 0, -1).float()[None, None]
    valid = torch.ones(1, 8, dtype=torch.bool)
    prefix = spatial_candidates(scores, valid, xy)
    semantics = torch.tensor([[[0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.8, 0.9]]])
    result = supplement_from_full_grid(prefix, semantics, valid, xy)
    assert prefix["indices"].tolist() == [[[0, 1, 2, 3]]]
    assert result["indices"].tolist() == [[[0, 1, 7, 6]]]
    # Semantic7/6 would be lost under a structural top4 prescreen.
    with pytest.raises(ValueError, match="complete target grid"):
        supplement_from_full_grid(prefix, semantics[..., :4], valid, xy)


def test_same_sequential_NMS_as_fusion_and_no_invalid_fill():
    torch.manual_seed(61)
    xy = torch.stack([torch.arange(16) * 8, torch.zeros(16)], 1).float()
    scores = torch.randn(2, 3, 16)
    bias = torch.rand_like(scores) * 2 - 1
    valid = torch.rand(2, 16) > 0.3
    prefix = spatial_candidates(scores, valid, xy)
    expected = routed_candidates(scores, bias, valid, xy, 1.0)
    actual = supplement_from_full_grid(prefix, scores + bias, valid, xy)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    valid[:] = False
    valid[:, 0] = True
    prefix = spatial_candidates(scores, valid, xy)
    sparse = supplement_from_full_grid(prefix, bias, valid, xy)
    assert (sparse["indices"][..., 1:] == -1).all()
    assert (~sparse["valid"][..., 1:]).all()


def test_rescue_denominator_is_original_misses_and_invalid_queries_excluded():
    a = np.array([2.0, 20.0, 30.0, 4.0, 40.0])
    s = np.array([20.0, 3.0, 40.0, 4.0, 2.0])
    result = paired_counts(a, s, np.array([True, True, True, True, False]))
    assert result["new_hits"] == 1 and result["lost_hits"] == 1
    assert result["rescue_rate"] == 0.5 and result["loss_rate"] == 0.5
    assert result["net_hits"] == 0 and result["recall_at_4"] == 0.5
    assert paired_counts(a, s, np.zeros(5, dtype=bool))["rescue_rate"] is None


def test_family_resampling_recomputes_query_weighted_paired_ratios():
    rows = [
        dict(
            effective_queries=10,
            structure_hits=5,
            fusion_hits=5,
            semantic_hits=6,
            fusion_new_hits=0,
            fusion_lost_hits=0,
            semantic_new_hits=2,
            semantic_lost_hits=1,
        )
    ] * 3
    result = paired_family_intervals(rows, seed=7, replicates=50)
    np.testing.assert_allclose(
        result["interval95"]["semantic_minus_structure_recall4"], [0.1, 0.1]
    )
    np.testing.assert_allclose(result["interval95"]["semantic_rescue_rate"], [0.4, 0.4])
