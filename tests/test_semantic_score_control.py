import torch

from sdgm.semantic_score_control import probability_scores
from sdgm.coarse import spatial_candidates
from sdgm.semantic_supplement import supplement_from_full_grid


def test_distance_identity_and_no_mutation_or_extra_normalization():
    torch.manual_seed(8)
    p = torch.rand(2, 32, 5)
    p /= p.sum(1, keepdim=True)
    q = torch.rand(2, 32, 9)
    q /= q.sum(1, keepdim=True)
    before_p = p.clone()
    before_q = q.clone()
    s = probability_scores(p, q)
    direct = -0.5 * (
        p.transpose(1, 2)[:, :, None, :] - q.transpose(1, 2)[:, None, :, :]
    ).square().sum(-1)
    torch.testing.assert_close(s["dist"], direct, atol=1e-7, rtol=1e-5)
    assert torch.equal(p, before_p) and torch.equal(q, before_q)
    # The same algebra must hold even for nonunit vectors: no hidden normalize.
    s = probability_scores(p * 2, q * 3)
    direct = -0.5 * (
        (p * 2).transpose(1, 2)[:, :, None, :] - (q * 3).transpose(1, 2)[:, None, :, :]
    ).square().sum(-1)
    torch.testing.assert_close(s["dist"], direct, atol=1e-7, rtol=1e-5)


def test_target_sharpness_can_reverse_dot_rank_without_new_representation():
    p = torch.tensor([[[0.6], [0.4]]])
    q = torch.tensor([[[1.0, 0.6], [0.0, 0.4]]])
    s = probability_scores(p, q)
    assert s["dot"].argmax(-1).item() == 0
    assert s["dist"].argmax(-1).item() == 1
    torch.testing.assert_close(s["dist"][..., 1], torch.zeros(1, 1), atol=1e-7, rtol=0)


def test_distance_full_grid_supplement_keeps_structural_prefix_and_budget():
    xy = torch.stack([torch.arange(8) * 32, torch.zeros(8)], 1).float()
    structural = torch.arange(8, 0, -1).float()[None, None]
    prefix = spatial_candidates(structural, torch.ones(1, 8, dtype=torch.bool), xy)
    p = torch.tensor([[[0.6], [0.4]]])
    q = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.55],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.45],
            ]
        ]
    )
    s = probability_scores(p, q)
    c = supplement_from_full_grid(
        prefix, s["dist"], torch.ones(1, 8, dtype=torch.bool), xy
    )
    assert c["indices"].tolist() == [[[0, 1, 6, 7]]]
