"""GT-assisted diagnostics only; never used by deployment candidate extraction."""

import torch


@torch.no_grad()
def hard_negative_peaks(
    scores, positive, target_valid, target_xy, count=16, radius=24.0
):
    available = target_valid[:, None].expand_as(scores).clone() & ~positive
    indices, flags = [], []
    for _ in range(count):
        index = scores.masked_fill(~available, torch.finfo(scores.dtype).min).argmax(-1)
        valid = available.any(-1)
        center = target_xy[index]
        suppressed = ((target_xy[None, None] - center[:, :, None]) ** 2).sum(
            -1
        ) <= radius**2
        available &= ~(suppressed & valid[..., None])
        indices.append(torch.where(valid, index, -torch.ones_like(index)))
        flags.append(valid)
    return torch.stack(indices, -1), torch.stack(flags, -1)


@torch.no_grad()
def comparison_stats(scores, positive, negative, atol=1e-6):
    """Best-P rank interval, best-P/max-N comparison, all-P/all-N comparisons."""
    p_count = positive.sum(-1)
    n_count = negative.sum(-1)
    pmax = scores.masked_fill(~positive, -torch.inf).max(-1).values
    nmax = scores.masked_fill(~negative, -torch.inf).max(-1).values
    competition = positive | negative
    valid = (p_count > 0) & (n_count > 0)
    ranking_valid = p_count > 0
    safe_p = torch.where(ranking_valid, pmax, torch.zeros_like(pmax))
    minimum_rank = 1 + ((scores > safe_p[..., None] + atol) & competition).sum(-1)
    # Rank the first positive within a tied block, not an arbitrarily indexed
    # positive. Other tied positives cannot worsen the first-positive rank.
    maximum_rank = 1 + ((scores >= safe_p[..., None] - atol) & negative).sum(-1)
    delta = torch.where(valid, pmax - nmax, torch.zeros_like(pmax))
    return dict(
        positive_count=p_count,
        negative_count=n_count,
        valid=valid,
        ranking_valid=ranking_valid,
        positive_max=torch.where(ranking_valid, pmax, torch.zeros_like(pmax)),
        negative_max=torch.where(n_count > 0, nmax, torch.zeros_like(nmax)),
        delta=delta,
        win=valid & (delta > atol),
        loss=valid & (delta < -atol),
        tie=valid & (delta.abs() <= atol),
        exact_tie=valid & (delta == 0),
        rank_min=torch.where(
            ranking_valid, minimum_rank, torch.zeros_like(minimum_rank)
        ),
        rank_max=torch.where(
            ranking_valid, maximum_rank, torch.zeros_like(maximum_rank)
        ),
        rank1_guaranteed=ranking_valid & (maximum_rank == 1),
        rank1_possible=ranking_valid & (minimum_rank == 1),
    )


@torch.no_grad()
def pairwise_comparisons(scores, positive, negative_indices, negative_valid, atol=1e-6):
    """Every common P versus every common selected N, ties separate."""
    nscore = scores.gather(-1, negative_indices.clamp_min(0))
    difference = scores[..., None] - nscore[:, :, None, :]
    keep = positive[..., None] & negative_valid[:, :, None, :]
    return dict(
        pairwise_count=keep.sum((-1, -2)),
        pairwise_win=((difference > atol) & keep).sum((-1, -2)),
        pairwise_loss=((difference < -atol) & keep).sum((-1, -2)),
        pairwise_tie=((difference.abs() <= atol) & keep).sum((-1, -2)),
    )


@torch.no_grad()
def full_grid_rank(scores, positive, target_valid, atol=1e-6):
    support = target_valid[:, None].expand_as(scores)
    valid = positive.any(-1)
    best = scores.masked_fill(~positive, -torch.inf).max(-1).values
    best = torch.where(valid, best, torch.zeros_like(best))
    rank_min = 1 + ((scores > best[..., None] + atol) & support).sum(-1)
    rank_max = 1 + ((scores >= best[..., None] - atol) & support & ~positive).sum(-1)
    # Top4 attainable and guaranteed across the boundary-tie block, independent
    # of stable sort/argmax index order. Count P members within the tie block.
    above = (scores > best[..., None] + atol) & support
    tied = (
        (scores >= best[..., None] - atol)
        & (scores <= best[..., None] + atol)
        & support
    )
    negative_tied = (tied & ~positive).sum(-1)
    top4_possible = valid & (rank_min <= 4)
    top4_guaranteed = valid & (above.sum(-1) + negative_tied < 4)
    return dict(
        valid=valid,
        rank_min=torch.where(valid, rank_min, torch.zeros_like(rank_min)),
        rank_max=torch.where(valid, rank_max, torch.zeros_like(rank_max)),
        top4_possible=top4_possible,
        top4_guaranteed=top4_guaranteed,
    )
