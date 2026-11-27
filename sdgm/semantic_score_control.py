"""Probability distance scoring without changing representations or normalization."""

import torch


def probability_scores(query, target):
    """query B,M,Q; target B,M,T, already original normalized probabilities.

    No normalization/temperature is introduced here. Include query constant in
    the exported actual distance score; target correction is exported for audit.
    """
    dot = torch.bmm(query.transpose(1, 2), target)
    pnorm = query.square().sum(1)
    qnorm = target.square().sum(1)
    dist = dot - 0.5 * qnorm[:, None, :] - 0.5 * pnorm[:, :, None]
    if not torch.isfinite(dot).all() or not torch.isfinite(dist).all():
        raise FloatingPointError("Nonfinite fixed-representation score")
    return dict(dot=dot, dist=dist, query_norm2=pnorm, target_norm2=qnorm)
