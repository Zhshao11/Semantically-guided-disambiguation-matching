"""GT-free masked multiscale composition followed by two independent point MLPs."""

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_routing import sample_native_map


def orthogonal_projection(seed, channels=128, extra=32):
    if channels < extra:
        raise ValueError("Need channels >= extra for row-orthogonal projection")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(channels, extra, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.sign(r.diag())
    signs[signs == 0] = 1
    return (q * signs).T.contiguous().float()


def masked_mean_map(x, mask, window, eps=1e-6):
    if window < 1 or window % 2 != 1:
        raise ValueError("Window must be positive odd")
    if mask.shape != (x.shape[0], 1, *x.shape[-2:]):
        raise ValueError("Mask/feature spatial shape mismatch")
    clean = torch.where(mask.bool(), x, torch.zeros_like(x))
    count = F.avg_pool2d(
        mask.to(x.dtype), window, stride=1, padding=window // 2, divisor_override=1
    )
    total = F.avg_pool2d(
        clean, window, stride=1, padding=window // 2, divisor_override=1
    )
    pooled = total / (count + eps)
    pooled = torch.where(mask.bool(), pooled, torch.zeros_like(pooled))
    return pooled, count / (window * window)


def multiscale_maps(feature, extra, mask, context=True):
    if feature.shape[0] != extra.shape[0] or feature.shape[-2:] != extra.shape[-2:]:
        raise ValueError("Feature/extra spatial shape mismatch")
    valid = mask.bool()
    f = F.normalize(
        torch.where(valid, feature, torch.zeros_like(feature)), dim=1, eps=1e-6
    )
    e = F.normalize(torch.where(valid, extra, torch.zeros_like(extra)), dim=1, eps=1e-6)
    maps = [(f, e)]
    fractions = [mask.to(feature.dtype)]
    for window in [3, 5, 9]:
        if context:
            pf, frac = masked_mean_map(f, mask, window)
            pe, _ = masked_mean_map(e, mask, window)
            maps.append((pf, pe))
        else:
            # Only mask counts are inspected for diagnostics. Point-control
            # feature values outside the center are never pooled or sampled.
            count = F.avg_pool2d(
                mask.to(f.dtype),
                window,
                stride=1,
                padding=window // 2,
                divisor_override=1,
            )
            frac = count / (window * window)
            maps.append((f, e))
        fractions.append(frac)
    return maps, torch.cat(fractions, 1)


def sample_information_blocks(maps, native_xy, image_hw=(256, 256)):
    blocks = []
    for f, e in maps:
        # Normalize each modality component AFTER interpolating its scale map.
        # Concatenate exactly [norm(A_k(F)),norm(A_k(E))] as specified.
        sf = F.normalize(sample_native_map(f, native_xy, image_hw), dim=1, eps=1e-6)
        se = F.normalize(sample_native_map(e, native_xy, image_hw), dim=1, eps=1e-6)
        blocks.append(torch.cat([sf, se], 1))
    return torch.cat(blocks, 1).transpose(1, 2)


class ContextDescriptorHead(nn.Module):
    def __init__(self, channels=128, extra=32):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(4 * (channels + extra), 256), nn.GELU(), nn.Linear(256, 128)
            )

        self.optical_head = mlp()
        self.sar_head = mlp()

    def forward(self, optical_maps, sar_maps, source_xy, target_xy):
        optical_input = sample_information_blocks(optical_maps, source_xy)
        sar_input = sample_information_blocks(sar_maps, target_xy)
        first = F.normalize(self.optical_head(optical_input), dim=-1, eps=1e-6)
        second = F.normalize(self.sar_head(sar_input), dim=-1, eps=1e-6)
        scores = torch.bmm(first, second.transpose(1, 2))
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite learned context descriptors")
        return dict(scores=scores, optical=first, sar=second)


def full_grid_pair_mass_loss(cosine_scores, geometry, temperature=0.1):
    if temperature <= 0:
        raise ValueError("Positive fixed loss temperature required")
    support = geometry["target_valid"][:, None].expand_as(cosine_scores)
    positive = geometry["positive"].detach() & support
    keep = geometry["supervised"].detach() & positive.any(-1) & support.any(-1)
    # Only valid rows reach logsumexp to avoid all-negative-infinity autograd.
    per_query = torch.zeros(
        cosine_scores.shape[:2], device=cosine_scores.device, dtype=cosine_scores.dtype
    )
    if keep.any():
        score = cosine_scores[keep] / temperature
        num = torch.logsumexp(score.masked_fill(~positive[keep], -torch.inf), -1)
        den = torch.logsumexp(score.masked_fill(~support[keep], -torch.inf), -1)
        per_query[keep] = den - num
    counts = keep.sum(-1)
    valid_pair = counts > 0
    pair_loss = per_query.sum(-1) / counts.clamp_min(1)
    loss = pair_loss[valid_pair].mean() if valid_pair.any() else cosine_scores.sum() * 0
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite per-pair full-grid loss")
    return loss, dict(
        supervised=keep,
        per_query=per_query,
        per_pair=pair_loss,
        valid_pair=valid_pair,
        empty_positive=~positive.any(-1),
        empty_valid_target=~support.any(-1),
    )
