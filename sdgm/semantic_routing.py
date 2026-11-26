"""Frozen-front prototype distillation and bounded routing without GT candidates."""

import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .coarse import spatial_candidates


class SARPrototypeHead(nn.Module):
    def __init__(self, channels=128, prototypes=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 128, 1), nn.GELU(), nn.Conv2d(128, prototypes, 1)
        )

    def forward(self, descriptors):
        return self.layers(descriptors).softmax(dim=1)


def probability_normalize(probabilities):
    if not torch.isfinite(probabilities).all() or (probabilities < -1e-6).any():
        raise FloatingPointError("Invalid prototype probabilities")
    probabilities = probabilities.clamp_min(0)
    mass = probabilities.sum(1, keepdim=True)
    if (mass <= 0).any():
        raise ValueError("Empty prototype probability distribution")
    result = probabilities / mass
    if not torch.allclose(result.sum(1), torch.ones_like(result[:, 0]), atol=1e-5):
        raise AssertionError("Probability normalization failed")
    return result


def sample_native_map(feature, native_xy, image_hw=(256, 256)):
    """Pixel-center resize map, align_corners=False, border extrapolation.

    For an18x18 token field, native x maps to token (x+.5)*18/256-.5.
    For a32x32 structural field this is (x-3.5)/8. Border extension is
    explicit: existing image-valid queries need not lie inside feature centers.
    """
    if native_xy.ndim == 2:
        native_xy = native_xy[None].expand(feature.shape[0], -1, -1)
    scale = native_xy.new_tensor([image_hw[1], image_hw[0]])
    grid = 2 * (native_xy + 0.5) / scale - 1
    if not torch.isfinite(grid).all():
        raise FloatingPointError("Nonfinite sampling coordinates")
    return F.grid_sample(
        feature, grid[:, None], mode="bilinear", padding_mode="border", align_corners=False
    )[:, :, 0]


def distillation_loss(teacher, sar_probabilities, source_xy, geometry):
    teacher_query = probability_normalize(sample_native_map(teacher.detach(), source_xy))
    sar_query = probability_normalize(
        sample_native_map(sar_probabilities, geometry["gt_pixel_xy"])
    )
    divergence = (
        teacher_query
        * (teacher_query.clamp_min(1e-8).log() - sar_query.clamp_min(1e-8).log())
    ).sum(1)
    keep = geometry["supervised"]
    loss = divergence[keep].mean() if keep.any() else sar_probabilities.sum() * 0
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite distillation KL")
    return loss, divergence, sar_query, teacher_query


def semantic_bias(teacher_queries, sar_probability):
    # teacher_queries: B,M,Q; output B,Q,T. No coordinate labels enter this API.
    bias = 2 * torch.bmm(teacher_queries.transpose(1, 2), sar_probability.flatten(2)) - 1
    if not torch.isfinite(bias).all() or (bias.abs() > 1 + 1e-5).any():
        raise FloatingPointError("Semantic bias outside [-1,1]")
    return bias


@torch.no_grad()
def routed_candidates(scores, bias, target_valid, target_xy, weight, radius=24.0):
    """Reserve original top2, then greedily fill2 from fused scores with union NMS."""
    if weight < 0 or not np.isfinite(weight):
        raise ValueError("Routing weight must be finite and nonnegative")
    original = spatial_candidates(scores, target_valid, target_xy, count=4, nms_radius_px=radius)
    if weight == 0:
        return original
    if bias.shape != scores.shape or not torch.isfinite(bias).all():
        raise ValueError("Bias must match structural scores")
    fused = scores + weight * bias
    available = target_valid[:, None].expand_as(scores).clone()
    indices = [original["indices"][..., i] for i in range(2)]
    validity = [original["valid"][..., i] for i in range(2)]
    values = [original["scores"][..., i] for i in range(2)]
    for index, valid in zip(indices, validity):
        xy = target_xy[index.clamp_min(0)]
        suppress = ((target_xy[None, None] - xy[:, :, None]) ** 2).sum(-1) <= radius**2
        available &= ~(suppress & valid[..., None])
    for _ in range(2):
        index = fused.masked_fill(~available, torch.finfo(fused.dtype).min).argmax(-1)
        valid = available.any(-1)
        value = fused.gather(-1, index[..., None]).squeeze(-1)
        xy = target_xy[index]
        suppress = ((target_xy[None, None] - xy[:, :, None]) ** 2).sum(-1) <= radius**2
        available &= ~(suppress & valid[..., None])
        indices.append(torch.where(valid, index, -torch.ones_like(index)))
        validity.append(valid)
        values.append(torch.where(valid, value, torch.zeros_like(value)))
    index = torch.stack(indices, -1)
    valid = torch.stack(validity, -1)
    xy = target_xy[index.clamp_min(0)]
    return dict(
        indices=index,
        valid=valid,
        pixel_xy=torch.where(valid[..., None], xy, torch.zeros_like(xy)),
        scores=torch.stack(values, -1),
    )


def shuffle_semantic(probability, sample_ids, seed):
    result = []
    for p, sample_id in zip(probability, sample_ids):
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        permutation = torch.as_tensor(rng.permutation(p.shape[-1] * p.shape[-2]), device=p.device)
        result.append(p.flatten(1)[:, permutation].reshape_as(p))
    return torch.stack(result)


@torch.no_grad()
def spherical_kmeans(features, count, iterations, seed):
    features = F.normalize(features.float(), dim=1)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    # Seeded cosine-distance kmeans++ initialization, no validation-driven restarts.
    first = int(torch.randint(len(features), (1,), device=features.device, generator=generator))
    centers = [features[first]]
    nearest = (1 - features @ centers[0]).clamp_min(0)
    for _ in range(1, count):
        if float(nearest.sum()) <= 1e-8:
            raise ValueError("Not enough distinct directions for prototypes")
        index = torch.multinomial(nearest, 1, generator=generator).item()
        centers.append(features[index])
        nearest = torch.minimum(nearest, (1 - features @ centers[-1]).clamp_min(0))
    centers = torch.stack(centers)
    history = []
    for iteration in range(iterations):
        similarity = features @ centers.T
        best, labels = similarity.max(1)
        counts = torch.bincount(labels, minlength=count)
        sums = torch.zeros_like(centers).index_add_(0, labels, features)
        empty = counts == 0
        if empty.any():
            sums[empty] = features[best.argsort()[: int(empty.sum())]]
        centers = F.normalize(sums, dim=1)
        history.append(dict(iteration=iteration + 1, mean_cosine=float(best.mean()), empty=int(empty.sum())))
    labels = (features @ centers.T).argmax(1)
    return centers, torch.bincount(labels, minlength=count).cpu().tolist(), history
