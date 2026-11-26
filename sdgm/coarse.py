"""Structural-only coarse retrieval with explicit pixel-center geometry."""

import torch
from torch import nn
from torch.nn import functional as functional


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )

    def forward(self, feature):
        return functional.silu(feature + self.layers(feature))


def input_adapter():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1, bias=False),
        nn.GroupNorm(4, 16),
        nn.SiLU(),
    )


class StructuralRetriever(nn.Module):
    """Two modality adapters and shared encoder producing Bx128x(H/8)x(W/8).

    Three 2x2 average pools establish feature centers at 8*i+3.5 in input
    pixels. All convolutions use stride one, odd kernels and symmetric padding.
    """

    def __init__(self, temperature=0.1, image_hw=(512, 512), source_grid_hw=None):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.image_hw = tuple(image_hw)
        if len(self.image_hw) != 2 or any(v < 8 or v % 8 for v in self.image_hw):
            raise ValueError(
                "Native image dimensions must be positive multiples of eight"
            )
        self.feature_hw = tuple(v // 8 for v in self.image_hw)
        if source_grid_hw is None:
            source_grid_hw = tuple(max(1, round(v * 24 / 64)) for v in self.feature_hw)
        if len(source_grid_hw) != 2 or any(
            q < 1 or q > n for q, n in zip(source_grid_hw, self.feature_hw)
        ):
            raise ValueError("Source grid must fit the native feature map")
        self.optical_adapter = input_adapter()
        self.sar_adapter = input_adapter()
        stages = []
        channels = 16
        for output_channels in [32, 64, 128]:
            stages.extend(
                [
                    nn.AvgPool2d(2),
                    nn.Conv2d(channels, output_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(8, output_channels),
                    nn.SiLU(),
                    ResidualBlock(output_channels),
                ]
            )
            channels = output_channels
        self.encoder = nn.Sequential(*stages)
        axis_y = (
            torch.linspace(0, self.feature_hw[0] - 1, source_grid_hw[0]).round().long()
        )
        axis_x = (
            torch.linspace(0, self.feature_hw[1] - 1, source_grid_hw[1]).round().long()
        )
        yy, xx = torch.meshgrid(axis_y, axis_x, indexing="ij")
        self.register_buffer(
            "source_feature_xy", torch.stack([xx, yy], -1).reshape(-1, 2)
        )
        target_y, target_x = torch.meshgrid(
            torch.arange(self.feature_hw[0]),
            torch.arange(self.feature_hw[1]),
            indexing="ij",
        )
        self.register_buffer(
            "target_feature_xy", torch.stack([target_x, target_y], -1).reshape(-1, 2)
        )

    @property
    def source_pixel_xy(self):
        return self.source_feature_xy.float() * 8 + 3.5

    @property
    def target_pixel_xy(self):
        return self.target_feature_xy.float() * 8 + 3.5

    def forward(self, optical, sar):
        if (
            optical.shape != sar.shape
            or optical.ndim != 4
            or optical.shape[1:] != (1, *self.image_hw)
        ):
            raise ValueError(
                f"Expected matching Bx1x{self.image_hw[0]}x{self.image_hw[1]} native single-band inputs"
            )
        first = functional.normalize(
            self.encoder(self.optical_adapter(optical)), dim=1, eps=1e-6
        )
        second = functional.normalize(
            self.encoder(self.sar_adapter(sar)), dim=1, eps=1e-6
        )
        indices = (
            self.source_feature_xy[:, 1] * self.feature_hw[1]
            + self.source_feature_xy[:, 0]
        )
        queries = first.flatten(2).index_select(2, indices).transpose(1, 2)
        scores = torch.bmm(queries, second.flatten(2)) / self.temperature
        return dict(optical_descriptors=first, sar_descriptors=second, scores=scores)


def apply_affine(matrix, points):
    return torch.einsum("bij,nj->bni", matrix[:, :2, :2], points) + matrix[
        :, :2, 2, None
    ].transpose(1, 2)


def sample_support(mask, points, image_hw):
    """Require complete bilinear support at zero-based original pixel centers."""
    if mask.ndim != 4 or tuple(mask.shape[1:]) != (1, *tuple(image_hw)):
        raise ValueError("Support mask must match native Bx1xHxW input")
    if points.ndim == 2:
        points = points[None].expand(mask.shape[0], -1, -1)
    scale = points.new_tensor([image_hw[1], image_hw[0]])
    grid = 2 * (points + 0.5) / scale - 1
    finite = torch.isfinite(grid).all(-1)
    grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
    value = functional.grid_sample(
        mask.to(dtype=points.dtype),
        grid[:, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0, 0]
    return finite & (value >= 1 - 1e-6)


def supervision_geometry(
    matrix,
    source_pixel_xy,
    target_pixel_xy,
    positive_radius_px=8.0,
    image_hw=(512, 512),
    source_support=None,
    target_support=None,
):
    """Physical support is derived from the affine map and image extent.

    Zero-valued image pixels remain eligible. The native optical/base-SAR
    relation is treated only as the derivative-label convention for this smoke.
    """
    if positive_radius_px <= 0:
        raise ValueError("positive_radius_px must be positive")
    extent = matrix.new_tensor([image_hw[1] - 1, image_hw[0] - 1])
    gt = apply_affine(matrix, source_pixel_xy)
    inverse = torch.linalg.inv(matrix)
    base_target = apply_affine(inverse, target_pixel_xy)
    target_valid = (
        torch.isfinite(base_target).all(-1)
        & (base_target >= 0).all(-1)
        & (base_target <= extent).all(-1)
    )
    if target_support is not None:
        target_valid = sample_support(target_support, target_pixel_xy, image_hw)
    source_valid = torch.ones(gt.shape[:2], dtype=torch.bool, device=matrix.device)
    if source_support is not None:
        source_valid = sample_support(source_support, source_pixel_xy, image_hw)
    gt_support = torch.ones_like(source_valid)
    if target_support is not None:
        gt_support = sample_support(target_support, gt, image_hw)
    gt_in_frame = (
        torch.isfinite(gt).all(-1) & (gt >= 0).all(-1) & (gt <= extent).all(-1)
    )
    distance_squared = ((gt[:, :, None] - target_pixel_xy[None, None]) ** 2).sum(-1)
    positive = (distance_squared <= positive_radius_px**2) & target_valid[:, None]
    supervised = gt_in_frame & gt_support & source_valid & positive.any(-1)
    return dict(
        gt_pixel_xy=gt,
        target_valid=target_valid,
        positive=positive,
        supervised=supervised,
        gt_outside=~gt_in_frame,
        empty_positive=gt_in_frame & ~positive.any(-1),
        invalid_source=~source_valid,
        gt_invalid_support=gt_in_frame & ~gt_support,
    )


def neighborhood_mass_loss(scores, geometry):
    """-log(sum of probabilities at valid GT-neighborhood positions)."""
    minimum = torch.finfo(scores.dtype).min
    denominator = torch.logsumexp(
        scores.masked_fill(~geometry["target_valid"][:, None], minimum), -1
    )
    numerator = torch.logsumexp(scores.masked_fill(~geometry["positive"], minimum), -1)
    supervised = geometry["supervised"]
    per_query = torch.where(
        supervised, denominator - numerator, torch.zeros_like(denominator)
    )
    loss = per_query.sum() / supervised.sum().clamp_min(1)
    return loss


@torch.no_grad()
def spatial_candidates(
    scores, target_valid, target_pixel_xy, count=4, nms_radius_px=24.0
):
    """Greedy spatial NMS depends on scores, target geometry and support only."""
    if count < 1 or count > scores.shape[-1] or nms_radius_px < 0:
        raise ValueError("Invalid candidate count or NMS radius")
    batch, queries, targets = scores.shape
    available = target_valid[:, None].expand(batch, queries, targets).clone()
    chosen_indices, chosen_scores, chosen_valid = [], [], []
    minimum = torch.finfo(scores.dtype).min
    for _ in range(count):
        index = scores.masked_fill(~available, minimum).argmax(-1)
        valid = available.any(-1)
        value = scores.gather(-1, index[..., None]).squeeze(-1)
        center = target_pixel_xy[index]
        suppressed = ((target_pixel_xy[None, None] - center[:, :, None]) ** 2).sum(
            -1
        ) <= nms_radius_px**2
        available &= ~suppressed
        chosen_indices.append(torch.where(valid, index, -torch.ones_like(index)))
        chosen_scores.append(torch.where(valid, value, torch.zeros_like(value)))
        chosen_valid.append(valid)
    indices = torch.stack(chosen_indices, -1)
    valid = torch.stack(chosen_valid, -1)
    coordinates = target_pixel_xy[indices.clamp_min(0)]
    coordinates = torch.where(
        valid[..., None], coordinates, torch.zeros_like(coordinates)
    )
    return dict(
        indices=indices,
        pixel_xy=coordinates,
        valid=valid,
        scores=torch.stack(chosen_scores, -1),
    )
