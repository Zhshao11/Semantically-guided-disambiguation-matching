"""Geometry convention shared by augmentation, sampling, and affine estimation.

H maps optical pixel centers to SAR pixel centers. Matrices act on column
vectors. Image sizes are (height, width). grid_sample uses align_corners=False.
"""

from dataclasses import dataclass

import numpy as np


def homogeneous(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 2x3 or 3x3 matrix")
    if not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Only affine transforms are supported")
    return matrix


def transform_points(matrix, points):
    matrix = homogeneous(matrix)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite points in (..., 2) x,y order")
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def update_after_augmentation(matrix, optical_augmentation, sar_augmentation):
    """A_o and A_s map old pixels to augmented pixels, including crop/resize."""
    return (
        homogeneous(sar_augmentation)
        @ homogeneous(matrix)
        @ np.linalg.inv(homogeneous(optical_augmentation))
    )


def resize_transform(source_hw, target_hw):
    """Forward pixel-center transform for OpenCV / align_corners=False resize."""
    source = np.asarray(source_hw, dtype=float)[::-1]
    target = np.asarray(target_hw, dtype=float)[::-1]
    if np.any(source <= 0) or np.any(target <= 0):
        raise ValueError("Image sizes must be positive")
    scale = target / source
    return np.array(
        [
            [scale[0], 0, (scale[0] - 1) / 2],
            [0, scale[1], (scale[1] - 1) / 2],
            [0, 0, 1],
        ]
    )


def image_to_feature(points, image_hw, feature_hw):
    """Mapping for features defined on a resized pixel-center grid.

    Strided convolution architectures must supply their actual stride/offset
    transform if their receptive-field centers use a different grid.
    """
    return transform_points(resize_transform(image_hw, feature_hw), points)


def feature_to_normalized(points, feature_hw):
    size = np.asarray(feature_hw, dtype=float)[::-1]
    if np.any(size <= 0):
        raise ValueError("Feature size must be positive")
    return 2 * (np.asarray(points, dtype=float) + 0.5) / size - 1


def image_to_normalized(points, image_hw, feature_hw):
    return feature_to_normalized(
        image_to_feature(points, image_hw, feature_hw), feature_hw
    )


def flow_to_map(flow):
    flow = np.asarray(flow)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("Expected finite HxWx2 flow")
    y, x = np.indices(flow.shape[:2])
    return np.stack([x, y], -1) + flow


@dataclass
class AffineResult:
    valid: bool
    matrix: np.ndarray | None
    reason: str
    condition: float | None = None
    rmse: float | None = None


def solve_affine(source, target, weights=None, max_condition=1e8):
    """Weighted, centered affine least squares with explicit degeneracy status."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape:
        raise ValueError("source and target must both be Nx2")
    weights = (
        np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    )
    if (
        weights.shape != (len(source),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Weights must be finite nonnegative values with shape N")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return AffineResult(False, None, "nonfinite_correspondence")
    active = weights > 0
    if active.sum() < 3:
        return AffineResult(False, None, "insufficient_positive_weight_points")
    p, q, w = source[active], target[active], weights[active]
    w = w / w.sum()
    pc, qc = (p * w[:, None]).sum(0), (q * w[:, None]).sum(0)
    scale = np.sqrt(((p - pc) ** 2 * w[:, None]).sum())
    if scale < 1e-12:
        return AffineResult(False, None, "collapsed_source")
    design = np.column_stack([(p - pc) / scale, np.ones(len(p))])
    design_w = design * np.sqrt(w[:, None])
    singular = np.linalg.svd(design_w, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], np.finfo(float).tiny))
    if singular[-1] <= singular[0] / max_condition:
        return AffineResult(
            False, None, "rank_deficient_or_ill_conditioned_source", condition
        )
    coefficients = np.linalg.lstsq(
        design_w, (q - qc) * np.sqrt(w[:, None]), rcond=None
    )[0]
    linear = coefficients[:2].T / scale
    linear_singular = np.linalg.svd(linear, compute_uv=False)
    if (
        linear_singular[-1] < 1e-10
        or linear_singular[0] / linear_singular[-1] > max_condition
    ):
        return AffineResult(
            False, None, "singular_or_ill_conditioned_target", condition
        )
    matrix = np.column_stack([linear, qc + coefficients[2] - linear @ pc])
    residual = transform_points(matrix, p) - q
    rmse = float(np.sqrt((w * (residual**2).sum(1)).sum()))
    return AffineResult(True, matrix, "ok", condition, rmse)
