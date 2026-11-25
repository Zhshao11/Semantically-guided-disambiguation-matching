import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from sdgm.geometry import (
    feature_to_normalized,
    flow_to_map,
    homogeneous,
    image_to_feature,
    image_to_normalized,
    resize_transform,
    solve_affine,
    transform_points,
    update_after_augmentation,
)


def test_optical_to_sar_direction_against_raster_warp():
    optical = np.zeros((64, 80), np.float32)
    optical[15, 20] = 1
    matrix = np.array([[1, 0, 7], [0, 1, -4]], float)
    sar = cv2.warpAffine(optical, matrix, (80, 64), flags=cv2.INTER_NEAREST)
    prediction = transform_points(matrix, [[20, 15]])[0]
    y, x = np.unravel_index(np.argmax(sar), sar.shape)
    np.testing.assert_allclose(prediction, [x, y], atol=0)
    np.testing.assert_allclose(
        transform_points(np.linalg.inv(homogeneous(matrix)), [prediction]), [[20, 15]]
    )


def test_augmentation_composition_with_crop_resize_and_flip():
    matrix = np.array([[1.03, 0.12, 17], [-0.08, 0.94, -9], [0, 0, 1]])
    crop = np.array([[1, 0, -13], [0, 1, -27], [0, 0, 1]])
    flip = np.array([[-1, 0, 255], [0, 1, 0], [0, 0, 1]])
    optical_aug = flip @ resize_transform((400, 500), (256, 256)) @ crop
    sar_aug = resize_transform((600, 700), (320, 384)) @ np.array(
        [[1, 0, -7], [0, 1, -3], [0, 0, 1]]
    )
    points = np.random.default_rng(3).uniform(30, 200, (37, 2))
    updated = update_after_augmentation(matrix, optical_aug, sar_aug)
    np.testing.assert_allclose(
        transform_points(updated, transform_points(optical_aug, points)),
        transform_points(sar_aug, transform_points(matrix, points)),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "image_hw,feature_hw",
    [((512, 512), (64, 64)), ((513, 769), (65, 97)), ((21, 31), (1, 7))],
)
def test_half_pixel_resize_and_normalization(image_hw, feature_hw):
    h, w = image_hw
    points = np.array([[0, 0], [w - 1, h - 1], [(w - 1) / 2, (h - 1) / 2]], float)
    feature = image_to_feature(points, image_hw, feature_hw)
    scale = np.array(feature_hw)[::-1] / np.array(image_hw)[::-1]
    np.testing.assert_allclose(feature, (points + 0.5) * scale - 0.5)
    np.testing.assert_allclose(
        feature_to_normalized(feature, feature_hw),
        2 * (points + 0.5) / np.array(image_hw)[::-1] - 1,
    )


def test_feature_sampling_with_real_torch_grid_sample():
    h, w = 17, 23
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    feature = (xx + 100 * yy).double()[None, None]
    feature_points = np.array([[2.25, 3.75], [13, 7], [20.5, 15.25]])
    image_points = transform_points(
        resize_transform((h, w), (136, 184)), feature_points
    )
    grid = torch.from_numpy(image_to_normalized(image_points, (136, 184), (h, w))).view(
        1, 1, -1, 2
    )
    sampled = (
        functional.grid_sample(feature, grid, mode="bilinear", align_corners=False)
        .numpy()
        .ravel()
    )
    np.testing.assert_allclose(
        sampled, feature_points[:, 0] + 100 * feature_points[:, 1], atol=1e-10
    )


def test_affine_solve_full_six_degrees_of_freedom_and_large_origin():
    points = np.random.default_rng(8).normal(size=(41, 2)) * 200 + np.array([6e6, 3e6])
    matrix = np.array([[1.07, 0.19, 157], [-0.12, 0.93, -81]])
    result = solve_affine(points, transform_points(matrix, points))
    assert result.valid, result.reason
    np.testing.assert_allclose(result.matrix, matrix, atol=2e-5)
    assert result.rmse < 1e-7


def test_zero_weight_outlier_is_excluded():
    points = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [20, 20]], float)
    target = points + np.array([7, -3])
    target[-1] = [10000, -4000]
    result = solve_affine(points, target, [1, 1, 1, 1, 0])
    assert result.valid
    np.testing.assert_allclose(result.matrix, [[1, 0, 7], [0, 1, -3]], atol=1e-12)


@pytest.mark.parametrize(
    "points,reason",
    [
        ([[0, 0], [1, 1]], "insufficient_positive_weight_points"),
        ([[1, 1], [1, 1], [1, 1]], "collapsed_source"),
        ([[0, 0], [1, 1], [2, 2], [3, 3]], "rank_deficient_or_ill_conditioned_source"),
        (
            [[0, 0], [1, 0], [2, 1e-12], [3, 2e-12]],
            "rank_deficient_or_ill_conditioned_source",
        ),
    ],
)
def test_source_degeneracy_is_explicit(points, reason):
    result = solve_affine(points, points)
    assert not result.valid and result.matrix is None and result.reason == reason


def test_collapsed_target_is_invalid():
    result = solve_affine([[0, 0], [1, 0], [0, 1]], [[2, 2], [2, 2], [2, 2]])
    assert not result.valid and "target" in result.reason


def test_nonfinite_and_invalid_weights():
    assert not solve_affine(
        [[0, 0], [1, 0], [0, 1]], [[0, 0], [1, np.nan], [0, 1]]
    ).valid
    with pytest.raises(ValueError):
        solve_affine([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]], [1, -1, 1])
    assert not solve_affine(
        [[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]], [0, 0, 0]
    ).valid


def test_flow_is_forward_xy_map():
    flow = np.zeros((5, 7, 2))
    flow[..., 0] = 3
    flow[..., 1] = -2
    np.testing.assert_equal(flow_to_map(flow)[4, 6], [9, 2])


def test_resize_matches_opencv_pixel_centers():
    yy, xx = np.indices((20, 30))
    ramp = (xx + 3 * yy).astype(np.float64)
    scaled = cv2.resize(ramp, (15, 10), interpolation=cv2.INTER_LINEAR)
    target = np.array([[3, 2], [8, 6]])
    source = transform_points(
        np.linalg.inv(resize_transform((20, 30), (10, 15))), target
    )
    np.testing.assert_allclose(
        scaled[target[:, 1], target[:, 0]], source[:, 0] + 3 * source[:, 1]
    )
