import math

import pytest
import torch
import numpy as np

from sdgm.coarse import (
    StructuralRetriever,
    neighborhood_mass_loss,
    spatial_candidates,
    supervision_geometry,
)
from sdgm.geometry import transform_points


def test_metadata_direction_matches_independently_validated_geometry():
    matrix = np.array([[1.1, 0.2, 7], [-0.15, 0.95, 12], [0, 0, 1]], dtype=np.float32)
    source = np.array([[33, 77], [401, 123]], dtype=np.float32)
    model = StructuralRetriever()
    geometry = supervision_geometry(
        torch.from_numpy(matrix)[None], torch.from_numpy(source), model.target_pixel_xy
    )
    np.testing.assert_allclose(
        geometry["gt_pixel_xy"][0].numpy(), transform_points(matrix, source), atol=5e-5
    )


def test_shared_encoder_output_and_descriptor_norm():
    torch.set_num_threads(2)
    model = StructuralRetriever()
    with torch.no_grad():
        result = model(torch.randn(1, 1, 512, 512), torch.randn(1, 1, 512, 512))
    assert result["scores"].shape == (1, 576, 4096)
    for key in ["optical_descriptors", "sar_descriptors"]:
        assert result[key].shape == (1, 128, 64, 64)
        torch.testing.assert_close(result[key].norm(dim=1), torch.ones(1, 64, 64))
    assert model.source_pixel_xy.shape == (576, 2)
    assert len(torch.unique(model.source_feature_xy, dim=0)) == 576
    torch.testing.assert_close(
        model.source_pixel_xy[[0, -1]], torch.tensor([[3.5, 3.5], [507.5, 507.5]])
    )


def test_average_pool_centers_match_feature_sampling_convention():
    y, x = torch.meshgrid(torch.arange(512), torch.arange(512), indexing="ij")
    ramp = (x + 1000 * y).float()[None, None]
    pooled = torch.nn.functional.avg_pool2d(ramp, 8, 8)
    model = StructuralRetriever()
    expected = model.target_pixel_xy[:, 0] + 1000 * model.target_pixel_xy[:, 1]
    torch.testing.assert_close(pooled.flatten(), expected)


def test_probability_mass_uses_all_valid_targets_and_sums_positive_neighborhood():
    matrix = torch.eye(3)[None]
    source = torch.tensor([[10.0, 10.0]])
    target = torch.tensor([[10.0, 10.0], [11.0, 10.0], [30.0, 30.0], [40.0, 40.0]])
    geometry = supervision_geometry(matrix, source, target, positive_radius_px=1.1)
    scores = torch.zeros(1, 1, 4, requires_grad=True)
    loss = neighborhood_mass_loss(scores, geometry)
    assert float(loss.detach()) == pytest.approx(math.log(2), abs=1e-6)
    loss.backward()
    assert scores.grad[0, 0, 0] < 0 and scores.grad[0, 0, 1] < 0
    assert scores.grad[0, 0, 2] > 0 and scores.grad[0, 0, 3] > 0


@pytest.mark.parametrize(
    "translation,source,radius",
    [(1000.0, [[1.0, 1.0]], 8.0), (0.0, [[10.3, 10.3]], 0.01)],
)
def test_empty_supervision_has_finite_zero_loss_and_zero_gradients(
    translation, source, radius
):
    matrix = torch.eye(3)[None]
    matrix[0, 0, 2] = translation
    target = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    geometry = supervision_geometry(
        matrix, torch.tensor(source), target, positive_radius_px=radius
    )
    assert not geometry["supervised"].any()
    scores = torch.randn(1, 1, 2, requires_grad=True)
    loss = neighborhood_mass_loss(scores, geometry)
    assert torch.isfinite(loss) and float(loss.detach()) == 0
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_spatial_nms_does_not_inject_labels_or_propagate_gradients():
    target = torch.tensor(
        [[0.0, 0.0], [8.0, 0.0], [32.0, 0.0], [64.0, 0.0], [96.0, 0.0]]
    )
    scores = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0]]], requires_grad=True)
    candidates = spatial_candidates(
        scores, torch.ones(1, 5, dtype=torch.bool), target, nms_radius_px=16.0
    )
    torch.testing.assert_close(candidates["indices"], torch.tensor([[[0, 2, 3, 4]]]))
    assert not candidates["scores"].requires_grad
    assert candidates["valid"].all()


def test_empty_or_partial_candidate_support_is_explicit():
    target = torch.tensor([[0.0, 0.0], [8.0, 0.0], [32.0, 0.0], [64.0, 0.0]])
    scores = torch.zeros(2, 1, 4)
    support = torch.tensor([[False, False, False, False], [False, False, True, False]])
    result = spatial_candidates(scores, support, target, nms_radius_px=8.0)
    assert not result["valid"][0].any()
    assert result["valid"][1].sum() == 1
    assert result["indices"][1, 0, 0] == 2
    assert (result["indices"][~result["valid"]] == -1).all()
    assert torch.isfinite(result["pixel_xy"]).all()
    assert torch.isfinite(result["scores"]).all()


def test_validity_uses_warp_support_and_keeps_zero_pixel_locations():
    matrix = torch.eye(3)[None]
    matrix[0, 0, 2] = 100
    source = torch.tensor([[0.0, 0.0], [500.0, 0.0]])
    target = torch.tensor([[0.0, 0.0], [100.0, 0.0], [508.0, 0.0]])
    geometry = supervision_geometry(matrix, source, target, positive_radius_px=8)
    torch.testing.assert_close(
        geometry["target_valid"], torch.tensor([[False, True, True]])
    )
    torch.testing.assert_close(geometry["supervised"], torch.tensor([[True, False]]))


def test_native_256_uses_same_encoder_and_pixel_centers_without_resize():
    torch.set_num_threads(2)
    model = StructuralRetriever(image_hw=(256, 256))
    result = model(torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256))
    assert result["scores"].shape == (1, 144, 1024)
    assert result["optical_descriptors"].shape == (1, 128, 32, 32)
    torch.testing.assert_close(
        model.source_pixel_xy[[0, -1]], torch.tensor([[3.5, 3.5], [251.5, 251.5]])
    )
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in StructuralRetriever().parameters()
    )
    geometry = supervision_geometry(
        torch.eye(3)[None],
        model.source_pixel_xy,
        model.target_pixel_xy,
        image_hw=(256, 256),
    )
    neighborhood_mass_loss(result["scores"], geometry).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError, match="Native image"):
        StructuralRetriever(image_hw=(255, 256))


def test_actual_masks_override_inverse_affine_support_and_mask_source_gt():
    matrix = torch.eye(3)[None]
    source = torch.tensor([[10.0, 10.0], [20.0, 20.0], [250.0, 10.0], [256.0, 10.0]])
    target = torch.tensor([[10.0, 10.0], [20.0, 20.0], [250.0, 10.0]])
    optical_mask = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    sar_mask = optical_mask.clone()
    optical_mask[0, 0, 10, 10] = False
    sar_mask[0, 0, 20, 20] = False
    geometry = supervision_geometry(
        matrix,
        source,
        target,
        image_hw=(256, 256),
        source_support=optical_mask,
        target_support=sar_mask,
    )
    torch.testing.assert_close(
        geometry["supervised"], torch.tensor([[False, False, True, False]])
    )
    torch.testing.assert_close(
        geometry["target_valid"], torch.tensor([[True, False, True]])
    )
    # A mask of ones stays valid regardless of image intensity; zero images are not passed to geometry.
    assert geometry["supervised"][0, 2]
