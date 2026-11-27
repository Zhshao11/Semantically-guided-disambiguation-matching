import inspect
import numpy as np
import torch

from sdgm.context_descriptors import (
    masked_mean_map,
    multiscale_maps,
    sample_information_blocks,
    ContextDescriptorHead,
    orthogonal_projection,
    full_grid_pair_mass_loss,
)
from sdgm.coarse import StructuralRetriever, supervision_geometry
from sdgm.continuation_diagnostics import state_digest


def test_constant_pooling_boundary_and_invalid_values_do_not_change_valid_results():
    x = torch.full((1, 2, 5, 5), 3.0)
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    mask[..., 1, 2] = False
    y, frac = masked_mean_map(x, mask, 3)
    torch.testing.assert_close(
        y[mask.expand_as(y)],
        torch.full_like(y[mask.expand_as(y)], 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    changed = x.clone()
    changed[~mask.expand_as(x)] = 1000000
    yy, _ = masked_mean_map(changed, mask, 3)
    assert torch.equal(y, yy)
    assert (
        frac[0, 0, 0, 0].item() == 4 / 9 or abs(frac[0, 0, 0, 0].item() - 4 / 9) < 1e-7
    )
    assert y[0, 0, 1, 2].item() == 0


def test_small_pooling_matches_independent_manual_sum():
    x = torch.arange(18.0).reshape(1, 2, 3, 3)
    mask = torch.tensor([[[[1, 1, 0], [0, 1, 1], [1, 0, 1]]]], dtype=torch.bool)
    actual, frac = masked_mean_map(x, mask, 3)
    expected = np.zeros((2, 3, 3))
    m = mask[0, 0].numpy()
    a = x[0].numpy()
    for y in range(3):
        for z in range(3):
            if not m[y, z]:
                continue
            window = m[max(0, y - 1) : min(3, y + 2), max(0, z - 1) : min(3, z + 2)]
            value = a[:, max(0, y - 1) : min(3, y + 2), max(0, z - 1) : min(3, z + 2)]
            expected[:, y, z] = (value * window).sum((1, 2)) / (window.sum() + 1e-6)
    np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-6, rtol=1e-6)


def test_point_repeated_four_blocks_never_reads_other_positions():
    torch.manual_seed(2)
    f = torch.randn(1, 128, 5, 5)
    e = torch.randn(1, 32, 5, 5)
    m = torch.ones(1, 1, 5, 5, dtype=torch.bool)
    # Custom native extent40 maps centers8*k+3.5 exactly to5-grid.
    xy = torch.tensor([[19.5, 19.5]])
    maps, _ = multiscale_maps(f, e, m, False)
    inp = sample_information_blocks(maps, xy, image_hw=(40, 40))
    for i in range(1, 4):
        torch.testing.assert_close(inp[..., i * 160 : (i + 1) * 160], inp[..., :160])
    f2 = torch.randn_like(f) * 100
    e2 = torch.randn_like(e) * 100
    f2[..., 2, 2] = f[..., 2, 2]
    e2[..., 2, 2] = e[..., 2, 2]
    maps2, _ = multiscale_maps(f2, e2, m, False)
    torch.testing.assert_close(sample_information_blocks(maps2, xy, (40, 40)), inp)


def test_continuous_sampling_precedes_nonlinear_head_and_matches_pixel_center_chain():
    # A linear field interpolates exactly at a fractional native coordinate.
    grid = torch.arange(4) * 8 + 3.5
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    f = torch.stack([x + 1, y + 2])[None]
    e = torch.ones(1, 1, 4, 4)
    maps = [(f, e)] * 4
    xy = torch.tensor([[13.25, 18.75]])
    got = sample_information_blocks(maps, xy, (32, 32))[0, 0, :2]
    expected = torch.tensor([14.25, 20.75])
    expected /= expected.norm()
    torch.testing.assert_close(got, expected)
    # The network's interface accepts scale maps and sample coordinates, no GT.
    assert set(inspect.signature(ContextDescriptorHead.forward).parameters) == {
        "self",
        "optical_maps",
        "sar_maps",
        "source_xy",
        "target_xy",
    }
    # Demonstrate nonlinear interpolation substitution would give another result.
    torch.manual_seed(3)
    head = ContextDescriptorHead(channels=2, extra=1)
    grid_xy = torch.stack([x, y], -1).reshape(-1, 2)
    all_first = (
        torch.nn.functional.normalize(
            head.optical_head(sample_information_blocks(maps, grid_xy, (32, 32))),
            dim=-1,
        )
        .transpose(1, 2)
        .reshape(1, 128, 4, 4)
    )
    from sdgm.semantic_routing import sample_native_map

    wrong = torch.nn.functional.normalize(
        sample_native_map(all_first, xy, (32, 32)), dim=1
    ).transpose(1, 2)
    # For this test call the actual MLP directly with correctly sized32 inputs.
    correct = torch.nn.functional.normalize(
        head.optical_head(sample_information_blocks(maps, xy, (32, 32))), dim=-1
    )
    assert not torch.allclose(correct, wrong, atol=1e-7, rtol=1e-6)


def test_pair_mass_loss_matches_independent_formula_with_unequal_query_counts():
    score = torch.tensor(
        [[[0.2, 0.1, -0.3], [0.1, 0.7, 0.2]], [[0.3, 0.4, 0.1], [0.5, 0.1, 0.1]]],
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, False], [True, True, True]])
    positive = torch.tensor(
        [[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]], dtype=torch.bool
    )
    keep = torch.tensor([[True, True], [True, False]])
    loss, stats = full_grid_pair_mass_loss(
        score, dict(target_valid=valid, positive=positive, supervised=keep), 0.1
    )
    expected = []
    for b in range(2):
        queries = []
        for i in range(2):
            if not keep[b, i]:
                continue
            row = score[b, i].detach().double().numpy() / 0.1
            v = valid[b].numpy()
            p = positive[b, i].numpy() & v
            queries.append(np.log(np.exp(row[v]).sum()) - np.log(np.exp(row[p]).sum()))
        expected.append(np.mean(queries))
    np.testing.assert_allclose(loss.detach().numpy(), np.mean(expected), atol=1e-6)
    query_mean = stats["per_query"][keep].mean()
    assert abs(float((loss - query_mean).detach())) > 1e-3
    loss.backward()
    assert torch.isfinite(score.grad).all() and (score.grad[1, 1] == 0).all()


def test_empty_query_loss_safe_skips_pairs_and_has_finite_zero_gradient():
    score = torch.randn(2, 3, 4, requires_grad=True)
    geom = dict(
        target_valid=torch.zeros(2, 4, dtype=torch.bool),
        positive=torch.zeros(2, 3, 4, dtype=torch.bool),
        supervised=torch.ones(2, 3, dtype=torch.bool),
    )
    loss, stats = full_grid_pair_mass_loss(score, geom)
    assert not stats["valid_pair"].any() and loss.item() == 0
    loss.backward()
    assert torch.equal(score.grad, torch.zeros_like(score))


def test_projection_row_orthogonal_shared_and_four_heads_equal_parameters():
    r = orthogonal_projection(29)
    torch.testing.assert_close(r @ r.T, torch.eye(32), atol=1e-6, rtol=1e-6)
    assert torch.equal(r, orthogonal_projection(29))
    heads = [ContextDescriptorHead() for _ in range(4)]
    assert len({sum(p.numel() for p in h.parameters()) for h in heads}) == 1
    assert sum(p.numel() for p in heads[0].parameters()) == 393984
    assert not torch.equal(heads[0].optical_head[0].weight, heads[0].sar_head[0].weight)


def test_frozen_front_no_gradient_or_running_state_changes_and_both_heads_train():
    model = StructuralRetriever(image_hw=(256, 256)).requires_grad_(False).eval()
    before = state_digest(model.state_dict())
    head = ContextDescriptorHead()
    optim = torch.optim.AdamW(head.parameters(), lr=0.001)
    with torch.no_grad():
        pred = model(torch.randn(1, 1, 256, 256), torch.randn(1, 1, 256, 256))
        r = orthogonal_projection(3)
        first = pred["optical_descriptors"]
        second = pred["sar_descriptors"]
        mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        om, _ = multiscale_maps(first, torch.einsum("ec,bchw->behw", r, first), mask)
        sm, _ = multiscale_maps(second, torch.einsum("ec,bchw->behw", r, second), mask)
    out = head(om, sm, model.source_pixel_xy, model.target_pixel_xy)
    geom = supervision_geometry(
        torch.eye(3)[None],
        model.source_pixel_xy,
        model.target_pixel_xy,
        image_hw=(256, 256),
    )
    loss, _ = full_grid_pair_mass_loss(out["scores"], geom)
    loss.backward()
    for part in [head.optical_head, head.sar_head]:
        assert all(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in part.parameters()
        )
    optim.step()
    assert state_digest(model.state_dict()) == before and all(
        p.grad is None for p in model.parameters()
    )
    assert not model.training


def test_forward_unchanged_if_GT_matrix_changes_without_changing_inputs():
    from sdgm.context_descriptor_data import frozen_inputs, group_maps
    from sdgm.semantic_routing import SARPrototypeHead

    torch.manual_seed(41)
    model = StructuralRetriever(image_hw=(256, 256)).requires_grad_(False).eval()
    semantic = SARPrototypeHead().requires_grad_(False).eval()
    head = ContextDescriptorHead().eval()
    batch = dict(
        optical=torch.randn(1, 1, 256, 256),
        sar=torch.randn(1, 1, 256, 256),
        teacher=torch.rand(1, 32, 32, 32),
        source_support=torch.ones(1, 1, 256, 256, dtype=torch.bool),
        target_support=torch.ones(1, 1, 256, 256, dtype=torch.bool),
        matrix=torch.eye(3)[None],
    )
    with torch.no_grad():
        first = frozen_inputs(model, semantic, batch)
        batch["matrix"] = torch.tensor(
            [[[0.8, 0.1, 100.0], [-0.2, 1.2, -70.0], [0.0, 0.0, 1.0]]]
        )
        second = frozen_inputs(model, semantic, batch)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        a, b, _, _ = group_maps(first, orthogonal_projection(3), "semantic_context")
        c, d, _, _ = group_maps(second, orthogonal_projection(3), "semantic_context")
        out1 = head(a, b, model.source_pixel_xy, model.target_pixel_xy)
        out2 = head(c, d, model.source_pixel_xy, model.target_pixel_xy)
        assert all(torch.equal(out1[key], out2[key]) for key in out1)
