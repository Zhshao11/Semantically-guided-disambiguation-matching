import numpy as np
import pytest
import torch

from sdgm.coarse import spatial_candidates
from sdgm.semantic_routing import (
    SARPrototypeHead,
    distillation_loss,
    probability_normalize,
    routed_candidates,
    sample_native_map,
    semantic_bias,
    shuffle_semantic,
    spherical_kmeans,
)
from sdgm.semantic_teacher import prototype_sample, teacher_preprocess


def test_token_native_structure_pixel_centers_and_affine_sampling():
    # Analytic linear scalar field on18x18 token centers: 3*x+2*y.
    center = (torch.arange(18) + 0.5) * 256 / 18 - 0.5
    yy, xx = torch.meshgrid(center, center, indexing="ij")
    field = (3 * xx + 2 * yy)[None, None]
    xy = torch.tensor([[35.5, 51.5], [83.5, 91.5], [155.5, 139.5]])
    actual = sample_native_map(field, xy)[0, 0]
    torch.testing.assert_close(
        actual, 3 * xy[:, 0] + 2 * xy[:, 1], atol=1e-4, rtol=1e-6
    )
    # Sampling32x32 field after optical->SAR shift; reverse direction must differ.
    s = torch.arange(32) * 8 + 3.5
    y, x = torch.meshgrid(s, s, indexing="ij")
    sar = (3 * (x - 8) + 2 * (y + 16))[None, None]
    shifted = xy + torch.tensor([8.0, -16.0])
    torch.testing.assert_close(
        sample_native_map(sar, shifted)[0, 0], actual, atol=1e-4, rtol=1e-6
    )
    assert not torch.allclose(
        sample_native_map(sar, xy - torch.tensor([8.0, -16.0]))[0, 0], actual
    )
    # Explicit border policy at a valid native point outside token-center hull.
    torch.testing.assert_close(
        sample_native_map(field, torch.tensor([[0.0, 0.0]]))[0, 0, 0], field[0, 0, 0, 0]
    )


def test_KL_stops_teacher_grad_uses_existing_mask_and_normalizes_interpolation():
    head = SARPrototypeHead(prototypes=2)
    descriptor = torch.randn(1, 128, 32, 32)
    p = head(descriptor)
    teacher = torch.rand(1, 2, 32, 32, requires_grad=True)
    teacher = teacher / teacher.sum(1, keepdim=True)
    teacher.retain_grad()
    xy = torch.tensor([[35.5, 51.5], [83.5, 91.5]])
    g = dict(
        gt_pixel_xy=xy[None] + torch.tensor([2.25, 3.125]),
        supervised=torch.tensor([[True, False]]),
    )
    loss, kl, sampled, t = distillation_loss(teacher, p, xy, g)
    torch.testing.assert_close(loss, kl[0, 0])
    torch.testing.assert_close(sampled.sum(1), torch.ones(1, 2))
    loss.backward()
    assert teacher.grad is None
    assert all(
        x.grad is not None and torch.isfinite(x.grad).all() for x in head.parameters()
    )
    g["supervised"][:] = False
    empty, *_ = distillation_loss(teacher, head(descriptor), xy, g)
    assert empty.item() == 0
    empty.backward()


def test_lambda0_exact_fallback_and_reserved_peaks_union_NMS_and_invalid_slots():
    # A widely spaced target line makes selected peak order hand-checkable.
    xy = torch.stack([torch.arange(6) * 32, torch.zeros(6)], 1).float()
    scores = torch.tensor([[[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]])
    valid = torch.ones(1, 6, dtype=torch.bool)
    bias = torch.tensor([[[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0]]])
    original = spatial_candidates(scores, valid, xy, count=4)
    zero = routed_candidates(scores, bias, valid, xy, 0)
    assert all(torch.equal(original[k], zero[k]) for k in original)
    routed = routed_candidates(scores, bias, valid, xy, 2.0)
    assert routed["indices"].tolist() == [[[0, 1, 4, 5]]]
    valid[:, 2:] = False
    sparse = routed_candidates(scores, bias, valid, xy, 2.0)
    assert sparse["indices"].tolist() == [[[0, 1, -1, -1]]]
    assert sparse["valid"].sum() == 2
    assert (sparse["pixel_xy"][~sparse["valid"]] == 0).all()
    # Two nearby locations cannot both survive union suppression.
    close = xy.clone()
    close[4] = close[0] + torch.tensor([2.0, 0.0])
    selected = routed_candidates(scores, bias, torch.ones_like(valid), close, 2.0)
    assert 4 not in selected["indices"].flatten().tolist()


def test_bias_bounds_and_shuffle_preserves_histogram_but_changes_alignment():
    a = probability_normalize(torch.rand(2, 32, 5))
    p = probability_normalize(torch.rand(2, 32, 4, 4))
    b = semantic_bias(a, p)
    assert b.min() >= -1 and b.max() <= 1
    shuffled = shuffle_semantic(p, ["a", "b"], 17)
    assert torch.equal(shuffled, shuffle_semantic(p, ["a", "b"], 17))
    assert not torch.equal(p, shuffled)
    torch.testing.assert_close(
        p.flatten(2).sort(-1).values, shuffled.flatten(2).sort(-1).values
    )


def test_prototype_sampling_rejects_validation_and_balances_train_families():
    rows = [
        dict(sample_id=f"{f}/{i}", family_id=f, split="train", optical_sha256="x")
        for f in ["a", "b", "c"]
        for i in range(4)
    ]
    cfg = dict(seed=31, pairs_per_training_family=1, tokens_per_pair=8)
    chosen = prototype_sample(rows, 324, cfg)
    assert len(chosen) == 3 and {r["family_id"] for r in chosen} == {"a", "b", "c"}
    assert chosen == prototype_sample(rows, 324, cfg)
    assert all(len(set(r["token_indices"])) == 8 for r in chosen)
    rows[0]["split"] = "val"
    with pytest.raises(ValueError, match="Only training"):
        prototype_sample(rows, 324, cfg)


def test_spherical_centers_are_unit_length_and_separate_distinct_directions():
    x = torch.cat(
        [
            torch.tensor([[1.0, 0.0]]).repeat(20, 1),
            torch.tensor([[0.0, 1.0]]).repeat(20, 1),
        ]
    )
    c, counts, _ = spherical_kmeans(x, 2, 5, 9)
    torch.testing.assert_close(c.norm(dim=1), torch.ones(2))
    assert counts == [20, 20]
    assert sorted(c.argmax(1).tolist()) == [0, 1]
    gray = torch.full((1, 1, 256, 256), 127.5)
    transformed = teacher_preprocess(gray)
    assert transformed.shape == (1, 3, 252, 252)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(transformed[0, :, 100, 100], (0.5 - mean) / std)


def test_one_distillation_update_cannot_modify_frozen_structure():
    from sdgm.coarse import StructuralRetriever
    from sdgm.continuation_diagnostics import state_digest
    from scripts.train_os_semantic_routing import step_head

    model = StructuralRetriever(image_hw=(256, 256)).requires_grad_(False).eval()
    head = SARPrototypeHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    before = state_digest(model.state_dict())
    head_before = state_digest(head.state_dict())
    batch = dict(
        optical=torch.randn(1, 1, 256, 256),
        sar=torch.randn(1, 1, 256, 256),
        matrix=torch.eye(3)[None],
        source_support=torch.ones(1, 1, 256, 256, dtype=torch.bool),
        target_support=torch.ones(1, 1, 256, 256, dtype=torch.bool),
        teacher=probability_normalize(torch.rand(1, 32, 32, 32)),
    )
    row = step_head(
        model, head, batch, dict(positive_radius_px=8), optimizer, dict(gradient_clip=1)
    )
    assert row["structure_has_no_gradient"]
    assert state_digest(model.state_dict()) == before
    assert state_digest(head.state_dict()) != head_before
    assert not model.training and all(p.grad is None for p in model.parameters())


def test_teacher_cache_refuses_sample_hash_or_prototype_configuration_mismatch(
    tmp_path, monkeypatch
):
    import json
    from sdgm.semantic_teacher import TeacherView, verify_teacher_cache, json_digest

    class View:
        records = [
            dict(
                sample_id="a",
                family_id="F",
                split="train",
                optical_sha256="wrong",
                sar_sha256="s",
            )
        ]
        split = "train"

    (tmp_path / "cache_samples.jsonl").write_text(
        json.dumps(dict(View.records[0], optical_sha256="right")) + "\n"
    )
    with pytest.raises(ValueError, match="provenance"):
        TeacherView(View(), tmp_path)
    receipt = dict(teacher={"fixed": "identity"}, prototypes={"temperature": 0.05})
    receipt["cache_version"] = json_digest(receipt)
    receipt["file_hashes"] = {}
    (tmp_path / "cache_receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(
        "sdgm.semantic_teacher.teacher_identity", lambda c: {"fixed": "identity"}
    )
    with pytest.raises(ValueError, match="mismatch"):
        verify_teacher_cache(tmp_path, dict(prototypes={"temperature": 0.1}))
