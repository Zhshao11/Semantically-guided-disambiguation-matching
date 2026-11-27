import torch
import pytest
from torch.utils.data import DataLoader
from sdgm.correspondence_objective import (
    correspondence_loss,
    fixed_competition,
    continued_batch_schedule,
)


def test_corr_matches_logsumexp_mass_and_gradients_reach_both_positions():
    scores = torch.tensor([[[0.03, -0.02, 0.01, -0.1]]], requires_grad=True)
    p = torch.tensor([[[True, False, True, False]]])
    n = torch.tensor([[[False, True, False, True]]])
    c = dict(positive=p, negative=n, supervised=torch.ones(1, 1, dtype=torch.bool))
    loss, stats = correspondence_loss(scores, c, 0.02)
    expected = -torch.log(
        torch.exp(scores[0, 0, [0, 2]] / 0.02).sum()
        / torch.exp(scores[0, 0] / 0.02).sum()
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (scores.grad[p] < 0).all() and (scores.grad[n] > 0).all()


def test_empty_P_or_N_has_finite_zero_gradient_without_invalid_fill():
    score = torch.randn(1, 2, 4, requires_grad=True)
    p = torch.zeros_like(score, dtype=torch.bool)
    p[:, 0, 0] = True
    n = torch.zeros_like(p)
    n[:, 1, 1] = True
    loss, stats = correspondence_loss(
        score,
        dict(positive=p, negative=n, supervised=torch.ones(1, 2, dtype=torch.bool)),
        0.01,
    )
    assert loss.item() == 0 and not stats["eligible"].any()
    loss.backward()
    assert torch.equal(score.grad, torch.zeros_like(score))
    with pytest.raises(ValueError):
        correspondence_loss(
            score, dict(positive=p, negative=n, supervised=p.any(-1)), 0
        )


def test_structure_negative_choice_detached_and_independent_of_student():
    scores = torch.arange(8.0).reshape(1, 1, 8).requires_grad_()
    p = torch.zeros_like(scores, dtype=torch.bool)
    p[..., 7] = True
    geometry = dict(
        positive=p,
        target_valid=torch.ones(1, 8, dtype=torch.bool),
        supervised=p.any(-1),
        gt_pixel_xy=torch.zeros(1, 1, 2),
    )
    xy = torch.stack([torch.arange(8) * 32, torch.zeros(8)], 1).float()
    comp = fixed_competition(scores, geometry, xy, count=3)
    assert comp["negative_indices"].tolist() == [[[6, 5, 4]]]
    assert not comp["negative"].requires_grad and not comp["gt_pixel_xy"].requires_grad


def test_partial_epoch_sampler_resume_matches_uninterrupted_order():
    gen = torch.Generator().manual_seed(17)
    loader = DataLoader(list(range(19)), batch_size=4, shuffle=True, generator=gen)
    epoch = 1
    it = iter(loader)
    for _ in range(7):
        try:
            next(it)
        except StopIteration:
            epoch += 1
            it = iter(loader)
            next(it)
    state = gen.get_state().clone()
    expected = []
    for _ in range(6):
        try:
            b = next(it)
        except StopIteration:
            epoch += 1
            it = iter(loader)
            b = next(it)
        expected.append((epoch, b.tolist()))
    schedule, audit = continued_batch_schedule(19, 4, 17, 7, 6, state)
    assert [(r["augmentation_epoch"], r["indices"]) for r in schedule] == expected
    assert audit["partial_epoch_batches_consumed"] == 2


def test_restored_branches_have_identical_AdamW_moments_and_only_objective_differs():
    from sdgm.correspondence_objective import restored_head_and_optimizer
    from sdgm.semantic_routing import SARPrototypeHead
    from sdgm.continuation_diagnostics import state_digest

    h = SARPrototypeHead()
    o = torch.optim.AdamW(h.parameters(), lr=0.001, weight_decay=0.0001)
    h(torch.randn(1, 128, 2, 2)).square().sum().backward()
    o.step()
    saved = dict(head=h.state_dict(), optimizer=o.state_dict())
    a, oa = restored_head_and_optimizer(saved, SARPrototypeHead, "cpu")
    b, ob = restored_head_and_optimizer(saved, SARPrototypeHead, "cpu")
    assert (
        state_digest(a.state_dict())
        == state_digest(b.state_dict())
        == state_digest(saved["head"])
    )
    assert (
        state_digest(oa.state_dict())
        == state_digest(ob.state_dict())
        == state_digest(saved["optimizer"])
    )
    # Updating a does not mutate b or the original saved optimizer moments.
    before = state_digest(ob.state_dict())
    oa.zero_grad()
    a(torch.randn(1, 128, 2, 2)).square().sum().backward()
    oa.step()
    assert state_digest(ob.state_dict()) == before == state_digest(saved["optimizer"])


def test_SAR_probabilities_at_both_P_and_N_receive_gradient_teacher_stopped():
    from sdgm.semantic_score_control import probability_scores

    teacher = torch.tensor([[[0.7], [0.3]]], requires_grad=True)
    logits = torch.tensor([[[0.2, 0.8, 0.5], [-0.3, 0.1, 0.4]]], requires_grad=True)
    sar = logits.softmax(1)
    score = probability_scores(teacher.detach(), sar)["dist"]
    comp = dict(
        positive=torch.tensor([[[True, False, False]]]),
        negative=torch.tensor([[[False, True, False]]]),
        supervised=torch.ones(1, 1, dtype=torch.bool),
    )
    loss, _ = correspondence_loss(score, comp, 0.02)
    loss.backward()
    assert teacher.grad is None
    assert logits.grad[..., 0].abs().sum() > 0 and logits.grad[..., 1].abs().sum() > 0
    assert torch.equal(logits.grad[..., 2], torch.zeros_like(logits.grad[..., 2]))
