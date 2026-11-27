"""Detached structure negatives and stable positive-mass correspondence loss."""

import copy
import torch
from torch.utils.data import DataLoader

from .semantic_bottleneck import hard_negative_peaks


@torch.no_grad()
def fixed_competition(structural_scores, geometry, target_xy, count=16, radius=24.0):
    p = geometry["positive"].detach()
    indices, valid = hard_negative_peaks(
        structural_scores.detach(),
        p,
        geometry["target_valid"],
        target_xy,
        count,
        radius,
    )
    n = torch.zeros_like(p, dtype=torch.long)
    n.scatter_add_(-1, indices.clamp_min(0), valid.long())
    n = n.bool()
    return dict(
        positive=p,
        negative=n,
        negative_indices=indices,
        negative_valid=valid,
        supervised=geometry["supervised"].detach(),
        gt_pixel_xy=geometry["gt_pixel_xy"].detach(),
    )


def correspondence_loss(scores, competition, tau):
    if tau <= 0:
        raise ValueError("corr temperature must be positive")
    p = competition["positive"].detach()
    n = competition["negative"].detach()
    eligible = competition["supervised"].detach() & p.any(-1) & n.any(-1)
    # Select valid queries before logsumexp: all-masked rows must not produce
    # NaNs in autograd even if masked from the final mean later.
    if eligible.any():
        selected = scores[eligible] / tau
        pm = p[eligible]
        nm = n[eligible]
        numerator = torch.logsumexp(selected.masked_fill(~pm, -torch.inf), -1)
        denominator = torch.logsumexp(selected.masked_fill(~(pm | nm), -torch.inf), -1)
        per = denominator - numerator
        loss = per.mean()
    else:
        per = scores.new_empty(0)
        loss = scores.sum() * 0
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite corr loss")
    return loss, dict(
        eligible=eligible,
        per_eligible=per,
        empty_positive=~p.any(-1),
        empty_negative=~n.any(-1),
        negative_insufficient=n.sum(-1) < 16,
    )


def loss_gradients(kl, corr, head):
    params = list(head.parameters())
    gk = torch.autograd.grad(kl, params, retain_graph=True, allow_unused=False)
    gc = torch.autograd.grad(corr, params, retain_graph=True, allow_unused=False)
    k = torch.cat([v.detach().flatten() for v in gk])
    c = torch.cat([v.detach().flatten() for v in gc])
    nk = k.double().norm()
    nc = c.double().norm()
    if not torch.isfinite(k).all() or not torch.isfinite(c).all():
        raise FloatingPointError("Nonfinite component gradient")
    cosine = (k.double() @ c.double()) / (nk * nc).clamp_min(1e-30)
    return dict(
        KL_gradient_norm=float(nk),
        corr_gradient_norm=float(nc),
        gradient_cosine=float(cosine),
        conflicting=bool(cosine < 0),
        all_head_parameters_reached_by_both=True,
    )


def continued_batch_schedule(
    length, batch_size, seed, start_step, additional_steps, saved_sampler_state
):
    """Replay index-only original DataLoader to recover a saved partial epoch."""
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(range(length)),
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        num_workers=0,
    )
    epoch = 0
    it = None

    def next_batch():
        nonlocal epoch, it
        if it is None:
            epoch += 1
            it = iter(loader)
        try:
            return next(it)
        except StopIteration:
            epoch += 1
            it = iter(loader)
            return next(it)

    for _ in range(start_step):
        next_batch()
    if not torch.equal(gen.get_state(), saved_sampler_state):
        raise ValueError("Original sampler state replay mismatch")
    start_epoch = epoch
    rows = []
    for step in range(additional_steps):
        indices = next_batch().tolist()
        rows.append(
            dict(
                additional_step=step + 1,
                total_step=start_step + step + 1,
                augmentation_epoch=epoch,
                indices=indices,
            )
        )
    return rows, dict(
        start_epoch=start_epoch,
        replayed_steps=start_step,
        sampler_state_matches=True,
        original_batches_per_epoch=len(loader),
        partial_epoch_batches_consumed=start_step % len(loader),
        final_epoch=epoch,
        final_sampler_state=gen.get_state(),
    )


def restored_head_and_optimizer(saved, head_class, device):
    head = head_class().to(device)
    head.load_state_dict(copy.deepcopy(saved["head"]))
    group = saved["optimizer"]["param_groups"][0]
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=group["lr"], weight_decay=group["weight_decay"]
    )
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    return head, optimizer
