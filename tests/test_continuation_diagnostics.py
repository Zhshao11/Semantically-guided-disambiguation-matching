import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from sdgm.continuation_diagnostics import (
    state_digest,
    replay_sampler,
    select_monitor,
    FrozenDiagnosticView,
    aggregate_families,
)


def test_sampler_replay_reaches_same_next_epoch_without_loading_images():
    generator = torch.Generator().manual_seed(17)
    loader = DataLoader(
        list(range(19)), batch_size=4, shuffle=True, generator=generator, num_workers=0
    )
    for _ in range(20):
        for _ in loader:
            pass
    expected = [int(x) for batch in loader for x in batch]
    replayed = replay_sampler(17, 19, 4, 20)
    actual = [
        int(x)
        for batch in DataLoader(
            list(range(19)),
            batch_size=4,
            shuffle=True,
            generator=replayed,
            num_workers=0,
        )
        for x in batch
    ]
    assert actual == expected


def test_optimizer_digest_checks_moments_and_step_on_resume():
    a = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(a.parameters(), lr=0.0003)
    a(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved = optimizer.state_dict()
    digest = state_digest(saved)
    b = torch.nn.Linear(2, 1)
    restored = torch.optim.AdamW(b.parameters(), lr=0.1)
    restored.load_state_dict(saved)
    assert (
        state_digest(restored.state_dict()) == digest
        and restored.param_groups[0]["lr"] == 0.0003
    )
    restored.state[next(iter(restored.state))]["exp_avg"].add_(1)
    assert state_digest(restored.state_dict()) != digest


def test_monitor_covers_balanced_families_and_frozen_view_ignores_epoch_changes():
    rows = [
        dict(sample_id=f"train/{f}/{p}", family_id=f"T{f}")
        for f in range(5)
        for p in range(4)
    ]
    val = [dict(sample_id=f"val/{i}", family_id=f"V{i//4}") for i in range(8)]
    ids = select_monitor(rows, val, 5, 2, 4)
    assert len(ids) == 8 and len({rows[i]["family_id"] for i in ids}) == 2
    assert ids == select_monitor(rows, val, 5, 2, 4)

    class Base:
        pass

    base = Base()
    base.records = rows
    base.split = "train"
    base.arrays = np.zeros((20, 2, 256, 256), "uint8")
    base.epoch = 1
    plans = [
        dict(
            sample_id=rows[i]["sample_id"],
            family_id=rows[i]["family_id"],
            optical_input_to_sar_input=np.eye(3).tolist(),
        )
        for i in ids
    ]
    view = FrozenDiagnosticView(base, ids, plans)
    before = view[0]
    base.epoch = 30
    after = view[0]
    torch.testing.assert_close(before["matrix"], after["matrix"])
    torch.testing.assert_close(before["sar"], after["sar"])
    assert before["source_support"].all() and before["target_support"].all()
    with pytest.raises(ValueError, match="overlaps"):
        select_monitor(
            rows,
            [dict(sample_id="v", family_id="T" + str(i % 5)) for i in range(8)],
            5,
            2,
            4,
        )


def test_family_aggregate_uses_only_effective_source_points(tmp_path):
    import gzip

    rows = [
        dict(
            sample_id="a",
            family_id="F",
            count=2,
            recall1_correct=1,
            recall4_correct=2,
            recall_at_1=0.5,
            recall_at_4=1.0,
        ),
        dict(
            sample_id="b",
            family_id="F",
            count=1,
            recall1_correct=0,
            recall4_correct=0,
            recall_at_1=0.0,
            recall_at_4=0.0,
        ),
    ]
    with gzip.open(tmp_path / "x_per_pair.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    np.savez(
        tmp_path / "x_distances.npz",
        sample_id=np.array(["a", "b"]),
        family_id=np.array(["F", "F"]),
        nearest_distance_px=np.array([[2.0, 6.0], [20.0, 0.0]]),
        supervised=np.array([[True, True], [True, False]]),
    )
    summary = aggregate_families(tmp_path, "x")
    assert (
        summary["families"] == 1
        and summary["family_distributions"]["median_distance_px"]["50"] == 6
    )
    assert summary["family_distributions"]["recall_at_4"]["50"] == pytest.approx(2 / 3)
