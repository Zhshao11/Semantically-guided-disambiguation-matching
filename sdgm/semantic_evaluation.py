"""Fixed-view prototype diagnostics, paired candidate changes and family intervals."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .coarse import spatial_candidates
from .family_evaluation import quantiles
from .io import save_csv, save_json
from .semantic_routing import (
    distillation_loss,
    probability_normalize,
    routed_candidates,
    sample_native_map,
    semantic_bias,
    shuffle_semantic,
)
from scripts.train_structure_smoke import prepare, geometry_for


class DistributionStats:
    """Query-weighted diagnostics; spatial variability also computed per image."""

    def __init__(self, prototypes=32):
        self.n = 0
        self.mass = np.zeros(prototypes, dtype="float64")
        self.hard = np.zeros(prototypes, dtype="int64")
        self.entropy_sum = 0.0
        self.max_sum = 0.0
        self.spatial_rms = []
        self.spatial_l1 = []
        self.image_js = []

    def add(self, probabilities, keep):
        # B,M,Q, B,Q; only the existing supervised query mask is used.
        for p, k in zip(probabilities, keep):
            p = p[:, k].T.double()
            if not len(p):
                continue
            self.n += len(p)
            self.mass += p.sum(0).cpu().numpy()
            self.hard += (
                torch.bincount(p.argmax(-1), minlength=p.shape[-1]).cpu().numpy()
            )
            self.entropy_sum += float(-(p * p.clamp_min(1e-12).log()).sum())
            self.max_sum += float(p.max(-1).values.sum())
            mean = p.mean(0)
            self.spatial_rms.append(float(((p - mean).square().mean()).sqrt()))
            self.image_js.append(
                float(
                    (p * (p.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log()))
                    .sum(-1)
                    .mean()
                )
            )
            if len(p) > 1:
                self.spatial_l1.append(float((p[1:] - p[:-1]).abs().sum(-1).mean()))

    def summary(self):
        soft = self.mass / max(self.n, 1)
        hard = self.hard / max(self.n, 1)
        entropy = -(soft * np.log(np.maximum(soft, 1e-12))).sum()
        return dict(
            valid_queries=self.n,
            mean_soft_occupancy=soft.tolist(),
            hard_counts=self.hard.tolist(),
            hard_occupancy=hard.tolist(),
            hard_active_prototypes=int((self.hard > 0).sum()),
            hard_active_above_1pct=int((hard >= 0.01).sum()),
            largest_hard_fraction=float(hard.max()),
            mean_query_entropy_nats=self.entropy_sum / max(self.n, 1),
            mean_query_entropy_normalized=self.entropy_sum
            / max(self.n, 1)
            / np.log(len(soft)),
            marginal_entropy_nats=float(entropy),
            marginal_effective_prototypes=float(np.exp(entropy)),
            marginal_minus_conditional_entropy_nats=float(
                entropy - self.entropy_sum / max(self.n, 1)
            ),
            mean_max_probability=self.max_sum / max(self.n, 1),
            per_image_spatial_probability_rms=quantiles(self.spatial_rms),
            per_image_JS_to_image_mean_nats=quantiles(self.image_js),
            adjacent_source_query_L1=quantiles(self.spatial_l1),
            interpretation="hard assignments and entropy are diagnostics, not semantic class validation; adjacent source queries are row-major, not exclusively 2D neighbors",
        )


def frozen_views(config, train, validation):
    from .continuation_diagnostics import FrozenDiagnosticView, verify_monitor

    root = Path(config["diagnostic_root"])
    verify_monitor(root)
    lookup = {r["sample_id"]: i for i, r in enumerate(train.records)}
    monitor_plans = [
        json.loads(l)
        for l in (root / "monitor_transforms.jsonl").read_text().splitlines()
    ]
    val_plans = [
        json.loads(l)
        for l in (root / "validation_transforms.jsonl").read_text().splitlines()
    ]
    return (
        FrozenDiagnosticView(
            train, [lookup[p["sample_id"]] for p in monitor_plans], monitor_plans
        ),
        FrozenDiagnosticView(validation, range(len(validation)), val_plans),
    )


@torch.no_grad()
def distillation_diagnostics(
    model, head, dataset, parent, device, output, tag, consistency=False
):
    model.eval()
    head.eval()
    teacher_stats, student_stats = DistributionStats(), DistributionStats()
    rows = []
    sum_kl = 0.0
    n_total = 0
    consistency_sum = 0.0
    for batch in DataLoader(dataset, batch_size=16, shuffle=False):
        batch = prepare(batch, device)
        result = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        p = head(result["sar_descriptors"])
        _, kl, sampled, teacher = distillation_loss(
            batch["teacher"], p, model.source_pixel_xy, geom
        )
        keep = geom["supervised"]
        teacher_stats.add(teacher, keep)
        student_stats.add(sampled, keep)
        n = int(keep.sum())
        n_total += n
        sum_kl += float(kl[keep].sum())
        consistency_per = torch.zeros_like(kl)
        if consistency:
            # Same SAR native base, no new visibility assumptions. Sample/compare only
            # the existing valid transformed queries, using nominal identity base.
            unwarped = []
            view = dataset.view
            base = view.base
            ids = {r["sample_id"]: i for i, r in enumerate(base.records)}
            for sample_id in batch["id"]:
                unwarped.append(base.arrays[ids[sample_id], 1])
            native = (
                torch.from_numpy(np.stack(unwarped).copy()).to(device)[:, None].float()
                / 127.5
                - 1
            )
            desc = torch.nn.functional.normalize(
                model.encoder(model.sar_adapter(native)), dim=1, eps=1e-6
            )
            base_q = probability_normalize(
                sample_native_map(head(desc), model.source_pixel_xy)
            )
            mid = (sampled + base_q) / 2
            consistency_per = (
                0.5
                * (
                    sampled
                    * (sampled.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())
                    + base_q
                    * (base_q.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())
                )
            ).sum(1)
            consistency_sum += float(consistency_per[keep].sum())
        for i, sample in enumerate(batch["id"]):
            k = keep[i]
            ni = int(k.sum())
            rows.append(
                dict(
                    sample_id=sample,
                    family_id=batch["family_id"][i],
                    effective_queries=ni,
                    KL=float(kl[i, k].mean()) if ni else None,
                    augmentation_JS=(
                        float(consistency_per[i, k].mean())
                        if ni and consistency
                        else None
                    ),
                )
            )
    summary = dict(
        tag=tag,
        pairs=len(dataset),
        families=len({r["family_id"] for r in dataset.records}),
        effective_queries=n_total,
        KL=sum_kl / max(n_total, 1),
        teacher=teacher_stats.summary(),
        SAR=student_stats.summary(),
        augmentation_consistency_JS_nats=(
            consistency_sum / max(n_total, 1) if consistency else None
        ),
        consistency_definition="JS[qs(Ax), qs_base(x)] at same144 points and existing valid supervision mask; no inferred trustworthy-visibility mask",
        test_evaluated=False,
        UBC_evaluated=False,
    )
    save_json(Path(output) / f"{tag}_distillation.json", summary)
    save_csv(Path(output) / f"{tag}_distillation_per_pair.csv", rows)
    return summary


@torch.no_grad()
def calibrate_lambda(model, head, dataset, parent, config, device):
    model.eval()
    head.eval()
    total = 0
    variance_s = 0.0
    variance_b = 0.0
    mins = []
    maxs = []
    mean_b = 0.0
    mean_s = 0.0
    for batch in DataLoader(dataset, batch_size=16, shuffle=False):
        batch = prepare(batch, device)
        out = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        q = probability_normalize(
            sample_native_map(batch["teacher"], model.source_pixel_xy)
        )
        bias = semantic_bias(q, head(out["sar_descriptors"]))
        valid = geom["target_valid"][:, None].expand_as(bias)
        count = valid.sum(-1).clamp_min(1)
        keep = geom["supervised"]
        for value, label in [(out["scores"], "s"), (bias, "b")]:
            means = (value * valid).sum(-1) / count
            variances = (((value - means[..., None]) ** 2) * valid).sum(-1) / count
            if label == "s":
                variance_s += float(variances[keep].sum())
                mean_s += float(means[keep].sum())
            else:
                variance_b += float(variances[keep].sum())
                mean_b += float(means[keep].sum())
        total += int(keep.sum())
        mins.append(float(bias[valid].min()))
        maxs.append(float(bias[valid].max()))
    s = (variance_s / total) ** 0.5
    b = (variance_b / total) ** 0.5
    weight = (
        min(
            config["routing"]["lambda_cap"],
            config["routing"]["relative_std_target"] * s / b,
        )
        if b >= 1e-8
        else 0.0
    )
    return dict(
        lambda_value=weight,
        training_queries=total,
        structural_within_query_rms_std=s,
        semantic_bias_within_query_rms_std=b,
        structural_mean=mean_s / total,
        semantic_bias_mean=mean_b / total,
        observed_bias_min=min(mins),
        observed_bias_max=max(maxs),
        fused_bias_absolute_bound=weight,
        achieved_bias_to_structural_std_ratio=weight * b / s,
        desired_std_ratio=config["routing"]["relative_std_target"],
        cap=config["routing"]["lambda_cap"],
        split="fixed_training_monitor_only",
        validation_used=False,
        note="constant semantic offset cannot alter ranking; report centered variability separately from absolute bound",
    )


def candidate_arrays(candidate, geometry):
    d = torch.linalg.vector_norm(
        candidate["pixel_xy"] - geometry["gt_pixel_xy"][:, :, None], dim=-1
    )
    d = d.masked_fill(~candidate["valid"], float("inf"))
    return d[..., 0], d.min(-1).values


def family_bootstrap(family_rows, config):
    arrays = np.array(
        [
            [r["effective_queries"], r["A_hits"], r["C_hits"], r["shuffle_hits"]]
            for r in family_rows
        ],
        dtype="float64",
    )
    rng = np.random.default_rng(config["seed"])
    values = []
    for _ in range(config["replicates"]):
        total = arrays[rng.integers(0, len(arrays), len(arrays))].sum(0)
        values.append(
            [(total[2] - total[1]) / total[0], (total[2] - total[3]) / total[0]]
        )
    a = np.array(values)
    return dict(
        replicates=config["replicates"],
        seed=config["seed"],
        unit="resample whole families with replacement",
        C_minus_A_recall4_interval95=np.percentile(a[:, 0], [2.5, 97.5]).tolist(),
        C_minus_shuffle_recall4_interval95=np.percentile(a[:, 1], [2.5, 97.5]).tolist(),
        interpretation="descriptive family-sampling uncertainty conditional on this trained seed and split; not repeat-seed stability or geographic-independence evidence",
    )


@torch.no_grad()
def evaluate_routing(
    model, head, dataset, parent, config, device, output, split, weight
):
    model.eval()
    head.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods = ["A", "B", "C", "C_shuffle"]
    storage = {m: dict(first=[], nearest=[], indices=[], valid=[]) for m in methods}
    keeps = []
    pair_rows = []
    sample_ids = []
    families = []
    masked = defaultdict(int)
    identical = True
    for batch in DataLoader(dataset, batch_size=parent["batch_size"], shuffle=False):
        batch = prepare(batch, device)
        out = model(batch["optical"], batch["sar"])
        geom = geometry_for(model, batch, parent)
        ps = head(out["sar_descriptors"])
        qo = probability_normalize(
            sample_native_map(batch["teacher"], model.source_pixel_xy)
        )
        bias = semantic_bias(qo, ps)
        shuffled = semantic_bias(
            qo, shuffle_semantic(ps, batch["id"], config["routing"]["shuffle_seed"])
        )
        kwargs = dict(
            target_valid=geom["target_valid"], target_xy=model.target_pixel_xy
        )
        predictions = {
            "A": spatial_candidates(
                out["scores"],
                geom["target_valid"],
                model.target_pixel_xy,
                count=4,
                nms_radius_px=24,
            ),
            "B": routed_candidates(out["scores"], bias, weight=0, **kwargs),
            "C": routed_candidates(out["scores"], bias, weight=weight, **kwargs),
            "C_shuffle": routed_candidates(
                out["scores"], shuffled, weight=weight, **kwargs
            ),
        }
        for key in predictions["A"]:
            if not torch.equal(predictions["A"][key], predictions["B"][key]):
                raise AssertionError(f"lambda0 changed original path: {key}")
        for m in ["C", "C_shuffle"]:
            assert torch.equal(
                predictions[m]["indices"][..., :2], predictions["A"]["indices"][..., :2]
            )
        for key in [
            "gt_outside",
            "empty_positive",
            "invalid_source",
            "gt_invalid_support",
        ]:
            masked[key] += int(geom[key].sum())
        keep = geom["supervised"].cpu().numpy()
        keeps.extend(keep)
        distances = {m: candidate_arrays(predictions[m], geom) for m in methods}
        for m in methods:
            for key, value in zip(["first", "nearest"], distances[m]):
                array = value.cpu().numpy()
                if not np.isfinite(array[keep]).all():
                    raise FloatingPointError("No valid candidate for supervised query")
                storage[m][key].extend(array)
            for key in ["indices", "valid"]:
                storage[m][key].extend(predictions[m][key].cpu().numpy())
        for i, sample in enumerate(batch["id"]):
            k = keep[i]
            n = int(k.sum())
            sample_ids.append(sample)
            families.append(batch["family_id"][i])
            hits = {m: (distances[m][1][i].cpu().numpy() <= 8) & k for m in methods}
            row = dict(
                sample_id=sample, family_id=batch["family_id"][i], effective_queries=n
            )
            for m in methods:
                f, near = [v[i].cpu().numpy() for v in distances[m]]
                row.update(
                    {
                        f"{m}_hits": int(hits[m].sum()),
                        f"{m}_recall1": float(((f <= 8) & k).sum() / n) if n else None,
                        f"{m}_recall4": float(hits[m].sum() / n) if n else None,
                        f"{m}_median_px": float(np.median(near[k])) if n else None,
                        f"{m}_p95_px": float(np.percentile(near[k], 95)) if n else None,
                    }
                )
            row["new_hits_C"] = int((~hits["A"] & hits["C"] & k).sum())
            row["lost_hits_C"] = int((hits["A"] & ~hits["C"] & k).sum())
            row["new_hits_shuffle"] = int((~hits["A"] & hits["C_shuffle"] & k).sum())
            row["lost_hits_shuffle"] = int((hits["A"] & ~hits["C_shuffle"] & k).sum())
            row["changed_candidate_queries_C"] = int(
                (
                    (predictions["A"]["indices"][i] != predictions["C"]["indices"][i])
                    .any(-1)
                    .cpu()
                    .numpy()
                    & k
                ).sum()
            )
            pair_rows.append(row)
    keep = np.stack(keeps)
    arrays = {
        "sample_id": np.array(sample_ids),
        "family_id": np.array(families),
        "supervised": keep,
    }
    summary = dict(
        split=split,
        pairs=len(pair_rows),
        families=len(set(families)),
        effective_queries=int(keep.sum()),
        lambda_value=weight,
        A_B_bit_identical=identical,
        masked=dict(masked),
        methods={},
        source_family_isolation=True,
        geographic_independence="not_claimed",
        test_evaluated=False,
        UBC_evaluated=False,
    )
    for m in methods:
        for key in storage[m]:
            arrays[f"{m}_{key}"] = np.stack(storage[m][key])
        first = arrays[f"{m}_first"][keep]
        near = arrays[f"{m}_nearest"][keep]
        summary["methods"][m] = dict(
            recall_at_1=float((first <= 8).mean()),
            recall_at_4=float((near <= 8).mean()),
            nearest_candidate_distance_px=quantiles(near),
            invalid_candidate_slots=int((~arrays[f"{m}_valid"]).sum()),
        )
    family_rows = []
    for family in sorted(set(families)):
        ids = np.flatnonzero(np.array(families) == family)
        members = [pair_rows[i] for i in ids]
        k = keep[ids]
        n = int(k.sum())
        row = dict(family_id=family, pairs=len(ids), effective_queries=n)
        for m in methods:
            d = arrays[f"{m}_nearest"][ids][k]
            row.update(
                {
                    f"{m}_hits": int((d <= 8).sum()),
                    f"{m}_recall4": float((d <= 8).mean()),
                    f"{m}_median_px": float(np.median(d)),
                    f"{m}_p95_px": float(np.percentile(d, 95)),
                }
            )
        for key in [
            "new_hits_C",
            "lost_hits_C",
            "new_hits_shuffle",
            "lost_hits_shuffle",
            "changed_candidate_queries_C",
        ]:
            row[key] = sum(r[key] for r in members)
        row["net_hits_C"] = row["new_hits_C"] - row["lost_hits_C"]
        row["delta_recall4_C_minus_A"] = row["net_hits_C"] / n
        row["shuffle_hits"] = row["C_shuffle_hits"]
        family_rows.append(row)
    summary["replacement"] = {
        key: sum(r[key] for r in pair_rows)
        for key in [
            "new_hits_C",
            "lost_hits_C",
            "new_hits_shuffle",
            "lost_hits_shuffle",
            "changed_candidate_queries_C",
        ]
    }
    summary["replacement"]["net_hits_C"] = (
        summary["replacement"]["new_hits_C"] - summary["replacement"]["lost_hits_C"]
    )
    assert summary["replacement"]["net_hits_C"] == sum(
        r["C_hits"] - r["A_hits"] for r in family_rows
    )
    summary["family_paired_change"] = dict(
        improved=sum(r["net_hits_C"] > 0 for r in family_rows),
        degraded=sum(r["net_hits_C"] < 0 for r in family_rows),
        unchanged=sum(r["net_hits_C"] == 0 for r in family_rows),
        delta=quantiles([r["delta_recall4_C_minus_A"] for r in family_rows]),
    )
    summary["family_bootstrap"] = family_bootstrap(family_rows, config["bootstrap"])
    save_json(output / f"{split}_routing_summary.json", summary)
    save_csv(output / f"{split}_routing_per_pair.csv", pair_rows)
    save_csv(output / f"{split}_routing_per_family.csv", family_rows)
    np.savez_compressed(output / f"{split}_routing_queries.npz", **arrays)
    return summary
