"""Describe the already-frozen epoch20 cases; classifications never filter data."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sdgm.io import save_csv, save_json
from sdgm.os_family_data import _load_native
from sdgm.raw_lineage import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/os_continuation_diagnostic.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = Path(config["output"])
    observations = json.loads(
        (root / "fixed_failures/epoch20_query_observations.json").read_text()
    )
    if (root / "failure_query_freeze.json").exists():
        freeze = json.loads((root / "failure_query_freeze.json").read_text())
        assert freeze["queries_sha256"] == sha256_file(
            root / "fixed_failure_queries.json"
        )
    else:
        save_json(
            root / "failure_query_freeze.json",
            dict(
                queries_sha256=sha256_file(root / "fixed_failure_queries.json"),
                baseline_observations_sha256=sha256_file(
                    root / "fixed_failures/epoch20_query_observations.json"
                ),
                selection_epoch=20,
                all_future_comparisons_use_same_query_and_pair_IDs=True,
            ),
        )
    reviews = []
    for row in observations:
        category = row["proposed_category"]
        rank = int(np.argmin(row["candidate_distances_px"])) + 1
        if category == "near_gt_miss":
            label = "1_接近合成标签但未命中"
            status = "confirmed_relative_to_synthetic_label"
            confidence = "high_geometry_only"
            basis = f"最近候选距标签{row['nearest_distance_px']:.4f}px，刚超过固定8px门槛，位于候选排序第{rank}；top1误差{row['top1_distance_px']:.2f}px。"
            visual = "最近候选的坐标接近标签；部分案例top1仍在远处，因此还包含排序问题，不等同于所有候选只需精定位。"
        elif category == "similar_structure_candidate":
            label = "2_相似结构或沿重复边界别处匹配"
            status = "appearance_consistent_hypothesis_not_physical_identity_proof"
            confidence = "medium_visual"
            basis = f"最近候选误差{row['nearest_distance_px']:.2f}px、top1误差{row['top1_distance_px']:.2f}px，落在另一位置。"
            visual = "固定裁块可见重复田块条带、边界或弱纹理；候选处与标签处都有相似局部SAR响应。仅凭散斑纹理不能确认同一地面结构，保留沿边界滑动/重复外观歧义。"
        elif category == "large_range_error":
            label = "3_大范围错误"
            status = "confirmed_relative_to_synthetic_label"
            confidence = "high_geometry_only"
            basis = f"4个原生候选中最近仍为{row['nearest_distance_px']:.2f}px，远超8px门槛；top1为{row['top1_distance_px']:.2f}px。"
            visual = "候选窗口落在图中另一片区域，不能仅用小范围定位偏差解释。可见相似田块和散斑，不据此认定具体成因。"
        else:
            label = "4_有效域或标签待核查"
            status = "boundary_sensitivity_flag_no_confirmed_mask_or_label_error"
            confidence = "review_flag_only"
            basis = f"标签距实际有效支持边界约{row['actual_support_margin_px']:.1f}px；源点坐标{row['source_pixel_xy']}；当前仍被既定有效性定义监督。"
            visual = "光学源点或SAR标签靠近裁块/填充边界，局部上下文不完整。当前未发现图外GT、无效源或图内GT支持错误；原生对应精度未独立认证，不能将边界表现直接判为标签错。"
        reviews.append(
            dict(
                case_id=row["case_id"],
                sample_id=row["sample_id"],
                family_id=row["family_id"],
                query_index=row["query_index"],
                category=label,
                review_status=status,
                confidence=confidence,
                geometry_evidence=basis,
                visual_evidence=visual,
                nearest_candidate_rank=rank,
                nearest_distance_px=row["nearest_distance_px"],
                top1_distance_px=row["top1_distance_px"],
                actual_support_margin_px=row["actual_support_margin_px"],
                mask_error_confirmed=False,
                label_error_confirmed=False,
                observer="Codex same-observer visual screening; no independent human ground control audit",
                selected_at_epoch20=True,
                sample_removed=False,
                semantic_solution_assumed=False,
                uncertainty="all distances conditional on nominal native-base alignment and exact synthetic affine",
            )
        )
    save_json(
        root / "fixed_failure_review.json",
        dict(
            review_unit="source query; six fixed pairs can contain multiple failure types",
            cases=reviews,
            none_deleted=True,
            category4_confirmed_mask_errors=0,
            category4_confirmed_label_errors=0,
            semantic_remedy_not_assumed=True,
        ),
    )
    save_csv(root / "fixed_failure_review.csv", reviews)
    records = {
        r["sample_id"]: r
        for r in (
            json.loads(l)
            for l in (root / "validation_samples.jsonl").read_text().splitlines()
        )
    }
    plans = {
        r["sample_id"]: r
        for r in (
            json.loads(l)
            for l in (root / "validation_transforms.jsonl").read_text().splitlines()
        )
    }
    output = root / "failure_review_cards"
    output.mkdir(exist_ok=True)
    for epoch in config["comparison_epochs"]:
        path = root / f"fixed_failures/epoch{epoch:02d}_query_observations.json"
        if not path.exists():
            continue
        obs = json.loads(path.read_text())
        for sample_id in sorted({r["sample_id"] for r in obs}):
            cases = [r for r in obs if r["sample_id"] == sample_id]
            optical, base = _load_native(records[sample_id])
            matrix = np.asarray(plans[sample_id]["optical_input_to_sar_input"])
            sar = cv2.warpAffine(
                base,
                matrix[:2],
                (256, 256),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            canvas = Image.new("RGB", (768, 240 * len(cases)), "white")
            draw = ImageDraw.Draw(canvas)
            for k, row in enumerate(cases):
                q = row["query_index"]
                nearest = int(np.argmin(row["candidate_distances_px"]))
                y = k * 240
                draw.text(
                    (4, y + 1),
                    f"Q{q} {row['proposed_category']} | nearest={row['nearest_distance_px']:.2f}px rank={nearest+1} | top1={row['top1_distance_px']:.2f}px",
                    fill="black",
                )
                choices = [
                    (optical, row["source_pixel_xy"], "Optical source", "cyan"),
                    (sar, row["gt_pixel_xy"], "SAR label", "green"),
                    (sar, row["candidate_xy"][0], "SAR top1", "red"),
                    (
                        sar,
                        row["candidate_xy"][nearest],
                        "SAR nearest of four",
                        "orange",
                    ),
                ]
                for j, (a, xy, text, color) in enumerate(choices):
                    cx, cy = np.rint(xy).astype(int)
                    crop = (
                        Image.fromarray(a)
                        .convert("RGB")
                        .crop((cx - 48, cy - 48, cx + 48, cy + 48))
                        .resize((192, 192), Image.Resampling.NEAREST)
                    )
                    d = ImageDraw.Draw(crop)
                    d.line((88, 96, 104, 96), fill=color, width=2)
                    d.line((96, 88, 96, 104), fill=color, width=2)
                    canvas.paste(crop, (j * 192, y + 42))
                    draw.text((j * 192 + 3, y + 19), text, fill="black")
                    draw.text((j * 192 + 3, y + 29), str(np.round(xy, 1)), fill="black")
            canvas.save(output / f'epoch{epoch:02d}_{sample_id.rsplit("/",1)[-1]}.png')
    print(
        "Frozen case review saved; no sample, label or training configuration modified"
    )


if __name__ == "__main__":
    main()
