# Semantically Guided Disambiguation Matching

光学–SAR 精确配准研究的代码仓库。方法设计包含语义区域路由 SRR、假设推理 HR 和鲁棒配准 RR。

当前交付对应实验前置任务，包括三套数据的全量校验、WHU 底图对齐初筛和可重复运行的几何测试。神经网络训练与最终配准精度实验属于后续阶段。

## 在 zzvr 运行

仓库路径为 `/mnt/sda/zhushihao/Semantically-guided-disambiguation-matching`。

专用 `.venv` 继承已有 `/home/zzvr/miniconda3/envs/compared` 的 Python 3.10、PyTorch 和 OpenCV，并安装 rasterio、pyproj 和 pytest。实际环境记录见报告的 `environment.json`。

```bash
ssh zzvr
cd /mnt/sda/zhushihao/Semantically-guided-disambiguation-matching
export PYTHONPATH=.

.venv/bin/python -u scripts/audit_datasets.py --config configs/zzvr.json
.venv/bin/python scripts/check_whu_archives.py --config configs/zzvr.json
.venv/bin/python scripts/diagnose_alignment.py --config configs/zzvr.json
.venv/bin/python scripts/render_checkpoints.py --config configs/zzvr.json
.venv/bin/python scripts/verify_dataset_geometry.py --config configs/zzvr.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  --junitxml=reports/zzvr_20261125/geometry/junit.xml
```

其他机器使用 Python ≥3.10，并安装 `pip install -e '.[test]'`。数据路径和输出路径通过 JSON 配置指定。数据读取只访问原文件，所有报告写入配置中的输出目录。

## 数据校验

每套数据生成 `manifest.jsonl`、`anomalies.jsonl`、`summary.json` 和样本预览。异常分为 `error`、`warning`、`info`，出现 `error` 时审计命令返回非零退出码。TIFF 按块完整解码，记录波段、尺寸、类型、值域、非有限数、零值比例、有效掩码、NoData、CRS、仿射地理变换和边界。零值比例单独记录，实际无效像素依照有效掩码和非有限值统计。

### WHU

现有服务器文件是 512×512 灰度配准派生包。训练目录有 5600 个样本，测试目录有 700 个样本，每个样本包含光学 TIFF、扰动 SAR TIFF、`.flo` 和六元素 NumPy `datum`。

`datum[0]` 保存光学和扰动 SAR 两个通道，`datum[2]` 为合成仿射，`datum[4]` 保存扰动前的 SAR。读取旧式对象数组时使用受限的 NumPy 类型白名单。

校验四类文件的逐 ID 对应和完整解码，检查嵌入图像与 TIFF 相等，检查 flow 的有限值、尺寸、仿射性及其与 `datum[2]` 的正向映射一致性。另核对原 ZIP 索引中的文件名和解压尺寸。

该派生包没有原始 100 景影像的场景编号、RGB/NIR 波段、土地利用标签和地理参考。因此报告将原版 WHU 数据完整性、真实地理覆盖、父场景独立性与训练测试的空间泄漏检查标为未验证。现有训练包也没有官方验证划分。

### SRIF 光学–SAR

核验编号 1–200 的两张 JPG 与一份 `gt_*.txt`，并检查矩阵尺寸、有限值、行列式和条件数。GT 按作者 `demo_GT.m` 的使用方式解释为 image 1 → image 2。预览抽查中的 image 1 为光学，image 2 为 SAR，使用统一光学 → SAR 方向时沿用该角色约定。

### SpaceNet 9

核验 3 对训练影像和 2 对公开测试影像。训练 CSV 必须包含 `sar_row,sar_col,optical_row,optical_col`，检查有限值、越界、重复及小数坐标。影像读取和地理参考检查覆盖全部公开影像。

公开测试 CSV 和最终私测数据由官方保留，在异常表中以 `info` 记录其发布状态。公开训练场景可用于自行划分的本地评估，论文需明确其使用方式。

## WHU 底图初筛

固定随机种子从提供的两个 split 中选择 12 个分散编号的裁块，生成原始光学、扰动前 SAR、边缘叠加、扰动 SAR、flow 恢复图和局部位移代理图。

`configs/whu_screening_points.json` 保存 12 个裁块的地物判读记录及 8 个同名结构候选，其中 6 个用于初筛、2 个因放大复核后仍难以定位而弃用。保留的点坐标由图像目视估读，单侧判读半径为 5–6 像素，并保留对应的标记图和放大图。山地、森林和重复屋顶中难以确定的点保留为未选取。父场景关系无法由现有编号恢复。

梯度相位相关用于发现可能的整体与局部偏移，其输出明确标记为代理量。初步点和代理结果仅用于筛查，精度评估需要进一步独立复核的控制点。

## 几何约定与测试

统一采用零起点像素中心 `(x,y)`，其中 `x=col`、`y=row`。矩阵作用于列向量，`H` 表示光学 → SAR。两侧分别应用旧像素 → 新像素增强 `A_o`、`A_s` 后，矩阵更新为

```text
H_new = A_s @ H @ inverse(A_o)
```

图像尺寸使用 `(height,width)`。缩放坐标采用半像素约定，`grid_sample` 使用 `align_corners=False`。特征层默认采用与 resize 相同的像素中心网格，具有其他感受野中心的步长卷积需要提供实际 stride/offset 变换。

几何测试通过真实的 OpenCV 图像 warp、PyTorch 双线性采样以及非对称仿射验证方向和坐标约定，覆盖裁剪、缩放、翻转组合，奇数及非方形尺寸，加权六自由度仿射解、大坐标稳定性、零权重、非有限输入、点数不足、共线、近共线及目标坍缩。

`verify_dataset_geometry.py` 额外用真实 WHU 裁块比较 metadata 矩阵、flow、角点位移和图像 warp，并比较正确方向与反方向的重采样误差。该验证针对合成几何，光学与底图 SAR 的原生对齐仍由独立检查点判读。

## 数据来源

- WHU-OPT-SAR 原版发布和论文可从作者仓库核对。服务器派生 ZIP 缺少原始生成脚本和来源说明，当前报告保留这一溯源缺项。[原版发布](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
- SRIF 来自 Li, Hu and Zhang 的 2023 年多模态匹配基准。[官方仓库](https://github.com/LJY-RS/SRIF) · [论文](https://doi.org/10.1016/j.isprsjprs.2023.08.010)
- SpaceNet 9 使用官方公开文件布局和字段定义。[官方仓库](https://github.com/SpaceNetChallenge/SpaceNet9) · [发布页](https://spacenet.ai/sn9-challenge/)

数据集原始文件与 Python 环境保留在服务器。仓库的 `artifacts/zzvr_20261125/` 保存本次运行的压缩清单、异常、图像诊断和测试证据。
