# Semantically Guided Disambiguation Matching

最新完成[三层同位置语义瓶颈诊断](artifacts/os_semantic_bottleneck_20261127/REPORT_zh.md)。模型、语义头、原型数、温度、相似度、lambda、部署4候选与8px规则全部冻结。每个查询固定同一P（8px正网格）及N（正域外16个结构NMS难负峰），比较A原DINO同图余弦、B原型教师同图点积、C实际SAR跨模态点积。**A/B使用GT逆映射且有同图自相似优势，不是严格跨模态上界。**

| 验证查询组 | A：最佳正胜最高难负 | B：最佳正胜最高难负 | C：最佳正胜最高难负 | C全网格最佳正名次中位数 |
| --- | ---: | ---: | ---: | ---: |
| 全部116,953 | 99.5143% | 36.7071% | 11.9005% | 136 |
| 原结构漏检46,631 | 99.4403% | 34.1725% | 11.4924% | 142 |
| 新增277 | 99.6390% | 30.3249% | 88.0866% | 2 |
| 丢失5,645 | 99.5040% | 31.4969% | 10.0266% | 158 |

平局单列，不依赖数组顺序；完整名次区间、Delta分布、P×N胜负平、NMS前后覆盖、连续GT检查及family配对区间均已保存。C原始top4/语义独立NMS4覆盖为4.1906%/4.8729%；部署2+2的55.5385%仍主要来自结构前缀。丢失组新增2语义峰最近者在8–16px有229个、16–32px有623个、>32px有4,793个，且没有查询的P被结构前缀全部屏蔽。当前表现不能主要归结为8px边缘误差或NMS；压缩加评分和实际SAR排序都有落差，但未完成各原因的因果分离。70项测试与独立NumPy核验通过，无训练更新，测试池继续封存。

```bash
export PYTHONPATH=.
# 仅诊断；GT位置不进入部署候选。已有output拒绝覆盖。
.venv/bin/python scripts/diagnose_os_semantic_bottleneck.py --config configs/os_semantic_bottleneck.json
.venv/bin/python scripts/summarize_os_semantic_bottleneck.py --config configs/os_semantic_bottleneck.json
.venv/bin/python scripts/verify_os_semantic_bottleneck.py --config configs/os_semantic_bottleneck.json
```

此前完成[不重训的纯语义补充最小对照](artifacts/os_pure_semantic_supplement_20261126/REPORT_zh.md)。固定现有第30轮结构与同一5000步语义头，原生256、固定样本/扰动、4候选与8px门槛全部不变。纯语义分数为q_o^T q_s，保留相同前2结构峰，再从**全部有效1024目标网格**按原顺序24px NMS补2峰，没有结构候选池预筛。

| 验证设置 | @4 | 新增命中 | 丢失命中 | 原结构未命中补救率 |
| --- | ---: | ---: | ---: | ---: |
| 4结构 | 60.1284% | 0 | 0 | 0 |
| 2结构+2融合 | 60.1216% | 144 | 152 | 0.3088% |
| 2结构+2纯语义 | 55.5385% | 277 | 5,645 | 0.5940% |

纯语义验证净少5,368个命中、下降4.5899个百分点，238个family全部净退化；配对family重采样95%区间为[−4.8157,−4.3669]个百分点。当前冻结头的纯语义排序未带来净覆盖收益；区间只描述这一模型/种子及来源组下的样本不确定性。原结构与融合逐查询精确复现，独立NumPy对1,152个固定查询核验全网格最大值及NMS通过，全套66项测试通过。无新增训练更新，OS/UBCv2测试继续封存，所有距离相对合成标签解释。

```bash
export PYTHONPATH=.
# 复用既有结构、语义头和教师缓存，无训练；新评估用新output，已有输出拒绝覆盖。
.venv/bin/python scripts/evaluate_os_pure_semantic.py --config configs/os_pure_semantic_supplement.json
.venv/bin/python scripts/verify_os_pure_semantic.py --config configs/os_pure_semantic_supplement.json
.venv/bin/python scripts/summarize_os_pure_semantic.py --config configs/os_pure_semantic_supplement.json
```

此前完成[冻结结构后的DINOv2潜在原型蒸馏与4候选路由](artifacts/os_semantic_routing_20261126/REPORT_zh.md)。第30轮结构前端全冻结；官方DINOv2 ViT-S/14 patch教师、原生灰度复制三通道、252教师输入与半像素坐标映射固定，仅从1,976个训练family各取1对/64token拟合32个球面原型。先做32对冒烟，再重置SAR头，按预定预算运行5,000步KL-only蒸馏，没有解冻前端、训练门控或加入关系图。

**该轮未达到覆盖提升标准。** 验证KL由1.5520降至1.1424，SAR原型分布有空间差异但比教师更平滑；同一个最终语义头、固定训练集校准lambda=1（实际偏置波动约结构的3.98%），结果如下。A逐点复现第30轮，B与A候选逐元素一致；C保留2个结构峰、按原24px NMS补2个融合峰，仍为4候选/8原生px门槛。

| 验证设置 | @1 | @4 | 最近距离中位数/P95 |
| --- | ---: | ---: | ---: |
| A：4结构候选 | 47.8337% | 60.1284% | 6.5377 / 28.4306 px |
| B：同一语义头，lambda=0 | 47.8337% | 60.1284% | 6.5377 / 28.4306 px |
| C：2结构+2融合 | 47.8337% | 60.1216% | 6.5372 / 28.5561 px |
| C：打乱SAR语义空间 | 47.8337% | 60.1413% | 6.5370 / 28.5226 px |

验证新增144命中、丢失152、净少8；family改善/退化/不变为63/69/106，重采样区间跨0。低KL、A/B回退与空间打乱均不构成语义有效性的证据，不据本轮结果断言语义普遍无效。62项测试通过，最终头重载精确复现四种设置。OS/UBCv2测试继续封存；全部指标只相对于合成标签。教师缓存、头权重与原始数据留在zzvr，原型中心、抽样清单、每对/family/查询记录与图像随报告归档。

```bash
export PYTHONPATH=.
# 既有冻结数据与第30轮checkpoint；DINO官方代码/权重版本见配置及报告。
# 新训练须用新output；已有教师协议及训练目录拒绝覆盖。
.venv/bin/python scripts/build_os_semantic_teacher.py --config configs/os_semantic_routing.json
.venv/bin/python scripts/train_os_semantic_routing.py --config configs/os_semantic_routing.json
.venv/bin/python scripts/evaluate_os_semantic_routing.py --config configs/os_semantic_routing.json
.venv/bin/python scripts/verify_os_semantic_routing.py --config configs/os_semantic_routing.json
.venv/bin/python scripts/summarize_os_semantic_routing.py --config configs/os_semantic_routing.json
.venv/bin/python scripts/analyze_os_semantic_routing.py --config configs/os_semantic_routing.json
```

此前完成[第20—30轮统一诊断、优化器续训与固定失败分析](artifacts/os_continuation_diagnostic_20261126/REPORT_zh.md)。原训练指标来自固定扰动的 `eval()` 视图，在线随机增强用于训练损失。该轮冻结覆盖238个训练family的952对监测样本，与原952对验证使用完全相同的仿射集合、掩码、144源点、4候选及8原生像素门槛；两侧每轮均有116,953个有效查询。

从第20轮完整恢复模型及AdamW动量/步数，原训练无scheduler，续接常数学习率3e-4，保持架构、损失、温度、正邻域和NMS不变。旧checkpoint未保存global RNG；采样器状态通过原DataLoader索引重放恢复，新checkpoint保存完整RNG状态。

| epoch | 固定监测@4 | 验证@1 | 验证@4 | 验证最近距离中位数/P95 |
| --- | ---: | ---: | ---: | ---: |
| 20 | 68.67% | 47.63% | 59.83% | 6.59 / 28.57 px |
| 25 | 69.38% | 48.30% | 60.56% | 6.50 / 28.13 px |
| 30 | 70.11% | 47.83% | 60.13% | 6.54 / 28.43 px |

验证收益较小且有波动，不能宣称30轮已收敛。20—30每轮的点、图像对和family统计、距离数组及固定6对/24查询失败证据均已归档；有改善也有退化，6例大范围错误仍全部未命中，不预判语义可以解决。全部54项测试通过，终期checkpoint重载精确复现监测和验证。指标相对于合成标签解释；OS最终测试1,696对与UBCv2测试池1,625对未运行模型，WHU重建继续暂停。

```bash
export PYTHONPATH=.
# 使用服务器已有数据与第20轮checkpoint；输出存在时拒绝重训覆盖。
.venv/bin/python scripts/continue_os_diagnostic.py --config configs/os_continuation_diagnostic.json
.venv/bin/python scripts/review_fixed_os_failures.py --config configs/os_continuation_diagnostic.json
.venv/bin/python scripts/summarize_os_continuation.py --config configs/os_continuation_diagnostic.json
.venv/bin/python scripts/verify_os_continuation.py --config configs/os_continuation_diagnostic.json
```

上一轮完成[OS同源分组冻结、原生256正式基线与UBC风险披露](artifacts/os_family_baseline_20261126/REPORT_zh.md)。17,168条来源记录全部保留，512母块、256子块及重复别名按6,441个family分组；使用7,912对训练、952对来源组隔离验证，1,696对最终测试封存，新增3,803对RGB512隔离。已拟合32对的31个family全部归训练，不称地理独立划分。

该无语义基线从头训练20个完整epoch，恢复既定范围的逐epoch仿射扰动，验证种子及矩阵固定。第20epoch的全训练固定视图/验证recall@4为68.46%/59.83%，验证recall@1为47.63%，最近候选距离中位数6.59 px、P95 28.57 px（相对合成标签，8原生像素门槛）。该全训练视图不同于本轮952对监测子集，不能直接比较训练数值。小样本checkpoint在同一隔离验证上的recall@4仅13.62%；正式基线未从该小样本checkpoint继续训练。当时重载完整复现指标，50项测试通过。

UBCv2原1,625对池未改动、未运行新模型。完整88条交叠记录中，train–val 26、train–test 26、val–test 36；62条涉及16个测试样本，均在Jacksonville。只披露风险，不按这些记录删测试样本。

```bash
export PYTHONPATH=.
# 已冻结目录及已有训练run均拒绝覆盖；新版本须使用新output。
.venv/bin/python scripts/freeze_os_families.py --config configs/os_family_baseline.json
.venv/bin/python scripts/verify_os_family_freeze.py --config configs/os_family_baseline.json
.venv/bin/python scripts/train_os_family_baseline.py --config configs/os_family_baseline.json
.venv/bin/python scripts/verify_os_family_baseline.py --config configs/os_family_baseline.json
.venv/bin/python scripts/summarize_os_family_baseline.py --config configs/os_family_baseline.json
.venv/bin/python scripts/record_ubc_overlap_risk.py --config configs/os_family_baseline.json
```


本轮数据路线调整为OS主训练及域内评估候选、UBCv2模型冻结后的跨域基准，SRIF/SpaceNet9补充外部验证；WHU重建暂停。最新交付见[OS/UBCv2原生数据审计及工程拟合报告](artifacts/os_ubc_20261126/REPORT_zh.md)。UBCv2此前已用于分析，不作为盲测，也不用于选择超参数。

OS全量裁块17,168对，UBCv2原始影像7,314对，均完成配对及解码检查。旧OS的256与512有精确裁块父子关系，不能当成独立数据混合；完整地理父场景仍未知。UBCv2官方split间检出88组正面积足迹交叠，已冻结未来官方测试池1,625对，当前未运行UBC模型评估。

原生OS256的固定32对完成800步工程拟合：损失5.9329→0.1388，原生候选召回@1/@4为98.80%/99.19%（8原生像素），最近候选距离中位数59.97→4.68 px。结果仅相对于发布者已对齐前提加已知合成仿射解释；当时未开展隔离验证或真实精配准精度认证；后续来源组隔离结果见上方最新报告。网络仍为原无语义结构编码器，适配为128×32×32描述子、144源点×1024目标位置；默认512接口保持兼容。

```bash
export PYTHONPATH=.
# 审计须使用新的output配置；已有UBC冻结回执时拒绝覆盖。
.venv/bin/python scripts/audit_os_ubc.py --config configs/os_ubc_audit.json
# 仅OS官方train固定样本；重跑训练须指定新的output。
.venv/bin/python scripts/train_structure_smoke.py --config configs/structure_os_native_smoke.json
.venv/bin/python scripts/verify_structure_checkpoint.py --run reports/structure_os_native_20261126
.venv/bin/python scripts/summarize_os_ubc.py
```


光学–SAR 匹配研究的工程仓库。当前实现无语义结构检索、几何函数和原生数据审计；后续研究模块不属于本轮工程验证。

此前已完成三套数据的全量校验、WHU 底图对齐初筛、几何测试，以及单波段无语义结构检索的固定小样本工程拟合。源点与候选接口、稠密分数监督和训练循环均可复用。

本次双轨结果见 [来源核查与结构检索报告](artifacts/dual_track_20261125/REPORT_zh.md)。32 对现有训练样本经过 800 步拟合，损失 7.0817 → 2.0020，原生召回@1 为 88.05%、@4 为 92.98%，固定召回半径 8 原图像素。结果仅用于相对派生标签的工程诊断。

GDROS 官方下载目录已核查，两个 ZIP 的名称和字节数与本地包相符。生成脚本、父影像映射和原生监督精度仍未核实。此前选择的原始 WHU 重建路线现已暂停，当前路线及历史证据见 `configs/data_route.json`。SOPatch Image/SEN1-2 当前仍为待核实备选。

## 原始 WHU 追溯与准入审计

本轮仅下载并解码原版试点：39 个文件、14 个父影像、13 对完整光学–SAR；官方文件目录列出 100 个父影像的 300 个文件。四份清单及判读证据见 [本轮报告](artifacts/raw_rebuild_20261125/REPORT_zh.md)。原始四波段光学和单波段 SAR 保留，未覆盖为灰度。

TIFF 缺少 CRS、地理变换、GCP 和 RPC。使用独立发布的覆盖图与原始 RGB 内容匹配支持近似覆盖，按 3 km 假设缓冲后的连通分量分组，并按预定经度带划分；所有已分配划分均标为暂定，地理独立性未认证；4 个覆盖未确认父影像保持 `unassigned`。不把图幅名或无重复当作独立性证明。

底图抽样预览 10 个父影像的 30 个窗口，登记 8 个暂定同名结构点。名义残差 1.41–5.00 px，加上两侧判读不确定性的筛查量为 8.83–14.12 px；这些不是经独立人工复核的精度统计。当前没有区域获准用于固定 8 px 粗检索监督，也未认证 1–2 px 精定位。`tile_manifest.jsonl` 因此保持为空；生成器通过合成测试，但真实裁块、32 对重训和地理验证尚未运行。现有网络保持原状。

```bash
# 使用既有 zzvr 仓库 .venv；新环境安装 pip install -e '.[raw,smoke,test]'
export PYTHONPATH=.
# 下载必须显式列出试点父影像，不默认获取全套
.venv/bin/python scripts/acquire_raw_whu.py --parents NH49E001014 --modalities optical,sar,label
# 新克隆先将归档source目录复制到配置所指reports/.../source
# 固定顺序：覆盖关联 -> 地理分组 -> 划分 -> 底图审计 -> 准入后裁图
.venv/bin/python scripts/build_raw_manifests.py
.venv/bin/python scripts/prepare_raw_audit.py
.venv/bin/python scripts/audit_raw_alignment.py
# 当前准入下此命令预期退出非零，且不创建真实训练样本
.venv/bin/python scripts/generate_raw_tiles.py
.venv/bin/python scripts/summarize_raw_rebuild.py
```

`audit_raw_alignment.py` 只导出保守筛查记录，不会自动把少量目视点认证成整窗监督。后续准入需要足够精度、覆盖和复核证据。生成器在写入前核对训练与验证准入、冻结清单、原始文件及掩码附属文件哈希，记录完整变换链、插值/填充、有效掩码和固定种子。重复窗口的不同扰动单独计数，不当作独立地理样本。

## 无语义结构检索冒烟测试

单波段光学/SAR 分别经过输入适配器，再进入同一个结构编码器，得到归一化的 `[B,128,64,64]` 描述子。对固定 24×24 个源点计算全部 `576×4096` 分数，使用 GT 邻域概率质量损失训练，并通过空间 NMS 生成每个源点 4 个原生候选。

正邻域固定为原图 8 像素，NMS 半径为 24 像素。候选 API 接收分数、目标坐标和目标物理支持，GT 不作为候选输入。图外监督、空正邻域和候选不足具有显式状态。像素有效性由图像范围和仿射支持定义，零像素保留为正常像素。

本次固定种子从 5600 训练样本中取 32 对，关闭额外几何增强，未使用 700 测试样本调试训练，也未建立独立开发集。DINO、区域图、多仿射假设、精定位、协方差与拒配头属于后续方法阶段。

```bash
export PYTHONPATH=.
# 在现有 zzvr 环境中直接运行。其他环境可安装 pip install -e '.[smoke,test]'。
.venv/bin/python scripts/train_structure_smoke.py --config configs/structure_smoke.json
.venv/bin/python scripts/check_duplicates.py --config configs/zzvr.json
.venv/bin/python scripts/collect_release_evidence.py
.venv/bin/python scripts/summarize_dual_track.py
```

重跑训练时在配置中指定新的输出目录。固定样本及 SHA-256、配置、每步训练日志、全样本指标、距离全集、梯度检查、可视化和显存耗时记录随结果保存。服务器 checkpoint 位于 `reports/structure_smoke_20261125/last.pt`。

精确重复检查覆盖全部 6300 对的光学、原 SAR 和联合像素哈希。近重复采用全体紧阈值感知哈希及跨 split 最近邻筛查，再检验 SIFT/RANSAC/像素相关。本次均未检出确认重复。没有检出不能证明地理独立，父场景分组仍需原始映射。

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
