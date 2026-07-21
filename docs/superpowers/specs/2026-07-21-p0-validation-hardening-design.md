# P0 数值验证与联合判定加固设计规格

## 一、目标

让同语义多语言 hubness 实验流程能够在出现无效数值几何时安全停止，使用 FP32 重新运行 XGLM，并用“同层联合证据 + 表示与密度控制”替代当前按单项指标分别判断复现的宽松规则。

## 二、范围

本次修改只实现 `7.21任务清单/02_本周任务清单.md` 中的 P0 内容：

1. 在隐藏状态提取阶段增加有限值与向量范数检查；
2. 在指标计算阶段增加有限值、相似度矩阵和 kNN 质量守恒检查；
3. XGLM 使用 FP32 计算与存储；
4. 计算四项核心指标在同一层成立的联合连续区间；
5. 收紧单模型状态与跨模型复现判定；
6. 为每一种新增失败条件和状态分类补充测试。

本次不实现语言轮换、tokenization 控制、扩大样本和除“无效值断线显示”之外的完整图表重构。这些内容属于 P1/P2。

## 三、采用的方案

采用严格的 fail-fast 方案：无效数值不得被过滤、跳过或转化为部分指标。结果不完整或存在非有限值的模型统一标记为 `INVALID`，不得参与复现判定。

不采用以下方案：

- 跳过异常层继续计算：这会使不同模型的轨迹不可比，并可能掩盖数值故障；
- 检测失败后自动切换 dtype 重试：这会使实际运行条件不透明，削弱实验可复现性。

## 四、隐藏状态提取设计

修改 `src/extract_hidden.py`，提供可以在不下载模型的情况下进行单元测试的小型数值验证函数。

对于每个句子和每一层隐藏状态，必须验证：

- 完整隐藏状态张量全部为有限值；
- mean-pool 和 sentinel-EOS 向量全部为有限值；
- 两种表示的向量范数均为有限值，并且大于一个很小的正阈值；
- 转换为配置的落盘 dtype 后，向量仍为有限值。

异常信息必须包含：

- 模型名称；
- 表示方式；
- 行号；
- 语义 ID；
- 语言；
- 层号；
- 异常类型。

任意验证失败后，不得继续写入 `.npy`、`metadata.csv` 或 `extraction_manifest.json`。

在 `configs/xglm_1b7_24lang.json` 中显式覆盖：

```json
"dtype": "float32",
"storage_dtype": "float32"
```

这样既保证 XGLM 使用 FP32 推理，也避免在保存阶段重新转换为 FP16 后产生溢出。

## 五、指标几何验证设计

修改 `src/compute_metrics.py`，在多个边界实施防御性验证。

### 5.1 表示数组

加载的表示数组必须满足：

- 数组为三维结构：`行 × 层 × 隐藏维度`；
- 行数与 `metadata.csv` 一致；
- 层数和隐藏维度均大于零；
- 全部元素为有限值。

### 5.2 语义组向量

每个语义组、每一层的向量必须满足：

- 全部元素为有限值；
- 每种语言向量的范数均为有限正数；
- 不允许零范数向量进入归一化。

### 5.3 相似度矩阵

Cosine 和 local-scaled cosine 矩阵必须满足：

- 为方阵；
- 全部元素为有限值；
- 在数值容差内对称；
- 对角线为有限值。

### 5.4 kNN 图

`group_statistics` 必须验证：

- `1 <= k < n`；
- 每个 query 的 fractional top-k 权重和等于 k；
- 整个语义组的 occurrence 总质量等于 `n × k`；
- occurrence、centrality、percentile 和 medoid 全部为有限值；
- medoid 权重和等于 1。

### 5.5 Bootstrap

`bootstrap_mean_ci` 不再静默删除 NaN/Inf。只要输入中存在非有限观察值，就立即抛出异常。

原因是：删除部分语义组会改变统计分母，并可能生成看似有效、实际不可比较的置信区间。

## 六、四项指标同层联合证据设计

建立共享辅助函数，按层联合检查以下四项指标：

- `k_occurrence_excess`；
- `centrality_advantage`；
- `rank_percentile_advantage`；
- `medoid_rate_excess`。

一层只有同时满足以下条件，才能标记为“联合显著为正”：

1. 四项指标各存在且只存在一条记录；
2. 四个 `ci_lower` 都是有限值；
3. 四个 `ci_lower` 都大于 0。

连续区间必须按照真实、连续的整数层号计算。缺失层号必须中断连续区间，不能只根据剩余行的位置计算。

## 七、来源广度联合要求

主表示的正式支持还要求来源广度与四项核心指标在同一层成立。

来源广度阈值为：

- 支持英语的来源语言数不少于全部非英语来源语言的一半；
- 至少覆盖 4 种文字系统；
- 至少包含 3 种非拉丁文字语言。

Validation 输出必须记录：

- 四项指标共同显著的层号；
- 四项指标与来源广度共同成立的层号；
- 最长真实连续区间；
- 配置要求的最小连续层数。

## 八、模型状态设计

每个模型只能处于以下状态之一：

### `INVALID`

存在以下任一情况：

- 必需文件缺失；
- 层或指标缺失；
- 同一 `(layer, metric)` 出现重复记录；
- 存在 NaN/Inf；
- 近邻质量不守恒；
- 结果目录不完整。

### `NOT_SUPPORTED`

主条件 `mean-pool/cosine` 未通过“四项同层联合显著 + 来源广度 + 最小连续层数”规则。

### `REPRESENTATION_SENSITIVE`

主条件通过，但以下至少一项未通过联合连续区间规则：

- `sentinel-EOS/cosine`；
- `mean-pool/local-scaled cosine`。

### `ROBUST`

同时满足：

- `mean-pool/cosine` 通过四项联合证据和来源广度要求；
- `sentinel-EOS/cosine` 通过四项联合连续区间要求；
- `mean-pool/local-scaled cosine` 通过四项联合连续区间要求。

k 扫描作为独立控制项报告，不能把主条件或表示控制不通过的模型升级为 `ROBUST`。

## 九、跨模型复现设计

修改 `src/compare_models.py`，对每个模型进行完整性检查：

- 四项核心指标覆盖每个预期层；
- 在选定表示和相似度方法下，每个 `(layer, metric)` 恰好一条记录；
- `mean`、`ci_lower`、`ci_upper` 全部为有限值；
- 层号从 0 到最大层连续完整；
- 模型的 validation 文件存在且可以解析。

无效模型仍需出现在 verdict 中，并保存具体原因，但不得进入：

- AUC 计算；
- 有效模型计数；
- 复现模型计数；
- 正式对比轨迹。

跨模型 `REPLICATED` 必须至少有两个模型被判定为 `ROBUST`。

`REPRESENTATION_SENSITIVE` 可以作为“条件性证据”报告，但不能触发 `REPLICATED`。

## 十、跨模型图的最小安全调整

本次不进行完整图表重构，只处理 P0 级别的误导风险：

- 每个模型、每项指标使用显式曲线绘制；
- 缺失层或 NaN 必须显示为断点；
- 不允许绘图库跨越缺失区间自动连线；
- 无效模型不进入正式轨迹图，并在 verdict 中说明原因。

## 十一、兼容性与输出

- 保留现有 CSV 文件和主要入口脚本；
- 新增联合证据和模型状态字段，不删除现有基础指标；
- `run_pilot.py` 和 `run_model_suite.py --resume` 的使用方式保持不变；
- 因 XGLM 的解析后配置发生变化，旧 FP16 XGLM 结果不能被 `--resume` 判定为已完成；
- Qwen 和 BLOOM 可以复用已有隐藏向量，重新运行 metrics、validation 和 comparison。

## 十二、测试策略

严格按照测试驱动开发执行：先写失败测试并确认失败原因，再修改生产代码。

单元测试覆盖：

- NaN/Inf 隐藏向量被拒绝；
- 零范数向量被拒绝；
- 含 NaN 的相似度矩阵被拒绝；
- kNN 每行权重和与总 occurrence 质量守恒；
- bootstrap 遇到非有限值时抛出异常，而不是过滤；
- 四项指标必须在同一层成立；
- 缺失层号会中断连续区间；
- 四项指标各自在不同区间为正时不能通过联合判定；
- 无效模型不能参与跨模型复现；
- 必须至少两个 `ROBUST` 模型才能得到 `REPLICATED`。

单元测试通过后，运行现有 synthetic pipeline smoke test，确认在不下载模型的情况下仍能生成：

- metrics；
- validation；
- figures；
- model comparison。

## 十三、验收标准

- XGLM 解析后的计算和存储 dtype 均为 `float32`；
- 非有限或零范数句向量不能进入指标计算；
- 非法 kNN 图不能进入统计汇总；
- 四项指标位于互不重叠区间时不能通过联合 hubness 判定；
- 缺层或存在非有限证据的模型标记为 `INVALID`；
- 跨模型 `REPLICATED` 至少需要两个 `ROBUST` 模型；
- 单元测试、synthetic smoke test 和 Python 编译检查全部通过。

## 十四、本次实施后的直接运行方式

完成代码修改并同步至 AutoDL 后，运行：

```bash
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4

python src/run_model_suite.py \
  --suite configs/model_suite_24lang.json \
  --resume
```

预期行为：

- 已完成且配置一致的模型可以复用结果；
- XGLM 因 dtype 配置变化而重新运行；
- 任意非有限值或无效几何会立即停止并给出可定位的错误；
- 所有模型完成后，生成严格的控制感知跨模型 verdict。
