# P0 数值验证与联合判定加固实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 让无效隐状态、相似度与 kNN 图在产生正式结果前立即失败，并用同层四指标联合证据和严格的单模型/跨模型状态替代宽松的逐指标判定。

**架构：** 新增两个无模型依赖的轻量模块：`numerical_validation.py` 负责数组、相似度与质量守恒验证，`evidence_rules.py` 负责证据网格、连续区间和模型状态判定。提取、指标、验证、k 扫描与跨模型比较只调用这些共享规则，避免多套判定口径。

**技术栈：** Python 3.10、NumPy、pandas、PyTorch、Transformers、unittest、Matplotlib。

**当前实施状态（2026-07-21）：** 代码、配置与测试用例已完成；当前 Windows 工作区没有可用的 Python 解释器，因此运行态单元测试、合成 smoke test 与 `compileall` 保留为 AutoDL 上的交付前验证项。

## 全局约束

- 所有新功能先写失败测试，再写最小实现。
- NaN/Inf、零范数、缺层、缺指标和重复 `(layer, metric)` 不得被静默过滤。
- 保留现有入口、CSV 文件和 `--resume` 使用方式。
- XGLM 推理与保存均固定为 FP32；Qwen/BLOOM 配置不改。
- P0 不扩展到 tokenization、样本量和完整图表重构。

---

### 任务 1：建立共享数值验证边界

**文件：**

- 新建：`src/numerical_validation.py`
- 新建：`tests/test_numerical_validation.py`

- [ ] 写入测试，覆盖非有限数组、零范数行、非方阵/非对称/非有限相似度矩阵以及合法输入。
- [ ] 运行 `python -m unittest discover -s tests -p "test_numerical_validation.py" -v`，确认因模块缺失而失败。
- [ ] 实现以下接口：

```python
def require_finite(values, context): ...
def require_nonzero_row_norms(values, context, minimum=1e-12): ...
def validate_representation_array(values, expected_rows, context): ...
def validate_similarity_matrix(values, context, atol=1e-6): ...
```

每个异常必须使用 `ValueError`，并在消息中包含 `context` 与具体失败类型。

- [ ] 重跑该测试文件，确认通过。

### 任务 2：加固提取阶段并强制 XGLM FP32

**文件：**

- 修改：`src/extract_hidden.py`
- 修改：`configs/xglm_1b7_24lang.json`
- 新建：`tests/test_extraction_validation.py`

- [ ] 测试保存前/后表示验证会拒绝 NaN、Inf 和零范数，并检查错误上下文包含模型、行、语义 ID、语言、层和表示名。
- [ ] 运行新测试并确认失败。
- [ ] 在每个句子/层上验证完整 hidden state、mean-pool、sentinel-EOS；转成 `storage_dtype` 后再次验证。先积累全部表示并验证成功，再写 `.npy`、metadata 与 manifest。
- [ ] 将 XGLM 配置显式设置为：

```json
"dtype": "float32",
"storage_dtype": "float32"
```

- [ ] 测试 XGLM 解析后的两个 dtype 均为 `float32`，重跑新测试。

### 任务 3：加固指标计算、bootstrap 与 kNN 质量守恒

**文件：**

- 修改：`src/compute_metrics.py`
- 修改：`tests/test_same_semantics_metrics.py`

- [ ] 增加失败测试：bootstrap 输入 NaN/Inf；local scaling 输入非法矩阵；k 非法；fractional top-k 每行权重不为 k；总 occurrence 不为 `n*k`；medoid 权重不为 1。
- [ ] 运行 `python -m unittest discover -s tests -p "test_same_semantics_metrics.py" -v`，确认新增断言失败。
- [ ] `bootstrap_mean_ci` 改为先要求全部输入有限且非空，不再过滤观测值。
- [ ] `locally_scaled_similarity` 在输入和输出两侧验证矩阵。
- [ ] `group_statistics` 验证 `1 <= k < n`、每行 top-k 权重和、总 occurrence、所有输出以及 medoid 质量。
- [ ] `main()` 加载每个表示数组后验证三维结构、metadata 行数与全体有限值；每个语义组每层归一化前验证有限值和非零范数。
- [ ] 重跑指标测试并确认通过。

### 任务 4：建立同层四指标联合证据规则

**文件：**

- 新建：`src/evidence_rules.py`
- 新建：`tests/test_evidence_rules.py`
- 修改：`src/sweep_k.py`

- [ ] 写测试覆盖：四指标同层为正；四指标分别位于不同层不能通过；重复记录失败；缺指标/缺层失败；真实层号缺口中断连续区间。
- [ ] 运行新测试并确认失败。
- [ ] 实现常量和接口：

```python
REQUIRED_EVIDENCE_METRICS = (
    "k_occurrence_excess",
    "centrality_advantage",
    "rank_percentile_advantage",
    "medoid_rate_excess",
)
def validate_evidence_grid(frame, expected_layers=None): ...
def joint_positive_layers(frame): ...
def max_consecutive_layers(layers): ...
```

- [ ] `sweep_k.py` 改为比较每个 k 的联合正证据连续区间，而不是四项指标各自是否出现连续区间。
- [ ] 重跑证据规则测试并确认通过。

### 任务 5：收紧单模型验证状态

**文件：**

- 修改：`src/run_validations.py`
- 新建：`tests/test_model_status.py`

- [ ] 写测试构造 primary、EOS、density 和来源广度层集合，覆盖 `INVALID`、`NOT_SUPPORTED`、`REPRESENTATION_SENSITIVE`、`ROBUST`。
- [ ] 运行新测试并确认失败。
- [ ] 在 `evidence_rules.py` 增加纯函数：

```python
def classify_model_status(primary_layers, breadth_layers,
                          eos_layers, density_layers, min_run): ...
```

规则为：primary 与 breadth 的同层交集达标才支持主结论；EOS 与 density 也分别达标才是 `ROBUST`。

- [ ] `run_validations.py` 使用共享联合规则，输出每个控制条件的联合层、最长连续区间、`min_run` 和顶层 `model_status`；保留现有编号报告与 `overall_status` 兼容字段。
- [ ] 重跑模型状态测试并确认通过。

### 任务 6：严格跨模型比较与无效模型隔离

**文件：**

- 修改：`src/compare_models.py`
- 新建：`tests/test_compare_models.py`

- [ ] 写临时目录测试：缺 validation、缺层、重复指标、NaN 模型都进入 verdict 的 `INVALID`，不进入 AUC；只有至少两个 `ROBUST` 模型时为 `REPLICATED`。
- [ ] 运行新测试并确认失败。
- [ ] 抽出可测试的 `compare_suite(config_paths, output)`；逐模型捕获完整性错误并记录原因，合法模型才进入 AUC 与轨迹图。
- [ ] 轨迹图对每个模型/指标显式 `ax.plot`；遇缺失值保持断线；无效模型不绘制。
- [ ] verdict 至少输出 `model_statuses`、`valid_model_count`、`robust_models`、`conditional_models`、`replication_status` 和严格规则说明。
- [ ] 重跑跨模型测试并确认通过。

### 任务 7：回归验证与交付检查

**文件：**

- 可能修改：`tests/synthetic_pipeline_smoke.py`（仅适配严格输出字段）
- 修改：`docs/superpowers/plans/2026-07-21-p0-validation-hardening.md`（勾选完成项）

- [ ] 运行全部单元测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

- [ ] 运行无需模型下载的合成流水线：

```bash
python tests/synthetic_pipeline_smoke.py --output /root/autodl-tmp/language_hubness_p0_smoke
```

- [ ] 运行编译检查：

```bash
python -m compileall config src scripts tests
```

- [ ] 检查 `git diff --check`、配置差异与输出 schema；确认没有修改用户运行产物。
- [ ] 更新计划勾选状态并提供 AutoDL 直接运行命令与预期重算范围。
