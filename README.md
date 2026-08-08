# Language Hubness Trajectory

## 行为实验版本入口

- V1（冻结归档）：`src/behavior_v1/`、`configs/behavior_v1/`、`docs/behavior_v1/`
- V2（script-aware）：`src/behavior_v2/`、`configs/behavior_v2/`、`docs/behavior_v2/`
- V2 当前规范：`docs/behavior_v2/design_framework.md`（geometry-only 主实验）
- 结构主实验实现：`src/structure_v2/`、`docs/structure_v2/`
- 第三阶段行为关联：`src/behavior_association_v3/`、`configs/behavior_association_v3/`、`docs/behavior_association_v3/`
- 项目结构：`docs/project_structure.md`
- V2 服务器流程：`docs/behavior_v2/server_runbook.md`

V1 与 V2 使用独立输出根目录；顶层行为脚本仅为兼容入口，不包含重复实现。

## 一周冲刺正式入口（2026-08）

五个必跑模型：

```bash
python src/run_formal_suite.py \
  --suite configs/model_suite_week1_required_random200.json \
  --resume
```

Moonlight通过hidden-state审计后：

```bash
python src/run_formal_suite.py \
  --suite configs/model_suite_week1_with_moonlight_random200.json \
  --resume
```

新增离线模块包括竞争性余弦（hard margin、pairwise win rate、英语排名、完整source→candidate矩阵）以及范数、leave-one-out语义质心、全局质心、PC1、局部密度和相邻层轨迹。完整的离线权重下载、上传、Blackwell/H800审计和运行顺序见`docs/服务器执行指南_week1.md`。

本项目研究多语言因果语言模型中，英语是否在层间表示空间里成为 hub。当前正式协议只使用：

- `mean_pool`：对原句 tokenizer 文本 token 的 hidden state 求均值；可选 BOS 不进入均值。代码只允许这一种句向量表示。

模型、tokenizer 与 FLORES+ 的 Hugging Face 缓存统一配置在
`/root/autodl-tmp/huggingface`，避免占用 AutoDL 系统盘。修改
`huggingface_cache_dir` 即可切换到其他机器的缓存目录。

## 关键实验约束

1. 每个语义 ID 必须包含至少 20 种语言；默认配置使用 24 种语言。
2. 任何相似度、近邻排序和局部密度校正都只在同一语义 ID 的平行译句中进行。
3. 不把不同语义的句子作为候选或负例；bootstrap 的抽样单位是语义 ID。
4. “更接近英语”不等于 hubness。英语 hubness 至少由四类互补的操作化信号共同支持（它们并非统计独立）：
   - reverse-kNN `k_occurrence_excess`；
   - 平均图中心性 `centrality_advantage`；
   - 中心性排名 `rank_percentile_advantage`；
   - 成为组内 medoid 的频率 `medoid_rate_excess`。
5. 正式结论要求主证据、来源广度与局部密度校正在相同连续层成立，并通过不同 k 和多模型复现。

## 默认语言与模型

`configs/base_24lang_same_semantics.json` 配置 24 种语言，覆盖 Latin、Han、Arabic、Devanagari、Cyrillic、Japanese、Hangul、Thai、Greek、Tamil、Telugu 和 Bengali 等文字系统。语言集合取自 XGLM 明确列出的训练语言，以减少把“模型从未训练该语言”误当成 hubness 的风险。

首轮三模型套件：

- Qwen2.5-1.5B
- BLOOM-1.7B
- XGLM-1.7B

这三者参数量接近，且来自不同模型家族，适合先做快速结构复现；它们不能替代后续更强模型的确认实验。

## 单模型运行

首次准备 FLORES+ 前，需要在 Hugging Face 接受数据集条款并通过 `huggingface-cli login` 登录；也可以把已审核的 24 语言 JSONL 配成 `dataset.source=local_jsonl`，完全离线运行。

```bash
python src/run_pilot.py --config configs/qwen25_1_5b_mvp.json
```

只重算指标和图：

```bash
python src/run_pilot.py --config configs/qwen25_1_5b_mvp.json --skip-prepare --skip-extract
```

检查向量对应的原句和 token：

```bash
python src/inspect_hidden_states.py \
  --config configs/qwen25_1_5b_mvp.json \
  --rows 0,100,200 --layers 0,14,28 --show-token-sequence
```

## 多模型一键运行

```bash
python src/run_model_suite.py --suite configs/model_suite_24lang.json
```

## paper_v1 离线正式分析

模型已经完成 `mean_pool_v1` hidden-state 提取时，不要重新加载模型权重。直接运行：

```bash
python src/run_paper_analysis.py \
  --config configs/qwen25_1_5b_mvp.json \
  --resume
```

该入口只读取 `output_dir/hidden/metadata.csv` 与
`output_dir/hidden/sentence_layer_mean_pool.npy`，并把确认性结果写入
`output_dir/paper_v1/`。它依次计算 AlignmentGain、24 语言目标轮换、
max-statistic 标签置换、raw/local-scaled breadth、交叉拟合公共方向控制、
随机语义子集、language structure 和 semantic-ID split language probe。

方法与结论边界见 `docs/paper_v1_protocol.md`。旧 `metrics/` 和 `validation/`
不会被覆盖。

如果一个 suite 中只有部分模型已经完成兼容的 hidden-state 提取，可只对这些
模型做离线复算；缺失模型会被列出，但不会被自动启动：

```bash
python src/run_paper_suite.py \
  --suite configs/model_suite_24lang.json \
  --resume
```

正式随机样本的服务器执行顺序、断点语义和“旧结果复算/正式结果”边界见
`docs/服务器执行指南_paper_v1.md`。冻结后的三模型正式入口为：

```bash
python src/run_formal_suite.py \
  --suite configs/model_suite_paper_v1_random200.json \
  --resume
```

首个模型准备一次 FLORES 数据；后续模型复用经过哈希核对的完全相同数据。最后自动生成归一化层深的跨模型比较。

若已经完成了 Qwen 单模型试跑，可安全续跑；只有配置快照完全一致且必需输出齐全的模型才会被跳过：

```bash
python src/run_model_suite.py --suite configs/model_suite_24lang.json --resume
```

若先做快速 smoke test，可暂时跳过 k sweep：

```bash
python src/run_model_suite.py \
  --suite configs/model_suite_24lang.json \
  --skip-k-sweep
```

## 主要输出

```text
outputs/<experiment>/
  data/dataset_manifest.json
  hidden/sentence_layer_mean_pool.npy
  metrics/metrics_manifest.json
  metrics/within_semantic_pair_similarity.csv
  metrics/within_semantic_knn.csv
  metrics/hubness_by_language.csv
  metrics/hubness_global.csv
  metrics/english_hubness_evidence.csv
  metrics/english_source_group_attraction.csv
  metrics/english_hubness_breadth.csv
  validation/validation_summary.md
```

跨模型输出位于 `outputs/model_comparison_24lang/`。

## 解释边界

- 同语义设计控制了内容差异，但无法自动控制翻译质量、句长、token 数、训练语料比例和文字系统效应。
- 英语的平均 cosine 更高只能算中心接近证据；只有 reverse-kNN/中心性排名/medoid 等反复被选中证据才属于 hubness。
- 局部密度校正后消失的英语优势，更可能来自各向异性或密度差异。
- 只在 Qwen 上成立的轨迹不得写成通用多语言模型规律。
- 新协议不再运行旧版跨语义检索、语言 neighborhood purity 和 re-separation 指标；历史输出仅供追溯。
