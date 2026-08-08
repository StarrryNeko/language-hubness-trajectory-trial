# behavior_v1：行为关联实验协议（无激活干预）

## 研究问题与结论边界

本协议检验冻结的表示几何是否能够预测未参与几何分析的翻译行为。它是观察性关联实验，不读取、替换或编辑生成过程中的 activation，不支持因果措辞。

唯一的确认性检验是：在模型预先冻结的主层上，`english_minus_target_cosine` 是否正向预测非英语到非英语翻译中的 `unnecessary_english_leakage`。双侧 `p < 0.05` 且系数方向为正视为原始余弦支持；局部缩放版本同时成立时，才视为通过密度鲁棒性检查。

## 冻结设计

- 模型：XGLM-1.7B、Mistral-7B-v0.1、Aya-23-8B。
- 主层：XGLM 12、Mistral 30、Aya 31；相邻层只做层敏感性检查。
- 样本：从 FLORES+ `dev` 另抽 208 个语义 ID，使用 `configs/behavior_exclusions_seed1_seed2.json` 排除两个几何 seed 使用过的 363 个不重复 ID；该文件同时冻结两个来源 manifest 的哈希。
- 208 个行为语义 ID 的预期 SHA-256 已写入基础配置；验证器不只检查“不重叠”，也检查抽样结果是否与预注册哈希完全一致。
- 其中 8 个 ID 只用于固定 few-shot demonstrations，200 个 ID只用于评估。
- 评估语言：中文、阿拉伯语、印地语、西班牙语、俄语、斯瓦希里语、土耳其语、日语。
- 每个评估 ID 含 8 个非英→非英、8 个英→非英、8 个非英→英任务，共 4,800 个生成任务。非英→非英目标语随语义 ID 轮换，以避免固定源—目标配对混杂。
- 解码：greedy，`do_sample=false`、`num_beams=1`、最多 256 个新 token。

## 指标与统计

主要行为指标为目标语言保持、非必要英语泄漏和 sentence-level chrF++。参考译文门禁使用 fastText top-1 标签正确率且至少达到 0.95；参考文本置信度不参与该门禁。输出侧目标语言保持和整段英语判定必须同时满足冻结的 fastText 置信度阈值；英语片段判定继续使用结果揭示前冻结的 `english_span_threshold=0.15`。正式结果还要求 SacreBLEU chrF++，并要求空输出率不超过 1%。脚本启发式 LID 和字符 F-score 只允许本地 smoke test。

fastText 置信度阈值只能在独立校准集上扫描。校准集固定使用两个几何 seed 的 363 个语义 ID，必须通过 manifest 验证来源文件哈希、排除清单哈希、任务文件哈希，并证明与 208 个正式行为语义 ID 的交集为零。正式行为参考集只允许 top-1 标签审计，禁止阈值搜索。

关联模型仅使用非英→非英任务，加入源语言和目标语言固定效应、源句和目标句 token 数控制，并按语义 ID 计算 cluster-robust 标准误。确认性检验不参与多重校正；其余层、局部缩放、hubness、范数、质心距离、PC1 和局部密度检验统一在单模型内做 Benjamini–Hochberg 校正。

三模型比较预先把 XGLM 和 Aya 定义为几何阳性模型，把 Mistral 定义为负对照。只有两个阳性模型均支持确认性关联、负对照不支持、且三个模型均通过正式验证时，才报告跨模型复现；若两个阳性模型的局部缩放检验也都成立，状态明确写为 `REPLICATED_RAW_AND_LOCAL_SCALED_WITH_NEGATIVE_CONTROL_SPECIFICITY`，否则标记为 raw-only、density-sensitive 复现。

## 完整性门禁

验证器要求三个模型使用完全相同的任务哈希；行为样本不得与两个几何 seed 重叠；hidden-state 抽取、生成必须匹配同一 checkpoint SHA-256；生成清单必须明确包含 `activation_intervention=false`。任何失败都会把结果标为无效。

## 运行

服务器准备好本地模型、两个 seed 的 `dataset_manifest.json` 以及 `/root/autodl-tmp/lid/lid.176.bin` 后：

```bash
python src/run_behavior_suite.py \
  --suite configs/model_suite_behavior_v1.json \
  --resume
```

若 hidden states、checkpoint audit 和 generations 已存在，只重新做离线评估：

```bash
python src/run_behavior_suite.py \
  --suite configs/model_suite_behavior_v1.json \
  --resume --skip-extraction --skip-generation
```

输出位于各模型的 `outputs_behavior_v1/<model>/behavior_v1/`，跨模型汇总位于 `outputs_behavior_v1/model_comparison_three/`。
