# Language Hubness Trajectory

本项目研究多语言因果语言模型中，英语是否在层间表示空间里成为 hub。当前正式协议只使用：

- `mean_pool`：主表示；对原句 tokenizer 文本 token 的 hidden state 求均值，不包含额外 BOS/EOS。
- `sentinel_eos`：验证表示；在每个句子后追加该模型原生 EOS，读取 EOS 位置 hidden state。

旧版的 `last_token`、`last_content_token`、`shared_sentinel` 和 `content_mean_pool` 不再参与新实验。

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
5. 结论还需通过 sentinel-EOS、局部密度校正、不同 k、来源语系/文字系统覆盖和多模型复现。

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

检查向量对应的原句、token 和 EOS：

```bash
python src/inspect_hidden_states.py \
  --config configs/qwen25_1_5b_mvp.json \
  --rows 0,100,200 --layers 0,14,28 --show-token-sequence
```

## 多模型一键运行

```bash
python src/run_model_suite.py --suite configs/model_suite_24lang.json
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
  hidden/sentence_layer_sentinel_eos.npy
  metrics/metrics_manifest.json
  metrics/within_semantic_pair_similarity.csv
  metrics/within_semantic_knn.csv
  metrics/hubness_by_language.csv
  metrics/hubness_global.csv
  metrics/english_hubness_evidence.csv
  metrics/english_source_group_attraction.csv
  metrics/english_hubness_breadth.csv
  metrics/representation_agreement.csv
  validation/validation_summary.md
```

跨模型输出位于 `outputs/model_comparison_24lang/`。

## 解释边界

- 同语义设计控制了内容差异，但无法自动控制翻译质量、句长、token 数、训练语料比例和文字系统效应。
- 英语的平均 cosine 更高只能算中心接近证据；只有 reverse-kNN/中心性排名/medoid 等反复被选中证据才属于 hubness。
- 局部密度校正后消失的英语优势，更可能来自各向异性或密度差异。
- 只在 Qwen 上成立的轨迹不得写成通用多语言模型规律。
- 新协议不再运行旧版跨语义检索、语言 neighborhood purity 和 re-separation 指标；历史输出仅供追溯。
