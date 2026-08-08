# 第三阶段：behavior_association_v3

本阶段与 `structure_v2` 严格拆分。`structure_v2` 只验证804个 FLORES+ devtest
同语义 ID 的全语言 mean-pool 几何；本阶段只检验冻结几何是否预测生成行为，不进行激活干预，也不支持因果措辞。

## 冻结任务

- 4个语义 ID：few-shot demonstrations；
- 80个语义 ID：协议校准，不进入正式关联；
- 720个语义 ID：正式评估；
- 目标语固定为中文、阿拉伯语、印地语、俄语和日语，覆盖 Han、Arabic、Devanagari、Cyrillic 及 Han/Kana 混合文字；
- 每个正式 ID 对五种目标语各生成一次，共3,600条/模型；
- 英语既不是源语也不是目标语。

主预测量为：

```text
english_target_competition
= cosine(source, English) - cosine(source, intended target)
```

主结果为经过校准集盲审验证的 `english_lexical_leakage`。主层 raw cosine 是唯一确认性检验，local-scaled
版本是预指定密度稳健性检查。使用 binomial-logit GEE、语义 ID 聚类、源/目标语言固定效应及源/目标 token
长度控制。若正式英语泄漏事件少于30个，状态固定为
`PRIMARY_NOT_ESTIMABLE_DUE_TO_LOW_EVENT_COUNT`，不得临时修改检测器或更换主结果。

## 生成与停止

生成采用 greedy decoding。模型原生 EOS 记录为 `native_eos`；冻结的双换行或下一个语言标签记录为
`text_boundary`（单个前导换行不会产生空答案）；`max_new_tokens=192` 只是失控保护，达到它记录为
`token_ceiling`。输出不补齐到固定长度。

EOS 同时从 tokenizer、model config 和 generation config 收集并显式传给生成器。框架的
`forced_eos_token_id` 被关闭，因此安全上限处由框架追加的 EOS 不会伪装成 `native_eos`。每条记录保存
实际 EOS token ID 和位置，manifest 保存完整 EOS ID 集合及三类结束原因计数。

生成器按 prompt 长度排序，并在预设范围内根据显存峰值自动增加 batch；OOM 时减半。96 GiB GPU
的进程上限为91 GiB，目标利用率为94%，每次实际 batch、OOM 和峰值显存均写入 manifest。

正式分析门禁：空输出率不超过1%、token-ceiling rate 不超过1%、平均4-gram重复率不超过2%、
SacreBLEU chrF++ 可用、校准集检测器达到 precision/recall/FPR 门槛。

## 单模型顺序

以下命令均在仓库根目录执行，首先只使用 XGLM：

```bash
python src/run_behavior_association_v3_single.py \
  --config configs/behavior_association_v3/xglm_1b7.json --stage structure

python src/run_behavior_association_v3_single.py \
  --config configs/behavior_association_v3/xglm_1b7.json --stage prepare

python src/run_behavior_association_v3_single.py \
  --config configs/behavior_association_v3/xglm_1b7.json --stage calibrate --resume
```

填写生成的 `lexical_detector_calibration_audit.csv` 后验证：

```bash
python src/annotate_behavior_association_v3.py validate \
  --config configs/behavior_association_v3/xglm_1b7.json \
  --annotations outputs_behavior_v2/xglm_1b7/behavior_association_v3/measurement/lexical_detector_calibration_audit.csv \
  --output outputs_behavior_v2/xglm_1b7/behavior_association_v3/measurement/lexical_detector_validation.json
```

之后才能执行：

```bash
python src/run_behavior_association_v3_single.py \
  --config configs/behavior_association_v3/xglm_1b7.json --stage formal-generate --resume

python src/run_behavior_association_v3_single.py \
  --config configs/behavior_association_v3/xglm_1b7.json --stage analyze
```

XGLM 的完整性与测量门禁通过后，才以相同顺序运行 Mistral 和 Aya 配置。
三模型均完成后运行：

```bash
python src/compare_behavior_association_v3_models.py \
  --suite configs/behavior_association_v3/suite.json
```

跨模型复现只要求两个冻结预期阳性模型的方向和区间复现。Mistral 是冻结对照模型，但不会把
“未显著”错误解释为“等效为零”。
