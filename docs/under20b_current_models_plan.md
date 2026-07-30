# 20B 以下新模型的跨家族复验方案

## 主线范围

主实验只纳入总参数量严格小于 20B、可通过 `AutoModelForCausalLM` 提取逐层 hidden state 的预训练基座模型。这样可以在常见单卡或有限多卡环境中完成实验，同时避免 22B、32B、70B 模型使算力成为项目主导因素。

旧模型只用于历史对照，不进入“当前新模型是否仍存在该现象”的主结论。

## S/M/L 分档

在 `<20B` 范围内重新定义：

| 规格 | 总参数量 |
| --- | ---: |
| S | `<7B` |
| M | `7B–<12B` |
| L | `12B–<20B` |

这里的参数量指未量化模型的总参数量，不使用磁盘文件大小、量化后显存占用或 MoE 激活参数量。

## 主实验模型

| 规格 | 模型 | 参数量 | 家族 | 发布代际 | 角色 |
| --- | --- | ---: | --- | --- | --- |
| S | EuroMoE-2.6B-A0.6B-2512 | 总2.6B、激活0.6B | EuroMoE | 2025-12 | 新模型 |
| M | Apertus-8B-2509 | 8.0B | Swiss AI | 2025-09 | 新模型 |
| M | EuroLLM-9B-2512 | 9.0B | EuroLLM | 2025-12 / 2026 报告 | 新模型 |
| L | Qwen3-14B-Base | 14.8B | Qwen | 2025 | 新模型 |

Qwen2.5-1.5B 仅放在带基线的辅助套件中，用于判断新旧代际差异。

S档采用2025年末发布的EuroMoE，而不是更早的Qwen3-4B。规格按总参数量2.6B划分，同时单独记录每次前向激活0.6B参数，避免MoE口径混淆。

当前没有把 Qwen3.5、Gemma 3 或 Ministral 3 直接加入主套件：这些新模型包含视觉编码器或使用 conditional-generation 架构，不能直接套用现有 `AutoModelForCausalLM` hidden-state 提取协议；Ministral 3 模型卡明确列出的语言也不足 20 种。后续若增加统一的多模态文本骨干提取适配器，可作为第二阶段扩展。

## 共同语言

四个主模型使用相同的24种明确支持语言：

`en, zh, de, ar, hi, es, fr, ru, ja, ko, tr, fi, el, bg, it, pt, cs, nl, pl, ro, uk, sv, da, hu`

配置继承使用整段替换语义，并有测试保证最终语言数恰好为24，不会与旧语言字典意外合并。

## 运行

正式主实验：

```bash
python src/run_model_suite.py \
  --suite configs/model_suite_under20b_current_24lang.json \
  --resume
```

需要与旧 Qwen2.5 基线同数据比较时：

```bash
python src/run_model_suite.py \
  --suite configs/model_suite_under20b_with_baseline_24lang.json \
  --resume
```

套件在启动模型下载和推理前会检查：

1. 所有模型参数量必须 `<20B`；
2. S/M/L 三档必须全部覆盖；
3. 配置参数量与 `size_class` 必须一致；
4. 所有模型使用相同数据文件。

## 结论口径

逐模型继续使用 `NOT_SUPPORTED`、`DENSITY_SENSITIVE`、`ROBUST` 和 `INVALID`。

跨模型层面分别报告：

- 每个规格至少一个模型是否支持；
- 每个规格内所有有效模型是否一致支持；
- 按模型家族汇总的复现情况；
- 按模型代际汇总的复现情况；
- 原始 cosine 与局部密度控制结论是否一致。

主结论应优先回答“现象是否在这些 `<20B` 新模型中跨家族出现”，规模趋势作为第二层分析。由于 S 和 L 当前各只有一个新模型，不能把组间差异直接解释为纯参数规模因果效应。
