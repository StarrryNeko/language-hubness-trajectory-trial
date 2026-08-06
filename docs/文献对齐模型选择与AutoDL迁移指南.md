# 文献对齐模型选择与 AutoDL 迁移指南

## 1. 决策摘要

本轮新增四个模型：

```text
bigscience/bloom-3b
facebook/xglm-2.9B
mistralai/Mistral-7B-v0.1
CohereLabs/aya-23-8B
```

这组模型不是简单扩充数量，而是补齐三类对照：

1. **同论文、同模型族的尺度对照**：BLOOM-1.7B→3B，XGLM-1.7B→2.9B。
2. **英语中心模型的文献锚点**：Mistral-7B-v0.1。
3. **均衡多语训练的反事实对照**：Aya-23-8B。

前三个模型是 base checkpoint，可进入正文统一比较；Aya-23-8B 经历了多语指令微调，应作为独立扩展组报告，不能把其差异全部归因于预训练语言比例。

## 2. 为什么选择这些模型

### 2.1 bigscience/bloom-3b

Kojima 等人在 NAACL 2024 的语言特异神经元研究中比较了 BLOOM-560M、BLOOM-1.7B 和 BLOOM-3B。项目已经完成 `bigscience/bloom-1b7`，加入 `bigscience/bloom-3b` 后，可以在相同架构、训练语料和 tokenizer 家族内观察规模增加是否改变英语 hubness 的层间强度、峰值位置、竞争性近邻份额和向量范数。

它主要回答：**英语 hubness 是否只是小模型容量不足造成的几何现象，还是在同族扩展后仍然稳定？**

### 2.2 facebook/xglm-2.9B

同一篇 NAACL 2024 论文同时分析了 XGLM-564M、1.7B 和 2.9B。项目已有 `facebook/xglm-1.7B`，新增 2.9B 能形成第二组独立的同族尺度实验。

如果 BLOOM 与 XGLM 两个家族都呈现相似的规模轨迹，结论就不容易被解释为某一架构或 tokenizer 的偶然结果；如果两组方向不同，也能成为训练语料构成和架构差异的研究线索。

### 2.3 mistralai/Mistral-7B-v0.1

Mistral-7B-v0.1 被语言特异神经元研究和 Multilingual Workflow 研究用作基础模型。它不依赖 Meta Llama 的访问许可，规模适中，能够作为英语占主导训练模型的文献锚点。

它主要回答：**在更强、较大且主要由英语语料塑造的 base 模型中，英语中心性是否比均衡多语模型更明显？**

Mistral 与 Llama-2 并非同一 checkpoint，因此不能宣称直接复现 Llama-2 结果；应表述为使用先行研究采用的另一个公开模型进行概念复核。

### 2.4 CohereLabs/aya-23-8B

2025 年针对 Aya-23 内部语言表示的研究将 Aya-23-8B 与 Llama-3.1-8B、Chinese-LLaMA-2-7B 对比，发现均衡多语训练模型可能同时激活多个类型学相关语言，而非始终依赖单一英语枢纽。

它因此构成关键反事实：**如果英语 hubness 主要来自英语占主导的训练分布，那么均衡多语模型中该现象应减弱、变形或由多个语言竞争中心取代。**

但是 Aya-23-8B 是多语指令模型，和正文 base 模型存在 alignment confound。正式报告中应：

- 将 Aya 单列为“指令对齐扩展”；
- 不直接把 Aya 与 Mistral 的差异解释为语料平衡的纯因果效应；
- 后续资源允许时，再增加同家族 base/instruct 或其他可比配对。

## 3. 与已有模型的整体比较结构

| 分组 | 模型 | 主要作用 |
|---|---|---|
| 文献原 checkpoint | BLOOM-1.7B、XGLM-1.7B | 已有结果；与 Kojima et al. 使用的型号一致 |
| 同族尺度 | BLOOM-3B、XGLM-2.9B | 控制架构和模型族，研究规模效应 |
| 现代小模型泛化 | Qwen2.5-1.5B、Llama-3.2-3B | 检验旧模型结论能否推广到新一代模型 |
| 英语中心文献锚点 | Mistral-7B-v0.1 | 与多篇内部多语机制研究建立直接联系 |
| 均衡多语扩展 | Aya-23-8B | 检验均衡多语训练下英语中心性是否减弱 |

正文的主要统计检验应优先使用 base checkpoint。Aya 的结果放在正文末尾的扩展实验或附录，并显式标记训练阶段。

## 4. 可复现性要求

模型名称相同不代表字节级权重完全相同。每次下载必须记录：

- Hugging Face 仓库 ID；
- 不可变 revision/commit SHA；
- base、instruct 或 chat 阶段；
- tokenizer revision；
- 推理 dtype；
- 是否量化；
- `trust_remote_code` 状态。

本项目的下载脚本会在每个目录写入 `.lht_model_manifest.json`，并在根目录生成 `portable_models_manifest.json`。上传时应保留模型目录内的 manifest；正式实验使用 BF16、非量化权重。

## 5. 本地目录与服务器目录

本地便携权重根目录：

```text
D:\code\拔尖科研计划\language-hubness-trajectory\langhub_models
```

对应文件夹：

```text
bigscience__bloom-3b
facebook__xglm-2.9B
mistralai__Mistral-7B-v0.1
CohereLabs__aya-23-8B
```

服务器目标：

```text
/root/autodl-tmp/models/
```

最终必须形成：

```text
/root/autodl-tmp/models/bigscience__bloom-3b/config.json
```

不能多套一层 `langhub_models`。

## 6. 如何提高向 AutoDL 的传输速度

### 6.1 首选：服务器直接从 Hugging Face 下载

如果 AutoDL 到 Hugging Face 的线路稳定，服务器直接下载通常最快，因为不经过本地家庭上行带宽。将本地目录作为断网备份即可。Aya 需要服务器所用 Hugging Face 账号也已接受许可。

### 6.2 次选：对象存储或公网网盘中转

AutoDL 官方文档优先推荐公网网盘或 OSS。先从 Windows 上传到同区域对象存储，再由 AutoDL 内部下载，通常比家庭网络直传 SSH 更稳定，并方便断点续传和多实例复用。四个模型约 45–50GB，AutoDL 免费网盘 20GB 不足，应选容量足够的 OSS、阿里云盘或其他受支持网盘。

### 6.3 直传时使用单个 tar 流或分模型归档

AutoDL 官方指出，SCP 直接传输包含大量小文件的目录会很慢。每个模型可打成一个**不压缩 tar**，减少握手和元数据开销：

```powershell
tar -cf bloom-3b.tar -C "D:\code\拔尖科研计划\language-hubness-trajectory\langhub_models" bigscience__bloom-3b
```

上传后：

```bash
tar -xf bloom-3b.tar -C /root/autodl-tmp/models
```

不要使用 zip、7z 高压缩或 `tar.gz` 期待明显缩小体积：safetensors 中的浮点权重通常几乎不可压缩，只会额外消耗 CPU 和时间。不压缩 tar 的价值是合并小文件，而不是缩小权重。

### 6.4 需要断点续传时

优先使用支持断点续传的对象存储客户端；若继续走 SSH，可使用 WinSCP/FileZilla 的续传功能。不要在失败后删除服务器上已经传完的模型目录。

### 6.5 分模型上传与校验

一次只传一个模型，完成后立即校验，再传下一个。服务器执行：

```bash
cd /root/autodl-tmp/language-hubness-trajectory
conda activate langhub
export LHT_MODEL_ROOT=/root/autodl-tmp/models

python src/download_model_weights.py \
  --suite configs/model_suite_literature_additions_random200.json \
  --verify-root "$LHT_MODEL_ROOT"
```

四个模型全部存在时，命令应输出 `PORTABLE_MODEL_ROOT_VERIFIED`。

## 7. 文献依据

- Kojima et al. (2024), *On the Multilingual Ability of Decoder-based Pre-trained Language Models*: https://aclanthology.org/2024.naacl-long.384.pdf
- Tang et al. (2024), *Language-Specific Neurons*: https://aclanthology.org/2024.acl-long.309.pdf
- Zhao et al. (2024), *How do Large Language Models Handle Multilingualism?*: https://proceedings.neurips.cc/paper_files/paper/2024/file/1bd359b32ab8b2a6bbafa1ed2856cf40-Paper-Conference.pdf
- Trinley et al. (2025), *What Language(s) Does Aya-23 Think In?*: https://aclanthology.org/2025.globalnlp-1.18.pdf
- AutoDL 上传数据：https://www.autodl.com/docs/scp/
- AutoDL 公网网盘与对象存储：https://www.autodl.com/docs/netdisk/
