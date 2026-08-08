# behavior_v2 冻结实验协议

## 研究问题

V2 将结构问题与行为问题分成两层：

1. 在同一语义的 24 种语言表征中，Latin 文字语言是否形成更紧密的区域？
2. 英语相对其他 Latin 语言是否还具有额外的中心性或 hubness 优势？
3. 这种英语特异优势及 Latin 区域吸引是否与非 Latin 目标翻译中的 Latin 文字侵入、英语词汇泄露和翻译质量相关？

所有 hidden-state 指标只使用 `mean_pool`，协议版本为 `mean_pool_v1`。PCA 只用于二维展示，不能作为确认性证据。

## 数据与任务

- 数据：FLORES+ `devtest`，与 V1 使用的 `dev` 分开。
- 每种语言冻结抽取 804 个相同语义 ID：4 个固定演示，800 个正式评估。
- 每个评估 ID 生成 5 个任务，目标固定为 `zh/ar/hi/ru/ja`。
- 源语言从 `zh/ar/hi/es/ru/sw/tr/ja` 中确定性轮换，并排除与目标相同的语言。
- 每个模型共 4,000 条生成；三个模型使用完全相同的任务文件哈希。
- 英语既不是源语言也不是目标语言，因此输出中的英语片段属于非必要泄露候选。

## 结构指标

每个冻结分析层都输出：

- 全 24 语言的成对余弦相似度、余弦距离及置信区间；
- 语言平均中心性及 K=`3/5/10` 的 reverse-kNN occurrence；
- Latin 内部、非 Latin 内部、Latin—非 Latin 之间相似度；
- `latin_concentration = within_latin - between_script`；
- 英语相对其他 Latin 语言的中心性，以及 K=`3/5/10` 各自的 hubness excess；
- raw cosine 与 local-scaled cosine 下的任务级预测量。

主行为预测量是 `english_specific_advantage`：源语言到英语的相似度减去源语言到其他 Latin 语言的平均相似度。`latin_attraction` 则是源语言到其他 Latin 语言的平均相似度减去到非 Latin 语言的平均相似度。

## 行为指标

- `latin_script_fraction`：输出字母中 Latin 字符比例；
- `has_latin_span`：是否存在至少 3 个连续 Latin 单词；
- `english_lexical_leakage`：上述连续片段中是否包含至少一个高特异英语标记，或至少两个普通英语功能词；
- `semantic_quality_chrfpp`：SacreBLEU chrF++；
- 空输出、4-gram 重复率和达到 256-token 上限的比例作为质量门控。

V2 不再使用整段 fastText LID 来定义英语泄露。检测器必须经过人工盲审：标注表不展示自动标签，抽取全部自动阳性并按目标语言随机抽取自动阴性；验证时对阴性分层样本做逆概率加权。正式分析只接受 `passed=true` 且达到 precision、recall、FPR 门槛的报告。

## 生成协议

- plain-text prompt，`add_special_tokens=false`；
- greedy decoding，无采样、无 beam search；
- 模型自然终止；256 token 只是安全上限；
- 终止控制 token 不写入文本，其他 tokenizer 控制 token 全部屏蔽；
- 达到 token 上限的比例不得超过 1%；
- 不做激活干预。

## 确认性统计

冻结主检验：主层上，`english_specific_advantage` 是否正向预测 `english_lexical_leakage`。

使用联合模型，同时放入英语特异优势、Latin 吸引、源—目标对齐、源/目标长度以及源/目标语言固定效应；标准误按语义 ID 聚类。二元泄露使用 binomial logit，连续指标使用 OLS。local-scaled 版本是密度稳健性检验，其余层、指标和预测量统一作为次要检验并做 BH 校正。

任何支持都只能写成观察性关联，不能写成激活层面的因果机制。
