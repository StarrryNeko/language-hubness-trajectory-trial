# 同语义 24 语言、多模型 Hubness 实验方案

## 研究问题

在完全相同语义的平行译句集合中，英语表示是否比其他语言更频繁地成为多语言表示图的中心或 reverse-kNN hub？这一现象是否跨层、跨来源语言群体、跨表示方法和跨模型家族稳定？

## 实验单元

一个 FLORES 语义 ID 是一个独立实验单元，包含 24 种语言各一个译句。该组形成 24×24 cosine 矩阵。矩阵计算、top-k、local scaling 和排名都不得访问其他语义 ID。

## 主要因变量

对每个模型、表示、层、语义 ID：

1. `k_occurrence`：某语言被其余语言放入 top-k 的次数；平衡零假设为 `k`。
2. `centrality`：该语言与同组其他 23 种语言的平均相似度。
3. `centrality_rank_percentile`：中心性在 24 种语言中的百分位排名。
4. `medoid_rate`：该语言是否为该语义组的中心语言；并列时平分权重。
5. `k_occurrence_skewness` 与 `gini`：整个语言图是否存在 hub 集中，而不是所有语言均匀被选中。

英语证据使用语义 ID 配对差值并 bootstrap：英语 occurrence−k、英语中心性−其他语言均值、英语排名−其他语言均值、英语 medoid 权重−1/24。

## 预注册式判定

英语 hubness 的强结论建议同时要求：

- 四类英语证据在连续至少 3 层的 95% CI 下界大于 0；
- 不是只由一两个来源语言驱动，并覆盖多个语系与非 Latin 文字系统；
- `mean_pool` 与 `sentinel_eos` 方向一致；
- cosine 与 local-scaled cosine 方向一致；
- k=1/3/5/10 的结论一致；
- 至少两个不同模型家族复现，第三个模型用于判断边界条件。

若只满足其中一部分，应报告“局部/条件性英语 hubness”，而不是“英语是普遍 hub”。

## 运行顺序

1. 先在 Qwen2.5-1.5B 上用 24 语言×100 语义 ID 完成 smoke/pilot。
2. 查看 `validation/validation_summary.md`，确认无截断、EOS 一致、候选范围严格同语义。
3. 跑 k sweep；若结论敏感，先解释敏感性，不急于加样本。
4. 用完全相同数据跑 BLOOM-1.7B 与 XGLM-1.7B。
5. 只有三模型流程稳定后，把语义 ID 增加到 200–500，并加入更强的现代模型。

## 必要控制与后续增强

- 句长与 token 数：对英语 hub 指标做 token-length 差异回归或分层匹配。
- 文字系统：分别报告 Latin→English 与非-Latin→English 的选择率。
- 语系：以来源语言为单位做 leave-one-family-out，避免 24 种语言被当作 24 个独立语系。
- 翻译质量：抽查或用外部质量指标标记低质量语义组，再做剔除敏感性分析。
- 训练语料：若能取得模型语言配比，把语言暴露量作为解释变量，而不是把所有中心性都归因于“英语中介语”。
- 统计层级：正式论文建议使用 semantic ID 与 source language 的交叉随机效应或双向 cluster bootstrap。
- 因果验证：观察性 hubness 稳定后，再做英语方向移除、语言质心去偏或 activation intervention；不应直接把轨迹称为机制。
