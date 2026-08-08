# paper_v1 确认性分析协议

状态：方法与软件实现冻结候选版。新模型结果不得用于修改本文件中的主指标或通过门槛。

## 研究问题

1. 平行翻译是否形成高于 shuffled semantic baseline 的共享语义结构？
2. 英语是否是 24 种语言中具有选择校正后证据的特殊局部 hub？
3. 该证据能否通过来源广度、局部密度和公共方向控制？
4. hub 所处层的语言身份是否仍可由局部 purity 和线性 probe 读出？

生成行为关联与激活干预属于下一阶段协议；离线句向量结果不能单独支持功能或因果表述。

## 固定数据与表示

- 完整平行语义组；每组必须包含配置中的全部 24 种语言。
- `mean_pool` 是唯一确认性句向量；如模型支持则保留 BOS，但 BOS 不进入均值。
- layer 0 单独标记，不与 Transformer block 层混称。
- 所有模型复用同一 semantic-ID split 文件。
- hubness、alignment、language structure 使用彼此独立的候选范围。

## 固定主指标

- Alignment：paired cosine 减去 deranged semantic cosine；双向 Recall@1/5 为佐证。
- Hubness：k-occurrence excess、centrality advantage、rank-percentile advantage、medoid-rate excess。
- 英语特殊性：英语在 24 目标中的排名、相对最佳非英语目标的差值，以及以 `k_occurrence_excess` 为主统计量的 max-target/max-layer 标签置换检验。
- Breadth：来源语言的选择率 CI 下界超过平衡基线，并同时报告语言数、语系数、文字系统数和非拉丁文字语言数。
- 语言结构：semantic-ID-excluded neighborhood purity、centroid separation/within-language dispersion ratio、semantic-ID split 的 multinomial logistic probe macro-F1。

## 固定几何控制

确认性判定必须同时展示 raw cosine 与 local-scaled cosine。全局去均值和移除前 1/3/5 个主成分作为公共方向敏感性分析；中心和主成分只能由冻结 split 的训练语义估计。

`ROBUST` 不得只使用 raw breadth：正式密度稳健证据要求 raw 四指标、raw breadth、local-scaled 四指标和 local-scaled breadth 在相同连续层段内共同成立。

## 多重性与抽样

- bootstrap 单位为 semantic ID，而非句子行或语言对。
- shuffled baseline 使用无固定点 derangement。
- 标签置换在每个 semantic ID 内独立置换目标标签；保存每次置换的最大目标统计量。
- 主置换结论使用跨目标且跨层的 family-wise max statistic；逐层 p 值仅作定位。
- Alignment 的模块状态要求总体 AlignmentGain 连续至少 3 个非 embedding 层 CI 下界大于 0，且至少 80% 的有向语言对在某个非 embedding 层 Recall@1 CI 下界超过随机基线。
- 内部样本稳健性使用 10 个无放回随机 80-ID 子集；至少 80% 子集复现全样本模型状态记为 `REPLICATED`。该检查不能代替从完整 dev 随机抽样或完整 dev 确认。
- 新模型选择清单必须在读取其 `paper_v1` 结果前冻结。

## 允许与禁止的结论

- Alignment 通过：允许讨论共享语义表示；否则只能描述几何邻接。
- 英语选择校正通过：允许称英语具有特殊性；仅英语 CI>0 不足以支持该说法。
- 几何控制通过：允许称现象不完全由局部密度/公共方向解释。
- probe/purity 只支持“语言信息可读”，不支持模型主动使用该信息。
- 没有生成关联时不得声称功能影响；没有干预与对照时不得声称因果机制。

## 方法修改治理

允许修复实现错误、候选池泄漏、数值异常或补充预先解释的替代解释控制。禁止因模型未通过而降低连续层门槛、删除密度控制、挑选 pooling/k/层窗、隐藏反转层或不断追加模型直到获得正例。
