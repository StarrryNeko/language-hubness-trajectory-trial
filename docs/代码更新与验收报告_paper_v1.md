# paper_v1 代码更新与验收报告

## Overall Assessment: Ready for server method reanalysis

代码已经能够对兼容的 `mean_pool_no_eos_v1` hidden arrays 完成离线论文分析，并将旧 `first_n` 结果与正式随机样本结果分级。真实三模型结果尚未在本机重算，因此当前验收只覆盖软件正确性、合成端到端运行和历史 artifact 兼容性，不包含新的实证结论。

## 已实现模块

- AlignmentGain、无固定点 shuffled baseline、双向 Recall@1/5；
- 24 目标语言对称轮换与英语排名；
- 英语相对最佳非英语候选的效应；
- 跨目标、跨层最大统计量标签置换；
- raw 与 local-scaled 四指标及各自来源 breadth；
- 去均值、移除前 1/3/5 主成分的两折 semantic-ID cross-fitting；
- 各向异性诊断、participation-ratio effective rank；
- k=1/3/5/10 的完整状态、效应、排名、峰值层和置换 p 值；
- 随机 semantic subset 内部稳定性；
- 跨语义 neighborhood purity、centroid separation/dispersion；
- semantic-ID split 的线性语言 probe 与标签置换基线；
- 模块 manifest、semantic-ID hash、状态分层和 claim boundary；
- 已完成模型的离线 suite、跨模型汇总、归档导入、运行状态审计；
- 随机 200-ID 正式三模型配置及 extraction-only/formal suite 入口。

## 计算与候选范围核对

- Hubness：同一 semantic ID 内 24 个语言节点，排除自身；
- Alignment：相同语言对的 paired semantic 与 deranged semantic baseline；
- Retrieval：候选仅来自目标语言；
- Purity：排除 query 的完整平行语义组，保留同语言、不同语义候选；
- Probe：同一 semantic ID 的全部翻译进入相同 split；
- PCA/centering：测试 semantic 不参与自身变换方向拟合；
- Bootstrap：主要离线证据以 semantic ID 为抽样单位。

## 验收结果

- 单元测试：53 项通过；
- 合成端到端测试：通过；
- 合成流程覆盖 Alignment、geometry、hubness、k sweep、sample robustness、language structure、probe、figures 和 validation summary；
- 7 月 26 日三个 archived hidden arrays 均通过当前兼容性检查；
- 所有 JSON 配置解析通过；
- `git diff --check` 通过；
- 未覆盖或修改用户已有的 `.gitignore` 变更。

## 仍需服务器验证

1. 三个真实旧模型的 `paper_v1` 运行时间、内存和磁盘占用；
2. 真实 24×100/200 数据上的 PCA 与 purity 计算性能；
3. 三模型的 alignment、英语选择校正和 geometry status；
4. 随机 200-ID 正式样本与旧 `first_n` 样本之间的结果差异；
5. 完整 dev 复现是否必要。

## 结论边界

当前代码不会把离线几何结果升级为行为或因果结论。`behavior_association_status` 与 `intervention_status` 保持 `NOT_RUN`。生成任务、语言识别器、行为质量指标和激活方向应在正式三模型几何状态确定后冻结；否则容易根据有利层和模型反向设计干预。

