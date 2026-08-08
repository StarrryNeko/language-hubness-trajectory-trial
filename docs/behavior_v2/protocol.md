# behavior_v2 协议入口

当前 V2 已收敛为全语言表征几何主实验，不再将文本生成和行为关联纳入主流程。

规范性设计、指标定义、跨模型门槛、实现状态与下一步操作见：

- [V2 设计框架：全语言表征几何主实验](design_framework.md)

历史生成实验代码暂时保留用于追溯，但在 geometry-only 重构完成和结构准入门槛满足前，不得作为 V2 正式入口运行。

当前实现入口已拆分为：

- [structure_v2 纯几何协议](../structure_v2/protocol.md)
- [behavior_association_v3 第三阶段协议](../behavior_association_v3/protocol.md)
