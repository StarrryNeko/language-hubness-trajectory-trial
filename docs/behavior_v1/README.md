# behavior_v1 归档说明

V1 的正式结果与协议保持不变。此次整理只改变代码组织，不改变输出目录、字段或旧命令。

- 核心实现：`src/behavior_v1/`
- 整理后的配置入口：`configs/behavior_v1/`
- 服务器入口：`scripts/run_behavior_v1_gpu.sh`
- 历史顶层 Python 文件仍是兼容薄封装，因此原有命令、测试和已有结果均可继续使用。
- 输出仍写入 `outputs_behavior_v1/`，不会被 V2 覆盖。

V1 的表征协议仍只有 `mean_pool_v1`。本次整理不重算、不迁移也不修改任何已有 V1 结果。
