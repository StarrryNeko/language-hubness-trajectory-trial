# 行为实验代码结构

```text
src/
  behavior_v1/              # V1 核心实现（已冻结）
  behavior_v2/              # V2 核心实现
  *_behavior*.py            # 旧命令或服务器友好的薄入口
configs/
  behavior_v1/              # V1 整理入口；继承历史冻结配置
  behavior_v2/              # V2 base、三模型与 suite
scripts/
  run_behavior_v1_gpu.sh
  run_behavior_v2_gpu.sh
docs/
  behavior_v1/
  behavior_v2/
outputs_behavior_v1/        # 历史 V1 输出，不被 V2 写入
outputs_behavior_v2/        # 新 V2 输出
```

兼容薄入口只负责导入相应版本包并调用 `main()`，没有第二份实现。这样原服务器命令仍可用，同时 V1 与 V2 的协议逻辑不会混在同一文件中。
