# behavior_v2 服务器运行指南

以下命令均在仓库根目录 `/root/autodl-tmp/language-hubness-trajectory-trial` 执行。V2 不依赖 fastText 权重。

## 单模型先导检查

若要在三模型正式运行前先验证完整协议，使用 XGLM 单模型 suite：

```bash
export BEHAVIOR_V2_SUITE=configs/behavior_v2/suite_xglm_only.json
bash scripts/run_behavior_v2_gpu.sh prepare
bash scripts/run_behavior_v2_gpu.sh generate
bash scripts/run_behavior_v2_gpu.sh audit
```

完成盲审和检测器验证后再运行 `analyze`。单模型入口不会执行跨模型复现判断，也不会改变三模型正式 suite 的配置和输出。

## 0. 同步与静态检查

```bash
cd /root/autodl-tmp/language-hubness-trajectory-trial
export BEHAVIOR_MODEL_ROOT=/root/autodl-tmp/models
python -m pip install -r requirements.txt
python -m compileall src scripts tests
python -m unittest discover -s tests -p 'test_behavior_v2.py' -v
```

## 1. 准备数据、hidden states 和几何量（需要 GPU）

```bash
bash scripts/run_behavior_v2_gpu.sh prepare 2>&1 | tee behavior_v2_prepare.log
```

这一阶段冻结 FLORES+ devtest 样本、审计三个本地 checkpoint、抽取唯一的 mean-pool 表征，并计算全语言几何。成功标志为：

```text
behavior_v2 prepare stage complete: extraction, checkpoint audit, tasks, geometry.
```

## 2. 三个模型依次生成（需要 GPU）

```bash
bash scripts/run_behavior_v2_gpu.sh generate 2>&1 | tee behavior_v2_generate.log
```

默认批量为 XGLM 256、Mistral 64、Aya 48，并通过 PyTorch allocator 将单进程显存硬限制为 76 GiB；达到上限时批量自动减半，不会改变任务、解码或统计协议。需要单模型重跑时：

```bash
python -u src/generate_behavior_v2.py --config configs/behavior_v2/xglm_1b7.json --resume
python -u src/generate_behavior_v2.py --config configs/behavior_v2/mistral_7b_v01.json --resume
python -u src/generate_behavior_v2.py --config configs/behavior_v2/aya_23_8b.json --resume
```

## 3. 创建盲审表（可切换为无卡模式）

```bash
bash scripts/run_behavior_v2_gpu.sh audit 2>&1 | tee behavior_v2_audit.log
```

分别编辑以下文件中的 `human_english_leakage` 列，只填写 `0` 或 `1`，不要改变 `task_id`：

```text
outputs_behavior_v2/xglm_1b7/behavior_v2/measurement/lexical_detector_audit.csv
outputs_behavior_v2/mistral_7b_v01/behavior_v2/measurement/lexical_detector_audit.csv
outputs_behavior_v2/aya_23_8b/behavior_v2/measurement/lexical_detector_audit.csv
```

标注准则：只有输出中确实出现不属于目标文本所需的英语词汇片段才标 1；人名、地名、网址、缩写、音译和单个借词默认标 0，除非上下文明确形成英语短语。

## 4. 验证三份盲审结果

```bash
python src/annotate_behavior_v2.py validate \
  --config configs/behavior_v2/xglm_1b7.json \
  --annotations outputs_behavior_v2/xglm_1b7/behavior_v2/measurement/lexical_detector_audit.csv \
  --output outputs_behavior_v2/xglm_1b7/behavior_v2/measurement/lexical_detector_validation.json

python src/annotate_behavior_v2.py validate \
  --config configs/behavior_v2/mistral_7b_v01.json \
  --annotations outputs_behavior_v2/mistral_7b_v01/behavior_v2/measurement/lexical_detector_audit.csv \
  --output outputs_behavior_v2/mistral_7b_v01/behavior_v2/measurement/lexical_detector_validation.json

python src/annotate_behavior_v2.py validate \
  --config configs/behavior_v2/aya_23_8b.json \
  --annotations outputs_behavior_v2/aya_23_8b/behavior_v2/measurement/lexical_detector_audit.csv \
  --output outputs_behavior_v2/aya_23_8b/behavior_v2/measurement/lexical_detector_validation.json
```

三个命令都必须打印 `passed=True`。失败时不能调阈值后直接进入正式分析；应检查错误类型，并将任何规则修改作为新版本协议。

## 5. 正式关联、验证和跨模型比较（无卡可运行）

```bash
bash scripts/run_behavior_v2_gpu.sh analyze 2>&1 | tee behavior_v2_analyze.log
```

最终状态文件：

```text
outputs_behavior_v2/model_comparison_three/cross_model_status.json
```

各模型还会输出全语言距离矩阵、每层距离热图、语言质心 PCA、script concentration、English advantage、任务级测量、关联回归和完整 blocker 清单。不要在首次运行时直接使用 `all`，因为人工盲审是不可跳过的正式性断点。
