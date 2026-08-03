# 服务器执行指南：paper_v1

## 结论先行

不要先补跑 7 月 30 日新增但尚未完成的模型，也不要把旧的前 100 条结果直接命名为正式确认性实验。

正确顺序是：

1. 把已经完成的 Qwen、BLOOM、XGLM `hidden/` 结果放回各旧配置的 `output_dir`；
2. 只做离线 `paper_v1` 复算，验证新指标、目标轮换、密度 breadth、公共方向控制和 probe；
3. 冻结方法与模型清单；
4. 使用独立的 `outputs_paper_v1/` 和随机 200 semantic IDs 从头运行正式三模型实验；
5. 根据三模型的正式状态决定是否进入新模型扩展和生成干预。

旧结果的价值是方法校准和结果分支选择；正式结果必须使用预先冻结的随机样本，避免“前 100 条”和看过结果后改方法的问题。

## 0. 安装与工作目录

```bash
cd language-hubness-trajectory
pip install -r requirements.txt
```

所有相对 `output_dir` 都相对于上述仓库根目录解析。

## 1. 审计旧三模型文件

如果上传服务器时仍保留了 `7.26代码调整结果/outputs/` 归档，可在 Linux
服务器用符号链接把它安全映射到当前配置路径，避免复制约 1.4GB hidden arrays：

```bash
python src/import_archived_outputs.py \
  --source-root ../7.26代码调整结果/outputs \
  --suite configs/model_suite_24lang.json \
  --mode symlink
```

如果服务器文件系统不允许符号链接，改用 `--mode copy`。脚本拒绝覆盖任何
非空目标目录。

然后审计：

```bash
python src/inspect_run_state.py \
  --suite configs/model_suite_24lang.json
```

状态解释：

- `READY_FOR_PAPER_REANALYSIS` / `EXTRACTION_REUSABLE`：不要重新提取，直接离线复算；
- `PARTIAL_OR_INCOMPATIBLE_EXTRACTION`：当前版本不能从半个 `.npy` 安全续接，该模型需要重新提取，但暂时不要启动新模型；
- `DATA_READY_EXTRACTION_MISSING`：只有数据，没有完整 hidden；
- `PAPER_ANALYSIS_COMPLETE`：使用 `--resume` 即可跳过。

## 2. 先复算已完成的旧三模型

```bash
python src/run_paper_suite.py \
  --suite configs/model_suite_24lang.json \
  --resume
```

这个命令永远不会下载或加载模型权重。没有完整 extraction 的模型会被列出并跳过。旧结果写入每个模型的：

```text
output_dir/paper_v1/
```

重点查看：

```text
paper_v1/validation/paper_validation_summary.md
paper_v1/metrics/hubness/paper_model_status.json
paper_v1/metrics/hubness/target_rotation_summary.csv
paper_v1/metrics/hubness/target_rotation_permutation.csv
paper_v1/metrics/hubness/k_robustness_summary.csv
```

旧配置使用 `first_n`，验证汇总会明确标为 `METHOD_REANALYSIS_ONLY`，不会伪装成正式确认结果。

## 3. 方法冻结门槛

只有满足以下条件才开始正式提取：

- 所有模块没有 `INVALID` 或缺失 manifest；
- 24 目标语言轮换覆盖完整；
- raw 与 local-scaled breadth 均存在；
- max-target/max-layer permutation 输出完整；
- k=1/3/5/10 均有完整状态、效应、排名与 p 值；
- Alignment、language structure、probe 的 semantic-ID hash 一致；
- 旧三模型的差异能够落入预先声明的结果分支；
- 不再因为正负结果修改门槛。

## 4. 从头运行正式随机 200-ID 三模型实验

正式配置使用单独输出目录，不覆盖旧 hidden：

```bash
python src/run_formal_suite.py \
  --suite configs/model_suite_paper_v1_random200.json \
  --resume
```

该命令先准备所有语言共享的随机 semantic indices，再只做 hidden 提取，随后执行全部离线 `paper_v1` 模块。BLOOM 与 XGLM 复用 Qwen 准备的数据，并进行哈希核对。

若任务中断，再运行同一条 `--resume` 命令：已经生成完整 extraction manifest、metadata 和 `.npy` 的模型会被跳过；在 `.npy` 完成前中断的单个模型仍需从该模型开头重新提取，其他已完成模型不会重跑。

## 5. 何时扩展新模型

不要把“至少两个模型必须为正”设为启动条件。正式三模型完成后按识别需要决定：

- 如果英语选择校正与几何控制均不支持，优先写清伪影边界，不通过增加模型寻找正例；
- 如果只有 BLOOM 支持，新增模型应检验训练语言分布/模型家族，而不是重复尝试相近模型；
- 如果模型间差异稳定，冻结 `<20B` 扩展清单后再运行；
- 生成关联和激活干预先在三种边界模型上小规模开展，只有出现预测性关联才扩大干预。
