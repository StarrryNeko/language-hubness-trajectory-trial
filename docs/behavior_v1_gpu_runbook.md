# behavior_v1 GPU/24 核服务器操作流程

## 资源分工

- GPU：三个模型的 hidden-state 抽取、greedy generation、样本级 cosine 和 local-scaled cosine。
- 24 核 CPU：数据准备、fastText/chrF++ 并发评估、随机 PCA、cluster bootstrap、statsmodels cluster-robust 回归和文件校验。
- PCA 与回归没有强行迁移到 GPU：当前数据规模下传输和重写成本高于收益，CPU 数值库使用 24 线程更稳定。

基础配置已冻结 `cpu_threads=24`、`evaluation_workers=24`、`geometry_device=cuda`、`geometry_dtype=float32` 和 `allow_tf32=false`。关闭 TF32 是为了减少不同 GPU 之间的数值漂移。

## 0. 服务器准备

```bash
cd /root/autodl-tmp/language-hubness-trajectory

export LHT_MODEL_ROOT=/root/autodl-tmp/models
export CUDA_VISIBLE_DEVICES=0
export CPU_THREADS=24

python -m pip install -r requirements.txt
```

包装脚本会忽略终端中可能残留的旧 `LHT_MODEL_ROOT`，默认固定读取 `/root/autodl-tmp/models`。如需在其他服务器改根目录，使用 `BEHAVIOR_MODEL_ROOT=/new/path` 覆盖。

模型目录必须采用项目的 portable 名称：

```text
$LHT_MODEL_ROOT/
  facebook__xglm-1.7B/
  mistralai__Mistral-7B-v0.1/
  CohereLabs__aya-23-8B/
```

并准备：

```text
/root/autodl-tmp/lid/lid.176.bin
```

## 1. 一条命令运行全部流程

```bash
bash scripts/run_behavior_v1_gpu.sh all 2>&1 | tee behavior_v1_all.log
```

脚本自动设置 OpenMP、MKL、OpenBLAS、NumExpr 为 24 线程，检查 CUDA、fastText、SacreBLEU 和 statsmodels，然后依次运行抽样、hidden states、checkpoint audit、任务构造、GPU 几何预测量、生成、评估、关联回归、验证和三模型汇总。

## 2. 推荐的分阶段运行

### 阶段 A：准备数据、hidden states 和几何预测量

```bash
bash scripts/run_behavior_v1_gpu.sh prepare 2>&1 | tee behavior_v1_prepare.log
```

该阶段完成：

1. 排除两个几何 seed 的 363 个 ID；
2. 抽取冻结的 208 个新 ID；
3. 三模型 GPU hidden-state 抽取；
4. checkpoint SHA-256 审计；
5. 构造每模型 4,800 个任务；
6. 在 GPU 上计算 cosine/local-scaled predictors，并在 24 核 CPU 上计算 PCA。

### 阶段 B：三模型生成

```bash
bash scripts/run_behavior_v1_gpu.sh generate 2>&1 | tee behavior_v1_generate.log
```

生成过程逐批追加 JSONL；再次执行同一命令会核对 task、prompt、reference 和 checkpoint 哈希，然后从未完成 task 继续。

### 阶段 C：评估、关联统计和跨模型比较

```bash
bash scripts/run_behavior_v1_gpu.sh analyze 2>&1 | tee behavior_v1_analyze.log
```

该阶段使用 24 个并发 worker 进行 LID/chrF++ 评估，随后运行 24 线程 CPU 回归与 bootstrap，输出单模型验证和三模型复现状态。

## 3. 不使用包装脚本时的等价命令

```bash
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24
export RAYON_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0
export LHT_MODEL_ROOT=/root/autodl-tmp/models

python src/run_behavior_suite.py \
  --suite configs/model_suite_behavior_v1.json \
  --stage prepare --resume

python src/run_behavior_suite.py \
  --suite configs/model_suite_behavior_v1.json \
  --stage generate --resume

python src/run_behavior_suite.py \
  --suite configs/model_suite_behavior_v1.json \
  --stage analyze --resume
```

## 4. 关键输出

```text
outputs_behavior_v1/<model>/
  checkpoint_identity.json
  hidden/sentence_layer_mean_pool.npy
  behavior_v1/data/behavior_tasks.jsonl
  behavior_v1/generations/generations.jsonl
  behavior_v1/metrics/behavior_geometry_predictors.csv
  behavior_v1/metrics/behavior_item_results.csv
  behavior_v1/metrics/behavior_association_results.csv
  behavior_v1/validation/behavior_validation_summary.json

outputs_behavior_v1/model_comparison_three/
  behavior_model_metric_comparison.csv
  behavior_model_association_comparison.csv
  behavior_cross_model_status.json
```

显存不足时，先把对应模型配置中的 `behavior_v1.decoding.batch_size` 调低；不要修改样本 seed、排除清单、主层、任务数或统计规则。
