# 一周冲刺服务器执行指南

> 适用清单：`configs/model_suite_week1_required_random200.json`  
> 条件扩展：`configs/model_suite_week1_with_moonlight_random200.json`  
> 正式协议：24语言、随机200 semantic IDs、全部层、mean-pool、不追加EOS、BF16推理、FP32句向量存储。

## 结论先行

权重可以在另一台联网机器单独下载后上传。推荐使用本项目的便携目录格式，而不是复制 Hugging Face 缓存内部的 `blobs/refs/snapshots` 结构。服务器设置 `LHT_MODEL_ROOT` 后会自动把官方模型ID映射到上传目录；manifest继续记录官方ID和本地解析路径，跨机器结果仍可审计。

正式顺序是：

1. 新主机完成GPU与环境smoke test；
2. 验证上传权重目录；
3. 用7月26日旧hidden完成方法复算；
4. 冻结代码和配置；
5. 从头运行五个必跑模型的随机200-ID正式实验；
6. Moonlight通过两小时审计后再加入；
7. 生成跨模型汇总并备份。

## 一、在联网机器单独下载权重

如果GPU主机本身连接Hugging Face速度稳定，直接在GPU主机运行下载器最简单：

```bash
python src/download_model_weights.py \
  --suite configs/model_suite_week1_required_random200.json
```

脚本支持续传并把权重写入配置的`/root/autodl-tmp/huggingface`。但如果主机网络不稳定、租用期间下载计费明显，或希望以后换卡复用权重，再使用下面的便携目录方案。五个必跑模型约需33GB；加入Moonlight后约65GB。使用个人电脑会经历“下载一次、上传一次”两次网络传输，因此家庭上行较慢时，优先考虑平台的廉价CPU实例、对象存储或可挂载持久数据盘。

### 1.1 准备环境

```bash
cd language-hubness-trajectory
python -m venv .venv-download
source .venv-download/bin/activate
python -m pip install --upgrade pip
python -m pip install "huggingface_hub>=0.24"
```

Windows PowerShell激活命令：

```powershell
.\.venv-download\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "huggingface_hub>=0.24"
```

Llama是gated模型。先在Hugging Face网页接受模型许可，再在下载机器登录：

```bash
huggingface-cli login
```

Moonlight不需要把任何服务端API替代本地权重；主实验必须使用可返回全部hidden states的本地checkpoint。

### 1.2 下载五个必跑模型

```bash
python src/download_model_weights.py \
  --suite configs/model_suite_week1_required_random200.json \
  --output-root /data/langhub_models
```

Windows示例：

```powershell
python src/download_model_weights.py `
  --suite configs/model_suite_week1_required_random200.json `
  --output-root D:\langhub_models
```

下载结果使用稳定目录名：

```text
langhub_models/
  Qwen__Qwen2.5-1.5B/
  bigscience__bloom-1b7/
  facebook__xglm-1.7B/
  meta-llama__Llama-3.2-3B/
  meta-llama__Llama-3.1-8B/
  portable_models_manifest.json
```

下载Moonlight：

```bash
python src/download_model_weights.py \
  --suite configs/model_suite_week1_with_moonlight_random200.json \
  --model moonshotai/Moonlight-16B-A3B \
  --output-root /data/langhub_models
```

脚本只下载safetensors或必要的PyTorch权重、tokenizer、配置和custom code；不会下载重复的TensorFlow、Flax、ONNX或`original/`权重。

### 1.3 上传前验证

```bash
python src/download_model_weights.py \
  --suite configs/model_suite_week1_required_random200.json \
  --verify-root /data/langhub_models
```

需要看到：

```text
PORTABLE_MODEL_ROOT_VERIFIED
```

### 1.4 上传建议

safetensors本身几乎不可压缩，不建议花数小时制作高压缩比zip。优先使用云平台数据盘上传、对象存储或支持断点续传的`rsync`：

```bash
rsync -avP --partial /data/langhub_models/ user@server:/root/autodl-tmp/models/
```

如果只能使用`scp`：

```bash
scp -r /data/langhub_models user@server:/root/autodl-tmp/models
```

不要上传本机Hugging Face token文件。便携权重上传完成后，服务器离线运行不需要该token。

## 二、新主机环境

### 2.1 检查硬件

```bash
nvidia-smi
free -h
df -h /root/autodl-tmp
nproc
```

最低门槛：

- GPU显示完整96GB RTX PRO 6000或80GB H800，而不是MIG/vGPU切片；
- 系统内存至少64GB，推荐128GB；
- 数据盘至少300GB可用，推荐500GB；
- 至少16 vCPU。

### 2.2 安装环境

RTX PRO 6000 Blackwell优先使用平台提供的CUDA 12.8或更新的PyTorch镜像。不要仅根据`nvidia-smi`顶部显示的“CUDA Version”判断PyTorch是否支持Blackwell；必须实际运行下一节hidden-state审计。

```bash
cd /root/autodl-tmp/language-hubness-trajectory
mkdir -p logs audits
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

H800可使用平台成熟的CUDA 12.x PyTorch镜像；仍应把实际torch、driver、模型revision写入运行日志。

### 2.3 设置数据盘与离线权重

```bash
export LHT_MODEL_ROOT=/root/autodl-tmp/models
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/hub
```

确认所有权重已上传后，再启用严格离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

先验证目录：

```bash
python src/download_model_weights.py \
  --suite configs/model_suite_week1_required_random200.json \
  --verify-root "$LHT_MODEL_ROOT"
```

## 三、运行前模型审计

### 3.1 先审计最小模型

```bash
python src/audit_model_runtime.py \
  --suite configs/model_suite_week1_required_random200.json \
  --model Qwen/Qwen2.5-1.5B \
  --sentences 10 \
  --output audits/qwen_smoke.json
```

通过条件：

- 模型来自`LHT_MODEL_ROOT`；
- 返回embedding加全部Transformer层hidden states；
- 所有shape固定；
- 无NaN/Inf；
- 显存峰值合理。

再审计其余必跑模型：

```bash
python src/audit_model_runtime.py \
  --suite configs/model_suite_week1_required_random200.json \
  --sentences 10 \
  --continue-on-error \
  --output audits/required_models.json
```

### 3.2 Moonlight两小时止损审计

```bash
python src/audit_model_runtime.py \
  --suite configs/model_suite_week1_with_moonlight_random200.json \
  --model moonshotai/Moonlight-16B-A3B \
  --sentences 10 \
  --output audits/moonlight.json
```

两小时内仍不能通过时，保留`audits/moonlight.json`及错误日志，记录`TECHNICAL_EXCLUSION`，本周不继续修改框架或更换有利模型。

## 四、先复算旧hidden，不重新加载权重

如果上传了`7.26代码调整结果/outputs`：

```bash
python src/import_archived_outputs.py \
  --source-root ../7.26代码调整结果/outputs \
  --suite configs/model_suite_24lang.json \
  --mode symlink

python src/run_paper_suite.py \
  --suite configs/model_suite_24lang.json \
  --resume
```

这一步会新增：

```text
paper_v1/metrics/similarity_competition/
paper_v1/metrics/norm_trajectory/
paper_v1/figures/english_competition_by_layer.png
paper_v1/figures/english_geometry_mechanisms.png
paper_v1/figures/source_candidate_attraction_peak_layer.png
```

旧first-100输出只能标记为`METHOD_REANALYSIS_ONLY`，不能混入正式结论。

## 五、正式五模型实验

### 5.1 权重未全部到齐时先运行成员A四模型

如果Qwen权重仍在传输，但BLOOM、XGLM和Llama-3.2-3B已经通过目录验证和运行审计，可以先运行三模型执行子集：

```bash
python -u src/run_formal_suite.py \
  --suite configs/model_suite_week1_available_three_random200.json \
  --resume 2>&1 | tee logs/week1_available_three.log
```

三模型子集使用与四模型、五模型清单完全相同的数据选择协议和逐模型配置。它只用于提前完成可用模型的提取与分析，不能作为最终确认性跨模型比较。Qwen到位后继续使用下方四模型清单并加`--resume`。

成员A的四个模型已上传并通过运行审计后，可以先执行：

```bash
python -u src/run_formal_suite.py \
  --suite configs/model_suite_week1_member_a_random200.json \
  --resume 2>&1 | tee logs/week1_member_a.log
```

该子清单与正式清单使用完全相同的随机200-ID、种子、配置文件和逐模型输出目录，只把跨模型比较写入独立目录。它不改变预先冻结的五模型设计。第五个模型到位后运行下方正式五模型命令；`--resume`会复用通过manifest、revision、配置和数据哈希检查的前四个模型，仅提取新增模型并重做五模型比较。

### 5.2 五模型正式汇总

确认代码、配置和旧hidden复算均通过后，保存git commit或源码压缩包哈希，然后运行：

```bash
python -u src/run_formal_suite.py \
  --suite configs/model_suite_week1_required_random200.json \
  --resume 2>&1 | tee logs/week1_required.log
```

该命令会：

1. 准备一次随机200-ID、24语言的共同数据；
2. 逐个模型提取全部层mean-pool hidden；
3. 运行AlignmentGain、全语言KNN、竞争性余弦、范数/质心/密度、逐层轨迹、purity/probe和稳健性分析；
4. 生成逐模型验证摘要；
5. 生成跨模型汇总。

模型是逐个加载的，不会同时占据GPU。中断后重新运行同一命令和`--resume`：完整模型会跳过；只有在`.npy`完成前中断的当前模型需要从该模型起点重新提取。

## 六、加入Moonlight

只有Moonlight审计通过后执行：

```bash
python -u src/run_formal_suite.py \
  --suite configs/model_suite_week1_with_moonlight_random200.json \
  --resume 2>&1 | tee logs/week1_with_moonlight.log
```

前五个模型若config snapshot、数据哈希和extraction manifest完全匹配会被复用，仅新增Moonlight并重做六模型比较。

## 七、验收与备份

逐模型检查：

```text
outputs_week1/<model>/extraction_manifest.json
outputs_week1/<model>/paper_v1/run_manifest.json
outputs_week1/<model>/paper_v1/validation/paper_validation_summary.json
outputs_week1/<model>/paper_v1/metrics/similarity_competition/similarity_competition_status.json
outputs_week1/<model>/paper_v1/metrics/norm_trajectory/norm_trajectory_manifest.json
```

跨模型检查：

```text
outputs_week1/model_comparison_required_random200/paper_v1/paper_model_comparison.csv
outputs_week1/model_comparison_required_random200/paper_v1/paper_model_comparison.json
```

备份优先级：配置、manifest、metadata、汇总CSV/JSON、图表、日志、hidden数组。不要只下载图片而丢失semantic hash和原始句向量。

## 八、常见失败处理

| 失败 | 处理 |
| --- | --- |
| 离线模式报告缺文件 | 回到下载机器对相同suite再次执行`--output-root`，然后增量上传；不在服务器临时改模型ID |
| Llama 401/403 | 下载机器尚未接受许可；在网页授权后重新下载 |
| Blackwell `no kernel image` | 换CUDA 12.8兼容PyTorch镜像；两小时仍失败则切H800 |
| CPU内存不足 | 把实例升级到64/128GB；不要通过量化正式权重规避 |
| Moonlight remote code失败 | 保存审计JSON并技术排除；不影响五模型主线 |
| 当前模型提取中断 | 用同一suite和`--resume`重跑；已完整模型不会重复 |
| 结果不支持英语假设 | 保留负结果并按预先定义的claim boundary报告；不改阈值或追加模型找正例 |
