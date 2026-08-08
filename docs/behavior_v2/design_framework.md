# V2 设计框架：全语言表征几何主实验

> 文档状态：当前 V2 的规范性设计说明  
> 表征协议：`mean_pool_v1`  
> 研究类型：同语义、跨语言、跨模型的观察性表征几何实验  
> 当前边界：V2 主实验不包含文本生成、行为关联或激活干预

## 技术摘要

V2 的核心目标是直接检验：在同一语义、同一模型层中，英语是否相对其他语言处于更中心、更高 hubness 的位置，以及这种优势能否与 Latin 文字语言的整体集中效应区分开来。

正式分析只读取各模型已经抽取的 `mean_pool` 句向量。对每个语义 ID，在冻结目标层构造 24 种语言的 `24 × 24` 相似度与距离矩阵，再以语义 ID 为独立统计单位汇总 Latin 集中度、英语特异中心性、reverse-kNN hubness、K 敏感性和局部密度校正结果。

文本生成、英语词汇泄露、翻译质量、人工盲审和行为回归不是 V2 主实验的组成部分。只有当几何效应满足预先冻结的跨模型门槛后，才另行启动行为关联实验。

## 1. 研究问题与可支持结论

### 1.1 Latin 语言是否形成集中区域

检验 Latin 文字语言之间的平均相似度是否高于 Latin 与非 Latin 语言之间的平均相似度。

若该差值稳定大于 0，可支持“Latin 语言在该层形成更紧密表征区域”的结构性描述。

### 1.2 英语是否具有超出 Latin 共性的额外优势

英语属于 Latin 文字系统，因此不能仅将英语与全部非英语语言比较。V2 将英语与“其他 Latin 语言的平均水平”比较，以区分：

- Latin 文字系统的共同集中效应；
- 英语在 Latin 语言内部仍然存在的特异中心性或 hubness 优势。

### 1.3 英语 hubness 是否对 K 和局部密度校正稳健

英语被其他语言反复选为近邻，才构成 hubness 证据。V2 在 K=`3/5/10` 下计算 reverse-kNN occurrence，并比较 raw cosine 与 local-scaled cosine。

若优势只存在于 raw cosine、在局部密度校正后消失，应优先解释为密度差异或各向异性效应，而不是稳健 hubness。

### 1.4 V2 不支持的结论

V2 是观察性的内部表征实验，不能单独证明：

- 英语中心性会导致输出中的英语泄露；
- 某一语言向量对生成具有因果控制作用；
- 激活干预能够改变模型行为；
- 所有多语言模型都遵循同一机制。

## 2. 数据、模型与分析范围

### 2.1 数据

- 数据源：FLORES+ `devtest`。
- 语言数：24。
- 每个语义 ID 必须具有完整且一一对应的 24 语言平行句。
- V2 与 V1 使用的数据 split 分开，避免复用 V1 `dev` 行为结果。
- 几何主实验直接使用冻结样本中的全部 804 个语义 ID，不再划分 demonstration 与 evaluation ID。

当前配置曾为生成实验保留 4 个 demonstration ID、使用 800 个评估 ID。几何主流程重构后应去掉这一划分，并对全部 804 个 ID 离线重算；该重算不需要重新加载模型或执行文本生成。

### 2.2 模型角色

| 模型 | 冻结角色 | 主层 | 相邻描述层 |
|---|---|---:|---|
| XGLM-1.7B | 预期阳性模型 | 12 | 11、13 |
| Mistral-7B-v0.1 | 冻结负对照 | 30 | 29、31 |
| Aya-23-8B | 预期阳性模型 | 31 | 30、32 |

模型角色沿用 V1 冻结设计，不根据 V2 结果重新选择模型。

### 2.3 唯一表征

正式协议只允许：

```text
mean_pool
```

协议版本为：

```text
mean_pool_v1
```

每个向量对应原始句子文本 token 的层级 hidden state 均值。任何其他句向量变体均不进入 V2 主分析。

## 3. 基本分析单元

设：

- `s` 为语义 ID；
- `l` 为模型层；
- `a,b` 为语言；
- `v(s,l,a)` 为语言 `a` 在语义 `s`、层 `l` 的 mean-pool 向量。

对每个语义 ID 和分析层计算：

```text
cosine_similarity(s,l,a,b) = cos(v(s,l,a), v(s,l,b))
cosine_distance(s,l,a,b)   = 1 - cosine_similarity(s,l,a,b)
```

因此，每个 `(语义 ID, 层)` 都得到一张 `24 × 24` 对称矩阵。所有比较必须限制在同一语义 ID 内，禁止将不同语义句子互相作为候选或近邻。

统计汇总和重采样的独立单位是语义 ID，而不是语言对，也不是矩阵中的单个单元格。

## 4. 核心指标

### 4.1 全语言成对距离

对每个语言对输出：

- 平均 cosine similarity；
- 平均 cosine distance；
- 以语义 ID 重采样得到的置信区间；
- 两种语言的文字系统与语系元数据。

该结果用于绘制全语言距离热图，并提供所有后续组间比较的基础数据。

### 4.2 Latin 集中度

对每个语义 ID 分别计算：

```text
within_latin
within_non_latin
between_latin_non_latin
latin_concentration = within_latin - between_latin_non_latin
```

计算组内均值时必须排除对角线和重复方向。`latin_concentration > 0` 表示 Latin 语言内部比跨文字系统语言对更接近。

### 4.3 语言中心性

语言 `a` 在某个语义 ID 中的中心性定义为其与其余 23 种语言的平均相似度：

```text
centrality(a) = mean cosine(a, all other languages)
```

英语特异中心性定义为：

```text
english_centrality_excess
= centrality(en) - mean centrality(other Latin languages)
```

该比较控制了英语属于 Latin 文字系统这一事实。

### 4.4 Reverse-kNN hubness

对每一种语言，将同语义的其余语言按相似度排序，选取前 K 个近邻。统计一种语言被多少其他语言选入其近邻集合：

```text
k_occurrence(a, K)
```

英语特异 hubness 定义为：

```text
english_hubness_excess(K)
= k_occurrence(en, K)
 - mean k_occurrence(other Latin languages, K)
```

正式报告 K=`3/5/10`，其中 K=`5` 为主 K，K=`3/10` 为敏感性检查。

### 4.5 局部密度校正

所有确认性英语优势至少需要同时报告：

- raw cosine；
- local-scaled cosine。

局部密度校正用于判断英语优势是否主要由局部高密度或表示空间各向异性造成。raw 与 local-scaled 结果必须分开解释，不能将二者混为同一证据。

### 4.6 排名与二维展示

辅助结果包括：

- 英语在全部语言和 Latin 语言内部的中心性排名；
- 英语在 K=`3/5/10` 下的 hubness 排名；
- 语义中心化后的语言质心 PCA；
- 每层的全语言距离热图。

PCA 只用于展示语言分布，不作为确认性统计证据，其坐标轴方向也不能跨模型直接解释。

## 5. 统计设计

### 5.1 主分析层与相邻层

- 每个模型的冻结主层用于确认性检验。
- 主层前后各一层用于检查局部层级稳定性。
- 不允许在观察 V2 结果后重新选择“最显著层”作为主层。

### 5.2 不确定性

对 Latin 集中度、英语中心性 excess 和英语 hubness excess：

- 以语义 ID 为单位进行配对 bootstrap；
- 默认 bootstrap 次数为 1,000；
- 报告 95% 置信区间；
- 同时报告点估计、有效语义 ID 数和方向一致性。

现有几何导出代码中的正态近似置信区间应在正式重构时替换为上述 semantic-ID bootstrap，避免把语言对单元误当作独立观测。

### 5.3 主证据与敏感性证据

冻结主证据建议为：

1. 主层 `latin_concentration`；
2. 主层 `english_centrality_excess`；
3. 主层 K=`5` 的 `english_hubness_excess`；
4. 上述英语优势的 local-scaled 对照。

K=`3/10`、相邻层、语言排名和 PCA 均属于稳健性或描述性证据。若对大量次要层和指标进行显著性检验，应统一采用 BH 校正；不能从次要结果中挑选显著项替代失败的主检验。

## 6. 跨模型复现与行为实验准入门槛

V2 的最终目标不是要求单个模型显著，而是判断结构模式是否跨模型复现。

建议将进入行为关联实验的门槛冻结为：

1. 所有模型通过数据、checkpoint、语义对齐和 `mean_pool_v1` 完整性检查；
2. 至少两个预期阳性模型的 `latin_concentration` 方向一致；
3. 至少两个预期阳性模型的英语中心性或 K=`5` hubness excess 为正，且 95% 置信区间不跨 0；
4. K=`3/5/10` 的英语 hubness excess 方向一致；
5. local-scaled 后英语优势至少部分保留；
6. Mistral 负对照单独报告，用于评估模型特异性，不根据结果删除或替换。

只有满足上述门槛，才进入独立的行为关联阶段。若结构优势不复现，应停止生成实验，并将其报告为有效的结构性零结果或模型特异结果。

## 7. 正式输出

每个模型至少应生成：

```text
structure_v2/
  geometry/
    language_pair_similarity.csv
    language_centrality.csv
    script_concentration.csv
    english_advantage.csv
    language_centroid_pca.csv
    geometry_manifest.json
  figures/
    language_distance_heatmap_layer_<L>.png
    language_centroid_pca_layer_<L>.png
    script_concentration_trajectory.png
    english_advantage_trajectory.png
    figure_manifest.json
  validation/
    validation_summary.json
    validation_summary.md
```

跨模型输出至少包括：

```text
outputs_structure_v2/model_comparison_three/
  geometry_comparison.csv
  k_sensitivity_comparison.csv
  cross_model_status.json
```

manifest 必须记录数据内容哈希、语义 ID 哈希、语言顺序哈希、checkpoint 哈希、层列表、K 列表、表示协议和输出文件哈希。

## 8. 明确排除的流程

以下内容不属于 V2 几何主实验：

- 行为 prompt 与 demonstration；
- 翻译文本生成；
- 生成终止规则与重复抑制；
- fastText 或其他输出语言识别；
- 英语词汇泄露检测；
- chrF++ 行为质量指标；
- 人工盲审；
- 几何到行为的回归；
- 激活干预或因果声明。

这些内容应保留在独立的后续目录中，不得成为 `structure_v2` 正式入口的隐式依赖。

## 9. 当前实现状态与处置

### 9.1 已经可复用的内容

XGLM 已完成的 `prepare` 阶段包含可复用的：

- FLORES+ devtest 平行数据；
- checkpoint identity；
- mean-pool hidden states；
- 800 个正式语义 ID 的几何 CSV；
- 距离热图和 PCA 等描述性图形。

当前生成实验不会修改或污染上述 hidden states 和几何结果。

### 9.2 不进入正式论文的内容

XGLM 已生成的 4,000 条行为输出属于提前运行的协议诊断。其 `token_budget_rate=0.472`，且存在错误语言复制与重复续写，因此不能作为正式行为证据，也不需要为了 V2 几何主实验进行修复或重跑。

### 9.3 正式运行前必须完成的代码收敛

1. 建立独立的 `src/structure_v2/` 主入口；
2. 移除几何分析对行为任务文件的依赖；
3. 使用全部 804 个语义 ID 重算几何；
4. 将置信区间统一改为 semantic-ID bootstrap；
5. 建立 geometry-only 验证器和三模型比较器；
6. 将现有 `behavior_v2/generate.py`、`evaluate.py`、`annotation.py` 和 `associate.py` 移出正式主流程；
7. 在结构准入门槛满足前，不再运行 `generate`、`audit` 或行为 `analyze`。

## 10. 局限性与解释边界

- Latin 集中度可能同时受到文字系统、语系、训练数据规模和 tokenizer 覆盖率影响。
- 不同模型的绝对层号不可直接视为完全同构；跨模型比较应同时报告归一化层深。
- cosine 中心性可能受空间各向异性影响，因此 local-scaled 结果是必要稳健性检查。
- 同语义设计控制了句子内容，但不能自动控制各语言译文长度和 token 数差异。
- PCA 是二维压缩展示，不能替代高维距离、中心性和 hubness 统计。
- 即使英语结构优势跨模型复现，也只能支持观察性内部表征结论，不能直接推出生成行为或因果机制。

## 11. 下一步

1. 停止当前 XGLM 行为生成分支，不进行人工盲审；
2. 将现有 XGLM 几何结果单独导出并审查；
3. 完成 geometry-only `structure_v2` 代码收敛和测试；
4. 对 XGLM 使用全部 804 个语义 ID 离线复算；
5. 确认单模型结果与 manifest 无误后，再运行 Mistral 和 Aya；
6. 完成三模型复现判断后，依据冻结准入门槛决定是否另行设计行为关联实验。

## 12. 待冻结问题

正式三模型运行前还需要最终确认：

- paired bootstrap 是否同时配套零中心符号翻转检验；
- `latin_concentration` 与英语 K=`5` hubness excess 中哪一个作为最高优先级主结果；
- 行为实验准入是否要求两个预期阳性模型同时通过 raw 与 local-scaled；
- Mistral 负对照未呈现英语优势时，是作为必要特异性条件还是仅作解释性证据。

这些选择必须在审查 XGLM 正式 804-ID 几何结果之前冻结，避免结果驱动的门槛调整。
