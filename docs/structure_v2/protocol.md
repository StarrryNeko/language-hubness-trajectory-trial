# structure_v2：纯几何主实验

`structure_v2` 不读取 prompt、demonstration、generation、人工标注或行为任务文件。它直接读取每个模型已有的
804-ID `mean_pool_v1` hidden states，在冻结主层和相邻层内计算：

- Latin concentration；
- 英语相对其他 Latin 语言的中心性 excess；
- K=`3/5/10` 的 raw 与 local-scaled reverse-kNN hubness excess。

所有候选限制在同一语义 ID 内，不确定性统一使用语义 ID bootstrap。单模型入口：

```bash
python src/run_structure_v2.py --config configs/behavior_association_v3/xglm_1b7.json
```
