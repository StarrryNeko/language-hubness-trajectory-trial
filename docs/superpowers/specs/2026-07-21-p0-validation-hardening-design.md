# P0 Validation Hardening Design

## Objective

Make the same-semantics multilingual hubness pipeline fail safely on invalid numerical geometry, rerun XGLM in FP32, and replace per-metric replication claims with joint, control-aware evidence states.

## Scope

This change implements only the P0 items recorded in `7.21任务清单/02_本周任务清单.md`:

1. finite-value and norm checks during extraction;
2. finite-value, similarity, and kNN mass-conservation checks during metrics;
3. FP32 compute and storage for XGLM;
4. same-layer joint evidence intervals;
5. stricter per-model and cross-model verdicts;
6. tests for each new failure and classification path.

Language rotation, tokenization controls, additional sampling, and chart redesign beyond invalid-gap handling remain out of scope.

## Chosen Approach

Use strict fail-fast validation. Invalid numerical values are never filtered, skipped, or converted into partial metrics. A model with incomplete or non-finite evidence is `INVALID` and cannot contribute to a replication verdict.

Rejected alternatives:

- Skipping invalid layers would make trajectories incomparable and could hide numerical failures.
- Automatically retrying with a different dtype would make model conditions implicit and weaken reproducibility.

## Extraction Design

`src/extract_hidden.py` will expose small validation helpers so tests can exercise numerical checks without downloading a model.

For every row and hidden layer:

- the full hidden tensor must be finite;
- mean-pool and sentinel-EOS vectors must be finite;
- both representation vectors must have a finite norm greater than a small positive threshold;
- vectors must remain finite after conversion to the configured storage dtype.

Errors will include model name, representation, row index, semantic ID, language, and layer. No `.npy`, metadata, or extraction manifest will be written after a validation failure.

`configs/xglm_1b7_24lang.json` will override both `dtype` and `storage_dtype` to `float32`. This makes the diagnostic rerun explicit and prevents float16 storage conversion from reintroducing overflow.

## Metric Geometry Design

`src/compute_metrics.py` will validate at multiple boundaries:

- loaded representation arrays must have rank 3, expected row count, positive layer/dimension sizes, and finite values;
- each semantic-group layer slice must be finite and contain no zero-norm vector;
- cosine and local-scaled similarity matrices must be square, finite, symmetric, and have a finite diagonal;
- `group_statistics` must receive `1 <= k < n`;
- each fractional top-k row must sum to k within tolerance;
- total occurrence mass must equal `n * k`;
- occurrence, centrality, percentile, and medoid outputs must be finite;
- medoid mass must sum to one.

`bootstrap_mean_ci` will no longer silently remove non-finite observations. It will raise with an explicit error, because partial deletion changes the semantic-group denominator and can fabricate valid-looking confidence intervals.

## Joint Evidence Design

A shared helper will calculate evidence by layer for the required metrics:

- `k_occurrence_excess`;
- `centrality_advantage`;
- `rank_percentile_advantage`;
- `medoid_rate_excess`.

A layer is jointly positive only when all four rows exist exactly once and all four `ci_lower` values are finite and greater than zero. A consecutive interval is computed only over actual ordered layer indices; missing layer numbers break a run.

Primary support additionally requires source breadth on the same layers:

- supported source languages at least half of all non-English source languages;
- at least four source scripts;
- at least three supported non-Latin languages.

The validation output will record:

- jointly positive layers;
- jointly positive plus broad-source layers;
- longest actual consecutive run;
- configured minimum run length.

## Control-Aware Status Design

Each model receives one of these statuses:

- `INVALID`: missing files, missing layers/metrics, duplicate evidence rows, or non-finite values;
- `NOT_SUPPORTED`: primary mean-pool/cosine evidence does not pass the joint same-layer and breadth rule;
- `REPRESENTATION_SENSITIVE`: primary evidence passes, but sentinel-EOS or local-scaled evidence does not pass the joint run rule;
- `ROBUST`: primary evidence with breadth passes, and both sentinel-EOS/cosine and mean-pool/local-scaled cosine pass the joint run rule.

The k-sweep remains a separately reported control. It cannot upgrade a model whose primary or representation controls fail.

Cross-model replication requires at least two `ROBUST` models. `REPRESENTATION_SENSITIVE` models may be reported as conditional evidence but do not produce `REPLICATED`.

## Comparison Design

`src/compare_models.py` will validate every model before summary calculation:

- all four required metrics are present for every expected layer;
- there is exactly one row per `(layer, metric)` under the selected representation and similarity method;
- `mean`, `ci_lower`, and `ci_upper` are finite;
- layer indices are complete from zero through the maximum layer.

Invalid models remain visible in the verdict with their reason, but do not contribute trajectories, AUC summaries, or replication counts. Plotting will use explicit per-model lines with NaN-preserving gaps if diagnostic rows are ever plotted; it will not allow seaborn aggregation to bridge missing values.

The comparison verdict will include per-model status, joint layers, longest run, control results, invalid reasons, and the strict cross-model replication status.

## Compatibility and Output

Existing CSV files remain available. New joint-evidence and status fields are additive. Existing `run_pilot.py` and `run_model_suite.py --resume` entry points remain unchanged.

Because the resolved XGLM configuration changes, `--resume` will not treat an older FP16 XGLM run as complete.

## Testing Strategy

Tests will be written before production changes and observed failing for the intended reason.

Unit tests will cover:

- rejection of NaN/Inf hidden vectors;
- rejection of zero-norm vectors;
- rejection of NaN similarity matrices;
- kNN row and total mass conservation;
- rejection rather than filtering of non-finite bootstrap values;
- joint intervals requiring all four metrics on the same layers;
- missing layer numbers breaking consecutive runs;
- disjoint per-metric runs not qualifying;
- invalid models excluded from replication;
- two robust models required for `REPLICATED`.

The existing synthetic pipeline smoke test will run after unit tests. It must still generate metrics, validation, figures, and comparison outputs without model downloads.

## Success Criteria

- XGLM resolves to float32 compute and float32 storage.
- No non-finite or zero-norm representation can enter metrics.
- No invalid kNN graph can enter aggregate evidence.
- Four disjoint per-metric runs cannot qualify as joint hubness.
- A model with missing or non-finite evidence is `INVALID`.
- Cross-model `REPLICATED` requires at least two `ROBUST` models.
- Unit tests, synthetic smoke test, and Python compilation all pass.
