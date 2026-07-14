# Executable Sentence-Level Pilot Plan

## Goal

Distinguish cross-lingual semantic alignment, English-specific proximity, English hubness, and late language re-separation inside Qwen2.5-1.5B.

## Phase 1: Sentence-Level Signal Check

Extract last-token and mean-pool sentence representations in the same forward pass. Validation and research metrics are computed together after extraction.

Run:

```bash
python src/prepare_flores.py --config configs/qwen25_1_5b_mvp.json
python src/extract_hidden.py --config configs/qwen25_1_5b_mvp.json
python src/compute_metrics.py --config configs/qwen25_1_5b_mvp.json
python src/plot_trajectories.py --config configs/qwen25_1_5b_mvp.json
```

Inspect:

- `figures/alignment_gain_by_layer.png`
- `figures/similarity_sanity_check.png`
- `figures/semantic_retrieval_recall1.png`
- `figures/anchor_specificity_by_layer.png`
- `figures/english_hub_attraction_by_layer.png`
- `figures/language_neighborhood_purity.png`
- `figures/centroid_separation_by_layer.png`
- `metrics/validation_report.txt`
- `metrics/research_summary.txt`
- `metrics/pooling_robustness_summary.csv`
- `figures/re_separation_strength.png`
- `figures/pooling_robustness_summary.png`

Decision rules:

- Semantic evidence requires positive AlignmentGain and above-chance parallel retrieval.
- English-specific proximity requires English to beat the rotated pseudo-anchors.
- English hubness requires English neighbor share above the balanced candidate baseline.
- Re-separation requires late recovery of neighborhood purity; centroid separation is supporting evidence.

Bootstrap confidence intervals are computed over semantic samples. For optional
kNN robustness after the main run:

```bash
python src/sweep_k.py --config configs/qwen25_1_5b_mvp.json --k-values 5 10 20
```

## Optional Phase 2: Token-Level Diagnostics

Enable token export:

```json
"token_export": {
  "enabled": true,
  "max_samples_per_language": 5
}
```

Then rerun `extract_hidden.py`.

Use exported files in `hidden/token_samples/` for manual inspection.

Only run this after a stable sentence-level result. Token export is diagnostic and is not part of the core pilot.

- Which tokens drift first?
- Are punctuation and named entities natural hubs?
- Do content tokens and function tokens behave differently?
- Do final layers correct earlier English drift?

## Optional Phase 3: Intervention

Only after re-separation is replicated in another model:

1. Pick the peak layer from `layer_summary.csv`.
2. Apply language-centering:
   - subtract English centroid direction;
   - optionally add own-language centroid direction.
3. Recompute metrics.

Success criteria:

- English hub attraction decreases.
- Non-English retrievability increases.
- Semantic similarity does not collapse.

## Phase 4: Scale Up

After the MVP works:

- Increase sample size to 500-1000 per language.
- Add languages: Arabic, Hindi, Swahili, Japanese.
- Add models: Llama 3.1 8B, Gemma 2 2B/9B, Aya Expanse 8B.
- Compare whether Qwen has weaker Chinese absorption than English-centric models.
