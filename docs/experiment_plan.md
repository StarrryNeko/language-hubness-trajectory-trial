# Executable Discovery Plan

## Goal

Find whether non-English representations are pulled toward an English hub inside Qwen2.5-1.5B, and whether this process is gradual, abrupt, or corrected in later layers.

## Phase 1: Sentence-Level Signal Check

Use sentence-level mean hidden states. This is cheap and answers whether the signal exists.

Run:

```bash
python src/prepare_flores.py --config configs/qwen25_1_5b_mvp.json
python src/extract_hidden.py --config configs/qwen25_1_5b_mvp.json
python src/compute_metrics.py --config configs/qwen25_1_5b_mvp.json
python src/plot_trajectories.py --config configs/qwen25_1_5b_mvp.json
```

Inspect:

- `figures/english_drift_by_layer.png`
- `figures/english_hub_attraction_by_layer.png`
- `figures/language_centrality_by_layer.png`
- `metrics/trajectory_shapes.csv`

Expected discoveries:

- Chinese may show lower English drift than Hausa if Qwen preserves Chinese better.
- German may be close to English early because of script and language-family proximity.
- Hausa may show stronger English attraction or weaker late correction.

## Phase 2: Token-Level Pilot

Enable token export:

```json
"token_export": {
  "enabled": true,
  "max_samples_per_language": 5
}
```

Then rerun `extract_hidden.py`.

Use exported files in `hidden/token_samples/` for manual inspection.

Token-level questions:

- Which tokens drift first?
- Are punctuation and named entities natural hubs?
- Do content tokens and function tokens behave differently?
- Do final layers correct earlier English drift?

## Phase 3: Correction Layer Test

After identifying peak drift layers:

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

