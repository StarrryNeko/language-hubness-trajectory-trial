# Language Hubness Trajectory

This project is a small, executable MVP for studying layer-wise language centralization in multilingual LLMs.

Core idea:

> Track whether non-English representations drift toward an English hub across model layers, and identify absorption / peak / correction layers.

The first target model is `Qwen/Qwen2.5-1.5B` on a CUDA 12.1 PyTorch environment.

## Research Flow

1. Prepare a small parallel multilingual dataset from FLORES-200.
2. Extract layer-wise hidden states from Qwen2.5-1.5B.
3. Validate that the extracted sentence representation carries semantics and that cosine similarity behaves normally.
4. Compute three core signals:
   - `AlignmentGain`: parallel-sentence cosine minus a shuffled semantic baseline.
   - `Anchor Specificity`: whether English is more special than Chinese, German, or Hausa pseudo-anchors.
   - `Cross-lingual Hub Attraction`: which language occupies balanced cross-language kNN slots.
5. Measure language re-separation with neighborhood purity and centroid separation.
6. Plot all validation and research trajectories.

## Recommended Cloud Environment

- GPU: RTX 3090 24GB or better
- RAM: 32GB or better
- OS: Linux / Ubuntu preferred
- Python: 3.10 or 3.11
- CUDA: 12.1

Install PyTorch for CUDA 12.1 first:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install project dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

From this project folder:

```bash
python src/run_pilot.py --config configs/qwen25_1_5b_mvp.json
```

The unified runner stops immediately if any stage fails. To rerun metrics and
figures without loading the model again:

```bash
python src/run_pilot.py --config configs/qwen25_1_5b_mvp.json --skip-prepare --skip-extract
```

The equivalent individual commands are:

```bash
python src/prepare_flores.py --config configs/qwen25_1_5b_mvp.json
python src/extract_hidden.py --config configs/qwen25_1_5b_mvp.json
python src/compute_metrics.py --config configs/qwen25_1_5b_mvp.json
python src/plot_trajectories.py --config configs/qwen25_1_5b_mvp.json
```

If the cloud machine cannot reach HuggingFace, first try a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

If the machine has no network at all, create a tiny toy dataset to test the rest of the pipeline:

```bash
python src/make_toy_data.py --config configs/qwen25_1_5b_mvp.json
python src/extract_hidden.py --config configs/qwen25_1_5b_mvp.json
python src/compute_metrics.py --config configs/qwen25_1_5b_mvp.json
python src/plot_trajectories.py --config configs/qwen25_1_5b_mvp.json
```

Outputs will be written to:

```text
outputs/qwen25_1_5b_mvp/
  data/
  hidden/
  metrics/
  figures/
```

The extraction step saves both `last_token` (primary) and `mean_pool` sentence
representations in one model pass. The metric step performs validation and the
pilot analysis together, so there is no second model run.

After `compute_metrics.py`, inspect the terminal report or:

```text
outputs/qwen25_1_5b_mvp/metrics/validation_report.txt
```

The main figures are:

```text
alignment_gain_by_layer.png
similarity_sanity_check.png
semantic_retrieval_recall1.png
anchor_specificity_by_layer.png
english_hub_attraction_by_layer.png
language_neighborhood_purity.png
centroid_separation_by_layer.png
```

## First Discovery Questions

Use the first plots to answer:

1. Is parallel-sentence similarity reliably above the shuffled baseline?
2. Is English specificity higher than all pseudo-anchor languages?
3. Does English occupy more cross-language kNN slots than the uniform baseline?
4. Does language neighborhood purity fall in middle layers and recover late?
5. Do last-token and mean-pool representations support the same broad pattern?

## Important Notes

- The primary representation is the final non-padding token because it has seen the full left context. Mean pooling is saved as a robustness check.
- Hidden states are contextual representations, not pure semantic outputs. Parallel retrieval and shuffled baselines test whether they carry usable sentence semantics.
- Keep the first run small: 100-300 sentences per language.
- Do not save every token from every sentence at first; hidden state files grow quickly.

## Interpretation Guardrails

- High raw cosine alone is not semantic alignment; require positive `AlignmentGain` and above-chance parallel retrieval.
- Positive English specificity is proximity evidence, not hubness; require English kNN attraction above the balanced baseline.
- Falling English attraction alone is not re-separation; require late recovery of language neighborhood purity, preferably with centroid separation.
- Do not call a late change a correction mechanism without a later intervention experiment.

## Suggested Paper Angle

Working title:

> Hook-Based Layer Trajectory Analysis for Detecting Language Centralization in Multilingual LLMs

Contribution:

1. A hook-style framework for extracting layer-wise multilingual representations.
2. Controlled sentence-level measures separating semantic alignment, English specificity, and hubness.
3. Evidence for or against late language re-separation.
4. Pooling and random-baseline validation of the observed trajectories.
