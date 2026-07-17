# Language Hubness Trajectory

This project is a small, executable MVP for studying layer-wise language centralization in multilingual LLMs.

Core idea:

> Track whether non-English representations drift toward English across model layers, distinguish
> neighbor-language attraction from point-level hubness, and test late language re-separation.

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

To print the source sentence and terminal-token audit for every saved hidden-state row during extraction:

```bash
python src/run_pilot.py --config configs/qwen25_1_5b_mvp.json --show-sentences all
```

The unified runner stops immediately if any stage fails. To rerun metrics and
figures without loading the model again:

```bash
python src/run_pilot.py --config configs/qwen25_1_5b_mvp.json --skip-prepare --skip-extract
```

Use the skip flags only when `data/dataset_manifest.json`, `extraction_manifest.json`, and the
config snapshot belong to the same run. An official rerun should not reuse unmanifested toy or
copied output directories.

The equivalent individual commands are:

```bash
python src/prepare_flores.py --config configs/qwen25_1_5b_mvp.json
python src/extract_hidden.py --config configs/qwen25_1_5b_mvp.json
python src/compute_metrics.py --config configs/qwen25_1_5b_mvp.json
python src/plot_trajectories.py --config configs/qwen25_1_5b_mvp.json
python src/sweep_k.py --config configs/qwen25_1_5b_mvp.json --k-values 5 10 20
python src/run_validations.py --config configs/qwen25_1_5b_mvp.json
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
  validation/
```

The extraction step saves five representations in one model pass:

- `last_token`: the original final non-padding token; this often is punctuation.
- `last_content_token`: the final token containing a Unicode letter, number, or mark.
- `shared_sentinel`: one identical EOS/sentinel token appended to every language after the sentence.
- `mean_pool`: the original all-token mean, excluding the appended sentinel.
- `content_mean_pool`: a mean over content-bearing tokens only.

`hidden/metadata.csv` and `hidden/hidden_state_sentence_index.jsonl` map every vector row back to
its source sentence, complete token sequence, original last token, last content token, and sentinel.
Use the inspector without loading the model again:

```bash
python src/inspect_hidden_states.py \
  --config configs/qwen25_1_5b_mvp.json \
  --rows 0,200,400,600 --layers 0,16,28 --show-token-sequence
```

After the unified run, begin with:

```text
outputs/qwen25_1_5b_mvp/validation/validation_summary.md
```

The numbered files in `validation/` preserve each validation question separately: dataset,
sentence/token audit, hidden integrity, semantics, English specificity, attraction/hubness,
re-separation, representation robustness, and k robustness. Each report records the method,
evidence, interpretation, and any required action.

Bootstrap confidence intervals, automatic claim verdicts, and pooling summaries
are written to:

```text
metrics/research_summary.txt
metrics/re_separation_summary.csv
metrics/pooling_robustness_summary.csv
figures/re_separation_strength.png
figures/pooling_robustness_summary.png
```

Run the optional kNN robustness sweep without reloading the model:

```bash
python src/sweep_k.py --config configs/qwen25_1_5b_mvp.json --k-values 5 10 20
```

This writes isolated `metrics/k5`, `metrics/k10`, and `metrics/k20`
directories plus `figures/k_robustness_summary.png`.

The main figures are:

```text
alignment_gain_by_layer.png
similarity_sanity_check.png
semantic_retrieval_recall1.png
anchor_specificity_by_layer.png
english_specificity_contrasts.png
english_hub_attraction_by_layer.png
hubness_occurrence_by_layer.png
language_neighborhood_purity.png
centroid_separation_by_layer.png
re_separation_strength.png
pooling_robustness_summary.png
```

## First Discovery Questions

Use the first plots to answer:

1. Is parallel-sentence similarity reliably above the shuffled baseline?
2. Is English specificity higher than all pseudo-anchor languages?
3. Does English occupy more cross-language kNN slots than the uniform baseline?
4. Does language neighborhood purity fall in middle layers and recover late?
5. Do last-token and mean-pool representations support the same broad pattern?

## Important Notes

- The raw final token remains the historical baseline because a causal decoder's final position has seen
  the full left context. The updated Qwen pilot uses `shared_sentinel` as its configured primary
  representation and treats raw final-token results as a control. Final-content-token and shared-sentinel
  results must agree before a terminal-token-sensitive claim is treated as robust.
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
