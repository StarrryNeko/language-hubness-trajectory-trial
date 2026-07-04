# Language Hubness Trajectory

This project is a small, executable MVP for studying layer-wise language centralization in multilingual LLMs.

Core idea:

> Track whether non-English representations drift toward an English hub across model layers, and identify absorption / peak / correction layers.

The first target model is `Qwen/Qwen2.5-1.5B` on a CUDA 12.1 PyTorch environment.

## Research Flow

1. Prepare a small parallel multilingual dataset from FLORES-200.
2. Extract layer-wise hidden states from Qwen2.5-1.5B.
3. Compute two discovery metrics:
   - `English Drift`: whether a sample is closer to the English centroid than its own-language centroid.
   - `English Hub Attraction`: how many nearest neighbors of a sample are English.
4. Plot layer-wise trajectories.
5. Inspect whether trajectories show gradual drift, threshold jumps, or late correction.

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

## First Discovery Questions

Use the first plots to answer:

1. Does Chinese drift toward the English centroid in middle layers?
2. Does Hausa / Swahili drift earlier or more strongly than Chinese?
3. Is German closer to English from the beginning, or does it also show a jump?
4. Does the drift decrease in the final layers, suggesting a correction layer?
5. Does Qwen2.5-1.5B show weaker English centralization than expected because of its Chinese-English training background?

## Important Notes

- The first version uses sentence-level mean hidden states. This is intentional: it is much cheaper and is enough to test whether the signal exists.
- After sentence-level trends appear, enable token-level exports in the config and inspect token trajectories.
- Keep the first run small: 100-300 sentences per language.
- Do not save every token from every sentence at first; hidden state files grow quickly.

## Suggested Paper Angle

Working title:

> Hook-Based Layer Trajectory Analysis for Detecting Language Centralization in Multilingual LLMs

Contribution:

1. A hook-style framework for extracting layer-wise multilingual representations.
2. Dynamic trajectory metrics for English drift and English hub attraction.
3. Evidence of whether language centralization forms gradually, abruptly, or is corrected in late layers.
4. A foundation for later lightweight correction at absorption / peak layers.
