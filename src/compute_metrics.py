import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import ensure_dirs, l2_normalize, load_config


def classify_curve(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return "too_short"

    start = values[0]
    end = values[-1]
    peak_idx = int(np.argmax(values))
    trough_idx = int(np.argmin(values))
    peak = float(values[peak_idx])
    trough = float(values[trough_idx])
    delta = end - start
    amplitude = peak - trough

    if amplitude < 0.05:
        return "stable"
    if delta > 0.10 and peak_idx < len(values) * 0.35:
        return "early_jump"
    if delta > 0.10 and peak_idx > len(values) * 0.65:
        return "late_jump"
    if delta > 0.10:
        return "gradual_increase"
    if peak_idx > len(values) * 0.25 and peak_idx < len(values) * 0.75 and end < peak - 0.05:
        return "inverted_u"
    if trough_idx > len(values) * 0.25 and trough_idx < len(values) * 0.75 and end > trough + 0.05:
        return "u_shaped"
    return "mixed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)

    hidden_path = Path(paths["hidden"]) / "sentence_layer_means.npy"
    meta_path = Path(paths["hidden"]) / "metadata.csv"
    vectors = np.load(hidden_path)
    meta = pd.read_csv(meta_path)

    english_lang = cfg["metrics"].get("english_language", "en")
    k = int(cfg["metrics"].get("nearest_neighbors_k", 10))

    n_samples, n_layers, _ = vectors.shape
    langs = meta["lang"].to_numpy()
    unique_langs = sorted(meta["lang"].unique())

    sample_records = []
    summary_records = []
    centrality_records = []

    for layer in tqdm(range(n_layers), desc="Computing layer metrics"):
        x = l2_normalize(vectors[:, layer, :], axis=1)

        centroids = {}
        for lang in unique_langs:
            centroids[lang] = l2_normalize(x[langs == lang].mean(axis=0, keepdims=True), axis=1)[0]
        en_centroid = centroids[english_lang]

        sim = x @ x.T
        np.fill_diagonal(sim, -np.inf)
        topk_idx = np.argpartition(-sim, kth=min(k, n_samples - 1), axis=1)[:, :k]

        neighbor_counts = {lang: 0 for lang in unique_langs}
        for row_neighbors in topk_idx:
            for idx in row_neighbors:
                neighbor_counts[langs[idx]] += 1

        total_neighbor_slots = n_samples * k
        for lang in unique_langs:
            centrality_records.append(
                {
                    "layer": layer,
                    "lang": lang,
                    "centrality_count": neighbor_counts[lang],
                    "centrality_rate": neighbor_counts[lang] / max(total_neighbor_slots, 1),
                }
            )

        for i in range(n_samples):
            own_lang = langs[i]
            own_centroid = centroids[own_lang]
            drift_en = float(x[i] @ en_centroid - x[i] @ own_centroid)
            eha = float(np.mean(langs[topk_idx[i]] == english_lang))
            sample_records.append(
                {
                    "row_idx": int(meta.loc[i, "row_idx"]),
                    "id": meta.loc[i, "id"],
                    "lang": own_lang,
                    "layer": layer,
                    "drift_en": drift_en,
                    "english_hub_attraction": eha,
                    "num_tokens": int(meta.loc[i, "num_tokens"]),
                }
            )

        layer_df = pd.DataFrame(sample_records[-n_samples:])
        for lang in unique_langs:
            lang_df = layer_df[layer_df["lang"] == lang]
            summary_records.append(
                {
                    "layer": layer,
                    "lang": lang,
                    "mean_drift_en": lang_df["drift_en"].mean(),
                    "mean_english_hub_attraction": lang_df["english_hub_attraction"].mean(),
                    "std_drift_en": lang_df["drift_en"].std(),
                    "std_english_hub_attraction": lang_df["english_hub_attraction"].std(),
                    "n": len(lang_df),
                }
            )

    sample_df = pd.DataFrame(sample_records)
    summary_df = pd.DataFrame(summary_records)
    centrality_df = pd.DataFrame(centrality_records)

    trajectory_records = []
    for row_idx, group in sample_df.groupby("row_idx"):
        group = group.sort_values("layer")
        trajectory_records.append(
            {
                "row_idx": row_idx,
                "lang": group["lang"].iloc[0],
                "drift_shape": classify_curve(group["drift_en"].to_numpy()),
                "eha_shape": classify_curve(group["english_hub_attraction"].to_numpy()),
                "max_drift_layer": int(group.loc[group["drift_en"].idxmax(), "layer"]),
                "max_eha_layer": int(group.loc[group["english_hub_attraction"].idxmax(), "layer"]),
                "final_minus_peak_drift": float(group["drift_en"].iloc[-1] - group["drift_en"].max()),
            }
        )
    trajectory_df = pd.DataFrame(trajectory_records)

    sample_df.to_csv(Path(paths["metrics"]) / "sample_layer_metrics.csv", index=False, encoding="utf-8")
    summary_df.to_csv(Path(paths["metrics"]) / "layer_summary.csv", index=False, encoding="utf-8")
    centrality_df.to_csv(Path(paths["metrics"]) / "language_centrality.csv", index=False, encoding="utf-8")
    trajectory_df.to_csv(Path(paths["metrics"]) / "trajectory_shapes.csv", index=False, encoding="utf-8")

    print(f"Saved metrics to {paths['metrics']}")


if __name__ == "__main__":
    main()

