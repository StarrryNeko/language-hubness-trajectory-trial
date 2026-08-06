"""Run a semantic-ID-split linear language probe for paper_v1."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
try:
    from tqdm import tqdm
except (ImportError, PermissionError):
    def tqdm(iterable, **_kwargs):
        return iterable

from common import load_config
from paper_common import (
    ensure_paper_dirs,
    load_hidden_dataset,
    load_or_create_splits,
    paper_settings,
    semantic_id_hash,
    write_manifest,
)


def semantic_split_masks(meta, split):
    ids = meta.sort_values("row_idx").id.astype(str).to_numpy()
    masks = {name: np.isin(ids, values) for name, values in split.items() if name in {"train", "validation", "test"}}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("every probe split must contain at least one semantic ID")
    if np.any(masks["train"] & masks["validation"]) or np.any(masks["train"] & masks["test"]):
        raise ValueError("semantic split leakage detected")
    return masks


def fit_probe(x_train, y_train, x_eval, c_value, seed):
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=float(c_value), solver="lbfgs", max_iter=1000, random_state=int(seed)
    )
    model.fit(scaler.transform(x_train), y_train)
    return model, scaler, model.predict(scaler.transform(x_eval))


def main():
    parser = argparse.ArgumentParser(description="paper_v1 semantic-ID-split language probe")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    settings = paper_settings(cfg)
    paths = ensure_paper_dirs(cfg)
    output = paths.metrics / "language_probe"
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_hidden_dataset(cfg)
    split_path = paths.splits / "probe_semantic_split.json"
    split = load_or_create_splits(
        split_path,
        dataset.semantic_ids,
        settings["seed"],
        settings["probe_split_ratios"],
    )
    masks = semantic_split_masks(dataset.meta, split)
    labels = dataset.row_language_indices
    c_values = [float(value) for value in cfg.get("paper_v1", {}).get("probe_c_values", [1.0])]
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("paper_v1.probe_c_values must contain positive values")
    n_permutations = int(cfg.get("paper_v1", {}).get("probe_permutations", 0))
    if n_permutations < 0:
        raise ValueError("paper_v1.probe_permutations must be non-negative")
    rng = np.random.default_rng(settings["seed"] + 301)
    score_records = []
    confusion_records = []
    n_layers = int(dataset.vectors.shape[1])
    for layer in tqdm(range(n_layers), desc="paper_v1 language probe"):
        x = np.asarray(dataset.vectors[:, layer, :], dtype=np.float32)
        best = None
        for c_value in c_values:
            model, scaler, prediction = fit_probe(
                x[masks["train"]], labels[masks["train"]],
                x[masks["validation"]], c_value, settings["seed"] + layer,
            )
            score = f1_score(labels[masks["validation"]], prediction, average="macro")
            candidate = (float(score), -c_value, c_value, model, scaler)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        _, _, selected_c, model, scaler = best
        test_prediction = model.predict(scaler.transform(x[masks["test"]]))
        macro_f1 = f1_score(labels[masks["test"]], test_prediction, average="macro")
        balanced = balanced_accuracy_score(labels[masks["test"]], test_prediction)

        permutation_scores = []
        for permutation in range(n_permutations):
            shuffled_labels = rng.permutation(labels[masks["train"]])
            perm_model = LogisticRegression(
                C=selected_c, solver="lbfgs", max_iter=1000,
                random_state=settings["seed"] + layer * 1000 + permutation,
            )
            perm_model.fit(scaler.transform(x[masks["train"]]), shuffled_labels)
            perm_prediction = perm_model.predict(scaler.transform(x[masks["test"]]))
            permutation_scores.append(
                f1_score(labels[masks["test"]], perm_prediction, average="macro")
            )
        empirical_p = (
            (1 + sum(value >= macro_f1 for value in permutation_scores))
            / (1 + len(permutation_scores))
            if permutation_scores else None
        )
        score_records.append({
            "representation": "mean_pool",
            "layer": layer,
            "layer_role": "embedding" if layer == 0 else "transformer_block_output",
            "macro_f1": float(macro_f1),
            "balanced_accuracy": float(balanced),
            "selected_c": selected_c,
            "validation_macro_f1": float(best[0]),
            "permutation_macro_f1_mean": (
                float(np.mean(permutation_scores)) if permutation_scores else None
            ),
            "permutation_macro_f1_std": (
                float(np.std(permutation_scores)) if permutation_scores else None
            ),
            "empirical_p_value": empirical_p,
            "chance_level": 1 / len(dataset.languages),
            "n_train_semantic_ids": len(split["train"]),
            "n_validation_semantic_ids": len(split["validation"]),
            "n_test_semantic_ids": len(split["test"]),
        })
        matrix = confusion_matrix(
            labels[masks["test"]], test_prediction,
            labels=np.arange(len(dataset.languages)), normalize=None,
        )
        for true_index, true_lang in enumerate(dataset.languages):
            for predicted_index, predicted_lang in enumerate(dataset.languages):
                confusion_records.append({
                    "representation": "mean_pool",
                    "layer": layer,
                    "true_lang": true_lang,
                    "predicted_lang": predicted_lang,
                    "count": int(matrix[true_index, predicted_index]),
                })

    pd.DataFrame(score_records).to_csv(output / "probe_scores.csv", index=False, encoding="utf-8")
    pd.DataFrame(confusion_records).to_csv(output / "probe_confusion.csv", index=False, encoding="utf-8")
    write_manifest(output / "probe_manifest.json", {
        "split_file": str(split_path),
        "split_unit": "semantic_id",
        "semantic_id_sha256": semantic_id_hash(dataset.semantic_ids),
        "standardizer_fit_scope": "train_only",
        "c_selection_scope": "validation_only",
        "test_used_for_selection": False,
        "classifier": "multinomial logistic regression (lbfgs)",
        "c_values": c_values,
        "label_permutations": n_permutations,
        "inference_mode": (
            "permutation_test" if n_permutations >= 199 else "effect_size_only"
        ),
        "primary_metric": "macro_f1",
        "claim_boundary": "linear decodability is not evidence of active model use",
        "output_files": ["probe_scores.csv", "probe_confusion.csv"],
    })
    validation = {
        "module": "11_language_probe",
        "status": "PASS",
        "checks": {
            "split_unit_semantic_id": True,
            "split_disjoint": True,
            "standardizer_train_only": True,
            "hyperparameter_selection_validation_only": True,
            "label_permutation_baseline_saved": n_permutations > 0,
            "inferential_p_value_claim_allowed": n_permutations >= 199,
        },
    }
    (paths.validation / "11_language_probe.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved paper_v1 language-probe outputs to {output}")


if __name__ == "__main__":
    main()
