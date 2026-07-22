"""Shared strict rules for layer-wise English hubness evidence."""

import numpy as np


REQUIRED_EVIDENCE_METRICS = (
    "k_occurrence_excess",
    "centrality_advantage",
    "rank_percentile_advantage",
    "medoid_rate_excess",
)

MODEL_STATUSES = {
    "INVALID",
    "NOT_SUPPORTED",
    "REPRESENTATION_SENSITIVE",
    "ROBUST",
}


def validate_evidence_grid(frame, expected_layers=None):
    required_columns = {"layer", "metric", "mean", "ci_lower", "ci_upper"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"evidence is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("evidence grid is empty")
    work = frame.copy()
    work["layer"] = work["layer"].astype(int)
    work = work[work["metric"].isin(REQUIRED_EVIDENCE_METRICS)]
    duplicates = work.duplicated(["layer", "metric"], keep=False)
    if duplicates.any():
        pairs = work.loc[duplicates, ["layer", "metric"]].drop_duplicates().to_dict("records")
        raise ValueError(f"evidence contains duplicate (layer, metric) records: {pairs[:5]}")
    layers = sorted(int(layer) for layer in work["layer"].unique())
    if expected_layers is None:
        if not layers:
            raise ValueError("evidence contains none of the four required metrics")
        expected_layers = list(range(layers[-1] + 1))
    else:
        expected_layers = sorted(int(layer) for layer in expected_layers)
    expected_pairs = {
        (layer, metric) for layer in expected_layers for metric in REQUIRED_EVIDENCE_METRICS
    }
    actual_pairs = set(zip(work["layer"], work["metric"]))
    missing_pairs = expected_pairs - actual_pairs
    extra_layers = set(layers) - set(expected_layers)
    if missing_pairs or extra_layers:
        preview = sorted(missing_pairs)[:5]
        raise ValueError(
            f"evidence grid is incomplete; missing (layer, metric)={preview}, "
            f"unexpected_layers={sorted(extra_layers)}"
        )
    numeric = work[["mean", "ci_lower", "ci_upper"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("evidence contains non-finite mean/CI values")
    return expected_layers


def joint_positive_layers(frame, expected_layers=None):
    layers = validate_evidence_grid(frame, expected_layers=expected_layers)
    work = frame[frame["metric"].isin(REQUIRED_EVIDENCE_METRICS)].copy()
    work["layer"] = work["layer"].astype(int)
    return [
        layer
        for layer in layers
        if bool((work.loc[work["layer"] == layer, "ci_lower"] > 0).all())
    ]


def max_consecutive_layers(layers):
    ordered = sorted(set(int(layer) for layer in layers))
    best = current = 0
    previous = None
    for layer in ordered:
        current = current + 1 if previous is not None and layer == previous + 1 else 1
        best = max(best, current)
        previous = layer
    return best


def classify_model_status(primary_layers, breadth_layers, eos_layers, density_layers, min_run):
    min_run = int(min_run)
    if min_run < 1:
        raise ValueError("min_run must be at least 1")
    primary_joint = sorted(set(map(int, primary_layers)) & set(map(int, breadth_layers)))
    primary_run = max_consecutive_layers(primary_joint)
    eos_run = max_consecutive_layers(eos_layers)
    density_run = max_consecutive_layers(density_layers)
    if primary_run < min_run:
        status = "NOT_SUPPORTED"
    elif eos_run >= min_run and density_run >= min_run:
        status = "ROBUST"
    else:
        status = "REPRESENTATION_SENSITIVE"
    return {
        "status": status,
        "primary_joint_layers": primary_joint,
        "primary_joint_longest_run": primary_run,
        "eos_joint_layers": sorted(set(map(int, eos_layers))),
        "eos_joint_longest_run": eos_run,
        "density_joint_layers": sorted(set(map(int, density_layers))),
        "density_joint_longest_run": density_run,
        "min_consecutive_layers": min_run,
    }


def validate_model_status_payload(summary):
    status = summary.get("model_status")
    if status not in MODEL_STATUSES:
        raise ValueError(f"validation model_status is missing or invalid: {status!r}")
    evidence = summary.get("joint_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("validation joint_evidence is missing or invalid")
    if status == "INVALID":
        return status
    required = {
        "status", "primary_joint_longest_run", "eos_joint_longest_run",
        "density_joint_longest_run", "min_consecutive_layers",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(f"validation joint_evidence is missing fields: {sorted(missing)}")
    if evidence["status"] != status:
        raise ValueError("validation model_status disagrees with joint_evidence.status")
    minimum = int(evidence["min_consecutive_layers"])
    primary = int(evidence["primary_joint_longest_run"])
    eos = int(evidence["eos_joint_longest_run"])
    density = int(evidence["density_joint_longest_run"])
    if minimum < 1 or min(primary, eos, density) < 0:
        raise ValueError("validation joint-run lengths and minimum are out of range")
    expected = (
        "NOT_SUPPORTED" if primary < minimum else
        "ROBUST" if eos >= minimum and density >= minimum else
        "REPRESENTATION_SENSITIVE"
    )
    if status != expected:
        raise ValueError(
            f"validation status is inconsistent with joint runs: reported={status}, expected={expected}"
        )
    return status
