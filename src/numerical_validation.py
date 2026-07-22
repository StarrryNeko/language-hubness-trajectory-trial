"""Fail-fast numerical validation shared by extraction and metric stages."""

import numpy as np


def require_finite(values, context):
    array = np.asarray(values)
    if not np.isfinite(array).all():
        bad = int(array.size - np.isfinite(array).sum())
        raise ValueError(f"{context}: found {bad} non-finite value(s) (NaN/Inf)")
    return array


def require_nonzero_row_norms(values, context, minimum=1e-12):
    array = require_finite(values, context)
    if array.ndim == 0:
        raise ValueError(f"{context}: expected at least one vector dimension")
    norms = np.linalg.norm(array, axis=-1)
    require_finite(norms, f"{context} norms")
    bad = np.argwhere(norms <= minimum)
    if len(bad):
        first = tuple(int(value) for value in bad[0])
        raise ValueError(
            f"{context}: found {len(bad)} zero/near-zero vector norm(s); "
            f"first index={first}, minimum={minimum}"
        )
    return array


def validate_representation_array(values, expected_rows, context):
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(
            f"{context}: representation array must be 3D "
            f"(rows, layers, hidden_dim), got shape={array.shape}"
        )
    if array.shape[0] != int(expected_rows):
        raise ValueError(
            f"{context}: row count {array.shape[0]} does not match metadata rows {expected_rows}"
        )
    if array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(f"{context}: layers and hidden_dim must be positive, got shape={array.shape}")
    require_finite(array, context)
    return array


def validate_similarity_matrix(values, context, atol=1e-6):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{context}: similarity matrix must be square, got shape={matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{context}: similarity matrix must contain at least two languages")
    require_finite(matrix, context)
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=atol):
        error = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(f"{context}: similarity matrix is not symmetric; max_error={error}")
    require_finite(np.diag(matrix), f"{context} diagonal")
    return matrix
