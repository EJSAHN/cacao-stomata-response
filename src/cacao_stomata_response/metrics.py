from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    n_x = len(x)
    n_y = len(y)
    if n_x < 2 or n_y < 2:
        return np.nan
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    if pooled_var <= 0:
        return np.nan
    return (np.mean(y) - np.mean(x)) / np.sqrt(pooled_var)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    d_value = cohen_d(x, y)
    if np.isnan(d_value):
        return np.nan
    n_total = len(x) + len(y)
    if n_total <= 3:
        return np.nan
    correction = 1.0 - (3.0 / (4.0 * n_total - 9.0))
    return correction * d_value


def vector_norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = vector_norm(a)
    norm_b = vector_norm(b)
    if norm_a == 0 or norm_b == 0:
        return np.nan
    return float(np.dot(a, b) / (norm_a * norm_b))


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = cosine_similarity(a, b)
    if np.isnan(cosine):
        return np.nan
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def project_vector(response: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    axis_norm = vector_norm(axis)
    response_norm = vector_norm(response)
    if axis_norm == 0 or response_norm == 0:
        return np.nan, np.nan
    axis_unit = axis / axis_norm
    parallel = float(np.dot(response, axis_unit))
    residual = response - parallel * axis_unit
    perpendicular = float(np.linalg.norm(residual))
    return parallel, perpendicular


def one_dimensional_wasserstein(control: np.ndarray, infected: np.ndarray) -> float:
    if len(control) == 0 or len(infected) == 0:
        return np.nan
    return float(wasserstein_distance(control, infected))


def gaussian_w2_distance(
    control_values: np.ndarray,
    infected_values: np.ndarray,
    regularization: float = 1e-6,
) -> float:
    if control_values.size == 0 or infected_values.size == 0:
        return np.nan

    control_values = np.asarray(control_values, dtype=float)
    infected_values = np.asarray(infected_values, dtype=float)

    if control_values.ndim == 1:
        control_values = control_values.reshape(-1, 1)
    if infected_values.ndim == 1:
        infected_values = infected_values.reshape(-1, 1)

    mean_control = control_values.mean(axis=0)
    mean_infected = infected_values.mean(axis=0)
    cov_control = np.cov(control_values, rowvar=False)
    cov_infected = np.cov(infected_values, rowvar=False)

    if np.ndim(cov_control) == 0:
        cov_control = np.array([[float(cov_control)]])
    if np.ndim(cov_infected) == 0:
        cov_infected = np.array([[float(cov_infected)]])

    dim = cov_control.shape[0]
    cov_control = cov_control + np.eye(dim) * regularization
    cov_infected = cov_infected + np.eye(dim) * regularization

    mean_term = np.sum((mean_control - mean_infected) ** 2)
    sqrt_cov_control = sqrtm(cov_control)
    middle = sqrt_cov_control @ cov_infected @ sqrt_cov_control
    sqrt_middle = sqrtm(middle)

    if np.iscomplexobj(sqrt_middle):
        sqrt_middle = np.real_if_close(sqrt_middle, tol=1000)
    if np.iscomplexobj(sqrt_cov_control):
        sqrt_cov_control = np.real_if_close(sqrt_cov_control, tol=1000)

    trace_term = np.trace(cov_control + cov_infected - 2.0 * sqrt_middle)
    value = float(np.sqrt(max(mean_term + trace_term, 0.0)))
    return value


def summarize_series(values: Iterable[float]) -> dict[str, float]:
    array = pd.Series(list(values), dtype=float).dropna()
    if array.empty:
        return {"mean": np.nan, "sd": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else np.nan,
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(len(array)),
    }
