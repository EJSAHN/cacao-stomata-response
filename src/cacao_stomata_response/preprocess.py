from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .feature_sets import ALL_FEATURES


@dataclass(frozen=True)
class ScalingReference:
    means: dict[str, float]
    sds: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for feature in ALL_FEATURES:
            rows.append(
                {
                    "feature": feature,
                    "control_mean": self.means[feature],
                    "control_sd": self.sds[feature],
                }
            )
        return pd.DataFrame(rows)


def clean_stimulus_data(frame: pd.DataFrame, control_label: str) -> pd.DataFrame:
    data = frame.copy()

    string_columns = ["time_label", "light_dark", "strain", "genotype", "condition_label"]
    for column in string_columns:
        data[column] = data[column].astype(str).str.strip()

    data["light_dark"] = data["light_dark"].str.upper()
    data["strain"] = data["strain"].str.strip()
    data["genotype"] = data["genotype"].str.strip()

    numeric_columns = ALL_FEATURES + ["trial", "hour"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numeric_columns + ["light_dark", "strain", "genotype"]).copy()
    data["trial"] = data["trial"].astype(int)
    data["hour"] = data["hour"].astype(int)
    data["is_control"] = data["strain"].eq(control_label)
    data["cell_id"] = np.arange(1, len(data) + 1)
    return data


def clean_leaf_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    string_columns = ["leaf_position", "side"]
    for column in string_columns:
        data[column] = data[column].astype(str).str.strip()

    numeric_columns = ALL_FEATURES + ["group", "leaf_sample"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numeric_columns).copy()
    data["group"] = data["group"].astype(int)
    data["leaf_sample"] = data["leaf_sample"].astype(int)
    return data


def compute_scaling_reference(data: pd.DataFrame, features: Sequence[str]) -> ScalingReference:
    control = data.loc[data["is_control"], list(features)]
    means = control.mean().to_dict()
    sds = control.std(ddof=1).to_dict()
    zero_sd = [feature for feature, value in sds.items() if pd.isna(value) or value == 0]
    if zero_sd:
        joined = ", ".join(zero_sd)
        raise ValueError(f"Control-based scaling failed because these features have zero or missing variance: {joined}")
    return ScalingReference(means=means, sds=sds)


def apply_standardization(data: pd.DataFrame, scaling: ScalingReference, features: Sequence[str]) -> pd.DataFrame:
    standardized = data.copy()
    for feature in features:
        standardized[f"{feature}_z"] = (standardized[feature] - scaling.means[feature]) / scaling.sds[feature]
    return standardized


def stimulus_counts(data: pd.DataFrame) -> pd.DataFrame:
    counts = (
        data.groupby(["genotype", "light_dark", "hour", "trial", "strain"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["genotype", "light_dark", "hour", "trial", "strain"])
    )
    return counts
