from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from .feature_sets import ALL_FEATURES, SHAPE_FEATURES, SIZE_FEATURES
from .metrics import (
    angle_degrees,
    cosine_similarity,
    gaussian_w2_distance,
    hedges_g,
    one_dimensional_wasserstein,
    project_vector,
    summarize_series,
    vector_norm,
)


@dataclass(frozen=True)
class GroupColumns:
    stimulus_reference: tuple[str, ...] = ("genotype", "light_dark", "hour", "trial")
    stimulus_condition: tuple[str, ...] = ("genotype", "light_dark", "hour", "trial", "strain")


def build_reference_issues(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    grouped = data.groupby(["genotype", "light_dark", "hour", "trial"], dropna=False)
    for keys, frame in grouped:
        control_count = int(frame["is_control"].sum())
        if control_count == 0:
            rows.append(
                {
                    "issue_type": "missing_control",
                    "genotype": keys[0],
                    "light_dark": keys[1],
                    "hour": keys[2],
                    "trial": keys[3],
                    "details": "No control cells available for this genotype, light regime, hour, and trial.",
                }
            )

    controls = data.loc[data["is_control"]]
    light_axis_check = (
        controls.groupby(["genotype", "trial", "hour", "light_dark"])
        .size()
        .reset_index(name="n_cells")
        .pivot_table(
            index=["genotype", "trial", "hour"],
            columns="light_dark",
            values="n_cells",
            aggfunc="first",
        )
        .reset_index()
    )
    if "L" not in light_axis_check.columns or "D" not in light_axis_check.columns:
        missing_rows = light_axis_check[["genotype", "trial", "hour"]].drop_duplicates()
    else:
        missing_rows = light_axis_check.loc[light_axis_check[["L", "D"]].isna().any(axis=1), ["genotype", "trial", "hour"]]

    for _, row in missing_rows.iterrows():
        rows.append(
            {
                "issue_type": "incomplete_light_axis",
                "genotype": row["genotype"],
                "light_dark": np.nan,
                "hour": int(row["hour"]),
                "trial": int(row["trial"]),
                "details": "Light-axis reference is unavailable because matching control cells were not found for both light regimes at this hour.",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["issue_type", "genotype", "light_dark", "hour", "trial", "details"]
        )

    return pd.DataFrame(rows).sort_values(["issue_type", "genotype", "trial", "hour"], na_position="last")


def build_control_reference(data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    controls = data.loc[data["is_control"]].copy()
    rows = []
    for keys, frame in controls.groupby(["genotype", "light_dark", "hour", "trial"], dropna=False):
        row: dict[str, object] = {
            "genotype": keys[0],
            "light_dark": keys[1],
            "hour": int(keys[2]),
            "trial": int(keys[3]),
            "n_control": int(len(frame)),
        }
        for feature in features:
            values = frame[f"{feature}_z"].to_numpy(dtype=float)
            row[f"{feature}_mean_z"] = float(np.mean(values))
            row[f"{feature}_sd_z"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["genotype", "light_dark", "hour", "trial"])


def build_matched_cell_data(data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    control_reference = build_control_reference(data, features)
    merged = data.merge(
        control_reference,
        on=["genotype", "light_dark", "hour", "trial"],
        how="left",
        validate="many_to_one",
    )

    for feature in features:
        merged[f"{feature}_centered_z"] = merged[f"{feature}_z"] - merged[f"{feature}_mean_z"]

    merged["comparison_group"] = np.where(merged["is_control"], "control", "infected")
    ordered_columns = [
        "cell_id",
        "genotype",
        "light_dark",
        "hour",
        "trial",
        "strain",
        "condition_label",
        "comparison_group",
        "time_label",
    ]
    for feature in features:
        ordered_columns.extend([feature, f"{feature}_z", f"{feature}_centered_z"])
    return merged[ordered_columns].sort_values(["genotype", "light_dark", "hour", "trial", "strain", "cell_id"])


def compute_trial_feature_effects(data: pd.DataFrame, features: Sequence[str], control_label: str) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(["genotype", "light_dark", "hour", "trial", "strain"], dropna=False)
    for keys, infected in grouped:
        genotype, light_dark, hour, trial, strain = keys
        if strain == control_label:
            continue

        control = data.loc[
            data["is_control"]
            & data["genotype"].eq(genotype)
            & data["light_dark"].eq(light_dark)
            & data["hour"].eq(hour)
            & data["trial"].eq(trial)
        ]
        if control.empty or infected.empty:
            continue

        for feature in features:
            control_raw = control[feature].to_numpy(dtype=float)
            infected_raw = infected[feature].to_numpy(dtype=float)
            control_std = control[f"{feature}_z"].to_numpy(dtype=float)
            infected_std = infected[f"{feature}_z"].to_numpy(dtype=float)

            rows.append(
                {
                    "genotype": genotype,
                    "light_dark": light_dark,
                    "hour": int(hour),
                    "trial": int(trial),
                    "strain": strain,
                    "feature": feature,
                    "n_control": int(len(control_raw)),
                    "n_infected": int(len(infected_raw)),
                    "control_mean_raw": float(np.mean(control_raw)),
                    "infected_mean_raw": float(np.mean(infected_raw)),
                    "mean_difference_raw": float(np.mean(infected_raw) - np.mean(control_raw)),
                    "control_mean_z": float(np.mean(control_std)),
                    "infected_mean_z": float(np.mean(infected_std)),
                    "mean_difference_z": float(np.mean(infected_std) - np.mean(control_std)),
                    "hedges_g": hedges_g(control_std, infected_std),
                    "wasserstein_1d_z": one_dimensional_wasserstein(control_std, infected_std),
                }
            )
    columns = [
        "genotype",
        "light_dark",
        "hour",
        "trial",
        "strain",
        "feature",
        "n_control",
        "n_infected",
        "control_mean_raw",
        "infected_mean_raw",
        "mean_difference_raw",
        "control_mean_z",
        "infected_mean_z",
        "mean_difference_z",
        "hedges_g",
        "wasserstein_1d_z",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["genotype", "light_dark", "hour", "trial", "strain", "feature"])


def compute_trial_response_vectors(data: pd.DataFrame, features: Sequence[str], control_label: str) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(["genotype", "light_dark", "hour", "trial", "strain"], dropna=False)
    for keys, infected in grouped:
        genotype, light_dark, hour, trial, strain = keys
        if strain == control_label:
            continue

        control = data.loc[
            data["is_control"]
            & data["genotype"].eq(genotype)
            & data["light_dark"].eq(light_dark)
            & data["hour"].eq(hour)
            & data["trial"].eq(trial)
        ]
        if control.empty or infected.empty:
            continue

        row: dict[str, object] = {
            "genotype": genotype,
            "light_dark": light_dark,
            "hour": int(hour),
            "trial": int(trial),
            "strain": strain,
            "n_control": int(len(control)),
            "n_infected": int(len(infected)),
        }

        vector_all = []
        vector_size = []
        vector_shape = []
        for feature in features:
            response_value = float(infected[f"{feature}_z"].mean() - control[f"{feature}_z"].mean())
            row[f"{feature}_response_z"] = response_value
            vector_all.append(response_value)
            if feature in SIZE_FEATURES:
                vector_size.append(response_value)
            if feature in SHAPE_FEATURES:
                vector_shape.append(response_value)

        row["all_response_norm"] = vector_norm(np.asarray(vector_all))
        row["size_response_norm"] = vector_norm(np.asarray(vector_size))
        row["shape_response_norm"] = vector_norm(np.asarray(vector_shape))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["genotype", "light_dark", "hour", "trial", "strain"])


def compute_trial_module_divergence(data: pd.DataFrame, control_label: str, regularization: float) -> pd.DataFrame:
    rows = []
    grouped = data.groupby(["genotype", "light_dark", "hour", "trial", "strain"], dropna=False)
    modules = {
        "size": SIZE_FEATURES,
        "shape": SHAPE_FEATURES,
        "all": ALL_FEATURES,
    }
    for keys, infected in grouped:
        genotype, light_dark, hour, trial, strain = keys
        if strain == control_label:
            continue

        control = data.loc[
            data["is_control"]
            & data["genotype"].eq(genotype)
            & data["light_dark"].eq(light_dark)
            & data["hour"].eq(hour)
            & data["trial"].eq(trial)
        ]
        if control.empty or infected.empty:
            continue

        row: dict[str, object] = {
            "genotype": genotype,
            "light_dark": light_dark,
            "hour": int(hour),
            "trial": int(trial),
            "strain": strain,
            "n_control": int(len(control)),
            "n_infected": int(len(infected)),
        }
        for module_name, module_features in modules.items():
            control_values = control[[f"{feature}_z" for feature in module_features]].to_numpy(dtype=float)
            infected_values = infected[[f"{feature}_z" for feature in module_features]].to_numpy(dtype=float)
            response_vector = infected_values.mean(axis=0) - control_values.mean(axis=0)
            row[f"{module_name}_gaussian_w2"] = gaussian_w2_distance(
                control_values=control_values,
                infected_values=infected_values,
                regularization=regularization,
            )
            row[f"{module_name}_mean_shift_norm"] = vector_norm(response_vector)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["genotype", "light_dark", "hour", "trial", "strain"])


def build_control_axes(data: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    controls = data.loc[data["is_control"]].copy()
    control_means = (
        controls.groupby(["genotype", "trial", "light_dark", "hour"], dropna=False)
        [[f"{feature}_z" for feature in features]]
        .mean()
        .reset_index()
    )
    rows = []

    # Light axes
    for (genotype, trial, hour), frame in control_means.groupby(["genotype", "trial", "hour"], dropna=False):
        if {"L", "D"} - set(frame["light_dark"].tolist()):
            continue
        light_row = frame.loc[frame["light_dark"].eq("L")].iloc[0]
        dark_row = frame.loc[frame["light_dark"].eq("D")].iloc[0]
        vector = []
        row = {
            "axis_family": "light",
            "axis_label": "D_to_L",
            "genotype": genotype,
            "trial": int(trial),
            "light_dark": np.nan,
            "start_hour": int(hour),
            "end_hour": int(hour),
            "reference_hour": int(hour),
        }
        for feature in features:
            value = float(light_row[f"{feature}_z"] - dark_row[f"{feature}_z"])
            row[f"{feature}_axis"] = value
            vector.append(value)
        row["axis_norm"] = vector_norm(np.asarray(vector))
        rows.append(row)

    # Time axes
    for (genotype, trial, light_dark), frame in control_means.groupby(["genotype", "trial", "light_dark"], dropna=False):
        hours = sorted(frame["hour"].astype(int).tolist())
        for start_hour, end_hour in combinations(hours, 2):
            start_row = frame.loc[frame["hour"].eq(start_hour)].iloc[0]
            end_row = frame.loc[frame["hour"].eq(end_hour)].iloc[0]
            vector = []
            row = {
                "axis_family": "time",
                "axis_label": f"{start_hour}_to_{end_hour}",
                "genotype": genotype,
                "trial": int(trial),
                "light_dark": light_dark,
                "start_hour": int(start_hour),
                "end_hour": int(end_hour),
                "reference_hour": int(end_hour),
            }
            for feature in features:
                value = float(end_row[f"{feature}_z"] - start_row[f"{feature}_z"])
                row[f"{feature}_axis"] = value
                vector.append(value)
            row["axis_norm"] = vector_norm(np.asarray(vector))
            rows.append(row)

    columns = [
        "axis_family",
        "axis_label",
        "genotype",
        "trial",
        "light_dark",
        "start_hour",
        "end_hour",
        "reference_hour",
    ] + [f"{feature}_axis" for feature in features] + ["axis_norm"]

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["axis_family", "genotype", "trial", "reference_hour", "light_dark", "axis_label"],
        na_position="last",
    )


def compute_response_alignment(response_vectors: pd.DataFrame, control_axes: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows = []
    for _, response in response_vectors.iterrows():
        response_vector = response[[f"{feature}_response_z" for feature in features]].to_numpy(dtype=float)

        candidate_axes = control_axes.loc[
            control_axes["genotype"].eq(response["genotype"])
            & control_axes["trial"].eq(response["trial"])
            & (
                ((control_axes["axis_family"] == "light") & control_axes["reference_hour"].eq(response["hour"]))
                | (
                    (control_axes["axis_family"] == "time")
                    & control_axes["light_dark"].eq(response["light_dark"])
                    & control_axes["end_hour"].eq(response["hour"])
                )
            )
        ]

        for _, axis in candidate_axes.iterrows():
            axis_vector = axis[[f"{feature}_axis" for feature in features]].to_numpy(dtype=float)
            parallel, perpendicular = project_vector(response_vector, axis_vector)
            rows.append(
                {
                    "genotype": response["genotype"],
                    "light_dark": response["light_dark"],
                    "hour": int(response["hour"]),
                    "trial": int(response["trial"]),
                    "strain": response["strain"],
                    "axis_family": axis["axis_family"],
                    "axis_label": axis["axis_label"],
                    "axis_light_dark": axis["light_dark"],
                    "axis_start_hour": axis["start_hour"],
                    "axis_end_hour": axis["end_hour"],
                    "response_norm": vector_norm(response_vector),
                    "axis_norm": vector_norm(axis_vector),
                    "cosine_similarity": cosine_similarity(response_vector, axis_vector),
                    "angle_deg": angle_degrees(response_vector, axis_vector),
                    "parallel_component": parallel,
                    "perpendicular_component": perpendicular,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["genotype", "light_dark", "hour", "trial", "strain", "axis_family", "axis_label"]
    )


def summarize_feature_consensus(trial_feature_effects: pd.DataFrame) -> pd.DataFrame:
    grouping = ["genotype", "light_dark", "hour", "strain", "feature"]
    rows = []
    for keys, frame in trial_feature_effects.groupby(grouping, dropna=False):
        summary_mean_diff = summarize_series(frame["mean_difference_z"].tolist())
        summary_g = summarize_series(frame["hedges_g"].tolist())
        summary_w = summarize_series(frame["wasserstein_1d_z"].tolist())
        mean_direction = np.sign(np.nanmean(frame["mean_difference_z"].to_numpy(dtype=float)))
        sign_consistency = float(
            np.mean(
                np.sign(frame["mean_difference_z"].to_numpy(dtype=float)) == mean_direction
            )
        ) if len(frame) else np.nan
        rows.append(
            {
                "genotype": keys[0],
                "light_dark": keys[1],
                "hour": int(keys[2]),
                "strain": keys[3],
                "feature": keys[4],
                "trial_count": int(len(frame)),
                "mean_difference_z_mean": summary_mean_diff["mean"],
                "mean_difference_z_sd": summary_mean_diff["sd"],
                "hedges_g_mean": summary_g["mean"],
                "hedges_g_sd": summary_g["sd"],
                "wasserstein_1d_z_mean": summary_w["mean"],
                "wasserstein_1d_z_sd": summary_w["sd"],
                "sign_consistency": sign_consistency,
            }
        )
    return pd.DataFrame(rows).sort_values(grouping)


def summarize_vector_consensus(trial_response_vectors: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    grouping = ["genotype", "light_dark", "hour", "strain"]
    rows = []
    for keys, frame in trial_response_vectors.groupby(grouping, dropna=False):
        row: dict[str, object] = {
            "genotype": keys[0],
            "light_dark": keys[1],
            "hour": int(keys[2]),
            "strain": keys[3],
            "trial_count": int(len(frame)),
        }
        for feature in features:
            values = frame[f"{feature}_response_z"].to_numpy(dtype=float)
            summary = summarize_series(values)
            row[f"{feature}_response_mean"] = summary["mean"]
            row[f"{feature}_response_sd"] = summary["sd"]
            row[f"{feature}_sign_consistency"] = float(
                np.mean(np.sign(values) == np.sign(np.nanmean(values)))
            ) if len(values) else np.nan

        for prefix in ["all", "size", "shape"]:
            summary = summarize_series(frame[f"{prefix}_response_norm"].tolist())
            row[f"{prefix}_response_norm_mean"] = summary["mean"]
            row[f"{prefix}_response_norm_sd"] = summary["sd"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(grouping)


def summarize_alignment_consensus(response_alignment: pd.DataFrame) -> pd.DataFrame:
    if response_alignment.empty:
        return pd.DataFrame(
            columns=[
                "genotype",
                "light_dark",
                "hour",
                "strain",
                "axis_family",
                "axis_label",
                "trial_count",
                "cosine_similarity_mean",
                "cosine_similarity_sd",
                "angle_deg_mean",
                "angle_deg_sd",
                "parallel_component_mean",
                "parallel_component_sd",
                "perpendicular_component_mean",
                "perpendicular_component_sd",
            ]
        )
    grouping = ["genotype", "light_dark", "hour", "strain", "axis_family", "axis_label"]
    rows = []
    for keys, frame in response_alignment.groupby(grouping, dropna=False):
        cosine_summary = summarize_series(frame["cosine_similarity"].tolist())
        angle_summary = summarize_series(frame["angle_deg"].tolist())
        parallel_summary = summarize_series(frame["parallel_component"].tolist())
        perpendicular_summary = summarize_series(frame["perpendicular_component"].tolist())
        rows.append(
            {
                "genotype": keys[0],
                "light_dark": keys[1],
                "hour": int(keys[2]),
                "strain": keys[3],
                "axis_family": keys[4],
                "axis_label": keys[5],
                "trial_count": int(len(frame)),
                "cosine_similarity_mean": cosine_summary["mean"],
                "cosine_similarity_sd": cosine_summary["sd"],
                "angle_deg_mean": angle_summary["mean"],
                "angle_deg_sd": angle_summary["sd"],
                "parallel_component_mean": parallel_summary["mean"],
                "parallel_component_sd": parallel_summary["sd"],
                "perpendicular_component_mean": perpendicular_summary["mean"],
                "perpendicular_component_sd": perpendicular_summary["sd"],
            }
        )
    return pd.DataFrame(rows).sort_values(grouping)


def compute_trial_robustness(trial_response_vectors: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    grouping = ["genotype", "light_dark", "hour", "strain"]
    rows = []
    for keys, frame in trial_response_vectors.groupby(grouping, dropna=False):
        if len(frame) < 2:
            continue

        full_vector = frame[[f"{feature}_response_z" for feature in features]].mean().to_numpy(dtype=float)
        full_norm = vector_norm(full_vector)

        for trial in sorted(frame["trial"].unique()):
            subset = frame.loc[frame["trial"].ne(trial)]
            if subset.empty:
                continue
            subset_vector = subset[[f"{feature}_response_z" for feature in features]].mean().to_numpy(dtype=float)
            subset_norm = vector_norm(subset_vector)
            rows.append(
                {
                    "genotype": keys[0],
                    "light_dark": keys[1],
                    "hour": int(keys[2]),
                    "strain": keys[3],
                    "omitted_trial": int(trial),
                    "full_norm": full_norm,
                    "subset_norm": subset_norm,
                    "norm_ratio": subset_norm / full_norm if full_norm != 0 else np.nan,
                    "cosine_similarity": cosine_similarity(full_vector, subset_vector),
                    "angle_deg": angle_degrees(full_vector, subset_vector),
                }
            )

    columns = [
        "genotype",
        "light_dark",
        "hour",
        "strain",
        "omitted_trial",
        "full_norm",
        "subset_norm",
        "norm_ratio",
        "cosine_similarity",
        "angle_deg",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["genotype", "light_dark", "hour", "strain", "omitted_trial"]
    )


def summarize_development_leaf_data(leaf_data: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    for (leaf_sample, group), frame in leaf_data.groupby(["leaf_sample", "group"], dropna=False):
        row: dict[str, object] = {
            "leaf_sample": int(leaf_sample),
            "group": int(group),
            "n_cells": int(len(frame)),
        }
        for feature in features:
            row[f"{feature}_mean"] = float(frame[feature].mean())
            row[f"{feature}_sd"] = float(frame[feature].std(ddof=1)) if len(frame) > 1 else np.nan
        group_rows.append(row)

    group_summary = pd.DataFrame(group_rows).sort_values(["leaf_sample", "group"])

    leaf_rows = []
    for leaf_sample, frame in leaf_data.groupby("leaf_sample", dropna=False):
        row = {
            "leaf_sample": int(leaf_sample),
            "n_cells": int(len(frame)),
        }
        for feature in features:
            row[f"{feature}_mean"] = float(frame[feature].mean())
            row[f"{feature}_sd"] = float(frame[feature].std(ddof=1)) if len(frame) > 1 else np.nan
        leaf_rows.append(row)

    leaf_summary = pd.DataFrame(leaf_rows).sort_values("leaf_sample")
    return group_summary, leaf_summary
