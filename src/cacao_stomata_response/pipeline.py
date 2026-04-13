from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import (
    build_control_axes,
    build_control_reference,
    build_matched_cell_data,
    build_reference_issues,
    compute_response_alignment,
    compute_trial_feature_effects,
    compute_trial_module_divergence,
    compute_trial_response_vectors,
    compute_trial_robustness,
    summarize_alignment_consensus,
    summarize_development_leaf_data,
    summarize_feature_consensus,
    summarize_vector_consensus,
)
from .excel_writer import build_readme_frame, write_excel_workbook
from .feature_sets import ALL_FEATURES
from .io import read_leaf_workbook, read_stimulus_workbook
from .preprocess import (
    apply_standardization,
    clean_leaf_data,
    clean_stimulus_data,
    compute_scaling_reference,
    stimulus_counts,
)


def run_pipeline(
    stomata_file: str | Path,
    output_file: str | Path,
    leaf_file: str | Path | None = None,
    control_label: str = "Control",
    regularization: float = 1e-6,
) -> dict[str, pd.DataFrame]:
    stomata_file = str(stomata_file)
    output_file = str(output_file)
    leaf_file_str = str(leaf_file) if leaf_file is not None else None

    stimulus_raw = read_stimulus_workbook(stomata_file)
    stimulus = clean_stimulus_data(stimulus_raw, control_label=control_label)
    scaling_reference = compute_scaling_reference(stimulus, ALL_FEATURES)
    stimulus = apply_standardization(stimulus, scaling=scaling_reference, features=ALL_FEATURES)

    stimulus_count_frame = stimulus_counts(stimulus)
    reference_issues = build_reference_issues(stimulus)
    control_reference = build_control_reference(stimulus, ALL_FEATURES)
    matched_cell_data = build_matched_cell_data(stimulus, ALL_FEATURES)
    trial_feature_effects = compute_trial_feature_effects(stimulus, ALL_FEATURES, control_label=control_label)
    trial_response_vectors = compute_trial_response_vectors(stimulus, ALL_FEATURES, control_label=control_label)
    trial_module_divergence = compute_trial_module_divergence(
        stimulus,
        control_label=control_label,
        regularization=regularization,
    )
    control_axes = build_control_axes(stimulus, ALL_FEATURES)
    response_alignment = compute_response_alignment(trial_response_vectors, control_axes, ALL_FEATURES)
    condition_feature_consensus = summarize_feature_consensus(trial_feature_effects)
    condition_vector_consensus = summarize_vector_consensus(trial_response_vectors, ALL_FEATURES)
    alignment_consensus = summarize_alignment_consensus(response_alignment)
    trial_robustness = compute_trial_robustness(trial_response_vectors, ALL_FEATURES)

    sheets: dict[str, pd.DataFrame] = {
        "readme": build_readme_frame(
            stomata_file=Path(stomata_file).name,
            leaf_file=Path(leaf_file_str).name if leaf_file_str else None,
            output_file=Path(output_file).name,
            control_label=control_label,
        ),
        "stimulus_counts": stimulus_count_frame,
        "reference_issues": reference_issues,
        "scaling_reference": scaling_reference.to_frame(),
        "control_reference": control_reference,
        "matched_cell_data": matched_cell_data,
        "trial_feature_effects": trial_feature_effects,
        "trial_response_vectors": trial_response_vectors,
        "trial_module_divergence": trial_module_divergence,
        "control_axes": control_axes,
        "response_alignment": response_alignment,
        "condition_feature_consensus": condition_feature_consensus,
        "condition_vector_consensus": condition_vector_consensus,
        "alignment_consensus": alignment_consensus,
        "trial_robustness": trial_robustness,
    }

    if leaf_file_str:
        leaf_raw = read_leaf_workbook(leaf_file_str)
        leaf_data = clean_leaf_data(leaf_raw)
        development_group_summary, development_leaf_summary = summarize_development_leaf_data(
            leaf_data=leaf_data,
            features=ALL_FEATURES,
        )
        sheets["development_group_summary"] = development_group_summary
        sheets["development_leaf_summary"] = development_leaf_summary

    write_excel_workbook(output_file, sheets)
    return sheets
