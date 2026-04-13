from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font


SHEET_DESCRIPTIONS = {
    "readme": "Run metadata and sheet descriptions.",
    "stimulus_counts": "Raw condition counts in the stimulus dataset.",
    "reference_issues": "Missing controls or incomplete axis references.",
    "scaling_reference": "Control-based scaling parameters for shared features.",
    "control_reference": "Matched control centroids in standardized feature space.",
    "matched_cell_data": "Cell-level standardized and control-centered data.",
    "trial_feature_effects": "Feature-level trial-wise control vs infected comparisons.",
    "trial_response_vectors": "Trial-wise response vectors in standardized feature space.",
    "trial_module_divergence": "Trial-wise multivariate divergence for size, shape, and all features.",
    "control_axes": "Control-derived light and time axes in standardized feature space.",
    "response_alignment": "Alignment of infection response vectors against available control axes.",
    "condition_feature_consensus": "Feature-level aggregation across trials.",
    "condition_vector_consensus": "Condition-level aggregation across trials.",
    "alignment_consensus": "Alignment aggregation across trials.",
    "trial_robustness": "Leave-one-trial-out response-vector stability.",
    "development_group_summary": "Optional developmental summary by leaf sample and group.",
    "development_leaf_summary": "Optional developmental summary by leaf sample.",
}


def build_readme_frame(
    stomata_file: str,
    output_file: str,
    control_label: str,
    leaf_file: str | None = None,
) -> pd.DataFrame:
    rows = [
        {"item": "stomata_file", "value": stomata_file},
        {"item": "leaf_file", "value": leaf_file if leaf_file else ""},
        {"item": "output_file", "value": output_file},
        {"item": "control_label", "value": control_label},
        {"item": "", "value": ""},
    ]
    rows.extend({"item": sheet, "value": description} for sheet, description in SHEET_DESCRIPTIONS.items())
    return pd.DataFrame(rows)


def write_excel_workbook(path: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    output_path = Path(path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_frame = frame.copy()
            safe_frame.to_excel(writer, index=False, sheet_name=sheet_name)

    workbook = load_workbook(output_path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True)

        widths: dict[str, int] = {}
        for column_cells in sheet.columns:
            max_len = 0
            column_letter = column_cells[0].column_letter
            for cell in column_cells:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(value))
            widths[column_letter] = min(max(max_len + 2, 12), 40)

        for column_letter, width in widths.items():
            sheet.column_dimensions[column_letter].width = width

    workbook.save(output_path)
