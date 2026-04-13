from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_label(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


@dataclass(frozen=True)
class ColumnMapping:
    original_to_standard: dict[str, str]

    def rename(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.rename(columns=self.original_to_standard).copy()


def _find_sheet_with_columns(path: Path, required_aliases: Iterable[set[str]]) -> str:
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name=sheet_name, nrows=5)
        normalized = {normalize_label(col): col for col in frame.columns}
        if all(any(alias in normalized for alias in aliases) for aliases in required_aliases):
            return sheet_name
    raise ValueError(f"No worksheet in {path.name} contains the required columns.")


def _build_mapping(columns: Iterable[str], aliases: dict[str, set[str]]) -> ColumnMapping:
    normalized = {normalize_label(col): col for col in columns}
    original_to_standard: dict[str, str] = {}
    for standard_name, valid_aliases in aliases.items():
        original_name = next((normalized[alias] for alias in valid_aliases if alias in normalized), None)
        if original_name is None:
            raise ValueError(f"Missing required column for '{standard_name}'.")
        original_to_standard[original_name] = standard_name
    return ColumnMapping(original_to_standard=original_to_standard)


def read_stimulus_workbook(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    aliases = {
        "time_label": {"time"},
        "area": {"area", "areasize"},
        "perimeter": {"perimeter", "perimeterlength"},
        "length": {"length"},
        "width": {"width"},
        "lwr": {"lwr"},
        "circularity": {"circularity"},
        "distance_is_cg": {"distancebetweenisandcg", "isandcg"},
        "trial": {"trial"},
        "light_dark": {"lightdark"},
        "hour": {"hour"},
        "strain": {"strain"},
        "genotype": {"genotype"},
        "condition_label": {"specificcondition"},
    }
    required_aliases = list(aliases.values())
    sheet_name = _find_sheet_with_columns(path, required_aliases)
    frame = pd.read_excel(path, sheet_name=sheet_name)
    mapping = _build_mapping(frame.columns, aliases)
    frame = mapping.rename(frame)
    frame["source_sheet"] = sheet_name
    frame["source_file"] = path.name
    return frame


def read_leaf_workbook(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    aliases = {
        "leaf_position": {"leafposition"},
        "area": {"areasize"},
        "perimeter": {"perimeterlength"},
        "length": {"length"},
        "width": {"width"},
        "lwr": {"lwr"},
        "circularity": {"circularity"},
        "distance_is_cg": {"isandcg"},
        "group": {"group"},
        "side": {"rorl"},
        "leaf_sample": {"leafsamplenumber"},
    }
    required_aliases = list(aliases.values())
    sheet_name = _find_sheet_with_columns(path, required_aliases)
    frame = pd.read_excel(path, sheet_name=sheet_name)
    mapping = _build_mapping(frame.columns, aliases)
    frame = mapping.rename(frame)
    frame["source_sheet"] = sheet_name
    frame["source_file"] = path.name
    return frame
