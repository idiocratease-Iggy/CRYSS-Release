# MIT License
#
# Copyright (c) [2026] [Alan A. Smith]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

DEBUG = False
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _normalize_header_key(key: object) -> str:
    text = str(key).strip().lower()
    for token in (" ", "-", "/", "\\", "(", ")", "."):
        text = text.replace(token, "_")
    text = text.replace("__", "_")
    return text.strip("_")


def _normalize_component_index(raw_index: str) -> str:
    idx = raw_index.strip().strip("_")
    return idx


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _normalize_salt_derivative(value: object) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    return text.casefold()


def load_slice_records(path: Path) -> list[dict]:
    """
    Load NODES slice_records from an Excel file with merged headers.
    Returns a list of dictionaries, one per row.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path, sheet_name="slice_records", header=[0, 1])
    normalized: dict[str, object] = {}

    for (block, comp), series in df.items():
        block_key = _normalize_header_key(block)
        comp_key = _normalize_header_key(comp)

        if comp_key.startswith("component"):
            idx = _normalize_component_index(comp_key.split("component")[-1])
            if not idx:
                continue
            if block_key.startswith("starting"):
                normalized[f"xc_{idx}"] = series
            elif block_key.startswith("solid"):
                normalized[f"xs_{idx}"] = series
            elif block_key.startswith("liquor"):
                normalized[f"xl_{idx}"] = series
            continue

        if comp_key in {"mixture_id", "slice_number", "recovery", "rel_sol_volume", "rmax", "rel_sol_volume_at_rmax", "rel_sol_volume_at_rmax_1", "id", "origin_id", "origin_plate_ref", "replicate_status", "replicate_index", "plate_number", "well_address", "block_number", "experiment_code", "salt_derivative", "salt", "derivative"}:
            normalized[comp_key] = series
        elif comp_key in {"", "unnamed", "unnamed_0_level_0"}:
            if block_key and block_key != "unnamed":
                normalized[block_key] = series
        else:
            normalized[comp_key] = series

    normalized_df = pd.DataFrame(normalized)
    records = normalized_df.to_dict(orient="records")

    cleaned_records: list[dict] = []
    for row_index, row in enumerate(records):
        if row.get("mixture_id") in (None, ""):
            continue

        cleaned = {}
        for key, value in row.items():
            if key is None:
                continue
            key_name = _normalize_header_key(key)
            if "unnamed" in str(key).lower():
                continue
            if key_name in {"salt_derivative", "salt", "derivative"}:
                cleaned["salt_derivative"] = _normalize_salt_derivative(value)
                cleaned["Salt/Derivative"] = _normalize_text(value)
                continue
            if pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value

        cleaned["R"] = None

        if not any(v is not None for v in cleaned.values()):
            continue
        cleaned_records.append(cleaned)

        xs_values = []
        xl_values = []
        xc_values = []
        xs_indices = sorted({int(key.split("_")[-1]) for key in cleaned if key.startswith("xs_")})
        xl_indices = sorted({int(key.split("_")[-1]) for key in cleaned if key.startswith("xl_")})
        xc_indices = sorted({int(key.split("_")[-1]) for key in cleaned if key.startswith("xc_")})
        for idx in xs_indices:
            if f"xs_{idx}" in cleaned:
                xs_values.append(cleaned[f"xs_{idx}"])
        for idx in xl_indices:
            if f"xl_{idx}" in cleaned:
                xl_values.append(cleaned[f"xl_{idx}"])
        for idx in xc_indices:
            if f"xc_{idx}" in cleaned:
                xc_values.append(cleaned[f"xc_{idx}"])

        if xc_values:
            cleaned["Xc"] = xc_values
        if xs_values:
            cleaned["Xs"] = xs_values
        if xl_values:
            cleaned["Xl"] = xl_values

        if DEBUG:
            logger.debug("LOADED row:")
            logger.debug("  mixture_id=%s", cleaned.get("mixture_id"))
            logger.debug("  row_index=%s", row_index)
            logger.debug("  Xs=%s", xs_values)
            logger.debug("  Xl=%s", xl_values)

    return cleaned_records
