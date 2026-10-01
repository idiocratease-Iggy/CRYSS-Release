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

from pathlib import Path
from typing import Dict

import numpy as np
from qc_settings import R_TOLERANCE

from . import mixture_set_detector as msd
from .excel_writer import write_diagnostic_excel
from .loader import load_slice_records
from .mixture_set_detector import QC_PRINT, build_mixture_sets, qc_event
from .slice_builder import MixtureSet, build_slices_for_mixture
from .xc_normalizer import compute_canonical_xc

import pandas as pd


REQUIRED_COLUMN_ID = "id"


def _looks_like_nodes_columns(columns: list[str]) -> bool:
    normalized = [str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in columns]
    return any("component" in col for col in normalized) or any(
        col.startswith("starting_") or col.startswith("solid_") or col.startswith("liquor_")
        for col in normalized
    )


def _is_crys_v2_sheet_name(sheet_name: str) -> bool:
    normalized = str(sheet_name).strip().lower()
    return normalized.startswith("cryss input file v2")


def detect_input_format(path: Path) -> str:
    """Return 'nodes' for NODES workbooks and 'crys' for standard CRYSS workbooks."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        with pd.ExcelFile(path) as xls:
            for name in xls.sheet_names:
                normalized = str(name).strip().lower()
                if normalized == "slice_records":
                    sheet = pd.read_excel(path, sheet_name=name, header=[0, 1])
                    if _looks_like_nodes_columns([str(col) for col in sheet.columns]):
                        return "nodes"
                elif _is_crys_v2_sheet_name(name):
                    return "crys"
    except Exception:
        pass

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        columns = [str(col).strip().lower() for col in df.columns]
        if _looks_like_nodes_columns(columns):
            return "nodes"
    except Exception:
        pass

    return "crys"


def _reject_degenerate_crys_rows(rows: list[dict]) -> list[dict]:
    """Reject CRYSS rows where the solid phase collapses to a single component."""
    valid_rows: list[dict] = []
    for row in rows:
        xs = row.get("Xs")
        if xs is None:
            valid_rows.append(row)
            continue

        xs_values = np.asarray(xs, dtype=float).reshape(-1)
        if xs_values.size == 0:
            valid_rows.append(row)
            continue

        dominant_component = float(np.max(xs_values))
        if dominant_component >= 0.99:
            others = np.delete(xs_values, int(np.argmax(xs_values)))
            if others.size == 0 or np.max(np.abs(others)) <= 1e-6:
                qc_event(
                    f"mixture {row.get('mixture_id', 'UNKNOWN')} rejected: degenerate one-component CRYSS slice"
                )
                continue

        valid_rows.append(row)

    return valid_rows


def _infer_standard_crys_canonical_xc(rows: list[dict], n_components: int) -> list[float]:
    """Infer a stable Xc estimate for CRYSS rows when the input file does not provide explicit xc_* columns."""
    estimates: list[list[float]] = []
    for row in rows:
        xs_value = row.get("Xs") if row.get("Xs") is not None else np.zeros(n_components, dtype=float)
        xl_value = row.get("Xl") if row.get("Xl") is not None else np.zeros(n_components, dtype=float)
        xs = np.asarray(xs_value, dtype=float).reshape(-1)
        xl = np.asarray(xl_value, dtype=float).reshape(-1)
        if xs.size < n_components:
            xs = np.pad(xs, (0, max(0, n_components - xs.size)), constant_values=0.0)
        if xl.size < n_components:
            xl = np.pad(xl, (0, max(0, n_components - xl.size)), constant_values=0.0)
        estimates.append(np.clip(0.5 * (xs + xl), 0.0, 1.0).tolist())

    if not estimates:
        return [0.0] * n_components

    return np.median(np.asarray(estimates, dtype=float), axis=0).tolist()


def _resolve_crys_sheet_name(path: Path):
    if path.suffix.lower() == ".csv":
        return None

    try:
        with pd.ExcelFile(path) as xls:
            names = [str(name).strip().lower() for name in xls.sheet_names]
            for name in xls.sheet_names:
                normalized = str(name).strip().lower()
                if normalized.startswith("cryss input file v2"):
                    return name
            if "slice_records" in names:
                return "slice_records"
            return xls.sheet_names[0] if xls.sheet_names else None
    except Exception:
        return None


def _flatten_multiindex_crys_rows(df: pd.DataFrame) -> list[dict]:
    normalized: dict[str, object] = {}
    for (block, comp), series in df.items():
        block_key = str(block).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        comp_key = str(comp).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

        if comp_key in {"", "unnamed", "unnamed_0_level_0"}:
            if block_key and block_key != "unnamed":
                normalized[block_key] = series
            continue

        if "derived_from" in comp_key or comp_key.startswith("notes"):
            continue

        if comp_key.startswith("component"):
            idx = comp_key.split("component")[-1].strip("_")
            if not idx:
                continue
            if block_key.startswith("starting"):
                normalized[f"xc_{idx}"] = series
            elif block_key.startswith("solid"):
                normalized[f"xs_{idx}"] = series
            elif block_key.startswith("liquor"):
                if comp_key.endswith("salt_derivative"):
                    normalized["salt_derivative"] = series
                else:
                    normalized[f"xl_{idx}"] = series
            continue

        if comp_key in {"mixture_id", "id", "salt_derivative", "salt", "derivative"}:
            normalized[comp_key] = series
            continue

        if block_key in {"liquor", "solid", "starting"}:
            continue

        normalized[comp_key] = series

    normalized_df = pd.DataFrame(normalized)
    records = normalized_df.to_dict(orient="records")
    cleaned_records: list[dict] = []
    for row in records:
        if row.get("mixture_id") in (None, ""):
            continue
        cleaned = {}
        for key, value in row.items():
            if key is None:
                continue
            if "unnamed" in str(key).lower():
                continue
            if pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        if cleaned.get("salt_derivative") is not None:
            cleaned["salt_derivative"] = str(cleaned["salt_derivative"]).strip().casefold()
        cleaned["R"] = None
        xs_values = []
        xl_values = []
        xc_values = []
        for idx in sorted({int(k.split("_")[-1]) for k in cleaned if k.startswith("xs_")}):
            if f"xs_{idx}" in cleaned:
                xs_values.append(cleaned[f"xs_{idx}"])
        for idx in sorted({int(k.split("_")[-1]) for k in cleaned if k.startswith("xl_")}):
            if f"xl_{idx}" in cleaned:
                xl_values.append(cleaned[f"xl_{idx}"])
        for idx in sorted({int(k.split("_")[-1]) for k in cleaned if k.startswith("xc_")}):
            if f"xc_{idx}" in cleaned:
                xc_values.append(cleaned[f"xc_{idx}"])
        if xc_values:
            cleaned["Xc"] = xc_values
        if xs_values:
            cleaned["Xs"] = xs_values
        if xl_values:
            cleaned["Xl"] = xl_values
        cleaned_records.append(cleaned)
    return cleaned_records


def load_slice_records_basecase(path: Path):
    """
    Load arbitrary CSV/XLSX files in the base‑case format.
    Only 'id' is required. Xs*/Xl* columns are optional.
    All other columns are retained as metadata for the CRYSS branch.
    """

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        sheet_name = _resolve_crys_sheet_name(path)
        if sheet_name is not None and str(sheet_name).strip().lower().startswith("cryss input file v2"):
            df = pd.read_excel(path, sheet_name=sheet_name, header=[0, 1])
            return _flatten_multiindex_crys_rows(df)
        df = pd.read_excel(path, sheet_name=sheet_name) if sheet_name else pd.read_excel(path)

    normalized_cols = [str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_") for c in df.columns]
    required_key = None
    for candidate in ("id", "mixture_id"):
        if candidate in normalized_cols:
            required_key = candidate
            break
    if required_key is None:
        raise ValueError("Base‑case parser: missing required column 'id' or 'Mixture_id'")

    xs_cols = [c for c in df.columns if str(c).lower().startswith("xs")]
    xl_cols = [c for c in df.columns if str(c).lower().startswith("xl")]

    slice_records = []

    for _, row in df.iterrows():
        mixture_value = row.get("Mixture_id", row.get("mixture_id", "UNKNOWN"))
        raw_id = row.get("id", None)
        if raw_id is None and mixture_value is not None:
            raw_id = str(mixture_value)
        rec = {
            "id": str(raw_id) if raw_id is not None else "UNKNOWN",
            "mixture_id": str(mixture_value if mixture_value is not None else "UNKNOWN"),
            "Xs": row[xs_cols].to_numpy(dtype=float) if xs_cols else None,
            "Xl": row[xl_cols].to_numpy(dtype=float) if xl_cols else None,
        }

        xs_values = np.asarray(rec["Xs"], dtype=float).reshape(-1) if rec["Xs"] is not None else []
        xl_values = np.asarray(rec["Xl"], dtype=float).reshape(-1) if rec["Xl"] is not None else []
        for idx, value in enumerate(xs_values, start=1):
            rec[f"xs_{idx}"] = float(value)
        for idx, value in enumerate(xl_values, start=1):
            rec[f"xl_{idx}"] = float(value)

        for col_name, value in row.items():
            key = str(col_name)
            norm_key = key.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            if key.lower().startswith("xs") or key.lower().startswith("xl"):
                continue
            if key.lower() in {"id", "mixture_id", "mixture id"}:
                continue
            if norm_key in {"salt_derivative", "salt", "derivative"}:
                rec["salt_derivative"] = str(value).strip().casefold() if value is not None and not pd.isna(value) else None
                rec["Salt/Derivative"] = None if pd.isna(value) else str(value).strip()
                continue
            rec[key] = None if pd.isna(value) else value
        slice_records.append(rec)

    return slice_records


def parse_nodes_excel(
    path: Path,
    write_excel: bool = False,
    output_path: Path | None = None,
    *,
    quiet: bool = True,
) -> Dict[str, MixtureSet]:

    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    msd.total_reps = 0
    msd.rejected_reps = 0
    msd.replicate_devs = []
    msd.QC_PRINT = not quiet
    msd.R_TOLERANCE = R_TOLERANCE

    input_format = detect_input_format(path)
    if input_format == "nodes":
        try:
            rows = load_slice_records(path)
        except Exception:
            rows = []
        full_nodes = True
    else:
        rows = load_slice_records_basecase(path)
        full_nodes = False
        rows = _reject_degenerate_crys_rows(rows)

    if full_nodes:
        # Existing NODES logic
        if not rows:
            return {}

        n_components = max(
            1,
            max(
                (len([key for key in row if key.startswith("xc_")]) for row in rows),
                default=0,
            ),
        )

    else:
        if not rows:
            return {}

        # Infer number of components from Xs or Xl
        if rows and rows[0].get("Xs") is not None:
            n_components = len(rows[0]["Xs"])
        elif rows and rows[0].get("Xl") is not None:
            n_components = len(rows[0]["Xl"])
        else:
            raise ValueError("Cannot infer number of components from base‑case file")

    # Build mixture sets (works for both full and base‑case)
    mixture_rows = build_mixture_sets(rows, n_components=n_components)
    mixtures: Dict[str, MixtureSet] = {}

    for mixture_id, mrows in mixture_rows.items():
        if not mrows:
            continue

        if full_nodes:
            canonical_xc = compute_canonical_xc(mrows, n_components)
        else:
            canonical_xc = _infer_standard_crys_canonical_xc(mrows, n_components)

        mixture = build_slices_for_mixture(
            mixture_id, mrows, canonical_xc, n_components
        )

        if not mixture.slices:
            qc_event(f"mixture {mixture_id} rejected: no valid mixture_set found")
            continue

        mixtures[mixture_id] = mixture
        qc_event(f"mixture {mixture_id} added to mixture_set {mixture_id}")

    # Diagnostic Excel export is intentionally disabled for the cryss-core flow.
    # Keep parsing inline and avoid writing statistics workbooks to disk.
    # if write_excel:
    #     out_path = Path(output_path) if output_path else path.with_name(path.stem + "_parsed.xlsx")
    #     write_diagnostic_excel(mixtures, out_path)

    if msd.QC_PRINT and msd.total_reps > 0:
        rejection_rate = msd.rejected_reps / msd.total_reps
        print(f"[QC] replicate rejection rate = {rejection_rate:.2f} ({msd.rejected_reps}/{msd.total_reps})")

    if msd.QC_PRINT and hasattr(msd, "replicate_devs") and msd.replicate_devs:
        median_dev = float(np.median(msd.replicate_devs))
        print("[QC] within-slice replicate dR stats:")
        print(f"  min dR = {min(msd.replicate_devs):.3f}")
        print(f"  median dR = {median_dev:.3f}")
        print(f"  max dR = {max(msd.replicate_devs):.3f}")

    return mixtures

