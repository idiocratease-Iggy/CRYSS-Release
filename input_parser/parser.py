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


def load_slice_records_basecase(path: Path):
    """
    Load arbitrary CSV/XLSX files in the base‑case format.
    Only 'id' is required. Xs*/Xl* columns are optional.
    All other columns are ignored.
    """

    # Accept CSV or Excel
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if REQUIRED_COLUMN_ID not in df.columns:
        raise ValueError("Base‑case parser: missing required column 'id'")

    # Flexible detection of solid/liquor columns
    xs_cols = [c for c in df.columns if c.lower().startswith("xs")]
    xl_cols = [c for c in df.columns if c.lower().startswith("xl")]

    slice_records = []

    for _, row in df.iterrows():
        rec = {
            "id": str(row["id"]),
            "mixture_id": str(row.get("Mixture_id", "UNKNOWN")),
            "Xs": row[xs_cols].to_numpy(dtype=float) if xs_cols else None,
            "Xl": row[xl_cols].to_numpy(dtype=float) if xl_cols else None,
            # Xc intentionally skipped — base‑case does NOT reconstruct Xc
            # R intentionally skipped — base‑case does NOT enforce R
        }
        slice_records.append(rec)

    return slice_records




def parse_nodes_excel(
    path: Path,
    write_excel: bool = False,
    output_path: Path | None = None,
    *,
    quiet: bool = False,
) -> Dict[str, MixtureSet]:

    if not Path(path).exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    msd.total_reps = 0
    msd.rejected_reps = 0
    msd.replicate_devs = []
    msd.QC_PRINT = not quiet
    msd.R_TOLERANCE = R_TOLERANCE

    # Detect whether this is a full NODES workbook before falling back.
    try:
        rows = load_slice_records(path)
        full_nodes = True
    except Exception:
        try:
            with pd.ExcelFile(path) as xls:
                sheet_names = set(str(name).lower() for name in xls.sheet_names)
                if "slice_records" in sheet_names:
                    sheet = pd.read_excel(path, sheet_name="slice_records", header=[0, 1])
                    columns = [str(col).lower() for col in sheet.columns]
                    has_nodes_header = any("component" in str(col) for col in columns) or any(
                        str(col).startswith("starting") or str(col).startswith("solid") or str(col).startswith("liquor")
                        for col in columns
                    )
                    if has_nodes_header:
                        raise ValueError("NODES workbook detected; do not fall back to base-case")
        except Exception:
            pass
        # If load_slice_records fails, fall back to base‑case only if this is not a NODES workbook.
        rows = load_slice_records_basecase(path)
        full_nodes = False

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
        # --- NEW: base‑case loader ---
        rows = load_slice_records_basecase(path)

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

        # Full NODES → compute canonical Xc
        # Base‑case → canonical_xc = None
        canonical_xc = compute_canonical_xc(mrows, n_components) if full_nodes else None

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
        print("[QC] replicate deviation stats:")
        print(f"  min ΔR = {min(msd.replicate_devs):.3f}")
        print(f"  median ΔR = {median_dev:.3f}")
        print(f"  max ΔR = {max(msd.replicate_devs):.3f}")

    return mixtures

