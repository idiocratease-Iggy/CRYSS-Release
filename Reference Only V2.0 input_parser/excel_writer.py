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

import pandas as pd
from pathlib import Path
from typing import Dict

from .slice_builder import MixtureSet
from .statistics import (
    collect_mixture_statistics,
    collect_slice_statistics,
    collect_component_statistics,
    collect_file_statistics,
)


def write_diagnostic_excel(mixtures: Dict[str, MixtureSet], out_path: Path) -> None:
    """
    Writes a multi-sheet Excel file with:
    - canonical slice summary
    - mixture-level statistics
    - slice-level statistics
    - component-level statistics
    - file-level summary
    """

    # ------------------------------------------------------------
    # Sheet 1 — canonical slice summary
    # ------------------------------------------------------------
    slice_rows = []
    for mid, mset in mixtures.items():
        for s in mset.slices:
            row = {
                "mixture_id": mid,
                "slice_number": s.slice_number,
                "R": s.R,
            }

            # canonical Xc
            for i, xc in enumerate(s.Xc, start=1):
                row[f"Xc_{i}"] = xc

            # canonical Xs
            for i, xs in enumerate(s.Xs, start=1):
                row[f"Xs_{i}"] = xs

            # canonical Xl
            for i, xl in enumerate(s.Xl, start=1):
                row[f"Xl_{i}"] = xl

            slice_rows.append(row)

    df_slice_summary = pd.DataFrame(slice_rows)

    # ------------------------------------------------------------
    # Sheet 2 — mixture statistics
    # ------------------------------------------------------------
    df_mixture_stats = pd.DataFrame(
        [collect_mixture_statistics(m) for m in mixtures.values()]
    )

    # ------------------------------------------------------------
    # Sheet 3 — slice statistics
    # ------------------------------------------------------------
    slice_stats_rows = []
    for m in mixtures.values():
        slice_stats_rows.extend(collect_slice_statistics(m))
    df_slice_stats = pd.DataFrame(slice_stats_rows)

    # ------------------------------------------------------------
    # Sheet 4 — mixture statistics   ← NEW STAGE 2 SHEET
    # ------------------------------------------------------------
    mixture_stats_rows = []
    for m in mixtures.values():
        mixture_stats_rows.append(collect_mixture_statistics(m))
    df_mixture_stats = pd.DataFrame(mixture_stats_rows)

    # ------------------------------------------------------------
    # Sheet 5 — component statistics
    # ------------------------------------------------------------
    comp_stats_rows = []
    for m in mixtures.values():
        comp_stats_rows.extend(collect_component_statistics(m))
    df_component_stats = pd.DataFrame(comp_stats_rows)

    # ------------------------------------------------------------
    # Sheet 6 — file summary
    # ------------------------------------------------------------
    df_file_summary = pd.DataFrame([collect_file_statistics(mixtures)])

# ------------------------------------------------------------
# Write all sheets
# ------------------------------------------------------------
    with pd.ExcelWriter(out_path) as writer:
        df_slice_summary.to_excel(writer, sheet_name="slice_summary", index=False)
        df_mixture_stats.to_excel(writer, sheet_name="mixture_stats", index=False)
        df_slice_stats.to_excel(writer, sheet_name="slice_stats", index=False)
        df_component_stats.to_excel(writer, sheet_name="component_stats", index=False)
        df_file_summary.to_excel(writer, sheet_name="file_summary", index=False)
# ============================================================
# CRYSS Excel Workbook Structure (Stage‑2 Integrated)
# ============================================================
#
# The output workbook contains six sheets, ordered to reflect
# the CRYSS data‑processing pipeline:
#
#   slice → mixture → component → file
#
# This ordering ensures downstream tools (optimizer, PCA,
# mechanistic model) read the correct data in the correct order.
#
# ------------------------------------------------------------
# Sheet 1 — slice_summary
# ------------------------------------------------------------
# • One row per slice
# • Canonical Xs, Xl, Xc
# • Canonical R
# • Slice geometry metrics
# • Replicate‑filtered values
#
# ------------------------------------------------------------
# Sheet 2 — mixture_stats   (Stage‑2)
# ------------------------------------------------------------
# • One row per mixture
# • Mixture‑level canonical R
# • Mixture‑level Xs/Xl medians across slices
# • Slice‑to‑slice variation (std)
# • Lever‑rule consistency (Xc median/std)
# • Purity index (mean |Xs − Xl|)
# • This sheet provides the mixture fingerprint used by the
#   optimizer and PCA.
#
# ------------------------------------------------------------
# Sheet 3 — slice_stats
# ------------------------------------------------------------
# • One row per slice per mixture
# • Stage‑1 slice‑level diagnostics
# • Replicate variation
# • Lever‑rule per component
# • Slice geometry indices
#
# ------------------------------------------------------------
# Sheet 4 — component_stats
# ------------------------------------------------------------
# • One row per component per mixture
# • Component‑level enrichment trends
# • Component‑level purity contributions
# • Component‑level slice variation
# • (Upgraded further in Stage‑3)
#
# ------------------------------------------------------------
# Sheet 5 — file_summary
# ------------------------------------------------------------
# • One row summarising the entire file
# • Total mixtures, slices, components
# • QC flags
# • Timing, metadata
#
# ============================================================
# End of workbook structure documentation
# ============================================================

