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





import copy
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from qc_settings import R_TOLERANCE

from .r_recalculator import compute_R_from_lever_rule, compute_row_R

QC_PRINT = True
_qc_seen = set()
total_reps = 0
rejected_reps = 0
replicate_devs = []


def qc_event(msg: str):
    if QC_PRINT:
        if msg not in _qc_seen:
            _qc_seen.add(msg)
            print(f"[QC] {msg}")





DEBUG = False
logger = logging.getLogger(__name__)

# R, Xs, Xl all use BASE_TOL as the noise floor.
XC_ANALYTICAL_TOL = 0.02
BASE_TOL = 0.05

xc_tol = XC_ANALYTICAL_TOL
r_tol = BASE_TOL

# mixture_set_detector.py

import numpy as np
from typing import List, Dict

XC_ANALYTICAL_TOL = 0.02  # mixture & replicate Xc tolerance


def _ensure_slice_metadata(rows: List[dict]) -> None:
    """Give each row a safe default slice/replicate identity before slice grouping."""
    for row in rows:
        if row.get("slice_number") is None:
            row["slice_number"] = 1
        if row.get("replicate_number") is None:
            row["replicate_number"] = 0


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def detect_slices_by_chemistry(rows: List[dict], tol: float = BASE_TOL) -> List[dict]:
    """Assign slice_number by computed R-driven gaps against the current slice representative."""
    _ensure_slice_metadata(rows)

    if not rows:
        return rows

    indexed = []
    for original_index, row in enumerate(rows):
        if row.get("R") is None:
            continue
        try:
            row_R = float(row["R"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(row_R):
            continue
        indexed.append((original_index, row, row_R))

    if not indexed:
        return rows

    indexed_sorted = sorted(indexed, key=lambda item: item[2], reverse=True)

    adaptive_R_tol = max(float(tol), 1e-9)
    current_slice = 1
    slice_rep_R = indexed_sorted[0][2]
    indexed_sorted[0][1]["slice_number"] = current_slice

    for _, row, R in indexed_sorted[1:]:
        if abs(R - slice_rep_R) > adaptive_R_tol:
            current_slice += 1
            slice_rep_R = R
        row["slice_number"] = current_slice

    for original_index, row, _ in indexed_sorted:
        rows[original_index]["slice_number"] = row["slice_number"]

    return rows


def group_by_mixture_id(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        mid = str(row["mixture_id"])
        groups[mid].append(row)
    return groups


def debug_row(row, canonical_xc, canonical_R, n_components):
    """Return a dict of debug info for a single row."""
    xc_devs = [
        abs(row[f"xc_{i+1}"] - canonical_xc[i])
        for i in range(n_components)
    ]
    Ri = compute_row_R(canonical_xc, row, n_components)
    r_dev = abs(Ri - canonical_R) if Ri is not None else None

    return {
        "row_id": row.get("id", None),
        "xc_deviation_per_component": xc_devs,
        "xc_deviation_max": max(xc_devs),
        "R_row": Ri,
        "R_deviation": r_dev,
        "slice_number": row.get("slice_number", None),
        "raw_row": row,
    }


def detect_replicates(rows, canonical_xc, n_components, xc_tol, r_tol):
    accepted = []
    rejected = []

    # Compute all R values
    row_Rs = []
    for row in rows:
        Ri = compute_row_R(canonical_xc, row, n_components)
        if Ri is not None:
            row_Rs.append(Ri)

    if not row_Rs:
        return rows, []

    row_Rs = np.array(row_Rs)

    # --- Outlier rejection for R ---
    R_median = float(np.median(row_Rs))
    deviations = np.abs(row_Rs - R_median)
    mad_R = np.median(deviations)
    threshold = 2.0 * mad_R

    mask = deviations <= threshold
    filtered_Rs = row_Rs[mask]

    if filtered_Rs.size == 0:
        filtered_Rs = row_Rs

    canonical_R = float(np.mean(filtered_Rs))

    global total_reps, rejected_reps, replicate_devs
    replicate_devs = []

    for row in rows:
        dbg = debug_row(row, canonical_xc, canonical_R, n_components)
        R_i = compute_row_R(canonical_xc, row, n_components)
        delta_r = abs(R_i - R_median) if R_i is not None else 0.0
        replicate_devs.append(delta_r)
        total_reps += 1

    for row in rows:
        dbg = debug_row(row, canonical_xc, canonical_R, n_components)

        xc_dev = dbg["xc_deviation_max"]
        R_i = compute_row_R(canonical_xc, row, n_components)
        r_dev = dbg["R_deviation"]
        delta_r = abs(R_i - R_median) if R_i is not None else 0.0

        if abs(R_i - R_median) > R_TOLERANCE:
            rejected_reps += 1
            rejected.append(row)
            mix_id = row.get("mixture_id", "UNKNOWN")
            slice_id = row.get("slice_number", "unknown")
            qc_event(
                f"mixture {mix_id} solvent_vol {slice_id} rejected: replicate outlier "
                f"(ΔR={abs(R_i - R_median):.3f})"
            )
            continue

        if xc_dev < xc_tol and (R_i is not None and abs(R_i - R_median) <= R_TOLERANCE):
            accepted.append(row)
        else:
            rejected.append(row)

    return accepted, rejected


# Global replicate rejection counters used for final QC summary.
# These are intentionally lightweight and terminal-only.



def compute_canonical_xc(rows: list[dict], n_components: int) -> list[float]:
    """
    Computes canonical Xc using mean instead of median,
    with optional outlier rejection per component.
    """
    xc_matrix = []
    for row in rows:
        xc = [row[f"xc_{i+1}"] for i in range(n_components)]
        xc_matrix.append(xc)

    xc_array = np.array(xc_matrix)

    # --- Optional: per-component outlier rejection ---
    # We compute deviations from the median (robust) and remove rows
    # where any component deviates too far. This stabilises the mean.
    median = np.median(xc_array, axis=0)
    deviations = np.abs(xc_array - median)

    # Threshold: 2× median absolute deviation (MAD)
    mad = np.median(deviations, axis=0)
    threshold = 2.0 * mad

    # Keep rows where all components are within threshold
    mask = np.all(deviations <= threshold, axis=1)
    filtered = xc_array[mask]

    # If filtering removes everything, fall back to original data
    if filtered.size == 0:
        filtered = xc_array

    # --- Canonical Xc using mean ---
    canonical = np.mean(filtered, axis=0)

    return canonical.tolist()




def filter_rows_by_xc(rows: list[dict], n_components: int, tol: float) -> list[dict]:
    median_xc = compute_canonical_xc(rows, n_components)
    accepted = []
    rejected = []

    for row in rows:
        xc_devs = [
            abs(row[f"xc_{i+1}"] - median_xc[i])
            for i in range(n_components)
        ]
        max_dev = max(xc_devs)

        if max_dev <= tol:
            accepted.append(row)
        else:
            rejected.append(row)

    return accepted



def build_mixture_sets(rows: list[dict], n_components: int) -> dict[str, list[dict]]:
    _ensure_slice_metadata(rows)
    grouped = group_by_mixture_id(rows)
    validated = {}

    for mid, mrows in grouped.items():
        for row_index, row in enumerate(mrows):
            xs_values = []
            xl_values = []
            xc_values = []
            xs_indices = sorted({int(key.split("_")[-1]) for key in row if key.startswith("xs_")})
            xl_indices = sorted({int(key.split("_")[-1]) for key in row if key.startswith("xl_")})
            xc_indices = sorted({int(key.split("_")[-1]) for key in row if key.startswith("xc_")})
            for idx in xs_indices:
                if f"xs_{idx}" in row:
                    xs_values.append(row[f"xs_{idx}"])
            for idx in xl_indices:
                if f"xl_{idx}" in row:
                    xl_values.append(row[f"xl_{idx}"])
            for idx in xc_indices:
                if f"xc_{idx}" in row:
                    xc_values.append(row[f"xc_{idx}"])
            if xs_values:
                row["Xs"] = xs_values
            if xl_values:
                row["Xl"] = xl_values
            if xc_values:
                row["Xc"] = xc_values

            value = row.get("R")
            if value is None or (isinstance(value, str) and not value.strip()) or value == 0.0:
                row["R"] = None
            elif isinstance(value, str):
                try:
                    row["R"] = float(value)
                except ValueError:
                    row["R"] = None
            elif pd.isna(value):
                row["R"] = None

            if row.get("R") is None and row.get("Xc") is not None and row.get("Xs") is not None and row.get("Xl") is not None:
                computed = compute_R_from_lever_rule(row["Xc"], row["Xs"], row["Xl"])
                if computed is not None:
                    row["R"] = float(computed)

            if DEBUG:
                logger.debug("AFTER mixture grouping:")
                logger.debug("  mixture_id=%s", mid)
                logger.debug("  row_index=%s", row_index)
                logger.debug("  Xs=%s", xs_values)
                logger.debug("  Xl=%s", xl_values)

    # R, Xs, Xl all use BASE_TOL as the noise floor.
    xc_tol = BASE_TOL
    r_tol = BASE_TOL

    for mid, mrows in grouped.items():
        row_R_values = [float(row["R"]) for row in mrows if row.get("R") is not None]
        if row_R_values:
            median_R = np.median(row_R_values)
            deviations = np.abs(np.asarray(row_R_values) - median_R)
            mad_R = np.median(deviations)
            threshold = 2.0 * mad_R if np.isfinite(mad_R) and mad_R > 0.0 else 0.0
            if threshold > 0.0:
                filtered = [
                    row for row in mrows
                    if row.get("R") is None or abs(float(row["R"]) - median_R) <= threshold
                ]
                if filtered:
                    mrows[:] = filtered

        detect_slices_by_chemistry(mrows, BASE_TOL)
        slice_count = len({int(row.get("slice_number")) for row in mrows if row.get("slice_number") is not None})
        if DEBUG:
            logger.debug("AFTER slice detection: mixture_id=%s, rows=%s, slices=%s", mid, len(mrows), slice_count)

        if len(mrows) == 1:
            global total_reps
            total_reps += 1
            validated[mid] = copy.deepcopy(mrows)
            r_values = [float(row["R"]) for row in mrows if row.get("R") is not None]
            r_min = min(r_values) if r_values else 0.0
            r_max = max(r_values) if r_values else 0.0
            qc_event(f"mixture {mid} accepted: {slice_count} solvent vols, R-range {r_min:.3f}–{r_max:.3f}")
            continue

        # Xc filtering uses xc_tol
        filtered = filter_rows_by_xc(mrows, n_components, xc_tol)
        if not filtered:
            qc_event(f"mixture {mid} rejected: no valid Xc rows")
            continue

        canonical_xc = compute_canonical_xc(filtered, n_components)

        # Replicate detection uses the same BASE_TOL noise floor as Xs/Xl.
        accepted, rejected = detect_replicates(
            filtered,
            canonical_xc,
            n_components,
            xc_tol=XC_ANALYTICAL_TOL,
            r_tol=BASE_TOL
        )

        for row in rejected:
            slice_id = row.get("slice_number", "unknown")
            qc_event(f"mixture {mid} solvent_vol {slice_id} rejected: replicate outlier")

        validated[mid] = copy.deepcopy(mrows)
        accepted_r_values = [float(row["R"]) for row in accepted if row.get("R") is not None]
        r_min = min(accepted_r_values) if accepted_r_values else 0.0
        r_max = max(accepted_r_values) if accepted_r_values else 0.0
        qc_event(f"mixture {mid} accepted: {slice_count} solvent vols, accepted R-range {r_min:.3f}–{r_max:.3f}")

    return validated


