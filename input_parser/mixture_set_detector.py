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
from typing import Dict, List

import numpy as np
import pandas as pd

from qc_settings import R_TOLERANCE

from .r_recalculator import compute_R_from_lever_rule, compute_row_R
from .slice_builder import detect_replicates_in_slice, detect_slices_by_chemistry as _volume_aware_detect_slices

QC_PRINT = False
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
XC_ANALYTICAL_TOL = 0.02  # mixture & replicate Xc tolerance
BASE_TOL = 0.05

xc_tol = XC_ANALYTICAL_TOL
r_tol = BASE_TOL


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
    """Use the explicit volume-aware implementation so distinct solvent-volume slices are preserved.

    The module previously had a duplicate R-gap heuristic with the same name; that heuristic
    collapses valid 3-volume mixtures when adjacent solvent-volume R values differ by less than
    the global tolerance. The real parser semantics are to retain distinct rel_sol_volume groups
    and treat repeated rows inside a volume as replicate measurements.
    """
    return _volume_aware_detect_slices(rows, tol)


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


def _is_degenerate_row(row: dict) -> bool:
    xs = row.get("Xs")
    if xs is None:
        return False

    xs_values = np.asarray(xs, dtype=float).reshape(-1)
    if xs_values.size == 0:
        return False

    dominant_component = float(np.max(xs_values))
    if dominant_component < 0.99:
        return False

    others = np.delete(xs_values, int(np.argmax(xs_values)))
    return others.size == 0 or np.max(np.abs(others)) <= 1e-6


def _reject_reason_for_row(row: dict, row_R: float | None, reference_R: float, tolerance: float) -> str:
    """Return a human-readable rejection reason for a row that is outside the acceptable band.

    The Rmax-0% label is reserved for physically degenerate rows, not for ordinary low-R points
    in a valid three-slice mixture. A normal low-but-physical slice should only be described as
    a replicate outlier.
    """
    if row_R is None:
        return "replicate outlier"

    r_value = float(row_R)
    if not np.isfinite(r_value):
        return "Rmax-0% record outside valid band"

    if _is_degenerate_row(row):
        return "Rmax-0% record outside valid band"

    if r_value <= max(0.0, reference_R - tolerance):
        return "replicate outlier"

    return "replicate outlier"


def detect_replicates(rows, canonical_xc, n_components, xc_tol, r_tol):
    accepted = []
    rejected = []

    grouped_rows = defaultdict(list)
    for row in rows:
        slice_id = row.get("slice_number")
        if slice_id is None:
            slice_id = 1
        grouped_rows[int(slice_id)].append(row)

    global total_reps, rejected_reps, replicate_devs
    if replicate_devs is None:
        replicate_devs = []

    for slice_id, slice_rows in sorted(grouped_rows.items()):
        if not slice_rows:
            continue

        row_Rs = []
        for row in slice_rows:
            Ri = compute_row_R(canonical_xc, row, n_components)
            if Ri is not None:
                row_Rs.append(Ri)

        if not row_Rs:
            accepted.extend(slice_rows)
            continue

        row_Rs = np.asarray(row_Rs, dtype=float)
        R_median = float(np.median(row_Rs))
        deviations = np.abs(row_Rs - R_median)
        mad_R = np.median(deviations)
        threshold = 2.0 * mad_R if np.isfinite(mad_R) and mad_R > 0.0 else 0.0
        if threshold <= 0.0:
            threshold = R_TOLERANCE

        canonical_R = float(np.mean(row_Rs))

        for row in slice_rows:
            dbg = debug_row(row, canonical_xc, canonical_R, n_components)
            R_i = compute_row_R(canonical_xc, row, n_components)
            delta_r = abs(R_i - R_median) if R_i is not None else 0.0
            replicate_devs.append(delta_r)
            total_reps += 1

        for row in slice_rows:
            dbg = debug_row(row, canonical_xc, canonical_R, n_components)
            xc_dev = dbg["xc_deviation_max"]
            R_i = compute_row_R(canonical_xc, row, n_components)
            delta_r = abs(R_i - R_median) if R_i is not None else 0.0

            if _is_degenerate_row(row):
                rejected_reps += 1
                rejected.append(row)
                mix_id = row.get("mixture_id", "UNKNOWN")
                qc_event(
                    f"mixture {mix_id} solvent_vol {slice_id} rejected: Rmax-0% record outside valid band "
                    f"(dR={abs(R_i - R_median) if R_i is not None else 0.0:.3f})"
                )
                continue

            if R_i is None:
                rejected_reps += 1
                rejected.append(row)
                continue

            if abs(R_i - R_median) > threshold:
                rejected_reps += 1
                rejected.append(row)
                mix_id = row.get("mixture_id", "UNKNOWN")
                reason = _reject_reason_for_row(row, R_i, R_median, threshold)
                qc_event(
                    f"mixture {mix_id} solvent_vol {slice_id} rejected: {reason} "
                    f"(dR={abs(R_i - R_median):.3f})"
                )
                continue

            if xc_dev < xc_tol and abs(R_i - R_median) <= threshold:
                accepted.append(row)
            else:
                rejected_reps += 1
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


def _count_accepted_measurements(rows: list[dict]) -> int:
    """Count only extra measurements beyond the first accepted row in each slice.

    A single valid reading in a solvent slice is not a replicate. Repeated rows in the
    same slice are replicate measurements, and their count is the number of extra rows
    beyond the first accepted row for that solvent volume.
    """
    if not rows:
        return 0

    slice_rows: Dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        slice_rows[int(row.get("slice_number", 1))].append(row)

    return sum(max(len(entries) - 1, 0) for entries in slice_rows.values())


def format_acceptance_summary(
    mixture_id: str,
    supplied_slice_count: int,
    accepted_slice_count: int,
    replicate_count: int,
    r_min: float,
    r_max: float,
) -> str:
    """Return a QC summary that distinguishes a genuine slice loss from normal replicate reduction.

    Real data can legitimately present two solvent-volume slices but still leave only one
    valid slice for the solver after noise filtering; in that case the wording must say that
    the accepted slice count is lower than the number supplied.
    """
    rejected_slice_count = max(supplied_slice_count - accepted_slice_count, 0)

    if supplied_slice_count > 0 and accepted_slice_count < supplied_slice_count:
        if accepted_slice_count == 1:
            slice_clause = (
                f"1 solvent vol retained for solver from {supplied_slice_count} supplied "
                f"({rejected_slice_count} rejected as noise)"
            )
        else:
            slice_clause = (
                f"{accepted_slice_count} solvent vols retained for solver from {supplied_slice_count} supplied "
                f"({rejected_slice_count} rejected as noise)"
            )
    elif accepted_slice_count == 1:
        slice_clause = "1 solvent vol retained for solver"
    else:
        slice_clause = f"{accepted_slice_count} solvent vols retained for solver"

    return (
        f"mixture {mixture_id} accepted: {slice_clause}, "
        f"{replicate_count} replicate rows, accepted R-range {r_min:.3f}-{r_max:.3f}"
    )


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
        detect_slices_by_chemistry(mrows, BASE_TOL)
        slice_count = len({int(row.get("slice_number")) for row in mrows if row.get("slice_number") is not None})
        if DEBUG:
            logger.debug("AFTER slice detection: mixture_id=%s, rows=%s, slices=%s", mid, len(mrows), slice_count)

        slice_groups: Dict[int, list[dict]] = defaultdict(list)
        for row in mrows:
            slice_groups[int(row.get("slice_number", 1))].append(row)

        for slice_id, slice_rows in list(slice_groups.items()):
            row_R_values = [float(row["R"]) for row in slice_rows if row.get("R") is not None]
            if row_R_values:
                median_R = np.median(row_R_values)
                deviations = np.abs(np.asarray(row_R_values) - median_R)
                mad_R = np.median(deviations)
                threshold = 2.0 * mad_R if np.isfinite(mad_R) and mad_R > 0.0 else 0.0
                if threshold <= 0.0:
                    threshold = BASE_TOL
                slice_rows[:] = [
                    row for row in slice_rows
                    if row.get("R") is None or abs(float(row["R"]) - median_R) <= threshold
                ]

        supplied_slice_count = len(slice_groups)
        mrows[:] = [row for rows in slice_groups.values() for row in rows]

        for slice_rows in slice_groups.values():
            replicate_groups = detect_replicates_in_slice(slice_rows, xc_tol=XC_ANALYTICAL_TOL, r_tol=BASE_TOL)
            for replicate_id, rep_rows in replicate_groups.items():
                for rep_row in rep_rows:
                    rep_row["replicate_number"] = replicate_id

        if len(mrows) == 1:
            global total_reps
            total_reps += 1
            validated[mid] = copy.deepcopy(mrows)
            r_values = [float(row["R"]) for row in mrows if row.get("R") is not None]
            r_min = min(r_values) if r_values else 0.0
            r_max = max(r_values) if r_values else 0.0
            replicate_count = _count_accepted_measurements(mrows)
            qc_event(
                format_acceptance_summary(
                    mid,
                    supplied_slice_count=supplied_slice_count,
                    accepted_slice_count=len({int(row.get("slice_number")) for row in mrows if row.get("slice_number") is not None}),
                    replicate_count=replicate_count,
                    r_min=r_min,
                    r_max=r_max,
                )
            )
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

        invalid_band_rejections = 0
        for row in rejected:
            slice_id = row.get("slice_number", "unknown")
            R_i = row.get("R")
            reference_R = float(np.median([float(r["R"]) for r in mrows if r.get("R") is not None])) if any(r.get("R") is not None for r in mrows) else 0.0
            reason = _reject_reason_for_row(row, R_i, reference_R, BASE_TOL)
            if reason == "Rmax-0% record outside valid band":
                invalid_band_rejections += 1
            qc_event(f"mixture {mid} solvent_vol {slice_id} rejected: {reason}")

        validated[mid] = copy.deepcopy(accepted)
        accepted_slice_count = len({int(row.get("slice_number")) for row in accepted if row.get("slice_number") is not None})
        accepted_replicate_count = _count_accepted_measurements(accepted)
        accepted_r_values = [float(row["R"]) for row in accepted if row.get("R") is not None]
        r_min = min(accepted_r_values) if accepted_r_values else 0.0
        r_max = max(accepted_r_values) if accepted_r_values else 0.0
        qc_event(
            format_acceptance_summary(
                mid,
                supplied_slice_count=supplied_slice_count,
                accepted_slice_count=accepted_slice_count,
                replicate_count=accepted_replicate_count,
                r_min=r_min,
                r_max=r_max,
            )
        )

    return validated


