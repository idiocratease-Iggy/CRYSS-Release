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


from .r_recalculator import compute_R_from_lever_rule, compute_row_R
from .salt_consistency import detail_lines as _salt_detail_lines, summary_lines as _salt_summary_lines
from .slice_builder import detect_replicates_in_slice, detect_slices_by_chemistry as _volume_aware_detect_slices

QC_PRINT = True
_qc_seen = set()
total_reps = 0
rejected_reps = 0
replicate_devs = []
salt_report = None
# Ledger of every rejected row with its reason; None until a parse run initialises it.
rejection_log = None

RMAX_0PCT_UNCONSTRAINED_REASON = "Rmax - 0% unconstrained"
LOW_R_VALID_BAND_REASON = "below valid-band threshold relative to slice reference"
XS_XL_NOISE_REASON = "Xs/Xl outside noise tolerance"
XC_ANALYTICAL_REASON = "Xc outside analytical tolerance"
R_OUTLIER_PREFILTER_REASON = "R outlier within slice"
REPLICATE_OUTLIER_REASON = "replicate outlier"
NO_VALID_R_REASON = "R could not be computed"


def _log_rejection(row: dict, reason: str, *, count_row: bool = False) -> None:
    """Record a rejected row. count_row also bumps the counters for rows rejected before replicate detection."""
    global rejection_log, total_reps, rejected_reps
    if rejection_log is None:
        rejection_log = []
    rejection_log.append({"mixture_id": str(row.get("mixture_id", "UNKNOWN")), "slice": row.get("slice_number"), "reason": reason})
    rejected_reps += 1
    if count_row:
        total_reps += 1


def qc_event(msg: str):
    if QC_PRINT:
        if msg not in _qc_seen:
            _qc_seen.add(msg)
            print(f"[QC] {msg}")





DEBUG = False
logger = logging.getLogger(__name__)

# Xc replicate tolerance (analytical error) vs Xs/Xl noise floor (BASE_TOL).
XC_ANALYTICAL_TOL = 0.02  # mixture & replicate Xc tolerance
BASE_TOL = 0.05


def dynamic_r_tolerance() -> float:
    """R-slice tolerance propagated (in quadrature) from the Xc analytical error and the Xs/Xl noise."""
    return float(np.sqrt(XC_ANALYTICAL_TOL ** 2 + 2.0 * BASE_TOL ** 2))


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


def _normalise_tag(value) -> str | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _experiment_tag_for_row(row: dict) -> str | None:
    tags: list[str] = []
    for key in ("salt_derivative", "Salt/Derivative", "salt", "derivative"):
        tag = _normalise_tag(row.get(key))
        if tag is not None:
            tags.append(tag)
    if not tags:
        return None
    return "|".join(tags)


def _has_explicit_xc(row: dict) -> bool:
    if row.get("Xc") is not None:
        return True
    return any(str(key).startswith("xc_") for key in row)


def group_by_mixture_id(rows: list[dict]) -> dict[str, list[dict]]:
    by_mid: dict[str, list[dict]] = defaultdict(list)
    distinct_tags_by_mid: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        mid = str(row.get("mixture_id") or "UNKNOWN")
        by_mid[mid].append(row)
        tag = _experiment_tag_for_row(row)
        if tag is not None:
            distinct_tags_by_mid[mid].add(tag)

    groups: dict[str, list[dict]] = defaultdict(list)
    for mid, mrows in by_mid.items():
        tags = distinct_tags_by_mid.get(mid, set())
        for row in mrows:
            tag = _experiment_tag_for_row(row)
            if tag is not None and len(tags) > 1:
                key = f"{mid}::{tag}"
            else:
                key = mid
            groups[key].append(row)
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


def _is_unconstrained_region_row(row: dict) -> bool:
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


def _effective_row_R(row: dict) -> float | None:
    """Return the row's effective R value, computing it from composition when the raw sheet omits R."""
    raw_value = row.get("R")
    if raw_value not in (None, "", "unknown"):
        try:
            value = float(raw_value)
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass

    x_c = row.get("Xc")
    x_s = row.get("Xs")
    x_l = row.get("Xl")
    if x_c is None and any(key.startswith("xc_") for key in row):
        x_c = [float(row[f"xc_{index}"]) for index in range(1, len([key for key in row if key.startswith("xc_")]) + 1)]
    if x_s is None and any(key.startswith("xs_") for key in row):
        x_s = [float(row[f"xs_{index}"]) for index in range(1, len([key for key in row if key.startswith("xs_")]) + 1)]
    if x_l is None and any(key.startswith("xl_") for key in row):
        x_l = [float(row[f"xl_{index}"]) for index in range(1, len([key for key in row if key.startswith("xl_")]) + 1)]

    if x_c is not None and x_s is not None and x_l is not None:
        try:
            computed = compute_R_from_lever_rule(x_c, x_s, x_l)
            if computed is not None and np.isfinite(float(computed)):
                return float(computed)
        except Exception:
            pass

    return None


def _reject_reason_for_row(row: dict, row_R: float | None, reference_R: float, tolerance: float) -> str:
    """Return a human-readable rejection reason for a row that is outside the acceptable band.

    The Rmax-0% label is reserved for physically unconstrained rows, not for ordinary low-R points
    in a valid three-slice mixture. A normal low-but-physical slice should only be described as
    a valid-band low-R reject.
    """
    if _is_unconstrained_region_row(row):
        return RMAX_0PCT_UNCONSTRAINED_REASON

    if row.get("_reject_reason") == XS_XL_NOISE_REASON:
        return XS_XL_NOISE_REASON

    effective_row_R = row_R if row_R is not None else _effective_row_R(row)
    if effective_row_R is None:
        return "replicate outlier"

    r_value = float(effective_row_R)
    if not np.isfinite(r_value):
        return RMAX_0PCT_UNCONSTRAINED_REASON

    threshold = max(0.0, float(reference_R) - float(tolerance))
    if r_value <= threshold:
        return LOW_R_VALID_BAND_REASON

    return "replicate outlier"


def _r_outlier_threshold(mad_R: float) -> float:
    """Replicate R outlier band: 2*MAD, but never tighter than the propagated R tolerance.

    Tight replicate groups would otherwise shrink the MAD band below measurement noise and
    discard good rows; only genuinely wild rows should fall outside this band.
    """
    mad_band = 2.0 * float(mad_R) if np.isfinite(mad_R) else 0.0
    return max(mad_band, dynamic_r_tolerance())


def _xs_xl_reference(slice_rows: list[dict], key: str) -> np.ndarray | None:
    """Per-slice median of Xs or Xl, or None when rows do not carry comparable vectors."""
    vectors = []
    for row in slice_rows:
        value = row.get(key)
        if value is None:
            continue
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.size and np.all(np.isfinite(vector)):
            vectors.append(vector)
    if not vectors or len({v.size for v in vectors}) != 1:
        return None
    return np.median(np.vstack(vectors), axis=0)


def _exceeds_noise_tolerance(row: dict, key: str, reference: np.ndarray | None) -> bool:
    if reference is None or row.get(key) is None:
        return False
    vector = np.asarray(row[key], dtype=float).reshape(-1)
    if vector.size != reference.size:
        return False
    return float(np.max(np.abs(vector - reference))) > BASE_TOL


def detect_replicates(rows, canonical_xc, n_components, xc_tol):
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
        threshold = _r_outlier_threshold(mad_R)

        canonical_R = float(np.mean(row_Rs))
        xs_ref = _xs_xl_reference(slice_rows, "Xs")
        xl_ref = _xs_xl_reference(slice_rows, "Xl")

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

            if _is_unconstrained_region_row(row):
                _log_rejection(row, RMAX_0PCT_UNCONSTRAINED_REASON)
                rejected.append(row)
                mix_id = row.get("mixture_id", "UNKNOWN")
                qc_event(
                    f"mixture {mix_id} solvent_vol {slice_id} rejected: {RMAX_0PCT_UNCONSTRAINED_REASON} "
                    f"(dR={abs(R_i - R_median) if R_i is not None else 0.0:.3f})"
                )
                continue

            if R_i is None:
                _log_rejection(row, NO_VALID_R_REASON)
                rejected.append(row)
                continue

            # Xs/Xl scatter only rejects a row when it also pushes R outside the R tolerance.
            if abs(R_i - R_median) > dynamic_r_tolerance() and (
                _exceeds_noise_tolerance(row, "Xs", xs_ref) or _exceeds_noise_tolerance(row, "Xl", xl_ref)
            ):
                row["_reject_reason"] = XS_XL_NOISE_REASON
                _log_rejection(row, XS_XL_NOISE_REASON)
                rejected.append(row)
                qc_event(f"mixture {row.get('mixture_id', 'UNKNOWN')} solvent_vol {slice_id} rejected: {XS_XL_NOISE_REASON}")
                continue

            if abs(R_i - R_median) > threshold:
                rejected.append(row)
                mix_id = row.get("mixture_id", "UNKNOWN")
                reason = _reject_reason_for_row(row, R_i, R_median, threshold)
                _log_rejection(row, reason)
                qc_event(
                    f"mixture {mix_id} solvent_vol {slice_id} rejected: {reason} "
                    f"(dR={abs(R_i - R_median):.3f})"
                )
                continue

            if xc_dev < xc_tol and abs(R_i - R_median) <= threshold:
                accepted.append(row)
            else:
                _log_rejection(row, XC_ANALYTICAL_REASON)
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
            _log_rejection(row, XC_ANALYTICAL_REASON, count_row=True)

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


def _summary_reason_counts(rows: list[dict], mixtures: dict) -> dict[str, int]:
    """Count the actual rejection reasons for the compact parser summary.

    The raw workbook rows are not guaranteed to have a stable external identifier or the same
    slice/replicate metadata that the validated parser state assigns during slice grouping. Summary
    classification therefore matches on a normalized row fingerprint built from the mixture, slice,
    and composition values rather than on mixture_id alone, which collapses valid accepted rows and
    rejected rows into the same bucket when the workbook rows are missing source IDs.
    """
    if rejection_log is not None:
        ledger_counts: dict[str, int] = defaultdict(int)
        for entry in rejection_log:
            ledger_counts[entry["reason"]] += 1
        return ledger_counts

    _ensure_slice_metadata(rows)

    def _vector_from_row(row: dict, key: str) -> tuple[float, ...]:
        raw = row.get(key)
        if raw is not None:
            if isinstance(raw, (list, tuple, np.ndarray)):
                return tuple(float(float(v)) for v in raw)
            return (float(raw),)

        values: list[float] = []
        index = 1
        while True:
            value = row.get(f"{key.lower()}_{index}")
            if value is None:
                break
            values.append(float(value))
            index += 1
        return tuple(values)

    def _identity_for_summary_row(row: dict, mixture_id: str) -> tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        xc_values = tuple(np.round(_vector_from_row(row, "Xc"), 12)) if row.get("Xc") is not None or any(key.startswith("xc_") for key in row) else tuple()
        xs_values = tuple(np.round(_vector_from_row(row, "Xs"), 12)) if row.get("Xs") is not None or any(key.startswith("xs_") for key in row) else tuple()
        xl_values = tuple(np.round(_vector_from_row(row, "Xl"), 12)) if row.get("Xl") is not None or any(key.startswith("xl_") for key in row) else tuple()
        return (
            str(mixture_id),
            xc_values,
            xs_values,
            xl_values,
        )

    accepted_keys: set[tuple[str, tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = set()
    for mixture_id, mixture in mixtures.items():
        for slice_record in mixture.slices:
            for row in slice_record.raw_rows:
                accepted_keys.add(_identity_for_summary_row(row, mixture_id))

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        mixture_id = str(row.get("mixture_id") or "")
        if not mixture_id:
            continue
        if mixture_id not in mixtures:
            continue
        if _identity_for_summary_row(row, mixture_id) in accepted_keys:
            continue

        ref_R = None
        same_mixture_rows = [r for r in rows if str(r.get("mixture_id")) == mixture_id]
        valid_ref_values = [value for value in (_effective_row_R(r) for r in same_mixture_rows) if value is not None]
        if valid_ref_values:
            ref_R = float(np.median(valid_ref_values))
        if ref_R is None:
            ref_R = 0.0

        reason = _reject_reason_for_row(row, _effective_row_R(row), ref_R, dynamic_r_tolerance())
        if reason in {RMAX_0PCT_UNCONSTRAINED_REASON, LOW_R_VALID_BAND_REASON}:
            counts[reason] += 1
        else:
            counts["replicate outlier / other reject"] += 1

    return counts


def build_parser_summary(
    rows: list[dict],
    mixtures: dict,
    *,
    rejected_reps: int | None = None,
    total_reps: int | None = None,
    replicate_devs: list[float] | None = None,
) -> str:
    """Return a compact human-readable parser summary for terminal and GUI display."""
    _ensure_slice_metadata(rows)
    report = globals().get("salt_report")
    salt_rejected = report.rejected_count if report is not None else 0
    total_rows = len(rows) + salt_rejected
    total_mixtures = len(mixtures)

    slice_counts: dict[int, int] = {}
    for mixture in mixtures.values():
        count = len(mixture.slices)
        slice_counts[count] = slice_counts.get(count, 0) + 1

    replicates_absorbed = 0
    for mixture in mixtures.values():
        for slice_record in mixture.slices:
            replicates_absorbed += max(len(slice_record.raw_rows) - 1, 0)

    if rejected_reps is None:
        rejected_reps = globals().get("rejected_reps", 0)
    if total_reps is None:
        total_reps = globals().get("total_reps", 0)
    if replicate_devs is None:
        replicate_devs = globals().get("replicate_devs", [])

    if total_reps > 0:
        rejection_rate = rejected_reps / total_reps
    else:
        rejection_rate = 0.0

    if replicate_devs:
        dR_values = [float(d) for d in replicate_devs if np.isfinite(d)]
        if dR_values:
            dR_min = min(dR_values)
            dR_median = float(np.median(dR_values))
            dR_max = max(dR_values)
        else:
            dR_min = dR_median = dR_max = 0.0
    else:
        dR_min = dR_median = dR_max = 0.0

    distribution_lines: list[str] = []
    for slice_count in sorted(slice_counts):
        count = slice_counts[slice_count]
        if count == 1:
            distribution_lines.append(f"1 mixture with {slice_count} composition slice")
        else:
            distribution_lines.append(f"{count} mixtures with {slice_count} composition slices")
    if not distribution_lines:
        distribution_lines.append("0 mixtures retained")

    reason_counts = _summary_reason_counts(rows, mixtures)
    reason_labels = {
        RMAX_0PCT_UNCONSTRAINED_REASON: "Rmax - 0% unconstrained",
        LOW_R_VALID_BAND_REASON: "below valid-band threshold",
    }
    reason_parts = [
        f"{count} x {reason_labels.get(reason, reason)}"
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
        if count
    ]
    if rejection_log is not None:
        rejected_reps = len(rejection_log)
    reason_text = "; ".join(reason_parts) if reason_parts else "none"

    lines = [
        "Input File Parser Summary",
        "========================",
        f"No. of records read = {total_rows}",
        f"No. of unique mixture IDs = {total_mixtures}",
    ]
    lines.extend(distribution_lines)
    lines.extend(
        [
            f"Rejected records outside valid band = {rejected_reps}",
            f"  reason(s): {reason_text}",
            f"No of Replicates absorbed = {replicates_absorbed}",
            f"Replicate rejection rate = {rejection_rate:.2f} ({rejected_reps}/{total_reps})",
            "Within-slice replicate dR stats:",
            f"  min dR = {dR_min:.3f}",
            f"  median dR = {dR_median:.3f}",
            f"  max dR = {dR_max:.3f}",
        ]
    )
    salt_lines = _salt_summary_lines(report)
    if salt_lines:
        lines.append("")
        lines.extend(salt_lines)
    return "\n".join(lines)


def build_verbose_qc_report(rows: list[dict], mixtures: dict) -> str:
    """Return the retained slice composition summary used for fitting.

    This report is intentionally the same compact per-mixture slice record that is emitted in the
    terminal during parsing and that downstream fitting logic consumes. The detailed reject-by-band
    list is not included because it is diagnostic noise for the saved export and does not represent
    the actual accepted fit data.
    """
    lines: list[str] = [
        "Detailed Parser QC Trace",
        "=======================",
        f"records_in_file={len(rows) + (globals().get('salt_report').rejected_count if globals().get('salt_report') else 0)}",
        f"mixtures_retained={len(mixtures)}",
        f"Xc_tolerance={XC_ANALYTICAL_TOL * 100:.1f}% (analytical error: Xc agreement within a mixture_id)",
        f"Xs_Xl_tolerance={BASE_TOL * 100:.1f}% (noise: Xs/Xl row acceptance)",
        f"R_tolerance={dynamic_r_tolerance() * 100:.1f}% (propagated: sqrt(Xc^2 + 2*noise^2), R fallback threshold)",
        f"Rmax_0pct_guard={dynamic_r_tolerance() * 100:.1f}% (unconstrained region gate)",
        "",
    ]
    lines.extend(_salt_detail_lines(globals().get("salt_report")))

    for mixture_id, mixture in sorted(mixtures.items()):
        lines.append(f"mixture {mixture_id}:")
        for slice_record in mixture.slices:
            canonical_R = float(slice_record.R)
            recovery_pct = canonical_R * 100.0
            xc_pct = [float(v) * 100.0 for v in slice_record.Xc]
            xs_pct = [float(v) * 100.0 for v in slice_record.Xs]
            xl_pct = [float(v) * 100.0 for v in slice_record.Xl]
            lines.append(
                f"  slice {slice_record.slice_number}: rows={len(slice_record.raw_rows)}, "
                f"recovery={recovery_pct:.4f}%, "
                f"Xc=[{', '.join(f'{v:.4f}%' for v in xc_pct)}], "
                f"Xs=[{', '.join(f'{v:.4f}%' for v in xs_pct)}], "
                f"Xl=[{', '.join(f'{v:.4f}%' for v in xl_pct)}]"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_mixture_sets(rows: list[dict], n_components: int) -> dict[str, list[dict]]:
    _ensure_slice_metadata(rows)
    rows = [row for row in rows if _has_explicit_xc(row)]
    if not rows:
        return {}

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
                threshold = _r_outlier_threshold(mad_R)
                kept_rows = []
                for row in slice_rows:
                    if row.get("R") is None or abs(float(row["R"]) - median_R) <= threshold:
                        kept_rows.append(row)
                    else:
                        _log_rejection(row, R_OUTLIER_PREFILTER_REASON, count_row=True)
                slice_rows[:] = kept_rows

        supplied_slice_count = len(slice_groups)
        mrows[:] = [row for rows in slice_groups.values() for row in rows]

        for slice_rows in slice_groups.values():
            replicate_groups = detect_replicates_in_slice(slice_rows, xc_tol=XC_ANALYTICAL_TOL)
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
        filtered = filter_rows_by_xc(mrows, n_components, XC_ANALYTICAL_TOL)
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
        )

        invalid_band_rejections = 0
        for row in rejected:
            slice_id = row.get("slice_number", "unknown")
            R_i = row.get("R")
            reference_R = float(np.median([float(r["R"]) for r in mrows if r.get("R") is not None])) if any(r.get("R") is not None for r in mrows) else 0.0
            reason = _reject_reason_for_row(row, R_i, reference_R, dynamic_r_tolerance())
            if reason == RMAX_0PCT_UNCONSTRAINED_REASON:
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


