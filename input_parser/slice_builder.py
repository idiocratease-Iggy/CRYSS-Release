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
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
import numpy as np

from .r_recalculator import canonical_R_for_slice

DEBUG = False
logger = logging.getLogger(__name__)


@dataclass
class SliceRecord:
    mixture_id: str
    slice_number: int
    R: float
    Xc: List[float]
    Xs: List[float]
    Xl: List[float]
    raw_rows: List[Dict]


@dataclass
class MixtureSet:
    mixture_id: str
    slices: List[SliceRecord]


# slice_builder.py

import numpy as np
from typing import List, Dict

BASE_TOL = 0.05      # R, Xs, Xl all use BASE_TOL as the noise floor.
XC_ANALYTICAL_TOL = 0.02


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _ensure_slice_metadata(rows: List[dict]) -> None:
    """Normalize missing slice metadata before row grouping."""
    for row in rows:
        if row.get("slice_number") is None:
            row["slice_number"] = 1
        if row.get("replicate_number") is None:
            row["replicate_number"] = 0


def detect_slices_by_chemistry(rows: List[dict], tol: float = BASE_TOL) -> List[dict]:
    """Assign slice_number by Xs/Xl chemistry for rows within the same mixture."""
    _ensure_slice_metadata(rows)

    if not rows:
        return rows

    def xs_xl_values(row: dict) -> tuple[list[float], list[float]]:
        xs_values = []
        xl_values = []
        xs_indices = sorted({int(key.split("_")[-1]) for key in row if key.startswith("xs_")})
        xl_indices = sorted({int(key.split("_")[-1]) for key in row if key.startswith("xl_")})
        for idx in xs_indices:
            if f"xs_{idx}" in row:
                xs_values.append(row[f"xs_{idx}"])
        for idx in xl_indices:
            if f"xl_{idx}" in row:
                xl_values.append(row[f"xl_{idx}"])
        if not xs_values and row.get("Xs") is not None:
            xs_values = np.asarray(row.get("Xs"), dtype=float).tolist()
        if not xl_values and row.get("Xl") is not None:
            xl_values = np.asarray(row.get("Xl"), dtype=float).tolist()
        return xs_values, xl_values

    slice_reps: list[dict] = []
    for row_index, row in enumerate(rows):
        mixture_id = row.get("mixture_id", "UNKNOWN")
        xs, xl = xs_xl_values(row)
        slice_number_before = row.get("slice_number")

        placed = False
        for sn, rep in enumerate(slice_reps, start=1):
            rep_xs = np.asarray(rep.get("Xs") or [], dtype=float)
            rep_xl = np.asarray(rep.get("Xl") or [], dtype=float)

            xs_diff = 0.0 if len(xs) == 0 or rep_xs.size == 0 else max_abs_diff(np.asarray(xs, dtype=float), rep_xs)
            xl_diff = 0.0 if len(xl) == 0 or rep_xl.size == 0 else max_abs_diff(np.asarray(xl, dtype=float), rep_xl)

            if xs_diff < tol and xl_diff < tol:
                row["slice_number"] = sn
                placed = True
                break

        if not placed:
            row["slice_number"] = len(slice_reps) + 1
            slice_reps.append(copy.deepcopy(row))

        if DEBUG:
            logger.debug("AFTER slice detection:")
            logger.debug("  mixture_id=%s", mixture_id)
            logger.debug("  row_index=%s", row_index)
            logger.debug("  slice_number=%s", row.get("slice_number"))
            logger.debug("  Xs=%s", xs)
            logger.debug("  Xl=%s", xl)

    return rows


def group_rows_by_slice_composition(rows: List[dict]) -> Dict[int, List[dict]]:
    """
    Chemistry‑first slice grouping:
    - Group rows into slices by Xs/Xl similarity within BASE_TOL.
    - Assign synthetic slice_number.
    """
    slices: Dict[int, List[dict]] = {}
    slice_reps: List[dict] = []  # representative row per slice

    for row in rows:
        xs = np.array(row["Xs"], dtype=float) if row.get("Xs") is not None else None
        xl = np.array(row["Xl"], dtype=float) if row.get("Xl") is not None else None

        placed = False
        for sn, rep in enumerate(slice_reps, start=1):
            xs_ref = np.array(rep["Xs"], dtype=float) if rep.get("Xs") is not None else None
            xl_ref = np.array(rep["Xl"], dtype=float) if rep.get("Xl") is not None else None

            xs_ok = xs is None or xs_ref is None or max_abs_diff(xs, xs_ref) < BASE_TOL
            xl_ok = xl is None or xl_ref is None or max_abs_diff(xl, xl_ref) < BASE_TOL

            if xs_ok and xl_ok:
                row["slice_number"] = sn
                slices.setdefault(sn, []).append(row)
                placed = True
                break

        if not placed:
            sn_new = len(slice_reps) + 1
            row["slice_number"] = sn_new
            slice_reps.append(row)
            slices.setdefault(sn_new, []).append(row)

    return slices


def detect_replicates_in_slice(
    rows: List[dict],
    xc_tol: float = XC_ANALYTICAL_TOL,
    r_tol: float = BASE_TOL,
) -> Dict[int, List[dict]]:
    """Group rows inside a slice by Xs/Xl similarity and assign replicate_number."""
    _ensure_slice_metadata(rows)

    reps: List[dict] = []
    replicates: Dict[int, List[dict]] = {}

    for row in rows:
        xs = np.asarray(row.get("Xs") or [], dtype=float)
        xl = np.asarray(row.get("Xl") or [], dtype=float)

        placed = False
        for rep in reps:
            rep_xs = np.asarray(rep.get("Xs") or [], dtype=float)
            rep_xl = np.asarray(rep.get("Xl") or [], dtype=float)

            xs_diff = 0.0 if xs.size == 0 or rep_xs.size == 0 else max_abs_diff(xs, rep_xs)
            xl_diff = 0.0 if xl.size == 0 or rep_xl.size == 0 else max_abs_diff(xl, rep_xl)

            if xs_diff < BASE_TOL and xl_diff < BASE_TOL:
                row["replicate_number"] = rep["replicate_number"]
                replicates.setdefault(row["replicate_number"], []).append(row)
                placed = True
                break

        if not placed:
            rn_new = len(reps) + 1
            row["replicate_number"] = rn_new
            rep_copy = copy.deepcopy(row)
            rep_copy["replicate_number"] = rn_new
            reps.append(rep_copy)
            replicates.setdefault(rn_new, []).append(row)

    return replicates



from collections import defaultdict
from typing import Dict, List


def group_rows_by_slice_number(rows: List[dict]) -> Dict[int, List[dict]]:
    _ensure_slice_metadata(rows)
    grouped = defaultdict(list)

    for row_index, row in enumerate(rows):
        mixture_id = row.get("mixture_id", "UNKNOWN")
        sn = row.get("slice_number")
        if DEBUG:
            logger.debug("GROUPING row:")
            logger.debug("  mixture_id=%s", mixture_id)
            logger.debug("  row_index=%s", row_index)
            logger.debug("  slice_number_before=%s", int(sn) if sn is not None else None)
            logger.debug("  Xs=%s", np.asarray(row.get("Xs") or [], dtype=float).tolist())
            logger.debug("  Xl=%s", np.asarray(row.get("Xl") or [], dtype=float).tolist())
        grouped[int(sn)].append(copy.deepcopy(row))

    return grouped



def canonical_Xs_for_slice(rows: List[dict], n_components: int) -> List[float]:
    if DEBUG:
        for row_index, row in enumerate(rows):
            logger.debug("CANONICAL INPUT:")
            logger.debug("  mixture_id=%s", row.get("mixture_id", "UNKNOWN"))
            logger.debug("  slice_number=%s", row.get("slice_number"))
            logger.debug("  row_index=%s", row_index)
            logger.debug("  Xs=%s", np.asarray(row.get("Xs") or [], dtype=float).tolist())
            logger.debug("  Xl=%s", np.asarray(row.get("Xl") or [], dtype=float).tolist())

    xs_matrix = [
        [row[f"xs_{i+1}"] for i in range(n_components)]
        for row in rows
    ]
    canonical_xs = np.median(np.array(xs_matrix), axis=0).tolist()
    if DEBUG:
        for row_index, row in enumerate(rows):
            logger.debug("CANONICAL OUTPUT:")
            logger.debug("  mixture_id=%s", row.get("mixture_id", "UNKNOWN"))
            logger.debug("  slice_number=%s", row.get("slice_number"))
            logger.debug("  canonical_Xs=%s", canonical_xs)
            break
    return canonical_xs


def canonical_Xl_for_slice(rows: List[dict], n_components: int) -> List[float]:
    if DEBUG:
        for row_index, row in enumerate(rows):
            logger.debug("CANONICAL INPUT:")
            logger.debug("  mixture_id=%s", row.get("mixture_id", "UNKNOWN"))
            logger.debug("  slice_number=%s", row.get("slice_number"))
            logger.debug("  row_index=%s", row_index)
            logger.debug("  Xs=%s", np.asarray(row.get("Xs") or [], dtype=float).tolist())
            logger.debug("  Xl=%s", np.asarray(row.get("Xl") or [], dtype=float).tolist())

    xl_matrix = [
        [row[f"xl_{i+1}"] for i in range(n_components)]
        for row in rows
    ]
    canonical_xl = np.median(np.array(xl_matrix), axis=0).tolist()
    if DEBUG:
        for row_index, row in enumerate(rows):
            logger.debug("CANONICAL OUTPUT:")
            logger.debug("  mixture_id=%s", row.get("mixture_id", "UNKNOWN"))
            logger.debug("  slice_number=%s", row.get("slice_number"))
            logger.debug("  canonical_Xl=%s", canonical_xl)
            break
    return canonical_xl


def print_slice_summary(mixture_id: str, slice_records: List[SliceRecord]) -> None:
    if not DEBUG:
        return
    logger.debug("Mixture %s", mixture_id)
    component_count = len(slice_records[0].Xc) if slice_records else 0
    total_rows = sum(len(s.raw_rows) for s in slice_records)
    logger.debug("  Components: %s", component_count)
    logger.debug("  Rows: %s", total_rows)

    for record in slice_records:
        logger.debug("\n  Slice %s:", record.slice_number)
        logger.debug("    Rows: %s", len(record.raw_rows))
        logger.debug("    Xs canonical: %s", record.Xs)
        logger.debug("    Xl canonical: %s", record.Xl)
        logger.debug("    R canonical: %s", record.R)
        logger.debug("    Replicates:")

        replicate_groups: Dict[int, List[dict]] = defaultdict(list)
        for row in record.raw_rows:
            replicate_groups[int(row.get("replicate_number", 0))].append(row)

        for replicate_number in sorted(replicate_groups):
            rows_ref = list(range(len(replicate_groups[replicate_number])))
            logger.debug("      Replicate %s: rows %s", replicate_number, rows_ref)

    logger.debug("\n-----------------------------------------")


def build_slices_for_mixture(
    mixture_id: str,
    rows: List[dict],
    canonical_xc: List[float],
    n_components: int
) -> MixtureSet:

    if DEBUG:
        logger.debug("build_slices_for_mixture mixture_id=%s, rows=%s", mixture_id, len(rows))
    grouped = group_rows_by_slice_number(rows)
    if DEBUG:
        logger.debug("grouped slices for mixture_id=%s: %s", mixture_id, sorted(grouped.keys()))
    slice_records = []

    for slice_number, slice_rows in sorted(grouped.items()):
        if DEBUG:
            logger.debug("processing slice_number=%s in mixture_id=%s, row_count=%s", slice_number, mixture_id, len(slice_rows))
        for row_index, row in enumerate(slice_rows):
            if DEBUG:
                logger.debug("BEFORE canonical:")
                logger.debug("  mixture_id=%s", mixture_id)
                logger.debug("  slice_number=%s", slice_number)
                logger.debug("  row_index=%s", row_index)
                logger.debug("  Xs=%s", np.asarray(row.get("Xs") or [], dtype=float).tolist())
                logger.debug("  Xl=%s", np.asarray(row.get("Xl") or [], dtype=float).tolist())

        detect_replicates_in_slice(slice_rows, XC_ANALYTICAL_TOL, BASE_TOL)
        R_slice = canonical_R_for_slice(slice_rows, canonical_xc, n_components)
        Xs_slice = canonical_Xs_for_slice(slice_rows, n_components)
        Xl_slice = canonical_Xl_for_slice(slice_rows, n_components)

        slice_records.append(
            SliceRecord(
                mixture_id=mixture_id,
                slice_number=slice_number,
                R=R_slice,
                Xc=canonical_xc,
                Xs=Xs_slice,
                Xl=Xl_slice,
                raw_rows=slice_rows,
            )
        )

    print_slice_summary(mixture_id, slice_records)
    return MixtureSet(mixture_id=mixture_id, slices=slice_records)

