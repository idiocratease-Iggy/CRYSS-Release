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

from typing import Any, Dict, List, Optional

import math

try:
    from multx_slicing_engine.slice_generator import generate_slices_for_mixture
except ImportError:  # pragma: no cover
    from slice_generator import generate_slices_for_mixture


def print_matrix_layer(layer: Dict[str, Any]) -> None:
    """Print the matrix-layer records to the terminal for comparison with the base block."""
    # print("\n==============================================")
    # print(" MATRIX LAYER SUMMARY")
    # print("==============================================\n")

    # print("[MIXTURE ANCHORS]")
    for anchor in layer.get("mixture_anchors", []):
        pass

    # print("\n[SEGMENT ROWS]")
    for segment in layer.get("segments", []):
        pass

    # print("\n[SLICE ROWS]")
    for scenario_name, rows in sorted(layer.get("slice_arrays", {}).items()):
        for row in rows:
            pass


def _safe_list(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


def _compute_liquor_from_slice(Xc: List[float], Xs: List[float], R: float) -> List[float]:
    if abs(R - 1.0) < 1e-12:
        return [0.0 for _ in Xc]

    return [((R * xs) - xc) / (R - 1.0) for xc, xs in zip(Xc, Xs)]


def _clip_tiny_negative_values(values: List[float], threshold: float = 1e-12) -> List[float]:
    return [0.0 if value < 0.0 else value for value in values]


def _compute_implied_recovery_for_zero_liquor(Xc: List[float], Xs: List[float], R: float) -> float:
    if abs(R - 1.0) < 1e-12:
        return R

    xliq = _compute_liquor_from_slice(Xc, Xs, R)
    implied_R = None
    for xc, xs, liquor_value in zip(Xc, Xs, xliq):
        if liquor_value >= 0.0:
            continue
        if abs(xs) < 1e-12:
            continue
        candidate_R = xc / xs
        if implied_R is None or abs(candidate_R - R) < abs(implied_R - R):
            implied_R = candidate_R

    return implied_R if implied_R is not None else R


def _diagnose_slice_liquor(Xc: List[float], Xs: List[float], R: float) -> tuple[float, List[float], int, bool]:
    """Compute the raw liquor values, count negative entries, and report a local adjusted recovery.

    This keeps the original slice R untouched and focuses the diagnostic on the raw
    liquor calculation, the number of negative components, and the implied recovery
    that would arise if the negative liquor component were clamped to zero.
    """

    if abs(R - 1.0) < 1e-12:
        return R, [0.0 for _ in Xc], 0, False

    xliq = _compute_liquor_from_slice(Xc, Xs, R)
    negative_count = sum(1 for value in xliq if value < 0.0)
    clamped_xliq = _clip_tiny_negative_values(xliq)
    correction_flagged = negative_count > 0

    if not correction_flagged:
        return R, clamped_xliq, negative_count, False

    adjusted_R = _compute_implied_recovery_for_zero_liquor(Xc, Xs, R)
    return adjusted_R, clamped_xliq, negative_count, True


def _compute_r_check(Xc: List[float], Xs: List[float], Xl: List[float]) -> List[float]:
    results: List[float] = []
    for xc, xs, xl in zip(Xc, Xs, Xl):
        denom = xs - xl
        if abs(denom) < 1e-12:
            results.append(float("nan"))
        else:
            results.append((xc - xl) / denom)
    return results


def build_matrix_layer(canonical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a structured matrix-layer view from canonical harness results.

    The base pipeline remains unchanged. This layer consumes the existing summary
    records and exposes them in a simpler shape for later matrix construction.
    """

    mixture_anchors: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    slice_arrays: Dict[str, List[Dict[str, Any]]] = {}

    for summary in canonical_results:
        mixture = summary.get("mixture")
        if mixture is None:
            continue

        state = summary.get("state")
        slices = summary.get("slices") or []
        num_slices = int(summary.get("num_slices", len(slices) or 3))
        configured_slices = list(slices)

        Xc = _safe_list(getattr(mixture, "X", []))

        anchor = {
            "mixture_id": getattr(mixture, "mixture_id", ""),
            "component_count": len(Xc),
            "Xc": Xc,
            "Rmax": float(summary.get("Rmax", 0.0)),
            "RelSolVol_Rmax": float(summary.get("RelSolVol_Rmax", 0.0)),
            "slice_count": num_slices,
            "segment_count": len(getattr(state, "segments", []) or []),
        }
        mixture_anchors.append(anchor)

        for seg in getattr(state, "segments", []) or []:
            if getattr(seg, "end_R", 0.0) == getattr(seg, "start_R", 0.0):
                continue

            segment_rows.append({
                "mixture_id": getattr(mixture, "mixture_id", ""),
                "segment_index": getattr(seg, "index", 0),
                "start_R": float(getattr(seg, "start_R", 0.0)),
                "end_R": float(getattr(seg, "end_R", 0.0)),
                "solvent_start": float(getattr(seg, "solvent_start", 0.0)),
                "solvent_end": float(getattr(seg, "solvent_end", 0.0)),
                "Xliq_start": _safe_list(getattr(seg, "Xliq_start", [])),
                "Xliq_end": _safe_list(getattr(seg, "Xliq_end", [])),
            })

        for scenario_count in range(1, num_slices + 1):
            if scenario_count == num_slices and configured_slices:
                scenario_slices = configured_slices
            else:
                scenario_slices = generate_slices_for_mixture(summary, scenario_count)

            for slice_index, slice_result in enumerate(scenario_slices, start=1):
                Xs = _safe_list(getattr(slice_result, "Xliq", []))
                original_R = float(getattr(slice_result, "R", 0.0))
                relsolvol = float(getattr(slice_result, "solvent", 0.0))
                derived_xliq = _compute_liquor_from_slice(Xc, Xs, original_R)
                adjusted_R, adjusted_xliq, negative_liquor_count, correction_flagged = _diagnose_slice_liquor(
                    Xc,
                    Xs,
                    original_R,
                )
                derived_xliq = _clip_tiny_negative_values(adjusted_xliq)
                R = original_R
                r_check = _compute_r_check(Xc, Xs, derived_xliq)

                row = {
                    "mixture_id": getattr(mixture, "mixture_id", ""),
                    "slice_count": scenario_count,
                    "slice_index": slice_index,
                    "R": R,
                    "RelSolVol": relsolvol,
                    "Xc": Xc,
                    "Xs": Xs,
                    "Xliq": derived_xliq,
                    "R_check": r_check,
                    "mass_balance_ok": not correction_flagged,
                    "correction_applied": correction_flagged,
                    "adjusted_R": adjusted_R,
                    "negative_liquor_count": negative_liquor_count,
                    "correction_delta_R": abs(original_R - adjusted_R) if correction_flagged else 0.0,
                    "correction_flagged": correction_flagged,
                }

                scenario_key = f"slice_count_{scenario_count}"
                slice_arrays.setdefault(scenario_key, []).append(row)

    return {
        "mixture_anchors": mixture_anchors,
        "segments": segment_rows,
        "slice_arrays": slice_arrays,
    }
