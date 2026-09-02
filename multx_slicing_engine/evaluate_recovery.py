# evaluate_recovery.py

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


from dataclasses import dataclass
from typing import List

try:
    from multxeu_core.multx_slicing_engine.canonical_types import EutecticPathState
    from multxeu_core.multx_slicing_engine.canonical_gradients import SegmentGradients
except ImportError:
    from multxeu_core.multx_slicing_engine.canonical_types import EutecticPathState
    from multxeu_core.multx_slicing_engine.canonical_gradients import SegmentGradients


@dataclass
class SliceResult:
    R: float
    Xliq: List[float]
    Xsol: List[float]
    Ks: float
    kl: float
    solvent: float
    seg_index: int | None = None
    seg_start_R: float | None = None
    seg_end_R: float | None = None
    seg_Xsol: List[float] | None = None


def _segment_for_recovery(segments, R: float):
    """Pick the unique active segment for a recovery value, even when boundaries collapse."""
    candidates = []
    for seg in segments:
        if abs(seg.start_R - seg.end_R) <= 1e-12:
            continue
        if seg.end_R - 1e-12 <= R <= seg.start_R + 1e-12:
            candidates.append(seg)

    if not candidates:
        return segments[-1]

    return min(
        candidates,
        key=lambda seg: (abs(seg.start_R - R), abs(seg.end_R - R), seg.index),
    )


def evaluate_at_recovery(state: EutecticPathState,
                         segment_grads: List[SegmentGradients],
                         R: float) -> SliceResult:
    """
    Evaluate liquor composition, solid composition, solvent volume,
    and mass balance at arbitrary recovery R.

    Parser-supplied recoveries are treated as exact slice targets and are not
    silently clamped or rewritten. The segment lookup must only decide which
    canonical interval contains that exact R, without mutating the supplied value.
    """

    if not state.segments:
        raise ValueError("state.segments is empty")

    # ------------------------------------------------------------
    # 1. Find correct segment using the exact requested R.
    # ------------------------------------------------------------
    seg = _segment_for_recovery(state.segments, R)
    seg_index = seg.index
    sg = segment_grads[seg_index]

    # ------------------------------------------------------------
    # 3. Solid composition is explicit on the segment itself.
    # ------------------------------------------------------------
    Xsol = list(seg.Xsol)

    # ------------------------------------------------------------
    # 4. Liquor composition (segment interpolation)
    # ------------------------------------------------------------
    start_R = seg.start_R
    end_R = seg.end_R
    if abs(start_R - end_R) <= 1e-12:
        Xliq = list(seg.Xliq_start)
    else:
        frac = (R - start_R) / (end_R - start_R)
        Xliq = []
        for x0, x1 in zip(seg.Xliq_start, seg.Xliq_end):
            x = x0 + (x1 - x0) * frac
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            Xliq.append(x)

    # ------------------------------------------------------------
    # 5. Mass balance
    # ------------------------------------------------------------
    # At the segment boundary, a component can approach zero in both the liquor and
    # solid split. In that regime the denominator Xsol - Xliq becomes tiny and the
    # arithmetic is dominated by floating-point jitter rather than physical drift.
    # We ignore only these near-zero contributions so the aggregate Ks remains
    # stable without changing the source composition or cascade logic.
    Ks_values = []
    for i in range(len(Xliq)):
        denom = (Xsol[i] - Xliq[i])
        if abs(denom) < 1e-10:
            continue
        if abs(Xsol[i]) < 1e-10 and abs(Xliq[i]) < 1e-10:
            continue
        Ks_values.append((state.initial_mixture.X[i] - Xliq[i]) / denom)

    Ks = sum(Ks_values) / len(Ks_values) if Ks_values else 0.0
    kl = 1 - Ks

    # ------------------------------------------------------------
    # 6. Solvent volume
    # ------------------------------------------------------------
    solvent = sg.solvol_grad * R + sg.solvol_int

    return SliceResult(
        R=R,
        Xliq=Xliq,
        Xsol=Xsol,
        Ks=Ks,
        kl=kl,
        solvent=solvent,
        seg_index=seg_index,
        seg_start_R=seg.start_R,
        seg_end_R=seg.end_R,
        seg_Xsol=list(seg.Xsol),
    )
