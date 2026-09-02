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
    from multxeu_core.multx_slicing_engine.canonical_types import EutecticPathState, Segment
except ImportError:
    from multxeu_core.multx_slicing_engine.canonical_types import EutecticPathState, Segment


@dataclass
class SegmentGradients:
    index: int
    gradients: List[float]
    intercepts: List[float]
    solvol_grad: float
    solvol_int: float
    start_R: float
    end_R: float


class GradientEngine:
    """Build piecewise-linear segment gradients for canonical state data."""

    def __init__(self, state: EutecticPathState):
        self.state = state
        self.segments: List[SegmentGradients] = []
        self._build_gradients()

    def _build_gradients(self) -> None:
        self.segments.clear()

        for seg in self.state.segments:
            R0 = seg.start_R
            R1 = seg.end_R
            X0 = seg.Xliq_start
            X1 = seg.Xliq_end
            S0 = float(seg.solvent_start)
            S1 = float(seg.solvent_end)

            dR = R1 - R0
            if abs(dR) < 1e-12:
                gradients = [0.0 for _ in X0]
                intercepts = list(X0)
                solvol_grad = 0.0
                solvol_int = S0
            else:
                gradients = [(x1 - x0) / dR for x0, x1 in zip(X0, X1)]
                intercepts = [x0 - m * R0 for x0, m in zip(X0, gradients)]
                # Canonical solvent gradients must use the corrected solvent values
                # directly; no plateau or degeneracy-adjustment logic is applied.
                solvol_grad = (S1 - S0) / dR
                solvol_int = S0 - solvol_grad * R0

            sg = SegmentGradients(
                index=seg.index,
                gradients=gradients,
                intercepts=intercepts,
                solvol_grad=solvol_grad,
                solvol_int=solvol_int,
                start_R=R0,
                end_R=R1,
            )
            self.segments.append(sg)

    def solvent_value_at_recovery(self, R: float) -> float:
        """Evaluate the segment solvent value at a given recovery using the direct linear form."""
        for seg in self.segments:
            if seg.start_R >= R >= seg.end_R or seg.start_R <= R <= seg.end_R:
                if abs(seg.end_R - seg.start_R) < 1e-12:
                    return float(seg.solvol_int)
                return seg.solvol_grad * R + seg.solvol_int
        if not self.segments:
            return 0.0
        return self.segments[-1].solvol_grad * R + self.segments[-1].solvol_int

    def get_segment_gradients(self) -> List[SegmentGradients]:
        return self.segments
