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

from typing import Iterable


def _segment_for_recovery(segments, R: float):
    candidates = []
    for seg in segments:
        if abs(seg.start_R - seg.end_R) <= 1e-12:
            continue
        # Direct recovery lookup only; no solvent plateau or degenerate_main_pair
        # adjustment is applied in the canonical slice logic.
        if seg.end_R - 1e-12 <= R <= seg.start_R + 1e-12:
            candidates.append(seg)

    if not candidates:
        return None

    return min(candidates, key=lambda seg: (abs(seg.start_R - R), abs(seg.end_R - R), seg.index))


def compute_slices(state, slice_R: Iterable[float]):
    """Return slice metadata for the requested recovery cuts."""
    slices = []
    for target in list(slice_R):
        selected = _segment_for_recovery(state.segments, float(target))
        if selected is None:
            slices.append({"R": float(target), "segment_index": None, "segment": None})
        else:
            slices.append({"R": float(target), "segment_index": selected.index, "segment": selected})
    return slices
