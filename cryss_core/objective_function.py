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

"""Objective functions used by the CRYSS optimizer and MC fitter."""

from typing import Any, Dict, List, Tuple


def compute_objective_with_slices(
    summary: Dict[str, Any],
    Xs_obs: List[List[float]],
) -> Tuple[float, List[float]]:
    """Compute the slice-wise sum-squared objective against observed Xs."""
    slices_pred = summary["slices"]

    if len(slices_pred) != len(Xs_obs):
        raise ValueError(f"Slice count mismatch: pred={len(slices_pred)}, obs={len(Xs_obs)}")

    slice_errors: List[float] = []
    total_error = 0.0

    for s, sl in enumerate(slices_pred):
        Xs_pred = sl["Xs"]
        Xs_true = Xs_obs[s]

        if len(Xs_pred) != len(Xs_true):
            raise ValueError(
                f"Component count mismatch at slice {s}: "
                f"pred={len(Xs_pred)}, obs={len(Xs_true)}"
            )

        err_s = 0.0
        for i in range(len(Xs_true)):
            diff = Xs_pred[i] - Xs_true[i]
            err_s += diff * diff

        slice_errors.append(err_s)
        total_error += err_s

    return total_error, slice_errors


def compute_tensor_objective(summary: Dict[str, Any], Xs_obs: List[List[float]]) -> float:
    """Return a secondary rank-weighted objective used for MC/optimizer comparison."""
    slices_pred = summary.get("slices", [])
    if not slices_pred:
        return 0.0

    total = 0.0
    n_slices = max(1, len(slices_pred))
    for s, sl in enumerate(slices_pred):
        Xs_pred = sl.get("Xs", [])
        Xs_true = Xs_obs[s] if s < len(Xs_obs) else []
        slice_weight = 1.0 + (float(s) / max(1, n_slices - 1))
        n_components = max(1, min(len(Xs_pred), len(Xs_true)))
        for i in range(n_components):
            diff = float(Xs_pred[i]) - float(Xs_true[i])
            component_weight = 1.0 + (float(i) / max(1, n_components - 1))
            total += slice_weight * component_weight * diff * diff

    return total


compute_objective = compute_objective_with_slices

__all__ = ["compute_objective", "compute_objective_with_slices", "compute_tensor_objective"]
