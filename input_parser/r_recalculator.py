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



import numpy as np


def _filter_outlier_Rs(Rs: list[float]) -> list[float]:
    """Robustly reject outliers across component-level or row-level R values."""
    if not Rs:
        return []

    Rs_array = np.asarray(Rs, dtype=float)
    median_R = np.median(Rs_array)
    deviations = np.abs(Rs_array - median_R)
    mad_R = np.median(deviations)

    if np.isclose(mad_R, 0.0):
        return Rs_array.tolist()

    threshold = 2.0 * mad_R
    mask = deviations <= threshold
    filtered_Rs = Rs_array[mask]

    if filtered_Rs.size == 0:
        filtered_Rs = Rs_array

    return filtered_Rs.tolist()


def compute_R_from_lever_rule(Xc: list[float] | np.ndarray | None, Xs: list[float] | np.ndarray | None, Xl: list[float] | np.ndarray | None) -> float | None:
    """Compute row-level R from Xc/Xs/Xl using the lever rule with division safety."""
    if Xc is None or Xs is None or Xl is None:
        return None

    xc_array = np.asarray(Xc, dtype=float)
    xs_array = np.asarray(Xs, dtype=float)
    xl_array = np.asarray(Xl, dtype=float)

    component_Rs: list[float] = []
    for xc, xs, xl in zip(xc_array, xs_array, xl_array):
        denom = float(xs - xl)
        if abs(denom) < 1e-12:
            continue
        Ri = float((xc - xl) / denom)
        if np.isfinite(Ri):
            component_Rs.append(Ri)

    if not component_Rs:
        return None

    filtered_Rs = _filter_outlier_Rs(component_Rs)
    return float(np.mean(filtered_Rs))


def compute_component_Rs(canonical_xc: list[float], row: dict, n_components: int) -> list[float]:
    """
    Computes component-level R_i values with robust handling of
    cross-contamination noise, denominator stability, and outlier rejection.
    """

    Rs = []

    for i in range(n_components):
        xc = canonical_xc[i]
        xs = row[f"xs_{i+1}"]
        xl = row[f"xl_{i+1}"]

        # --- Denominator stability check ---
        denom = xs - xl
        if abs(denom) < 1e-9:   # slightly more forgiving than 1e-12
            continue

        Ri = (xc - xl) / denom

        # --- Basic physical sanity checks ---
        # R should be between 0 and 1 for valid slice geometry.
        # We allow up to 1.1 to absorb mild noise.
        if Ri < -0.05 or Ri > 1.2:
            continue

        Rs.append(Ri)

    if not Rs:
        return []

    return _filter_outlier_Rs(Rs)


def compute_row_R(canonical_xc: list[float], row: dict, n_components: int) -> float | None:
    """
    Computes row-level R using mean instead of median,
    with outlier rejection across component-level R_i values.
    """

    row_xc = row.get("Xc")
    if row_xc is None:
        row_xc = [row.get(f"xc_{i+1}") for i in range(n_components)]

    xs_values = row.get("Xs")
    xl_values = row.get("Xl")
    if row_xc is not None and xs_values is not None and xl_values is not None:
        computed = compute_R_from_lever_rule(row_xc, xs_values, xl_values)
        if computed is not None:
            return computed

    Rs = compute_component_Rs(canonical_xc, row, n_components)

    if not Rs:
        return None

    return float(np.mean(_filter_outlier_Rs(Rs)))



def canonical_R_for_slice(rows: list[dict], canonical_xc: list[float], n_components: int) -> float:
    R_values = []

    for row in rows:
        Ri = compute_row_R(canonical_xc, row, n_components)
        if Ri is not None:
            R_values.append(Ri)

    if not R_values:
        return float(rows[0].get("r", 0.0))

    return float(np.median(R_values))
