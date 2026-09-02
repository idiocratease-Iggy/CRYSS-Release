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

def extract_xc_matrix(rows: list[dict], n_components: int) -> np.ndarray:
    """
    Extracts the Xc values from each row into a matrix of shape (n_rows, n_components).
    """
    xc_matrix = []
    for row in rows:
        xc = [row[f"xc_{i+1}"] for i in range(n_components)]
        xc_matrix.append(xc)
    return np.array(xc_matrix)


def compute_canonical_xc(rows: list[dict], n_components: int) -> list[float]:
    """
    Computes the canonical Xc for a mixture set using the median across replicates.
    """
    xc_matrix = extract_xc_matrix(rows, n_components)
    canonical = np.median(xc_matrix, axis=0)
    return canonical.tolist()


def compute_xc_diagnostics(rows: list[dict], n_components: int) -> dict:
    """
    Optional: returns mean, median, std, and per-row deviations.
    Useful for debugging and writing diagnostic Excel files.
    """
    xc_matrix = extract_xc_matrix(rows, n_components)

    median = np.median(xc_matrix, axis=0)
    mean = np.mean(xc_matrix, axis=0)
    std = np.std(xc_matrix, axis=0)

    deviations = []
    for xc in xc_matrix:
        deviations.append(np.abs(xc - median))

    return {
        "median": median.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "deviations": deviations,  # list of per-row deviation vectors
    }
