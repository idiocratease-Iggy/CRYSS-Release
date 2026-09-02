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
from typing import Dict, List
from .slice_builder import MixtureSet, SliceRecord


# ------------------------------------------------------------
# 1. Mixture-level statistics
# ------------------------------------------------------------

def collect_mixture_statistics(mixture: MixtureSet) -> Dict:
    """
    Stage 2 — Mixture-level statistics.
    These metrics summarise slice-level chemistry into a stable mixture fingerprint.

    Outputs:
    - canonical mixture R
    - per-component Xs/Xl medians across slices
    - slice-to-slice variation (std)
    - lever-rule consistency across slices
    - mixture purity index
    """

    # --- Gather slice-level arrays ---
    slice_R = []
    slice_Xs = []
    slice_Xl = []

    for s in mixture.slices:
        slice_R.append(s.R)
        slice_Xs.append(np.array(s.Xs))
        slice_Xl.append(np.array(s.Xl))

    slice_R = np.array(slice_R)
    slice_Xs = np.array(slice_Xs)   # shape: (n_slices, n_components)
    slice_Xl = np.array(slice_Xl)

    # --- Canonical mixture R ---
    mixture_R = float(np.median(slice_R))

    # --- Per-component medians across slices ---
    Xs_median = np.median(slice_Xs, axis=0).tolist()
    Xl_median = np.median(slice_Xl, axis=0).tolist()

    # --- Slice-to-slice variation ---
    Xs_std = np.std(slice_Xs, axis=0).tolist()
    Xl_std = np.std(slice_Xl, axis=0).tolist()
    R_std = float(np.std(slice_R))

    # --- Lever-rule consistency ---
    # For each slice: Xc = Xs + Xl
    slice_Xc = slice_Xs + slice_Xl
    Xc_median = np.median(slice_Xc, axis=0).tolist()
    Xc_std = np.std(slice_Xc, axis=0).tolist()

    # --- Purity index ---
    # A simple but meaningful metric:
    # purity = mean(|Xs - Xl|)
    purity_index = float(np.mean(np.abs(slice_Xs - slice_Xl)))

    return {
        "mixture_id": mixture.mixture_id,

        # Canonical mixture-level R
        "R_mixture": mixture_R,
        "R_std": R_std,

        # Per-component mixture-level Xs/Xl medians
        "Xs_median_mix": Xs_median,
        "Xl_median_mix": Xl_median,

        # Slice-to-slice variation
        "Xs_std_mix": Xs_std,
        "Xl_std_mix": Xl_std,

        # Lever-rule consistency
        "Xc_median_mix": Xc_median,
        "Xc_std_mix": Xc_std,

        # Purity index
        "purity_index": purity_index,
    }



# ------------------------------------------------------------
# 2. Slice-level statistics
# ------------------------------------------------------------

def collect_slice_statistics(mixture: MixtureSet) -> List[Dict]:
    """
    Chemically meaningful slice statistics:
    - canonical R
    - per-component Xs and Xl medians
    - per-component replicate variation (std)
    - lever-rule consistency per component
    - slice geometry diagnostics
    - slice purity index
    """

    slice_stats = []

    for s in mixture.slices:
        # Canonical per-component values for this slice
        Xs = np.array(s.Xs)   # median across replicates already
        Xl = np.array(s.Xl)
        Xc = np.array(s.Xc)

        # Replicate-level variation (if replicates stored)
        if hasattr(s, "replicates") and s.replicates:
            Xs_reps = np.array([r.Xs for r in s.replicates])
            Xl_reps = np.array([r.Xl for r in s.replicates])
            R_reps  = np.array([r.R  for r in s.replicates])

            Xs_std = np.std(Xs_reps, axis=0).tolist()
            Xl_std = np.std(Xl_reps, axis=0).tolist()
            R_std  = float(np.std(R_reps))
        else:
            Xs_std = [0.0] * len(Xs)
            Xl_std = [0.0] * len(Xl)
            R_std  = 0.0

        # Lever-rule consistency per component
        R_components = []
        for i in range(len(Xc)):
            denom = (Xs[i] - Xl[i])
            if abs(denom) < 1e-12:
                R_components.append(None)
            else:
                R_components.append(float((Xc[i] - Xl[i]) / denom))

        # Slice geometry diagnostics
        solid_enrichment_index  = float(np.sum((Xs - Xc)**2))
        liquor_enrichment_index = float(np.sum((Xl - Xc)**2))
        solid_liquor_separation = float(np.sum((Xs - Xl)**2))

        # Purity index (how close slice is to pure component)
        slice_purity_index = float(np.max(Xs))

        slice_stats.append({
            "mixture_id": mixture.mixture_id,
            "slice_number": s.slice_number,
            "R": s.R,

            # Canonical per-component values
            "Xs_median": Xs.tolist(),
            "Xl_median": Xl.tolist(),

            # Replicate variation
            "Xs_std_per_component": Xs_std,
            "Xl_std_per_component": Xl_std,
            "R_std": R_std,

            # Lever-rule diagnostics
            "R_component_values": R_components,

            # Geometry diagnostics
            "solid_enrichment_index": solid_enrichment_index,
            "liquor_enrichment_index": liquor_enrichment_index,
            "solid_liquor_separation": solid_liquor_separation,

            # Purity / sharpness
            "slice_purity_index": slice_purity_index,
        })

    return slice_stats




# ------------------------------------------------------------
# 3. Component-level statistics
# ------------------------------------------------------------

def collect_component_statistics(mixture: MixtureSet) -> List[Dict]:
    """
    For each component:
    - Xc_i median
    - Xs_i median across slices
    - Xl_i median across slices
    - R median across slices
    """

    n_components = len(mixture.slices[0].Xc)
    comp_stats = []

    # Collect per-component values across slices
    Xs_all = [[] for _ in range(n_components)]
    Xl_all = [[] for _ in range(n_components)]
    R_all = []

    for s in mixture.slices:
        R_all.append(s.R)
        for i in range(n_components):
            Xs_all[i].append(s.Xs[i])
            Xl_all[i].append(s.Xl[i])

    for i in range(n_components):
        comp_stats.append({
            "mixture_id": mixture.mixture_id,
            "component": i + 1,

            "Xc": mixture.slices[0].Xc[i],

            "Xs_median": float(np.median(Xs_all[i])),
            "Xl_median": float(np.median(Xl_all[i])),
            "R_median": float(np.median(R_all)),
        })

    return comp_stats


# ------------------------------------------------------------
# 4. File-level statistics
# ------------------------------------------------------------

def collect_file_statistics(mixtures: Dict[str, MixtureSet]) -> Dict:
    """
    High-level summary of the entire parsed file.
    """

    num_mixtures = len(mixtures)
    num_slices = sum(len(m.slices) for m in mixtures.values())

    return {
        "num_mixtures": num_mixtures,
        "num_slices": num_slices,
    }
