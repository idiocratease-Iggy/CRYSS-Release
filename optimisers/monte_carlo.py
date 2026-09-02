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

import math
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from cryss_core.harness import assign_pd_svl, run_single_case


def _linear_slice_weights(slice_r: Sequence[float] | None) -> List[float]:
    """Return slice weights where the lowest-R slice gets weight 1 and higher-R slices are smaller.

    For example, with R values [0.90, 0.80, 0.70], this gives a gradient in which the
    lowest-R slice is the most heavily weighted. The exact ordering is preserved by
    mapping weights back to the original slice positions.
    """
    if not slice_r:
        return [1.0]

    cleaned = [float(r) for r in slice_r if math.isfinite(float(r))]
    if not cleaned:
        return [1.0] * len(slice_r)

    if len(cleaned) == 1:
        return [1.0]

    weights = [0.0] * len(cleaned)
    ranked = sorted(range(len(cleaned)), key=lambda idx: cleaned[idx])
    for rank, idx in enumerate(ranked):
        weights[idx] = 1.0 - (rank / (len(cleaned) - 1))

    return weights


def _exp_slice_weights(slice_r: Sequence[float] | None, *, gamma: float = 3.0) -> List[float]:
    """Return exponentially decaying weights so the lowest-R slice dominates most strongly."""
    if not slice_r:
        return [1.0]

    cleaned = [float(r) for r in slice_r if math.isfinite(float(r))]
    if not cleaned:
        return [1.0] * len(slice_r)

    if len(cleaned) == 1:
        return [1.0]

    ranked = sorted(range(len(cleaned)), key=lambda idx: cleaned[idx])
    max_rank = max(len(cleaned) - 1, 1)
    weights = [0.0] * len(cleaned)
    for rank, idx in enumerate(ranked):
        normalized = rank / max_rank
        weights[idx] = math.exp(-gamma * normalized)

    # Renormalise so the lowest-R slice stays at weight 1.0.
    weight_min = min(weights)
    if weight_min < 1.0:
        weights = [w / max(weights) for w in weights]

    return weights


def _slice_weights(slice_r: Sequence[float] | None, *, weight_mode: str = "linear") -> List[float]:
    """Return slice weights for the objective.

    Supported modes:
      - "none": all weights equal to 1.0
      - "linear": lowest-R slice carries the highest weight, tapering linearly
      - "exp": lowest-R slice carries the highest weight with a curved exponential decay
    """
    mode = (weight_mode or "linear").lower()
    if mode == "none":
        return [1.0] * len(slice_r) if slice_r else [1.0]
    if mode == "exp":
        return _exp_slice_weights(slice_r)
    return _linear_slice_weights(slice_r)


def compute_monte_carlo_objective(summary, Xs_obs, slice_R=None, *, weight_mode: str = "linear"):
    """Slice-wise sum-of-squares objective with selectable slice weighting.

    Use weight_mode="none" for the raw unweighted SSE, "linear" for a simple
    low-R emphasis, or "exp" for a stronger curved decay that keeps the lowest-R
    slice dominant while reducing the influence of higher-R slices.
    """
    slices_pred = summary.get("slices", [])
    if not slices_pred:
        return 0.0

    if slice_R is None:
        slice_R = [float(slice_pred.get("slice_R", 0.0)) for slice_pred in slices_pred]

    weights = _slice_weights(slice_R, weight_mode=weight_mode)
    if len(weights) < len(slices_pred):
        weights = weights + [1.0] * (len(slices_pred) - len(weights))

    total_error = 0.0
    for s, slice_pred in enumerate(slices_pred):
        X_pred = slice_pred.get("Xs", [])
        X_obs = Xs_obs[s] if s < len(Xs_obs) else []
        weight = float(weights[s]) if s < len(weights) else 1.0

        for i in range(min(len(X_pred), len(X_obs))):
            diff = float(X_pred[i]) - float(X_obs[i])
            total_error += weight * diff * diff

    return total_error


def _sample_vba_like_mpt_dh(
    base_tm: float,
    base_dh: float,
    *,
    base_temperature: float = 50.0,
    melting_point_range: float = 150.0,
    heat_of_fusion_range: float = 1.0,
    rng: random.Random | None = None,
) -> Tuple[float, float]:
    """Match the legacy VBA proposal rule used for MPT/DH randomization.

    VBA logic in the legacy app:
        T(n) = BaseTemperature + RandomNumber * MeltingPointRange
        DH(n) = 0.07 * (273 + T(n)) * (0.5 + HeatOfFusionRange * RandomNumber)

    Important:
    - Tm values are generated in °C, exactly as the VBA code does.
    - The solver converts to Kelvin internally before evaluating the eutectic model.
    - The sample is centered on the legacy defaults used in the original CRYSS app:
      BaseTemperature=50, MeltingPointRange=150, HeatOfFusionRange=1.
    """
    if rng is None:
        rng = random.Random()

    u1 = rng.random()
    u2 = rng.random()
    trial_tm = float(base_temperature) + u1 * float(melting_point_range)
    trial_dh = 0.07 * (273.0 + trial_tm) * (0.5 + float(heat_of_fusion_range) * u2)
    return trial_tm, trial_dh


def monte_carlo_fit(
    mixture_case,
    Xs_obs,
    *,
    n_samples: int,
    bounds: Dict[str, Sequence[float] | Iterable[float]],
    rng: random.Random | None = None,
):
    """Random-sample Tm/DH values and keep the best objective result.

    Parameters
    ----------
    mixture_case : dict
        Parsed mixture case containing component values and slice metadata.
    Xs_obs : list[list[float]]
        Observed slice compositions for the mixture.
    n_samples : int
        Number of random proposals to evaluate.
    bounds : dict
        Maps component index or name to a (min, max) bound pair for Tm and DH.
        The expected shape is {"Tm": (lo, hi), "DH": (lo, hi)}.
    rng : random.Random, optional
        Random number generator. If omitted, a local generator is created.

    Returns
    -------
    tuple
        (best_summary, best_Tm, best_DH, best_objective)
    """
    if rng is None:
        rng = random.Random()

    # Parsed workbook state is observational only. For each trial we generate a fresh
    # Tm/DH profile using the legacy VBA-style proposal and never mutate or intercept that
    # random state during the eutectic cascade evaluation.
    Tm = [50.0] * len(mixture_case.get("Xc", mixture_case.get("X", [])))
    Xc = list(mixture_case.get("Xc", mixture_case.get("X", [])))
    DH = [0.07 * (273.0 + 50.0)] * len(Xc)
    slice_R = list(mixture_case.get("slice_R", []))

    # The PD/SVL state is a trial property, not a static parser mutation.
    # For each Monte Carlo draw we assign PD flags to the minor components in
    # the current composition vector, keeping the original component order intact.
    base_modes = assign_pd_svl(Xc, rng=rng)

    if not Xc:
        raise ValueError("mixture_case must provide Xc values for Monte Carlo fitting.")
    if not slice_R:
        raise ValueError(f"No slice_R values were available for mixture {mixture_case.get('mixture_id')}." )

    tm_bounds = bounds.get("Tm", (min(Tm), max(Tm))) if isinstance(bounds, dict) else (min(Tm), max(Tm))
    dh_bounds = bounds.get("DH", (min(DH), max(DH))) if isinstance(bounds, dict) else (min(DH), max(DH))

    best_summary = None
    best_Tm = Tm[:]
    best_DH = DH[:]
    best_objective = float("inf")

    weight_mode = "linear"

    for _ in range(max(1, int(n_samples))):
        trial_Tm = []
        trial_DH = []
        for i in range(len(Tm)):
            # Use the legacy defaults exactly as the VBA app did when drawing each new
            # component profile for a trial mixture. We never re-use or overwrite the
            # parsed case values here; each pass creates fresh thermodynamic state.
            base_tm = 50.0
            base_dh = 0.07 * (273.0 + base_tm)
            trial_tm, trial_dh = _sample_vba_like_mpt_dh(
                base_tm,
                base_dh,
                base_temperature=50.0,
                melting_point_range=150.0,
                heat_of_fusion_range=1.0,
                rng=rng,
            )
            trial_Tm.append(trial_tm)
            trial_DH.append(trial_dh)

        trial_modes = assign_pd_svl(Xc, rng=rng)
        summary = run_single_case(trial_Tm, Xc, trial_DH, trial_modes, slice_R=slice_R)
        objective = compute_monte_carlo_objective(
            summary,
            Xs_obs,
            slice_R=slice_R,
            weight_mode=weight_mode,
        )

        if objective < best_objective:
            best_objective = objective
            best_summary = summary
            best_Tm = trial_Tm[:]
            best_DH = trial_DH[:]

    if best_summary is None:
        raise RuntimeError("Monte Carlo fit did not produce a valid result.")

    return best_summary, best_Tm, best_DH, best_objective
