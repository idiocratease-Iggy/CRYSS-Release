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



"""
input_writer.py
----------------
Generates a single configuration JSON capturing the full state from
PagePhysChem. This JSON is later consumed by PageGenerateMatrix to
produce multiple mixtures.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import random

import numpy as np


@dataclass
class Params:
    n_components: int = 1
    main_component_index: int = 0
    main_component_level: float = 0.5
    stochastic_variance: float = 0.0
    max_addition_compounds: int = 0
    degeneracy_enabled: bool = False
    tm_base: float = 0.0
    tm_sd_factor: float = 1.0
    dh_sd_factor: float = 1.0
    max_solvent_volumes: int = 0
    include_rmax_slice: bool = False
    notes: str = ""


def _coerce_params(params, main_component_level=None, max_addition_compounds=None,
                  stochastic_variance=None, tm_base=None, tm_sd_factor=None,
                  dh_sd_factor=None):
    if isinstance(params, Params):
        return params

    if isinstance(params, dict):
        meta = params
        return Params(
            n_components=int(meta.get("n_components", 1)),
            main_component_index=int(meta.get("main_component_index", 0)),
            main_component_level=float(meta.get("main_component_level", 0.5)),
            stochastic_variance=float(meta.get("stochastic_variance", 0.0)),
            max_addition_compounds=int(meta.get("max_addition_compounds", 0)),
           
            tm_base=float(meta.get("tm_base", tm_base if tm_base is not None else 0.0)),
            tm_sd_factor=float(meta.get("tm_sd_factor", tm_sd_factor if tm_sd_factor is not None else 1.0)),
            dh_sd_factor=float(meta.get("dh_sd_factor", dh_sd_factor if dh_sd_factor is not None else 1.0)),
            max_solvent_volumes=int(meta.get("max_solvent_volumes", 0)),
           
        )

    return Params(
        n_components=int(params),
        main_component_index=0,
        main_component_level=float(main_component_level if main_component_level is not None else 0.5),
        stochastic_variance=float(stochastic_variance if stochastic_variance is not None else 0.0),
        max_addition_compounds=int(max_addition_compounds if max_addition_compounds is not None else 0),
        degeneracy_enabled=False,
        tm_base=float(tm_base if tm_base is not None else 0.0),
        tm_sd_factor=float(tm_sd_factor if tm_sd_factor is not None else 1.0),
        dh_sd_factor=float(dh_sd_factor if dh_sd_factor is not None else 1.0),
        max_solvent_volumes=0,
        
    )


def generate_config_id():
    return datetime.now().strftime("%y%m%d")


def normalize_composition_vector(values):
    """Return a finite, non-negative vector that sums to 1 exactly within tolerance."""
    cleaned = [float(value) for value in values]
    if not cleaned:
        raise ValueError("Composition cannot be empty.")
    if any(not math.isfinite(value) or value < 0.0 for value in cleaned):
        raise ValueError(f"Composition contains non-finite or negative values: {cleaned}")

    total = sum(cleaned)
    if total <= 0.0:
        raise ValueError(f"Composition sum must be positive: {cleaned}")
    if abs(total - 1.0) > 1e-12:
        cleaned = [value / total for value in cleaned]
    return cleaned


def generate_composition(params: Params, main_component_level=None,
                        max_addition_compounds=None, stochastic_variance=None,
                        *args, **kwargs):
    """Generate a composition vector and addition indices."""
    params = _coerce_params(params, main_component_level, max_addition_compounds,
                           stochastic_variance)

    n = params.n_components
    main_idx = params.main_component_index
    var = params.stochastic_variance
    max_add = params.max_addition_compounds

    Xc = np.zeros(n)
    Xc[main_idx] = params.main_component_level

    remaining = 1.0 - params.main_component_level
    add_count = min(max_add, max(0, n - 1))
    add_indices = []

    if add_count > 0:
        impurity_positions = [i for i in range(n) if i != main_idx]
        add_indices = np.random.choice(impurity_positions, add_count, replace=False).tolist()

    impurity_indices = [i for i in range(n) if i not in [main_idx] + add_indices]
    n_imp = len(impurity_indices)

    if n_imp > 0:
        base = remaining / (n_imp + add_count) if (n_imp + add_count) > 0 else 0.0
        imp_values = []
        for _ in impurity_indices:
            sigma = base * var
            val = np.random.normal(base, sigma)
            imp_values.append(max(val, 0.0))

        for idx, val in zip(impurity_indices, imp_values):
            Xc[idx] = val

    if add_count > 0:
        base_add = remaining / (n_imp + add_count) if (n_imp + add_count) > 0 else 0.0
        for idx in add_indices:
            sigma = base_add * var
            val = np.random.normal(base_add, sigma)
            Xc[idx] = max(val, 0.0)

    if params.degeneracy_enabled and n_imp >= 2:
        deg_pair = np.random.choice(impurity_indices, 2, replace=False)
        avg_val = (Xc[deg_pair[0]] + Xc[deg_pair[1]]) / 2
        Xc[deg_pair[0]] = avg_val
        Xc[deg_pair[1]] = avg_val

    Xc = np.asarray(normalize_composition_vector(Xc.tolist()), dtype=float)
    return Xc.tolist(), add_indices


def generate_physchem(params: Params, Xc, add_indices=None, tm_base=None,
                     tm_sd_factor=None, dh_sd_factor=None):
    """Generate Tm and DH values for each component."""
    params = _coerce_params(params, tm_base=tm_base, tm_sd_factor=tm_sd_factor,
                           dh_sd_factor=dh_sd_factor)

    n = params.n_components
    var = params.stochastic_variance
    physchem = []

    for _ in range(n):
        tm_value = params.tm_base
        sigma_t = tm_value * var
        tm_mod = max(np.random.normal(tm_value, sigma_t), 1.0)
        dh_base = 0.0714 * (273.0 + tm_mod)
        sigma_h = dh_base * var
        dh_mod = max(np.random.normal(dh_base, sigma_h), 0.1)
        physchem.append({"Tm": tm_mod, "DH": dh_mod})

    return physchem


def assign_pd_svl(params: Params, Xc, add_indices=None, pd_probability: float = 0.25):
    """Assign PD/SVL flags using the minor-component rule from the VBA/NODES workflow.

    The PD flag is distributed across the lowest-composition components only, and
    the number of PD-bearing components is sampled uniformly from 0..floor(n/2).
    The original component ordering is preserved.
    """
    n = len(Xc)
    if n == 0:
        return []

    max_pd_cap = int(params.get("max_addition_compounds", n // 2)) if isinstance(params, dict) else (n // 2)
    max_pd = min(max_pd_cap, n // 2)
    pd_count = int(np.random.randint(0, max_pd + 1)) if max_pd > 0 else 0

    ordered_indices = sorted(range(n), key=lambda idx: float(Xc[idx]))
    pd_indices = set(ordered_indices[:pd_count])

    modes = ["SVL"] * n
    for idx in pd_indices:
        modes[idx] = "PD"

    return modes


def build_configuration_json(params: Params):
    config_id = generate_config_id()

    json_data = {
        "config_id": config_id,
        "metadata": {
            "n_components": params.n_components,
            "main_component_index": params.main_component_index,
            "main_component_level": params.main_component_level,
            "stochastic_variance": params.stochastic_variance,
            "max_addition_compounds": params.max_addition_compounds,
            "num_unique_mixtures": getattr(params, "num_unique_mixtures", 10),
            #"degeneracy_enabled": params.degeneracy_enabled,
            #"notes": params.notes,
        },
        "physchem_rules": {
            "tm_base": params.tm_base,
            "tm_sd_factor": params.tm_sd_factor,
            "dh_sd_factor": params.dh_sd_factor,
            "dh_correlation": "0.0714*(273+Tm)",
        },
        "slicing": {
            "max_solvent_volumes": params.max_solvent_volumes,
          #  "include_rmax_slice": params.include_rmax_slice,
        },
    }

    return json_data


def write_json(path: str, json_data: dict):
    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)


def main():
    params = Params(
        n_components=6,
        main_component_index=0,
        main_component_level=0.80,
        stochastic_variance=0.10,
        max_addition_compounds=1,
        degeneracy_enabled=False,
        tm_base=130,
        tm_sd_factor=1.0,
        dh_sd_factor=1.0,
        max_solvent_volumes=3,
        include_rmax_slice=True,
        notes="Example configuration",
    )
    write_json("config.json", build_configuration_json(params))


if __name__ == "__main__":
    main()
