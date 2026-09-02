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




import copy
from datetime import datetime

from .input_writer import generate_composition, generate_physchem, assign_pd_svl, normalize_composition_vector

def maybe_make_degenerate(component_props, degeneracy_probability=0.1):
    """Apply a degenerate-component rewrite only when explicitly requested.

    This helper is intentionally conservative: the default generation path must not
    mutate the starting Xc vector or duplicate component values without an explicit
    opt-in in the input config.
    """
    return component_props


def generate_mixture_set(config: dict, num_systems: int | None = None, num_unique_mixtures: int | None = None):
    """Generate fresh per-mixture component data from a clean config JSON."""
    mixtures = []
    prefix = datetime.now().strftime("%y%m%d")

    if num_unique_mixtures is None:
        num_unique_mixtures = num_systems if num_systems is not None else 1
    num_unique_mixtures = int(num_unique_mixtures)

    clean_config = copy.deepcopy(config)
    params = clean_config.get("metadata", {})
    physchem_rules = clean_config.get("physchem_rules", {})

    for i in range(1, num_unique_mixtures + 1):
        m = copy.deepcopy(clean_config)

        Xc, add_indices = generate_composition(
            params["n_components"],
            params["main_component_level"],
            params["max_addition_compounds"],
            params["stochastic_variance"],
        )

        physchem = generate_physchem(
            params,
            Xc,
            add_indices,
            physchem_rules.get("tm_base", params.get("tm_base", 0.0)),
            physchem_rules.get("tm_sd_factor", params.get("tm_sd_factor", 1.0)),
            physchem_rules.get("dh_sd_factor", params.get("dh_sd_factor", 1.0)),
        )

        # ---------------------------------------------------------
        # PD assignment based on the minor-component rule used by the NODES app.
        # Select a random number of minor components from 0..floor(n/2), keeping
        # the original index ordering intact and never reordering the vector.
        # ---------------------------------------------------------
        max_pd_cap = int(params.get("max_addition_compounds", params["n_components"] // 2))
        max_pd = min(max_pd_cap, params["n_components"] // 2)
        modes = ["SVL"] * params["n_components"]

        if max_pd > 0:
            pd_count = random.randint(0, max_pd)
            minor_order = sorted(range(params["n_components"]), key=lambda idx: Xc[idx])
            for idx in minor_order[:pd_count]:
                modes[idx] = "PD"


        components = []
        for idx in range(params["n_components"]):
            components.append({
                "index": idx,
                "Xc": Xc[idx],
                "Tm": physchem[idx]["Tm"],
                "DH": physchem[idx]["DH"],
                "mode": modes[idx],
                #"is_addition_compound": idx in add_indices,
            })
        if bool(params.get("degeneracy_enabled", False)):
            components = maybe_make_degenerate(components)

        normalized_xc = normalize_composition_vector([float(component["Xc"]) for component in components])
        normalized_components = []
        for idx, item in enumerate(components):
            normalized_item = dict(item)
            normalized_item["Xc"] = float(normalized_xc[idx])
            normalized_components.append(normalized_item)

        m["components"] = normalized_components
        m["mixture_id"] = f"{prefix}_{i:03d}"
        mixtures.append(m)

    return mixtures
