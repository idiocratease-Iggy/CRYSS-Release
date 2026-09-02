# engine/cascade.py
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


from multxeu_core.solvers.halley import solve_eutectic_halley
from multxeu_core.solvers.vba import solve_eutectic_walkdown

from multxeu_core.engine.xeu import compute_xeu


def run_cascade(mixture, modes, solver="halley"):
    """Evaluate the eutectic cascade for one mixture.

    The solver choice is applied at the mixture level, not component-by-component.
    In particular, if a PD flag is present anywhere in the mixture, the production
    logic resolves to the VBA walkdown solver for the whole cascade step. This is
    deliberate: mixing solver formulations within the same Tk solve would create an
    inconsistent thermodynamic state.
    """
    mix = mixture.copy()
    nodes_xeu = []
    nodes_mix = []
    lambdas = []
    node_recoveries = []
    cumulative_recovery = 1.0
    tk_nodes = []
    solvent_values = []
    solvent_total = 0.0
    removed_by_step = []
    degenerate_cascade = False

    while len(mix.active_components()) > 1:
        # The cascade keeps one solver policy for the full mixture at each step. We
        # never switch between Halley and VBA on a per-component basis inside the same
        # solve. The only fallback is a whole-mixture fallback from a requested hybrid
        # solver to VBA when Halley is unstable.
        if solver == "vba":
            Tk = solve_eutectic_walkdown(mix, modes)
            result = {"T_eutectic": Tk}
        elif solver == "hybrid":
            try:
                result = solve_eutectic_halley(mix, modes)
            except RuntimeError:
                Tk = solve_eutectic_walkdown(mix, modes)
                result = {"T_eutectic": Tk}
        elif solver == "halley":
            result = solve_eutectic_halley(mix, modes)
        else:
            raise NotImplementedError(f"Solver '{solver}' is not wired at present.")

        Tk_node = result["T_eutectic"]
        tk_nodes.append(Tk_node)

        Xeu = compute_xeu(Tk_node, mix, modes)
        nodes_xeu.append(Xeu)

        active = mix.active_components()
        ratios = {
            i: Xeu[i] / mix.X[i]
            for i in active
            if mix.X[i] > 0.0 and Xeu[i] > 0.0
        }

        if not ratios:
            break

        dominant_idx = max(ratios, key=ratios.get)
        lam = ratios[dominant_idx]

        scaled = [lam * mix.X[i] for i in range(mix.n_components)]
        S_scaled = sum(scaled)

        new_X = [max(scaled[i] - Xeu[i], 0.0) for i in range(mix.n_components)]
        S_new = sum(new_X)

        if S_new <= 0.0:
            break

        removed_now = [
            i for i in range(mix.n_components)
            if mix.X[i] > 1e-12 and new_X[i] <= 1e-12
        ]
        if len(removed_now) > 1:
            degenerate_cascade = True
        removed_by_step.append(list(removed_now))

        Rk = S_new / S_scaled
        node_recoveries.append(Rk)
        cumulative_recovery *= Rk

        solvent_total += max(0.0, 1.0 - Rk)
        solvent_values.append(solvent_total)

        mix.X = [x / S_new for x in new_X]
        nodes_mix.append(mix.X[:])
        lambdas.append(lam)

    final_purity = 0.0
    if nodes_mix:
        active_indices = [i for i, x in enumerate(mix.X) if x > 1e-12]
        if active_indices:
            final_purity = mix.X[active_indices[0]]

    return (
        nodes_xeu,
        nodes_mix,
        lambdas,
        node_recoveries,
        cumulative_recovery,
        final_purity,
        tk_nodes,
        solvent_values,
        degenerate_cascade,
        removed_by_step,
    )