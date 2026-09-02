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

import math

from multxeu_core.solvers.halley import solve_eutectic_halley
from multxeu_core.solvers.vba import solve_eutectic_walkdown
from multxeu_core.engine.cascade import run_cascade
from multxeu_core.engine.xeu import compute_xeu


def _resolve_mixture_solver(requested_solver: str, modes) -> str:
    """Choose the whole-mixture solver.

    Important: we do not switch solvers component-by-component. When a mixture
    contains any PD component, the entire Tk solve is carried out with the VBA
    walkdown solver for that mixture, because the Halley implementation becomes
    numerically unstable for PD-bearing states.
    """
    if requested_solver == "halley" and any(str(mode).upper() == "PD" for mode in modes):
        return "vba"
    return requested_solver


def eutectic_path(mixture, modes, solver="halley"):
    active_ts = [mixture.T[i] for i in mixture.active_components()]

    # We intentionally solve the entire mixture with one solver state. This keeps
    # the physical model internally consistent: either the whole mixture is treated
    # with the standard SVL/Halley path, or the whole mixture is evaluated with the
    # VBA walkdown path when any PD flag is present.
    solver = _resolve_mixture_solver(solver, modes)

    if solver == "halley":
        if mixture.use_bracket:
            result = solve_eutectic_halley(mixture, modes)

    elif solver == "vba":
        Tk = solve_eutectic_walkdown(mixture, modes)
        result = {"T_eutectic": Tk}
    else:
        raise NotImplementedError(f"Solver '{solver}' is not wired at present.")

    Tk = result["T_eutectic"]
    Xeu = compute_xeu(Tk, mixture, modes)
    raw_result = run_cascade(mixture.copy(), modes, solver=solver)
    if len(raw_result) >= 10:
        nodes_xeu, nodes_mix, lambdas, node_recoveries, Rmax, final_purity, tk_nodes, solvent_values, degenerate_cascade, removed_by_step = raw_result
    else:
        nodes_xeu, nodes_mix, lambdas, node_recoveries, Rmax, final_purity, tk_nodes, solvent_values = raw_result
        degenerate_cascade = False
        removed_by_step = []

    recovery = {
        "node_recoveries": node_recoveries,
        "Rmax": Rmax,
        "final_purity": final_purity,
    }

    return {
        "solver": solver,
        "Tk": Tk,
        "Xeu": Xeu,
        "cascade": (nodes_xeu, nodes_mix),
        "recovery": recovery,
        "tk_nodes": tk_nodes,
        "solvent": solvent_values,
        "degenerate_cascade": bool(degenerate_cascade),
        "removed_components_by_step": list(removed_by_step),
        "removed_by_step": list(removed_by_step),
    }
