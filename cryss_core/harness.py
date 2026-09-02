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

import json
import random
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from input_parser.parser import parse_nodes_excel
from .objective_function import compute_objective_with_slices


def prompt_for_excel_file(initial_dir: str | Path | None = None) -> Path | None:
    """Open the Excel file chooser used by the CRYSS input-parser flow."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = str(initial_dir) if initial_dir else str(Path(__file__).resolve().parent / "data")
    selected = filedialog.askopenfilename(
        title="Select a NODES Excel workbook",
        initialdir=initial,
        filetypes=[("Excel files", "*.xlsx"), ("Legacy Excel files", "*.xls"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(selected) if selected else None


def choose_directory(initial_dir: str | Path | None = None, *, title: str = "Select directory") -> Path | None:
    """Open a directory chooser and return the selected folder."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = str(initial_dir) if initial_dir else str(Path(__file__).resolve().parent / "data")
    selected = filedialog.askdirectory(title=title, initialdir=initial)
    root.destroy()
    return Path(selected) if selected else None


def _timestamped_crys_rmax_filename() -> str:
    """Return the legacy timestamped CRYSS Rmax filename pattern."""
    return datetime.now().strftime("%y%m%d_%H%M") + "_CRYSS_Rmax.xlsx"


def _input_based_crys_rmax_filename(input_file: str | Path | None = None) -> str:
    """Return a workbook-based CRYSS Rmax filename such as my_input_CRYSS_Rmax.xlsx."""
    if input_file is None:
        return _timestamped_crys_rmax_filename()
    stem = Path(input_file).stem
    return f"{stem}_CRYSS_Rmax.xlsx"


def _prompt_for_renamed_output_path(current_path: Path) -> Path:
    """Ask the user for a replacement output path when the default file already exists."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.asksaveasfilename(
        title="CRYSS output file already exists; choose a different name",
        initialdir=str(current_path.parent),
        initialfile=current_path.name,
        defaultextension=".xlsx",
        filetypes=[("Excel Workbook", "*.xlsx")],
    )
    root.destroy()
    if not chosen:
        raise FileExistsError(f"Output file already exists: {current_path}. Please choose a new name.")
    return Path(chosen)


def export_crys_rmax_results(
    results: List[Dict[str, Any]] | None,
    *,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    prompt_for_dir: bool = False,
    input_file: str | Path | None = None,
) -> Path:
    """Write a simple Excel workbook with mixture_id and Rmax_CRYSS values.

    Default names now follow the input workbook stem, such as my_input_CRYSS_Rmax.xlsx.
    If that file already exists, the user is asked to rename it before the workbook is written.
    """
    if not results:
        raise ValueError("No CRYSS Rmax results were supplied for export.")

    export_rows = []
    for item in results:
        if not isinstance(item, dict):
            continue
        mixture_id = item.get("mixture_id") or item.get("case_id")
        rmax = item.get("Rmax_CRYSS")
        if mixture_id is None or rmax is None:
            continue
        export_rows.append({
            "mixture_id": str(mixture_id),
            "Rmax_CRYSS": float(rmax),
        })

    if not export_rows:
        raise ValueError("No valid CRYSS Rmax rows were found in the supplied results.")

    if output_path is not None:
        target = Path(output_path)
    else:
        directory = Path(output_dir) if output_dir is not None else (
            Path(input_file).resolve().parent if input_file is not None else Path.cwd()
        )
        if prompt_for_dir:
            chosen_dir = choose_directory(directory)
            if chosen_dir is None:
                raise FileNotFoundError("No output directory was selected for the CRYSS Rmax export.")
            directory = chosen_dir
        target = directory / _input_based_crys_rmax_filename(input_file)

        if target.exists():
            target = _prompt_for_renamed_output_path(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(export_rows, columns=["mixture_id", "Rmax_CRYSS"])
    writer = pd.ExcelWriter(target, engine="openpyxl")
    df.to_excel(writer, index=False)
    writer.close()

    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        workbook = load_workbook(target)
        worksheet = workbook.active
        header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
        header_font = Font(bold=True, color="1F1F1F")
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in worksheet.iter_rows(min_row=2, min_col=2, max_col=2):
            row[0].number_format = "0.00%"
        worksheet.column_dimensions["A"].width = 22
        worksheet.column_dimensions["B"].width = 22
        worksheet.freeze_panes = "A2"
        workbook.save(target)
    except Exception:
        pass

    return target


def resolve_case_input_path(
    explicit_path: str | Path | None = None,
    *,
    allow_prompt: bool = False,
) -> Path:
    """Return the configured workbook or an explicitly supplied Excel path.

    Root-harness execution is deliberately non-interactive. The only file chooser
    remains in input_parser/parser_harness.py, which is the parser entry point for the
    Excel workbook selection workflow.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Explicit Excel input not found: {path}")

    configured = PARAM_BLOCK.get("case_json_path")
    configured_path = Path(configured) if configured else None
    if configured_path and configured_path.exists():
        return configured_path

    raise FileNotFoundError("No Excel input file was configured for the root harness. Use the parser harness to select a workbook.")


def _stable_seed(value: Any) -> int:
    """Convert a case id or other key into a stable integer seed."""
    text = str(value)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % (2 ** 31 - 1)


def assign_pd_svl(Xc: List[float] | None, *, rng: random.Random | None = None, max_pd: int | None = None) -> List[str]:
    """Return a PD/SVL mode vector using the NODES-style rule.

    The PD flag is assigned only to the lowest-composition components, with the
    number of PD-bearing components chosen uniformly from 0 to floor(n/2). This
    matches the legacy generator behaviour: the minor components are the ones with
    the smallest Xc values, and the component order itself is never rearranged.
    """
    if Xc is None:
        raise ValueError("Xc must be provided to assign PD/SVL modes.")

    components = [float(v) for v in Xc]
    n = len(components)
    if n == 0:
        return []

    if rng is None:
        rng = random.Random()

    pd_cap = n // 2 if max_pd is None else max(0, min(max_pd, n // 2))
    pd_count = int(rng.randint(0, pd_cap))

    ordered_indices = sorted(range(n), key=lambda idx: components[idx])
    pd_indices = set(ordered_indices[:pd_count])

    modes = ["SVL"] * n
    for idx in pd_indices:
        modes[idx] = "PD"
    return modes


from multxeu_core.mixture_input import MixtureInput
from multxeu_core.engine.eutectic_path import eutectic_path
from multxeu_core.multx_slicing_engine.canonical_eutectic import build_canonical_state
from multxeu_core.multx_slicing_engine.canonical_gradients import compute_segment_gradients
from multxeu_core.multx_slicing_engine.evaluate_recovery import evaluate_at_recovery
from .report_component_errors import (
    compute_structured_gradient,
    compute_structured_objective,
    report_component_errors,
    report_component_slice_deltas,
)



# ---------------------------------------------------------------------------
# Default architecture note:
#   - The most likely production path is a single physchem state (one Tm/DH/mode
#     configuration) evaluated at multiple slice R values.
#   - Batch / list-of-cases support remains available for future scanning or
#     stochastic/optimized sweeps, but the wrapper should not force that shape
#     onto the common single-run use case.
# ---------------------------------------------------------------------------
OPTIMISER_DEFAULTS: Dict[str, Any] = {
    "eta_T": 0.5,
    "eta_DH": 0.5,
    "eps_T": 0.05,
    "eps_DH": 0.05,
    "max_iter": 20,
    "tol": 1e-6,
}

PARAM_BLOCK: Dict[str, Any] = {
    "case_json_path": r"E:\Dev\05 cryss core\data\260830_1133_Vol01.xlsx",
    "solver": "vba",
    "slice_R": [0.90, 0.80, 0.70],
    "use_bracket": True,
    "optimiser": dict(OPTIMISER_DEFAULTS),
    "notes": "Primary project flow reads Excel input through the input_parser pipeline; JSON is compatibility-only.",
}


def build_cases_from_parsed_mixtures(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert parser output into harness case dicts while preserving MixtureSet metadata."""
    cases: List[Dict[str, Any]] = []
    for mixture_id, mixset in parsed.items():
        if not mixset.slices:
            continue

        slice_records = list(mixset.slices)
        first_slice = slice_records[0]
        slice_R_values = [float(slice_record.R) for slice_record in slice_records]
        slice_Xs_values = [[float(v) for v in slice_record.Xs] for slice_record in slice_records]

        rmax_candidates = []
        for slice_record in slice_records:
            for row in getattr(slice_record, "raw_rows", []):
                raw_rmax = row.get("rmax", row.get("Rmax", row.get("r_max")))
                if raw_rmax is None:
                    continue
                try:
                    rmax_candidates.append(float(raw_rmax))
                except (TypeError, ValueError):
                    continue

        # A minimal component-only workbook does not contain a NODES Rmax reference.
        # Do not infer this from the observed slice recoveries; that would fabricate a
        # comparison axis and trigger misleading plots.
        rmax_nodes = rmax_candidates[0] if rmax_candidates else None

        x_values = [float(v) for v in first_slice.Xc]
        case = {
            "case_id": mixture_id,
            "mixture_id": mixture_id,
            "Xc": x_values,
            # Parsed workbook data provides composition and slice observations only.
            # The Monte Carlo optimizer must generate its own per-trial Tm/DH values.
            # Do not derive Tm from Xc or hard-code DH to 25.0 here.
            "Tm": [50.0] * len(x_values),
            "DH": [0.07 * (273.0 + 50.0)] * len(x_values),
            "mode": assign_pd_svl(x_values, rng=random.Random(_stable_seed(mixture_id))),
            "slice_R": slice_R_values,
            "slice_Xs": slice_Xs_values,
            "parsed_mixture": mixset,
            "Rmax_NODES": float(rmax_nodes) if rmax_nodes is not None else None,
            "RelSolVol_Rmax_NODES": None,
            "slice_metadata": [
                {
                    "slice_number": slice_record.slice_number,
                    "R": float(slice_record.R),
                    "Xs": [float(v) for v in slice_record.Xs],
                    "Xl": [float(v) for v in slice_record.Xl],
                }
                for slice_record in slice_records
            ],
        }
        cases.append(case)

    if not cases:
        raise ValueError("No valid mixtures were parsed from the Excel workbook.")

    return cases


def load_excel_cases(path: str | Path) -> List[Dict[str, Any]]:
    """Convert parsed Excel mixtures into the harness case dicts used by the optimiser."""
    input_path = Path(path)
    parsed = parse_nodes_excel(input_path, write_excel=False)
    return build_cases_from_parsed_mixtures(parsed)


def resolve_optimiser_params(mixture_case: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return optimiser hyperparameters with per-mixture overrides layered on defaults."""
    params = dict(OPTIMISER_DEFAULTS)
    params.update(PARAM_BLOCK.get("optimiser", {}))

    if isinstance(mixture_case, dict):
        case_overrides = mixture_case.get("optimiser", {})
        if isinstance(case_overrides, dict):
            params.update(case_overrides)

    return params


def build_mixture(case: Dict[str, Any], *, use_bracket: bool = True) -> MixtureInput:
    parsed = case.get("parsed_mixture")
    if parsed is not None and getattr(parsed, "slices", None):
        first_slice = parsed.slices[0]
        X = [float(v) for v in first_slice.Xc]
        # Parsed Xc is observational data only; do not treat it as the optimizer state.
        # The Monte Carlo path generates trial Tm/DH values independently for each pass.
        T = [50.0] * len(X)
        DH = [0.07 * (273.0 + 50.0)] * len(X)
        modes = ["SVL"] * len(X)
    else:
        components = case.get("components")
        if isinstance(components, list) and components:
            X = [float(item["Xc"]) for item in components]
            T = [float(item["Tm"]) for item in components]
            DH = [float(item["DH"]) for item in components]
            modes = [str(item.get("mode", "SVL")).upper() for item in components]
        else:
            X = [float(v) for v in case["Xc"]]
            T = [float(v) for v in case["Tm"]]
            DH = [float(v) for v in case["DH"]]
            default_modes = case.get("mode", ["SVL"] * len(X))
            modes = [str(v).upper() for v in default_modes]

    return MixtureInput(
        T=T,
        DH=DH,
        X=X,
        case_id=str(case.get("case_id", case.get("mixture_id", "case_001"))),
        modes=modes,
        use_bracket=use_bracket,
    )


def run_case(case: Dict[str, Any], *, solver: str, slice_R: List[float]) -> Dict[str, Any]:
    mix = build_mixture(case, use_bracket=PARAM_BLOCK["use_bracket"])
    raw = eutectic_path(mix, mix.modes, solver=solver)
    state = build_canonical_state(mix, modes=mix.modes, solver=solver)
    grads = compute_segment_gradients(state)

    slice_results = []
    for R in slice_R:
        result = evaluate_at_recovery(state, grads, float(R))
        xliq = [float(v) for v in result.Xliq]
        xsol = [float(v) for v in result.Xsol]
        slice_results.append({
            "R": float(result.R),
            "Xs": xliq,
            "Xliq": xliq,
            "Xsol": xsol,
            "RelSolVol_at_R": float(result.solvent),
            "seg_index": result.seg_index,
            "seg_start_R": result.seg_start_R,
            "seg_end_R": result.seg_end_R,
        })

    return {
        "case_id": mix.case_id,
        "solver": solver,
        "Xc": [float(v) for v in mix.X],
        "Tm": [float(v) for v in mix.T],
        "DH": [float(v) for v in mix.DH],
        "mode": [str(v).upper() for v in mix.modes],
        "raw_result": raw,
        "state": state,
        "segment_grads": grads,
        "slice_results": slice_results,
        "Rmax": float(getattr(state, "Rmax", raw.get("recovery", {}).get("Rmax", 0.0))),
        "RelSolVol_at_Rmax": float(getattr(state, "solvent_at_Rmax", 0.0)),
    }
def fd_gradient_Tm_DH(Tm, Xc, DH, modes, Xs_obs, eps_T=0.05, eps_DH=0.05, alpha=1.0, beta=1.0, slice_R=None):
    """Finite-difference gradients for Tm and DH."""
    slice_R = list(slice_R) if slice_R is not None else list(PARAM_BLOCK["slice_R"])

    # Base objective
    base_summary = run_single_case(Tm, Xc, DH, modes, slice_R=slice_R)
    base_obj = compute_structured_objective(base_summary, Xs_obs, alpha=alpha, beta=beta)["total_objective"]

    G_Tm = [0.0] * len(Tm)
    G_DH = [0.0] * len(DH)

    # Tm gradients
    for i in range(len(Tm)):
        Tm_neg = Tm[:]; Tm_neg[i] = Tm[i] - eps_T
        Tm_pos = Tm[:]; Tm_pos[i] = Tm[i] + eps_T

        obj_neg = compute_structured_objective(run_single_case(Tm_neg, Xc, DH, modes, slice_R=slice_R), Xs_obs, alpha, beta)["total_objective"]
        obj_pos = compute_structured_objective(run_single_case(Tm_pos, Xc, DH, modes, slice_R=slice_R), Xs_obs, alpha, beta)["total_objective"]

        G_Tm[i] = (obj_pos - obj_neg) / (2.0 * eps_T)

    # DH gradients
    for i in range(len(DH)):
        DH_neg = DH[:]; DH_neg[i] = DH[i] - eps_DH
        DH_pos = DH[:]; DH_pos[i] = DH[i] + eps_DH

        obj_neg = compute_structured_objective(run_single_case(Tm, Xc, DH_neg, modes, slice_R=slice_R), Xs_obs, alpha, beta)["total_objective"]
        obj_pos = compute_structured_objective(run_single_case(Tm, Xc, DH_pos, modes, slice_R=slice_R), Xs_obs, alpha, beta)["total_objective"]

        G_DH[i] = (obj_pos - obj_neg) / (2.0 * eps_DH)

    return base_summary, base_obj, G_Tm, G_DH

def optimise_Tm_DH(Tm_init, Xc, DH_init, modes, Xs_obs,
                   alpha=1.0, beta=1.0,
                   eta_T=OPTIMISER_DEFAULTS["eta_T"], eta_DH=OPTIMISER_DEFAULTS["eta_DH"],
                   eps_T=OPTIMISER_DEFAULTS["eps_T"], eps_DH=OPTIMISER_DEFAULTS["eps_DH"],
                   max_iter=OPTIMISER_DEFAULTS["max_iter"], tol=OPTIMISER_DEFAULTS["tol"],
                   slice_R=None):
    Tm = Tm_init[:]
    DH = DH_init[:]
    slice_R = list(slice_R) if slice_R is not None else list(PARAM_BLOCK["slice_R"])

    for k in range(max_iter):
        summary, obj, G_Tm, G_DH = fd_gradient_Tm_DH(
            Tm, Xc, DH, modes, Xs_obs,
            eps_T=eps_T, eps_DH=eps_DH,
            alpha=alpha, beta=beta,
            slice_R=slice_R,
        )

        # print(f"\nIter {k}: objective = {obj:.6f}")
        # print("  Tm gradients:", ["{:+.4f}".format(g) for g in G_Tm])
        # print("  DH gradients:", ["{:+.4f}".format(g) for g in G_DH])

        # Gradient descent updates
        Tm_new = [t - eta_T * g for t, g in zip(Tm, G_Tm)]
        DH_new = [h - eta_DH * g for h, g in zip(DH, G_DH)]

        # Check improvement
        new_summary = run_single_case(Tm_new, Xc, DH_new, modes, slice_R=slice_R)
        new_obj = compute_structured_objective(new_summary, Xs_obs, alpha=alpha, beta=beta)["total_objective"]

        # print(f"  Proposed objective: {new_obj:.6f}")

        if abs(obj - new_obj) < tol:
            # print("\nConverged (Δobj < tol).")
            return new_summary, Tm_new, DH_new

        # Accept step
        Tm, DH = Tm_new, DH_new

    # print("\nReached max_iter without strict convergence.")
    final_summary = run_single_case(Tm, Xc, DH, modes, slice_R=slice_R)
    final_obj = compute_structured_objective(final_summary, Xs_obs, alpha=alpha, beta=beta)["total_objective"]
    # print(f"Final objective: {final_obj:.6f}")
    return final_summary, Tm, DH


def summarize_core_output(case: Dict[str, Any], *, solver: str, slice_R: List[float]) -> Dict[str, Any]:
    result = run_case(case, solver=solver, slice_R=slice_R)
    summary = {
        "case_id": result["case_id"],
        "solver": result["solver"],
        "Xc": result["Xc"],
        "Rmax": result["Rmax"],
        "RelSolVol_at_Rmax": result["RelSolVol_at_Rmax"],
        "Tm": result["Tm"],
        "DH": result["DH"],
        "mode": result["mode"],
        "slices": [],
    }

    for item in result["slice_results"]:
        summary["slices"].append({
            "slice_R": item["R"],
            "Xs": item["Xs"],
            "Xliq": item["Xliq"],
            "Xsol": item["Xsol"],
            "RelSolVol_at_R": item["RelSolVol_at_R"],
            "Rmax": result["Rmax"],
            "RelSolVol_at_Rmax": result["RelSolVol_at_Rmax"],
        })

    return summary


def print_core_summary(summary: Dict[str, Any]) -> None:
    print(f"\nCase {summary['case_id']} | solver={summary['solver']}")
    print(f"Xc = {summary['Xc']}")
    print(f"Rmax = {summary['Rmax']:.6f}")
    print(f"RelSolVol@Rmax = {summary['RelSolVol_at_Rmax']:.6f}")
    print(f"Tm = {summary['Tm']}")
    print(f"DH = {summary['DH']}")
    print(f"mode = {summary['mode']}")

    for slice_info in summary["slices"]:
        print(
            f"slice R={slice_info['slice_R']:.4f} | "
            f"Xs={slice_info['Xs']} | "
            f"Xliq={slice_info['Xliq']} | "
            f"RelSolVol@R={slice_info['RelSolVol_at_R']:.6f} | "
            f"Rmax={slice_info['Rmax']:.6f} | "
            f"RelSolVol@Rmax={slice_info['RelSolVol_at_Rmax']:.6f}"
        )


def run_case_summary(case: Dict[str, Any], *, solver: str, slice_R: List[float]) -> Dict[str, Any]:
    """Return the clean summary dictionary for one physchem state.

    This is the preferred default path for the application: one Tm/DH/mode
    configuration evaluated across multiple slice R values. Batch handling is
    still available via run_harness() for future scans or sweeps.
    """
    resolved_slice_R = list(case.get("slice_R", slice_R)) if isinstance(case, dict) else list(slice_R)
    if not resolved_slice_R:
        resolved_slice_R = list(slice_R)
    return summarize_core_output(case, solver=solver, slice_R=resolved_slice_R)


def run_harness() -> List[Dict[str, Any]]:
    """Batch wrapper for the active Excel-driven flow.

    JSON input is intentionally disabled here. The project flow must use the
    Excel parser at input_parser/parser.py and never silently execute legacy JSON.
    """
    input_path = resolve_case_input_path()
    suffix = input_path.suffix.lower()

    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(
            f"Excel-only mode is enabled; got '{input_path}' instead of an .xlsx/.xls file. "
            "Please select a workbook through the parser pipeline."
        )

    cases = load_excel_cases(input_path)

    summaries = []
    for case in cases:
        parser_slice_R = case.get("slice_R") or list(PARAM_BLOCK["slice_R"])
        summary = run_case_summary(
            case,
            solver=PARAM_BLOCK["solver"],
            slice_R=list(parser_slice_R),
        )
        summaries.append(summary)
    return summaries


def run_single_case(Tm, Xc, DH, modes, slice_R=None):
    """Run the solver for a single perturbed physchem state."""
    case = {
        "case_id": "perturbation_case",
        "Xc": list(Xc),
        "Tm": list(Tm),
        "DH": list(DH),
        "mode": list(modes),
    }
    selected_slice_R = list(slice_R) if slice_R is not None else list(PARAM_BLOCK["slice_R"])
    return run_case_summary(
        case,
        solver=PARAM_BLOCK["solver"],
        slice_R=selected_slice_R,
    )


def compute_tensor_objective(summary: Dict[str, Any], Xs_obs: List[List[float]]) -> float:
    """Return a secondary rank-weighted objective for MC/optimizer comparison."""
    from .objective_function import compute_tensor_objective as _compute_tensor_objective

    return float(_compute_tensor_objective(summary, Xs_obs))


def compute_direct_slice_objective(summary: Dict[str, Any], Xs_obs: List[List[float]], *, slice_R: List[float] | None = None, weights: List[float] | None = None) -> float:
    """Return the direct slice-fit objective used for both MC ranking and local refinement.

    This intentionally matches the MC-style objective: compare the model-predicted Xs at
    the known slice recoveries against the observed Xs at those slice recoveries. The
    tensor objective remains a secondary diagnostic and is not used to drive the optimizer.
    """
    from optimisers.monte_carlo import compute_monte_carlo_objective

    return float(compute_monte_carlo_objective(summary, Xs_obs, slice_R=slice_R, weight_mode="linear"))


def run_mc_candidate_pool(case: Dict[str, Any], Xs_obs: List[List[float]], *, n_samples: int = 1000, top_k: int = 10, bounds: Dict[str, Any] | None = None, rng: random.Random | None = None):
    """Generate the Monte Carlo sample pool and keep the best candidate set.

    Returns a dict with both the full population and the top-K seed candidates.
    """
    if rng is None:
        rng = random.Random()

    Xc = [float(v) for v in case.get("Xc", [])]
    if not Xc:
        raise ValueError("case must include Xc to build the Monte Carlo candidate pool.")

    Tm = [float(v) for v in case.get("Tm", [50.0] * len(Xc))]
    DH = [float(v) for v in case.get("DH", [0.07 * (273.0 + 50.0)] * len(Xc))]
    modes = [str(v).upper() for v in case.get("mode", ["SVL"] * len(Xc))]
    slice_R = list(case.get("slice_R", PARAM_BLOCK.get("slice_R", [])))

    from optimisers.monte_carlo import _sample_vba_like_mpt_dh, compute_monte_carlo_objective

    all_candidates = []
    for sample_id in range(max(1, int(n_samples))):
        trial_Tm = []
        trial_DH = []
        for i in range(len(Xc)):
            trial_tm, trial_dh = _sample_vba_like_mpt_dh(
                50.0,
                0.07 * (273.0 + 50.0),
                base_temperature=50.0,
                melting_point_range=150.0,
                heat_of_fusion_range=1.0,
                rng=rng,
            )
            trial_Tm.append(float(trial_tm))
            trial_DH.append(float(trial_dh))

        trial_modes = assign_pd_svl(Xc, rng=rng)
        summary = run_single_case(trial_Tm, Xc, trial_DH, trial_modes, slice_R=slice_R)
        objective = float(compute_monte_carlo_objective(summary, Xs_obs, slice_R=slice_R, weight_mode="linear"))
        tensor_objective = float(compute_tensor_objective(summary, Xs_obs))
        candidate = {
            "sample_id": sample_id,
            "source": "mc",
            "Tm": list(trial_Tm),
            "DH": list(trial_DH),
            "modes": list(trial_modes),
            "objective": objective,
            "tensor_objective": tensor_objective,
            "Rmax": float(summary.get("Rmax", 0.0)),
            "RelSolVol_at_Rmax": float(summary.get("RelSolVol_at_Rmax", 0.0)),
            "summary": summary,
        }
        all_candidates.append(candidate)

    all_candidates.sort(key=lambda item: item["objective"])
    top_candidates = all_candidates[: max(1, min(len(all_candidates), int(top_k)))]
    return {
        "all_candidates": all_candidates,
        "top_candidates": top_candidates,
        "best_objective": all_candidates[0]["objective"] if all_candidates else None,
        "best_candidate": all_candidates[0] if all_candidates else None,
    }


def refine_top_candidates(case: Dict[str, Any], top_candidates: List[Dict[str, Any]], Xs_obs: List[List[float]], *, slice_R: List[float] | None = None, alpha: float = 1.0, beta: float = 1.0, best_only: bool = False):
    """Refine MC seeds using a direct slice-fit objective and a bounded local search.

    The comparison run you want to test is the legacy top-K behavior: multiple MC seeds
    are refined and the full set is retained for plotting. The single-best seed remains
    available as an explicit choice via ``best_only=True`` when needed for a narrower.
    """
    if not top_candidates:
        return []

    ranked_candidates = sorted(top_candidates, key=lambda item: float(item.get("objective", float("inf"))))
    chosen_candidates = ranked_candidates[:1] if best_only else ranked_candidates

    Xc = [float(v) for v in case.get("Xc", [])]
    if not Xc:
        raise ValueError("case must include Xc to refine the top Monte Carlo candidates.")

    default_modes = [str(v).upper() for v in case.get("mode", ["SVL"] * len(Xc))]
    selected_slice_R = list(slice_R) if slice_R is not None else list(case.get("slice_R", PARAM_BLOCK.get("slice_R", [])))

    refined = []
    for idx, seed in enumerate(chosen_candidates):
        base_Tm = [float(v) for v in seed.get("Tm", [50.0] * len(Xc))]
        base_DH = [float(v) for v in seed.get("DH", [0.07 * (273.0 + 50.0)] * len(Xc))]
        seed_modes = [str(v).upper() for v in seed.get("modes", default_modes)]

        best_summary = run_single_case(base_Tm, Xc, base_DH, seed_modes, slice_R=selected_slice_R)
        best_obj = compute_direct_slice_objective(best_summary, Xs_obs, slice_R=selected_slice_R)
        best_Tm = base_Tm[:]
        best_DH = base_DH[:]

        for _ in range(6):
            improved = False
            for i in range(len(Xc)):
                for tm_step in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20):
                    trial_Tm = best_Tm[:]
                    trial_Tm[i] = best_Tm[i] * (1.0 + tm_step)
                    for dh_step in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20):
                        trial_DH = best_DH[:]
                        trial_DH[i] = best_DH[i] * (1.0 + dh_step)
                        trial_summary = run_single_case(trial_Tm, Xc, trial_DH, seed_modes, slice_R=selected_slice_R)
                        trial_obj = compute_direct_slice_objective(trial_summary, Xs_obs, slice_R=selected_slice_R)
                        if trial_obj < best_obj:
                            best_obj = trial_obj
                            best_summary = trial_summary
                            best_Tm = trial_Tm[:]
                            best_DH = trial_DH[:]
                            improved = True
            if not improved:
                break

        refined_candidate = {
            "sample_id": seed.get("sample_id", idx),
            "source": "mc_plus_optimizer",
            "Tm": list(best_Tm),
            "DH": list(best_DH),
            "modes": list(seed_modes),
            "objective": float(best_obj),
            "tensor_objective": float(compute_tensor_objective(best_summary, Xs_obs)),
            "Rmax": float(best_summary.get("Rmax", 0.0)),
            "RelSolVol_at_Rmax": float(best_summary.get("RelSolVol_at_Rmax", 0.0)),
            "summary": best_summary,
        }
        refined.append(refined_candidate)

    refined.sort(key=lambda item: item["objective"])
    return refined


def plot_mc_vs_optimizer_results(mc_candidates: List[Dict[str, Any]], refined_candidates: List[Dict[str, Any]] | None = None, *, x_key: str = "Rmax", y_key: str = "objective"):
    """Scatter-plot MC-only and MC+optimizer candidate sets on the same axes."""
    mc_values = list(mc_candidates) if isinstance(mc_candidates, list) else []
    ref_values = list(refined_candidates) if isinstance(refined_candidates, list) else []

    if not mc_values:
        raise ValueError("mc_candidates must contain at least one candidate.")

    fig, ax = plt.subplots()
    ax.scatter(
        [float(item.get(x_key, 0.0)) for item in mc_values],
        [float(item.get(y_key, 0.0)) for item in mc_values],
        color="tab:blue",
        alpha=0.7,
        label="MC",
    )
    if ref_values:
        ax.scatter(
            [float(item.get(x_key, 0.0)) for item in ref_values],
            [float(item.get(y_key, 0.0)) for item in ref_values],
            color="tab:orange",
            alpha=0.8,
            label="MC + optimizer",
        )

    ax.set_xlabel("Rmax")
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.set_title("Monte Carlo vs MC + optimizer fit distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def analyse_mc_and_optimizer_runs(case: Dict[str, Any], Xs_obs: List[List[float]], *, n_runs: int = 10, n_samples: int = 200, top_k: int = 10, seed: int = 42):
    """Run the MC pool and optimizer refinement across several seeds for run-to-run error estimation.

    Returns the full per-run candidate store and a compact summary used for plotting.
    """
    profiles = []
    for run_idx in range(max(1, int(n_runs))):
        rng = random.Random(seed + run_idx)
        pool = run_mc_candidate_pool(case, Xs_obs, n_samples=n_samples, top_k=top_k, rng=rng)
        refined = refine_top_candidates(case, pool["top_candidates"], Xs_obs, slice_R=list(case.get("slice_R", [])))
        best_mc = pool["top_candidates"][0] if pool["top_candidates"] else None
        best_opt = refined[0] if refined else None
        rmax_nodes = case.get("Rmax_NODES")
        profiles.append({
            "run_id": run_idx,
            "seed": seed + run_idx,
            "Rmax_NODES": float(rmax_nodes) if rmax_nodes is not None else None,
            "mc_top_candidates": pool["top_candidates"],
            "optimized_top_candidates": refined,
            "best_mc_candidate": best_mc,
            "best_optimized_candidate": best_opt,
            "mc_best_Rmax": float(best_mc["Rmax"]) if best_mc is not None else None,
            "optimized_best_Rmax": float(best_opt["Rmax"]) if best_opt is not None else None,
            "mc_best_objective": float(best_mc["objective"]) if best_mc is not None else None,
            "optimized_best_objective": float(best_opt["objective"]) if best_opt is not None else None,
        })
    return profiles


def plot_mc_optimizer_run_summary(run_profiles: List[Dict[str, Any]], *, save_path: str | Path | None = None):
    """Create a two-panel view:
    1) top 10 MC and top 10 optimized candidates for a single run
    2) NODES-vs-CRYSS spread across multiple runs, with the best optimized result per run.
    """
    if not run_profiles:
        raise ValueError("run_profiles must contain at least one run. ")

    run = run_profiles[0]
    mc_top = run.get("mc_top_candidates", [])[:10]
    opt_top = run.get("optimized_top_candidates", [])[:10]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    if mc_top:
        axes[0].scatter(
            [float(c["Rmax"]) for c in mc_top],
            [float(c["objective"]) for c in mc_top],
            color="tab:blue",
            alpha=0.7,
            label="MC best 10",
        )
    if opt_top:
        axes[0].scatter(
            [float(c["Rmax"]) for c in opt_top],
            [float(c["objective"]) for c in opt_top],
            color="tab:orange",
            alpha=0.8,
            label="Optimized best 10",
        )
    axes[0].set_xlabel("Rmax (CRYSS)")
    axes[0].set_ylabel("Objective")
    axes[0].set_title("Top 10 Monte Carlo seeds vs top 10 optimized candidates")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    nodes_values = []
    cryss_values = []
    run_labels = []
    for profile in run_profiles:
        nodes_rmax = float(profile.get("Rmax_NODES", 0.0))
        best_opt = profile.get("best_optimized_candidate")
        if best_opt is None:
            continue
        nodes_values.append(100.0 * nodes_rmax)
        cryss_values.append(100.0 * float(best_opt["Rmax"]))
        run_labels.append(f"run {profile.get('run_id', 0)}")

    if nodes_values:
        axes[1].scatter(nodes_values, cryss_values, color="tab:green", s=50, alpha=0.8)
        for x_val, y_val, label in zip(nodes_values, cryss_values, run_labels):
            axes[1].annotate(label, (x_val, y_val), fontsize=8)
        axes[1].plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1.0, label="y=x")
        axes[1].set_xlim(0, 100)
        axes[1].set_ylim(0, 100)
        axes[1].set_xlabel("Rmax (NODES, %)")
        axes[1].set_ylabel("Rmax (CRYSS, %)")
        axes[1].set_title("Best optimized Rmax across multiple runs")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(Path(save_path), dpi=200)
    plt.show()
    return fig, axes


def summarize_mc_optimizer_improvements(run_profiles: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Return per-run improvement metrics from the best MC candidate to the best optimized candidate."""
    rows: List[Dict[str, float]] = []
    for profile in run_profiles:
        best_mc = profile.get("best_mc_candidate")
        best_opt = profile.get("best_optimized_candidate")
        if best_mc is None or best_opt is None:
            continue
        mc_rmax = float(best_mc["Rmax"])
        opt_rmax = float(best_opt["Rmax"])
        delta_rmax = opt_rmax - mc_rmax
        relative_improvement = 100.0 * ((mc_rmax - opt_rmax) / max(abs(mc_rmax), 1e-9))
        rows.append({
            "run_id": float(profile.get("run_id", 0)),
            "mc_best_rmax": mc_rmax,
            "optimized_best_rmax": opt_rmax,
            "delta_rmax": delta_rmax,
            "relative_improvement_percent": relative_improvement,
        })
    return rows


def perturbation_test(summary, Xs_obs, Tm, Xc, DH, modes, eps=0.1):
    """Evaluate local objective changes for component-wise Tm perturbations."""
    print("\n=== Perturbation Test ===")
    print(f"Using eps = {eps}\n")

    base_obj = compute_structured_objective(summary, Xs_obs)["total_objective"]
    print(f"Base objective: {base_obj:.6f}\n")

    for i in range(len(Tm)):
        print(f"Component {i+1}:")

        Tm_neg = Tm[:]
        Tm_neg[i] = Tm[i] - eps
        summary_neg = run_single_case(Tm_neg, Xc, DH, modes)
        obj_neg = compute_structured_objective(summary_neg, Xs_obs)["total_objective"]

        Tm_pos = Tm[:]
        Tm_pos[i] = Tm[i] + eps
        summary_pos = run_single_case(Tm_pos, Xc, DH, modes)
        obj_pos = compute_structured_objective(summary_pos, Xs_obs)["total_objective"]

        print(f"  Tm[i] - eps: {Tm_neg[i]:.4f}, objective = {obj_neg:.6f}")
        print(f"  Tm[i] + eps: {Tm_pos[i]:.4f}, objective = {obj_pos:.6f}")

        if obj_neg > base_obj and obj_pos > base_obj:
            print("  → Local minimum (convex)")
        elif obj_neg < base_obj and obj_pos < base_obj:
            print("  → Local maximum (concave)")
        else:
            print("  → Mixed behaviour (saddle or flat)")

        print()


def _extract_summary_state(summary: Dict[str, Any]):
    return summary["Tm"], summary["Xc"], summary["DH"], summary["mode"]


def debug_reference_physchem_against_slices(
    case: Dict[str, Any],
    *,
    slice_R: List[float] | None = None,
    Xs_obs: List[List[float]] | None = None,
    solver: str = PARAM_BLOCK["solver"],
):
    """Run a known-good physchem profile directly through the eutectic cascade.

    This is intended as an interactive debug harness for validating the model against
    workbook slice data. The passed case should already contain the true Tm/DH values
    and component composition. No Monte Carlo trial values are involved here.
    """
    if not isinstance(case, dict):
        raise TypeError("case must be a dictionary-like mixture payload")

    Xc = [float(v) for v in case.get("Xc", [item["Xc"] for item in case.get("components", [])])]
    Tm = [float(v) for v in case.get("Tm", [item["Tm"] for item in case.get("components", [])])]
    DH = [float(v) for v in case.get("DH", [item["DH"] for item in case.get("components", [])])]
    modes = [str(v).upper() for v in case.get("mode", [item.get("mode", "SVL") for item in case.get("components", [])])]

    if not slice_R:
        slice_R = list(case.get("slice_R", []))
    if not slice_R:
        raise ValueError("No slice_R values were supplied for the debug reference case.")

    summary = run_single_case(Tm, Xc, DH, modes, slice_R=slice_R)
    print("\n=== Direct cascade debug ===")
    print(f"case_id={case.get('mixture_id', case.get('case_id'))}")
    print("Tm:", Tm)
    print("DH:", DH)
    print("slice_R:", slice_R)

    for idx, sl in enumerate(summary["slices"]):
        print(f"\nSlice {idx+1} at R={sl['slice_R']}: Xs_pred={sl['Xs']}")
        if Xs_obs is not None and idx < len(Xs_obs):
            print(f"Xs_obs={Xs_obs[idx]}")

    if Xs_obs is not None:
        obj = compute_structured_objective(summary, Xs_obs, alpha=1.0, beta=1.0)
        print("\nStructured objective:", obj)
        return summary, obj

    return summary, None


def _print_objective_report(summary: Dict[str, Any], Xs_obs: List[List[float]]) -> None:
    obj = compute_structured_objective(summary, Xs_obs, alpha=1.0, beta=1.0)

    print(f"\nStructured objective value: {obj['total_objective']:.6f}")
    print(f"  Pointwise error term:     {obj['pointwise_error']:.6f}")
    print(f"  Slice-delta penalty term: {obj['delta_penalty']:.6f}")

    print("\nSlice-wise errors:")
    for idx, err in enumerate(obj["slice_errors"]):
        print(f"  Slice {idx+1}: {err:.6f}")

    print("\nComponent-wise errors:")
    for idx, err in enumerate(obj["component_errors"]):
        print(f"  Component {idx+1}: {err:.6f}")

    print("\nComponent slice-delta penalties:")
    for idx, pen in enumerate(obj["component_delta_penalties"]):
        print(f"  Component {idx+1}: {pen:.6f}")


def _print_optimised_summary(final_summary: Dict[str, Any], Tm_opt, DH_opt) -> None:
    print("\nOptimised Tm:", Tm_opt)
    print("Optimised DH:", DH_opt)
    print(f"Optimised Rmax = {final_summary['Rmax']:.6f}")
    print(f"Optimised RelSolVol@Rmax = {final_summary['RelSolVol_at_Rmax']:.6f}")


def optimise_mixture(mixture_case, Xs_obs):
    """Run the active front-end optimiser for a parsed mixture.

    The legacy gradient-based fitter remains available as ``optimise_Tm_DH`` and
    ``fd_gradient_Tm_DH`` for later stages. This entry point now swaps the
    optimizer driver to the Monte Carlo front-end while preserving the existing
    harness return contract.
    """
    # The parsed mixture may carry legacy placeholder Tm/DH values for compatibility,
    # but the optimizer must only ever optimize the trial values generated during each
    # Monte Carlo pass. Ignore any intercepted parsed-state numbers here.
    Tm = [50.0] * len(mixture_case.get("Xc", mixture_case.get("X", [])))
    Xc = list(mixture_case.get("Xc", mixture_case.get("X", [])))
    DH = [0.07 * (273.0 + 50.0)] * len(Xc)
    modes = list(mixture_case.get("mode", mixture_case.get("modes", ["SVL"] * len(Xc))))

    if not Tm or not Xc or not DH:
        raise ValueError("mixture_case must provide Tm/Xc/DH values for optimisation.")

    if Xs_obs is None:
        Xs_obs = mixture_case.get("slice_Xs", [])
    if not Xs_obs:
        raise ValueError(f"No parser-supplied Xs observations were found for mixture {mixture_case.get('mixture_id')}.")

    slice_R = list(mixture_case.get("slice_R", PARAM_BLOCK["slice_R"]))
    n_samples = 1000
    tm_min = min(Tm) if Tm else 0.0
    tm_max = max(Tm) if Tm else 1.0
    dh_min = min(DH) if DH else 0.0
    dh_max = max(DH) if DH else 1.0

    bounds = {
        "Tm": (tm_min * 0.75, tm_max * 1.25),
        "DH": (dh_min * 0.75, dh_max * 1.25),
    }

    final_summary, Tm_opt, DH_opt, final_obj = monte_carlo_fit(
        mixture_case,
        Xs_obs,
        n_samples=n_samples,
        bounds=bounds,
    )

    Rmax_CRYSS = float(final_summary["Rmax"])
    RelSolVol_Rmax_CRYSS = float(final_summary["RelSolVol_at_Rmax"])
    mixture_case["Rmax_CRYSS"] = Rmax_CRYSS
    mixture_case["RelSolVol_Rmax_CRYSS"] = RelSolVol_Rmax_CRYSS

    mode_opt = list(final_summary.get("mode", modes))
    #print(
    #    f"[MC fit] mixture_id={mixture_case.get('mixture_id', mixture_case.get('case_id'))}: "
    #    f"objective={final_obj:.6f}, Tm={Tm_opt}, DH={DH_opt}, mode={mode_opt}"
    #)

    return Tm_opt, DH_opt, final_obj, Rmax_CRYSS, RelSolVol_Rmax_CRYSS, mode_opt


def apply_optimised_physchem(mixture_case, optimised_Tm, optimised_DH):
    """Write optimised Tm/DH back into the original mixture dict and return it."""
    if not isinstance(mixture_case, dict):
        raise TypeError("mixture_case must be a dictionary-like mixture payload.")

    if len(optimised_Tm) != len(optimised_DH):
        raise ValueError(
            "optimised_Tm and optimised_DH must have the same length: "
            f"{len(optimised_Tm)} != {len(optimised_DH)}"
        )

    if "components" in mixture_case and isinstance(mixture_case["components"], list):
        if len(mixture_case["components"]) != len(optimised_Tm):
            raise ValueError(
                "Component count mismatch: "
                f"components={len(mixture_case['components'])}, optimised_Tm={len(optimised_Tm)}"
            )

        for component, Tm_val, DH_val in zip(mixture_case["components"], optimised_Tm, optimised_DH):
            if isinstance(component, dict):
                component["Tm"] = float(Tm_val)
                component["DH"] = float(DH_val)

        mixture_case["Tm"] = [float(v) for v in optimised_Tm]
        mixture_case["DH"] = [float(v) for v in optimised_DH]
        return mixture_case

    if "Tm" in mixture_case and isinstance(mixture_case.get("Tm"), list):
        mixture_case["Tm"] = [float(v) for v in optimised_Tm]
    else:
        mixture_case["Tm"] = [float(v) for v in optimised_Tm]

    if "DH" in mixture_case and isinstance(mixture_case.get("DH"), list):
        mixture_case["DH"] = [float(v) for v in optimised_DH]
    else:
        mixture_case["DH"] = [float(v) for v in optimised_DH]

    return mixture_case


def optimise_all(mixtures, Xs_obs_list):
    """Optimise each mixture in order and return a list of per-mixture result dicts."""
    if len(mixtures) != len(Xs_obs_list):
        raise ValueError(
            f"mixtures and Xs_obs_list length mismatch: {len(mixtures)} != {len(Xs_obs_list)}"
        )

    results = []
    for idx, mixture_case in enumerate(mixtures):
        mixture_id = mixture_case.get("mixture_id", mixture_case.get("case_id", f"mixture_{idx}"))
        Tm_opt, DH_opt, final_obj, Rmax_CRYSS, relsolvol_CRYSS, mode_opt = optimise_mixture(mixture_case, Xs_obs_list[idx])
        result = {
            "mixture_id": mixture_id,
            "Rmax_NODES": mixture_case.get("Rmax_NODES"),
            "RelSolVol_Rmax_NODES": mixture_case.get("RelSolVol_Rmax_NODES"),
            "Rmax_CRYSS": Rmax_CRYSS,
            "RelSolVol_Rmax_CRYSS": relsolvol_CRYSS,
            "objective": final_obj,
            "final_objective": final_obj,
            "Tm_opt": Tm_opt,
            "DH_opt": DH_opt,
            "mode_opt": mode_opt,
            "optimised_Tm": Tm_opt,
            "optimised_DH": DH_opt,
            "mode": mode_opt,
        }
        results.append(result)

    return results


def optimise_all_parsed_mixtures(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Optimise every unique parsed mixture using its own slice-level observations and R values."""
    cases = build_cases_from_parsed_mixtures(parsed)
    xs_obs_list = [case.get("slice_Xs", []) for case in cases]
    if any(not xs for xs in xs_obs_list):
        raise ValueError("One or more parsed mixtures are missing slice_Xs values.")

    # Keep the optimizer aligned to the actual parsed R values and Xs coordinates.
    for case in cases:
        case["slice_R"] = list(case.get("slice_R", []))
        if not case["slice_R"]:
            raise ValueError(f"Missing slice R values for mixture {case.get('mixture_id')}")
        case["Xs_obs"] = [list(xs) for xs in case.get("slice_Xs", [])]

    return optimise_all(cases, xs_obs_list)


def print_result_summary(results):
    """Print a compact per-mixture summary table before plotting."""
    if not results:
        return

    print("\nPer-mixture NODES vs CRYSS summary:")
    print("-" * 90)
    print(f"{'mixture_id':<18} {'Rmax_NODES':>12} {'Rmax_CRYSS':>12} {'objective':>12}")
    print("-" * 90)
    for item in results:
        mixture_id = str(item.get("mixture_id", ""))
        rmax_nodes = item.get("Rmax_NODES")
        rmax_crys = item.get("Rmax_CRYSS")
        objective = item.get("final_objective")
        print(f"{mixture_id:<18} {rmax_nodes if rmax_nodes is not None else 'n/a':>12} {rmax_crys if rmax_crys is not None else 'n/a':>12} {objective if objective is not None else 'n/a':>12}")
    print("-" * 90)


def _identity_r2(x_values: List[float], y_values: List[float]) -> float:
    """Return the R² score for y versus the identity line y = x.

    This is the usual fit metric for checking how closely a predicted series tracks
    the reference series when the same quantity is being compared on both axes.
    """
    if len(x_values) != len(y_values) or len(x_values) == 0:
        return 1.0

    if len(x_values) == 1:
        return 1.0 if abs(float(y_values[0]) - float(x_values[0])) < 1e-12 else 0.0

    y_mean = sum(float(y) for y in y_values) / len(y_values)
    ss_res = sum((float(y) - float(x)) ** 2 for x, y in zip(x_values, y_values))
    ss_tot = sum((float(y) - y_mean) ** 2 for y in y_values)

    if abs(ss_tot) < 1e-14:
        return 1.0 if ss_res < 1e-12 else 0.0

    return 1.0 - ss_res / ss_tot


def plot_results(results, *, candidate_runs: List[Dict[str, Any]] | None = None, show: bool = True):
    """Scatter plot comparing the best MC and best optimized candidates.

    If a NODES reference is present, the plot uses X = Rmax(NODES) and Y = best MC / best
    optimized Rmax(CRYSS). If it is not present, the plot falls back to the best MC-vs-
    best optimized comparison and reports the corresponding R² against y = x.
    """
    if not show:
        return None, None

    if not results and not candidate_runs:
        raise ValueError("results or candidate_runs must contain at least one item.")

    plotted = [
        item for item in results
        if item.get("Rmax_NODES") is not None and item.get("Rmax_CRYSS") is not None
    ] if results else []

    fig, ax = plt.subplots()
    best_mc_points = []
    best_opt_points = []
    ref_points = []

    if candidate_runs:
        for profile in candidate_runs:
            best_mc = profile.get("best_mc_candidate")
            best_opt = profile.get("best_optimized_candidate")
            if best_mc is None or best_opt is None:
                continue

            n_nodes = profile.get("Rmax_NODES")
            mc_rmax = float(best_mc["Rmax"])
            opt_rmax = float(best_opt["Rmax"])

            if n_nodes is not None:
                ref_points.append(100.0 * float(n_nodes))
                best_mc_points.append(100.0 * mc_rmax)
                best_opt_points.append(100.0 * opt_rmax)
            else:
                best_mc_points.append(100.0 * mc_rmax)
                best_opt_points.append(100.0 * opt_rmax)

    if ref_points:
        mc_x = ref_points
        mc_y = best_mc_points
        opt_x = ref_points
        opt_y = best_opt_points
        r2_mc = _identity_r2(mc_x, mc_y)
        r2_opt = _identity_r2(opt_x, opt_y)

        ax.scatter(mc_x, mc_y, color="tab:blue", s=18, alpha=0.8, label=f"MC best (R²={r2_mc:.3f})")
        ax.scatter(opt_x, opt_y, color="tab:orange", s=18, alpha=0.8, label=f"MC+opt best (R²={r2_opt:.3f})")
        ax.set_xlabel("Rmax (NODES, %)")
        ax.set_ylabel("Rmax (CRYSS, %)")
        ax.set_title("Best MC vs best MC+optimizer, lowest objective")
    else:
        if best_mc_points:
            x_values = best_mc_points
            y_values = best_opt_points
            r2_pair = _identity_r2(x_values, y_values)
            ax.scatter(x_values, y_values, color="tab:orange", s=18, alpha=0.8, label=f"MC+opt best (R²={r2_pair:.3f})")
            ax.scatter(x_values, x_values, color="tab:blue", s=18, alpha=0.7, label="MC best")
            ax.set_xlabel("Best MC Rmax (%)")
            ax.set_ylabel("Best MC+optimizer Rmax (%)")
            ax.set_title("Best MC vs best MC+optimizer, lowest objective")
        else:
            if not plotted:
                raise ValueError("No valid plotted data were found for the comparison figure.")
            x_plot_values = [100.0 * float(item["Rmax_NODES"]) for item in plotted]
            y_plot_values = [100.0 * float(item["Rmax_CRYSS"]) for item in plotted]
            labels = [str(item.get("mixture_id", "")) for item in plotted]
            ax.scatter(x_plot_values, y_plot_values, color="tab:purple", s=60, label="Best optimized run")
            for label, x_value, y_value in zip(labels, x_plot_values, y_plot_values):
                ax.annotate(label, (x_value, y_value))
            ax.set_xlabel("Rmax (NODES, %)")
            ax.set_ylabel("Rmax (CRYSS, %)")
            ax.set_title("NODES vs CRYSS Rmax comparison")

    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1.0, label="y=x")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


from .objective_function import compute_objective_with_slices
from optimisers.monte_carlo import monte_carlo_fit


def main() -> None:
    if len(sys.argv) <= 1:
        print("Root harness is optimizer-only and non-interactive.")
        print("Run the Excel picker first: python input_parser/parser_harness.py")
        print("Or pass an explicit Excel path: python harness.py path/to/file.xlsx")
        return

    arg = sys.argv[1].lower()
    if arg in {"--prompt", "-p", "--default", "-d"}:
        print("Root harness is optimizer-only and non-interactive.")
        print("Run the Excel picker first: python input_parser/parser_harness.py")
        print("Or pass an explicit Excel path: python harness.py path/to/file.xlsx")
        return

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"File not found: {target}")
        return

    if target.suffix.lower() not in {".xlsx", ".xls"}:
        print(f"Excel-only mode is enabled; '{target}' is not an Excel workbook.")
        return

    PARAM_BLOCK["case_json_path"] = str(target)
    parsed = parse_nodes_excel(Path(target), write_excel=False)
    cases = build_cases_from_parsed_mixtures(parsed)

    results = []
    candidate_runs = []
    for case in cases:
        Xs_obs = [list(xs) for xs in case.get("slice_Xs", [])]
        if not Xs_obs:
            raise ValueError(f"No parsed Xs observations found for mixture {case.get('mixture_id')}")

        summary = run_case_summary(
            case,
            solver=PARAM_BLOCK["solver"],
            slice_R=list(case.get("slice_R", [])) if case.get("slice_R") else list(PARAM_BLOCK["slice_R"]),
        )

        print_core_summary(summary)
        _print_objective_report(summary, Xs_obs)
        report_component_errors(summary, Xs_obs)
        report_component_slice_deltas(summary, Xs_obs)

        Tm_opt, DH_opt, final_obj, Rmax_CRYSS, relsolvol_CRYSS = optimise_mixture(case, Xs_obs)
        result = {
            "mixture_id": case.get("mixture_id"),
            "Rmax_NODES": case.get("Rmax_NODES"),
            "Rmax_CRYSS": Rmax_CRYSS,
            "final_objective": final_obj,
            "optimised_Tm": Tm_opt,
            "optimised_DH": DH_opt,
        }
        results.append(result)

        run_profiles = analyse_mc_and_optimizer_runs(case, Xs_obs, n_runs=1, n_samples=200, top_k=10, seed=42)
        if run_profiles:
            candidate_runs.append(run_profiles[0])

    if results:
        plot_results(results, candidate_runs=candidate_runs)






if __name__ == "__main__":
    main()

