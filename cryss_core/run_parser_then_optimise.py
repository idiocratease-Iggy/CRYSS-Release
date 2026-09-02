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

import sys
import threading
import time
import tkinter as tk
from pathlib import Path

from input_parser import parser_harness
from input_parser.parser_harness import run_parser
from .harness import (
    analyse_mc_and_optimizer_runs,
    build_cases_from_parsed_mixtures,
    export_crys_rmax_results,
    optimise_all_parsed_mixtures,
    plot_results,
)


def _find_status_image() -> Path | None:
    """Look for the CRYSS logo in source and bundled app layouts."""
    root = Path(__file__).resolve().parent
    bundle_root = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", None) else None
    repo_root = root.parent

    explicit = [
        root / "cryss.png",
        root / "cryss.ico",
        root / "cryss_fitting_routine_icon.png",
        root / "data" / "cryss.png",
        root / "docs" / "cryss.png",
        repo_root / "cryss.png",
        repo_root / "data" / "cryss.png",
        repo_root / "docs" / "cryss.png",
        bundle_root / "cryss.png" if bundle_root else None,
        bundle_root / "cryss_core" / "cryss.png" if bundle_root else None,
        bundle_root / "data" / "cryss.png" if bundle_root else None,
        bundle_root / "docs" / "cryss.png" if bundle_root else None,
        Path(sys.executable).resolve().parent / "cryss.png",
    ]
    for path in explicit:
        if path and path.exists():
            return path

    roots = [root, repo_root, root / "data", root / "docs", repo_root / "data", repo_root / "docs"]
    if bundle_root is not None:
        roots.extend([bundle_root, bundle_root / "cryss_core", bundle_root / "data", bundle_root / "docs"])

    candidates = []
    for folder in roots:
        if not folder.exists():
            continue
        for pattern in ("**/*.png", "**/*.ico", "**/*.jpg", "**/*.jpeg", "**/*.gif"):
            candidates.extend(folder.glob(pattern))

    for path in sorted(set(candidates), key=lambda p: str(p).lower()):
        name = path.name.lower()
        if "cryss" in name or "logo" in name:
            return path
    return None


def _show_status_window(message: str = "Running CRYSS Monte Carlo + optimizer…") -> tuple[tk.Tk, tk.Toplevel]:
    """Create the splash window in the main thread only."""
    root = tk.Tk()
    root.withdraw()
    dialog = tk.Toplevel(root)
    dialog.title("CRYSS")
    dialog.configure(bg="#f7f7f7")
    dialog.attributes("-topmost", True)

    image_path = _find_status_image()
    if image_path and image_path.exists():
        try:
            if image_path.suffix.lower() == ".ico":
                dialog.iconbitmap(str(image_path))
            photo = tk.PhotoImage(file=str(image_path))
            image_label = tk.Label(dialog, image=photo, bg="#f7f7f7")
            image_label.image = photo
            image_label.pack(padx=22, pady=(18, 8))
            dialog.iconphoto(False, photo)
        except Exception:
            pass

    badge = tk.Label(dialog, text="CRYSS", font=("Segoe UI", 16, "bold"), fg="#2c3e50", bg="#f7f7f7")
    badge.pack(pady=(12, 0))

    text = tk.Label(dialog, text=message, font=("Segoe UI", 11), fg="#1f1f1f", bg="#f7f7f7", justify="center")
    text.pack(padx=28, pady=(0, 16))
    dialog.update_idletasks()

    width = dialog.winfo_reqwidth() + 24
    height = dialog.winfo_reqheight() + 12
    dialog.geometry(f"{width}x{height}")
    dialog.resizable(False, False)
    return root, dialog


def run_with_status_window(task, *, message: str = "Running CRYSS Monte Carlo + optimizer…"):
    """Display a splash while heavy work runs, but never from a background Tk thread."""
    root, dialog = _show_status_window(message)
    result = {}
    error = {}

    def worker():
        try:
            result["value"] = task()
        except Exception as exc:  # pragma: no cover - UI-only error path
            error["value"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while thread.is_alive():
        root.update()
        time.sleep(0.03)

    dialog.destroy()
    root.destroy()

    if "value" in error:
        raise error["value"]
    return result.get("value")


def _run_parser_then_optimise_logic(
    parsed=None,
    *,
    show: bool | None = None,
):
    if parsed is None:
        raise ValueError("parsed mixture data must be supplied before the worker thread starts.")

    if not parsed:
        raise ValueError("No parsed workbook data was returned by the parser harness.")

    results = optimise_all_parsed_mixtures(parsed)
    cases = build_cases_from_parsed_mixtures(parsed)
    has_nodes_reference = any(case.get("Rmax_NODES") is not None for case in cases)
    if show is None:
        show = has_nodes_reference

    candidate_runs = []
    for case in cases:
        xs_obs = case.get("slice_Xs", [])
        if not xs_obs:
            continue
        run_profiles = analyse_mc_and_optimizer_runs(case, xs_obs, n_runs=1, n_samples=200, top_k=10, seed=42)
        if run_profiles:
            candidate_runs.append(run_profiles[0])

    return {
        "results": results,
        "candidate_runs": candidate_runs,
        "show_plot": bool(show),
        "has_nodes_reference": has_nodes_reference,
    }


def run_parser_then_optimise(
    excel_path: str | Path | None = None,
    *,
    show: bool | None = None,
    export_results: bool = False,
    output_dir: str | Path | None = None,
    prompt_for_dir: bool = False,
):
    """Run the Excel parser and optimizer with the requested UX ordering.

    Sequence is: file select -> splash while fitting -> hide splash -> show comparison plot if relevant
    -> then ask for export directory if requested or if a non-NODES workbook requires output.
    """
    if excel_path is None:
        parsed = run_parser()
        input_file = getattr(parser_harness, "LAST_INPUT_FILE", None)
    else:
        parsed = run_parser(Path(excel_path))
        input_file = Path(excel_path)

    if not parsed:
        raise ValueError("No parsed workbook data was returned by the parser harness.")

    payload = run_with_status_window(
        lambda: _run_parser_then_optimise_logic(parsed, show=show),
        message="CRYSS is running\nMonte Carlo + optimizer…",
    )

    results = payload["results"]
    candidate_runs = payload["candidate_runs"]
    show_plot = payload["show_plot"]
    has_nodes_reference = payload["has_nodes_reference"]

    if show_plot:
        plot_results(results, candidate_runs=candidate_runs)

    should_export = export_results or not has_nodes_reference
    if should_export:
        if output_dir is not None:
            export_crys_rmax_results(results, output_dir=output_dir, prompt_for_dir=False, input_file=input_file)
        elif prompt_for_dir:
            export_crys_rmax_results(results, prompt_for_dir=True, input_file=input_file)
        else:
            export_crys_rmax_results(
                results,
                output_dir=Path(__file__).resolve().parent / "data" / "tmp",
                input_file=input_file,
            )

    return results


if __name__ == "__main__":
    run_parser_then_optimise(show=None, export_results=True, prompt_for_dir=True)
