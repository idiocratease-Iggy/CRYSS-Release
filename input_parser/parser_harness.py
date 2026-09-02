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
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from input_parser import mixture_set_detector as msd
from input_parser.parser import parse_nodes_excel

LAST_INPUT_FILE: Path | None = None


def _prompt_for_path(prompt: str, initial_dir: Path | None = None, filetypes: tuple[tuple[str, str], ...] | None = None) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=prompt,
        initialdir=str(initial_dir) if initial_dir else str(Path.cwd()),
        filetypes=list(filetypes) if filetypes else None,
    )
    root.destroy()
    return Path(path) if path else None


def _prompt_for_output_path(initial_dir: Path | None = None) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title="Choose output Excel location",
        initialdir=str(initial_dir) if initial_dir else str(Path.cwd()),
        defaultextension=".xlsx",
        filetypes=[("Excel Workbook", "*.xlsx")],
    )
    root.destroy()
    return Path(path) if path else None


def run_parser(input_file: Path | None = None, output_file: Path | None = None):
    """
    Parse a standalone NODES Excel file without emitting the diagnostic workbook.
    The parser is kept in-process for the cryss-core flow.

    Returns the raw parsed dictionary of MixtureSet objects so downstream
    optimizer code can consume the exact parser-generated mixture/slice metadata.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if input_file is None:
        input_file = _prompt_for_path("Select a NODES workbook to parse", data_dir, (("Excel files", "*.xlsx"), ("All files", "*.*")))
        if input_file is None:
            print("No input file selected; exiting.")
            return {}

    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    global LAST_INPUT_FILE
    LAST_INPUT_FILE = input_file

    # The diagnostic Excel export is intentionally disabled in the cryss-core flow.
    mixtures = parse_nodes_excel(input_file, write_excel=False, output_path=output_file, quiet=True)
    msd.QC_PRINT = False
    return mixtures


if __name__ == "__main__":
    run_parser()
