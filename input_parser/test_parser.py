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

from pathlib import Path

import pandas as pd

from input_parser import mixture_set_detector as msd
from input_parser.loader import load_slice_records
from input_parser.mixture_set_detector import build_mixture_sets
from input_parser.parser import parse_nodes_excel


def _write_test_workbook(path: Path) -> None:
    columns = [
        ("Starting Composition", "Component 1"),
        ("Starting Composition", "Component 2"),
        ("Starting Composition", "Component 3"),
        ("Solid Composition", "Component 1"),
        ("Solid Composition", "Component 2"),
        ("Solid Composition", "Component 3"),
        ("Liquor Composition", "Component 1"),
        ("Liquor Composition", "Component 2"),
        ("Liquor Composition", "Component 3"),
        ("", "mixture_id"),
        ("", "slice_number"),
    ]
    df = pd.DataFrame(
        [[0.7, 0.2, 0.1, 0.8, 0.1, 0.1, 0.5, 0.2, 0.3, "mix_001", 1]],
        columns=pd.MultiIndex.from_tuples(columns),
    )

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="slice_records", index=True)


def test_load_slice_records_extracts_component_fields(tmp_path):
    path = tmp_path / "sample.xlsx"
    _write_test_workbook(path)

    rows = load_slice_records(path)

    assert len(rows) == 1
    row = rows[0]
    assert row["mixture_id"] == "mix_001"
    assert row["slice_number"] == 1
    assert row["xc_1"] == 0.7
    assert row["xs_2"] == 0.1
    assert row["xl_3"] == 0.3


def _write_test_workbook_without_slice_number(path: Path) -> None:
    columns = [
        ("Starting Composition", "Component 1"),
        ("Starting Composition", "Component 2"),
        ("Starting Composition", "Component 3"),
        ("Solid Composition", "Component 1"),
        ("Solid Composition", "Component 2"),
        ("Solid Composition", "Component 3"),
        ("Liquor Composition", "Component 1"),
        ("Liquor Composition", "Component 2"),
        ("Liquor Composition", "Component 3"),
        ("", "mixture_id"),
    ]
    df = pd.DataFrame(
        [[0.7, 0.2, 0.1, 0.8, 0.1, 0.1, 0.5, 0.2, 0.3, "mix_001"]],
        columns=pd.MultiIndex.from_tuples(columns),
    )

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="slice_records", index=True)


def test_build_mixture_sets_assigns_missing_slice_number(tmp_path):
    path = tmp_path / "sample_no_slice.xlsx"
    _write_test_workbook_without_slice_number(path)

    rows = load_slice_records(path)
    mixtures = build_mixture_sets(rows, n_components=3)

    assert list(mixtures) == ["mix_001"]
    assert all(row.get("slice_number") == 1 for row in mixtures["mix_001"])


def test_parse_nodes_excel_returns_mixture_set(tmp_path):
    path = tmp_path / "sample.xlsx"
    _write_test_workbook(path)

    mixtures = parse_nodes_excel(path)

    assert list(mixtures) == ["mix_001"]
    assert len(mixtures["mix_001"].slices) == 1
    assert mixtures["mix_001"].slices[0].slice_number == 1


def test_load_slice_records_ignores_file_r_values(tmp_path):
    path = tmp_path / "sample_recovery.xlsx"
    columns = [
        ("Starting Composition", "Component 1"),
        ("Starting Composition", "Component 2"),
        ("Solid Composition", "Component 1"),
        ("Solid Composition", "Component 2"),
        ("Liquor Composition", "Component 1"),
        ("Liquor Composition", "Component 2"),
        ("", "mixture_id"),
        ("", "recovery"),
    ]
    df = pd.DataFrame(
        [[0.7, 0.3, 0.8, 0.1, 0.5, 0.2, "mix_001", 0.75]],
        columns=pd.MultiIndex.from_tuples(columns),
    )

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="slice_records", index=True)

    rows = load_slice_records(path)

    assert len(rows) == 1
    assert rows[0]["R"] is None


def test_build_mixture_sets_uses_computed_r_from_lever_rule(tmp_path):
    path = tmp_path / "sample_missing_r.xlsx"
    columns = [
        ("Starting Composition", "Component 1"),
        ("Starting Composition", "Component 2"),
        ("Starting Composition", "Component 3"),
        ("Solid Composition", "Component 1"),
        ("Solid Composition", "Component 2"),
        ("Solid Composition", "Component 3"),
        ("Liquor Composition", "Component 1"),
        ("Liquor Composition", "Component 2"),
        ("Liquor Composition", "Component 3"),
        ("", "mixture_id"),
    ]
    df = pd.DataFrame(
        [[0.7, 0.2, 0.1, 0.8, 0.1, 0.1, 0.5, 0.2, 0.3, "mix_001"]],
        columns=pd.MultiIndex.from_tuples(columns),
    )

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="slice_records", index=True)

    rows = load_slice_records(path)
    mixtures = build_mixture_sets(rows, n_components=3)

    assert list(mixtures) == ["mix_001"]
    assert abs(mixtures["mix_001"][0]["R"] - 0.5555555556) < 1e-6


def test_parse_nodes_excel_prints_replicate_rejection_summary(capsys, tmp_path):
    path = tmp_path / "sample.xlsx"
    columns = [
        ("Starting Composition", "Component 1"),
        ("Starting Composition", "Component 2"),
        ("Starting Composition", "Component 3"),
        ("Solid Composition", "Component 1"),
        ("Solid Composition", "Component 2"),
        ("Solid Composition", "Component 3"),
        ("Liquor Composition", "Component 1"),
        ("Liquor Composition", "Component 2"),
        ("Liquor Composition", "Component 3"),
        ("", "mixture_id"),
    ]
    df = pd.DataFrame(
        [[0.7, 0.2, 0.1, 0.8, 0.1, 0.1, 0.5, 0.2, 0.3, "mix_001"]],
        columns=pd.MultiIndex.from_tuples(columns),
    )

    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="slice_records", index=True)

    parse_nodes_excel(path)
    out = capsys.readouterr().out
    assert "[QC] replicate rejection rate =" in out


def test_qc_event_prints_and_can_be_disabled(capsys):
    old_flag = msd.QC_PRINT
    msd.QC_PRINT = True
    msd.qc_event("mixture mix_001 accepted: 1 solvent vols, R-range 0.100–0.200")
    out = capsys.readouterr().out
    assert "[QC] mixture mix_001 accepted: 1 solvent vols, R-range 0.100–0.200" in out

    msd.QC_PRINT = False
    msd.qc_event("mixture mix_001 rejected: invalid")
    assert capsys.readouterr().out == ""
    msd.QC_PRINT = old_flag
