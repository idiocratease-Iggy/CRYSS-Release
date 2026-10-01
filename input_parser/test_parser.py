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

    parse_nodes_excel(path, quiet=False)
    out = capsys.readouterr().out
    assert "[QC] replicate rejection rate =" in out


def test_qc_event_prints_and_can_be_disabled(capsys):
    old_flag = msd.QC_PRINT
    msd.QC_PRINT = True
    msd.qc_event("mixture mix_001 accepted: 1 solvent vols, R-range 0.100-0.200")
    out = capsys.readouterr().out
    assert "[QC] mixture mix_001 accepted: 1 solvent vols, R-range 0.100-0.200" in out

    msd.QC_PRINT = False
    msd.qc_event("mixture mix_001 rejected: invalid")
    assert capsys.readouterr().out == ""
    msd.QC_PRINT = old_flag


def _write_standard_crys_workbook(path: Path, *, pure_component: bool = False) -> None:
    row = {
        "id": "row_1",
        "Mixture_id": "mix_001",
        "xs_1": 1.0 if pure_component else 0.7,
        "xs_2": 0.0 if pure_component else 0.3,
        "xl_1": 0.0 if pure_component else 0.2,
        "xl_2": 1.0 if pure_component else 0.8,
        "extra_field": "keep_me",
    }
    pd.DataFrame([row]).to_excel(path, index=False)


def test_detect_replicates_in_slice_keeps_different_salt_derivative_species_separate():
    from input_parser.slice_builder import detect_replicates_in_slice

    rows = [
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.80,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "salt_derivative": "NaCl",
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.80,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "salt_derivative": "KCl",
        },
    ]

    replicates = detect_replicates_in_slice(rows)

    assert len(replicates) == 2


def test_detect_replicates_in_slice_ignores_case_for_salt_derivative():
    from input_parser.slice_builder import detect_replicates_in_slice

    rows = [
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.80,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "salt_derivative": "NaCl",
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.80,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "salt_derivative": "nacl",
        },
    ]

    replicates = detect_replicates_in_slice(rows)

    assert len(replicates) == 1


def test_standard_crys_parse_preserves_extra_fields(tmp_path):
    path = tmp_path / "standard_crys.xlsx"
    _write_standard_crys_workbook(path)

    mixtures = parse_nodes_excel(path, quiet=True)

    assert "mix_001" in mixtures
    assert mixtures["mix_001"].slices[0].raw_rows[0]["extra_field"] == "keep_me"


def test_standard_crys_parse_rejects_degenerate_component_slice(tmp_path):
    path = tmp_path / "degenerate_crys.xlsx"
    _write_standard_crys_workbook(path, pure_component=True)

    mixtures = parse_nodes_excel(path, quiet=True)

    assert mixtures == {}


def test_parse_standard_crys_v2_workbook_from_data_folder():
    path = Path("data/Standard_CRYSS_input_file_format_V2.0.xlsx")

    mixtures = parse_nodes_excel(path, quiet=True)

    assert "260902_001" in mixtures
    assert any(row.get("salt_derivative") == "salt former 01" for row in mixtures["260902_001"].slices[0].raw_rows)


def test_build_mixture_sets_keeps_valid_three_slice_set(capsys):
    rows = [
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.95,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.80,
            "xs_2": 0.20,
            "xl_1": 0.10,
            "xl_2": 0.90,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "Xc": [0.70, 0.30],
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 2,
            "R": 0.87,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.85,
            "xs_2": 0.15,
            "xl_1": 0.20,
            "xl_2": 0.80,
            "Xs": [0.85, 0.15],
            "Xl": [0.20, 0.80],
            "Xc": [0.70, 0.30],
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 3,
            "R": 0.81,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.90,
            "xs_2": 0.10,
            "xl_1": 0.30,
            "xl_2": 0.70,
            "Xs": [0.90, 0.10],
            "Xl": [0.30, 0.70],
            "Xc": [0.70, 0.30],
        },
    ]

    accepted = build_mixture_sets(rows, n_components=2)

    assert list(accepted) == ["mix_001"]
    assert len(accepted["mix_001"]) == 3
    assert "Rmax-0% record outside valid band" not in capsys.readouterr().out


def test_detect_replicates_does_not_label_normal_low_slice_as_rmax_zero_percent(capsys):
    rows = [
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.86,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.80,
            "xs_2": 0.20,
            "xl_1": 0.10,
            "xl_2": 0.90,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "Xc": [0.70, 0.30],
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 2,
            "R": 0.77,
            "xc_1": 0.72,
            "xc_2": 0.28,
            "xs_1": 0.82,
            "xs_2": 0.18,
            "xl_1": 0.08,
            "xl_2": 0.92,
            "Xs": [0.82, 0.18],
            "Xl": [0.08, 0.92],
            "Xc": [0.72, 0.28],
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 3,
            "R": 0.64,
            "xc_1": 0.65,
            "xc_2": 0.35,
            "xs_1": 0.90,
            "xs_2": 0.10,
            "xl_1": 0.20,
            "xl_2": 0.80,
            "Xs": [0.90, 0.10],
            "Xl": [0.20, 0.80],
            "Xc": [0.65, 0.35],
        },
    ]

    import input_parser.mixture_set_detector as msd
    from input_parser.mixture_set_detector import detect_replicates

    old_flag = msd.QC_PRINT
    msd.QC_PRINT = True
    try:
        detect_replicates(rows, canonical_xc=[0.70, 0.30], n_components=2, xc_tol=0.05, r_tol=0.05)
        out = capsys.readouterr().out
    finally:
        msd.QC_PRINT = old_flag

    assert "Rmax-0% record outside valid band" not in out


def test_detect_replicates_reports_rmax_zero_percent_outlier(capsys):
    rows = [
        {
            "mixture_id": "mix_001",
            "slice_number": 1,
            "R": 0.80,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.80,
            "xs_2": 0.20,
            "xl_1": 0.10,
            "xl_2": 0.90,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "Xc": [0.70, 0.30],
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 2,
            "R": 0.78,
            "xc_1": 0.72,
            "xc_2": 0.28,
            "xs_1": 0.82,
            "xs_2": 0.18,
            "xl_1": 0.08,
            "xl_2": 0.92,
            "Xs": [0.82, 0.18],
            "Xl": [0.08, 0.92],
            "Xc": [0.72, 0.28],
        },
        {
            "mixture_id": "mix_001",
            "slice_number": 3,
            "R": 0.60,
            "xc_1": 0.65,
            "xc_2": 0.35,
            "xs_1": 1.00,
            "xs_2": 0.00,
            "xl_1": 0.20,
            "xl_2": 0.80,
            "Xs": [1.00, 0.00],
            "Xl": [0.20, 0.80],
            "Xc": [0.65, 0.35],
        },
    ]

    import input_parser.mixture_set_detector as msd
    from input_parser.mixture_set_detector import detect_replicates

    old_flag = msd.QC_PRINT
    msd.QC_PRINT = True
    try:
        detect_replicates(rows, canonical_xc=[0.70, 0.30], n_components=2, xc_tol=0.05, r_tol=0.05)
        out = capsys.readouterr().out
    finally:
        msd.QC_PRINT = old_flag

    assert "Rmax-0% record outside valid band" in out


def test_build_mixture_sets_reports_replica_count(capsys):
    rows = [
        {
            "mixture_id": "mix_999_rep",
            "slice_number": 1,
            "R": 0.90,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.80,
            "xs_2": 0.20,
            "xl_1": 0.10,
            "xl_2": 0.90,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "Xc": [0.70, 0.30],
            "salt_derivative": "naCl",
        },
        {
            "mixture_id": "mix_999_rep",
            "slice_number": 1,
            "R": 0.92,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.80,
            "xs_2": 0.20,
            "xl_1": 0.10,
            "xl_2": 0.90,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "Xc": [0.70, 0.30],
            "salt_derivative": "nacl",
        },
        {
            "mixture_id": "mix_999_rep",
            "slice_number": 1,
            "R": 0.91,
            "xc_1": 0.70,
            "xc_2": 0.30,
            "xs_1": 0.80,
            "xs_2": 0.20,
            "xl_1": 0.10,
            "xl_2": 0.90,
            "Xs": [0.80, 0.20],
            "Xl": [0.10, 0.90],
            "Xc": [0.70, 0.30],
            "salt_derivative": "NaCl",
        },
    ]

    import input_parser.mixture_set_detector as msd

    old_flag = msd.QC_PRINT
    msd.QC_PRINT = True
    msd._qc_seen.clear()
    try:
        build_mixture_sets(rows, n_components=2)
        out = capsys.readouterr().out
    finally:
        msd.QC_PRINT = old_flag

    assert "mixture mix_999_rep accepted: 1 solvent vol retained for solver, 2 replicate rows" in out


def test_build_mixture_sets_reports_true_slice_rejection_in_qc():
    summary = msd.format_acceptance_summary(
        "mix_002",
        supplied_slice_count=2,
        accepted_slice_count=1,
        replicate_count=0,
        r_min=0.810,
        r_max=0.900,
    )

    assert (
        "mixture mix_002 accepted: 1 solvent vol retained for solver from 2 supplied (1 rejected as noise), "
        "0 replicate rows, accepted R-range 0.810-0.900"
    ) == summary


def test_perturbed_261001_keeps_two_volume_slices_for_replicates():
    mixtures = parse_nodes_excel(Path("data/261001_1244_Vol02_err6pc.xlsx"), quiet=True)

    for mix_id in ("261001_002", "261001_004", "261001_008"):
        mix = mixtures[mix_id]
        assert len(mix.slices) == 2, f"{mix_id} collapsed to {len(mix.slices)} slices"
        assert sum(len(s.raw_rows) for s in mix.slices) == 3, f"{mix_id} should keep 3 rows across 2 slices"


def test_perturbed_261001_keeps_three_distinct_volume_slices_when_volume_metadata_is_distinct():
    mixtures = parse_nodes_excel(Path("data/261001_1244_Vol03_err6pc.xlsx"), quiet=True)

    mix = mixtures["261001_003"]
    assert len(mix.slices) == 3, f"261001_003 collapsed to {len(mix.slices)} slices"
    assert sum(len(s.raw_rows) for s in mix.slices) == 4, "261001_003 should keep 4 rows across 3 slices"
    assert {s.slice_number for s in mix.slices} == {1, 2, 3}


def test_perturbed_261001_keeps_three_distinct_volume_slices_when_volume_metadata_is_missing():
    mixtures = parse_nodes_excel(Path("data/261001_1244_Vol03_err6pc_sans.xlsx"), quiet=True)

    mix = mixtures["261001_003"]
    assert len(mix.slices) == 3, f"261001_003 collapsed to {len(mix.slices)} slices without rel_sol_volume metadata"
    assert sum(len(s.raw_rows) for s in mix.slices) == 4, "261001_003 should keep 4 rows across 3 slices even when metadata is stripped"
    assert {s.slice_number for s in mix.slices} == {1, 2, 3}


def test_parse_nodes_excel_reports_nonzero_replicate_deviation_stats_for_noisy_data():
    parse_nodes_excel(Path("data/261001_1244_Vol02_err6pc.xlsx"), quiet=True)

    assert len(msd.replicate_devs) > 0
    assert max(msd.replicate_devs) > 0.0


def test_parse_standard_crys_rejects_invalid_rmax_band_and_counts_surviving_records(capsys):
    path = Path("data/Standard CRYSS Test Read (one record floppy R).xlsx")

    parse_nodes_excel(path, quiet=False)
    out = capsys.readouterr().out

    assert "mixture 260905_001 accepted: 2 solvent vols retained for solver from 3 supplied (1 rejected as noise), 0 replicate rows, accepted R-range 0.780-0.908" in out
