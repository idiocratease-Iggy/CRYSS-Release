# CRYSS Release v1.0

Core CRYSS Pipeline for generating, parsing, evaluating, and optimizing mixture compositions.

## Scientific Scope and Industrial Positioning

CRYSS is a **thermodynamically grounded, mechanistic scientific software engine** provided in a **lightweight, non-GUI Python implementation**.

The current release executes a focused, production-relevant workflow:

1. Ingest structured experimental/process inputs (Excel-based),
2. Run the CRYSS Core Engine,
3. Return ranked and classified candidates based on **predicted maximum crystallization yield under optimization constraints**.

This architecture is intentionally suited for **API-first deployment** and integration into broader digital and physical development environments.  
For organizations operating in hardware, process engineering, formulation, or automated lab ecosystems, CRYSS can function as an integration-ready decision core for:

- automated execution pipelines,
- rapid scenario screening,
- closed-loop process analysis,
- and downstream knowledge generation.

In practical terms, CRYSS is designed to accelerate progression from:

**data → information → actionable knowledge**

## Repository

- **Owner:** `idiocratease-Iggy`
- **Repo:** `CRYSS-Release`
- **Description:** `Core CRYSS Pipeline`

## What this project contains

This release bundles the CRYSS app suite code and assets for:

- Input parsing from Excel-based batches
- Mixture generation and objective-function scoring
- Eutectic/gradient engine calculations
- Multi-slice evaluation flows
- Monte Carlo optimization experiments
- Report and plot outputs for analysis

## Top-level structure (tailored)

```text
CRYSS-Release/
├── main.py
├── mixture_input.py
├── qc_settings.py
├── CRYSS.spec
├── LICENSE
├── .gitignore
├── cryss_core/
├── engine/
├── input_parser/
├── multxeu_core/
├── multx_slicing_engine/
├── optimisers/
├── solvers/
├── data/
│   └── tmp/
├── docs/
└── tree.txt
```

## Module map

### `main.py`
Primary application entry point for CRYSS v1.0 runs.

### `cryss_core/`
Core orchestration and objective/report logic:
- `harness.py`
- `mixture_generator.py`
- `objective_function.py`
- `report_component_errors.py`
- `run_parser_then_optimise.py`
- `input_writer.py`

### `engine/`
Numerical and eutectic path engine components:
- `cascade.py`
- `eutectic_path.py`
- `gradient.py`
- `solvent_gradient.py`
- `xeu.py`

### `input_parser/`
Excel/parser pipeline and normalization:
- `loader.py`
- `parser.py`
- `mixture_set_detector.py`
- `slice_builder.py`
- `r_recalculator.py`
- `xc_normalizer.py`
- `statistics.py` (plus legacy/duplicate `statisitcs.py`)
- `excel_writer.py`
- `settings.py`
- `parser_harness.py`
- `test_parser.py`

### `multxeu_core/`
Alternative packaged core including:
- `mixture_input.py`
- nested `engine/`, `multx_slicing_engine/`, `solvers/`

### `multx_slicing_engine/`
Canonical slicing/recovery pipeline:
- `canonical_eutectic.py`
- `canonical_gradients.py`
- `canonical_types.py`
- `evaluate_recovery.py`
- `gradient_engine.py`
- `matrix_layer.py`
- `mixture_loader.py`
- `slice_engine.py`
- `slice_generator.py`

### `optimisers/`
- `monte_carlo.py`

### `solvers/`
- `halley.py`
- `vba.py`

### `docs/`
Pipeline and algorithm documentation PDFs.

### `data/`
Input/output workbooks, plots, JSON configs, and temporary run artifacts.

---

## Requirements

- Python 3.10+ recommended
- pip
- Windows/Linux/macOS (Windows paths are already in active use)

## Setup

```bash
git clone https://github.com/idiocratease-Iggy/CRYSS-Release.git
cd CRYSS-Release
python -m venv .venv
```

### Activate virtual environment

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (CMD)**
```bat
.venv\Scripts\activate.bat
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

## Install dependencies

If `requirements.txt` is present:
```bash
pip install -r requirements.txt
```

If not, install required packages manually for your environment.

## Run

Primary run command:

```bash
python main.py
```

## Data and output notes

- `data/` currently includes both source inputs and generated outputs.
- `data/tmp/` and `*/tmp/` are intermediate artifacts and are typically not suitable for long-term tracking.
- Consider storing only baseline inputs + selected final outputs in git; archive bulk run artifacts externally if repository size grows.

## Known cleanup opportunities (non-breaking)

- `cryss_core/ojective_finction.py` appears to be a typo/duplicate name of `objective_function.py`.
- `input_parser/statisitcs.py` appears to be a typo/duplicate of `statistics.py`.

(Keep as-is for reproducibility in v1.0 unless you explicitly refactor.)

## Versioning

Suggested annotated tag for this release:

```bash
git tag -a v1.0.0 -m "CRYSS Release v1.0.0"
git push origin v1.0.0
```

## Citation

See [`citation.md`](./citation.md) for the recommended citation format.

## License

See [`LICENSE`](./LICENSE).

## Maintainer

GitHub: [@idiocratease-Iggy](https://github.com/idiocratease-Iggy)
