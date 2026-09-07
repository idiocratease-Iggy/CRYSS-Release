# CRYSS v1.0 Executable Distribution Notes

This document describes the recommended files to include when distributing the CRYSS v1.0 executable build.

## Built executable

The PyInstaller configuration in `CRYSS.spec` produces the application executable named:

- `CRYSS` / `CRYSS.exe`

## Recommended release package contents

When sharing the packaged executable, include these files alongside it:

- `CRYSS.exe` — the application binary
- `LICENSE` — MIT License text
- `citation.md` — human-readable citation guidance
- `CITATION.cff` — machine-readable citation metadata
- `README.txt` or `Release-Notes.txt` — optional short usage and attribution note

## Suggested release folder structure

```text
CRYSS-v1.0/
├── CRYSS.exe
├── LICENSE
├── citation.md
├── CITATION.cff
└── README.txt
```

## Why include these files?

- `LICENSE` documents redistribution rights under the MIT License.
- `citation.md` gives users a simple citation format for academic and technical work.
- `CITATION.cff` helps GitHub and citation-aware tools recognize the project.
- A short readme or release note helps end users understand what the executable is and how to cite it.

## Notes

- These files are recommended for the **release archive** distributed to users.
- They do not need to be bundled into the Python source code itself unless you want them available in-repo too.
- If you publish the executable separately from GitHub, make sure the license and citation files remain easy to find.
