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


"""Per-mixture consistency check for the optional Salt/Derivative label.

Rule (applied independently to each Mixture_ID):
  * all rows blank            -> valid (no salt/derivative information supplied)
  * all rows labelled         -> valid
  * labelled and blank mixed  -> blank rows are rejected, labelled rows are kept
Whatever string is supplied is used as-is (case/whitespace-insensitive); there is no
placeholder interpretation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

MISSING_SALT_REASON = "missing salt/derivative string"
_TAG_KEYS = ("salt_derivative", "Salt/Derivative", "salt", "derivative")


@dataclass
class SaltConsistencyReport:
    total_records: int = 0
    # (record number in file order, mixture id, row id)
    rejected: list[tuple[int, str, str | None]] = field(default_factory=list)
    # mixture id -> salt labels present on retained rows (only for mixtures with a problem or several labels)
    labels_by_mixture: dict[str, list[str]] = field(default_factory=dict)
    multi_label_mixtures: dict[str, list[str]] = field(default_factory=dict)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def rejected_by_mixture(self) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for record_no, mid, _ in self.rejected:
            grouped[mid].append(record_no)
        return dict(grouped)


def _row_label(row: dict) -> str | None:
    for key in _TAG_KEYS:
        value = row.get(key)
        if value is None:
            continue
        try:
            if value != value:  # NaN
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text:
            return text.casefold()
    return None


def reject_inconsistent_salt_rows(rows: list[dict]) -> tuple[list[dict], SaltConsistencyReport]:
    """Return (kept_rows, report); input order is preserved."""
    report = SaltConsistencyReport(total_records=len(rows))
    labels: dict[str, set[str]] = defaultdict(set)
    blanks: dict[str, int] = defaultdict(int)
    row_labels: list[str | None] = []
    mids: list[str] = []

    for row in rows:
        mid = str(row.get("mixture_id") or "UNKNOWN")
        label = _row_label(row)
        mids.append(mid)
        row_labels.append(label)
        if label is None:
            blanks[mid] += 1
        else:
            labels[mid].add(label)

    inconsistent = {mid for mid in labels if blanks.get(mid, 0) > 0}
    kept: list[dict] = []
    for idx, (row, mid, label) in enumerate(zip(rows, mids, row_labels), start=1):
        if label is None and mid in inconsistent:
            report.rejected.append((idx, mid, None if row.get("id") is None else str(row.get("id"))))
            continue
        kept.append(row)

    for mid in inconsistent:
        report.labels_by_mixture[mid] = sorted(labels[mid])
    for mid, found in labels.items():
        if len(found) > 1:
            report.multi_label_mixtures[mid] = sorted(found)
    return kept, report


def summary_lines(report: SaltConsistencyReport | None) -> list[str]:
    """Compact warning block for the widget and the .txt summary (empty when consistent)."""
    if report is None or report.rejected_count == 0:
        return []
    mids = ", ".join(sorted(report.rejected_by_mixture()))
    return [
        "WARNING: input file labelling is inconsistent",
        f"  {report.rejected_count} record(s) rejected - {MISSING_SALT_REASON}",
        f"  affected mixture(s): {mids}",
    ]


def detail_lines(report: SaltConsistencyReport | None) -> list[str]:
    """Detailed block for the .txt export (empty when nothing to report)."""
    if report is None or (report.rejected_count == 0 and not report.multi_label_mixtures):
        return []
    lines = ["Salt/Derivative Consistency Check", "=================================="]
    if report.rejected_count:
        lines.append(
            f"{report.rejected_count} of {report.total_records} records rejected: {MISSING_SALT_REASON} "
            "(other records for the same Mixture_ID carry a label; labelled records were kept)."
        )
        for mid, records in sorted(report.rejected_by_mixture().items()):
            labels = ", ".join(report.labels_by_mixture.get(mid, []))
            recs = ", ".join(str(r) for r in records)
            lines.append(f"  mixture {mid}: record(s) {recs} (file order, data rows only); labelled as: {labels}")
    for mid, found in sorted(report.multi_label_mixtures.items()):
        lines.append(
            f"NOTE: mixture {mid} carries {len(found)} distinct salt/derivative labels "
            f"({', '.join(found)}); treated as separate experiments."
        )
    lines.append("")
    return lines
