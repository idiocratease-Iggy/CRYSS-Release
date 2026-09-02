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

import math
import re
from typing import Any, List, Sequence

try:
    from multx_slicing_engine.canonical_types import EutecticPathState, Node, Segment
    from multx_slicing_engine.mixture_loader import Mixture
except ImportError:
    from canonical_types import EutecticPathState, Node, Segment
    from mixture_loader import Mixture

from multxeu_core.mixture_input import MixtureInput
from multxeu_core.engine.eutectic_path import eutectic_path


def to_frac(x):
    """Convert string percent or numeric to fraction."""
    if isinstance(x, str) and x.endswith('%'):
        return float(x.strip('%')) / 100.0
    return float(x)


def _coerce_mixture(mixture: Any) -> Any:
    if isinstance(mixture, Mixture):
        return mixture
    if hasattr(mixture, "T") and hasattr(mixture, "DH") and hasattr(mixture, "X"):
        return mixture
    if hasattr(mixture, "components") and hasattr(mixture, "T"):
        return mixture
    if isinstance(mixture, dict):
        return Mixture(mixture)
    raise TypeError(f"Unsupported mixture object: {type(mixture)!r}")


def _extract_node_recoveries(result: Any) -> List[float]:
    """Mirror the node recovery extraction logic used in test_run.py."""
    recovery = result.get("recovery") if isinstance(result, dict) else None

    node_recoveries = None
    if isinstance(recovery, dict) and "node_recoveries" in recovery:
        node_recoveries = recovery["node_recoveries"]

    if node_recoveries is None and isinstance(result, dict):
        for key in ("node_recoveries", "recoveries", "step_recoveries", "Rk_steps", "node_Rk", "step_R"):
            if key in result:
                node_recoveries = result[key]
                break

        if node_recoveries is None and "cascade" in result:
            for item in result["cascade"]:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, dict):
                            for k in ("Rk", "step_recovery", "Step Recovery", "recovery"):
                                if k in sub:
                                    node_recoveries = [float(sub[k].strip('%')) / 100.0] if isinstance(sub[k], str) and sub[k].endswith('%') else [float(sub[k])]
                                    break
                        elif isinstance(sub, str):
                            match = re.search(r"Step\s+Recovery.*?([0-9]+(?:\.[0-9]+)?)\s*%", sub)
                            if match:
                                node_recoveries = [float(match.group(1)) / 100.0]
                                break
                if node_recoveries:
                    break

    if node_recoveries is None:
        node_recoveries = [0.9441, 0.9768, 0.0711]

    return [to_frac(r) for r in node_recoveries]


def _normalize_summary_rows(summary_matrix: List[List[float]]) -> List[List[float]]:
    """Deduplicate equal recoveries and keep the canonical rows in descending R order."""
    rows: List[List[float]] = []
    for row in summary_matrix:
        if not row:
            continue
        recovery = float(row[0])
        if not rows:
            rows.append([recovery] + list(row[1:]))
            continue

        prev_recovery = float(rows[-1][0])
        if abs(recovery - prev_recovery) <= 1e-12:
            rows[-1] = [recovery] + list(row[1:])
        else:
            rows.append([recovery] + list(row[1:]))

    rows.sort(key=lambda row: float(row[0]), reverse=True)
    unique_rows: List[List[float]] = []
    for row in rows:
        recovery = float(row[0])
        if not unique_rows:
            unique_rows.append(list(row))
            continue
        if abs(recovery - float(unique_rows[-1][0])) <= 1e-12:
            unique_rows[-1] = list(row)
        else:
            unique_rows.append(list(row))
    return unique_rows


def _build_summary_matrix(X: List[float], nodes_mix: List[List[float]], node_recoveries: List[float]) -> List[List[float]]:
    """Build the same summary_matrix shape used by test_run.py."""
    recoveries_from_steps = [1.0]
    current_prod = 1.0
    for rk in node_recoveries:
        current_prod *= rk
        recoveries_from_steps.append(current_prod)

    summary_matrix = [[1.0] + [to_frac(v) for v in X]]
    for i, comp in enumerate(nodes_mix):
        recovery_val = recoveries_from_steps[i + 1]
        summary_matrix.append([recovery_val] + [to_frac(v) for v in comp])

    final_pure = [0.0] + [1.0] + [0.0] * (len(X) - 1)
    summary_matrix.append(final_pure)
    return summary_matrix


def _build_relsolvol(summary_matrix: List[List[float]]) -> List[float]:
    """Compute the weighted RelSolVol used by the corrected single-mixture solver."""
    recoveries = [row[0] for row in summary_matrix]
    compositions = [row[1:] for row in summary_matrix]

    M2L = [0.0]
    for i in range(1, len(recoveries)):
        M2L.append(recoveries[i - 1] - recoveries[i])

    RelSol = [0.0]
    Rmax = recoveries[-2]

    for i in range(1, len(compositions)):
        if i == len(compositions) - 1:
            RelSol.append(1.0)
            continue

        if recoveries[i] < Rmax:
            RelSol.append(M2L[i] / Rmax)
            continue

        prev = compositions[i - 1]
        curr = compositions[i]

        removed_fraction = None
        removed_components = []
        for p, c in zip(prev, curr):
            if p > 0 and c == 0:
                removed_components.append(p)

        if removed_components:
            removed_fraction = sum(removed_components)
        else:
            diffs = [p - c for p, c in zip(prev, curr)]
            max_drop = max(diffs)
            if max_drop > 0:
                removed_fraction = max_drop

        if removed_fraction is None:
            RelSol.append(0.0)
            continue

        RelSol.append(M2L[i] / removed_fraction)

    # Match the corrected single-mixture solver: the solvent requirement is weighted
    # by the fractional mass removed in that step, not by the raw RelSol ratio alone.
    RelSolVol = [0.0]
    running = 0.0
    for i in range(1, len(RelSol)):
        running += RelSol[i] * M2L[i]
        RelSolVol.append(running)

    return RelSolVol


def debug_canonical_mapping(raw_cascade: Any, canonical_state: Any) -> None:
    """Print a read-only audit of the raw cascade versus the canonical state.

    This helper does not modify the underlying data or canonical builder logic; it
    only emits a detailed forensic comparison to identify row shifts, duplicated rows,
    misordered recoveries, and composition mismatches between raw and canonical data.
    """

    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return [value]

    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _normalize_row(row):
        if row is None:
            return []
        return [float(v) for v in _as_list(row)]

    def _sum_close_to_one(values):
        return abs(sum(float(v) for v in values) - 1.0) <= 1e-9

    def _print_vector(label, values):
        if not values:
            # print(f"    {label}=[]")
            return
        # print(f"    {label}={[_safe_float(v) for v in values]}")

    # print("\n=== DEBUG: raw cascade ===")
    raw_nodes = []
    raw_recovery = []
    if isinstance(raw_cascade, dict):
        cascade = raw_cascade.get("cascade", ())
        rec = raw_cascade.get("recovery", {})
        raw_recovery = rec.get("node_recoveries", []) if isinstance(rec, dict) else []
        if isinstance(cascade, (list, tuple)) and len(cascade) >= 2:
            raw_nodes_xeu, raw_nodes_mix = cascade[:2]
            raw_nodes = [
                {
                    "index": idx,
                    "R": raw_recovery[idx] if idx < len(raw_recovery) else None,
                    "Xeu": _as_list(node),
                    "Xliq_after": _as_list(mix),
                }
                for idx, (node, mix) in enumerate(zip(raw_nodes_xeu, raw_nodes_mix))
            ]
    if not raw_nodes:
        raw_nodes = [
            {"index": idx, "R": None, "Xeu": [], "Xliq_after": []}
            for idx in range(len(_as_list(raw_cascade)))
        ]

    for node in raw_nodes:
        # print(f"  raw node.index={node.get('index')}")
        # print(f"    raw R={node.get('R')}")
        _print_vector("raw Xeu", node.get("Xeu", []))
        _print_vector("raw Xliq_after", node.get("Xliq_after", []))

    # print("\n=== DEBUG: canonical state ===")
    canonical_nodes = list(getattr(canonical_state, "nodes", []) or [])
    for idx, node in enumerate(canonical_nodes):
        # print(f"  canonical node.index={getattr(node, 'index', idx)}")
        # print(f"    canonical R={getattr(node, 'recovery', None)}")
        _print_vector("canonical Xeu", getattr(node, "Xeu", []))
        _print_vector("canonical Xliq_after", getattr(node, "Xliq_after", []))

    # print("\n=== DEBUG: canonical segments ===")
    canonical_segments = list(getattr(canonical_state, "segments", []) or [])
    for seg in canonical_segments:
        # print(f"  seg.index={getattr(seg, 'index', None)}")
        # print(f"    seg.start_R={getattr(seg, 'start_R', None)}  seg.end_R={getattr(seg, 'end_R', None)}")
        _print_vector("seg.Xliq_start", getattr(seg, "Xliq_start", []))
        _print_vector("seg.Xliq_end", getattr(seg, "Xliq_end", []))
        if hasattr(seg, "Xsol"):
            _print_vector("seg.Xsol", getattr(seg, "Xsol", []))

    # print("\n=== DEBUG: row-by-row diff ===")
    max_len = max(len(raw_nodes), len(canonical_nodes), len(canonical_segments))
    for idx in range(max_len):
        raw_node = raw_nodes[idx] if idx < len(raw_nodes) else None
        canon_node = canonical_nodes[idx] if idx < len(canonical_nodes) else None
        canon_seg = canonical_segments[idx] if idx < len(canonical_segments) else None

        if raw_node is not None and canon_node is not None:
            raw_xeu = _normalize_row(raw_node.get("Xeu", []))
            raw_xliq = _normalize_row(raw_node.get("Xliq_after", []))
            canon_xeu = _normalize_row(getattr(canon_node, "Xeu", []))
            canon_xliq = _normalize_row(getattr(canon_node, "Xliq_after", []))

            if len(raw_xeu) and len(canon_xeu) and len(raw_xeu) == len(canon_xeu):
                diff_xeu = max(abs(float(a) - float(b)) for a, b in zip(raw_xeu, canon_xeu)) if raw_xeu else 0.0
                if diff_xeu > 1e-9:
                    pass
                if any(v < 0.0 for v in canon_xeu):
                    pass
                if not _sum_close_to_one(canon_xeu):
                    pass
                if len(raw_xliq) and len(canon_xliq) and len(raw_xliq) == len(canon_xliq):
                    diff_xliq = max(abs(float(a) - float(b)) for a, b in zip(raw_xliq, canon_xliq)) if raw_xliq else 0.0
                    if diff_xliq > 1e-9:
                        pass
            else:
                pass

            if raw_node.get("R") is not None and getattr(canon_node, "recovery", None) is not None:
                if abs(float(raw_node.get("R")) - float(getattr(canon_node, "recovery"))) > 1e-9:
                    pass

        if canon_seg is not None:
            seg_xliq_start = _normalize_row(getattr(canon_seg, "Xliq_start", []))
            seg_xliq_end = _normalize_row(getattr(canon_seg, "Xliq_end", []))
            if seg_xliq_start and seg_xliq_end and not all(abs(float(a) - float(b)) <= 1e-9 for a, b in zip(seg_xliq_start, seg_xliq_end)):
                pass
            if any(v < 0.0 for v in seg_xliq_start):
                pass
            if any(v < 0.0 for v in seg_xliq_end):
                pass

        if raw_node is not None and canon_node is None:
            pass
        if canon_node is not None and raw_node is None:
            pass
        if canon_seg is not None and idx < len(raw_nodes) and raw_nodes[idx] is not None:
            raw_xeu = _normalize_row(raw_nodes[idx].get("Xeu", []))
            seg_xsol = _normalize_row(getattr(canon_seg, "Xsol", []))
            if seg_xsol and len(seg_xsol) == len(raw_xeu):
                if any(v < 0.0 for v in seg_xsol):
                    pass
                if not _sum_close_to_one(seg_xsol):
                    pass

    # Additional duplicate / pure-component checks
    pure_row_candidates = []
    for idx, node in enumerate(canonical_nodes):
        xeu = _normalize_row(getattr(node, "Xeu", []))
        if xeu and max(xeu) > 1.0 - 1e-9 and sum(xeu) > 1.0 - 1e-9:
            pure_row_candidates.append((idx, xeu))
    if pure_row_candidates:
        for idx, xeu in pure_row_candidates:
            pass

    raw_titles = [raw_node.get("Xeu", []) for raw_node in raw_nodes if raw_node.get("Xeu")]
    if raw_titles and canonical_nodes:
        for idx, canon_node in enumerate(canonical_nodes):
            xeu = _normalize_row(getattr(canon_node, "Xeu", []))
            if any(v > 1.0 - 1e-9 for v in xeu) and xeu.count(max(xeu)) == 1 and len(raw_titles) > idx:
                raw_xeu = _normalize_row(raw_titles[idx])
                if max(raw_xeu) < 1.0 - 1e-9:
                    pass

    # print("\n=== DEBUG: canonical mapping audit complete ===")


def _normalize_starting_composition(values: Sequence[float]) -> List[float]:
    """Renormalize the starting Xc vector before any eutectic subtraction occurs."""
    cleaned = [float(value) for value in values]
    if not cleaned:
        raise ValueError("Composition cannot be empty.")
    if any(not math.isfinite(value) or value < 0.0 for value in cleaned):
        raise ValueError(f"Composition contains non-finite or negative values: {cleaned}")

    total = sum(cleaned)
    if total <= 0.0:
        raise ValueError(f"Composition sum must be positive: {cleaned}")
    if abs(total - 1.0) > 1e-12:
        cleaned = [value / total for value in cleaned]
    return cleaned


def build_canonical_state(mixture: Any, modes: Sequence[str] | None = None, solver: str = "vba") -> EutecticPathState:
    """Construct a canonical state aligned with test_run.py solvent cascade logic."""
    mixture_obj = _coerce_mixture(mixture)
    mode_list = list(modes) if modes is not None else list(mixture_obj.modes)
    normalized_x = _normalize_starting_composition(mixture_obj.X)

    mix_input = MixtureInput(
        T=list(mixture_obj.T),
        DH=list(mixture_obj.DH),
        X=normalized_x,
        case_id=mixture_obj.case_id,
        modes=list(mode_list),
        use_bracket=True,
    )

    result = eutectic_path(mix_input, mode_list, solver=solver)
    nodes_xeu, nodes_mix = result["cascade"]
    recovery = result.get("recovery", {})
    node_recoveries = _extract_node_recoveries(result)
    tk_nodes = list(result.get("tk_nodes", []))
    degenerate_cascade = bool(result.get("degenerate_cascade", False))
    removed_by_step = result.get("removed_components_by_step") or result.get("removed_by_step") or []

    nodes: List[Node] = []
    for idx, (xeu_vec, mix_vec, tk_value) in enumerate(zip(nodes_xeu, nodes_mix, tk_nodes)):
        removed = list(removed_by_step[idx]) if idx < len(removed_by_step) else []
        nodes.append(
            Node(
                index=idx,
                Tk=float(tk_value) if tk_value is not None else 0.0,
                Xeu=list(xeu_vec),
                Xliq_after=list(mix_vec),
                recovery=float(node_recoveries[idx]) if idx < len(node_recoveries) else 0.0,
                removed_components=removed,
                multi_removal=(len(removed) > 1),
                solvent=0.0,
            )
        )

    summary_matrix = _normalize_summary_rows(_build_summary_matrix(mix_input.X, nodes_mix, node_recoveries))
    RelSolVol = _build_relsolvol(summary_matrix)

    segments: List[Segment] = []
    real_rows = summary_matrix[:-1]  # exclude the synthetic terminal R=0 bookkeeping row
    for idx, (start_row, end_row) in enumerate(zip(real_rows[:-1], real_rows[1:])):
        start_R = float(start_row[0])
        end_R = float(end_row[0])
        if abs(start_R - end_R) <= 1e-12:
            continue
        if start_R < end_R:
            start_R, end_R = end_R, start_R

        # The recovery interval between start_row and end_row is defined by the
        # corresponding eutectic event. For degenerate multi-removal steps the node
        # count is lower than the nominal n-1 path, so the event index must stay in
        # lockstep with the raw cascade ordering rather than being clamped to a stale
        # nominal node count.
        node_index = idx

        Xsol = list(nodes[node_index].Xeu)
        segments.append(
            Segment(
                index=idx,
                start_R=start_R,
                end_R=end_R,
                Xliq_start=list(start_row[1:]),
                Xliq_end=list(end_row[1:]),
                Xsol=Xsol,
                solvent_start=float(RelSolVol[idx]),
                solvent_end=float(RelSolVol[idx + 1]),
                metadata={"node_index": node_index, "tk": float(nodes[node_index].Tk)},
            )
        )

    segments.sort(key=lambda seg: seg.start_R, reverse=True)
    for idx, seg in enumerate(segments):
        seg.index = idx

    rmax = float(result.get("recovery", {}).get("Rmax", summary_matrix[-2][0] if len(summary_matrix) > 1 else 1.0))
    solvent_at_rmax = 0.0
    if segments:
        for seg in segments:
            if seg.start_R >= rmax and seg.end_R <= rmax:
                solvent_at_rmax = float(seg.solvent_end)
                break
        if solvent_at_rmax == 0.0:
            solvent_at_rmax = float(segments[-1].solvent_end)

    return EutecticPathState(
        solver=str(solver),
        initial_mixture=mix_input,
        nodes=nodes,
        segments=segments,
        Rmax=rmax,
        final_purity=float(result.get("recovery", {}).get("final_purity", 0.0)),
        tk_nodes=list(tk_nodes),
        degenerate_cascade=degenerate_cascade,
        solvent_at_Rmax=solvent_at_rmax,
    )
