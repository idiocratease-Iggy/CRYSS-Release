# multx_slicing_engine/slice_generator.py
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


from multxeu_core.multx_slicing_engine.evaluate_recovery import evaluate_at_recovery

try:
    from multxeu_core.multx_slicing_engine.evaluate_recovery import _segment_for_recovery
except ImportError:
    from multxeu_core.multx_slicing_engine.evaluate_recovery import _segment_for_recovery


def debug_slicing_state(state, slice_results):
    """Print the canonical node/segment summary and the resulting slice values."""
    # print("\n=== DEBUG: canonical nodes ===")
    for node in getattr(state, "nodes", []):
        pass

    # print("\n=== DEBUG: canonical segments ===")
    for seg in getattr(state, "segments", []):
        pass

    segments = list(getattr(state, "segments", []))

    # print("\n=== DEBUG: slice values ===")
    for slice_result in slice_results:
        if isinstance(slice_result, dict):
            R = float(slice_result.get("R", 0.0))
            slice_xsol = list(slice_result.get("Xsol", []))
            slice_xliq = list(slice_result.get("Xliq", []))
        else:
            R = float(getattr(slice_result, "R", 0.0))
            slice_xsol = list(getattr(slice_result, "Xsol", []))
            slice_xliq = list(getattr(slice_result, "Xliq", []))

        seg = _segment_for_recovery(segments, R)
        seg_xsol = list(getattr(seg, "Xsol", []))
        pass

    # print("\n=== DEBUG: slice summary complete ===")


def generate_slices_for_mixture(summary, num_slices, slice_R=None):
    """
    Generate slice points between R=1.0 and Rmax for a single mixture.

    If an explicit slice_R list is supplied, it is used verbatim. Otherwise the
    function falls back to evenly spaced anchors. In either case, there is no
    random perturbation and the downstream calculations receive the exact target
    recovery values.
    """
    state = summary["state"]
    grads = summary["segment_grads"]
    Rmax = summary["Rmax"]

    if num_slices <= 0:
        return []

    if slice_R is not None:
        anchor_points = [float(v) for v in list(slice_R)]
    else:
        explicit = summary.get("slice_R") if isinstance(summary, dict) else None
        if explicit is not None:
            anchor_points = [float(v) for v in list(explicit)]
        else:
            interval = max(1e-9, 1.0 - Rmax)
            step = interval / (num_slices + 1)
            anchor_points = [1.0 - step * (k + 1) for k in range(num_slices)]

    anchor_points = sorted(anchor_points, reverse=True)

    results = []
    for R in anchor_points:
        res = evaluate_at_recovery(state, grads, R)
        results.append(res)

    debug_slicing_state(state, results)
    return results




def print_slices_to_terminal(mixture_id, slice_results, state=None):
    """
    Pretty-print slice results to terminal.
    """
    # print("\n==============================================")
    # print(f" SLICES FOR MIXTURE {mixture_id}")
    # print("==============================================\n")

    if state is not None:
        debug_slicing_state(state, slice_results)

    for s in slice_results:
        xs = ", ".join(f"{v*100:.2f}%" for v in s.Xliq)
        xsol = ", ".join(f"{v:.6f}" for v in s.Xsol)
        # print(f"Slice at R={s.R:.4f}")
        # print(f"  seg.index={s.seg_index}  seg.start_R={s.seg_start_R:.6f}  seg.end_R={s.seg_end_R:.6f}")
        # print(f"  seg.Xsol=[{xsol}]")
        # print(f"  Xliq=[{xs}]")
        # print(f"  sum(Xliq)={sum(s.Xliq):.6f}  min={min(s.Xliq):.6f}  max={max(s.Xliq):.6f}")
        # print(f"  Solvent(RelSolVol@slice): {s.solvent:.4f}")
        # print("")


def resolve_slice_count(summary, fallback_num_slices=3):
    """Resolve the requested slice count from the loaded mixture JSON, with a safe fallback."""
    mixture = summary.get("mixture") if isinstance(summary, dict) else None
    if mixture is None:
        return int(fallback_num_slices)

    raw = getattr(mixture, "raw", mixture) or {}
    if not isinstance(raw, dict):
        return int(fallback_num_slices)

    raw_slicing = raw.get("slicing") if isinstance(raw.get("slicing"), dict) else {}
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata", {}), dict) else {}
    metadata_slicing = metadata.get("slicing") if isinstance(metadata.get("slicing"), dict) else {}
    slicing_cfg = raw_slicing or metadata_slicing or {}

    configured_slices = slicing_cfg.get("max_solvent_volumes")
    if configured_slices is not None:
        try:
            count = int(configured_slices)
            if count > 0:
                return count
        except (TypeError, ValueError):
            pass

    return int(fallback_num_slices)


def generate_slices_all_mixtures(canonical_results, num_slices=3):
    """
    Generate slices for all mixtures in canonical_results.
    Returns a list of dicts:
        {
            "mixture_id": ...,
            "slices": [slice_result, ...]
        }
    """
    all_slices = []

    for summary in canonical_results:
        mixture = summary["mixture"]
        mixture_id = mixture.mixture_id

        slice_count = resolve_slice_count(summary, fallback_num_slices=num_slices)
        slice_results = generate_slices_for_mixture(summary, slice_count)

        all_slices.append({
            "mixture_id": mixture_id,
            "slices": slice_results
        })

    return all_slices
