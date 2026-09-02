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

def report_component_errors(summary, Xs_obs):
    """
    Print component-wise squared error for each slice and total error per component.

    Parameters
    ----------
    summary : dict
        Cascade summary from harness.py (contains summary["slices"][s]["Xs"]).

    Xs_obs : list[list[float]]
        Observed solid compositions per slice from mixture_set.
        Shape: [n_slices][n_components].
    """

    slices_pred = summary["slices"]
    n_slices = len(slices_pred)
    n_components = len(slices_pred[0]["Xs"])

    # Accumulate total error per component across slices
    total_component_error = [0.0] * n_components

    print("\nComponent-wise error analysis:")
    print("--------------------------------")

    for s in range(n_slices):
        Xs_pred = slices_pred[s]["Xs"]
        Xs_true = Xs_obs[s]

        print(f"\nSlice {s+1} (R={slices_pred[s]['slice_R']:.4f}):")

        for i in range(n_components):
            diff = Xs_pred[i] - Xs_true[i]
            err = diff * diff
            total_component_error[i] += err

            print(f"  Component {i+1} error: {err:.6f} "
                  f"(pred={Xs_pred[i]:.6f}, obs={Xs_true[i]:.6f})")

    print("\nTotal error per component across all slices:")
    print("--------------------------------------------")
    for i, err in enumerate(total_component_error):
        print(f"  Component {i+1}: {err:.6f}")

    print()
    pass

def report_component_slice_deltas(summary, Xs_obs):
    """
    Report how each component's solid composition changes across slices,
    comparing predicted vs observed slice deltas.

    Parameters
    ----------
    summary : dict
        Cascade summary from harness.py (contains summary["slices"][s]["Xs"]).

    Xs_obs : list[list[float]]
        Observed solid compositions per slice from mixture_set.
        Shape: [n_slices][n_components].
    """

    slices_pred = summary["slices"]
    n_slices = len(slices_pred)
    n_components = len(slices_pred[0]["Xs"])

    # Extract predicted Xs per slice
    Xs_pred = [sl["Xs"] for sl in slices_pred]

    print("\nComponent slice-delta analysis:")
    print("--------------------------------")

    for i in range(n_components):
        # Predicted deltas across slices
        pred_values = [Xs_pred[s][i] for s in range(n_slices)]
        pred_delta = max(pred_values) - min(pred_values)

        # Observed deltas across slices
        obs_values = [Xs_obs[s][i] for s in range(n_slices)]
        obs_delta = max(obs_values) - min(obs_values)

        print(f"\nComponent {i+1}:")
        print(f"  Predicted slice delta: {pred_delta:.6f} "
              f"(min={min(pred_values):.6f}, max={max(pred_values):.6f})")
        print(f"  Observed   slice delta: {obs_delta:.6f} "
              f"(min={min(obs_values):.6f}, max={max(obs_values):.6f})")

        # Optional diagnostic: difference between predicted and observed deltas
        delta_diff = pred_delta - obs_delta
        print(f"  Delta mismatch (pred - obs): {delta_diff:.6f}")
        pass

def compute_structured_objective(summary, Xs_obs, alpha=1.0, beta=1.0):
    """
    Structured objective over slices × components:
      - pointwise fit term (alpha)
      - slice-delta shape term (beta)
    """

    slices_pred = summary["slices"]
    n_slices = len(slices_pred)
    n_components = len(slices_pred[0]["Xs"])

    # ---------- Pointwise fit ----------
    pointwise_error = 0.0
    slice_errors = [0.0] * n_slices
    component_errors = [0.0] * n_components

    for s in range(n_slices):
        X_pred = slices_pred[s]["Xs"]
        X_obs = Xs_obs[s]
        for i in range(n_components):
            diff = X_pred[i] - X_obs[i]
            sq = diff * diff
            pointwise_error += sq
            slice_errors[s] += sq
            component_errors[i] += sq

    # ---------- Slice-delta shape ----------
    Xs_pred = [sl["Xs"] for sl in slices_pred]
    delta_penalty = 0.0
    component_delta_penalties = [0.0] * n_components

    for i in range(n_components):
        pred_vals = [Xs_pred[s][i] for s in range(n_slices)]
        obs_vals  = [Xs_obs[s][i]   for s in range(n_slices)]

        pred_delta = max(pred_vals) - min(pred_vals)
        obs_delta  = max(obs_vals)  - min(obs_vals)

        d = pred_delta - obs_delta
        sq = d * d
        delta_penalty += sq
        component_delta_penalties[i] = sq

    total_objective = alpha * pointwise_error + beta * delta_penalty

    return {
        "total_objective": total_objective,
        "pointwise_error": pointwise_error,
        "delta_penalty": delta_penalty,
        "slice_errors": slice_errors,
        "component_errors": component_errors,
        "component_delta_penalties": component_delta_penalties,
    }
def compute_structured_gradient(summary, Xs_obs, alpha=1.0, beta=1.0):
    slices_pred = summary["slices"]
    n_slices = len(slices_pred)
    n_components = len(slices_pred[0]["Xs"])

    # Gradient w.r.t. each component
    G = [0.0] * n_components

    # Pointwise error contribution
    for s in range(n_slices):
        X_pred = slices_pred[s]["Xs"]
        X_obs = Xs_obs[s]
        for i in range(n_components):
            G[i] += 2 * alpha * (X_pred[i] - X_obs[i])

    # Slice-delta shape contribution
    Xs_pred = [sl["Xs"] for sl in slices_pred]
    for i in range(n_components):
        pred_vals = [Xs_pred[s][i] for s in range(n_slices)]
        obs_vals  = [Xs_obs[s][i]   for s in range(n_slices)]

        pred_delta = max(pred_vals) - min(pred_vals)
        obs_delta  = max(obs_vals)  - min(obs_vals)

        d = pred_delta - obs_delta
        sign = 1.0 if pred_delta >= obs_delta else -1.0
        for s in range(n_slices):
            if Xs_pred[s][i] == max(pred_vals):
                G[i] += 2 * beta * d * sign
            elif Xs_pred[s][i] == min(pred_vals):
                G[i] -= 2 * beta * d * sign

    return G

