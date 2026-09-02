
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


# print("Loaded solvent_gradient.py from:", __file__)


class PurificationModel:
    def __init__(self, run_id, cumR, comp_after, gradients, intercepts, metadata=None):
        self.run_id = run_id
        self.cumR = cumR
        self.comp_after = comp_after
        self.gradients = gradients
        self.intercepts = intercepts
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<PurificationModel run_id={self.run_id}>"


class SolventGradientEngine:
    def __init__(self, result_data):
        """
        result_data must contain:
        - cumR: list of cumulative recoveries after each node (descending from 1 to 0)
        - solvent: list of scalar solvent values at each node (RelSolVol)
        """
        self.cascade = result_data
        self.cumR = result_data.get("cumR", [])
        self.solvent = result_data.get("solvent", [])
        self.gradients = None
        self.intercepts = None

    def find_interval(self, R):
        """
        Given a recovery R (0–1), find the cascade interval [k, k+1]
        such that cumR[k+1] <= R <= cumR[k].
        """
        for k in range(len(self.cumR) - 1):
            if self.cumR[k+1] <= R <= self.cumR[k]:
                return k

        return len(self.cumR) - 2

    def compute_gradients(self):
        """
        Compute gradient (m) and intercept (b) for each segment
        between each cascade node.
        """
        cumR = self.cumR
        solvent = self.solvent

        gradients = []
        intercepts = []

        num_intervals = len(cumR) - 1

        if len(solvent) < num_intervals + 1:
            last_value = solvent[-1]
            while len(solvent) < num_intervals + 1:
                solvent.append(last_value)

        for k in range(num_intervals):
            Ck = cumR[k]
            Ck1 = cumR[k+1]

            solvent_k = solvent[k]
            solvent_k1 = solvent[k+1]

            if Ck1 == Ck:
                m = 0.0
            else:
                m = (solvent_k1 - solvent_k) / (Ck1 - Ck)

            b = solvent_k - m * Ck

            gradients.append(m)
            intercepts.append(b)

        self.gradients = gradients
        self.intercepts = intercepts

    def value_at_recovery(self, R):
        """Compute scalar value at arbitrary recovery R (0–1)."""
        if self.gradients is None or self.intercepts is None:
            raise ValueError("Gradients not computed. Call compute_gradients() first.")

        k = self.find_interval(R)

        m = self.gradients[k]
        b = self.intercepts[k]

        return m * R + b

    def generate_profile(self, R_values):
        """
        Generate solvent profile over a list of recovery values.
        Returns a list of tuples: (R, scalar_value)
        """
        profile = []

        for R in R_values:
            value = self.value_at_recovery(R)
            profile.append((R, value))

        return profile

