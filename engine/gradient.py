# engine/gradient.py

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




import profile
# print("Loaded gradient.py from:", __file__)


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
    



class GradientEngine:

    def __init__(self, result_data):
        """
        result_data must contain:
        - cumR: list of cumulative recoveries after each node (descending from 1 to 0)
        - comp_after: list of compositions after each node (list of lists)
        """
        self.cascade = result_data
        self.cumR = result_data.get("cumR", [])
        self.comp_after = result_data.get("comp_after", [])
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

        # If R is exactly zero, return the last interval
        return len(self.cumR) - 2

    

    def compute_gradients(self):
        """
        Compute gradient (m) and intercept (b) for each component
        between each cascade node.
        """
        cumR = self.cumR
        comp = self.comp_after

        gradients = []
        intercepts = []

        # Number of intervals is len(cumR) - 1
        num_intervals = len(cumR) - 1
        
        # We need to ensure comp has the same number of points as cumR
        # If comp_after is shorter than cumR, we might need to pad it or handle it.
        # In this case, if num_intervals is 3, we need indices 0, 1, 2, 3 to be valid.
        # Therefore, len(comp) must be at least num_intervals + 1.
        
        if len(comp) < num_intervals + 1:
            # Pad comp with the last available composition if it's too short
            last_comp = comp[-1]
            while len(comp) < num_intervals + 1:
                comp.append(list(last_comp))

        for k in range(num_intervals):
            Ck = cumR[k]
            Ck1 = cumR[k+1]

            comp_k = comp[k]
            comp_k1 = comp[k+1]

            interval_grad = []
            interval_int = []

            for i in range(len(comp_k)):
                # Gradient
                m = (comp_k1[i] - comp_k[i]) / (Ck1 - Ck)

                # Intercept
                b = comp_k[i] - m * Ck

                interval_grad.append(m)
                interval_int.append(b)

            gradients.append(interval_grad)
            intercepts.append(interval_int)

        self.gradients = gradients
        self.intercepts = intercepts


    def composition_at_recovery(self, R):
        """Compute composition at arbitrary recovery R (0–1).
        Uses the linear gradient model between cascade nodes.
        """
        # Ensure gradients are computed
        if self.gradients is None or self.intercepts is None:
            raise ValueError("Gradients not computed. Call compute_gradients() first.")

        # Find interval
        k = self.find_interval(R)

        m = self.gradients[k]
        b = self.intercepts[k]

        # Compute composition for each component
        comp_R = []
        for i in range(len(m)):
            Xi = m[i] * R + b[i]
            comp_R.append(Xi)

        return comp_R


    def generate_profile(self, R_values):
        """
        Generate purification profile over a list of recovery values.
        Returns a list of tuples: (R, composition_vector)
        """
        profile = []

        for R in R_values:
            comp = self.composition_at_recovery(R)
            profile.append((R, comp))

        return profile
    
    # ============================================================
    # GradientEngine → build PurificationModel
    # ============================================================
    def build_model(self, run_id="run_001", metadata=None):
        self.compute_gradients()
        return PurificationModel(
            run_id=run_id,
            cumR=self.cumR,
            comp_after=self.comp_after,
            gradients=self.gradients,
            intercepts=self.intercepts,
            metadata=metadata or {}
        )

