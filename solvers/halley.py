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


import math

def solve_eutectic_halley(mixture, modes, active=None):
    T = mixture.T[:]              # melting points (°C)
    DH = mixture.DH[:]            # enthalpies (kJ/mol)
    n = mixture.n_components
    R = 0.008314  # kJ/mol/K

    if active is None:
        active = mixture.active_components()

    TiK = [Ti + 273.15 for Ti in T]          # melting points in Kelvin
    Ci  = [DH[i] / R for i in range(n)]      # DH/R
    Bi  = [Ci[i] / TiK[i] for i in range(n)] # Ci/Ti
    PD_twoCi = [2 * Ci[i] for i in range(n)]

    # convenience flag: do we have any PD terms?
    has_pd = any(modes[i] == "PD" for i in active)

    T_guess = min(T[i] for i in active)

    # -----------------------------
    # f(Tk): Sum Xeu - 1
    # -----------------------------
    def f(Tk):
        Tg = Tk + 273.15
        SumXs = 0.0

        for i in active:
            if modes[i] == "PD":
                # compute A first
                A = PD_twoCi[i] * (1.0 / TiK[i] - 1.0 / Tg)

                # clip A to avoid overflow in exp(A)
                MAX_EXP_ARG = 700.0  # exp(700) ~ 1e304
                A_clipped = max(min(A, MAX_EXP_ARG), -MAX_EXP_ARG)

                c = -math.exp(A_clipped)

                # physical clamp
                if c < -1.0:
                    c = -1.0

                root = math.sqrt(16.0 + 16.0 * c)
                Xeu_i = (-4.0 + root) / -8.0
            else:  # SVL
                Xeu_i = math.exp(Bi[i] - Ci[i] / Tg)

            SumXs += Xeu_i

        return SumXs - 1.0

    # -----------------------------
    # df(Tk): first derivative
    # -----------------------------
    def df(Tk):
        # if any PD present, use numerical derivative for the full mixture
        if has_pd:
            h = 1e-4
            return (f(Tk + h) - f(Tk - h)) / (2.0 * h)

        # otherwise, pure SVL: analytic derivative
        Tg = Tk + 273.15
        invT2 = 1.0 / (Tg * Tg)

        dSum = 0.0
        for i in active:
            # SVL only here
            Xeu_i = math.exp(Bi[i] - Ci[i] / Tg)
            dE = Ci[i] * invT2          # d/dTg of exponent
            dX = Xeu_i * dE
            dSum += dX

        return dSum

    # -----------------------------
    # d2f(Tk): second derivative
    # -----------------------------
    def d2f(Tk):
        # if any PD present, numerical second derivative for full mixture
        if has_pd:
            h = 1e-4
            return (df(Tk + h) - df(Tk - h)) / (2.0 * h)

        # otherwise, pure SVL: analytic second derivative
        Tg = Tk + 273.15
        invT2 = 1.0 / (Tg * Tg)
        invT3 = invT2 / Tg

        d2Sum = 0.0
        for i in active:
            Xeu_i = math.exp(Bi[i] - Ci[i] / Tg)
            Ci_i = Ci[i]
            dE  = Ci_i * invT2              # first derivative of exponent
            d2E = -2.0 * Ci_i * invT3      # second derivative of exponent

            d2X = Xeu_i * (dE * dE + d2E)
            d2Sum += d2X

        return d2Sum

    # -----------------------------
    # Halley iteration
    # -----------------------------
    max_iter = 50
    tol_F = 1e-6
    tol_T = 1e-7
    
    iterations_used = 0
    
    max_step = 20.0        # limit how far Tk can jump in one iteration
    MAX_DENOM = 1e300      # prevent overflow in denominator

    for k in range(max_iter):
        fval = f(T_guess)
        f1   = df(T_guess)
        f2   = d2f(T_guess)

        iterations_used = k + 1

        # Safety: if anything is non-finite, Halley is unstable → fallback
        if not math.isfinite(fval) or not math.isfinite(f1) or not math.isfinite(f2):
            raise RuntimeError("Halley produced NaN/inf")

        # Convergence check
        if abs(fval) < tol_F:
            break

        # Halley denominator
        denom = 2.0 * (f1 * f1) - fval * f2

        # Overflow or near-zero denominator → Halley unstable → fallback
        if (not math.isfinite(denom)) or abs(denom) < 1e-16 or abs(denom) > MAX_DENOM:
            raise RuntimeError("Halley denominator unstable")

        # Halley update
        delta = (2.0 * fval * f1) / denom

        # Step-size damping
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

        T_guess -= delta

        # Convergence in Tk
        if abs(delta) < tol_T:
            break



    T_eutectic = T_guess
    residual = f(T_eutectic)

    return {
        "T_eutectic": T_eutectic,
        "iterations": iterations_used,
        "residual": residual,
    }

