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

def solve_eutectic_walkdown(mixture, modes,
                            criteria=1e-4,
                            initial_step=5,
                            max_iter=2000):

    # Start from a safe high temperature:
    # lowest melting point + margin
    HighTemp = max(mixture.T[i] for i in mixture.active_components())


    Change = initial_step
    Convergence = False

    for _ in range(max_iter):
        SumOfXs = 0.0

        for i in mixture.active_components():
            Ti = mixture.T[i]
            DH = mixture.DH[i]
            R = 0.008314  #kJ/mol/K

            if modes[i] == "PD":
                # PG quadratic model
                c = -math.exp(2 * DH / R * (1/(Ti+273) - 1/(HighTemp+273)))
                if c < -1.0:
                    c = -1.0
                Xeu_i = (-4.0 + math.sqrt(16.0 + 16.0*c)) / -8.0
            else:
                # SVL model
                Xeu_i = math.exp(DH / R * (1/(Ti+273) - 1/(HighTemp+273)))

            SumOfXs += Xeu_i

        diff = SumOfXs - 1.0

        if diff > criteria:
            # too eutectic-rich → step down
            HighTemp -= Change
            Convergence = False

        elif diff < 0.0:
            # too lean → step up and shrink step
            HighTemp += Change
            Change /= 2.0
            Convergence = False

        else:
            Convergence = True

        if Convergence:
            break

    return HighTemp
