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

"""Minimal compatibility wrapper for the legacy root `mixture_input` import.

The production implementation is package-based; this file preserves the older
import surface used by the codebase while keeping the actual logic under the
`cryss_core` and `multxeu_core` folders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class MixtureInput:
    """Simple container for a mixture state used by the solver and cascade paths."""

    T: List[float]
    DH: List[float]
    X: List[float]
    case_id: str = "case_001"
    modes: List[str] = field(default_factory=lambda: ["SVL"])
    use_bracket: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T = [float(v) for v in self.T]
        self.DH = [float(v) for v in self.DH]
        self.X = [float(v) for v in self.X]
        self.modes = [str(v).upper() for v in self.modes]
        if len(self.T) != len(self.X) or len(self.DH) != len(self.X):
            raise ValueError("MixtureInput requires matching component lengths for T, DH, and X.")
        self.n_components = len(self.X)

    def __len__(self) -> int:
        return len(self.X)

    def __iter__(self):
        yield from zip(self.T, self.DH, self.X, self.modes)

    def active_components(self) -> List[int]:
        return [idx for idx, value in enumerate(self.X) if value > 1e-12]

    def copy(self) -> "MixtureInput":
        return MixtureInput(
            T=list(self.T),
            DH=list(self.DH),
            X=list(self.X),
            case_id=self.case_id,
            modes=list(self.modes),
            use_bracket=self.use_bracket,
            metadata=dict(self.metadata),
        )

    def remove_component(self, idx: int) -> None:
        if 0 <= idx < len(self.X):
            self.X[idx] = 0.0
            total = sum(self.X)
            if total > 0.0:
                self.X = [value / total for value in self.X]
            self.n_components = len(self.X)
