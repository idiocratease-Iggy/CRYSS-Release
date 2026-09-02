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


from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Node:
    index: int
    Tk: float
    Xeu: List[float]
    Xliq_after: List[float]
    recovery: float
    removed_components: List[int]
    multi_removal: bool
    solvent: float


@dataclass
class Segment:
    index: int
    start_R: float
    end_R: float
    Xliq_start: List[float]
    Xliq_end: List[float]
    Xsol: List[float]
    solvent_start: float
    solvent_end: float
    metadata: Dict[str, Any] = None


@dataclass
class EutecticPathState:
    solver: str
    initial_mixture: Any
    nodes: List[Node]
    segments: List[Segment]
    Rmax: float
    final_purity: float
    tk_nodes: List[float]
    degenerate_cascade: bool
    solvent_at_Rmax: float = 0.0
