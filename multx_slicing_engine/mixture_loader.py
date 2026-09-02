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
from copy import deepcopy
from typing import Any, Dict, List


def _normalize_component_vector(values: List[float]) -> List[float]:
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


class Mixture:
    """Compatibility wrapper for MultXeu configuration and component JSON."""

    def __init__(self, data: Dict[str, Any]):
        self.raw = data or {}
        self.config_id = self.raw.get("config_id")
        self.mixture_id = self.raw.get("mixture_id")
        self.metadata = self.raw.get("metadata", {})
        self.components = self.raw.get("components", [])
        self.q = self._extract_q(self.raw)
        self.case_id = self.config_id or self.mixture_id
        self.use_bracket = True

        parsed = self._parse_components(self.raw)
        self.components = parsed["components"]
        self.modes = parsed["modes"]
        self.T = parsed["T"]
        self.DH = parsed["DH"]
        self.X = parsed["X"]

    @staticmethod
    def _extract_q(data: Dict[str, Any]) -> float:
        for key in ("q", "Q", "main_component_level", "main_component_level_pct"):
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    value = value.strip("%")
                value = float(value)
                return value / 100.0 if value > 1.0 and key in ("main_component_level", "main_component_level_pct") else value
        metadata = data.get("metadata", {})
        for key in ("main_component_level", "main_component_level_pct"):
            if key in metadata:
                value = float(metadata[key])
                return value / 100.0 if value > 1.0 else value
        return 1.0

    @staticmethod
    def _parse_components(data: Dict[str, Any]) -> Dict[str, List[float]]:
        components = data.get("components")
        if isinstance(components, list) and components and all(isinstance(item, dict) for item in components):
            X = _normalize_component_vector([float(item.get("Xc", item.get("X", 0.0))) for item in components])
            T = [float(item.get("Tm", item.get("T", 0.0))) for item in components]
            DH = [float(item.get("DH", item.get("dH", 0.0))) for item in components]
            modes = [str(item.get("mode", item.get("mode_id", "SVL"))).upper() for item in components]
            for index, item in enumerate(components):
                item["Xc"] = X[index]
            return {"components": components, "modes": modes, "T": T, "DH": DH, "X": X}

        n_components = int(data.get("n_components") or data.get("metadata", {}).get("n_components") or 1)
        main_level = data.get("main_component_level")
        if main_level is None and "metadata" in data:
            main_level = data["metadata"].get("main_component_level")
        if main_level is None:
            main_level = 0.8
        if isinstance(main_level, str):
            main_level = float(main_level.strip("%")) / 100.0
        elif main_level > 1.0:
            main_level = float(main_level) / 100.0

        main_level = float(main_level)
        if n_components <= 1:
            X = [1.0]
        else:
            X = [main_level] + [(1.0 - main_level) / (n_components - 1) for _ in range(n_components - 1)]

        T = [float(50.0 + 150.0 * i / max(n_components - 1, 1)) for i in range(n_components)]
        DH = [0.0714 * (273.0 + t) for t in T]
        modes = ["SVL"] * n_components

        if "modes" in data and isinstance(data["modes"], list):
            modes = [str(m).upper() for m in data["modes"]]
        elif "metadata" in data and isinstance(data["metadata"].get("modes"), list):
            modes = [str(m).upper() for m in data["metadata"]["modes"]]

        components = []
        for i in range(n_components):
            components.append({
                "index": i,
                "Xc": X[i],
                "Tm": T[i],
                "DH": DH[i],
                "mode": modes[i],
            })

        return {"components": components, "modes": modes, "T": T, "DH": DH, "X": X}

    def active_components(self):
        return [i for i, x in enumerate(self.X) if x > 1e-12]

    def copy(self):
        return Mixture(deepcopy(self.raw))

    def remove_component(self, idx):
        self.X[idx] = 0.0
        s = sum(self.X)
        if s > 0.0:
            self.X = [x / s for x in self.X]


def mixture_from_json(data: Any) -> Mixture:
    """Build a valid MultXeu mixture from either a full component JSON or a minimal single-system config."""
    if isinstance(data, Mixture):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object, got {type(data).__name__}")

    if "components" in data or "n_components" in data or "main_component_level" in data:
        return Mixture(data)

    if "mixture" in data and isinstance(data["mixture"], dict):
        return Mixture(data["mixture"])

    if "system" in data and isinstance(data["system"], dict):
        return Mixture(data["system"])

    raise ValueError("Unsupported MultXeu JSON structure: expected a full component set or a single-system config.")
