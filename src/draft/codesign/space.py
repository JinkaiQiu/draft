"""Versioned, bounded physical design coordinates shared by training and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from draft.paths import repo_root


GEOMETRY_NAMES = (
    "upper_leg_link_length", "lower_leg_link_length", "torso_lx", "torso_ly",
    "torso_lz", "shoulder_width", "hip_roll_link_length",
)


@dataclass(frozen=True)
class DesignParameter:
    name: str
    lower: float
    upper: float
    initial: float

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Invalid parameter name: {self.name!r}")
        if not all(math.isfinite(x) for x in (self.lower, self.upper, self.initial)):
            raise ValueError(f"{self.name}: bounds and initial value must be finite")
        if not 0 < self.lower < self.upper or not self.lower < self.initial < self.upper:
            raise ValueError(f"{self.name}: require 0 < lower < initial < upper")


@dataclass(frozen=True)
class DesignSpace:
    """A stable order of physical coordinates, with an immutable reference spec.

    Normalized coordinates are in [-1, 1]. The JSON reference is deliberately
    immutable and serializable so spawned compiler workers and checkpoints use
    exactly the same definition.
    """

    parameters: tuple[DesignParameter, ...]
    reference_json: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if self.version != 1 or not self.parameters:
            raise ValueError("Unsupported or empty design-space schema")
        if self.names != GEOMETRY_NAMES:
            raise ValueError("Quadruped geometry coordinates must use the complete declared order")
        reference = json.loads(self.reference_json)
        if not isinstance(reference, dict):
            raise ValueError("Design reference must be a parameter mapping")
        for key in ("allow_hypothetical", "allow_motor_overlap", "allow_actuator_in_structure"):
            if reference.get(key):
                raise ValueError(f"Co-design cannot disable feasibility through {key}")
        for p in self.parameters:
            if p.name not in reference or not isinstance(reference[p.name], (int, float)):
                raise ValueError(f"Reference lacks numeric parameter {p.name}")

    @property
    def dim(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def lower(self) -> tuple[float, ...]:
        return tuple(p.lower for p in self.parameters)

    @property
    def upper(self) -> tuple[float, ...]:
        return tuple(p.upper for p in self.parameters)

    @property
    def center(self) -> tuple[float, ...]:
        return tuple(p.initial for p in self.parameters)

    @property
    def reference_parameters(self) -> dict[str, Any]:
        return json.loads(self.reference_json)

    def _array(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != self.dim or not np.isfinite(values).all():
            raise ValueError(f"Expected finite design coordinates with final dimension {self.dim}")
        return values

    def normalize(self, physical: Sequence[float] | np.ndarray) -> np.ndarray:
        values = self._array(physical)
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        result = 2.0 * (values - lo) / (hi - lo) - 1.0
        if (np.abs(result) > 1.0 + 1e-5).any():
            raise ValueError("Physical design lies outside its declared bounds")
        return np.clip(result, -1.0, 1.0)

    def denormalize(self, normalized: Sequence[float] | np.ndarray) -> np.ndarray:
        values = self._array(normalized)
        if (np.abs(values) > 1.0 + 1e-5).any():
            raise ValueError("Normalized design lies outside [-1, 1]")
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        return lo + (np.clip(values, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)

    def overrides(self, physical: Sequence[float]) -> dict[str, Any]:
        values = self._array(physical)
        if values.ndim != 1:
            raise ValueError("One physical design is required")
        values = self.denormalize(self.normalize(values))
        return {**self.reference_parameters, **dict(zip(self.names, map(float, values)))}

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version,
                "parameters": [asdict(p) for p in self.parameters],
                "reference": self.reference_parameters}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DesignSpace:
        unknown = set(data) - {"version", "parameters", "reference"}
        if unknown:
            raise ValueError(f"Unknown design-space keys: {sorted(unknown)}")
        return cls(tuple(DesignParameter(**p) for p in data["parameters"]),
                   json.dumps(data["reference"], sort_keys=True, allow_nan=False),
                   version=data.get("version", 1))

    @classmethod
    def quadruped(cls, *, bounds: Mapping[str, Sequence[float]] | None = None,
                  reference_overrides: Mapping[str, Any] | None = None,
                  relative_range: float = 0.15) -> DesignSpace:
        """The seven symmetric geometry dimensions around the cheetah design.

        Bounds are physical [lower, upper] pairs. Motor parameters, topology,
        and fitted physical properties remain part of the frozen reference.
        """
        if not math.isfinite(relative_range) or not 0 < relative_range < 1:
            raise ValueError("relative_range must lie strictly between zero and one")
        root = repo_root()
        base = root / "src/draft/robots/quadruped/parameters.yaml"
        variant = root / "experiments/quadruped_variants/cheetah.yaml"
        reference = {**yaml.safe_load(base.read_text()), **yaml.safe_load(variant.read_text()),
                     **dict(reference_overrides or {})}
        # The co-design domain is constrained to the existing generation gates.
        for key in ("allow_hypothetical", "allow_motor_overlap", "allow_actuator_in_structure"):
            if reference_overrides and reference_overrides.get(key):
                raise ValueError(f"Co-design cannot disable feasibility through {key}")
            reference[key] = False
        unknown = set(bounds or {}) - set(GEOMETRY_NAMES)
        if unknown:
            raise ValueError(f"Unknown geometry bounds: {sorted(unknown)}")
        parameters = []
        for name in GEOMETRY_NAMES:
            initial = float(reference[name])
            limits = (bounds or {}).get(name, (initial * (1 - relative_range),
                                                initial * (1 + relative_range)))
            if len(limits) != 2:
                raise ValueError(f"{name}: bounds must contain lower and upper")
            parameters.append(DesignParameter(name, float(limits[0]), float(limits[1]), initial))
        return cls(tuple(parameters), json.dumps(reference, sort_keys=True, allow_nan=False))
