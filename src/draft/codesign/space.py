"""Versioned, bounded physical design coordinates shared by training and export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from draft.paths import repo_root
from .topology import ACTIVE_NAMES, JOINT_GROUP_NAMES, LOCK_ANGLE_NAMES


GEOMETRY_NAMES = (
    "upper_leg_link_length", "lower_leg_link_length", "torso_lx", "torso_ly",
    "torso_lz", "shoulder_width", "hip_roll_link_length",
)
DEFAULT_LOCK_ANGLE_BOUNDS = ((-.8, .8), (-2.5, 2.5), (-2.8, 0.),
                             (-.8, .8), (-2.5, 2.5), (-2.8, 0.))


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
    lock_angle_bounds: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        try:
            angle_bounds = tuple(tuple(pair) for pair in self.lock_angle_bounds)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed lock-angle bounds") from exc
        object.__setattr__(self, "lock_angle_bounds", angle_bounds)
        if type(self.version) is not int or self.version not in (1, 2) or not self.parameters:
            raise ValueError("Unsupported or empty design-space schema")
        if self.version == 1 and self.lock_angle_bounds:
            raise ValueError("Version one cannot contain topology bounds")
        if self.version == 2:
            if len(self.lock_angle_bounds) != 6 or any(
                len(pair) != 2 or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                          and math.isfinite(x) for x in pair) or pair[0] >= pair[1]
                for pair in self.lock_angle_bounds
            ):
                raise ValueError("Six finite ordered lock-angle bounds are required")
        if tuple(p.name for p in self.parameters) != GEOMETRY_NAMES:
            raise ValueError("Quadruped geometry coordinates must use the complete declared order")
        reference = json.loads(self.reference_json)
        if not isinstance(reference, dict):
            raise ValueError("Design reference must be a parameter mapping")
        for key in ("allow_hypothetical", "allow_motor_overlap", "allow_actuator_in_structure"):
            if reference.get(key):
                raise ValueError(f"Co-design cannot disable feasibility through {key}")
        for p in self.parameters:
            if (p.name not in reference or not isinstance(reference[p.name], (int, float))
                    or isinstance(reference[p.name], bool) or not math.isfinite(reference[p.name])):
                raise ValueError(f"Reference lacks numeric parameter {p.name}")
        if self.topology_enabled:
            roles = ("hip_roll", "hip_pitch", "knee") * 2
            for group, role, (lower, upper) in zip(JOINT_GROUP_NAMES, roles, self.lock_angle_bounds):
                limits = reference.get(f"{role}_range")
                if (not isinstance(limits, (list, tuple)) or len(limits) != 2
                        or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                   and math.isfinite(x) for x in limits)
                        or limits[0] >= limits[1]):
                    raise ValueError(f"Missing physical joint limits for {role}")
                if lower < limits[0] or upper > limits[1]:
                    raise ValueError(f"{group}: lock-angle bounds exceed physical joint limits")
                if role == "hip_roll" and (-upper < limits[0] or -lower > limits[1]):
                    raise ValueError(f"{group}: mirrored roll bounds exceed physical joint limits")

    @property
    def dim(self) -> int:
        return len(self.parameters) + (12 if self.topology_enabled else 0)

    @property
    def topology_enabled(self) -> bool:
        return self.version == 2

    @property
    def conditioning_dim(self) -> int:
        return self.dim

    def geometry_space(self) -> DesignSpace:
        return DesignSpace(self.parameters, self.reference_json)

    @property
    def names(self) -> tuple[str, ...]:
        geometry = tuple(p.name for p in self.parameters)
        return geometry + ACTIVE_NAMES + LOCK_ANGLE_NAMES if self.topology_enabled else geometry

    @property
    def lower(self) -> tuple[float, ...]:
        geometry = tuple(p.lower for p in self.parameters)
        return geometry + (0.,) * 6 + tuple(pair[0] for pair in self.lock_angle_bounds) if self.topology_enabled else geometry

    @property
    def upper(self) -> tuple[float, ...]:
        geometry = tuple(p.upper for p in self.parameters)
        return geometry + (1.,) * 6 + tuple(pair[1] for pair in self.lock_angle_bounds) if self.topology_enabled else geometry

    @property
    def center(self) -> tuple[float, ...]:
        geometry = tuple(p.initial for p in self.parameters)
        return geometry + (1.,) * 6 + (0.,) * 6 if self.topology_enabled else geometry

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
        if self.topology_enabled and not np.isin(values[..., 7:13], (0., 1.)).all():
            raise ValueError("Topology active masks must be binary")
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        result = 2.0 * (values - lo) / (hi - lo) - 1.0
        outside = np.abs(result) > 1.0 + 1e-5
        if self.topology_enabled:
            outside[..., 13:19] &= values[..., 7:13] == 0.
        if outside.any():
            raise ValueError("Physical design lies outside its declared bounds")
        result = np.clip(result, -1.0, 1.0)
        if self.topology_enabled:
            result[..., 7:13] = values[..., 7:13]
            result[..., 13:19] *= 1. - values[..., 7:13]
        return result

    def denormalize(self, normalized: Sequence[float] | np.ndarray) -> np.ndarray:
        values = self._array(normalized)
        if (np.abs(values) > 1.0 + 1e-5).any():
            raise ValueError("Normalized design lies outside [-1, 1]")
        if self.topology_enabled and not np.isin(values[..., 7:13], (0., 1.)).all():
            raise ValueError("Topology active masks must be binary")
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        result = lo + (np.clip(values, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)
        if self.topology_enabled:
            result[..., 7:13] = values[..., 7:13]
            result[..., 13:19] *= 1. - values[..., 7:13]
        return result

    def overrides(self, physical: Sequence[float]) -> dict[str, Any]:
        values = self._array(physical)
        if values.ndim != 1:
            raise ValueError("One physical design is required")
        values = self.denormalize(self.normalize(values))
        result = {**self.reference_parameters,
                  **dict(zip(GEOMETRY_NAMES, map(float, values[:7])))}
        if self.topology_enabled:
            result["joint_topology"] = {
                "active": [int(x) for x in values[7:13]],
                "lock_angles": [float(x) for x in values[13:19]],
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        result = {"version": self.version,
                  "parameters": [asdict(p) for p in self.parameters],
                  "reference": self.reference_parameters}
        if self.topology_enabled:
            result["joint_topology"] = {
                "groups": list(JOINT_GROUP_NAMES),
                "active_names": list(ACTIVE_NAMES),
                "lock_angle_names": list(LOCK_ANGLE_NAMES),
                "lock_angle_bounds": [list(pair) for pair in self.lock_angle_bounds],
                "symmetry": "paired_left_right_right_roll_negated",
            }
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DesignSpace:
        version = data.get("version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported design-space schema version")
        unknown = set(data) - ({"version", "parameters", "reference", "joint_topology"} if version == 2
                               else {"version", "parameters", "reference"})
        if unknown:
            raise ValueError(f"Unknown design-space keys: {sorted(unknown)}")
        topology = data.get("joint_topology", {})
        if version == 2:
            if not isinstance(topology, Mapping) or set(topology) != {"groups", "active_names", "lock_angle_names", "lock_angle_bounds", "symmetry"} or (topology["groups"] != list(JOINT_GROUP_NAMES) or
                topology["active_names"] != list(ACTIVE_NAMES) or
                topology["lock_angle_names"] != list(LOCK_ANGLE_NAMES) or
                topology["symmetry"] != "paired_left_right_right_roll_negated"):
                raise ValueError("Invalid joint topology schema")
            if not isinstance(topology["lock_angle_bounds"], (list, tuple)):
                raise ValueError("Invalid joint topology lock-angle bounds")
        return cls(tuple(DesignParameter(**p) for p in data["parameters"]),
                   json.dumps(data["reference"], sort_keys=True, allow_nan=False),
                   version=version,
                   lock_angle_bounds=topology["lock_angle_bounds"] if version == 2 else ())

    @classmethod
    def quadruped(cls, *, bounds: Mapping[str, Sequence[float]] | None = None,
                  reference_overrides: Mapping[str, Any] | None = None,
                  relative_range: float = 0.15, joint_topology: bool = False,
                  lock_angle_bounds: Mapping[str, Sequence[float]] | None = None) -> DesignSpace:
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
        if not isinstance(joint_topology, bool):
            raise ValueError("joint_topology must be a boolean")
        if lock_angle_bounds and not joint_topology:
            raise ValueError("Lock-angle bounds require joint topology")
        if lock_angle_bounds is not None and not isinstance(lock_angle_bounds, Mapping):
            raise ValueError("lock_angle_bounds must be a mapping")
        unknown = set(lock_angle_bounds or {}) - set(JOINT_GROUP_NAMES)
        if unknown:
            raise ValueError(f"Unknown joint lock-angle bounds: {sorted(unknown)}")
        angle_bounds = tuple(tuple(map(float, (lock_angle_bounds or {}).get(name, default)))
                             for name, default in zip(JOINT_GROUP_NAMES, DEFAULT_LOCK_ANGLE_BOUNDS))
        return cls(tuple(parameters), json.dumps(reference, sort_keys=True, allow_nan=False),
                   version=2 if joint_topology else 1,
                   lock_angle_bounds=angle_bounds if joint_topology else ())
