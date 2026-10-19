"""Observation-only morphology conditioning for native quadruped tasks."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from mjlab.managers.observation_manager import ObservationTermCfg

from draft.codesign.space import DesignSpace, GEOMETRY_NAMES
from draft.tasks.quadruped.speed_limits import _resolve_robot_xml


class MorphologyObservation:
    """Cache immutable normalized geometry once, including heterogeneous rows."""

    def __init__(self, cfg, env):
        values = torch.as_tensor(cfg.params["values"], dtype=torch.float32, device=env.device)
        self.values = values.expand(env.num_envs, -1) if values.ndim == 1 else values
        if self.values.shape != (env.num_envs, len(GEOMETRY_NAMES)):
            raise ValueError("Morphology observation must have seven features per environment")

    def __call__(self, env, values) -> torch.Tensor:
        del env, values
        return self.values


def append_morphology_observations(cfg, values: Sequence[float] | np.ndarray,
                                   *, metadata: dict[str, Any] | None = None):
    """Append seven features to actor/critic without changing any native terms.

    ``values`` may be one shared normalized vector or one vector per world.
    This API does not change terrain, rewards, reset state, action offsets,
    randomization, episode duration, or any other environment configuration.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape not in ((len(GEOMETRY_NAMES),), (cfg.scene.num_envs, len(GEOMETRY_NAMES))):
        raise ValueError("Expected seven morphology features, shared or one row per environment")
    if not np.isfinite(values).all() or (np.abs(values) > 1.0 + 1e-5).any():
        raise ValueError("Morphology features must be finite and normalized to [-1, 1]")
    for name in ("actor", "critic"):
        if name not in cfg.observations or "morphology" in cfg.observations[name].terms:
            raise ValueError("Actor and critic must exist and have no morphology term yet")
    serialized = np.clip(values, -1.0, 1.0).tolist()
    for name in ("actor", "critic"):
        cfg.observations[name].terms["morphology"] = ObservationTermCfg(
            func=MorphologyObservation, params={"values": copy.deepcopy(serialized)})
    if metadata is not None:
        cfg.morphology_conditioning_metadata = copy.deepcopy(metadata)
    return cfg


def describe_morphology(robot_xml: str | Path, *,
                        space: DesignSpace | dict | None = None) -> dict:
    """Read the generated design's physical coordinates in the declared schema.

    Resolved generator parameters take precedence over GUI input or a source
    parameter table. Missing or out-of-domain geometry is rejected explicitly.
    """
    space = DesignSpace.quadruped() if space is None else space
    if isinstance(space, dict):
        space = DesignSpace.from_dict(space)
    if not isinstance(space, DesignSpace) or space.names != GEOMETRY_NAMES:
        raise ValueError("Morphology conditioning requires the complete ordered quadruped geometry space")
    xml_path = _resolve_robot_xml(Path(robot_xml)).resolve()
    candidates = tuple(xml_path.parent / name for name in (
        "parameters_resolved.yaml", "parameters_from_gui.yaml", "parameters.yaml"))
    parameter_path = next((path for path in candidates if path.is_file()), None)
    if parameter_path is None:
        raise FileNotFoundError(f"No generated quadruped parameter table beside {xml_path}")
    parameters = yaml.safe_load(parameter_path.read_text())
    if not isinstance(parameters, dict):
        raise ValueError(f"Expected a parameter mapping in {parameter_path}")
    try:
        physical = np.asarray([parameters[name] for name in GEOMETRY_NAMES], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Parameter table must contain all seven numeric geometry coordinates: {parameter_path}") from exc
    if not np.isfinite(physical).all() or (physical <= 0).any():
        raise ValueError("Physical geometry coordinates must be finite and positive")
    normalized = space.normalize(physical)
    metadata = dict(version=1, observation_term="morphology", feature_names=list(GEOMETRY_NAMES),
                    space_schema=space.to_dict(), physical=physical.tolist(), normalized=normalized.tolist(),
                    robot_xml=str(xml_path), parameters_path=str(parameter_path),
                    robot_xml_sha256=hashlib.sha256(xml_path.read_bytes()).hexdigest(),
                    parameters_sha256=hashlib.sha256(parameter_path.read_bytes()).hexdigest())
    return metadata


def condition_quadruped_cfg(cfg, robot_xml: str | Path, *,
                            space: DesignSpace | dict | None = None):
    """Append the generated robot's declared geometry to native observations."""
    metadata = describe_morphology(robot_xml, space=space)
    return append_morphology_observations(cfg, metadata["normalized"], metadata=metadata)
