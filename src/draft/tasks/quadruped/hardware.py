"""Opt-in geometry-aware posture and ground contact for the quadruped task."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from .quadruped_entity import _stance_height, default_stance
from .speed_limits import _resolve_robot_xml


def prepare_hardware_entity(entity, robot_xml, stance_xml=None, *, collision_subdivisions=None,
                            model: mujoco.MjModel | None = None,
                            stance_model: mujoco.MjModel | None = None) -> float:
    """Install a feasible nominal pose, measured root height and ground proxies."""
    from .hardware_collision import configure_hardware_collisions

    if model is None:
        model = mujoco.MjModel.from_xml_path(str(_resolve_robot_xml(Path(robot_xml)).resolve()))
    entity.init_state.joint_pos = default_stance(robot_xml, stance_xml, model=model,
                                               stance_model=stance_model)
    height = _stance_height(model, mujoco.MjData(model), -entity.init_state.joint_pos[".*knee"])
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Hardware geometry must have a positive finite nominal stance height")
    x, y, _ = entity.init_state.pos
    entity.init_state.pos = (x, y, height)
    configure_hardware_collisions(entity, subdivisions=collision_subdivisions, model=model)
    return float(height)


def set_hardware_heights(cfg, heights, replicas: int = 1) -> None:
    """Bind per-world height targets in design-major D × R order."""
    heights = np.asarray(heights, dtype=float)
    if (heights.ndim != 1 or not len(heights) or not np.isfinite(heights).all()
            or (heights <= 0).any() or isinstance(replicas, bool)
            or not isinstance(replicas, int) or replicas < 1):
        raise ValueError("Expected positive nominal stance heights and integer replicas")
    rows = np.repeat(heights, replicas)
    if len(rows) not in (1, cfg.scene.num_envs):
        raise ValueError("Nominal stance heights must match the hardware cohort layout")
    cfg.hardware_nominal_heights = tuple(map(float, heights))
    for term in (cfg.rewards["base_height"], cfg.terminations["base_too_low"]):
        term.params["nominal_heights"] = float(rows[0]) if len(rows) == 1 else rows.tolist()


def configure_hardware_task(cfg, robot_xml, stance_xml=None) -> None:
    """Correct internal posture criteria without rescaling the external task."""
    from .hardware_mdp import TerrainRelativeCollapse, TerrainRelativeHeightReward

    height = prepare_hardware_entity(cfg.scene.entities["robot"], robot_xml, stance_xml)
    cfg.hardware_aware = True
    cfg.rewards["base_height"] = RewardTermCfg(
        func=TerrainRelativeHeightReward, weight=cfg.rewards["base_height"].weight,
        params={"nominal_heights": [height], "sensor_name": "foot_height_scan", "asset_name": "robot"})
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=TerrainRelativeCollapse,
        params={"nominal_heights": [height], "sensor_name": "foot_height_scan", "asset_name": "robot"})
    sensor_name = "torso_ground_contact"
    cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors if sensor.name != sensor_name) + (
        ContactSensorCfg(name=sensor_name,
                         primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found",), reduce="none", num_slots=1),)
    cfg.terminations["torso_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact, params={"sensor_name": sensor_name})
    set_hardware_heights(cfg, [height])
