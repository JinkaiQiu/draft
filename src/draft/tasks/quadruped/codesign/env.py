"""Native flat-ground locomotion with immutable heterogeneous geometry cohorts."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import logging
import math
import sys
from typing import Any, Sequence

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scene import Scene
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from draft.codesign.batch import (
    FOOT_SITES, HeterogeneousModelAdapter, PreparedCohort, prepare_cohort,
)
from draft.tasks.quadruped.quadruped_entity import (
    KP_PER_NM, REF_LEG_LENGTH, _stance_height, default_stance, make_quadruped_entity_cfg,
)


_LOG = logging.getLogger(__name__)


def morphology(env: "CoDesignEnv") -> torch.Tensor:
    return env.morphology


def remaining_time(env: "CoDesignEnv") -> torch.Tensor:
    return (1.0 - env.episode_length_buf.to(torch.float32) /
            env.max_episode_length).clamp(0.0, 1.0)[:, None]


def base_height(env: "CoDesignEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square((z - 0.95 * env.nominal_height) /
                                   (0.25 * env.nominal_height)))


def base_too_low(env: "CoDesignEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene["robot"]
    feet_z = robot.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=-1)
    return robot.data.root_link_pos_w[:, 2] - feet_z < 0.5 * env.nominal_height


def vertical_velocity_cost(env: "CoDesignEnv") -> torch.Tensor:
    # Dimensionless gravity/length normalization, independently for each design.
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square() / env.length_scale


def foot_clearance(env: "CoDesignEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene["robot"]
    height = robot.data.site_pos_w[:, asset_cfg.site_ids, 2]
    height = height - env.scene.env_origins[:, None, 2]
    velocity = robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1)
    target = 0.10 * env.length_scale[:, None]
    error = ((height - target) / env.length_scale[:, None]).square()
    moving = env.command_manager.get_command("twist")[:, :2].norm(dim=-1) > 0.05
    return (error * velocity).sum(dim=-1) * moving


def _robot_spec(cohort: PreparedCohort, index: int) -> mujoco.MjSpec:
    spec = cohort.robot_spec(index)
    # Attached entity options are intentionally replaced by the common task's
    # cfg.sim.mujoco options, applied to every final CPU/GPU scene below.
    spec.option = mujoco.MjSpec().option
    return spec


def _entity_cfg(cohort: PreparedCohort, index: int):
    design = cohort.designs[index]
    model = mujoco.MjModel.from_xml_path(str(design.xml_path))
    stance = default_stance(design.xml_path)
    # Exact kinematic stance height includes the generated links and foot sphere.
    height = _stance_height(model, mujoco.MjData(model), -stance[".*knee"]) + 0.01
    cfg = make_quadruped_entity_cfg(design.xml_path, init_height=height, geom_group=1)
    cfg.spec_fn = lambda: _robot_spec(cohort, index)
    return cfg, height


def _preserve_mesh_bank(spec: mujoco.MjSpec) -> None:
    # Entity collision filtering disables non-foot geoms. Keep bank meshes
    # referenced for native convex-hull compilation on a separate collision bit;
    # no task geom uses bit 2, and these static references are 1 km underground.
    for geom in spec.geoms:
        if geom.name.startswith("robot/bank_cohort_"):
            geom.contype = geom.conaffinity = 2


def make_codesign_env_cfg(cohort: PreparedCohort, *, episode_length_s: float = 20.0,
                          seed: int = 0, env_kwargs: dict | None = None,
                          reward_overrides: dict | None = None):
    """Common absolute velocity demands; morphology only scales physical shaping.

    Co-design intentionally has its own task: a plane, no curriculum, no model
    randomization and a fixed wall-clock horizon. Existing quadruped tasks retain
    their original terrain, reward and episode-scaling behavior.
    """
    options = dict(env_kwargs or {})
    allowed = {"lin_vel_x", "lin_vel_y", "ang_vel_z", "timestep", "decimation",
               "action_scale", "enable_corruption", "verbose", "render_mode",
               "render_width", "render_height"}
    if set(options) - allowed:
        raise ValueError(f"unknown co-design environment options: {sorted(set(options) - allowed)}")
    if not isinstance(options.get("verbose", False), bool):
        raise ValueError("verbose must be a boolean")
    if options.get("render_mode") not in (None, "rgb_array"):
        raise ValueError("render_mode must be None or 'rgb_array'")
    for name in ("render_width", "render_height"):
        value = options.get(name, 640 if name == "render_width" else 480)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2:
            raise ValueError(f"{name} must be a positive even integer")
    if not math.isfinite(episode_length_s) or episode_length_s <= 0:
        raise ValueError("episode_length_s must be finite and positive")
    cfg = make_velocity_env_cfg()
    entity_cfgs, heights = zip(*(_entity_cfg(cohort, i) for i in range(len(cohort.designs))))
    cfg.scene.entities = {"robot": entity_cfgs[0]}
    cfg.scene.num_envs = cohort.num_envs
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = ()
    cfg.scene.spec_fn = _preserve_mesh_bank
    cfg.seed = seed
    cfg.episode_length_s = float(episode_length_s)
    cfg.is_finite_horizon = True
    # Fall resets protect numerical stability; the trial learner masks all
    # samples after the first fall through the fixed outer horizon.
    cfg.auto_reset = True
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.events = {name: term for name, term in cfg.events.items()
                  if name in ("reset_base", "reset_robot_joints")}
    cfg.events["reset_base"].params["pose_range"] = {"yaw": (-math.pi, math.pi)}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.actions["joint_pos"].scale = float(options.get("action_scale", 0.25 / KP_PER_NM))
    cfg.sim.mujoco.timestep = float(options.get("timestep", 0.005))
    decimation = options.get("decimation", 4)
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
        raise ValueError("decimation must be a positive integer")
    cfg.decimation = decimation
    if not math.isfinite(cfg.sim.mujoco.timestep) or cfg.sim.mujoco.timestep <= 0:
        raise ValueError("timestep must be finite and positive")
    twist = cfg.commands["twist"]
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_heading_envs = 0.0
    for name, default in (("lin_vel_x", (-1.0, 1.0)), ("lin_vel_y", (-0.6, 0.6)),
                          ("ang_vel_z", (-0.5, 0.5))):
        value = tuple(float(x) for x in options.get(name, default))
        if len(value) != 2 or not np.isfinite(value).all() or value[0] > value[1]:
            raise ValueError(f"invalid common command range: {name}")
        setattr(twist.ranges, name, value)
    for group in cfg.observations.values():
        group.terms = {name: term for name, term in group.terms.items()
                       if name in ("base_lin_vel", "base_ang_vel", "projected_gravity",
                                   "joint_pos", "joint_vel", "actions", "command")}
        group.terms["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)
        group.terms["base_ang_vel"] = ObservationTermCfg(func=mdp.base_ang_vel)
        group.terms["morphology"] = ObservationTermCfg(func=morphology)
        group.terms["remaining_time"] = ObservationTermCfg(func=remaining_time)
    cfg.observations["actor"].enable_corruption = bool(options.get("enable_corruption", True))
    cfg.rewards = {name: term for name, term in cfg.rewards.items()
                   if name in ("track_linear_velocity", "track_angular_velocity", "upright",
                               "dof_pos_limits", "action_rate_l2")}
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.5
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["base_height"] = RewardTermCfg(func=base_height, weight=0.4)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=vertical_velocity_cost, weight=-2.0)
    sites = SceneEntityCfg("robot", site_names=FOOT_SITES)
    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=foot_clearance, weight=-2.0, params={"asset_cfg": sites})
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"].params["limit_angle"] = 1.0
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=base_too_low, params={"asset_cfg": copy.deepcopy(sites)})
    for name, weight in (reward_overrides or {}).items():
        if name not in cfg.rewards or not math.isfinite(float(weight)):
            raise ValueError(f"invalid co-design reward override: {name}")
        cfg.rewards[name].weight = float(weight)
    cfg.viewer.body_name = "torso_link"
    cfg.viewer.env_idx = 0
    cfg.viewer.max_extra_envs = 0
    cfg.viewer.width = options.get("render_width", 640)
    cfg.viewer.height = options.get("render_height", 480)
    cfg.viewer.distance = max(1.0, 4.0 * cohort.designs[0].leg_length)
    cfg.viewer.elevation = -25.0
    cfg.viewer.azimuth = 120.0
    return cfg, entity_cfgs, heights


class CoDesignEnv(ManagerBasedRlEnv):
    """One native simulation for D designs, each with R contiguous replicas."""

    def __init__(self, cfg, cohort: PreparedCohort, entity_cfgs, heights, *, device: str,
                 render_mode: str | None = None):
        self.cohort = cohort
        self._cohort_entity_cfgs = entity_cfgs
        self._cohort_heights = heights
        # ObservationManager evaluates terms during construction.
        self.morphology = torch.tensor([d.morphology for d in cohort.designs],
                                       dtype=torch.float32, device=device).repeat_interleave(
                                           cohort.replicas, dim=0)
        self.design_index = torch.as_tensor(cohort.design_index, device=device)
        self.nominal_height = torch.tensor(heights, dtype=torch.float32,
                                           device=device).repeat_interleave(cohort.replicas)
        self.length_scale = torch.tensor([d.leg_length / REF_LEG_LENGTH for d in cohort.designs],
                                         dtype=torch.float32, device=device).repeat_interleave(
                                             cohort.replicas)
        super().__init__(cfg, device=device, render_mode=render_mode)
        self.horizon_steps = self.max_episode_length

    def load_managers(self):
        # Construct independent complete CPU scenes using the exact same terrain,
        # collision setup and motor conversion as the GPU scene.
        models = []
        for entity_cfg in self._cohort_entity_cfgs:
            scene_cfg = copy.deepcopy(self.cfg.scene)
            scene_cfg.entities = {"robot": entity_cfg}
            model = Scene(scene_cfg, device="cpu").compile()
            self.cfg.sim.mujoco.apply(model)
            models.append(model)
        self.cpu_models = tuple(models)
        HeterogeneousModelAdapter.apply(self, self.cohort, models)
        robot = self.scene["robot"]
        for index, entity_cfg in enumerate(self._cohort_entity_cfgs):
            rows = slice(index * self.cohort.replicas, (index + 1) * self.cohort.replicas)
            robot.data.default_root_state[rows, 2] = self._cohort_heights[index]
            from mjlab.utils.string import resolve_expr
            stance = resolve_expr(entity_cfg.init_state.joint_pos, robot.joint_names, 0.0)
            robot.data.default_joint_pos[rows] = torch.tensor(stance, device=self.device)
        # Entity initialization cloned the first design's joint limits before
        # field expansion. Refresh those caches from the heterogeneous model.
        joint_ids = [self.sim.mj_model.joint(f"robot/{name}").id for name in robot.joint_names]
        limits = self.sim.model.jnt_range[:, joint_ids].clone()
        robot.data.default_joint_pos_limits = limits.clone()
        robot.data.joint_pos_limits = limits.clone()
        center = limits.mean(dim=-1)
        half = 0.45 * (limits[..., 1] - limits[..., 0])
        robot.data.soft_joint_pos_limits = torch.stack((center - half, center + half), dim=-1)
        # Offsets clone default_joint_pos when the action manager is created.
        super().load_managers()


class CoDesignVecEnv(RslRlVecEnvWrapper):
    def step(self, actions):
        obs, reward, dones, extras = super().step(actions)
        # Native finite-horizon wrappers omit time_outs. Expose both causes
        # explicitly so final timeout never receives a physical fall penalty.
        extras["codesign_terminated"] = self.unwrapped.reset_terminated.clone()
        extras["time_outs"] = self.unwrapped.reset_time_outs.clone()
        return obs, reward, dones, extras

    @property
    def step_dt(self) -> float:
        return self.unwrapped.step_dt

    @property
    def horizon_steps(self) -> int:
        return self.unwrapped.horizon_steps


def build_env(designs: Sequence[Any], replicas: int, *, device: str,
              episode_length_s: float = 20.0, seed: int = 0,
              env_kwargs: dict | None = None, reward_overrides: dict | None = None):
    cohort = prepare_cohort(designs, replicas)
    cfg, entities, heights = make_codesign_env_cfg(
        cohort, episode_length_s=episode_length_s, seed=seed,
        env_kwargs=env_kwargs, reward_overrides=reward_overrides)
    if (env_kwargs or {}).get("verbose", False):
        return CoDesignVecEnv(CoDesignEnv(cfg, cohort, entities, heights, device=device,
                                         render_mode=(env_kwargs or {}).get("render_mode")), clip_actions=1.0)
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            env = CoDesignVecEnv(CoDesignEnv(cfg, cohort, entities, heights, device=device,
                                         render_mode=(env_kwargs or {}).get("render_mode")), clip_actions=1.0)
    except BaseException:
        sys.stderr.write(output.getvalue())
        raise
    _LOG.debug("Native cohort initialization:\n%s", output.getvalue())
    return env
