"""Native reset and reward terms for a cohort with locked quadruped joints."""

from __future__ import annotations

import torch

from mjlab.tasks.velocity import mdp

from draft.codesign.batch import JOINT_NAMES


def paired_encoder_bias(env, env_ids, bias_range, asset_cfg):
    """Sample one calibration error per mirrored front/rear joint pair."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    low, high = map(float, bias_range)
    pairs = low + (high - low) * torch.rand((len(env_ids), 6), device=env.device)
    reflected = torch.cat((pairs[:, :3], pairs[:, :3] * pairs.new_tensor([-1, 1, 1]),
                           pairs[:, 3:], pairs[:, 3:] * pairs.new_tensor([-1, 1, 1])), dim=1)
    order = [JOINT_NAMES.index(name) for name in robot.joint_names]
    robot.data.encoder_bias[env_ids] = reflected[:, order]


def reset_joints_topology(env, env_ids, position_range, velocity_range, asset_cfg):
    """Use the authored reset, then pin frozen coordinates without reset noise."""
    mdp.reset_joints_by_offset(env, env_ids, position_range, velocity_range, asset_cfg)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[asset_cfg.name]
    active = env.hardware_joint_mask[env_ids]
    pos = torch.where(active, robot.data.joint_pos[env_ids],
                      robot.data.default_joint_pos[env_ids])
    vel = torch.where(active, robot.data.joint_vel[env_ids],
                      torch.zeros_like(robot.data.joint_vel[env_ids]))
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


def active_action_rate_l2(env):
    difference = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(difference) * env.hardware_action_mask, dim=1)


def active_joint_pos_limits(env):
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits
    pos = robot.data.joint_pos
    low = (limits[..., 0] - pos).clamp(min=0)
    high = (pos - limits[..., 1]).clamp(min=0)
    return torch.sum((low + high) * env.hardware_joint_mask, dim=1)
