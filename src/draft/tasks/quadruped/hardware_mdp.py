"""Terrain-relative height terms scaled by each world's nominal stance height."""

import torch

from mjlab.managers.manager_base import ManagerTermBase


class _TerrainRelativeHeight(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.sensor_name = cfg.params.get("sensor_name", "foot_height_scan")
        self.asset_name = cfg.params.get("asset_name", "robot")
        root = env.scene[self.asset_name].data.root_link_pos_w
        heights = torch.as_tensor(cfg.params["nominal_heights"], dtype=root.dtype, device=root.device)
        if heights.ndim == 0:
            heights = heights.expand(env.num_envs)
        if heights.shape != (env.num_envs,) or not torch.isfinite(heights).all() or (heights <= 0).any():
            raise ValueError("nominal_heights must be a positive finite scalar or one value per world")
        self.nominal_heights = heights.clone()

    def _clearance(self, env):
        """Estimate support terrain from valid foot-ring hits, never foot height.

        A native TerrainHeightSensor reports site_z - hit_z, with its default
        min reduction selecting the highest ground in each foot's ring. Its
        fabricated all-miss clearance cannot establish actual terrain support.
        Reconstruct that reduction from raw upward-facing hits instead, ignoring
        missing/nonfinite rays and backfaces (ray origin inside terrain). Average
        the supported foot-ring elevations, giving each foot equal weight.

        Partial ring/foot misses use remaining support. With no valid support,
        or a nonfinite root position, fail closed: zero reward and termination.
        """
        root = env.scene[self.asset_name].data.root_link_pos_w
        data = env.scene[self.sensor_name].data
        frames = data.frame_pos_w
        if frames.ndim != 3 or frames.shape[0] != env.num_envs or frames.shape[-1] != 3 or frames.shape[1] < 1:
            raise ValueError("height terms require per-foot frame poses and matching raw terrain rays")
        count, feet = frames.shape[:2]
        distances, hits, normals = data.distances, data.hit_pos_w, data.normals_w
        if (distances.ndim != 2 or distances.shape[0] != count or distances.shape[1] < feet or
                distances.shape[1] % feet or hits.shape != (*distances.shape, 3) or
                normals.shape != hits.shape):
            raise ValueError("height terms require per-foot frame poses and matching raw terrain rays")
        valid = (torch.isfinite(distances) & (distances >= 0) &
                 torch.isfinite(hits).all(-1) & torch.isfinite(normals).all(-1) & (normals[..., 2] > 0))
        valid = valid.reshape(count, feet, -1) & torch.isfinite(frames).all(-1, keepdim=True)
        terrain = hits[..., 2].reshape_as(valid)
        highest = torch.where(valid, terrain, -torch.inf).amax(-1)
        supported = valid.any(-1)
        support_count = supported.sum(-1)
        terrain_z = torch.where(supported, highest, 0.).sum(-1) / support_count.clamp_min(1)
        observed = (support_count > 0) & torch.isfinite(root).all(-1)
        clearance = torch.where(observed, root[:, 2] - terrain_z, 0.)
        return clearance, observed


class TerrainRelativeHeightReward(_TerrainRelativeHeight):
    """Unit height reward; set the reward-manager weight to 0.4."""

    def __call__(self, env, nominal_heights, sensor_name="foot_height_scan", asset_name="robot"):
        clearance, observed = self._clearance(env)
        reward = torch.exp(-torch.square((clearance - .95 * self.nominal_heights) /
                                        (.25 * self.nominal_heights)))
        return torch.where(observed, reward, 0.)


class TerrainRelativeCollapse(_TerrainRelativeHeight):
    """Collapse below half nominal height, or no observable support terrain."""

    def __call__(self, env, nominal_heights, sensor_name="foot_height_scan", asset_name="robot"):
        clearance, observed = self._clearance(env)
        return ~observed | (clearance < .5 * self.nominal_heights)
