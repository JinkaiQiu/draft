"""Native RSL model with bounded morphology inputs kept in their original units."""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel


class ConditionedMLPModel(MLPModel):
    """Keep trailing morphology features at identity scale within native PPO.

    Network structure, distribution, normalizer state keys, and the native PPO
    update loop remain unchanged. Only the declared trailing input coordinates
    bypass empirical centering/scaling, including after checkpoint loading.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, *args, conditioning_dims: int = 7, **kwargs):
        if isinstance(conditioning_dims, bool) or not isinstance(conditioning_dims, int) or conditioning_dims < 1:
            raise ValueError("conditioning_dims must be a positive integer")
        last_group = obs_groups[obs_set][-1]
        if obs[last_group].shape[-1] < conditioning_dims:
            raise ValueError("The last observation group must contain the trailing conditioning coordinates")
        trimmed = {key: value for key, value in obs.items()}
        trimmed[last_group] = obs[last_group][..., :-conditioning_dims]
        if sum(trimmed[group].shape[-1] for group in obs_groups[obs_set]) < 1:
            raise ValueError("Conditioned models require at least one ordinary state coordinate")
        # Initialize exactly the native state-only network, then append zero
        # input columns without any further random draws or fan-in rescaling.
        super().__init__(trimmed, obs_groups, obs_set, output_dim, *args, **kwargs)
        first = next(layer for layer in self.mlp if isinstance(layer, torch.nn.Linear))
        zeros = first.weight.new_zeros((first.out_features, conditioning_dims))
        first.weight = torch.nn.Parameter(torch.cat((first.weight.detach(), zeros), dim=1))
        first.in_features += conditioning_dims
        self.obs_dim += conditioning_dims
        if self.obs_normalization:
            normalizer = self.obs_normalizer
            for name in ("_mean", "_var", "_std"):
                previous = getattr(normalizer, name)
                extra = previous.new_zeros((1, conditioning_dims)) if name == "_mean" else previous.new_ones((1, conditioning_dims))
                setattr(normalizer, name, torch.cat((previous, extra), dim=-1))
        self.conditioning_dims = conditioning_dims
        self.preserve_conditioning_scale()
        self.register_load_state_dict_post_hook(self._restore_conditioning_scale)

    @torch.no_grad()
    def preserve_conditioning_scale(self):
        if self.obs_normalization:
            normalizer = self.obs_normalizer
            normalizer._mean[:, -self.conditioning_dims:] = 0.
            normalizer._std[:, -self.conditioning_dims:] = 1. - normalizer.eps
            normalizer._var[:, -self.conditioning_dims:] = (1. - normalizer.eps) ** 2

    def _restore_conditioning_scale(self, module, incompatible_keys):
        self.preserve_conditioning_scale()

    def update_normalization(self, obs):
        super().update_normalization(obs)
        self.preserve_conditioning_scale()
