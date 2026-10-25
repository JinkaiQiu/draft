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

    def __init__(self, obs, obs_groups, obs_set, output_dim, *args, conditioning_dims: int = 7,
                 joint_topology: bool = False, **kwargs):
        if not isinstance(joint_topology, bool):
            raise ValueError("joint_topology must be a boolean")
        if joint_topology and conditioning_dims != 19:
            raise ValueError("Joint topology requires 19 trailing hardware features")
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
        self.joint_topology = joint_topology
        self._hardware_action_mask = None
        if joint_topology and self.distribution is not None and output_dim != 12:
            raise ValueError("Joint topology requires the native 12-action controller")
        self.preserve_conditioning_scale()
        self.register_load_state_dict_pre_hook(self._prepare_normalizer_restore)
        self.register_load_state_dict_post_hook(self._restore_conditioning_scale)

    @torch.no_grad()
    def preserve_conditioning_scale(self):
        if self.obs_normalization:
            normalizer = self.obs_normalizer
            if not torch.is_inference_mode_enabled():
                # Native collection may replace _std with an inference tensor.
                # Checkpoint loading also invokes this hook outside collection.
                for field in ("_mean", "_var", "_std"):
                    value = getattr(normalizer, field)
                    if torch.is_inference(value):
                        setattr(normalizer, field, value.clone())
            normalizer._mean[:, -self.conditioning_dims:] = 0.
            normalizer._std[:, -self.conditioning_dims:] = 1. - normalizer.eps
            normalizer._var[:, -self.conditioning_dims:] = (1. - normalizer.eps) ** 2

    def _restore_conditioning_scale(self, module, incompatible_keys):
        self.preserve_conditioning_scale()

    def _prepare_normalizer_restore(self, module, *args):
        self.preserve_conditioning_scale()

    def update_normalization(self, obs):
        super().update_normalization(obs)
        self.preserve_conditioning_scale()

    def get_latent(self, obs, masks=None, hidden_state=None):
        if getattr(self, "joint_topology", False) and self.distribution is not None:
            paired = obs[self.obs_groups[-1]][..., -12:-6]
            order = torch.tensor((0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5), device=paired.device)
            self._hardware_action_mask = paired.index_select(-1, order).detach()
        return super().get_latent(obs, masks, hidden_state)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        result = super().forward(obs, masks, hidden_state, stochastic_output)
        if self.joint_topology and self.distribution is not None:
            result = result * self._hardware_action_mask
        return result

    def get_output_log_prob(self, outputs):
        if not self.joint_topology:
            return super().get_output_log_prob(outputs)
        mean, std = self.distribution.params
        return (torch.distributions.Normal(mean, std).log_prob(outputs) * self._hardware_action_mask).sum(-1)

    @property
    def output_entropy(self):
        if not self.joint_topology:
            return super().output_entropy
        mean, std = self.distribution.params
        return (torch.distributions.Normal(mean, std).entropy() * self._hardware_action_mask).sum(-1)

    def get_kl_divergence(self, old_params, new_params):
        if not self.joint_topology:
            return super().get_kl_divergence(old_params, new_params)
        old, new = (torch.distributions.Normal(*params) for params in (old_params, new_params))
        return (torch.distributions.kl_divergence(old, new) * self._hardware_action_mask).sum(-1)
