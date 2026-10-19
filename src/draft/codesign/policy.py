"""Episode-level morphology decisions and the native RSL control models."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.distributions import Normal


class HardwarePolicy(nn.Module):
    """A raw Gaussian action, fixed tanh decoder, and scalar pre-draw baseline.

    Likelihoods always describe raw actions. The compiler and decoder are never
    differentiated, so clipping, rejection, and repeated physical designs do not
    change the probability measure used by PPO.
    """

    def __init__(self, dimensions: int = 7, initial_std: float = 0.5, initial_normalized=None):
        super().__init__()
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("Hardware dimensions must be a positive integer")
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("Hardware initial_std must be finite and positive")
        initial = torch.zeros(dimensions) if initial_normalized is None else torch.as_tensor(initial_normalized, dtype=torch.float32)
        if initial.shape != (dimensions,) or not torch.isfinite(initial).all() or (initial.abs() >= 1).any():
            raise ValueError("Initial normalized design must lie strictly inside (-1, 1)")
        self.mean = nn.Parameter(initial.atanh())
        self.log_std = nn.Parameter(torch.full((dimensions,), math.log(initial_std)))
        self.baseline = nn.Parameter(torch.zeros(()))

    @property
    def dimensions(self) -> int:
        return self.mean.numel()

    def distribution(self) -> Normal:
        std = self.log_std.exp()
        if not torch.isfinite(self.mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Hardware Gaussian parameters must be finite with positive scale")
        return Normal(self.mean, std)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim not in (1, 2) or raw.shape[-1] != self.dimensions or not torch.isfinite(raw).all():
            raise ValueError("Raw hardware actions have an invalid shape or nonfinite values")
        return raw.detach().tanh()

    @torch.no_grad()
    def sample(self, count: int, generator: torch.Generator | None = None):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Design count must be a positive integer")
        dist = self.distribution()
        noise = torch.randn((count, self.dimensions), device=self.mean.device,
                            dtype=self.mean.dtype, generator=generator)
        raw = dist.mean + dist.stddev * noise
        return raw, dist.log_prob(raw).sum(-1), self.decode(raw), self.baseline.expand(count).clone()

    def evaluate(self, raw: torch.Tensor):
        if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != self.dimensions or not torch.isfinite(raw).all():
            raise ValueError("Hardware actions must be a finite [designs, dimensions] matrix")
        dist = self.distribution()
        return (dist.log_prob(raw.detach()).sum(-1), dist.entropy().sum().expand(raw.shape[0]),
                self.baseline.expand(raw.shape[0]))


class NativeControlPolicy(nn.Module):
    """Adapter retaining Draft's native RSL actor, critic and normalizers."""

    def __init__(self, actor: nn.Module, critic: nn.Module, normalization_exclusions=None):
        super().__init__()
        if getattr(actor, "is_recurrent", False) or getattr(critic, "is_recurrent", False):
            raise ValueError("Co-design currently requires feed-forward control models")
        self.actor, self.critic = actor, critic
        self.normalization_exclusions = {}
        for name, indices in (normalization_exclusions or {}).items():
            if name not in ("actor", "critic"):
                raise ValueError(f"Unknown control normalization group: {name}")
            model = getattr(self, name)
            resolved = tuple(index if index >= 0 else model.obs_dim + index for index in indices)
            if any(index < 0 or index >= model.obs_dim for index in resolved):
                raise ValueError(f"Normalization exclusion outside {name} input dimensions")
            self.normalization_exclusions[name] = resolved
        self.preserve_conditioning_scale()

    @classmethod
    def from_observation(cls, observation: Mapping[str, torch.Tensor], num_actions: int,
                         actor_cfg: dict, critic_cfg: dict, obs_groups: dict | None = None,
                         normalization_exclusions: dict | None = None):
        from rsl_rl.models import MLPModel
        from rsl_rl.utils import resolve_callable

        groups = obs_groups or {"actor": ["actor"], "critic": ["critic"]}
        configs = [copy.deepcopy(actor_cfg), copy.deepcopy(critic_cfg)]
        models = []
        for config in configs:
            model = resolve_callable(config.pop("class_name", "MLPModel"))
            if not isinstance(model, type) or not issubclass(model, MLPModel):
                raise ValueError(f"Unsupported co-design control model: {model}")
            models.append(model)
        return cls(models[0](observation, groups, "actor", num_actions, **configs[0]),
                   models[1](observation, groups, "critic", 1, **configs[1]),
                   normalization_exclusions)

    def evaluate(self, observation, actions=None):
        # Update the native distribution without sampling when scoring actions.
        self.actor.distribution.update(self.actor.mlp(self.actor.get_latent(observation)))
        if actions is None:
            actions = self.actor.distribution.sample()
        return (actions, self.actor.get_output_log_prob(actions), self.actor.output_entropy,
                self.critic(observation).squeeze(-1))

    @torch.no_grad()
    def deterministic(self, observation):
        return self.actor(observation)

    @torch.no_grad()
    def update_normalization(self, observation):
        self.actor.update_normalization(observation)
        self.critic.update_normalization(observation)
        self.preserve_conditioning_scale()

    @torch.no_grad()
    def preserve_conditioning_scale(self):
        """Keep bounded conditioning inputs in their declared physical encoding.

        RSL's native normalizer uses ``(x - mean) / (std + eps)``. Pinning these
        coordinates to identity retains its checkpoint/export format and avoids
        amplifying unseen morphologies after fixed-hardware pretraining.
        """
        for name, indices in self.normalization_exclusions.items():
            model = getattr(self, name)
            if model.obs_normalization and indices:
                normalizer = model.obs_normalizer
                # Native PPO replaces buffers such as _std while collecting in
                # inference mode. Make them writable at stage/checkpoint
                # boundaries without carrying inference tensors into restores.
                if not torch.is_inference_mode_enabled():
                    for field in ("_mean", "_var", "_std"):
                        value = getattr(normalizer, field)
                        if torch.is_inference(value):
                            setattr(normalizer, field, value.clone())
                normalizer._mean[:, indices] = 0.
                normalizer._var[:, indices] = (1. - normalizer.eps) ** 2
                normalizer._std[:, indices] = 1. - normalizer.eps


class JointPolicy(nn.Module):
    def __init__(self, control: nn.Module | None, hardware: HardwarePolicy | None = None):
        super().__init__()
        self.control = control
        self.hardware = hardware if hardware is not None else HardwarePolicy()
