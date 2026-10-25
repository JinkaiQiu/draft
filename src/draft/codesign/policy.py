"""Episode-level morphology decisions and the native RSL control models."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.distributions import Normal

from .space import DesignSpace


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


class TopologyHardwarePolicy(nn.Module):
    """One categorical topology and topology-conditioned raw Gaussian coordinates.

    Conditional angles for active joint groups are omitted from the probability
    measure. Raw actions still retain a fixed 14-column shape for PPO storage.
    """

    def __init__(self, space: DesignSpace, initial_std: float = 0.5,
                 hidden_dims: tuple[int, ...] = (64, 64),
                 initial_active_probability: float = 0.9):
        super().__init__()
        if not space.topology_enabled:
            raise ValueError("Topology hardware policy requires a topology-enabled design space")
        if isinstance(initial_std, bool) or not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("Hardware initial_std must be finite and positive")
        if isinstance(initial_active_probability, bool) or not math.isfinite(initial_active_probability) or not 0 < initial_active_probability < 1:
            raise ValueError("Initial active probability must lie strictly between zero and one")
        if not isinstance(hidden_dims, (tuple, list)) or not hidden_dims or any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in hidden_dims
        ):
            raise ValueError("Hardware hidden dimensions must be positive integers")
        self.space = space
        bits = torch.arange(64, dtype=torch.long)[:, None].bitwise_right_shift(torch.arange(6)).bitwise_and(1)
        self.register_buffer("topology_bits", bits.float(), persistent=False)
        prior = (bits.float() * math.log(initial_active_probability)
                 + (1 - bits.float()) * math.log1p(-initial_active_probability)).sum(-1)
        self.topology_logits = nn.Parameter(prior)
        layers: list[nn.Module] = []
        incoming = 6
        for width in hidden_dims:
            layers.extend((nn.Linear(incoming, width), nn.Tanh()))
            incoming = width
        output = nn.Linear(incoming, 26)
        nn.init.zeros_(output.weight)
        with torch.no_grad():
            initial = torch.as_tensor(space.normalize(space.center)[:7], dtype=torch.float32)
            initial = initial.clamp(-1. + 1e-6, 1. - 1e-6).atanh()
            output.bias[:7].copy_(initial)
            nominal = (0., .5, -1., 0., .5, -1.)
            angle_normalized = []
            for target, (lower, upper) in zip(nominal, space.lock_angle_bounds):
                margin = 1e-4 * (upper - lower)
                bounded = min(max(target, lower + margin), upper - margin)
                angle_normalized.append(2. * (bounded - lower) / (upper - lower) - 1.)
            output.bias[7:13].copy_(torch.tensor(angle_normalized, dtype=torch.float32).atanh())
            output.bias[13:].fill_(math.log(initial_std))
        layers.append(output)
        self.conditioner = nn.Sequential(*layers)
        self.baseline = nn.Parameter(torch.zeros(()))

    @property
    def dimensions(self) -> int:
        return 14

    @property
    def mean(self) -> torch.Tensor:
        return self.deterministic_raw()

    @property
    def log_std(self) -> torch.Tensor:
        # Diagnostics in decoded-coordinate order; masks have no Gaussian scale.
        index = self.topology_logits.argmax()
        _, scales = self._conditional(self.topology_bits[index])
        return torch.cat((scales[:7], torch.zeros_like(scales[:6]), scales[7:]))

    def _conditional(self, bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = self.conditioner(bits)
        means, log_stds = parameters[..., :13], parameters[..., 13:]
        if not torch.isfinite(parameters).all():
            raise ValueError("Hardware conditional Gaussian parameters must be finite")
        # The raw Gaussian is unbounded. Bound log scale only to avoid numerical
        # overflow in exp and the KL while keeping its gradient well defined.
        return means, log_stds.clamp(-20., 5.)

    def _validate_raw(self, raw: torch.Tensor, *, batch: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.ndim != (2 if batch else 1) or raw.shape[-1] != 14 or (batch and raw.shape[0] < 1):
            raise ValueError("Topology raw actions require 14 columns")
        if not torch.isfinite(raw).all():
            raise ValueError("Topology raw actions must be finite")
        index = raw[..., 0]
        if ((index < 0) | (index > 63) | (index != index.floor())).any():
            raise ValueError("Topology indices must be integers from 0 to 63")
        return index.long(), raw[..., 1:]

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        index, coordinates = self._validate_raw(raw, batch=raw.ndim == 2)
        bits = self.topology_bits[index]
        angles = coordinates[..., 7:].detach().tanh() * (1. - bits)
        return torch.cat((coordinates[..., :7].detach().tanh(), bits, angles), dim=-1)

    @torch.no_grad()
    def deterministic_raw(self) -> torch.Tensor:
        index = self.topology_logits.argmax()
        means, _ = self._conditional(self.topology_bits[index])
        means = means.clone()
        means[7:] *= 1. - self.topology_bits[index]
        return torch.cat((index.to(means.dtype).unsqueeze(0), means))

    @torch.no_grad()
    def fixed_raw(self, count: int) -> torch.Tensor:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Design count must be a positive integer")
        normalized = torch.as_tensor(self.space.normalize(self.space.center)[:7],
                                     device=self.topology_logits.device, dtype=self.topology_logits.dtype)
        row = torch.cat((self.topology_logits.new_tensor([63.]),
                         normalized.clamp(-1. + 1e-6, 1. - 1e-6).atanh(),
                         self.topology_logits.new_zeros(6)))
        return row.expand(count, -1).clone()

    @torch.no_grad()
    def sample(self, count: int, generator: torch.Generator | None = None):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Design count must be a positive integer")
        probabilities = self.topology_logits.softmax(-1)
        index = torch.multinomial(probabilities, count, replacement=True, generator=generator)
        bits = self.topology_bits[index]
        means, log_stds = self._conditional(bits)
        noise = torch.randn((count, 13), device=means.device, dtype=means.dtype, generator=generator)
        coordinates = means + noise * log_stds.exp()
        coordinates[:, 7:] *= 1. - bits
        raw = torch.cat((index[:, None].to(coordinates.dtype), coordinates), dim=-1)
        logprob, _, baseline = self.evaluate(raw)
        return raw, logprob, self.decode(raw), baseline.clone()

    def evaluate(self, raw: torch.Tensor):
        index, coordinates = self._validate_raw(raw, batch=True)
        bits = self.topology_bits[index]
        means, log_stds = self._conditional(bits)
        normal = Normal(means, log_stds.exp())
        mask = torch.cat((torch.ones((raw.shape[0], 7), device=bits.device, dtype=bits.dtype),
                          1. - bits), dim=-1)
        scored = torch.where(mask.bool(), coordinates.detach(), means.detach())
        logprob = self.topology_logits.log_softmax(-1)[index] + (normal.log_prob(scored) * mask).sum(-1)
        probability = self.topology_logits.softmax(-1)
        entropy = -(probability * self.topology_logits.log_softmax(-1)).sum()
        _, all_log_stds = self._conditional(self.topology_bits)
        all_entropy = all_log_stds + .5 * math.log(2 * math.pi * math.e)
        all_mask = torch.cat((torch.ones((64, 7), device=bits.device, dtype=bits.dtype),
                              1. - self.topology_bits), dim=-1)
        entropy = entropy + (probability * (all_entropy * all_mask).sum(-1)).sum()
        return logprob, entropy.expand(raw.shape[0]), self.baseline.expand(raw.shape[0])

    def distribution_state(self) -> dict[str, torch.Tensor]:
        means, log_stds = self._conditional(self.topology_bits)
        return {"logits": self.topology_logits.detach().clone(),
                "means": means.detach().clone(), "log_stds": log_stds.detach().clone()}

    def kl_from_state(self, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
        old_logits = state["logits"].to(self.topology_logits)
        old_means = state["means"].to(self.topology_logits)
        old_log_stds = state["log_stds"].to(self.topology_logits)
        if old_logits.shape != (64,) or old_means.shape != (64, 13) or old_log_stds.shape != (64, 13):
            raise ValueError("Invalid topology distribution state")
        means, log_stds = self._conditional(self.topology_bits)
        old_prob = old_logits.softmax(-1)
        categorical_kl = (old_prob * (old_logits.log_softmax(-1)
                                      - self.topology_logits.log_softmax(-1))).sum()
        old_var = (2 * old_log_stds).exp()
        new_var = (2 * log_stds).exp()
        gaussian_kl = log_stds - old_log_stds + (old_var + (old_means - means).square()) / (2 * new_var) - .5
        mask = torch.cat((torch.ones((64, 7), device=means.device, dtype=means.dtype),
                          1. - self.topology_bits), dim=-1)
        return categorical_kl + (old_prob * (gaussian_kl * mask).sum(-1)).sum()


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
            if getattr(self.actor, "joint_topology", False):
                actions = actions * self.actor._hardware_action_mask
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
