"""Finite-window controller importance correction for co-adaptation scores."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn


class WindowTrajectoryRecorder:
    """Borrowed native transitions copied to bounded storage for one window.

    A reset starts another episode in the *same* hardware scoring window. Thus
    done flags only determine the terminal penalty; they never mask a return.
    """

    def __init__(self, count: int, steps: int, device="cpu", terminal_penalty=0.):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        self.count, self.steps = count, steps
        self.device = torch.device(device)
        self.terminal_penalty = float(terminal_penalty)
        if not math.isfinite(self.terminal_penalty):
            raise ValueError("terminal_penalty must be finite")
        self._observations, self._actions, self._logprobs, self._rewards = [], [], [], []
        self._pending = False
        self.observations = self.actions = self.behavior_logprobs = self.rewards = None

    def before_step(self, observation, actions, behavior_logprob):
        if self.observations is not None or self._pending or len(self._rewards) >= self.steps:
            raise RuntimeError("Trajectory step is pending or the window is complete")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        obs = {key: value.detach().to(self.device).clone() for key, value in observation.items()}
        if not obs or any(value.shape[0] != self.count or not torch.isfinite(value).all()
                          for value in obs.values()):
            raise ValueError("Invalid trajectory observations")
        if self._observations and obs.keys() != self._observations[0].keys():
            raise ValueError("Trajectory observation schema changed")
        action = actions.detach().to(self.device).clone()
        logprob = behavior_logprob.detach().reshape(-1).to(self.device).clone()
        if action.shape[0] != self.count or logprob.shape != (self.count,) or not (
                torch.isfinite(action).all() and torch.isfinite(logprob).all()):
            raise ValueError("Invalid trajectory action or behavior likelihood")
        self._observations.append(obs)
        self._actions.append(action)
        self._logprobs.append(logprob)
        self._pending = True

    def after_step(self, rewards, dones, extras):
        if not self._pending:
            raise RuntimeError("A pre-action transition is required")
        reward = rewards.detach().reshape(-1).to(self.device).clone()
        done = dones.detach().reshape(-1).to(self.device).bool()
        timeout = torch.as_tensor(extras.get("time_outs", torch.zeros_like(done)), device=self.device).reshape(-1).bool()
        terminated = torch.as_tensor(extras.get("codesign_terminated", done & ~timeout),
                                     device=self.device).reshape(-1).bool()
        if any(value.shape != (self.count,) for value in (reward, done, timeout, terminated)) or not torch.isfinite(reward).all():
            raise ValueError("Invalid trajectory reward or reset flag")
        self._rewards.append(reward + terminated * self.terminal_penalty)
        self._pending = False

    def finalize(self):
        if self.observations is not None:
            return self
        if self._pending or len(self._rewards) != self.steps:
            raise ValueError("Trajectory must contain the complete window")
        if self.observations is None:
            self.observations = {key: torch.stack([obs[key] for obs in self._observations])
                                 for key in self._observations[0]}
            self.actions = torch.stack(self._actions)
            self.behavior_logprobs = torch.stack(self._logprobs)
            self.rewards = torch.stack(self._rewards)
            self._observations.clear()
            self._actions.clear()
            self._logprobs.clear()
            self._rewards.clear()
        return self

    @property
    def storage_bytes(self):
        if self.observations is not None:
            tensors = [*self.observations.values(), self.actions, self.behavior_logprobs, self.rewards]
        else:
            tensors = [tensor for obs in self._observations for tensor in obs.values()]
            tensors += [*self._actions, *self._logprobs, *self._rewards]
        return sum(t.numel() * t.element_size() for t in tensors)


@torch.no_grad()
def finite_window_vtrace(rewards, values, log_ratios, gamma=1., rho_clip=1., c_clip=1.):
    """Return corrected values at every time, with V[T]=0 and no reset masks."""
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 1 or values.shape != rewards.shape or log_ratios.shape != rewards.shape:
        raise ValueError("V-trace inputs must all have shape [time, worlds]")
    if not all(torch.isfinite(value).all() for value in (rewards, values, log_ratios)):
        raise ValueError("V-trace inputs must be finite")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (gamma, rho_clip, c_clip)) or not 0 <= gamma <= 1 or rho_clip <= 0 or not 0 < c_clip <= min(1., rho_clip):
        raise ValueError("Invalid V-trace discount or importance caps")
    rho = log_ratios.clamp(max=math.log(rho_clip)).exp()
    c = log_ratios.clamp(max=math.log(c_clip)).exp()
    result = torch.empty_like(rewards)
    next_value = torch.zeros_like(rewards[0])
    next_corrected = next_value
    for t in range(rewards.shape[0] - 1, -1, -1):
        current = values[t]
        corrected = current + rho[t] * (rewards[t] + gamma * next_value - current)
        corrected = corrected + gamma * c[t] * (next_corrected - next_value)
        result[t] = corrected
        next_value, next_corrected = current, corrected
    return result


class _FiniteWindowValue(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension + 1, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, latent, remaining_fraction, remaining_steps):
        value = self.net(torch.cat((latent, remaining_fraction), dim=-1)).squeeze(-1)
        return value * remaining_steps


class HardwareVTraceScorer:
    """Approximate latest-controller scores with an independent window critic.

    The actor and its normalizer stay fixed throughout ``score``. The native
    episode critic supplies normalized inputs only: its value output, discount
    and optimizer are never used. Clipping and fitting on the same data make
    this a surrogate, not an unbiased estimate of final-controller performance.
    """

    DEFAULTS = dict(rho_clip=1., c_clip=1., value_epochs=3, batch_size=4096,
                    learning_rate=1e-3)

    @classmethod
    def validate_config(cls, config=None):
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise ValueError("V-trace config must be a mapping")
        unknown = set(config) - set(cls.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown V-trace settings: {sorted(unknown)}")
        result = dict(cls.DEFAULTS, **config)
        if (any(isinstance(result[k], bool) or not isinstance(result[k], (int, float)) or not math.isfinite(result[k])
                for k in ("rho_clip", "c_clip", "learning_rate")) or
                result["rho_clip"] <= 0 or not 0 < result["c_clip"] <= min(1., result["rho_clip"]) or
                result["learning_rate"] <= 0):
            raise ValueError("Invalid V-trace importance caps or critic learning rate")
        if any(isinstance(result[k], bool) or not isinstance(result[k], int) or result[k] < 1
               for k in ("value_epochs", "batch_size")):
            raise ValueError("V-trace value_epochs and batch_size must be positive integers")
        return result

    def __init__(self, control, config=None, device="cpu", seed=0):
        self.control, self.device = control, torch.device(device)
        self.config = self.validate_config(config)
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.value_model = None
        self.optimizer = None
        self.input_dimension = None
        self.trained_windows = 0
        self._pending_state = None

    def _initialize(self, dimension):
        if self.value_model is not None:
            if dimension != self.input_dimension:
                raise ValueError("V-trace critic observation dimension changed")
            return
        self.input_dimension = dimension
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            self.value_model = _FiniteWindowValue(dimension).to(self.device)
        self.optimizer = torch.optim.Adam(self.value_model.parameters(), lr=self.config["learning_rate"])
        if self._pending_state is not None:
            self.value_model.load_state_dict(self._pending_state["value_model"])
            self.optimizer.load_state_dict(self._pending_state["optimizer"])
            self._pending_state = None

    def _batch_observation(self, trajectory, indices):
        size = trajectory.steps * trajectory.count
        return {key: value.reshape(size, *value.shape[2:])[indices].to(self.device)
                for key, value in trajectory.observations.items()}

    def _latent(self, observation):
        return self.control.critic.get_latent(observation).detach()

    def _fit_value(self, trajectory, targets):
        targets = targets.detach().reshape(-1)
        total = targets.numel()
        loss_sum = 0.
        batches = 0
        self.value_model.train()
        for indices in torch.randperm(total, generator=self.generator).split(self.config["batch_size"]):
            obs = self._batch_observation(trajectory, indices)
            with torch.no_grad():
                latent = self._latent(obs)
            remaining = trajectory.steps - torch.div(indices, trajectory.count, rounding_mode="floor")
            fraction = (remaining.float() / trajectory.steps).to(self.device).unsqueeze(-1)
            predicted = self.value_model(latent, fraction, remaining.to(self.device).float())
            loss = torch.nn.functional.mse_loss(predicted, targets[indices].to(self.device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite V-trace critic loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 10., error_if_nonfinite=True)
            self.optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        self.value_model.eval()
        return loss_sum / batches

    def _estimate(self, trajectory, *, likelihoods=True):
        total = trajectory.steps * trajectory.count
        values = torch.empty(total, dtype=trajectory.rewards.dtype)
        log_ratios = torch.empty_like(values) if likelihoods else None
        self.value_model.eval()
        actor = self.control.actor
        with torch.no_grad():
            for start in range(0, total, self.config["batch_size"]):
                stop = min(start + self.config["batch_size"], total)
                indices = slice(start, stop)
                obs = self._batch_observation(trajectory, indices)
                latent = self._latent(obs)
                remaining = trajectory.steps - torch.arange(start, stop) // trajectory.count
                fraction = (remaining.float() / trajectory.steps).to(self.device).unsqueeze(-1)
                values[indices] = self.value_model(latent, fraction, remaining.to(self.device).float()).cpu()
                if likelihoods:
                    action = trajectory.actions.reshape(total, *trajectory.actions.shape[2:])[indices].to(self.device)
                    actor.distribution.update(actor.mlp(actor.get_latent(obs)))
                    target_logprob = actor.get_output_log_prob(action).reshape(-1)
                    behavior = trajectory.behavior_logprobs.reshape(-1)[indices].to(self.device)
                    log_ratios[indices] = (target_logprob - behavior).cpu()
        return (values.reshape(trajectory.steps, trajectory.count),
                log_ratios.reshape(trajectory.steps, trajectory.count) if likelihoods else None)

    def score(self, trajectory: WindowTrajectoryRecorder, gamma=1.):
        trajectory.finalize()
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        actor, critic = self.control.actor, self.control.critic
        actor_mode, critic_mode = actor.training, critic.training
        actor.eval()
        critic.eval()
        try:
            with torch.no_grad():
                sample = self._batch_observation(trajectory, slice(0, 1))
                self._initialize(self._latent(sample).shape[-1])
            # One Monte Carlo warmup only for the first window. Subsequent
            # targets come from the final actor's frozen likelihood ratios.
            if self.trained_windows == 0:
                warmup = torch.empty_like(trajectory.rewards)
                tail = torch.zeros_like(warmup[0])
                for t in range(trajectory.steps - 1, -1, -1):
                    tail = trajectory.rewards[t] + gamma * tail
                    warmup[t] = tail
                self._fit_value(trajectory, warmup)
            values, log_ratios = self._estimate(trajectory)
            for _ in range(self.config["value_epochs"]):
                targets = finite_window_vtrace(trajectory.rewards.cpu(), values, log_ratios,
                                               gamma, self.config["rho_clip"], self.config["c_clip"])
                value_loss = self._fit_value(trajectory, targets)
                values, _ = self._estimate(trajectory, likelihoods=False)
            corrected = finite_window_vtrace(trajectory.rewards.cpu(), values, log_ratios,
                                              gamma, self.config["rho_clip"], self.config["c_clip"])
            self.trained_windows += 1
            raw = torch.zeros(trajectory.count)
            for t in range(trajectory.steps - 1, -1, -1):
                raw = trajectory.rewards[t].cpu() + gamma * raw
            logw = log_ratios.reshape(-1).double()
            weights = (logw - logw.max()).exp()
            ess = weights.sum().square() / weights.square().sum()
            metrics = dict(vtrace_raw_mean=float(raw.mean()), vtrace_score_mean=float(corrected[0].mean()),
                           vtrace_score_delta=float((corrected[0] - raw).mean()),
                           vtrace_rho_clipped_fraction=float((log_ratios > math.log(self.config["rho_clip"])).float().mean()),
                           vtrace_c_clipped_fraction=float((log_ratios > math.log(self.config["c_clip"])).float().mean()),
                           vtrace_action_ess_fraction=float(ess / logw.numel()),
                           vtrace_mean_abs_log_ratio=float(log_ratios.abs().mean()),
                           vtrace_value_loss=value_loss, vtrace_value_fit_in_sample=1,
                           vtrace_storage_bytes=trajectory.storage_bytes,
                           vtrace_trained_windows=self.trained_windows)
            return corrected[0].to(self.device), metrics
        finally:
            actor.train(actor_mode)
            critic.train(critic_mode)

    def state_dict(self):
        return dict(version=1, config=copy.deepcopy(self.config), seed=self.seed,
                    input_dimension=self.input_dimension, trained_windows=self.trained_windows,
                    generator_state=self.generator.get_state(),
                    value_model=(self.value_model.state_dict() if self.value_model is not None else
                                 self._pending_state["value_model"] if self._pending_state is not None else None),
                    optimizer=(self.optimizer.state_dict() if self.optimizer is not None else
                               self._pending_state["optimizer"] if self._pending_state is not None else None))

    def load_state_dict(self, state):
        if state.get("version") != 1 or state["config"] != self.config or state["seed"] != self.seed:
            raise ValueError("V-trace checkpoint configuration changed")
        self.generator.set_state(state["generator_state"].cpu())
        self.trained_windows = state["trained_windows"]
        dimension = state["input_dimension"]
        if dimension is None:
            if state["value_model"] is not None or state["optimizer"] is not None:
                raise ValueError("Invalid uninitialized V-trace checkpoint")
        elif self.value_model is None:
            self._pending_state = copy.deepcopy(state)
            self.input_dimension = dimension
        else:
            self._initialize(dimension)
            self.value_model.load_state_dict(state["value_model"])
            self.optimizer.load_state_dict(state["optimizer"])
