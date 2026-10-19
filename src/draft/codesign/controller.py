"""Bounded native PPO controller training between frozen hardware evaluations."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Number
import time

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan, resolve_callable

from .policy import NativeControlPolicy


class _EpisodeMetrics:
    """Logging-only cohort state; retain O(worlds + metric keys), never extras."""

    def __init__(self, count, device):
        self.returns = torch.zeros(count, device=device)
        self.lengths = torch.zeros(count, device=device, dtype=torch.long)
        # Return sum, length sum, completed count; block and cohort totals.
        self.completed = torch.zeros(3, device=device, dtype=torch.float64)
        self.total_completed = torch.zeros_like(self.completed)
        self.reward_sum = torch.zeros((), device=device)
        self.total_reward_sum = torch.zeros_like(self.reward_sum)
        self.extras = {}

    def step(self, rewards, dones, extras):
        rewards, done = rewards.detach().reshape(-1), dones.detach().reshape(-1).bool()
        self.returns.add_(rewards)
        self.lengths.add_(1)
        count = done.sum()
        self.completed.add_(torch.stack(((self.returns * done).sum(),
                                         (self.lengths * done).sum(), count)))
        self.returns.masked_fill_(done, 0.)
        self.lengths.masked_fill_(done, 0)
        self.reward_sum.add_(rewards.sum())
        # Legacy episode extras take precedence if the same tag occurs in log.
        info = {}
        for source in ("log", "episode"):
            values = extras.get(source, {})
            if isinstance(values, Mapping):
                for key, value in values.items():
                    if isinstance(key, str) and isinstance(value, (Number, torch.Tensor)):
                        tag = key if "/" in key else "Episode/" + key
                        info[tag] = value, source == "episode"
        for tag, (value, episode) in info.items():
            value = torch.as_tensor(value, device=self.returns.device).detach()
            if not value.numel():
                continue
            if tag.startswith("Episode_Termination/"):
                # Native termination extras are reset counts, not episode means.
                amount, weight = torch.where(count > 0, value.sum(), 0.), None
            elif not tag.startswith("Metrics/") and (episode or tag.startswith(
                    ("Episode_Reward/", "Episode_Metrics/", "Episode/"))):
                amount, weight = torch.where(count > 0, value.float().mean() * count, 0.), count
            else:
                # Metrics/* may be emitted on every step, even without resets.
                amount, weight = value.float().sum(), value.numel()
            if tag not in self.extras:
                self.extras[tag] = [torch.zeros((), device=value.device, dtype=torch.float64),
                                    None if weight is None else torch.zeros((), device=value.device, dtype=torch.float64)]
            self.extras[tag][0].add_(amount)
            if weight is not None:
                self.extras[tag][1].add_(weight)

    def finish_block(self, writer, iteration, transitions):
        reward, length, count = self.completed.tolist()
        partial = self.partial()
        if writer is not None:
            writer.add_scalar("Train/completed_episodes", count, iteration)
            if count:
                writer.add_scalar("Train/mean_reward", reward / count, iteration)
                writer.add_scalar("Train/mean_episode_length", length / count, iteration)
            writer.add_scalar("Controller/mean_transition_reward", self.reward_sum.item() / transitions, iteration)
            for key, value in partial.items():
                writer.add_scalar("Controller/PartialEpisode/" + key, value, iteration)
            for tag, (amount, weight) in self.extras.items():
                denominator = weight.item() if weight is not None else 1.
                if denominator:
                    writer.add_scalar(tag, amount.item() / denominator, iteration)
        self.total_completed.add_(self.completed)
        self.total_reward_sum.add_(self.reward_sum)
        self.completed.zero_()
        self.reward_sum.zero_()
        self.extras.clear()

    def partial(self):
        active = self.lengths > 0
        count = int(active.sum())
        return dict(mean_return=(self.returns * active).sum().item() / max(count, 1),
                    mean_length=self.lengths.sum().item() / max(count, 1), count=count,
                    world_fraction=count / self.lengths.numel())

    def summary(self, transitions):
        reward, length, count = self.total_completed.tolist()
        metrics = dict(controller_completed_episode_count=int(count),
                       controller_mean_transition_reward=self.total_reward_sum.item() / transitions)
        if count:
            metrics.update(controller_mean_episode_reward=reward / count,
                           controller_mean_episode_length=length / count)
        metrics.update({"controller_partial_episode_" + key: value for key, value in self.partial().items()})
        return metrics


class NativeControllerTrainer:
    """Keep native PPO/Adam state across cohorts, retaining only one short rollout.

    Models belong to ``control`` and are never reconstructed. This trainer owns
    only their independent native PPO optimizer, counters, and temporary native
    rollout storage. Hardware policies never enter its optimizer.
    """

    VERSION = 1

    def __init__(self, control: NativeControlPolicy, train_cfg: dict, device):
        self.control = control.to(device)
        self.device = torch.device(device)
        self.config = copy.deepcopy(train_cfg)
        self.steps = self.config.get("num_steps_per_env", 24)
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("num_steps_per_env must be a positive integer")
        if getattr(control.actor, "is_recurrent", False) or getattr(control.critic, "is_recurrent", False):
            raise ValueError("Alternating controller training requires feed-forward native models")
        self.algorithm_config = copy.deepcopy(self.config["algorithm"])
        if resolve_callable(self.algorithm_config.pop("class_name", "PPO")) is not PPO:
            raise ValueError("NativeControllerTrainer requires the native RSL PPO algorithm")
        for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if self.algorithm_config.get(name) is not None:
                raise ValueError(f"Alternating controller training does not support {name}")
        if self.config.get("multi_gpu") or self.algorithm_config.pop("share_cnn_encoders", False):
            raise ValueError("Alternating controller training does not support multi-GPU or shared CNNs")
        if self.config.get("torch_compile_mode") is not None:
            raise ValueError("Alternating controller training currently requires uncompiled native models")
        self.alg: PPO | None = None
        self.iteration = 0
        self.total_transitions = 0
        self.schema = None
        self._pending_optimizer = None
        self._learning_rate = self.algorithm_config.get("learning_rate", .001)
        self._training = False

    @property
    def learning_rate(self):
        return self.alg.learning_rate if self.alg is not None else self._learning_rate

    def _time(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _observations(self, value, count):
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, TensorDict):
            value = TensorDict(value, batch_size=[count])
        return value.to(self.device)

    def _initialize(self, env, obs):
        schema = dict(observation_shapes={key: list(value.shape[1:]) for key, value in obs.items()},
                      num_actions=int(env.num_actions))
        if self.schema is not None and schema != self.schema:
            raise ValueError("Controller cohort changed the native observation/action schema")
        self.schema = schema
        for model in (self.control.actor, self.control.critic):
            dimension = sum(obs[group].shape[-1] for group in model.obs_groups)
            if dimension != model.obs_dim:
                raise ValueError("Controller observation dimensions differ from the existing native models")
        storage = RolloutStorage("rl", env.num_envs, self.steps, obs, [env.num_actions], str(self.device))
        if self.alg is None:
            self.alg = PPO(self.control.actor, self.control.critic, storage,
                           device=str(self.device), **self.algorithm_config)
            if self._pending_optimizer is not None:
                self.alg.optimizer.load_state_dict(self._pending_optimizer)
                self._pending_optimizer = None
            self.alg.learning_rate = self._learning_rate
            for group in self.alg.optimizer.param_groups:
                group["lr"] = self._learning_rate
        else:
            self.alg.storage = storage
        return storage

    @staticmethod
    def _storage_bytes(storage):
        """Count allocated rollout tensors once, including lazy distribution buffers."""
        tensors = []
        for value in vars(storage).values():
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, TensorDict):
                tensors.extend(value.values())
            elif isinstance(value, tuple):
                tensors.extend(item for item in value if isinstance(item, torch.Tensor))
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def train(self, env, num_updates, writer=None, elapsed_env_steps=0):
        if isinstance(num_updates, bool) or not isinstance(num_updates, int) or num_updates < 0:
            raise ValueError("num_updates must be a nonnegative integer")
        if isinstance(elapsed_env_steps, bool) or not isinstance(elapsed_env_steps, int) or elapsed_env_steps < 0:
            raise ValueError("elapsed_env_steps must be a nonnegative integer")
        if self._training:
            raise RuntimeError("Controller training is already active")
        metrics = dict(controller_updates=0, controller_iterations=self.iteration, controller_transitions=0,
                       total_controller_transitions=self.total_transitions, elapsed_env_steps=elapsed_env_steps,
                       controller_storage_bytes_peak=0, controller_collect_seconds=0., controller_update_seconds=0.,
                       controller_learning_rate=self.learning_rate, controller_mean_transition_reward=0.,
                       controller_completed_episode_count=0)
        if not num_updates:
            self.control.eval()
            return metrics
        self._training = True
        storage = None
        losses = {}
        episodes = _EpisodeMetrics(env.num_envs, self.device)
        try:
            native = env.unwrapped
            if hasattr(native, "common_step_counter"):
                native.common_step_counter = elapsed_env_steps
            obs = self._observations(env.reset(), env.num_envs)
            storage = self._initialize(env, obs)
            self.control.train()
            self.alg.train_mode()
            for _ in range(num_updates):
                started = self._time()
                with torch.inference_mode():
                    for _ in range(self.steps):
                        actions = self.alg.act(obs)
                        next_obs, rewards, dones, extras = env.step(actions.to(env.device))
                        obs = self._observations(next_obs, env.num_envs)
                        rewards, dones = rewards.to(self.device), dones.to(self.device)
                        if self.config.get("check_for_nan", True):
                            check_nan(obs, rewards, dones)
                        # Account for raw environment rewards before native PPO
                        # adds timeout bootstrap values to its training targets.
                        episodes.step(rewards, dones, extras)
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        self.control.preserve_conditioning_scale()
                    collected = self._time()
                    self.alg.compute_returns(obs)
                report = self.alg.update()
                finished = self._time()
                self.iteration += 1
                transitions = env.num_envs * self.steps
                self.total_transitions += transitions
                episodes.finish_block(writer, self.iteration, transitions)
                metrics["controller_updates"] += 1
                metrics["controller_transitions"] += transitions
                metrics["controller_collect_seconds"] += collected - started
                metrics["controller_update_seconds"] += finished - collected
                metrics["controller_storage_bytes_peak"] = max(metrics["controller_storage_bytes_peak"], self._storage_bytes(storage))
                for key, value in report.items():
                    losses[key] = losses.get(key, 0.) + float(value)
                    if writer is not None:
                        writer.add_scalar("Controller/Loss/" + key, value, self.iteration)
                if writer is not None:
                    writer.add_scalar("Controller/learning_rate", self.alg.learning_rate, self.iteration)
                    writer.add_scalar("Controller/transitions", self.total_transitions, self.iteration)
                    writer.add_scalar("Controller/collection_seconds", collected - started, self.iteration)
                    writer.add_scalar("Controller/update_seconds", finished - collected, self.iteration)
                    writer.add_scalar("Policy/mean_std", self.control.actor.output_std.mean().item(), self.iteration)
            metrics.update(controller_iterations=self.iteration, total_controller_transitions=self.total_transitions,
                           elapsed_env_steps=elapsed_env_steps + num_updates * self.steps,
                           controller_learning_rate=self.alg.learning_rate)
            metrics.update(episodes.summary(metrics["controller_transitions"]))
            metrics.update({"controller_loss_" + key: value / num_updates for key, value in losses.items()})
            self._learning_rate = self.alg.learning_rate
            return metrics
        finally:
            if self.alg is not None:
                self.alg.optimizer.zero_grad(set_to_none=True)
                self.alg.transition.clear()
                self.alg.storage = None
                self.alg.eval_mode()
            self.control.eval()
            self.control.preserve_conditioning_scale()
            storage = None
            self._training = False

    def state_dict(self):
        if self._training:
            raise RuntimeError("Controller checkpoints require a completed native PPO block")
        optimizer = self.alg.optimizer.state_dict() if self.alg is not None else self._pending_optimizer
        return dict(version=self.VERSION, actor_state_dict=self.control.actor.state_dict(),
                    critic_state_dict=self.control.critic.state_dict(), optimizer_state_dict=optimizer,
                    learning_rate=self.learning_rate, iteration=self.iteration,
                    total_transitions=self.total_transitions, schema=self.schema,
                    algorithm_config=copy.deepcopy(self.algorithm_config), num_steps_per_env=self.steps)

    def load_state_dict(self, state):
        if self._training:
            raise RuntimeError("Controller restore requires a completed native PPO block")
        if state.get("version") != self.VERSION or state["algorithm_config"] != self.algorithm_config or state["num_steps_per_env"] != self.steps:
            raise ValueError("Controller checkpoint changes native PPO configuration")
        if self.schema is not None and state["schema"] != self.schema:
            raise ValueError("Controller checkpoint observation/action schema mismatch")
        rate = state["learning_rate"]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Controller checkpoint learning rate must be finite and positive")
        self.control.actor.load_state_dict(state["actor_state_dict"], strict=True)
        self.control.critic.load_state_dict(state["critic_state_dict"], strict=True)
        self.control.preserve_conditioning_scale()
        self.schema = copy.deepcopy(state["schema"])
        self.iteration, self.total_transitions = state["iteration"], state["total_transitions"]
        self._learning_rate = rate
        if self.alg is None:
            self._pending_optimizer = copy.deepcopy(state["optimizer_state_dict"])
        else:
            if state["optimizer_state_dict"] is not None:
                self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.alg.learning_rate = rate
            for group in self.alg.optimizer.param_groups:
                group["lr"] = rate
