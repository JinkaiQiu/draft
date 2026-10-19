"""Finite-design-window PPO with one hardware decision per compiled cohort row."""

from __future__ import annotations

import dataclasses
import hashlib
from importlib.metadata import PackageNotFoundError, version
import gc
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from .policy import HardwarePolicy, JointPolicy, NativeControlPolicy
from .metrics import tensorboard_scalars


def _observation(value):
    if isinstance(value, tuple):
        value = value[0]
    return {key: tensor.detach().clone() for key, tensor in value.items()}


def finite_horizon_advantages(rewards, values, active, gamma, lam):
    """Zero terminal bootstrap, including at the fixed outer design boundary."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        continuing = active[step + 1] if step + 1 < rewards.shape[0] else torch.zeros_like(active[step])
        next_value = values[step + 1] if step + 1 < rewards.shape[0] else torch.zeros_like(values[step])
        delta = rewards[step] + gamma * next_value * continuing - values[step]
        carry = (delta + gamma * lam * carry * continuing) * active[step]
        advantages[step] = carry
    return advantages, (advantages + values) * active


def ppo_loss(logprob, old_logprob, entropy, values, returns, advantages, *, clip,
             value_coef, entropy_coef, normalize=True, old_values=None):
    advantages = advantages.detach()
    if normalize and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    logratio = logprob - old_logprob.detach()
    ratio = logratio.exp()
    policy_loss = -torch.minimum(ratio * advantages,
                                ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    value_errors = (values - returns.detach()).square()
    if old_values is not None:
        clipped_values = old_values + (values - old_values).clamp(-clip, clip)
        value_errors = torch.maximum(value_errors, (clipped_values - returns.detach()).square())
    value_loss = .5 * value_errors.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
    kl = ((ratio - 1) - logratio).mean().detach()
    return loss, {"policy_loss": policy_loss.detach(), "value_loss": value_loss.detach(), "kl": kl}


class CodesignRunner:
    """Collect a complete cohort with frozen policies before any PPO update.

    ``compiler`` provides ``space.denormalize`` and ``compile_many``. Each compile
    result has ``design`` and ``error``; rejected rows remain in the hardware
    batch. ``env_factory`` builds the native design-major G*R vector environment.
    Native auto-reset is permitted internally: all transitions after each
    replica's first done are excluded from rewards, normalization, and PPO.
    """

    CHECKPOINT_VERSION = 1
    DEFAULTS = dict(num_designs=8, replicas=8, horizon_steps=None, episode_length_s=20., device="cuda",
                    seed=0, learning_rate=1e-3, gamma=.99, lam=.95, clip_param=.2,
                    value_loss_coef=1., entropy_coef=.01, max_grad_norm=1., num_learning_epochs=5,
                    num_mini_batches=4, desired_kl=.01, hardware_target_kl=.02,
                    hardware_loss_weight=1., hardware_value_loss_coef=.5, hardware_entropy_coef=0.,
                    terminal_penalty=-10., invalid_design_return=-1000., hardware_std=.5,
                    optimize_hardware=True, hardware_sampling_mode="stochastic", logger="tensorboard", wandb_project="draft_codesign",
                    use_clipped_value_loss=True, schedule="fixed",
                    eval_interval=25, eval_replicas=16, final_evaluation=True, final_export=True,
                    eval_first_update=False,
                    eval_video_interval=0, eval_video_length_s=10., eval_video_fps=25,
                    eval_video_width=640, eval_video_height=480,
                    rollout_storage_device=None, max_control_batch_size=None,
                    hardware_update_schedule="once_per_epoch", save_interval=50,
                    actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
                    activation="elu", actor_obs_normalization=True, critic_obs_normalization=True,
                    normalization_exclusions={"actor": tuple(range(-8, -1)), "critic": tuple(range(-8, -1))},
                    init_std=.5, std_type="scalar", env_kwargs={}, reward_overrides={})

    def __init__(self, policy: JointPolicy | None, compiler, env_factory, config,
                 *, log_dir: str | Path | None = None, metadata: dict | None = None):
        supplied = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
        self.config = {**self.DEFAULTS, **supplied}
        self._validate_config()
        self.device = torch.device(self.config["device"])
        self.storage_device = torch.device(self.config.get("rollout_storage_device") or self.device)
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        self.hardware_generator = torch.Generator(device=self.device)
        self.hardware_generator.manual_seed(self.config["seed"] + 8_675_309)
        self.compiler, self.env_factory = compiler, env_factory
        initial = compiler.space.normalize(compiler.space.center) if hasattr(compiler.space, "normalize") else None
        self.policy = policy or JointPolicy(None, HardwarePolicy(compiler.space.dim, self.config["hardware_std"], initial))
        self.policy.to(self.device)
        if self.config["hardware_sampling_mode"] == "fixed" and initial is not None:
            with torch.no_grad():
                self.policy.hardware.mean.copy_(torch.as_tensor(initial, device=self.device, dtype=self.policy.hardware.mean.dtype).atanh())
        self._hardware_prior_state = {key: value.detach().clone() for key, value in self.policy.hardware.state_dict().items()}
        self.optimizer = torch.optim.Adam(self.policy.hardware.parameters(), lr=self.config["learning_rate"])
        self.control_initialized = False
        self.control_schema = None
        if self.policy.control is not None:
            self._register_control()
        self.iteration = 0
        self.global_step = 0
        self.elapsed_env_steps = 0
        self.phase = "boundary"
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        source_root = Path(__file__).resolve().parent
        sources = [*source_root.glob("*.py"),
                   *(source_root.parent / "tasks/quadruped/codesign").glob("*.py")]
        self.metadata["implementation_sha256"] = {
            str(path.relative_to(source_root.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)}
        versions = {"torch": torch.__version__}
        for name in ("mjlab", "rsl-rl-lib", "mujoco", "mujoco-warp", "warp-lang"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = None
        self.metadata["runtime_versions"] = versions
        self.writer = None
        self.wandb_run = None

    @property
    def current_learning_iteration(self):
        return self.iteration

    def _validate_config(self):
        c = self.config
        cap = c.get("max_control_batch_size")
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError("max_control_batch_size must be a positive integer or None")
        for name in ("num_designs", "replicas", "num_learning_epochs", "num_mini_batches", "save_interval", "eval_replicas"):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if c["hardware_sampling_mode"] not in ("stochastic", "fixed"):
            raise ValueError("hardware_sampling_mode must be stochastic or fixed")
        if c["hardware_sampling_mode"] == "fixed" and c["optimize_hardware"]:
            raise ValueError("Fixed nominal hardware requires optimize_hardware=False")
        if c["num_designs"] == 1 and c["optimize_hardware"]:
            raise ValueError("num_designs=1 requires a fixed or frozen hardware policy")
        for name in ("eval_video_interval", "eval_video_fps", "eval_video_width", "eval_video_height"):
            minimum = 0 if name == "eval_video_interval" else (2 if name in ("eval_video_width", "eval_video_height") else 1)
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            if name in ("eval_video_width", "eval_video_height") and c[name] % 2:
                raise ValueError(f"{name} must be even for video encoding")
        if not math.isfinite(c["eval_video_length_s"]) or c["eval_video_length_s"] <= 0:
            raise ValueError("eval_video_length_s must be finite and positive")
        if not isinstance(c["eval_interval"], int) or c["eval_interval"] < 0:
            raise ValueError("eval_interval must be a nonnegative integer (0 disables periodic evaluation)")
        if c["horizon_steps"] is not None and (not isinstance(c["horizon_steps"], int) or c["horizon_steps"] < 1):
            raise ValueError("horizon_steps must be positive or None (native environment horizon)")
        for name in ("gamma", "lam"):
            if not math.isfinite(c[name]) or not 0 <= c[name] <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("learning_rate", "max_grad_norm", "episode_length_s", "hardware_std"):
            if not math.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("value_loss_coef", "entropy_coef", "hardware_loss_weight", "hardware_value_loss_coef",
                     "hardware_entropy_coef", "desired_kl", "hardware_target_kl"):
            if not math.isfinite(c[name]) or c[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < c["clip_param"] < 1:
            raise ValueError("clip_param must be in (0, 1)")
        for name in ("terminal_penalty", "invalid_design_return"):
            if not math.isfinite(c[name]) or c[name] > 0:
                raise ValueError(f"{name} must be a finite nonpositive penalty")
        if c["hardware_update_schedule"] != "once_per_epoch":
            raise ValueError("Supported hardware_update_schedule is once_per_epoch")
        if c["schedule"] != "fixed":
            raise ValueError("Co-design uses fixed learning rate and independent control/hardware KL guards")

    def _register_control(self):
        self.optimizer.add_param_group({"params": self.policy.control.parameters()})
        self.control_initialized = True

    def _time(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _ensure_control(self, observation, num_actions):
        schema = {"observation_shapes": {key: list(value.shape[1:]) for key, value in observation.items()},
                  "num_actions": int(num_actions)}
        if self.control_schema is not None and schema != self.control_schema:
            raise ValueError("Co-design cohort changed the controller observation/action schema")
        if not self.control_initialized:
            c = self.config
            common = dict(activation=c["activation"])
            actor = c.get("actor_cfg") or dict(**common, hidden_dims=c["actor_hidden_dims"], obs_normalization=c["actor_obs_normalization"],
                         distribution_cfg=dict(class_name="GaussianDistribution", init_std=c["init_std"], std_type=c["std_type"]))
            critic = c.get("critic_cfg") or dict(**common, hidden_dims=c["critic_hidden_dims"], obs_normalization=c["critic_obs_normalization"])
            self.policy.control = NativeControlPolicy.from_observation(observation, num_actions, actor, critic,
                                                                     c.get("obs_groups"),
                                                                     normalization_exclusions=c["normalization_exclusions"]).to(self.device)
            self._register_control()
            self.policy.eval()
        self.control_schema = schema

    def collect(self):
        if self.phase != "boundary":
            raise RuntimeError("A cohort must be updated before collecting another")
        self.phase = "collecting"
        started = self._time()
        c = self.config
        self.policy.eval()
        if c["hardware_sampling_mode"] == "fixed":
            physical = np.broadcast_to(np.asarray(self.compiler.space.center, dtype=np.float64),
                                       (c["num_designs"], self.compiler.space.dim)).copy()
            coordinates = self.compiler.space.normalize(physical) if hasattr(self.compiler.space, "normalize") else physical
            normalized = torch.as_tensor(coordinates, dtype=self.policy.hardware.mean.dtype, device=self.device)
            raw = normalized.atanh()
            with torch.no_grad():
                logprob, _, baseline = self.policy.hardware.evaluate(raw)
                logprob, baseline = logprob.clone(), baseline.clone()
        else:
            raw, logprob, normalized, baseline = self.policy.hardware.sample(c["num_designs"], self.hardware_generator)
            physical = self.compiler.space.denormalize(normalized.cpu().numpy())
        compiled = self.compiler.compile_many(physical)
        compiled_at = self._time()
        if len(compiled) != c["num_designs"]:
            raise ValueError("Compiler must retain exactly one result per hardware draw")
        valid_indices = [i for i, record in enumerate(compiled) if record.design is not None]
        scores = raw.new_full((c["num_designs"],), c["invalid_design_return"])
        batch = dict(raw=raw, old_logprob=logprob, baseline=baseline, scores=scores,
                     normalized=normalized, physical=physical, compilation_errors=[r.error for r in compiled],
                     design_ids=[getattr(r, "design_id", None) or (r.design.design_id if r.design is not None else None)
                                 for r in compiled],
                     valid_indices=valid_indices, control=None,
                     timings=dict(compile_seconds=compiled_at - started, build_seconds=0., rollout_seconds=0.))
        if valid_indices:
            env = self.env_factory([compiled[i].design for i in valid_indices], c["replicas"],
                                   device=str(self.device), episode_length_s=c["episode_length_s"],
                                   seed=c["seed"] + self.iteration, env_kwargs=c["env_kwargs"],
                                   reward_overrides=c["reward_overrides"])
            built_at = self._time()
            batch["timings"]["build_seconds"] = built_at - compiled_at
            try:
                batch["control"] = self._collect_control(env, len(valid_indices))
                batch["timings"]["rollout_seconds"] = self._time() - built_at
            finally:
                env.close()
                del env
                gc.collect()  # Native manager cycles otherwise retain the old GPU cohort.
            scores[valid_indices] = self._design_scores(batch["control"], len(valid_indices)).to(self.device)
        batch["timings"]["collect_seconds"] = self._time() - started
        self.phase = "collected"
        return batch

    def _design_scores(self, control, num_designs):
        rewards = control["rewards"].reshape(-1, num_designs, self.config["replicas"])
        discounts = rewards.new_tensor(self.config["gamma"]).pow(torch.arange(len(rewards), device=rewards.device))
        return (rewards.mean(-1) * discounts[:, None]).sum(0)

    @torch.no_grad()
    def _collect_control(self, env, num_designs):
        c = self.config
        if hasattr(env.unwrapped, "common_step_counter"):
            env.unwrapped.common_step_counter = self.elapsed_env_steps
        observation = _observation(env.reset())
        self._ensure_control(observation, env.num_actions)
        count = num_designs * c["replicas"]
        horizon = int(getattr(env.unwrapped, "horizon_steps", 0) or
                      math.ceil(c["episode_length_s"] / env.unwrapped.step_dt))
        if c["horizon_steps"] is not None and horizon != c["horizon_steps"]:
            raise ValueError("Configured horizon_steps differs from native fixed design duration")
        if horizon < 1:
            raise ValueError("Environment must expose a positive horizon_steps")
        active = torch.ones(count, dtype=torch.bool, device=self.device)
        data = {"observations": {key: torch.empty((horizon, *value.shape), dtype=value.dtype, device=self.storage_device)
                                  for key, value in observation.items()}}
        for key, shape, dtype in (("actions", (env.num_actions,), torch.float32), ("logprob", (), torch.float32),
                                  ("values", (), torch.float32), ("rewards", (), torch.float32), ("active", (), torch.bool)):
            data[key] = torch.empty((horizon, count, *shape), dtype=dtype, device=self.storage_device)
        for step in range(horizon):
            if any(value.shape[0] != count or not torch.isfinite(value).all() for value in observation.values()):
                raise ValueError("Nonfinite or incorrectly sized native observations")
            actions, logprob, _, values = self.policy.control.evaluate(observation)
            safe_actions = torch.where(active[:, None], actions, torch.zeros_like(actions))
            result = env.step(safe_actions)
            if len(result) == 4:
                next_obs, rewards, done, extras = result
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool().reshape(-1)
                terminated = done.bool().reshape(-1) & ~timeout
            else:
                next_obs, rewards, terminated, timeout, extras = result
                terminated, timeout = terminated.bool().reshape(-1), timeout.bool().reshape(-1)
                done = terminated | timeout
            if "codesign_terminated" in extras:
                terminated = extras["codesign_terminated"].bool().reshape(-1)
                done = done.bool().reshape(-1) | terminated
            rewards = rewards.reshape(-1)
            if not torch.isfinite(rewards).all() or not torch.isfinite(actions).all() or not torch.isfinite(values).all():
                raise ValueError("Nonfinite control rollout")
            rewards = (rewards + terminated * c["terminal_penalty"]) * active
            for key, value in observation.items():
                data["observations"][key][step].copy_(value)
            for key, value in (("actions", actions), ("logprob", logprob), ("values", values),
                               ("rewards", rewards), ("active", active)):
                data[key][step].copy_(value)
            active = active & ~done.bool().reshape(-1)
            observation = _observation(next_obs)
        self.global_step += horizon * count
        self.elapsed_env_steps = int(getattr(env.unwrapped, "common_step_counter", self.elapsed_env_steps + horizon))
        return data

    def update(self, batch):
        if self.phase != "collected":
            raise RuntimeError("Joint PPO updates require a complete frozen cohort")
        self.phase = "updating"
        started = self._time()
        c, control = self.config, batch["control"]
        metrics = {"hardware_updates": 0, "control_updates": 0, "hardware_records": len(batch["raw"]),
                   "design_score_mean": batch["scores"].mean().item(),
                   "invalid_designs": len(batch["raw"]) - len(batch["valid_indices"])}
        h_advantages = batch["scores"] - batch["baseline"]
        control_open, hardware_open = control is not None, c["optimize_hardware"]
        if control is not None:
            advantages, returns = finite_horizon_advantages(control["rewards"], control["values"],
                                                            control["active"], c["gamma"], c["lam"])
            keep = control["active"].flatten().nonzero(as_tuple=False).squeeze(-1)
            observations = {k: v.flatten(0, 1) for k, v in control["observations"].items()}
            flat = {k: control[k].flatten(0, 1) for k in ("actions", "logprob", "values")}
            advantages, returns = advantages.flatten()[keep], returns.flatten()
            # Normalize on the complete valid control batch, not tiny minibatches.
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(c["num_learning_epochs"]):
            parts = torch.randperm(len(advantages), device=keep.device).tensor_split(c["num_mini_batches"]) if control_open else []
            indices = [chunk for part in parts if part.numel()
                       for chunk in part.split(c.get("max_control_batch_size") or len(part))]
            for minibatch in range(max(1, len(indices))):
                loss = None
                use_control, use_hardware = False, False
                if control_open and minibatch < len(indices):
                    index = indices[minibatch]
                    rows = keep[index]
                    states = {k: v[rows].to(self.device) for k, v in observations.items()}
                    _, lp, entropy, values = self.policy.control.evaluate(states, flat["actions"][rows].to(self.device))
                    c_loss, report = ppo_loss(lp, flat["logprob"][rows].to(self.device), entropy, values,
                                              returns[rows].to(self.device), advantages[index].to(self.device),
                                              clip=c["clip_param"], value_coef=c["value_loss_coef"], entropy_coef=c["entropy_coef"], normalize=False,
                                              old_values=flat["values"][rows].to(self.device) if c["use_clipped_value_loss"] else None)
                    metrics.update({"control_" + k: v.item() for k, v in report.items()})
                    control_open = report["kl"] <= c["desired_kl"]
                    if control_open:
                        loss, use_control = c_loss, True
                if hardware_open and minibatch == 0:
                    lp, entropy, values = self.policy.hardware.evaluate(batch["raw"])
                    h_loss, report = ppo_loss(lp, batch["old_logprob"], entropy, values, batch["scores"], h_advantages,
                                              clip=c["clip_param"], value_coef=c["hardware_value_loss_coef"], entropy_coef=c["hardware_entropy_coef"])
                    metrics.update({"hardware_" + k: v.item() for k, v in report.items()})
                    hardware_open = report["kl"] <= c["hardware_target_kl"]
                    if hardware_open and c["hardware_loss_weight"] > 0:
                        weighted = c["hardware_loss_weight"] * h_loss
                        loss = weighted if loss is None else loss + weighted
                        use_hardware = True
                if loss is not None:
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite joint PPO loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c["max_grad_norm"], error_if_nonfinite=True)
                    self.optimizer.step()
                    metrics["control_updates"] += int(use_control)
                    metrics["hardware_updates"] += int(use_hardware)
                if not control_open:
                    break
            if not control_open and not hardware_open:
                break
        if control is not None and hasattr(self.policy.control, "update_normalization"):
            # This changes the next cohort's policy only after all current PPO ratios.
            self.policy.control.train()
            try:
                for rows in keep.split(c.get("max_control_batch_size") or len(keep)):
                    self.policy.control.update_normalization({k: v[rows].to(self.device) for k, v in observations.items()})
            finally:
                self.policy.control.eval()
        metrics.update(batch.get("timings", {}))
        metrics["update_seconds"] = self._time() - started
        metrics["iteration_seconds"] = metrics.get("collect_seconds", 0.) + metrics["update_seconds"]
        metrics["simulated_transitions"] = control["rewards"].numel() if control is not None else 0
        metrics["control_transitions"] = int(control["active"].sum()) if control is not None else 0
        metrics["sps"] = metrics["simulated_transitions"] / max(metrics["iteration_seconds"], 1e-9)
        physical_mean = self.compiler.space.denormalize(self.policy.hardware.decode(self.policy.hardware.mean).cpu().numpy())
        std = self.policy.hardware.log_std.detach().exp().cpu().tolist()
        for name, mean, scale in zip(self.compiler.space.names, physical_mean, std):
            metrics["hardware_mean_" + name] = float(mean)
            metrics["hardware_latent_std_" + name] = scale
        self.iteration += 1
        if self.log_dir:
            with (self.log_dir / "design_records.jsonl").open("a") as stream:
                for i in range(len(batch["raw"])):
                    record = dict(iteration=self.iteration, draw_index=i, design_id=batch["design_ids"][i],
                                  raw=batch["raw"][i].cpu().tolist(), normalized=batch["normalized"][i].cpu().tolist(),
                                  physical=batch["physical"][i].tolist(), score=batch["scores"][i].item(),
                                  old_logprob=batch["old_logprob"][i].item(), baseline=batch["baseline"][i].item(),
                                  error=batch["compilation_errors"][i])
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.phase = "boundary"
        return metrics

    def learn(self, num_learning_iterations: int):
        if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int) or num_learning_iterations < 0:
            raise ValueError("num_learning_iterations must be a nonnegative integer")
        if self.log_dir and self.writer is None and self.config["logger"] == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.log_dir))
        if self.log_dir and self.wandb_run is None and self.config["logger"] == "wandb":
            import wandb
            self.wandb_run = wandb.init(project=self.config["wandb_project"], dir=str(self.log_dir),
                                       config=self.config, name=self.config.get("run_name"))
        if self.log_dir and not (self.log_dir / "model_initial.pt").exists():
            self.save(self.log_dir / "model_initial.pt")
        history = []
        for _ in range(num_learning_iterations):
            metrics = self.update(self.collect())
            history.append(metrics)
            mean = [metrics["hardware_mean_" + name] for name in self.compiler.space.names]
            std = [metrics["hardware_latent_std_" + name] for name in self.compiler.space.names]
            print(f"Update {self.iteration}: score={metrics['design_score_mean']:.3f} "
                  f"valid={metrics['hardware_records'] - metrics['invalid_designs']}/{metrics['hardware_records']} "
                  f"SPS={metrics['sps']:.0f} "
                  f"compile/build/rollout/update={metrics.get('compile_seconds', 0.):.2f}/"
                  f"{metrics.get('build_seconds', 0.):.2f}/{metrics.get('rollout_seconds', 0.):.2f}/"
                  f"{metrics['update_seconds']:.2f}s "
                  f"hardware_mean={np.round(mean, 4).tolist()} latent_std={np.round(std, 3).tolist()}", flush=True)
            if self.writer:
                for tag, value in tensorboard_scalars(metrics).items():
                    self.writer.add_scalar(tag, value, self.iteration)
            if self.wandb_run:
                self.wandb_run.log(metrics, step=self.iteration)
            if self.log_dir:
                self.save(self.log_dir / "model_latest.pt")
                with (self.log_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(iteration=self.iteration, global_step=self.global_step, **metrics)) + "\n")
                if self.iteration % self.config["save_interval"] == 0:
                    self.save(self.log_dir / f"model_{self.iteration}.pt")
                first_eval = self.iteration == 1 and self.config["eval_first_update"]
                periodic_eval = self.config["eval_interval"] and self.iteration % self.config["eval_interval"] == 0
                video_eval = self.config["eval_video_interval"] and self.iteration % self.config["eval_video_interval"] == 0
                if first_eval or periodic_eval or video_eval:
                    force_video = True if first_eval and self.config["eval_video_interval"] > 0 else None
                    self._record_evaluation(self.evaluate(record_video=force_video))
        if self.log_dir:
            if self.config["final_evaluation"]:
                final = self.evaluate()
                self._record_evaluation(final)
            self.save(self.log_dir / "model_final.pt")
            if self.config["final_export"] and self.control_initialized:
                from .evaluation import selected_design
                result, _, _ = selected_design(self)
                if result.design is not None:
                    self.export(self.log_dir / f"export_{self.iteration}")
                else:
                    (self.log_dir / "export_failure.json").write_text(json.dumps(dict(
                        iteration=self.iteration, selection="decoded hardware mean", error=result.error)) + "\n")
                    print(f"Final export unavailable: decoded hardware mean is infeasible: {result.error}", flush=True)
        if self.writer:
            self.writer.flush()
        return history

    def evaluate(self, *, replicas=None, seed=None, record_video=None):
        from .evaluation import evaluate
        kwargs = dict(replicas=replicas, seed=seed)
        if record_video is not None:
            kwargs["record_video"] = record_video
        return evaluate(self, **kwargs)

    def export(self, destination):
        from .evaluation import export
        return export(self, destination)

    def _record_evaluation(self, report):
        if report["valid"]:
            print(f"Evaluation {self.iteration}: return={report['return_mean']:.3f} "
                  f"survival={report['survival_fraction']:.1%} "
                  f"tracking_rmse={report.get('velocity_tracking_rmse_m_s')}", flush=True)
        else:
            print(f"Evaluation {self.iteration}: infeasible decoded hardware mean: {report['error']}", flush=True)
        with (self.log_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        for key, value in report.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if self.writer:
                    self.writer.add_scalar("eval/" + key, value, self.iteration)
                if self.wandb_run:
                    self.wandb_run.log({"eval/" + key: value}, step=self.iteration)
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None
        if self.wandb_run:
            self.wandb_run.finish()
            self.wandb_run = None

    def warmstart(self, path):
        """Transfer compatible nets/normalizers, starting a fresh optimization run.

        Sampling mode and trainability may change, including nominal controller
        pretraining followed by joint co-design. Full-state continuation is load().
        """
        if self.phase != "boundary":
            raise RuntimeError("Warm start requires a completed update boundary")
        data = torch.load(path, map_location=self.device, weights_only=False)
        if "policy" not in data and "actor_state_dict" in data:
            self._warmstart_native(data)
            self._fresh_warmstart_state(path, data.get("iter", 0), "native_conditioned")
            return
        if data.get("space_schema") != self._space_schema():
            raise ValueError("Warm start requires the same explicit design-space schema")
        if data.get("compiler_fingerprint") != getattr(self.compiler, "fingerprint", None):
            raise ValueError("Warm start changes the compiler source or dependency fingerprint")
        schema = data.get("control_schema")
        if data.get("control_initialized") and schema is None:
            raise ValueError("Warm start checkpoint lacks the native controller schema")
        if self.control_schema is not None and schema != self.control_schema:
            raise ValueError("Warm start changes the controller observation/action schema")
        if schema and not self.control_initialized:
            self._ensure_control({key: torch.zeros((1, *shape), device=self.device)
                                  for key, shape in schema["observation_shapes"].items()}, schema["num_actions"])
        self.policy.load_state_dict(data["policy"], strict=True)
        if hasattr(self.policy.control, "preserve_conditioning_scale"):
            self.policy.control.preserve_conditioning_scale()
        if self.config["hardware_sampling_mode"] == "fixed":
            self.policy.hardware.load_state_dict(self._hardware_prior_state, strict=True)
        self.control_schema = schema
        self._fresh_warmstart_state(path, data["iteration"], data["config"].get("hardware_sampling_mode", "stochastic"))

    def _warmstart_native(self, data):
        infos = data.get("infos")
        transfer = infos.get("codesign_transfer") if isinstance(infos, dict) else None
        if not isinstance(transfer, dict) or transfer.get("version") != 1:
            raise ValueError("Native warm start requires versioned infos.codesign_transfer with explicit design, observation and model schemas")
        if transfer.get("space_schema") != self._space_schema():
            raise ValueError("Native warm start requires the same explicit design-space schema")
        fingerprint = transfer.get("compiler_fingerprint")
        if fingerprint is not None and fingerprint != getattr(self.compiler, "fingerprint", None):
            raise ValueError("Native warm start changes the compiler fingerprint")
        source_schema = transfer.get("control_schema")
        target_schema = self.control_schema or self.config.get("native_transfer_schema")
        if target_schema is None:
            raise ValueError(f"Native controller schema {source_schema} requires an explicitly verified matching target control schema. "
                             "The existing finite-window co-design environment includes remaining time and cannot silently consume native observations.")
        if source_schema != target_schema:
            raise ValueError(f"Native controller observation/action schema mismatch: source={source_schema}, target={target_schema}")
        groups = self.config.get("obs_groups") or {"actor": ["actor"], "critic": ["critic"]}
        source_groups = transfer.get("obs_groups")
        if not isinstance(source_groups, dict) or {k: list(v) for k, v in source_groups.items()} != {k: list(v) for k, v in groups.items()}:
            raise ValueError("Native warm start observation group order mismatch")
        for role in ("actor", "critic"):
            model_cfg = transfer.get(role + "_cfg")
            if not isinstance(model_cfg, dict):
                raise ValueError("Native warm start requires explicit actor_cfg and critic_cfg")
            target_cfg = self.config.get(role + "_cfg") or dict(
                hidden_dims=self.config[role + "_hidden_dims"], activation=self.config["activation"],
                obs_normalization=self.config[role + "_obs_normalization"])
            if role == "actor" and "distribution_cfg" not in target_cfg:
                target_cfg = dict(target_cfg, distribution_cfg=dict(class_name="GaussianDistribution",
                                  init_std=self.config["init_std"], std_type=self.config["std_type"]))
            for key in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg"):
                source_value, target_value = model_cfg.get(key), target_cfg.get(key)
                if key == "hidden_dims":
                    source_value, target_value = tuple(source_value or ()), tuple(target_value or ())
                if source_value != target_value:
                    raise ValueError(f"Native warm start changes {role} model configuration: {key}")
            dimension = sum(source_schema["observation_shapes"][group][0] for group in groups[role])
            exclusions = self.config["normalization_exclusions"].get(role, ())
            resolved = [index if index >= 0 else dimension + index for index in exclusions]
            if transfer.get("conditioning_layout", {}).get(role) != resolved:
                raise ValueError("Native warm start morphology feature layout differs from the target; inputs cannot be reordered or padded implicitly")
        if not self.control_initialized:
            obs = {key: torch.zeros((1, *shape), device=self.device) for key, shape in source_schema["observation_shapes"].items()}
            self._ensure_control(obs, source_schema["num_actions"])
        self.policy.control.actor.load_state_dict(data["actor_state_dict"], strict=True)
        self.policy.control.critic.load_state_dict(data["critic_state_dict"], strict=True)
        self.policy.control.preserve_conditioning_scale()
        self.policy.hardware.load_state_dict(self._hardware_prior_state, strict=True)
        self.control_schema = source_schema

    def _fresh_warmstart_state(self, path, source_iteration, source_sampling_mode):
        self.optimizer = torch.optim.Adam(self.policy.hardware.parameters(), lr=self.config["learning_rate"])
        if self.control_initialized:
            self._register_control()
        self.iteration = self.global_step = self.elapsed_env_steps = 0
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        self.hardware_generator.manual_seed(self.config["seed"] + 8_675_309)
        self.metadata["warmstart"] = dict(checkpoint=str(Path(path).resolve()),
                                          sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                          source_iteration=source_iteration,
                                          source_sampling_mode=source_sampling_mode,
                                          scope="network and normalizer transfer; fresh optimizer, counters and RNG")

    def _space_schema(self):
        space = self.compiler.space
        if hasattr(space, "to_dict"):
            return space.to_dict()
        return {name: list(getattr(space, name)) for name in ("names", "lower", "upper", "center")}

    def _extra_checkpoint_state(self):
        return {}

    def _load_extra_checkpoint_state(self, data):
        pass

    def save(self, path):
        if self.phase != "boundary":
            raise RuntimeError("Checkpoints are supported only at completed update boundaries")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=self.CHECKPOINT_VERSION, policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                       iteration=self.iteration, global_step=self.global_step, config=self.config,
                       elapsed_env_steps=self.elapsed_env_steps,
                       space_schema=self._space_schema(), control_schema=self.control_schema,
                       control_initialized=self.control_initialized, metadata=self.metadata,
                       compiler_fingerprint=getattr(self.compiler, "fingerprint", None),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                       hardware_rng=self.hardware_generator.get_state(),
                       resume_scope="Completed update boundary; new cohort rebuilt, simulator state not restored")
        payload.update(self._extra_checkpoint_state())
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, path):
        if self.phase != "boundary":
            raise RuntimeError("Resume requires a completed update boundary")
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data["version"] != self.CHECKPOINT_VERSION or data["space_schema"] != self._space_schema():
            raise ValueError("Checkpoint format or design-space schema mismatch")
        if data.get("compiler_fingerprint") != getattr(self.compiler, "fingerprint", None):
            raise ValueError("Resume changes the compiler source or dependency fingerprint")
        mutable = {"device", "save_interval", "log_dir", "run_name", "max_iterations", "logger", "wandb_project", "upload_model",
                   "eval_interval", "eval_replicas", "final_evaluation", "final_export"}
        mutable.update({"eval_video_interval", "eval_video_length_s", "eval_video_fps", "eval_video_width", "eval_video_height"})
        mutable.add("eval_first_update")
        for key in set(self.config) | set(data["config"]):
            current, saved = self.config.get(key), data["config"].get(key)
            if key == "controller_train_cfg" and isinstance(current, dict) and isinstance(saved, dict):
                # Native runner metadata may change when resuming into a new
                # output directory; all model and PPO settings remain strict.
                native_mutable = mutable | {"experiment_name"}
                current = {name: value for name, value in current.items() if name not in native_mutable}
                saved = {name: value for name, value in saved.items() if name not in native_mutable}
            if key not in mutable and current != saved:
                raise ValueError(f"Resume changes protected co-design setting: {key}")
        schema = data["control_schema"]
        if data["control_initialized"] and not self.control_initialized:
            if schema is None:
                raise ValueError("Checkpoint lacks the native controller schema")
            observation = {key: torch.zeros((1, *shape), device=self.device) for key, shape in schema["observation_shapes"].items()}
            self._ensure_control(observation, schema["num_actions"])
        self.policy.load_state_dict(data["policy"], strict=True)
        self.optimizer.load_state_dict(data["optimizer"])
        self.control_schema = schema
        self.iteration, self.global_step = data["iteration"], data["global_step"]
        self.elapsed_env_steps = data.get("elapsed_env_steps", 0)
        self.metadata = data["metadata"]
        random.setstate(data["python_rng"])
        np.random.set_state(data["numpy_rng"])
        torch.set_rng_state(data["torch_rng"].cpu())
        self.hardware_generator.set_state(data["hardware_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in data["cuda_rng"]])
        self._load_extra_checkpoint_state(data)
        return data["resume_scope"]
