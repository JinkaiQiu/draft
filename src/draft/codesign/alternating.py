"""Native controller PPO followed by independent frozen-policy hardware trials."""

from __future__ import annotations

import gc
import math

import numpy as np
import torch

from .runner import CodesignRunner, _observation


class AlternatingCodesignRunner(CodesignRunner):
    """Train control on one cohort, then score fresh hardware draws without storage."""

    DEFAULTS = dict(CodesignRunner.DEFAULTS, training_schedule="alternating",
                    controller_updates_per_cycle=20, max_training_design_attempts=128,
                    controller_train_cfg=None, hardware_evaluation_steps=None, hardware_bootstrap=False)

    def __init__(self, *args, **kwargs):
        self.controller_trainer = None
        self.controller_cycles = 0
        self.hardware_evaluation_transitions = 0
        super().__init__(*args, **kwargs)
        self.training_generator = torch.Generator(device=self.device)
        self.training_generator.manual_seed(self.config["seed"] + 15_485_863)

    def _validate_config(self):
        super()._validate_config()
        if self.config["training_schedule"] != "alternating":
            raise ValueError("Alternating runner requires training_schedule='alternating'")
        for name in ("controller_updates_per_cycle", "max_training_design_attempts"):
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.config["max_training_design_attempts"] < self.config["num_designs"]:
            raise ValueError("max_training_design_attempts must be at least num_designs")
        if not isinstance(self.config["controller_train_cfg"], dict):
            raise ValueError("Alternating runner requires native controller_train_cfg")
        if self.config["controller_train_cfg"].get("num_steps_per_env") != 24:
            raise ValueError("Alternating controller PPO requires native 24-step rollouts")
        cap = self.config["hardware_evaluation_steps"]
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError("hardware_evaluation_steps must be a positive integer or None")
        if not isinstance(self.config["hardware_bootstrap"], bool):
            raise ValueError("hardware_bootstrap must be a boolean")
        if self.config["hardware_bootstrap"] and cap is None:
            raise ValueError("hardware_bootstrap requires explicit hardware_evaluation_steps")

    def _register_control(self):
        # Native PPO owns control parameters and its adaptive optimizer state.
        self.control_initialized = True

    def _ensure_controller_trainer(self):
        if self.controller_trainer is None:
            from .controller import NativeControllerTrainer
            self.controller_trainer = NativeControllerTrainer(
                self.policy.control, self.config["controller_train_cfg"], self.device)
        return self.controller_trainer

    def _training_designs(self):
        """Resample infeasible training draws only; these are never hardware records."""
        c = self.config
        designs, physical_designs, draws = [], [], 0
        while len(designs) < c["num_designs"] and draws < c["max_training_design_attempts"]:
            remaining = min(c["num_designs"] - len(designs), c["max_training_design_attempts"] - draws)
            if c["hardware_sampling_mode"] == "fixed":
                physical = np.broadcast_to(np.asarray(self.compiler.space.center),
                                           (remaining, self.compiler.space.dim)).copy()
            else:
                _, _, normalized, _ = self.policy.hardware.sample(remaining, self.training_generator)
                physical = self.compiler.space.denormalize(normalized.cpu().numpy())
            # Include retained designs so cache pruning cannot invalidate them.
            compiled = self.compiler.compile_many([*physical_designs, *physical])
            if len(compiled) != len(designs) + remaining:
                raise ValueError("Compiler must retain exactly one result per training draw")
            new_records = compiled[len(designs):]
            draws += remaining
            for coordinates, record in zip(physical, new_records):
                if record.design is not None:
                    designs.append(record.design)
                    physical_designs.append(coordinates)
        if len(designs) != c["num_designs"]:
            raise RuntimeError(f"Unable to fill controller training cohort with feasible independent draws after {draws} draws")
        # Renew the complete cohort lease before the native model reads XMLs.
        renewed = self.compiler.compile_many(physical_designs)
        if len(renewed) != len(designs) or any(record.design is None for record in renewed):
            raise RuntimeError("Previously feasible controller training cohort failed compilation")
        return [record.design for record in renewed], draws

    def collect(self):
        if self.phase != "boundary":
            raise RuntimeError("A cohort must be updated before collecting another")
        self.phase = "training"
        started = self._time()
        c = self.config
        designs, draws = self._training_designs()
        compiled_at = self._time()
        env = self.env_factory(designs, c["replicas"], device=str(self.device),
                               episode_length_s=c["episode_length_s"],
                               seed=c["seed"] + 1_000_003 + self.iteration,
                               env_kwargs=c["env_kwargs"], reward_overrides=c["reward_overrides"])
        built_at = self._time()
        try:
            if not self.control_initialized:
                self._ensure_control(_observation(env.reset()), env.num_actions)
            training = self._ensure_controller_trainer().train(
                env, c["controller_updates_per_cycle"], writer=self.writer,
                elapsed_env_steps=self.elapsed_env_steps)
            self.elapsed_env_steps = int(training["elapsed_env_steps"])
            self.global_step += int(training["controller_transitions"])
            self.controller_cycles += 1
        finally:
            env.close()
            del env
            gc.collect()
        trained_at = self._time()
        self.policy.eval()
        self.phase = "boundary"
        batch = super().collect()
        batch["controller_training"] = dict(training, controller_training_draws=draws,
                                             controller_training_rejections=draws - c["num_designs"],
                                             controller_compile_seconds=compiled_at - started,
                                             controller_build_seconds=built_at - compiled_at,
                                             controller_stage_seconds=trained_at - started)
        batch["timings"]["hardware_collect_seconds"] = batch["timings"]["collect_seconds"]
        batch["timings"]["collect_seconds"] = self._time() - started
        return batch

    @torch.no_grad()
    def _collect_control(self, env, num_designs):
        """Keep only replica returns and counters, independent of trial duration."""
        c = self.config
        self.policy.eval()
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
        steps = min(horizon, c["hardware_evaluation_steps"] or horizon)
        capped = steps < horizon
        active = torch.ones(count, dtype=torch.bool, device=self.device)
        returns = torch.zeros(count, device=self.device)
        active_transitions = torch.zeros((), dtype=torch.long, device=self.device)
        discount = 1.
        for _ in range(steps):
            if any(value.shape[0] != count or not torch.isfinite(value).all() for value in observation.values()):
                raise ValueError("Nonfinite or incorrectly sized native observations")
            actions, _, _, _ = self.policy.control.evaluate(observation)
            if not torch.isfinite(actions).all():
                raise ValueError("Nonfinite control rollout")
            result = env.step(torch.where(active[:, None], actions, torch.zeros_like(actions)))
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
            if not torch.isfinite(rewards).all():
                raise ValueError("Nonfinite control rollout")
            returns.add_((rewards + terminated * c["terminal_penalty"]) * active, alpha=discount)
            active_transitions.add_(active.sum())
            active &= ~done.bool().reshape(-1)
            discount *= c["gamma"]
            observation = _observation(next_obs)
        observed_return_mean = returns.mean().item()
        bootstrap_mean, bootstrap_replicas = 0., 0
        if capped and c["hardware_bootstrap"] and active.any():
            expected = self.control_schema["observation_shapes"]
            if set(observation) != set(expected) or any(
                    list(value.shape) != [count, *expected[key]] or not torch.isfinite(value).all()
                    for key, value in observation.items()):
                raise ValueError("Nonfinite or incorrectly sized final native observations")
            final_observation = {key: value[active] for key, value in observation.items()}
            values = self.policy.control.critic(final_observation)
            bootstrap_replicas = int(active.sum())
            if values.shape not in ((bootstrap_replicas,), (bootstrap_replicas, 1)) or not torch.isfinite(values).all():
                raise ValueError("Nonfinite or incorrectly sized hardware bootstrap values")
            bootstrap = discount * values.reshape(-1)
            returns[active] += bootstrap
            bootstrap_mean = bootstrap.sum().item() / count
        transitions = steps * count
        self.global_step += transitions
        self.hardware_evaluation_transitions += transitions
        # Held-out trials do not advance the controller's native curriculum.
        return dict(replica_returns=returns.reshape(num_designs, c["replicas"]),
                    simulated_transitions=transitions, active_transitions=int(active_transitions),
                    horizon_steps=steps, native_horizon_steps=horizon, survivors=int(active.sum()),
                    evaluation_capped=int(capped), observed_return_mean=observed_return_mean,
                    bootstrap_mean=bootstrap_mean, bootstrap_replicas=bootstrap_replicas)

    def _design_scores(self, control, num_designs):
        return control["replica_returns"].mean(-1)

    def update(self, batch):
        summary, training = batch["control"], batch["controller_training"]
        batch["control"] = None  # Base PPO sees precisely the fresh hardware records.
        self.policy.eval()
        for parameter in self.policy.control.parameters():
            parameter.grad = None
        metrics = super().update(batch)
        metrics.update(training)
        metrics["control_updates"] = training["controller_updates"]
        metrics["controller_cycles"] = self.controller_cycles
        metrics["hardware_evaluation_transitions"] = summary["simulated_transitions"] if summary else 0
        metrics["hardware_active_transitions"] = summary["active_transitions"] if summary else 0
        for key in ("native_horizon_steps", "survivors", "evaluation_capped",
                    "observed_return_mean", "bootstrap_mean", "bootstrap_replicas"):
            metrics["hardware_" + key] = summary[key] if summary else 0
        metrics["hardware_evaluation_steps"] = summary["horizon_steps"] if summary else 0
        metrics["hardware_evaluation_cap_steps"] = self.config["hardware_evaluation_steps"] or 0
        metrics["hardware_bootstrap_enabled"] = int(self.config["hardware_bootstrap"])
        metrics["hardware_survival_fraction"] = summary["survivors"] / summary["replica_returns"].numel() if summary else 0.
        metrics["total_hardware_evaluation_transitions"] = self.hardware_evaluation_transitions
        metrics["control_transitions"] = training["controller_transitions"]
        metrics["simulated_transitions"] = training["controller_transitions"] + metrics["hardware_evaluation_transitions"]
        metrics["sps"] = metrics["simulated_transitions"] / max(metrics["iteration_seconds"], 1e-9)
        return metrics

    def _fresh_warmstart_state(self, *args):
        super()._fresh_warmstart_state(*args)
        self.policy.hardware.load_state_dict(self._hardware_prior_state, strict=True)
        self.controller_trainer = None
        self.controller_cycles = self.hardware_evaluation_transitions = 0
        self.training_generator.manual_seed(self.config["seed"] + 15_485_863)

    def _extra_checkpoint_state(self):
        return dict(alternating_state=dict(
            training_rng=self.training_generator.get_state(), controller_cycles=self.controller_cycles,
            hardware_evaluation_transitions=self.hardware_evaluation_transitions,
            controller_trainer=self.controller_trainer.state_dict() if self.controller_trainer else None))

    def _load_extra_checkpoint_state(self, data):
        state = data.get("alternating_state")
        if not isinstance(state, dict):
            raise ValueError("Alternating resume requires controller optimizer and training RNG state")
        self.training_generator.set_state(state["training_rng"].cpu())
        self.controller_cycles = state["controller_cycles"]
        self.hardware_evaluation_transitions = state["hardware_evaluation_transitions"]
        if state["controller_trainer"] is not None:
            self._ensure_controller_trainer().load_state_dict(state["controller_trainer"])
