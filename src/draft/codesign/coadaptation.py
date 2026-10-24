"""Hardware PPO on the complete return earned during controller adaptation."""

from __future__ import annotations

import math
import gc
from contextlib import nullcontext

import torch

from .alternating import AlternatingCodesignRunner
from .runner import CodesignRunner, _observation


class _TrainingWindowScore:
    """One discounted accumulator per world, spanning every reset and PPO block."""

    def __init__(self, count, device, gamma, terminal_penalty, *, block_steps=None, replicas=None):
        self.returns = torch.zeros(count, device=device)
        self.true_terminations = torch.zeros((), dtype=torch.long, device=device)
        self.timeouts = torch.zeros_like(self.true_terminations)
        self.gamma, self.terminal_penalty = gamma, terminal_penalty
        self.steps, self.discount = 0, 1.
        self.block_steps, self.replicas = block_steps, replicas
        self.blocks = []

    @torch.no_grad()
    def step(self, rewards, dones, extras):
        rewards = rewards.detach().reshape(-1).to(self.returns.device)
        done = dones.detach().reshape(-1).to(self.returns.device).bool()
        timeout = torch.as_tensor(extras.get("time_outs", torch.zeros_like(done)),
                                  device=self.returns.device).reshape(-1).bool()
        terminated = torch.as_tensor(extras.get("codesign_terminated", done & ~timeout),
                                     device=self.returns.device).reshape(-1).bool()
        if any(value.shape != self.returns.shape for value in (rewards, done, timeout, terminated)):
            raise ValueError("Training score requires exactly one transition per world")
        if not torch.isfinite(rewards).all():
            raise ValueError("Nonfinite controller training rewards")
        value = rewards + terminated * self.terminal_penalty
        self.returns.add_(value, alpha=self.discount)
        if self.block_steps:
            position = self.steps % self.block_steps
            if not position:
                self.blocks.append(torch.zeros(self.returns.numel() // self.replicas,
                                              device=self.returns.device))
            self.blocks[-1].add_(value.reshape(-1, self.replicas).mean(-1), alpha=self.gamma ** position)
        self.true_terminations.add_(terminated.sum())
        self.timeouts.add_((timeout & ~terminated).sum())
        self.steps += 1
        self.discount *= self.gamma

    @property
    def storage_bytes(self):
        return sum(tensor.numel() * tensor.element_size()
                   for tensor in (self.returns, self.true_terminations, self.timeouts, *self.blocks))

    def block_metrics(self, valid_indices):
        metrics = {}
        for block, scores in enumerate(self.blocks):
            metrics[f"HardwareTraining/Block_{block}/score_mean"] = scores.mean().item()
            for index, score in zip(valid_indices, scores.tolist()):
                metrics[f"HardwareTraining/Block_{block}/Design_{index}/score"] = score
        if len(self.blocks) > 1 and len(valid_indices) > 1:
            first, last = self.blocks[0], self.blocks[-1]
            # Pairwise disagreement handles ties without arbitrary argsort ranks.
            pairs = torch.triu(torch.ones((len(first), len(first)), dtype=torch.bool,
                                         device=first.device), diagonal=1)
            a, b = first[:, None] - first[None, :], last[:, None] - last[None, :]
            comparable = pairs & (a != 0) & (b != 0)
            metrics["HardwareTraining/first_last_block_ranking_disagreement"] = (
                ((a * b < 0) & comparable).sum().item() / max(comparable.sum().item(), 1))
        return metrics


class CoadaptationCodesignRunner(AlternatingCodesignRunner):
    """Score unconditional hardware draws over their shared adaptation window.

    This optimizes performance during the specified controller training schedule.
    The controller adapts to the whole sampled cohort, so own-design returns are
    a hardware PPO surrogate rather than an unbiased gradient of the complete
    coupled cohort objective. Independent frozen-controller validation remains
    available through the inherited evaluation schedule.
    """

    TRAINING_SCHEDULE = "coadaptation"
    DEFAULTS = dict(AlternatingCodesignRunner.DEFAULTS, training_schedule="coadaptation",
                    coadaptation_cohort_lifetime="episode", hardware_gamma=1.,
                    staleness_diagnostics=True, staleness_probe_samples_per_design=64,
                    hardware_vtrace=False, hardware_vtrace_cfg={})

    def __init__(self, *args, **kwargs):
        self.reused_training_transitions = 0
        self._cohort = None
        self._vtrace_scorer = None
        self.cohorts_built = self.resume_cohort_restarts = 0
        super().__init__(*args, **kwargs)

    def _validate_config(self):
        super()._validate_config()
        from .vtrace import HardwareVTraceScorer
        c = self.config
        if not isinstance(c["hardware_vtrace"], bool):
            raise ValueError("hardware_vtrace must be a boolean")
        c["hardware_vtrace_cfg"] = HardwareVTraceScorer.validate_config(c["hardware_vtrace_cfg"])
        if self.config["hardware_evaluation_steps"] is not None:
            raise ValueError("Co-adaptation does not use hardware_evaluation_steps")
        if self.config["hardware_bootstrap"]:
            raise ValueError("Co-adaptation does not use hardware_bootstrap")
        if self.config["coadaptation_cohort_lifetime"] not in ("episode", "window"):
            raise ValueError("coadaptation_cohort_lifetime must be 'episode' or 'window'")
        gamma = self.config["hardware_gamma"]
        if isinstance(gamma, bool) or not isinstance(gamma, (float, int)) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("hardware_gamma must be in [0, 1]")
        if not isinstance(self.config["staleness_diagnostics"], bool):
            raise ValueError("staleness_diagnostics must be a boolean")
        samples = self.config["staleness_probe_samples_per_design"]
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("staleness_probe_samples_per_design must be a positive integer")
        self.config["controller_train_cfg"] = dict(self.config["controller_train_cfg"],
            staleness_diagnostics=self.config["staleness_diagnostics"],
            staleness_probe_samples_per_design=samples)

    def _ensure_vtrace_scorer(self):
        if self._vtrace_scorer is None:
            from .vtrace import HardwareVTraceScorer
            self._vtrace_scorer = HardwareVTraceScorer(self.policy.control,
                self.config["hardware_vtrace_cfg"], device=self.device, seed=self.config["seed"] + 73_856_093)
        return self._vtrace_scorer

    def collect(self):
        if self.config["coadaptation_cohort_lifetime"] == "episode":
            return self._collect_persistent()
        # Reuse the unconditional draw/compile path, preserving rejected rows.
        # Its one environment contains exactly the valid subset, without retries.
        batch = CodesignRunner.collect(self)
        summary = batch["control"]
        training = summary.pop("controller_training") if summary else dict(
            controller_updates=0,
            controller_iterations=getattr(self.controller_trainer, "iteration", 0),
            controller_transitions=0,
            total_controller_transitions=getattr(self.controller_trainer, "total_transitions", 0),
            elapsed_env_steps=self.elapsed_env_steps,
            controller_storage_bytes_peak=0)
        training.update(controller_training_draws=len(batch["raw"]),
                        controller_training_rejections=len(batch["raw"]) - len(batch["valid_indices"]),
                        controller_compile_seconds=batch["timings"]["compile_seconds"],
                        controller_build_seconds=batch["timings"]["build_seconds"],
                        controller_stage_seconds=batch["timings"]["collect_seconds"])
        batch["controller_training"] = training
        return batch

    def _close_cohort(self):
        cohort, self._cohort = self._cohort, None
        if cohort is None:
            return
        env, lease = cohort.pop("env"), cohort.pop("lease")
        try:
            try:
                if self.controller_trainer is not None:
                    self.controller_trainer.release_environment(env)
            finally:
                env.close()
        finally:
            lease.__exit__(None, None, None)
            del env, cohort
            gc.collect()

    def _collect_persistent(self):
        if self.phase != "boundary":
            raise RuntimeError("A cohort must be updated before collecting another")
        self.phase = "collecting"
        started, c = self._time(), self.config
        fresh = self._cohort is None
        try:
            if fresh:
                template, designs = self._prepare_hardware_batch()
                if designs:
                    retain = getattr(self.compiler, "retain_assets", None)
                    lease = retain(template["design_ids"]) if retain else nullcontext()
                    lease.__enter__()
                    try:
                        built_at = self._time()
                        env = self.env_factory(designs, c["replicas"], device=str(self.device),
                                               episode_length_s=c["episode_length_s"],
                                               seed=c["seed"] + c["training_seed_offset"] + self.iteration,
                                               env_kwargs=c["env_kwargs"], reward_overrides=c["reward_overrides"])
                    except BaseException:
                        lease.__exit__(None, None, None)
                        raise
                    template["timings"]["build_seconds"] = self._time() - built_at
                    # Transfer ownership before reading simulator metadata: all
                    # failures after construction must close the env and lease.
                    self._cohort = dict(env=env, lease=lease)
                    del env
                    horizon = int(getattr(self._cohort["env"].unwrapped, "horizon_steps", 0) or
                                  math.ceil(c["episode_length_s"] / self._cohort["env"].unwrapped.step_dt))
                    window = c["controller_updates_per_cycle"] * c["controller_train_cfg"]["num_steps_per_env"]
                    self.cohorts_built += 1
                    self._cohort.update(template=template, age=0,
                                        lifetime=math.ceil(horizon / window) * window,
                                        born=self.iteration, first_done=torch.zeros(self._cohort["env"].num_envs, dtype=torch.bool,
                                                                                 device=self.device),
                                        lengths=torch.zeros(self._cohort["env"].num_envs, dtype=torch.long, device=self.device),
                                        behavior_mean=self.policy.hardware.mean.detach().clone(),
                                        behavior_std=self.policy.hardware.log_std.detach().exp().clone())
                else:
                    # No world was built, but every rejected raw draw still trains hardware.
                    batch = template
            if self._cohort is not None:
                cohort = self._cohort
                batch = dict(cohort["template"])
                batch["scores"] = cohort["template"]["scores"].clone()
                batch["timings"] = (dict(cohort["template"]["timings"]) if fresh else
                                    dict(compile_seconds=0., build_seconds=0., rollout_seconds=0.))
                rollout_at = self._time()
                batch["control"] = self._collect_control(cohort["env"], len(batch["valid_indices"]))
                batch["timings"]["rollout_seconds"] = self._time() - rollout_at
                batch["scores"][batch["valid_indices"]] = self._design_scores(batch["control"], len(batch["valid_indices"]))
                cohort["age"] += batch["control"]["scoring_steps"]
                replacing = cohort["age"] >= cohort["lifetime"]
                batch["cohort_metrics"] = dict(cohort_rebuilt=int(fresh), cohorts_built=self.cohorts_built,
                    cohort_age_steps=cohort["age"], cohort_lifetime_steps=cohort["lifetime"],
                    cohort_completed=int(replacing), hardware_sample_age_updates=self.iteration - cohort["born"],
                    cohort_initial_episode_completion_fraction=cohort["first_done"].float().mean().item(),
                    cohort_interrupted_episodes=int((cohort["lengths"] > 0).sum()) if replacing else 0,
                    cohort_interrupted_episode_mean_length=(cohort["lengths"].sum().item() /
                        max(int((cohort["lengths"] > 0).sum()), 1)) if replacing else 0.,
                    resume_cohort_restarts=self.resume_cohort_restarts)
                batch["hardware_behavior"] = (cohort["behavior_mean"], cohort["behavior_std"])
                if replacing:
                    self._close_cohort()
            summary = batch["control"]
            training = summary.pop("controller_training") if summary else dict(
                controller_updates=0, controller_iterations=getattr(self.controller_trainer, "iteration", 0),
                controller_transitions=0, total_controller_transitions=getattr(self.controller_trainer, "total_transitions", 0),
                elapsed_env_steps=self.elapsed_env_steps, controller_storage_bytes_peak=0)
            training.update(controller_training_draws=len(batch["raw"]) if fresh else 0,
                            controller_training_rejections=len(batch["raw"]) - len(batch["valid_indices"]) if fresh else 0,
                            controller_compile_seconds=batch["timings"]["compile_seconds"],
                            controller_build_seconds=batch["timings"]["build_seconds"])
            batch["timings"]["collect_seconds"] = self._time() - started
            training["controller_stage_seconds"] = batch["timings"]["collect_seconds"]
            batch["controller_training"] = training
            self.phase = "collected"
            return batch
        except BaseException:
            self._close_cohort()
            raise

    def _collect_control(self, env, num_designs):
        c = self.config
        count = num_designs * c["replicas"]
        if env.num_envs != count:
            raise ValueError("Controller cohort must have an equal replica budget for every valid design")
        horizon = int(getattr(env.unwrapped, "horizon_steps", 0) or
                      math.ceil(c["episode_length_s"] / env.unwrapped.step_dt))
        if horizon < 1 or (c["horizon_steps"] is not None and horizon != c["horizon_steps"]):
            raise ValueError("Configured horizon_steps differs from native fixed design duration")
        if not self.control_initialized or self.control_schema is None:
            self._ensure_control(_observation(env.reset()), env.num_actions)
        diagnostics = c["staleness_diagnostics"]
        score = _TrainingWindowScore(count, self.device, c["hardware_gamma"], c["terminal_penalty"],
                                    block_steps=c["controller_train_cfg"]["num_steps_per_env"] if diagnostics else None,
                                    replicas=c["replicas"])
        def observe(rewards, dones, extras):
            score.step(rewards, dones, extras)
            if self._cohort is not None:
                cohort = self._cohort
                done = dones.reshape(-1).bool().to(self.device)
                cohort["first_done"] |= done
                cohort["lengths"].add_(1).masked_fill_(done, 0)
        options = (dict(preserve_env_state=True, design_count=num_designs)
                   if self._cohort is not None else dict(design_count=num_designs))
        trajectory = None
        if c["hardware_vtrace"]:
            from .vtrace import WindowTrajectoryRecorder
            expected_steps = c["controller_updates_per_cycle"] * c["controller_train_cfg"]["num_steps_per_env"]
            trajectory = WindowTrajectoryRecorder(count, expected_steps, terminal_penalty=c["terminal_penalty"])
            options["trajectory_observer"] = trajectory
        training = self._ensure_controller_trainer().train(
            env, c["controller_updates_per_cycle"], writer=self.writer,
            elapsed_env_steps=self.elapsed_env_steps, transition_callback=observe, **options)
        valid_indices = self._cohort["template"]["valid_indices"] if self._cohort else range(num_designs)
        training.update(score.block_metrics(valid_indices))
        if diagnostics:
            # Probe design slots enumerate only feasible rows; retain their
            # correspondence with unconditional draw indices in hardware logs.
            for slot, index in enumerate(valid_indices):
                training[f"Controller/Staleness/Design_{slot}/draw_index"] = index
        replica_returns = score.returns.reshape(num_designs, c["replicas"])
        raw_replica_returns = replica_returns
        if trajectory is not None:
            corrected, correction_metrics = self._ensure_vtrace_scorer().score(trajectory, gamma=c["hardware_gamma"])
            replica_returns = corrected.reshape(num_designs, c["replicas"])
            training.update(correction_metrics)
            training["vtrace_design_score_mae_from_raw"] = (
                replica_returns.mean(-1) - raw_replica_returns.mean(-1)).abs().mean().item()
            training["vtrace_target_controller_iteration"] = training["controller_iterations"]
        objective_records = None
        if c["hardware_mass_cost_weight"] > 0:
            from .mass_cost import apply_mass_cost
            gamma, steps = c["hardware_gamma"], score.steps
            discounted_steps = steps if gamma == 1 else (1 - gamma ** steps) / (1 - gamma)
            replica_returns, cost_metrics, objective_records = apply_mass_cost(
                replica_returns, getattr(env.unwrapped, "hardware_masses_kg", None),
                reference_kg=c["hardware_mass_reference_kg"], weight=c["hardware_mass_cost_weight"],
                discounted_seconds=discounted_steps * env.unwrapped.step_dt)
            training.update(cost_metrics)
        training["HardwareTraining/replicas_per_design"] = c["replicas"]
        training["HardwareTraining/between_design_score_std"] = (
            replica_returns.mean(-1).std(unbiased=True).item() if num_designs > 1 else 0.)
        if c["replicas"] > 1:
            spread = replica_returns.std(-1, unbiased=True)
            # IID-replica estimates, not confidence intervals: shared terrain,
            # commands or controller adaptation can correlate replica returns.
            sem = spread / math.sqrt(c["replicas"])
            training["HardwareTraining/replica_return_std_mean"] = spread.mean().item()
            training["HardwareTraining/naive_score_sem_mean"] = sem.mean().item()
            for index, std, error in zip(valid_indices, spread.tolist(), sem.tolist()):
                training[f"HardwareTraining/Design_{index}/replica_return_std"] = std
                training[f"HardwareTraining/Design_{index}/naive_score_sem"] = error
        expected_steps = c["controller_updates_per_cycle"] * c["controller_train_cfg"]["num_steps_per_env"]
        transitions = expected_steps * count
        if score.steps != expected_steps or training["controller_transitions"] != transitions:
            raise ValueError("Controller training must score the complete equal-budget adaptation window")
        self.elapsed_env_steps = int(training["elapsed_env_steps"])
        self.global_step += transitions
        self.reused_training_transitions += transitions
        self.controller_cycles += 1
        self.policy.eval()
        result = dict(replica_returns=replica_returns,
                      scoring_steps=score.steps, score_accumulator_bytes=score.storage_bytes,
                      true_terminations=int(score.true_terminations), timeouts=int(score.timeouts),
                      controller_training=training)
        if objective_records is not None:
            result["hardware_objective_records"] = objective_records
        return result

    def update(self, batch):
        if self.phase != "collected":
            raise RuntimeError("Co-adaptation updates require a complete scored cohort")
        summary, training = batch["control"], batch["controller_training"]
        if summary and summary.get("hardware_objective_records") is not None:
            batch["hardware_objective_records"] = dict(zip(batch["valid_indices"], summary["hardware_objective_records"]))
        batch["control"] = None
        self.policy.eval()
        if self.policy.control is not None:
            for parameter in self.policy.control.parameters():
                parameter.grad = None
        # Native PPO already updated control; this optimizer contains hardware only.
        drift = self._hardware_staleness(batch, "before") if self.config["staleness_diagnostics"] else {}
        metrics = CodesignRunner.update(self, batch)
        if self.config["staleness_diagnostics"]:
            drift.update(self._hardware_staleness(batch, "after"))
        metrics.update(drift)
        metrics.update(batch.get("cohort_metrics", {}))
        metrics["hardware_gamma"] = self.config["hardware_gamma"]
        metrics.update(training)
        metrics.update(control_updates=training["controller_updates"],
                       controller_cycles=self.controller_cycles,
                       hardware_evaluation_transitions=0, total_hardware_evaluation_transitions=0,
                       reused_training_transitions=training["controller_transitions"],
                       total_reused_training_transitions=self.reused_training_transitions,
                       control_transitions=training["controller_transitions"],
                       simulated_transitions=training["controller_transitions"])
        for key in ("scoring_steps", "score_accumulator_bytes", "true_terminations", "timeouts"):
            metrics["hardware_" + key] = summary[key] if summary else 0
        metrics["sps"] = metrics["simulated_transitions"] / max(metrics["iteration_seconds"], 1e-9)
        return metrics

    @torch.no_grad()
    def _hardware_staleness(self, batch, when):
        logprob, _, _ = self.policy.hardware.evaluate(batch["raw"])
        delta = logprob - batch["old_logprob"]
        weights = (delta - delta.max()).exp()
        ess = weights.sum().square() / weights.square().sum()
        prefix = f"HardwareStaleness/{when}/"
        metrics = {prefix + "mean_abs_log_ratio": delta.abs().mean().item(),
                   prefix + "clip_fraction": ((delta.exp() - 1).abs() > self.config["clip_param"]).float().mean().item(),
                   prefix + "sampled_ess_fraction": (ess / len(delta)).item()}
        if "hardware_behavior" in batch:
            mean, std = batch["hardware_behavior"]
            current = self.policy.hardware.distribution()
            metrics[prefix + "kl"] = torch.distributions.kl_divergence(
                torch.distributions.Normal(mean, std), current).sum().item()
        for index, value in enumerate(delta.tolist()):
            metrics[prefix + f"Design_{index}/log_ratio"] = value
        return metrics

    def close(self):
        try:
            self._close_cohort()
        finally:
            super().close()

    def _fresh_warmstart_state(self, *args):
        self._close_cohort()
        super()._fresh_warmstart_state(*args)
        self.reused_training_transitions = 0
        self._vtrace_scorer = None
        self.cohorts_built = self.resume_cohort_restarts = 0

    def _extra_checkpoint_state(self):
        state = super()._extra_checkpoint_state()
        state["coadaptation_state"] = dict(reused_training_transitions=self.reused_training_transitions,
            cohorts_built=self.cohorts_built, resume_cohort_restarts=self.resume_cohort_restarts,
            resume_discards_live_cohort=self._cohort is not None,
            live_cohort_age_steps=self._cohort["age"] if self._cohort else 0,
            vtrace_scorer=self._vtrace_scorer.state_dict() if self._vtrace_scorer else None,
            resume_scope="Controller/hardware optimizer boundary; resume builds a fresh cohort, not simulator continuation")
        return state

    def _load_extra_checkpoint_state(self, data):
        self._close_cohort()
        state = data.get("coadaptation_state")
        if not isinstance(state, dict):
            raise ValueError("Co-adaptation resume requires reused training transition counters")
        total = state.get("reused_training_transitions")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("Invalid co-adaptation training transition counter")
        super()._load_extra_checkpoint_state(data)
        if self.hardware_evaluation_transitions:
            raise ValueError("Co-adaptation cannot resume frozen hardware evaluation transitions")
        self.reused_training_transitions = total
        self.cohorts_built = state.get("cohorts_built", 0)
        self.resume_cohort_restarts = state.get("resume_cohort_restarts", 0) + int(state.get("resume_discards_live_cohort", False))
        self._vtrace_scorer = None
        if state.get("vtrace_scorer") is not None:
            if not self.config["hardware_vtrace"]:
                raise ValueError("V-trace critic requires hardware_vtrace=True")
            self._ensure_vtrace_scorer().load_state_dict(state["vtrace_scorer"])
