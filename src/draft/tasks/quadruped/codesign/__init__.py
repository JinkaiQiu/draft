"""Native geometry-conditioned quadruped control and hardware optimization."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path


def make_runner_cfg():
    from .runner_cfg import make_runner_cfg as factory
    return factory()


def play(runner_cfg) -> None:
    from .playback import play_checkpoint
    play_checkpoint(runner_cfg)


def train(runner_cfg, *, log_dir: str | Path, resume: str | None = None,
          metadata: dict | None = None) -> None:
    """Task-owned learner lifecycle, selected by the ordinary training script."""
    from draft.codesign.compiler import DesignCompiler
    from draft.codesign import get_runner_class
    from draft.codesign.space import DesignSpace
    from draft.tasks.quadruped.dispatch import native_codesign_factory

    runner_cfg.validate()
    space = DesignSpace.quadruped(bounds=runner_cfg.geometry_bounds,
                                  reference_overrides=runner_cfg.reference_overrides,
                                  relative_range=runner_cfg.geometry_relative_range)
    config = dataclasses.asdict(runner_cfg)
    config["episode_length_s"] = runner_cfg.max_episode_length
    native = runner_cfg.build_rsl_rl_cfg()
    for role in ("actor", "critic"):
        native[role]["class_name"] = "draft.codesign.models:ConditionedMLPModel"
    config.update(actor_cfg=native["actor"], critic_cfg=native["critic"],
                  obs_groups=native["obs_groups"], environment_family="quadruped_native",
                  normalization_exclusions={role: tuple(range(-7, 0)) for role in ("actor", "critic")})
    native_controller = runner_cfg.training_schedule in ("alternating", "coadaptation")
    if native_controller:
        native["algorithm"]["schedule"] = runner_cfg.controller_schedule
        if runner_cfg.training_schedule == "coadaptation":
            native.update(staleness_diagnostics=runner_cfg.staleness_diagnostics,
                          staleness_probe_samples_per_design=runner_cfg.staleness_probe_samples_per_design)
        config["controller_train_cfg"] = native
    learners = {
        "alternating": f"alternating native {runner_cfg.num_steps_per_env}-step controller PPO and frozen-controller hardware PPO",
        "coadaptation": f"native {runner_cfg.num_steps_per_env}-step controller PPO and hardware PPO scored on shared training trajectories",
        "full_window": "full-window joint PPO (custom control optimizer)",
    }
    if runner_cfg.training_schedule == "coadaptation" and runner_cfg.hardware_vtrace:
        learners["coadaptation"] += "; latest-controller V-trace hardware scoring"
    if not runner_cfg.optimize_hardware:
        learners[runner_cfg.training_schedule] = "native controller PPO with fixed hardware; hardware optimizer disabled"
    runner_type = get_runner_class(config)
    build_env = native_codesign_factory(runner_cfg.env_kwargs["robot_xml"],
                                        horizon_steps=runner_cfg.horizon_steps)
    with DesignCompiler(space, runner_cfg.cache_dir, workers=runner_cfg.compiler_workers,
                        max_cache_entries=runner_cfg.max_cache_entries) as compiler:
        runner = runner_type(None, compiler, build_env, config, log_dir=log_dir,
                                metadata={**(metadata or {}), "compiler_fingerprint": compiler.fingerprint,
                                          "learner": learners[runner_cfg.training_schedule],
                                          "environment": "original quadruped native cohort"})
        if resume or runner_cfg.warmstart_checkpoint:
            import gc
            from draft.codesign.runner import _observation
            probe = build_env([compiler.compile(space.center)], 1, device=runner_cfg.device,
                              episode_length_s=runner_cfg.max_episode_length, seed=runner_cfg.seed,
                              env_kwargs=runner_cfg.env_kwargs, reward_overrides=runner_cfg.reward_overrides)
            try:
                runner._ensure_control(_observation(probe.reset()), probe.num_actions)
            finally:
                probe.close()
                del probe
                gc.collect()
        if resume:
            runner.load(resume)
        elif runner_cfg.warmstart_checkpoint:
            runner.warmstart(runner_cfg.warmstart_checkpoint)
        try:
            runner.learn(runner_cfg.max_iterations)
        finally:
            runner.close()


def __getattr__(name):
    if name == "make_env_cfg":
        return importlib.import_module(".env", __name__).make_codesign_env_cfg
    raise AttributeError(name)
