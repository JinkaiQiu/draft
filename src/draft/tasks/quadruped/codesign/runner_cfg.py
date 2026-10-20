"""Typed configuration for quadruped geometry co-design."""

from dataclasses import dataclass, field
import math

from draft.tasks.quadruped.runner_cfg import QuadrupedLocomotionRunnerCfg


@dataclass
class CodesignRunnerCfg(QuadrupedLocomotionRunnerCfg):
    run_name: str = "quadruped_codesign"
    logger: str = "tensorboard"
    schedule: str = "fixed"
    n_envs: int = 512
    num_designs: int = 16
    replicas: int = 32
    training_schedule: str = "full_window"
    coadaptation_cohort_lifetime: str = "episode"
    hardware_gamma: float = 1.0
    hardware_vtrace: bool = False
    hardware_vtrace_cfg: dict = field(default_factory=dict)
    staleness_diagnostics: bool = True
    staleness_probe_samples_per_design: int = 64
    controller_updates_per_cycle: int = 20
    controller_schedule: str = "adaptive"
    training_seed_offset: int = 0
    max_training_design_attempts: int = 128
    horizon_steps: int | None = None
    hardware_evaluation_steps: int | None = None
    hardware_bootstrap: bool = False
    rollout_storage_device: str | None = None
    max_control_batch_size: int | None = None
    compiler_workers: int = 0
    max_cache_entries: int = 128
    cache_dir: str = "generated/codesign"
    geometry_bounds: dict = field(default_factory=dict)
    reference_overrides: dict = field(default_factory=dict)
    geometry_relative_range: float = 0.15
    optimize_hardware: bool = True
    hardware_sampling_mode: str = "stochastic"
    normalization_exclusions: dict = field(default_factory=lambda: {
        "actor": tuple(range(-8, -1)), "critic": tuple(range(-8, -1)),
    })
    hardware_std: float = 0.5
    hardware_target_kl: float = 0.02
    hardware_loss_weight: float = 1.0
    hardware_value_loss_coef: float = 0.5
    hardware_entropy_coef: float = 0.0
    hardware_update_schedule: str = "once_per_epoch"
    terminal_penalty: float = -10.0
    invalid_design_return: float = -1000.0
    eval_interval: int = 25
    eval_replicas: int = 16
    final_evaluation: bool = True
    final_export: bool = True
    eval_first_update: bool = False
    eval_video_interval: int = 0
    eval_video_length_s: float = 10.0
    eval_video_fps: int = 25
    eval_video_width: int = 640
    eval_video_height: int = 480

    def validate(self) -> None:
        if isinstance(self.training_seed_offset, bool) or not isinstance(self.training_seed_offset, int) or self.training_seed_offset < 0:
            raise ValueError("training_seed_offset must be a nonnegative integer")
        if self.training_schedule not in ("full_window", "alternating", "coadaptation"):
            raise ValueError("training_schedule must be full_window, alternating, or coadaptation")
        if not isinstance(self.hardware_vtrace, bool):
            raise ValueError("hardware_vtrace must be a boolean")
        if self.hardware_vtrace and self.training_schedule != "coadaptation":
            raise ValueError("hardware_vtrace requires training_schedule='coadaptation'")
        from draft.codesign.vtrace import HardwareVTraceScorer
        HardwareVTraceScorer.validate_config(self.hardware_vtrace_cfg)
        if self.coadaptation_cohort_lifetime not in ("episode", "window"):
            raise ValueError("coadaptation_cohort_lifetime must be 'episode' or 'window'")
        if isinstance(self.hardware_gamma, bool) or not isinstance(self.hardware_gamma, (int, float)) or not math.isfinite(self.hardware_gamma) or not 0 <= self.hardware_gamma <= 1:
            raise ValueError("hardware_gamma must be in [0, 1]")
        if not isinstance(self.staleness_diagnostics, bool):
            raise ValueError("staleness_diagnostics must be a boolean")
        samples = self.staleness_probe_samples_per_design
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("staleness_probe_samples_per_design must be a positive integer")
        if self.controller_schedule not in ("fixed", "adaptive"):
            raise ValueError("controller_schedule must be fixed or adaptive")
        steps = self.hardware_evaluation_steps
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
            raise ValueError("hardware_evaluation_steps must be a positive integer or None")
        if not isinstance(self.hardware_bootstrap, bool):
            raise ValueError("hardware_bootstrap must be a boolean")
        if self.training_schedule == "coadaptation" and (steps is not None or self.hardware_bootstrap):
            raise ValueError("coadaptation scores the controller training window; "
                             "remove hardware_evaluation_steps and hardware_bootstrap")
        if (steps is not None or self.hardware_bootstrap) and self.training_schedule != "alternating":
            raise ValueError("Hardware evaluation limits and bootstrapping require training_schedule='alternating'")
        if self.hardware_bootstrap and steps is None:
            raise ValueError("hardware_bootstrap requires hardware_evaluation_steps")
        for name in ("controller_updates_per_cycle", "max_training_design_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        cap = self.max_control_batch_size
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            raise ValueError("max_control_batch_size must be a positive integer or None")
        if self.hardware_sampling_mode not in ("stochastic", "fixed"):
            raise ValueError("hardware_sampling_mode must be stochastic or fixed")
        if self.hardware_sampling_mode == "fixed" and self.optimize_hardware:
            raise ValueError("Fixed nominal hardware requires optimize_hardware=False")
        if self.num_designs == 1 and self.optimize_hardware:
            raise ValueError("num_designs=1 requires a fixed or frozen hardware policy")
        for name, minimum in (("num_designs", 1), ("replicas", 1),
                              ("compiler_workers", 0), ("max_cache_entries", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.max_training_design_attempts < self.num_designs:
            raise ValueError("max_training_design_attempts must be at least num_designs")
        for name, minimum in (("eval_video_interval", 0), ("eval_video_fps", 1),
                              ("eval_video_width", 2), ("eval_video_height", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
            if name in ("eval_video_width", "eval_video_height") and value % 2:
                raise ValueError(f"{name} must be even for video encoding")
        if not math.isfinite(self.eval_video_length_s) or self.eval_video_length_s <= 0:
            raise ValueError("eval_video_length_s must be finite and positive")
        if self.n_envs != self.num_designs * self.replicas:
            raise ValueError("n_envs must equal num_designs * replicas")
        if not math.isfinite(self.geometry_relative_range) or not 0 < self.geometry_relative_range < 1:
            raise ValueError("geometry_relative_range must be in (0, 1)")
        if self.logger not in ("tensorboard", "wandb", "none"):
            raise ValueError("co-design logger must be tensorboard, wandb, or none")
        if self.std_type != "scalar":
            raise ValueError("co-design currently supports scalar control standard deviations")
        if isinstance(self.num_steps_per_env, bool) or not isinstance(self.num_steps_per_env, int) or self.num_steps_per_env < 1:
            raise ValueError("num_steps_per_env must be a positive integer")
        if self.training_schedule == "full_window" and self.num_steps_per_env != 24:
            raise ValueError("full_window requires num_steps_per_env=24; native alternating and "
                             "coadaptation schedules support configurable controller rollout lengths")


def make_runner_cfg() -> CodesignRunnerCfg:
    return CodesignRunnerCfg()
