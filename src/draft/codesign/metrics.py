"""Compact TensorBoard views of the complete, unchanged co-design metrics."""

from collections.abc import Mapping


_TAGS = {
    "design_score_mean": "codesign/design_score",
    "hardware_policy_loss": "HardwareLoss/policy",
    "hardware_value_loss": "HardwareLoss/value",
    "hardware_kl": "HardwareLoss/kl",
    "hardware_updates": "HardwareLoss/updates",
    "control_policy_loss": "ControlLoss/policy",
    "control_value_loss": "ControlLoss/value",
    "control_kl": "ControlLoss/kl",
    "hardware_observed_return_mean": "HardwareEvaluation/observed_return",
    "hardware_bootstrap_mean": "HardwareEvaluation/bootstrap_contribution",
    "hardware_survival_fraction": "HardwareEvaluation/survival_fraction",
    "compile_seconds": "Perf/CoDesign/compile_seconds",
    "build_seconds": "Perf/CoDesign/build_seconds",
    "rollout_seconds": "Perf/CoDesign/rollout_seconds",
    "hardware_collect_seconds": "Perf/CoDesign/hardware_collect_seconds",
    "iteration_seconds": "Perf/CoDesign/iteration_seconds",
    "sps": "Perf/transitions_per_second",
    "controller_compile_seconds": "Perf/Controller/compile_seconds",
    "controller_build_seconds": "Perf/Controller/build_seconds",
    "controller_collect_seconds": "Perf/Controller/rollout_seconds",
    "controller_update_seconds": "Perf/Controller/update_seconds",
    "controller_stage_seconds": "Perf/Controller/stage_seconds",
    "controller_cycles": "Progress/cycles",
    "controller_iterations": "Progress/controller_updates",
    "total_controller_transitions": "Progress/total_controller_transitions",
    "total_hardware_evaluation_transitions": "Progress/total_hardware_transitions",
    "elapsed_env_steps": "Progress/elapsed_env_steps",
    "cohort_interrupted_episodes": "Controller/Cohort/interrupted_episodes",
}

# Constants and redundant summaries remain available in metrics.jsonl/config.
_OMIT = {
    "iteration", "global_step", "hardware_records", "invalid_designs", "collect_seconds",
    "hardware_active_transitions", "hardware_evaluation_transitions", "hardware_survivors",
    "hardware_horizon_steps", "hardware_native_horizon_steps", "hardware_evaluation_steps",
    "hardware_evaluation_cap_steps", "hardware_evaluation_capped", "hardware_bootstrap_enabled",
    "hardware_bootstrap_replicas", "controller_updates", "controller_transitions",
    "controller_learning_rate", "controller_mean_reward", "controller_mean_transition_reward",
    "controller_completed_episode_count", "controller_mean_episode_reward", "controller_mean_episode_length",
    "controller_partial_episode_mean_return", "controller_partial_episode_mean_length",
    "controller_partial_episode_count", "controller_partial_episode_world_fraction",
    "controller_training_draws", "controller_training_rejections",
    "reused_training_transitions", "total_reused_training_transitions", "hardware_scoring_steps",
    "hardware_score_accumulator_bytes", "hardware_true_terminations", "hardware_timeouts", "hardware_gamma",
    "cohort_rebuilt", "cohorts_built", "cohort_age_steps", "cohort_lifetime_steps", "cohort_completed",
    "cohort_initial_episode_completion_fraction", "cohort_interrupted_episode_mean_length",
    "hardware_sample_age_updates", "resume_cohort_restarts",
}

# A fixed dashboard budget, independent of design count and PPO block count.
# Hardware staleness is measured before this window's hardware optimizer update.
# Detailed before/after, per-block and per-design values remain in metrics.jsonl.
_SUMMARY_TAGS = {
    "HardwareTraining/between_design_score_std": "HardwareTraining/design_score_std",
    "HardwareTraining/naive_score_sem_mean": "HardwareTraining/estimated_score_error",
    "HardwareTraining/first_last_block_ranking_disagreement": "HardwareTraining/ranking_disagreement",
    "HardwareStaleness/before/kl": "HardwareStaleness/kl",
    "HardwareStaleness/before/clip_fraction": "HardwareStaleness/clip_fraction",
    "HardwareStaleness/before/sampled_ess_fraction": "HardwareStaleness/ess_fraction",
    "Controller/Cohort/interrupted_episodes": "Controller/Cohort/interrupted_episodes",
}
for _name in ("kl", "clip_fraction", "sampled_ess_fraction", "early_late_kl_gap"):
    _tag = "Controller/Staleness/Window/" + _name
    _SUMMARY_TAGS[_tag] = _tag
_SUMMARY_TAGS.update({tag: tag for tag in tuple(_SUMMARY_TAGS.values())})
_COMPACT_GROUPS = ("HardwareTraining/", "HardwareStaleness/", "Controller/Cohort/", "Controller/Staleness/")


def _summary_tag(tag):
    if tag.startswith(_COMPACT_GROUPS):
        return _SUMMARY_TAGS.get(tag)
    return tag


def tensorboard_scalars(metrics: Mapping[str, float]) -> dict[str, float]:
    """Route scalar cards without changing raw metrics or creating duplicate tags.

    Native controller blocks already emit their own losses and episode metrics.
    HardwareStd contains the Gaussian's latent standard deviations; HardwareMean
    contains decoded physical dimensions. Explicit namespaced tags take priority.
    Hardware score quality and staleness each have three summary cards; cohort
    bookkeeping and detailed probes are available only in raw metrics.
    """
    native_control = "controller_iterations" in metrics
    result = {tag: value for key, value in metrics.items() if "/" in key
              and (tag := _summary_tag(key)) is not None}
    for key, value in metrics.items():
        if "/" in key or key in _OMIT or native_control and key.startswith("controller_loss_"):
            continue
        if key.startswith("controller_staleness_"):
            # These probes are already written on the native controller-update axis.
            continue
        if key.startswith("hardware_mean_"):
            tag = "HardwareMean/" + key.removeprefix("hardware_mean_")
        elif key.startswith("hardware_latent_std_"):
            tag = "HardwareStd/latent_" + key.removeprefix("hardware_latent_std_")
        elif key == "controller_storage_bytes_peak":
            tag, value = "Memory/controller_rollout_mib", value / 2**20
        elif key == "update_seconds":
            tag = "Perf/CoDesign/" + ("hardware_update_seconds" if native_control else "joint_update_seconds")
        elif key == "control_updates":
            if native_control:
                continue
            tag = "ControlLoss/updates"
        elif key in ("simulated_transitions", "control_transitions"):
            if native_control:
                continue
            tag = "Progress/" + key
        else:
            tag = _TAGS.get(key, "Diagnostics/" + key)
        tag = _summary_tag(tag)
        if tag is not None:
            result.setdefault(tag, value)
    records = metrics.get("hardware_records", 0)
    if records > 0 and "invalid_designs" in metrics:
        result.setdefault("codesign/valid_fraction", 1. - metrics["invalid_designs"] / records)
    return result
