"""Optional hardware optimization inside the ordinary quadruped task."""

from __future__ import annotations

import dataclasses


def _native_conditioning(config):
    """Normalize the fixed/randomized special cases without changing PPO knobs."""
    if config is None or config.codesign is None:
        return True
    if not isinstance(config.codesign, dict):
        raise ValueError("codesign must be an optional mapping")
    mode = config.codesign.get("mode", "optimize")
    if mode not in ("fixed", "randomized", "optimize"):
        raise ValueError("codesign.mode must be fixed, randomized, or optimize")
    if mode == "optimize":
        return False
    if mode == "fixed" and "training_schedule" in config.codesign:
        if config.codesign["training_schedule"] != "coadaptation":
            raise ValueError("Fixed cohort controller training requires training_schedule='coadaptation'")
        return False
    # Keep the user-visible mapping intact for metadata and repeat selection.
    config.env_kwargs = {**config.env_kwargs, "morphology_conditioning": dict(config.codesign)}
    return True


def codesign_config(config):
    """Merge explicit geometry options with the ordinary task's run settings."""
    from .codesign.runner_cfg import CodesignRunnerCfg

    options = config.codesign
    if not isinstance(options, dict):
        raise ValueError("codesign must be an optional mapping of hardware-optimization settings")
    options = dict(options)
    mode = options.pop("mode", "optimize")
    fixed_cohort = mode == "fixed" and options.get("training_schedule") == "coadaptation"
    if mode != "optimize" and not fixed_cohort:
        raise ValueError("Cohort training requires codesign.mode=optimize or fixed with training_schedule='coadaptation'")
    if fixed_cohort:
        if options.get("optimize_hardware", False) or options.get("hardware_sampling_mode", "fixed") != "fixed":
            raise ValueError("Fixed cohort training requires optimize_hardware=False and hardware_sampling_mode='fixed'")
        options.update(optimize_hardware=False, hardware_sampling_mode="fixed")
    valid = {field.name for field in dataclasses.fields(CodesignRunnerCfg)} - {"codesign"}
    unknown = set(options) - valid
    if unknown:
        raise ValueError(f"Unknown quadruped codesign settings: {sorted(unknown)}")
    values = {key: value for key, value in dataclasses.asdict(config).items() if key in valid}
    # Hardware PPO retains a fixed rate. Alternating controller training uses
    # the original task's independent native PPO schedule.
    values["controller_schedule"] = config.schedule
    values["schedule"] = "fixed"
    values.update(options)
    values["env_kwargs"] = {**config.env_kwargs, **options.get("env_kwargs", {})}
    values["reward_overrides"] = {**config.reward_overrides, **options.get("reward_overrides", {})}
    designs = values.get("num_designs", 16)
    if "replicas" not in options and isinstance(designs, int) and designs > 0 and values["n_envs"] % designs == 0:
        values["replicas"] = values["n_envs"] // designs
    values["normalization_exclusions"] = {"actor": tuple(range(-7, 0)), "critic": tuple(range(-7, 0))}
    resolved = CodesignRunnerCfg(**values)
    resolved.validate()
    if not resolved.optimize_hardware and not fixed_cohort:
        raise ValueError("codesign.mode=optimize selects hardware optimization; use codesign.mode=fixed or randomized for native controller training")
    if not resolved.env_kwargs.get("robot_xml"):
        raise ValueError("Quadruped codesign requires the same explicit env_kwargs.robot_xml task reference")
    return resolved


def get_training_handler(config):
    if _native_conditioning(config):
        return None
    codesign_config(config)
    return train_codesign


def get_playback_handler(config):
    if _native_conditioning(config):
        return None
    codesign_config(config)
    return play_codesign


def train_codesign(config, **kwargs):
    from .codesign import train
    return train(codesign_config(config), **kwargs)


def play_codesign(config):
    from .codesign import play
    return play(codesign_config(config))


def native_codesign_factory(reference_xml, *, horizon_steps=None):
    """Use the original task/cohort environment with the joint optimizer."""
    from .codesign.env import CoDesignVecEnv
    from .codesign.native_env import build_native_env

    def build(designs, replicas, *, device, episode_length_s, seed, env_kwargs=None, reward_overrides=None):
        task_reference = (env_kwargs or {}).get("task_geometry_xml") or reference_xml
        options = {key: value for key, value in (env_kwargs or {}).items()
                   if key not in ("robot_xml", "task_geometry_xml", "morphology_conditioning")}
        native = build_native_env(designs, replicas, reference_xml=task_reference, device=device,
                                  episode_length_s=episode_length_s, seed=seed, env_kwargs=options,
                                  reward_overrides=reward_overrides, horizon_steps=horizon_steps)
        native.horizon_steps = native.max_episode_length
        if horizon_steps is not None and native.horizon_steps != horizon_steps:
            native.close()
            raise ValueError("Configured horizon_steps differs from native fixed design duration")
        return CoDesignVecEnv(native)
    return build
