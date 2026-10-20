"""The registered task is configurable, and interrupted runs remain playable."""

from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
pytest.importorskip("torch")

from draft.tasks import get_task, list_tasks
from draft.tasks.config_loader import load_run_config
from draft.tasks.quadruped.codesign.playback import _checkpoint_path
from draft.tasks.quadruped.dispatch import codesign_config, native_codesign_factory


def test_registered_default_configuration_is_consistent():
    task = get_task("quadruped")
    assert "quadruped_codesign" not in list_tasks()
    run = load_run_config(task.path / "codesign/config.yaml")
    assert run["task"] == "quadruped"
    cfg = task.make_runner_cfg()
    cfg.apply_overrides(run["runner_overrides"])
    cfg.env_kwargs = run["env_kwargs"]
    optimized = codesign_config(cfg)
    assert optimized.n_envs == optimized.num_designs * optimized.replicas == 512
    assert optimized.max_episode_length == 20
    assert optimized.schedule == "fixed"
    assert cfg.schedule == "adaptive"
    assert optimized.normalization_exclusions["actor"] == tuple(range(-7, 0))
    assert callable(task.training_handler(cfg)) and callable(task.playback_handler(cfg))
    cfg.n_envs = 1
    with pytest.raises(ValueError, match="num_designs"):
        codesign_config(cfg)


def test_fixed_randomized_and_unconditioned_keep_native_learner():
    task = get_task("quadruped")
    cfg = task.make_runner_cfg()
    for conditioning in (False, True, {"mode": "fixed"}, {"mode": "randomized", "num_designs": 16}):
        cfg.env_kwargs = dict(morphology_conditioning=conditioning)
        assert task.training_handler(cfg) is None
        assert task.playback_handler(cfg) is None
    assert cfg.n_envs == 4096 and cfg.num_steps_per_env == 24 and cfg.schedule == "adaptive"


def test_joint_optimization_resolves_full_cohort_with_bounded_control_memory():
    task = get_task("quadruped")
    cfg = task.make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="generated/cheetah/quadruped.xml")
    cfg.codesign = dict(mode="optimize", num_designs=16, replicas=256,
                        rollout_storage_device="cpu", max_control_batch_size=24576)
    optimized = codesign_config(cfg)
    assert optimized.n_envs == optimized.num_designs * optimized.replicas == 4096
    assert optimized.optimize_hardware and optimized.hardware_sampling_mode == "stochastic"
    assert optimized.rollout_storage_device == "cpu" and optimized.max_control_batch_size == 24576
    assert callable(task.training_handler(cfg))


def test_alternating_schedule_keeps_independent_native_controller_configuration():
    from draft.codesign import get_runner_class
    from draft.codesign.alternating import AlternatingCodesignRunner
    from draft.codesign.runner import CodesignRunner
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="generated/cheetah/quadruped.xml")
    cfg.codesign = dict(mode="optimize", training_schedule="alternating",
                        controller_updates_per_cycle=20, num_designs=16, replicas=256,
                        hardware_evaluation_steps=256, hardware_bootstrap=True)
    optimized = codesign_config(cfg)
    assert optimized.n_envs == 4096 and optimized.num_steps_per_env == 24
    assert optimized.schedule == "fixed" and optimized.controller_schedule == "adaptive"
    assert optimized.controller_updates_per_cycle == 20
    assert optimized.hardware_evaluation_steps == 256 and optimized.hardware_bootstrap
    assert optimized.horizon_steps is None and optimized.max_episode_length == 20
    assert get_runner_class(vars(optimized)) is AlternatingCodesignRunner
    assert get_runner_class({}) is CodesignRunner  # Existing saved checkpoints.
    with pytest.raises(ValueError, match="Unknown co-design"):
        get_runner_class(dict(training_schedule="invalid"))


def test_coadaptation_is_an_optional_native_controller_schedule():
    from draft.codesign import get_runner_class
    from draft.codesign.coadaptation import CoadaptationCodesignRunner
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode="optimize", training_schedule="coadaptation",
                        controller_updates_per_cycle=20, num_designs=16, replicas=256)
    optimized = codesign_config(cfg)
    assert optimized.training_schedule == "coadaptation"
    assert optimized.n_envs == 4096 and optimized.num_steps_per_env == 24
    assert optimized.controller_schedule == "adaptive" and optimized.schedule == "fixed"
    assert optimized.hardware_evaluation_steps is None and not optimized.hardware_bootstrap
    assert get_runner_class(vars(optimized)) is CoadaptationCodesignRunner


def test_fixed_cohort_continuation_uses_same_native_controller_schedule():
    from draft.codesign import get_runner_class
    from draft.codesign.coadaptation import CoadaptationCodesignRunner
    task = get_task("quadruped")
    cfg = task.make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml", hardware_aware=True)
    cfg.codesign = dict(mode="fixed", training_schedule="coadaptation",
                        controller_updates_per_cycle=20, num_designs=16, replicas=256,
                        training_seed_offset=1_000_003)
    options = cfg.codesign.copy()
    assert callable(task.training_handler(cfg)) and callable(task.playback_handler(cfg))
    resolved = codesign_config(cfg)
    assert cfg.codesign == options
    assert resolved.n_envs == resolved.num_designs * resolved.replicas == 4096
    assert resolved.num_steps_per_env == 24 and resolved.controller_updates_per_cycle == 20
    assert resolved.seed == 0 and resolved.training_seed_offset == 1_000_003
    assert resolved.training_schedule == "coadaptation"
    assert resolved.controller_schedule == "adaptive" and resolved.schedule == "fixed"
    assert not resolved.optimize_hardware and resolved.hardware_sampling_mode == "fixed"
    assert resolved.env_kwargs == cfg.env_kwargs
    assert get_runner_class(vars(resolved)) is CoadaptationCodesignRunner


@pytest.mark.parametrize("options", [dict(optimize_hardware=True),
                                     dict(hardware_sampling_mode="stochastic"),
                                     dict(training_schedule="alternating")])
def test_fixed_cohort_rejects_hardware_optimization_or_alternating_schedule(options):
    task = get_task("quadruped")
    cfg = task.make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode="fixed", training_schedule="coadaptation") | options
    with pytest.raises(ValueError, match="Fixed cohort"):
        task.training_handler(cfg)


@pytest.mark.parametrize("options", [dict(hardware_evaluation_steps=256),
                                     dict(hardware_bootstrap=True),
                                     dict(hardware_evaluation_steps=256, hardware_bootstrap=True)])
def test_coadaptation_rejects_frozen_trial_settings(options):
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode="optimize", training_schedule="coadaptation", **options)
    with pytest.raises(ValueError, match="coadaptation.*training window"):
        codesign_config(cfg)


@pytest.mark.parametrize("mode,schedule", [("optimize", "alternating"),
                                          ("optimize", "coadaptation"), ("fixed", "coadaptation")])
def test_native_training_entrypoint_wires_both_controller_schedules(monkeypatch, tmp_path, mode, schedule):
    import draft.codesign
    from draft.codesign import compiler
    from draft.tasks.quadruped import dispatch
    from draft.tasks.quadruped.codesign import train

    recorded = {}

    class Compiler:
        fingerprint = "test-source"
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class Runner:
        def __init__(self, policy, compiler, factory, config, **kwargs):
            recorded.update(config=config, metadata=kwargs["metadata"])
        def learn(self, iterations):
            recorded["iterations"] = iterations
        def close(self):
            recorded["closed"] = True

    monkeypatch.setattr(compiler, "DesignCompiler", Compiler)
    monkeypatch.setattr(draft.codesign, "get_runner_class", lambda config: Runner)
    monkeypatch.setattr(dispatch, "native_codesign_factory", lambda xml, **kwargs: object())
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode=mode, training_schedule=schedule)
    optimized = codesign_config(cfg)
    train(optimized, log_dir=tmp_path)
    config = recorded["config"]
    assert config["training_schedule"] == schedule
    assert config["controller_train_cfg"]["num_steps_per_env"] == 24
    assert config["controller_train_cfg"]["algorithm"]["schedule"] == "adaptive"
    assert config["controller_train_cfg"]["actor"]["class_name"] == "draft.codesign.models:ConditionedMLPModel"
    learner = recorded["metadata"]["learner"]
    if mode == "fixed":
        assert "hardware optimizer disabled" in learner and "hardware PPO" not in learner
    else:
        assert ("shared training trajectories" in learner) == (schedule == "coadaptation")
    assert recorded["closed"] and recorded["iterations"] == optimized.max_iterations


@pytest.mark.parametrize("options", [dict(training_schedule="invalid"),
                                     dict(training_schedule="alternating", controller_updates_per_cycle=0),
                                     dict(training_schedule="alternating", controller_schedule="invalid")])
def test_alternating_configuration_rejected_before_environment_creation(options):
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode="optimize", **options)
    with pytest.raises(ValueError):
        codesign_config(cfg)


@pytest.mark.parametrize("options", [
    dict(hardware_evaluation_steps=0), dict(hardware_evaluation_steps=-1),
    dict(hardware_evaluation_steps=True), dict(hardware_evaluation_steps=2.5),
    dict(hardware_bootstrap=1), dict(hardware_bootstrap=True),
])
def test_hardware_evaluation_configuration_rejects_invalid_settings(options):
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode="optimize", training_schedule="alternating", **options)
    with pytest.raises(ValueError):
        codesign_config(cfg)


def test_hardware_evaluation_limit_requires_alternating_schedule():
    cfg = get_task("quadruped").make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(mode="optimize", hardware_evaluation_steps=256)
    with pytest.raises(ValueError, match="require training_schedule='alternating'"):
        codesign_config(cfg)


@pytest.mark.parametrize("options", [dict(mode="fixed"), dict(mode="randomized", num_designs=16, latent_std=.4)])
def test_high_level_codesign_modes_normalize_idempotently_to_native_conditioning(options):
    task = get_task("quadruped")
    cfg = task.make_runner_cfg()
    cfg.codesign = options.copy()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    assert task.training_handler(cfg) is None
    assert task.playback_handler(cfg) is None
    assert cfg.codesign == options
    assert cfg.env_kwargs["morphology_conditioning"] == options
    assert cfg.env_kwargs["robot_xml"] == "stock.xml"
    native = cfg.build_rsl_rl_cfg()
    assert native["actor"]["class_name"] == "draft.codesign.models:ConditionedMLPModel"
    assert cfg.n_envs == 4096 and native["num_steps_per_env"] == 24
    assert native["algorithm"]["schedule"] == "adaptive"


def test_nested_codesign_rejects_unknown_settings_and_controller_only_mode():
    task = get_task("quadruped")
    cfg = task.make_runner_cfg()
    cfg.env_kwargs = dict(robot_xml="stock.xml")
    cfg.codesign = dict(typo=1)
    with pytest.raises(ValueError, match="Unknown"):
        task.training_handler(cfg)
    cfg.codesign = dict(optimize_hardware=False)
    with pytest.raises(ValueError, match="codesign.mode=fixed or randomized"):
        task.training_handler(cfg)


def test_optimized_factory_uses_original_native_environment(monkeypatch):
    from draft.tasks.quadruped.codesign import native_env, env
    calls = []
    def make(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(max_episode_length=123)
    monkeypatch.setattr(native_env, "build_native_env", make)
    def wrap(native, **kwargs):
        assert "clip_actions" not in kwargs
        return native
    monkeypatch.setattr(env, "CoDesignVecEnv", wrap)
    factory = native_codesign_factory("stock.xml")
    result = factory(["design"], 2, device="cpu", episode_length_s=20., seed=7,
                     env_kwargs=dict(robot_xml="stock.xml", task_geometry_xml="reference.xml", morphology_conditioning=True, terrain_scan_fix="reward"))
    assert result.horizon_steps == 123
    assert calls[0][1]["reference_xml"] == "reference.xml"
    assert calls[0][1]["env_kwargs"] == dict(terrain_scan_fix="reward")
    assert calls[0][1]["horizon_steps"] is None

    monkeypatch.setattr(native_env, "build_native_env",
                        lambda *args, **kwargs: (calls.append((args, kwargs)) or
                                                 SimpleNamespace(max_episode_length=1280)))
    factory = native_codesign_factory("stock.xml", horizon_steps=1280)
    result = factory(["design"], 2, device="cpu", episode_length_s=25.6, seed=7)
    assert result.horizon_steps == 1280
    assert calls[-1][1]["horizon_steps"] == 1280


def test_playback_selects_latest_boundary_before_first_periodic_checkpoint(tmp_path):
    run = tmp_path / "codesign" / "timestamp"
    run.mkdir(parents=True)
    latest = run / "model_latest.pt"
    latest.touch()
    cfg = SimpleNamespace(checkpoint=None, log_dir=str(tmp_path), run_name="codesign")
    assert _checkpoint_path(cfg) == latest
    cfg.checkpoint = "zeros"
    with pytest.raises(ValueError, match="joint checkpoint"):
        _checkpoint_path(cfg)
