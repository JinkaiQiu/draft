"""Native PPO integration of the opt-in latest-controller hardware scorer."""

import copy
import json

import pytest
import torch

pytest.importorskip("rsl_rl")

from draft.codesign.metrics import tensorboard_scalars
from draft.codesign.vtrace import HardwareVTraceScorer
from test_codesign_alternating import RecordingCompiler
from test_codesign_controller import config
from test_codesign_persistent_cohort import make_runner as persistent_runner


def make_runner(*, enabled=True, full_window=False, **overrides):
    native = config()
    native["num_steps_per_env"] = 64 if full_window else 8
    native["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    cfg = dict(hardware_vtrace=enabled, controller_train_cfg=native,
               hardware_vtrace_cfg=dict(value_epochs=1, batch_size=128),
               horizon_steps=256 if full_window else 64)
    cfg.update(overrides)
    return persistent_runner([], **cfg)


def test_complete_64_by_4_window_uses_latest_controller_without_changing_native_learning_or_rng():
    plain, scored = make_runner(enabled=False, full_window=True), make_runner(full_window=True)
    scored.policy.control.load_state_dict(copy.deepcopy(plain.policy.control.state_dict()))
    try:
        torch.manual_seed(923)
        raw = plain.collect()
        expected_rng = torch.get_rng_state().clone()
        expected = copy.deepcopy(plain.policy.control.state_dict())
        torch.manual_seed(923)
        corrected = scored.collect()
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key, value in scored.policy.control.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        training = corrected["controller_training"]
        assert training["controller_updates"] == training["vtrace_target_controller_iteration"] == 4
        assert training["vtrace_trained_windows"] == 1
        assert training["vtrace_raw_mean"] == raw["scores"].mean().item()
        assert training["vtrace_value_fit_in_sample"] == 1
        assert training["vtrace_storage_bytes"] > 0
        assert 0 < training["vtrace_action_ess_fraction"] <= 1.000001
        assert torch.isfinite(corrected["scores"]).all()
        assert not torch.equal(corrected["scores"], raw["scores"])
        assert scored._vtrace_scorer.optimizer.state
        assert all(parameter.grad is None for parameter in scored.policy.control.parameters())
        metrics = scored.update(corrected)
        assert metrics["hardware_evaluation_transitions"] == 0
        assert metrics["reused_training_transitions"] == metrics["simulated_transitions"] == 4 * 256
        tags = tensorboard_scalars(metrics)
        assert {tag for tag in tags if tag.startswith("VTrace/")} == {
            "VTrace/score_correction", "VTrace/action_ess_fraction", "VTrace/clipped_fraction", "VTrace/value_loss"}
        assert not any(tag.startswith("Diagnostics/vtrace_") for tag in tags)
        json.dumps(metrics, allow_nan=False)
    finally:
        plain.close()
        scored.close()


def test_value_optimizer_checkpoint_roundtrip_and_warmstart_reset(tmp_path):
    source = make_runner(coadaptation_cohort_lifetime="window")
    target = make_runner(coadaptation_cohort_lifetime="window")
    try:
        source.update(source.collect())
        path = tmp_path / "vtrace.pt"
        source.save(path)
        target.load(path)
        assert target._vtrace_scorer.trained_windows == 1
        expected = source.collect()
        target.load(path)  # Restore the same controller action RNG as source's boundary.
        actual = target.collect()
        torch.testing.assert_close(actual["scores"], expected["scores"], rtol=0, atol=0)
        assert actual["controller_training"]["vtrace_value_loss"] == expected["controller_training"]["vtrace_value_loss"]
        target.update(actual)
        target.warmstart(path)
        assert target._vtrace_scorer is None and target.controller_trainer is None
        assert target.global_step == target.reused_training_transitions == 0
    finally:
        source.close()
        target.close()


def test_raw_checkpoints_predating_the_flag_still_resume_with_flag_off(tmp_path):
    source, target = make_runner(enabled=False), make_runner(enabled=False)
    enabled = make_runner()
    try:
        path = tmp_path / "raw.pt"
        source.save(path)
        payload = torch.load(path, weights_only=False)
        del payload["config"]["hardware_vtrace"]
        del payload["config"]["hardware_vtrace_cfg"]
        del payload["coadaptation_state"]["vtrace_scorer"]
        torch.save(payload, path)
        # The old feature had the same default critic settings implicitly.
        target.config["hardware_vtrace_cfg"] = HardwareVTraceScorer.validate_config()
        target.load(path)
        assert target._vtrace_scorer is None
        with pytest.raises(ValueError, match="protected.*hardware_vtrace"):
            enabled.load(path)
    finally:
        source.close()
        target.close()
        enabled.close()


def test_all_invalid_hardware_does_not_allocate_a_value_model():
    runner = make_runner()
    runner.compiler = RecordingCompiler(lambda call, count: [False] * count)
    try:
        metrics = runner.update(runner.collect())
        assert metrics["invalid_designs"] == 2 and metrics["hardware_updates"] == 1
        assert metrics["controller_updates"] == 0 and runner._vtrace_scorer is None
        assert runner.global_step == 0
    finally:
        runner.close()


@pytest.mark.parametrize("schedule", ["alternating", "full_window"])
def test_typed_vtrace_only_supports_coadaptation(schedule):
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg
    with pytest.raises(ValueError, match="hardware_vtrace requires"):
        CodesignRunnerCfg(training_schedule=schedule, hardware_vtrace=True).validate()


def test_typed_flag_and_value_config_validation():
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg
    cfg = CodesignRunnerCfg(training_schedule="coadaptation", hardware_vtrace=True,
                           num_steps_per_env=64, controller_updates_per_cycle=4)
    cfg.validate()
    assert cfg.hardware_vtrace
    with pytest.raises(ValueError, match="boolean"):
        make_runner(hardware_vtrace=1)
    with pytest.raises(ValueError, match="Unknown V-trace"):
        make_runner(hardware_vtrace_cfg={"misspelled_clip": 1})
