"""Full native episode coverage across short hardware/controller updates."""

import json

import pytest
import torch

pytest.importorskip("rsl_rl")

from draft.codesign.coadaptation import CoadaptationCodesignRunner, _TrainingWindowScore
from draft.codesign.metrics import tensorboard_scalars
from draft.codesign.policy import HardwarePolicy, JointPolicy
from test_codesign_alternating import RecordingCompiler
from test_codesign_controller import Env, config, policy


class LongEnv(Env):
    horizon_steps = 1045
    step_dt = 1.

    def __init__(self, designs, replicas, **kwargs):
        super().__init__(len(designs) * replicas)
        self.closed = False
        self.resets = 0

    def reset(self):
        self.resets += 1
        return super().reset()

    def step(self, actions):
        observation, _, _, _ = super().step(actions)
        done = torch.full((self.num_envs,), self.steps % self.horizon_steps == 0, dtype=torch.bool)
        return observation, torch.tensor([1., 1., 2., 2.]), done, {"time_outs": done.clone()}

    def close(self):
        self.closed = True


def make_runner(environments, **overrides):
    native = config()
    native["num_steps_per_env"] = 64
    native["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    cfg = dict(device="cpu", num_designs=2, replicas=2, horizon_steps=1045,
               episode_length_s=1045., controller_updates_per_cycle=4,
               controller_train_cfg=native, logger="none", num_learning_epochs=1,
               hardware_target_kl=100., staleness_diagnostics=False, terminal_penalty=0.)
    cfg.update(overrides)

    def factory(*args, **kwargs):
        env = LongEnv(*args, **kwargs)
        env.horizon_steps = cfg["horizon_steps"]
        environments.append(env)
        return env

    return CoadaptationCodesignRunner(JointPolicy(policy(), HardwarePolicy(2)),
                                      RecordingCompiler(), factory, cfg)


def test_world_and_episode_state_survive_five_hardware_windows_then_replace():
    environments = []
    runner = make_runner(environments)
    original_raw = original_logprob = original_baseline = initial_rng = None
    try:
        for window in range(5):
            batch = runner.collect()
            if not window:
                original_raw = batch["raw"].clone()
                original_logprob = batch["old_logprob"].clone()
                original_baseline = batch["baseline"].clone()
                initial_rng = runner.hardware_generator.get_state().clone()
                resets = environments[0].resets
            assert len(environments) == len(runner.compiler.cohorts) == 1
            assert environments[0].resets == resets
            assert environments[0].steps == (window + 1) * 256
            assert environments[0].closed == (window == 4)
            for key, expected in (("raw", original_raw), ("old_logprob", original_logprob),
                                  ("baseline", original_baseline)):
                torch.testing.assert_close(batch[key], expected, rtol=0, atol=0)
            torch.testing.assert_close(runner.hardware_generator.get_state(), initial_rng, rtol=0, atol=0)
            torch.testing.assert_close(batch["scores"], torch.tensor([256., 512.]))
            metrics = runner.update(batch)
            assert metrics["controller_updates"] == 4 and metrics["hardware_updates"] == 1
            assert metrics["cohort_rebuilt"] == (window == 0)
            assert metrics["cohort_lifetime_steps"] == 1280
            assert metrics["hardware_sample_age_updates"] == window
            if window:
                assert metrics["compile_seconds"] == metrics["build_seconds"] == 0
        assert metrics["controller_completed_episode_count"] == 4
        assert metrics["controller_mean_episode_length"] == 1045
        assert metrics["cohort_initial_episode_completion_fraction"] == 1
        assert metrics["cohort_interrupted_episodes"] == 4
        assert metrics["cohort_interrupted_episode_mean_length"] == 235
        assert runner._cohort is None and runner.controller_trainer._session_env is None
        assert runner.global_step == runner.reused_training_transitions == 5 * 256 * 4
        next_batch = runner.collect()
        assert len(environments) == 2 and not environments[1].closed
        assert not torch.equal(next_batch["raw"], original_raw)
        assert next_batch["cohort_metrics"]["hardware_sample_age_updates"] == 0
        runner.update(next_batch)
    finally:
        runner.close()
    assert all(env.closed for env in environments)


def test_final_actor_probes_and_per_block_design_scores_are_serializable():
    environments = []
    runner = make_runner(environments, staleness_diagnostics=True, staleness_probe_samples_per_design=16)
    try:
        first = runner.update(runner.collect())
        assert first["hardware_gamma"] == 1. and runner.controller_trainer.alg.gamma == .99
        for block in range(4):
            assert first[f"HardwareTraining/Block_{block}/Design_0/score"] == 64
            assert first[f"HardwareTraining/Block_{block}/Design_1/score"] == 128
            assert first[f"controller_staleness_final_block_{block}_probe_count"] == 8
            assert first[f"controller_staleness_final_block_{block}_kl"] >= 0
        assert first["controller_staleness_window_probe_count"] == 32
        assert first["HardwareTraining/between_design_score_std"] == pytest.approx(128 * 2 ** .5)
        assert first["HardwareTraining/replica_return_std_mean"] == 0
        assert first["HardwareTraining/naive_score_sem_mean"] == 0
        assert 0 < first["controller_staleness_window_sampled_ess_fraction"] <= 1.000001
        second = runner.update(runner.collect())
        assert second["HardwareStaleness/before/kl"] > 0
        assert second["hardware_sample_age_updates"] == 1
        json.dumps(second, allow_nan=False)
        tags = tensorboard_scalars(second)
        assert "HardwareStaleness/kl" in tags
        assert "Controller/Cohort/age_steps" not in tags
        assert tags["HardwareTraining/design_score_std"] == second["HardwareTraining/between_design_score_std"]
        assert not any(key.startswith("Diagnostics/controller_staleness") for key in tags)
    finally:
        runner.close()


def test_explicit_1280_episode_matches_replacement_without_survivor_truncation():
    environments = []
    runner = make_runner(environments, horizon_steps=1280, episode_length_s=1280.)
    try:
        for window in range(5):
            metrics = runner.update(runner.collect())
            assert len(environments) == 1
            assert metrics["cohort_age_steps"] == (window + 1) * 256
            assert metrics["cohort_lifetime_steps"] == 1280
            assert metrics["controller_completed_episode_count"] == (4 if window == 4 else 0)
        assert metrics["controller_mean_episode_length"] == 1280
        assert metrics["cohort_initial_episode_completion_fraction"] == 1.
        assert metrics["cohort_interrupted_episodes"] == 0
        assert environments[0].closed and runner._cohort is None
    finally:
        runner.close()


def test_live_cohort_checkpoint_explicitly_restarts_simulator_on_resume(tmp_path):
    source_envs, target_envs = [], []
    source, target = make_runner(source_envs), make_runner(target_envs)
    try:
        source.update(source.collect())
        path = tmp_path / "live_cohort.pt"
        source.save(path)
        payload = torch.load(path, weights_only=False)
        state = payload["coadaptation_state"]
        assert state["resume_discards_live_cohort"] and state["live_cohort_age_steps"] == 256
        assert "not simulator continuation" in state["resume_scope"]
        target.load(path)
        assert target._cohort is None and target.resume_cohort_restarts == 1
        assert target.elapsed_env_steps == 256
        metrics = target.update(target.collect())
        assert metrics["cohort_rebuilt"] == 1 and metrics["cohort_age_steps"] == 256
        assert target_envs[0].steps == 256 and target.elapsed_env_steps == 512
    finally:
        source.close()
        target.close()


def test_hardware_discount_and_block_scores_are_independent_of_controller_gamma():
    score = _TrainingWindowScore(4, "cpu", .5, -10., block_steps=2, replicas=2)
    for _ in range(4):
        score.step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
    torch.testing.assert_close(score.returns, torch.full((4,), 1.875))
    for block in score.blocks:
        torch.testing.assert_close(block, torch.full((2,), 1.5))


def test_simulator_metadata_failure_closes_environment_and_asset_lease():
    from contextlib import contextmanager
    environments, leases = [], []
    runner = make_runner(environments)

    @contextmanager
    def retain(ids):
        leases.append("opened")
        try:
            yield
        finally:
            leases.append("closed")

    runner.compiler.retain_assets = retain
    factory = runner.env_factory

    def broken(*args, **kwargs):
        env = factory(*args, **kwargs)
        env.horizon_steps = "invalid"
        return env

    runner.env_factory = broken
    with pytest.raises(ValueError):
        runner.collect()
    assert environments[0].closed and leases == ["opened", "closed"]
    assert runner._cohort is None
    runner.close()


def test_all_invalid_persistent_draws_update_without_building_worlds():
    environments = []
    runner = make_runner(environments)
    runner.compiler = RecordingCompiler(lambda call, count: [False] * count)
    try:
        metrics = runner.update(runner.collect())
        assert not environments and runner._cohort is None
        assert metrics["invalid_designs"] == 2 and metrics["hardware_updates"] == 1
        assert metrics["controller_updates"] == runner.global_step == 0
    finally:
        runner.close()


@pytest.mark.parametrize("overrides, message", [
    (dict(coadaptation_cohort_lifetime="bad"), "coadaptation_cohort_lifetime"),
    (dict(hardware_gamma=float("nan")), "hardware_gamma"),
    (dict(hardware_gamma=True), "hardware_gamma"),
    (dict(staleness_diagnostics=1), "staleness_diagnostics"),
    (dict(staleness_probe_samples_per_design=0), "staleness_probe_samples_per_design"),
])
def test_invalid_persistent_cohort_settings_fail_before_build(overrides, message):
    environments = []
    with pytest.raises(ValueError, match=message):
        make_runner(environments, **overrides)
    assert not environments
