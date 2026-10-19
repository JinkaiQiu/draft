from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from draft.codesign.alternating import AlternatingCodesignRunner
from draft.codesign.policy import HardwarePolicy, JointPolicy
from test_codesign_runner import Compiler, Control, Env


class Trainer:
    """Exercise runner ordering/state without importing the native GPU stack."""

    def __init__(self, control):
        self.control = control
        self.iteration = 0
        self.optimizer = torch.optim.Adam(control.parameters(), lr=.001)

    def train(self, env, num_updates, writer=None, elapsed_env_steps=0):
        self.control.train()
        self.optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in self.control.parameters()).backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.control.normalizer_count.add_(24 * num_updates)
        self.iteration += num_updates
        self.control.eval()
        return dict(controller_updates=num_updates, controller_iterations=self.iteration,
                    controller_transitions=24 * num_updates * env.count,
                    elapsed_env_steps=elapsed_env_steps + 24 * num_updates)

    def state_dict(self):
        return dict(iteration=self.iteration, optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        self.iteration = state["iteration"]
        self.optimizer.load_state_dict(state["optimizer"])


class NormalizedControl(Control):
    def __init__(self):
        super().__init__()
        self.register_buffer("normalizer_count", torch.zeros(()))

    def evaluate(self, *args, **kwargs):
        assert not self.training
        return super().evaluate(*args, **kwargs)


class RecordingCompiler(Compiler):
    def __init__(self, validity=None):
        super().__init__()
        self.cohorts = []
        self.validity = validity

    def compile_many(self, physical):
        self.cohorts.append(np.asarray(physical).copy())
        valid = self.validity(len(self.cohorts), len(physical)) if self.validity else [True] * len(physical)
        return [SimpleNamespace(design=SimpleNamespace(design_id=str(i)) if ok else None,
                                design_id=str(i), error=None if ok else "infeasible")
                for i, ok in enumerate(valid)]


def make_runner(compiler=None, env_factory=Env, **overrides):
    config = dict(device="cpu", num_designs=2, replicas=2, horizon_steps=4, episode_length_s=4.,
                  gamma=1., num_learning_epochs=3, hardware_target_kl=100.,
                  invalid_design_return=-50., logger="none", controller_updates_per_cycle=2,
                  controller_train_cfg=dict(num_steps_per_env=24))
    config.update(overrides)
    runner = AlternatingCodesignRunner(JointPolicy(NormalizedControl(), HardwarePolicy(2)),
                                      compiler or RecordingCompiler(), env_factory, config)
    runner._ensure_controller_trainer = lambda: ensure_trainer(runner)
    return runner


def ensure_trainer(runner):
    if runner.controller_trainer is None:
        runner.controller_trainer = Trainer(runner.policy.control)
    return runner.controller_trainer


def test_independent_training_then_frozen_streaming_hardware_trials():
    environments, snapshots = [], []
    runner = make_runner()

    class CheckedEnv(Env):
        def __init__(self, *args, **kwargs):
            if environments:
                assert environments[-1].closed  # Destroy training worlds before scoring.
                snapshots.append({key: value.clone() for key, value in runner.policy.control.state_dict().items()})
            super().__init__(*args, **kwargs)
            environments.append(self)

        def step(self, actions):
            assert not runner.policy.training and not runner.policy.control.training
            for key, value in snapshots[-1].items():
                torch.testing.assert_close(runner.policy.control.state_dict()[key], value, rtol=0, atol=0)
            return super().step(actions)

    runner.env_factory = CheckedEnv
    hardware_rng = runner.hardware_generator.get_state().clone()
    expected = torch.Generator().set_state(hardware_rng)
    expected_raw = runner.policy.hardware.sample(2, expected)[0]
    batch = runner.collect()
    torch.testing.assert_close(batch["raw"], expected_raw)
    assert not np.array_equal(runner.compiler.cohorts[0], runner.compiler.cohorts[-1])
    assert batch["scores"].tolist() == [-2., 4.]
    assert "observations" not in batch["control"] and "actions" not in batch["control"]
    snapshot = {key: value.clone() for key, value in runner.policy.control.state_dict().items()}
    metrics = runner.update(batch)
    for key, value in snapshot.items():
        torch.testing.assert_close(runner.policy.control.state_dict()[key], value, rtol=0, atol=0)
    assert metrics["controller_updates"] == metrics["control_updates"] == 2
    assert metrics["hardware_updates"] == 3 and metrics["hardware_records"] == 2
    assert metrics["control_transitions"] == 192
    assert metrics["hardware_evaluation_transitions"] == 16
    assert metrics["simulated_transitions"] == runner.global_step == 208
    assert runner.elapsed_env_steps == 48
    hardware_ids = {id(parameter) for parameter in runner.policy.hardware.parameters()}
    assert {id(parameter) for group in runner.optimizer.param_groups for parameter in group["params"]} == hardware_ids


@pytest.mark.parametrize("horizon", [4, 40])
def test_hardware_trial_memory_has_no_time_by_world_storage(monkeypatch, horizon):
    class LongEnv(Env):
        horizon_steps = horizon
    runner = make_runner(env_factory=LongEnv, horizon_steps=horizon)
    allocated = []
    empty = torch.empty
    def tracked(*args, **kwargs):
        shape = args[0] if isinstance(args[0], (tuple, list)) else args
        allocated.append(tuple(shape))
        return empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", tracked)
    batch = runner.collect()
    summary = batch["control"]
    assert summary["replica_returns"].numel() == 4
    assert not any(len(shape) >= 2 and shape[0] == horizon and shape[1] == 4 for shape in allocated)
    assert all(not isinstance(value, dict) for value in summary.values())


def test_training_resamples_only_its_own_invalid_draws():
    compiler = RecordingCompiler(lambda call, count: [False] * count if call == 1 else [True] * count)
    runner = make_runner(compiler)
    batch = runner.collect()
    assert batch["controller_training"]["controller_training_draws"] == 4
    assert batch["controller_training"]["controller_training_rejections"] == 2
    assert len(compiler.cohorts) == 4
    assert batch["raw"].shape == (2, 2)


def test_partial_training_rejection_preserves_full_cohort_cache_lease():
    compiler = RecordingCompiler(lambda call, count: [True, False] if call == 1 else [True] * count)
    runner = make_runner(compiler, max_training_design_attempts=3)
    batch = runner.collect()
    assert batch["controller_training"]["controller_training_draws"] == 3
    assert len(compiler.cohorts) == 4  # Initial draw, retry, full lease, held-out draw.
    np.testing.assert_array_equal(compiler.cohorts[0][0], compiler.cohorts[1][0])
    np.testing.assert_array_equal(compiler.cohorts[1], compiler.cohorts[2])


@pytest.mark.parametrize("timeout, expected", [(False, -.8125), (True, 1.6875)])
def test_streamed_discounted_returns_penalize_first_fall_but_not_timeout(timeout, expected):
    class TimeoutEnv(Env):
        def step(self, actions):
            obs, rewards, done, extras = super().step(actions)
            extras["time_outs"] = done.clone() if timeout else torch.zeros_like(done)
            return obs, rewards, done, extras
    batch = make_runner(env_factory=TimeoutEnv, gamma=.5).collect()
    assert batch["scores"][0].item() == expected
    assert batch["scores"][1].item() == 1.875


def test_invalid_hardware_records_keep_penalties_and_never_resample():
    compiler = RecordingCompiler(lambda call, count: [call <= 2] * count)
    runner = make_runner(compiler)
    batch = runner.collect()
    assert len(compiler.cohorts) == 3
    assert batch["scores"].tolist() == [-50., -50.]
    assert batch["control"] is None
    metrics = runner.update(batch)
    assert metrics["hardware_records"] == metrics["invalid_designs"] == 2
    assert metrics["hardware_evaluation_transitions"] == 0


def test_resume_preserves_separate_optimizer_counters_and_both_next_draws(tmp_path):
    runner = make_runner()
    runner.update(runner.collect())
    path = tmp_path / "alternating.pt"
    runner.save(path)
    resumed = make_runner()
    resumed.load(path)
    assert resumed.iteration == resumed.controller_cycles == 1
    assert resumed.controller_trainer.iteration == 2
    assert resumed.controller_trainer.optimizer.state
    first = runner.collect()
    second = resumed.collect()
    torch.testing.assert_close(first["raw"], second["raw"], rtol=0, atol=0)
    np.testing.assert_array_equal(runner.compiler.cohorts[-3], resumed.compiler.cohorts[-3])
    assert runner.global_step == resumed.global_step
    for key, value in runner.policy.control.state_dict().items():
        torch.testing.assert_close(value, resumed.policy.control.state_dict()[key], rtol=0, atol=0)


def test_resume_allows_changed_nested_native_runtime_metadata(tmp_path):
    native = dict(num_steps_per_env=24, run_name="source", experiment_name="source", max_iterations=20, log_dir="old/output",
                  algorithm=dict(learning_rate=.001, gamma=.99))
    source = make_runner(controller_train_cfg=native)
    source.update(source.collect())
    checkpoint = tmp_path / "runtime.pt"
    source.save(checkpoint)
    resumed = make_runner(controller_train_cfg=dict(native, run_name="resumed", experiment_name="resumed", max_iterations=100,
                                                   log_dir="new/output"))
    resumed.load(checkpoint)
    assert resumed.iteration == source.iteration
    assert resumed.controller_trainer.iteration == source.controller_trainer.iteration
    assert resumed.controller_trainer.optimizer.state
    assert resumed.config["controller_train_cfg"]["run_name"] == "resumed"


@pytest.mark.parametrize("algorithm", [dict(learning_rate=.002, gamma=.99), dict(learning_rate=.001, gamma=.95),
                                       dict(learning_rate=.001, gamma=.99, schedule="fixed")])
def test_resume_keeps_nested_native_ppo_configuration_protected(tmp_path, algorithm):
    native = dict(num_steps_per_env=24, algorithm=dict(learning_rate=.001, gamma=.99))
    source = make_runner(controller_train_cfg=native)
    checkpoint = tmp_path / "protected.pt"
    source.save(checkpoint)
    resumed = make_runner(controller_train_cfg=dict(native, algorithm=algorithm, run_name="resumed"))
    with pytest.raises(ValueError, match="protected.*controller_train_cfg"):
        resumed.load(checkpoint)


def test_warmstart_preserves_networks_but_resets_hardware_and_all_training_state(tmp_path):
    source = make_runner()
    source.update(source.collect())
    path = tmp_path / "source.pt"
    source.save(path)
    target = make_runner()
    prior = {key: value.clone() for key, value in target.policy.hardware.state_dict().items()}
    training_rng = target.training_generator.get_state().clone()
    target.update(target.collect())
    target.warmstart(path)
    assert target.iteration == target.global_step == target.elapsed_env_steps == target.controller_cycles == 0
    assert target.controller_trainer is None and not target.optimizer.state
    torch.testing.assert_close(target.training_generator.get_state(), training_rng)
    for key, value in source.policy.control.state_dict().items():
        torch.testing.assert_close(target.policy.control.state_dict()[key], value, rtol=0, atol=0)
    for key, value in prior.items():
        torch.testing.assert_close(target.policy.hardware.state_dict()[key], value, rtol=0, atol=0)


def test_training_feasibility_retry_limit_is_bounded():
    runner = make_runner(RecordingCompiler(lambda call, count: [False] * count), max_training_design_attempts=3)
    with pytest.raises(RuntimeError, match="feasible"):
        runner.collect()
    assert sum(len(cohort) for cohort in runner.compiler.cohorts) == 3


class BootstrapCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_observations = []

    def forward(self, observation):
        if isinstance(observation, dict):
            assert not self.training and not torch.is_grad_enabled()
            self.final_observations.append(observation["critic"].clone())
            observation = observation["critic"]
        return observation.new_full((len(observation), 1), 4.)


def test_cap_bootstraps_only_survivors_with_frozen_final_critic_and_logs_contributions():
    class MixedDoneEnv(Env):
        def step(self, actions):
            observation, rewards, done, extras = super().step(actions)
            done[1] = self.steps >= 2  # Timeout replica also resets and must not bootstrap.
            extras["time_outs"][1] = done[1]
            return observation, rewards, done, extras
    runner = make_runner(env_factory=MixedDoneEnv, gamma=.5,
                         hardware_evaluation_steps=2, hardware_bootstrap=True)
    critic = BootstrapCritic()
    runner.policy.control.critic = critic
    batch = runner.collect()
    assert batch["control"]["replica_returns"].tolist() == [[-3.5, 1.5], [2.5, 2.5]]
    assert len(critic.final_observations) == 1
    torch.testing.assert_close(critic.final_observations[0], torch.full((2, 2), 2.))
    control = {key: value.clone() for key, value in runner.policy.control.state_dict().items()}
    metrics = runner.update(batch)
    for key, value in control.items():
        torch.testing.assert_close(runner.policy.control.state_dict()[key], value, rtol=0, atol=0)
    assert metrics["hardware_observed_return_mean"] == .25
    assert metrics["hardware_bootstrap_mean"] == .5
    assert metrics["hardware_evaluation_steps"] == metrics["hardware_evaluation_cap_steps"] == 2
    assert metrics["hardware_native_horizon_steps"] == 4
    assert metrics["hardware_evaluation_capped"] == 1
    assert metrics["hardware_survivors"] == metrics["hardware_bootstrap_replicas"] == 2
    assert metrics["hardware_survival_fraction"] == .5
    assert metrics["hardware_evaluation_transitions"] == 8
    assert metrics["controller_transitions"] == 192
    assert runner.elapsed_env_steps == 48


@pytest.mark.parametrize("cap, bootstrap", [(2, False), (4, True), (20, True)])
def test_no_bootstrap_when_disabled_or_at_natural_trial_horizon(cap, bootstrap):
    runner = make_runner(hardware_evaluation_steps=cap, hardware_bootstrap=bootstrap)
    critic = BootstrapCritic()
    runner.policy.control.critic = critic
    batch = runner.collect()
    assert critic.final_observations == []
    assert batch["control"]["bootstrap_mean"] == batch["control"]["bootstrap_replicas"] == 0
    assert batch["control"]["horizon_steps"] == min(cap, 4)
    assert batch["control"]["native_horizon_steps"] == 4


def test_no_bootstrap_after_all_replicas_have_autoreset():
    class FinishedEnv(Env):
        def step(self, actions):
            observation, rewards, done, extras = super().step(actions)
            done[:] = True
            return observation, rewards, done, extras
    runner = make_runner(env_factory=FinishedEnv, hardware_evaluation_steps=2, hardware_bootstrap=True)
    critic = BootstrapCritic()
    runner.policy.control.critic = critic
    summary = runner.collect()["control"]
    assert critic.final_observations == []
    assert summary["survivors"] == summary["bootstrap_replicas"] == 0
    assert summary["replica_returns"].tolist() == [[-9., -9.], [-9., -9.]]


def test_final_critic_bootstrap_does_not_sample_an_extra_action():
    runner = make_runner(gamma=.5, hardware_evaluation_steps=1, hardware_bootstrap=True)
    runner.policy.control.critic = BootstrapCritic()
    torch_state = torch.get_rng_state().clone()
    bootstrapped = runner._collect_control(Env([None, None], 2), 2)
    after_bootstrap = torch.get_rng_state().clone()
    runner.config["hardware_bootstrap"] = False
    torch.set_rng_state(torch_state)
    observed = runner._collect_control(Env([None, None], 2), 2)
    torch.testing.assert_close(torch.get_rng_state(), after_bootstrap, rtol=0, atol=0)
    torch.testing.assert_close(bootstrapped["replica_returns"], observed["replica_returns"] + 2.)


@pytest.mark.parametrize("bad", ["nan", "shape", "observation_nan", "observation_shape"])
def test_bootstrap_validates_final_observations_and_critic_values(bad):
    class FinalEnv(Env):
        def step(self, actions):
            observation, rewards, done, extras = super().step(actions)
            if bad == "observation_nan":
                observation["critic"].fill_(float("nan"))
            elif bad == "observation_shape":
                observation["critic"] = torch.zeros(self.count, 3)
            return observation, rewards, done, extras
    class BadCritic(BootstrapCritic):
        def forward(self, observation):
            if isinstance(observation, dict):
                if bad == "nan":
                    return torch.full((len(observation["critic"]), 1), float("nan"))
                if bad == "shape":
                    return torch.zeros(len(observation["critic"]), 2)
            return super().forward(observation)
    runner = make_runner(env_factory=FinalEnv, hardware_evaluation_steps=1, hardware_bootstrap=True)
    runner.policy.control.critic = BadCritic()
    with pytest.raises(ValueError, match="(final native observations|bootstrap values)"):
        runner.collect()


def test_hardware_cap_preserves_native_horizon_assertion():
    runner = make_runner(horizon_steps=3, hardware_evaluation_steps=2)
    with pytest.raises(ValueError, match="native fixed design duration"):
        runner.collect()


@pytest.mark.parametrize("settings", [dict(hardware_evaluation_steps=value) for value in (0, -1, True, 1.5)] +
                         [dict(hardware_bootstrap=1), dict(hardware_bootstrap=True)])
def test_invalid_hardware_evaluation_settings_rejected(settings):
    with pytest.raises(ValueError, match="hardware_(evaluation_steps|bootstrap)"):
        make_runner(**settings)


@pytest.mark.parametrize("hardware", [{}, dict(hardware_evaluation_steps=2, hardware_bootstrap=True)])
def test_real_native_controller_blocks_release_storage_and_freeze_normalizers(tmp_path, hardware):
    pytest.importorskip("rsl_rl")
    from draft.codesign.policy import NativeControlPolicy
    observation = {role: torch.zeros(4, 2) for role in ("actor", "critic")}
    actor = dict(hidden_dims=(8,), activation="elu", obs_normalization=True,
                 distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    critic = dict(hidden_dims=(8,), activation="elu", obs_normalization=True)
    control = NativeControlPolicy.from_observation(observation, 1, actor, critic)
    config = dict(make_runner().config, controller_train_cfg=dict(num_steps_per_env=24,
                  algorithm=dict(class_name="PPO", schedule="adaptive", num_learning_epochs=1, num_mini_batches=1)))
    config.update(hardware)
    snapshots, environments = [], []
    runner = None

    class NativeEnv(Env):
        device = "cpu"
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_envs = self.count
            self.common_step_counter = 0
            if environments:
                assert environments[-1].closed
                snapshots.append({key: value.clone() for key, value in control.state_dict().items()})
            environments.append(self)

        def step(self, actions):
            self.common_step_counter += 1
            if snapshots:
                assert not control.training
                for key, value in snapshots[-1].items():
                    torch.testing.assert_close(control.state_dict()[key], value, rtol=0, atol=0)
            return super().step(actions)

    runner = AlternatingCodesignRunner(JointPolicy(control, HardwarePolicy(2)), RecordingCompiler(), NativeEnv, config)
    batch = runner.collect()
    assert runner.controller_trainer.iteration == 2
    assert runner.controller_trainer.alg.storage is None
    assert control.actor.obs_normalizer.count.item() == 48 * 4
    metrics = runner.update(batch)
    assert metrics["controller_storage_bytes_peak"] > 0
    assert metrics["controller_iterations"] == metrics["controller_updates"] == 2
    assert metrics["hardware_updates"] == 3
    for key, value in snapshots[-1].items():
        torch.testing.assert_close(control.state_dict()[key], value, rtol=0, atol=0)
    path = tmp_path / "native_controller.pt"
    runner.save(path)

    class ResumeEnv(Env):
        device = "cpu"
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_envs = self.count

    runner.env_factory = ResumeEnv
    expected = runner.collect()
    restored_control = NativeControlPolicy.from_observation(observation, 1, actor, critic)
    resumed_config = dict(config, controller_train_cfg=dict(config["controller_train_cfg"], run_name="resumed",
                                                           experiment_name="resumed", max_iterations=100, log_dir="new/output"))
    restored = AlternatingCodesignRunner(JointPolicy(restored_control, HardwarePolicy(2)), RecordingCompiler(), ResumeEnv, resumed_config)
    restored.load(path)
    assert restored.controller_trainer.iteration == 2
    assert restored.controller_trainer._pending_optimizer is not None
    actual = restored.collect()
    assert restored.controller_trainer.learning_rate == runner.controller_trainer.learning_rate
    torch.testing.assert_close(expected["raw"], actual["raw"], rtol=0, atol=0)
    torch.testing.assert_close(expected["scores"], actual["scores"], rtol=0, atol=0)
    for key, value in control.state_dict().items():
        torch.testing.assert_close(restored_control.state_dict()[key], value, rtol=0, atol=0)
