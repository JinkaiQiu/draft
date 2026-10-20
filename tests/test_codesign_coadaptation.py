import numpy as np
import pytest

torch = pytest.importorskip("torch")

from draft.codesign.coadaptation import CoadaptationCodesignRunner, _TrainingWindowScore
from draft.codesign.policy import HardwarePolicy, JointPolicy
from test_codesign_alternating import NormalizedControl, RecordingCompiler
from test_codesign_runner import Env


class TrainingEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_envs = self.count


class Trainer:
    """CPU trainer that delivers every raw transition and makes real Adam steps."""

    def __init__(self, control):
        self.control = control
        self.iteration = self.total_transitions = 0
        self.calls = 0
        self.optimizer = torch.optim.Adam(control.parameters(), lr=.001)

    def train(self, env, num_updates, writer=None, elapsed_env_steps=0, transition_callback=None, **kwargs):
        self.calls += 1
        self.control.train()
        env.reset()
        for _ in range(num_updates):
            for _ in range(24):
                _, rewards, dones, extras = env.step(torch.zeros(env.count, 1))
                transition_callback(rewards, dones, extras)
                # Native PPO may add a critic bootstrap in place after callback.
                rewards[extras["time_outs"]] += 999.
            self.optimizer.zero_grad(set_to_none=True)
            sum(parameter.square().sum() for parameter in self.control.parameters()).backward()
            self.optimizer.step()
            self.control.normalizer_count.add_(24)
            self.iteration += 1
        transitions = 24 * num_updates * env.count
        self.total_transitions += transitions
        self.control.eval()
        return dict(controller_updates=num_updates, controller_iterations=self.iteration,
                    controller_transitions=transitions, total_controller_transitions=self.total_transitions,
                    elapsed_env_steps=elapsed_env_steps + 24 * num_updates)

    def state_dict(self):
        return dict(iteration=self.iteration, total_transitions=self.total_transitions,
                    optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        self.iteration = state["iteration"]
        self.total_transitions = state["total_transitions"]
        self.optimizer.load_state_dict(state["optimizer"])


def make_runner(compiler=None, env_factory=TrainingEnv, lazy_control=False, **overrides):
    config = dict(device="cpu", num_designs=2, replicas=2, horizon_steps=4, episode_length_s=4.,
                  gamma=1., num_learning_epochs=3, hardware_target_kl=100., logger="none",
                  coadaptation_cohort_lifetime="window", staleness_diagnostics=False,
                  controller_updates_per_cycle=2, controller_train_cfg=dict(num_steps_per_env=24))
    config.update(overrides)
    runner = CoadaptationCodesignRunner(
        JointPolicy(None if lazy_control else NormalizedControl(), HardwarePolicy(2)),
        compiler or RecordingCompiler(), env_factory, config)

    def ensure_trainer():
        if runner.controller_trainer is None:
            runner.controller_trainer = Trainer(runner.policy.control)
        return runner.controller_trainer

    runner._ensure_controller_trainer = ensure_trainer
    return runner


def test_one_unconditional_cohort_is_frozen_while_controller_adapts():
    environments = []
    runner = make_runner()
    hardware = {key: value.clone() for key, value in runner.policy.hardware.state_dict().items()}
    control = {key: value.clone() for key, value in runner.policy.control.state_dict().items()}
    expected_rng = torch.Generator().set_state(runner.hardware_generator.get_state())
    raw, old_logprob, _, baseline = runner.policy.hardware.sample(2, expected_rng)

    class CheckedEnv(TrainingEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            environments.append(self)

        def step(self, actions):
            for key, value in hardware.items():
                torch.testing.assert_close(runner.policy.hardware.state_dict()[key], value, rtol=0, atol=0)
            return super().step(actions)

    runner.env_factory = CheckedEnv
    batch = runner.collect()
    torch.testing.assert_close(batch["raw"], raw, rtol=0, atol=0)
    torch.testing.assert_close(batch["old_logprob"], old_logprob, rtol=0, atol=0)
    torch.testing.assert_close(batch["baseline"], baseline, rtol=0, atol=0)
    assert len(runner.compiler.cohorts) == len(environments) == runner.controller_trainer.calls == 1
    assert environments[0].closed
    assert batch["scores"].tolist() == [-187., 48.]
    assert batch["control"]["scoring_steps"] == 48
    assert set(batch["control"]) == {"replica_returns", "scoring_steps", "score_accumulator_bytes",
                                     "true_terminations", "timeouts"}
    assert any(not torch.equal(value, runner.policy.control.state_dict()[key]) for key, value in control.items())
    trained_control = {key: value.clone() for key, value in runner.policy.control.state_dict().items()}
    metrics = runner.update(batch)
    for key, value in trained_control.items():
        torch.testing.assert_close(runner.policy.control.state_dict()[key], value, rtol=0, atol=0)
    assert metrics["hardware_updates"] == 3
    assert metrics["control_updates"] == metrics["controller_updates"] == 2
    assert metrics["hardware_evaluation_transitions"] == metrics["total_hardware_evaluation_transitions"] == 0
    assert metrics["reused_training_transitions"] == metrics["total_reused_training_transitions"] == 192
    assert metrics["simulated_transitions"] == metrics["control_transitions"] == runner.global_step == 192
    assert metrics["hardware_scoring_steps"] == runner.elapsed_env_steps == 48
    assert runner.controller_cycles == 1
    hardware_ids = {id(parameter) for parameter in runner.policy.hardware.parameters()}
    assert {id(parameter) for group in runner.optimizer.param_groups for parameter in group["params"]} == hardware_ids
    assert not hardware_ids & {id(parameter) for group in runner.controller_trainer.optimizer.param_groups
                               for parameter in group["params"]}


def test_partial_invalid_records_keep_penalty_without_feasible_resampling():
    compiler = RecordingCompiler(lambda call, count: [True, False, True])
    runner = make_runner(compiler, num_designs=3)
    batch = runner.collect()
    assert len(compiler.cohorts) == 1 and len(compiler.cohorts[0]) == 3
    assert batch["valid_indices"] == [0, 2]
    assert batch["scores"].tolist() == [-187., -1000., 48.]
    assert batch["compilation_errors"] == [None, "infeasible", None]
    assert batch["raw"].shape == (3, 2)
    assert batch["controller_training"]["controller_training_draws"] == 3
    assert batch["controller_training"]["controller_training_rejections"] == 1
    assert runner.update(batch)["hardware_records"] == 3


def test_all_invalid_draws_update_hardware_without_environment_or_controller(tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("No controller or environment is needed for invalid draws")

    compiler = RecordingCompiler(lambda call, count: [False] * count)
    runner = make_runner(compiler, env_factory=unexpected, lazy_control=True)
    runner._ensure_controller_trainer = unexpected
    batch = runner.collect()
    assert batch["scores"].tolist() == [-1000., -1000.]
    assert len(compiler.cohorts) == 1
    assert batch["control"] is None and runner.policy.control is None and runner.controller_trainer is None
    metrics = runner.update(batch)
    assert metrics["hardware_updates"] == 3
    assert metrics["invalid_designs"] == metrics["hardware_records"] == 2
    assert metrics["control_updates"] == metrics["reused_training_transitions"] == metrics["simulated_transitions"] == 0
    assert runner.global_step == runner.controller_cycles == runner.elapsed_env_steps == 0
    checkpoint = tmp_path / "invalid.pt"
    runner.save(checkpoint)
    restored = make_runner(compiler, env_factory=unexpected, lazy_control=True)
    restored.load(checkpoint)
    assert restored.controller_trainer is None and restored.reused_training_transitions == 0


def test_window_score_includes_timeouts_failures_autoresets_and_unfinished_trajectories():
    score = _TrainingWindowScore(3, "cpu", gamma=.5, terminal_penalty=-10.)
    score.step(torch.tensor([1., 2., 3.]), torch.tensor([True, True, False]),
               {"time_outs": torch.tensor([True, False, False])})
    score.step(torch.tensor([4., 5., 6.]), torch.tensor([True, False, False]),
               {"time_outs": torch.tensor([True, False, False]),
                "codesign_terminated": torch.tensor([True, False, False])})
    score.step(torch.tensor([7., 8., 9.]), torch.tensor([True, True, False]),
               {"time_outs": torch.tensor([False, True, False])})
    # No reset of discount or exclusion of rewards after any earlier done.
    torch.testing.assert_close(score.returns, torch.tensor([-2.75, -3.5, 8.25]))
    assert score.steps == 3 and score.discount == .125
    assert int(score.true_terminations) == 3 and int(score.timeouts) == 2


@pytest.mark.parametrize("steps", [24, 480, 960])
def test_score_storage_is_constant_in_window_duration(steps):
    score = _TrainingWindowScore(4, "cpu", .99, -10.)
    memory = score.storage_bytes
    for _ in range(steps):
        score.step(torch.ones(4), torch.zeros(4), {})
    assert score.storage_bytes == memory == 4 * 4 + 2 * 8
    assert all(value.ndim <= 1 for value in vars(score).values() if isinstance(value, torch.Tensor))
    assert score.returns.numel() == 4


def test_discount_progresses_across_native_ppo_updates():
    runner = make_runner(hardware_gamma=.5, terminal_penalty=0.)
    batch = runner.collect()
    torch.testing.assert_close(batch["scores"], torch.full((2,), 2.))
    assert batch["control"]["scoring_steps"] == 48


def test_twenty_native_updates_score_all_480_steps_with_constant_score_storage():
    runner = make_runner(controller_updates_per_cycle=20)
    batch = runner.collect()
    assert batch["control"]["scoring_steps"] == 480
    assert batch["control"]["score_accumulator_bytes"] == 4 * 4 + 2 * 8
    assert batch["scores"].tolist() == [-1915., 480.]
    metrics = runner.update(batch)
    assert metrics["controller_updates"] == 20
    assert metrics["reused_training_transitions"] == runner.global_step == 1920
    assert metrics["hardware_evaluation_transitions"] == 0


def test_training_timeouts_do_not_bootstrap_hardware_score():
    class TimeoutEnv(TrainingEnv):
        def step(self, actions):
            observation, rewards, done, extras = super().step(actions)
            done[1] = self.steps >= 2
            extras["time_outs"][1] = done[1]
            return observation, rewards, done, extras

    batch = make_runner(env_factory=TimeoutEnv).collect()
    assert batch["scores"].tolist() == [-187., 48.]
    assert batch["control"]["timeouts"] == 47


def test_resume_preserves_rng_independent_optimizers_and_transition_counters(tmp_path):
    source = make_runner()
    source.update(source.collect())
    path = tmp_path / "coadaptation.pt"
    source.save(path)
    target = make_runner()
    target.load(path)
    assert target.controller_trainer.iteration == 2 and target.controller_trainer.optimizer.state
    assert target.iteration == target.controller_cycles == 1
    assert target.global_step == target.reused_training_transitions == 192
    assert target.hardware_evaluation_transitions == 0
    first, second = source.collect(), target.collect()
    for key in ("raw", "old_logprob", "baseline", "scores"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    np.testing.assert_array_equal(source.compiler.cohorts[-1], target.compiler.cohorts[-1])
    assert target.global_step == source.global_step == 384
    assert target.elapsed_env_steps == source.elapsed_env_steps == 96
    for key, value in source.policy.control.state_dict().items():
        torch.testing.assert_close(value, target.policy.control.state_dict()[key], rtol=0, atol=0)


def test_warmstart_resets_adaptation_counters_and_optimizer(tmp_path):
    source = make_runner()
    source.update(source.collect())
    path = tmp_path / "source.pt"
    source.save(path)
    target = make_runner()
    prior = {key: value.clone() for key, value in target.policy.hardware.state_dict().items()}
    target.update(target.collect())
    target.warmstart(path)
    assert target.reused_training_transitions == target.global_step == target.controller_cycles == 0
    assert target.controller_trainer is None and not target.optimizer.state
    for key, value in prior.items():
        torch.testing.assert_close(target.policy.hardware.state_dict()[key], value, rtol=0, atol=0)


def test_resume_rejects_switching_training_schedule(tmp_path):
    from draft.codesign.alternating import AlternatingCodesignRunner

    source = make_runner()
    path = tmp_path / "coadaptation.pt"
    source.save(path)
    target = AlternatingCodesignRunner(
        JointPolicy(NormalizedControl(), HardwarePolicy(2)), RecordingCompiler(), TrainingEnv,
        dict(source.config, training_schedule="alternating"))
    with pytest.raises(ValueError, match="protected.*training_schedule"):
        target.load(path)


def test_phase_validation_prevents_recollect_update_and_save(tmp_path):
    runner = make_runner()
    batch = runner.collect()
    with pytest.raises(RuntimeError, match="updated"):
        runner.collect()
    with pytest.raises(RuntimeError, match="boundaries"):
        runner.save(tmp_path / "incomplete.pt")
    runner.update(batch)
    with pytest.raises(RuntimeError, match="complete scored cohort"):
        runner.update(batch)


def test_training_exception_closes_environment_and_blocks_incomplete_checkpoint(tmp_path):
    environments = []

    class BrokenEnv(TrainingEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            environments.append(self)

        def step(self, actions):
            raise RuntimeError("failed simulator")

    runner = make_runner(env_factory=BrokenEnv)
    with pytest.raises(RuntimeError, match="failed simulator"):
        runner.collect()
    assert len(environments) == 1 and environments[0].closed
    assert runner.phase != "boundary"
    with pytest.raises(RuntimeError, match="boundaries"):
        runner.save(tmp_path / "broken.pt")


@pytest.mark.parametrize("config", [dict(hardware_evaluation_steps=4),
                                    dict(hardware_evaluation_steps=4, hardware_bootstrap=True)])
def test_rejects_frozen_trial_cap_and_bootstrap(config):
    with pytest.raises(ValueError, match="Co-adaptation"):
        make_runner(**config)


def test_rejects_incomplete_callback_window_and_cleans_up():
    runner = make_runner()

    class IncompleteTrainer(Trainer):
        def train(self, env, num_updates, **kwargs):
            return super().train(env, num_updates - 1, **kwargs)

    runner.controller_trainer = IncompleteTrainer(runner.policy.control)
    with pytest.raises(ValueError, match="complete equal-budget"):
        runner.collect()
    assert runner.global_step == runner.reused_training_transitions == 0


@pytest.mark.parametrize("rewards, dones", [(torch.ones(2), torch.zeros(3)),
                                           (torch.tensor([float("nan")] * 3), torch.zeros(3))])
def test_score_rejects_invalid_transitions(rewards, dones):
    with pytest.raises(ValueError):
        _TrainingWindowScore(3, "cpu", 1., -10.).step(rewards, dones, {})
