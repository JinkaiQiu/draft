import copy
import weakref

import pytest
import torch

pytest.importorskip("rsl_rl")
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from draft.codesign.controller import NativeControllerTrainer, _EpisodeMetrics
from draft.codesign.policy import HardwarePolicy, JointPolicy, NativeControlPolicy


def policy(normalization_exclusions=None):
    obs = {key: torch.zeros(4, 9) for key in ("actor", "critic")}
    actor = dict(class_name="draft.codesign.models:ConditionedMLPModel", hidden_dims=(8,), activation="elu",
                 obs_normalization=True, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    critic = dict(class_name="draft.codesign.models:ConditionedMLPModel", hidden_dims=(8,), activation="elu", obs_normalization=True)
    return NativeControlPolicy.from_observation(obs, 2, actor, critic,
                                              normalization_exclusions=normalization_exclusions)


def config():
    return dict(num_steps_per_env=24, algorithm=dict(class_name="PPO", num_learning_epochs=2, num_mini_batches=2,
                learning_rate=.001, schedule="adaptive", desired_kl=.01, gamma=.99, lam=.95,
                clip_param=.2, value_loss_coef=1., entropy_coef=.01, max_grad_norm=1.,
                use_clipped_value_loss=True, rnd_cfg=None, symmetry_cfg=None, share_cnn_encoders=False))


class Env:
    device = "cpu"
    num_actions = 2

    def __init__(self, count=4):
        self.num_envs = count
        self.unwrapped = self
        self.common_step_counter = 0
        self.steps = 0
        self.reset_at = None

    def obs(self):
        value = torch.zeros(self.num_envs, 9)
        value[:, 0] = self.steps % 5 / 5.
        value[:, 1] = .3
        return TensorDict({"actor": value, "critic": value.clone()}, batch_size=[self.num_envs])

    def reset(self):
        self.steps = 0
        self.reset_at = self.common_step_counter
        return self.obs(), {}

    def step(self, actions):
        self.steps += 1
        self.common_step_counter += 1
        rewards = 1 - (actions[:, 0] - .15).square()
        dones = torch.full((self.num_envs,), self.steps % 7 == 0, dtype=torch.bool)
        return self.obs(), rewards, dones, {"time_outs": torch.zeros_like(dones)}


def test_native_24_step_training_changes_only_controller_and_releases_storage(monkeypatch):
    import draft.codesign.controller as module
    storage_refs = []
    original_storage = module.RolloutStorage
    def capture(*args, **kwargs):
        storage = original_storage(*args, **kwargs)
        storage_refs.append(weakref.ref(storage))
        return storage
    monkeypatch.setattr(module, "RolloutStorage", capture)
    joint = JointPolicy(policy(), HardwarePolicy())
    hardware = copy.deepcopy(joint.hardware.state_dict())
    before = copy.deepcopy(joint.control.state_dict())
    trainer = NativeControllerTrainer(joint.control, config(), "cpu")
    env = Env()
    metrics = trainer.train(env, 2, elapsed_env_steps=100)
    assert isinstance(trainer.alg, PPO)
    assert trainer.alg.update.__func__ is PPO.update
    assert metrics["controller_updates"] == metrics["controller_iterations"] == 2
    assert metrics["controller_transitions"] == 4 * 24 * 2
    assert metrics["elapsed_env_steps"] == env.common_step_counter == 148
    assert env.reset_at == 100
    assert 24 * 4 * 18 * 4 <= metrics["controller_storage_bytes_peak"] < 24 * 4 * 200
    assert trainer.alg.storage is None and storage_refs[0]() is None
    assert not joint.control.training
    assert all(parameter.grad is None for parameter in joint.control.parameters())
    assert any(not torch.equal(value, before[key]) for key, value in joint.control.state_dict().items() if key.startswith("actor.mlp"))
    for key, value in joint.hardware.state_dict().items():
        torch.testing.assert_close(value, hardware[key], rtol=0, atol=0)
    for model in (joint.control.actor, joint.control.critic):
        assert model.obs_normalizer.count.item() == 4 * 24 * 2
        torch.testing.assert_close(model.obs_normalizer._mean[:, -7:], torch.zeros(1, 7))
        torch.testing.assert_close(model.obs_normalizer._std[:, -7:] + .01, torch.ones(1, 7))


def test_native_optimizer_learning_rate_and_counters_persist_across_new_cohorts():
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    first = trainer.train(Env(), 1)
    optimizer = trainer.alg.optimizer
    initial_steps = [state["step"].item() for state in optimizer.state.values()]
    second = trainer.train(Env(8), 1, elapsed_env_steps=24)
    assert trainer.alg.optimizer is optimizer
    assert all(after["step"].item() > before for after, before in zip(optimizer.state.values(), initial_steps))
    assert first["controller_storage_bytes_peak"] * 2 == second["controller_storage_bytes_peak"]
    assert second["controller_iterations"] == 2
    assert second["total_controller_transitions"] == (4 + 8) * 24
    assert trainer.learning_rate == optimizer.param_groups[0]["lr"]
    assert trainer.alg.schedule == "adaptive"


def test_pending_checkpoint_load_reproduces_next_native_update():
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    trainer.train(Env(), 1)
    saved = copy.deepcopy(trainer.state_dict())
    torch.manual_seed(99)
    trainer.train(Env(), 1, elapsed_env_steps=24)
    expected = copy.deepcopy(trainer.control.state_dict())
    restored = NativeControllerTrainer(policy(), config(), "cpu")
    restored.load_state_dict(saved)
    assert restored.alg is None
    assert restored.iteration == 1
    assert restored.learning_rate == saved["learning_rate"]
    torch.manual_seed(99)
    restored.train(Env(), 1, elapsed_env_steps=24)
    for key, value in restored.control.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert restored.learning_rate == trainer.learning_rate
    assert restored.total_transitions == trainer.total_transitions


def test_shared_writer_receives_native_controller_metrics_without_a_new_writer():
    from types import SimpleNamespace
    calls = []
    writer = SimpleNamespace(add_scalar=lambda *args: calls.append(args))
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    trainer.train(Env(), 1, writer=writer)
    assert any(tag == "Controller/learning_rate" and step == 1 for tag, _, step in calls)
    assert {tag for tag, _, _ in calls if tag.startswith("Controller/Loss/")} == {
        "Controller/Loss/value", "Controller/Loss/surrogate", "Controller/Loss/entropy"}


@pytest.mark.parametrize("option", [dict(rnd_cfg={}), dict(symmetry_cfg={}), dict(multi_gpu_cfg={}),
                                    dict(share_cnn_encoders=True)])
def test_unsupported_native_extensions_rejected(option):
    cfg = config()
    cfg["algorithm"].update(option)
    with pytest.raises(ValueError):
        NativeControllerTrainer(policy(), cfg, "cpu")


def test_initial_state_and_zero_updates_need_no_environment():
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    state = copy.deepcopy(trainer.state_dict())
    assert state["optimizer_state_dict"] is None
    restored = NativeControllerTrainer(policy(), config(), "cpu")
    restored.load_state_dict(state)
    metrics = restored.train(None, 0, elapsed_env_steps=12)
    assert metrics["controller_updates"] == 0 and metrics["elapsed_env_steps"] == 12
    assert restored.alg is None


def test_actual_trailing_conditioning_exclusions_survive_native_inference_collection():
    exclusions = {role: tuple(range(-7, 0)) for role in ("actor", "critic")}
    trainer = NativeControllerTrainer(policy(exclusions), config(), "cpu")
    metrics = trainer.train(Env(), 2)
    assert metrics["controller_updates"] == 2
    for model in (trainer.control.actor, trainer.control.critic):
        normalizer = model.obs_normalizer
        assert not normalizer._std.is_inference()
        torch.testing.assert_close(normalizer._mean[:, -7:], torch.zeros(1, 7), rtol=0, atol=0)
        torch.testing.assert_close(normalizer._std[:, -7:] + normalizer.eps, torch.ones(1, 7), rtol=0, atol=0)
    # Re-entering native training and checkpointing must also remain valid.
    trainer.train(Env(), 1, elapsed_env_steps=48)
    state = copy.deepcopy(trainer.state_dict())
    restored = NativeControllerTrainer(policy(exclusions), config(), "cpu")
    restored.load_state_dict(state)
    assert restored.iteration == 3


class Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        assert (tag, step) not in self.scalars
        self.scalars[tag, step] = float(value)


class EpisodeEnv(Env):
    """Two fall/reset periods, one long survivor and one 48-step timeout."""

    def __init__(self):
        super().__init__()
        self.log = {}
        self.metric = torch.zeros(())

    def step(self, actions):
        observation, _, _, _ = super().step(actions)
        dones = torch.tensor([self.steps % 5 == 0, self.steps % 8 == 0,
                              False, self.steps % 48 == 0])
        timeouts = dones & torch.tensor([False, True, False, True])
        # Reuse dictionaries and a tensor exactly as native managers may do.
        self.metric.fill_(self.steps)
        self.log.clear()
        self.log["Metrics/step"] = self.metric
        return observation, torch.tensor([1., 2., 3., 4.]), dones, {
            "time_outs": timeouts, "log": self.log}


def test_exact_completed_and_partial_episode_statistics_across_native_blocks():
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    writer = Writer()
    metrics = trainer.train(EpisodeEnv(), 2, writer=writer)
    assert metrics["controller_completed_episode_count"] == 16
    assert metrics["controller_mean_episode_reward"] == pytest.approx(333 / 16)
    assert metrics["controller_mean_episode_length"] == pytest.approx(141 / 16)
    assert metrics["controller_mean_transition_reward"] == 2.5
    assert writer.scalars["Train/completed_episodes", 1] == 7
    assert writer.scalars["Train/completed_episodes", 2] == 9
    assert writer.scalars["Train/mean_reward", 1] == pytest.approx(68 / 7)
    assert writer.scalars["Train/mean_reward", 2] == pytest.approx(265 / 9)
    assert writer.scalars["Train/mean_episode_length", 2] == pytest.approx(97 / 9)
    assert writer.scalars["Controller/PartialEpisode/mean_length", 2] == 25.5
    assert {tag for tag, _ in writer.scalars if tag.startswith("Controller/PartialEpisode/")} == {
        "Controller/PartialEpisode/mean_length"}
    assert metrics["controller_partial_episode_count"] == 2
    assert not any(tag.startswith("Metrics/") for tag, _ in writer.scalars)
    assert ("Policy/mean_std", 2) in writer.scalars


def test_streaming_extras_weight_native_means_sum_termination_counts_and_copy_values():
    tracker, writer = _EpisodeMetrics(3, "cpu"), Writer()
    # Empty/stale reset means must not poison later completed-episode means.
    tracker.step(torch.zeros(3), torch.zeros(3, dtype=torch.bool), {"episode": {
        "Episode_Reward/alive": float("nan"), "Episode_Metrics/height": float("nan"),
        "Episode_Termination/fall": float("nan"), "score": float("nan")}})
    reward = torch.tensor(2.)
    log = {"Episode_Reward/alive": reward, "Episode_Metrics/height": 3.,
           "Episode_Termination/fall": 1, "Metrics/speed": torch.tensor([1., 3.]), "score": 4.}
    tracker.step(torch.ones(3), torch.tensor([True, False, False]), {"log": log})
    reward.fill_(8.)
    log.update({"Episode_Metrics/height": 9., "Episode_Termination/timeout": 1,
                "Metrics/speed": torch.tensor([5., 7.]), "score": 10.})
    tracker.step(torch.ones(3), torch.tensor([False, True, True]), {"episode": log})
    log.clear()
    reward.fill_(1000.)
    tracker.finish_block(writer, 1, transitions=9)
    assert writer.scalars["Episode_Reward/alive", 1] == 6.
    assert writer.scalars["Episode_Termination/fall", 1] == 2.
    assert writer.scalars["Episode_Termination/timeout", 1] == 1.
    assert not any(tag.startswith(("Episode_Metrics/", "Metrics/", "Episode/")) for tag, _ in writer.scalars)
    assert not tracker.extras


def test_raw_reward_tracking_precedes_native_ppo_timeout_bootstrap(monkeypatch):
    original = PPO.process_env_step
    def mutate_reward(self, obs, rewards, dones, extras):
        original(self, obs, rewards, dones, extras)
        # Guard against implementations that bootstrap the argument in place.
        rewards.add_(1000.)
    monkeypatch.setattr(PPO, "process_env_step", mutate_reward)
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    metrics = trainer.train(EpisodeEnv(), 2)
    assert metrics["controller_mean_episode_reward"] == pytest.approx(333 / 16)
    assert metrics["controller_mean_transition_reward"] == 2.5


def test_no_completions_are_partial_only_and_tracking_resets_for_new_cohort():
    class UnfinishedEnv(Env):
        def step(self, actions):
            observation, _, _, _ = super().step(actions)
            return observation, torch.ones(self.num_envs), torch.zeros(self.num_envs, dtype=torch.bool), {}
    trainer, writer = NativeControllerTrainer(policy(), config(), "cpu"), Writer()
    first = trainer.train(UnfinishedEnv(), 2, writer=writer)
    assert first["controller_completed_episode_count"] == 0
    assert "controller_mean_episode_reward" not in first
    assert not any(tag in ("Train/mean_reward", "Train/mean_episode_length") for tag, _ in writer.scalars)
    assert writer.scalars["Train/completed_episodes", 2] == 0
    assert writer.scalars["Controller/PartialEpisode/mean_length", 2] == 48
    assert first["controller_partial_episode_count"] == 4
    second = trainer.train(UnfinishedEnv(), 1, writer=writer, elapsed_env_steps=48)
    assert second["controller_partial_episode_mean_length"] == 24
    assert second["controller_partial_episode_mean_return"] == 24


def test_episode_logging_does_not_change_native_weights_optimizer_or_rng():
    control = policy()
    initial = copy.deepcopy(control.state_dict())
    with_metrics = NativeControllerTrainer(control, config(), "cpu")
    other_control = policy()
    other_control.load_state_dict(initial)
    without_metrics = NativeControllerTrainer(other_control, config(), "cpu")
    torch.manual_seed(431)
    with_metrics.train(EpisodeEnv(), 2, writer=Writer())
    expected_rng = torch.get_rng_state().clone()
    class NoExtrasEnv(EpisodeEnv):
        def step(self, actions):
            observation, reward, done, extras = super().step(actions)
            return observation, reward, done, {"time_outs": extras["time_outs"]}
    torch.manual_seed(431)
    without_metrics.train(NoExtrasEnv(), 2)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, value in with_metrics.control.state_dict().items():
        torch.testing.assert_close(without_metrics.control.state_dict()[key], value, rtol=0, atol=0)
    expected = with_metrics.alg.optimizer.state_dict()
    actual = without_metrics.alg.optimizer.state_dict()
    assert expected["param_groups"] == actual["param_groups"]
    for parameter, state in expected["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual["state"][parameter][key], value, rtol=0, atol=0)


@pytest.mark.parametrize("updates", [0, 1])
def test_transition_callback_must_be_callable_before_environment_or_controller_mutation(updates):
    trainer, env = NativeControllerTrainer(policy(), config(), "cpu"), Env()
    with pytest.raises(TypeError, match="transition_callback"):
        trainer.train(env, updates, elapsed_env_steps=100, transition_callback=3)
    assert env.reset_at is None and env.common_step_counter == 0
    assert trainer.alg is None and trainer.control.training and not trainer._training


def test_transition_callback_observes_every_raw_step_before_ppo_and_retains_no_transitions(monkeypatch):
    original = PPO.process_env_step
    pending, observed, reward_refs = [], [], []

    def observe(rewards, dones, extras):
        assert torch.is_inference_mode_enabled()
        torch.testing.assert_close(rewards, torch.tensor([1., 2., 3., 4.]), rtol=0, atol=0)
        pending.append((id(rewards), id(dones), id(extras)))
        observed.append((dones.clone(), extras["time_outs"].clone()))
        reward_refs.append(weakref.ref(rewards))

    def process(self, obs, rewards, dones, extras):
        assert pending.pop() == (id(rewards), id(dones), id(extras))
        torch.testing.assert_close(rewards, torch.tensor([1., 2., 3., 4.]), rtol=0, atol=0)
        original(self, obs, rewards, dones, extras)
        # A callback must consume values before PPO can modify its arguments.
        rewards.add_(1000.)

    monkeypatch.setattr(PPO, "process_env_step", process)
    trainer = NativeControllerTrainer(policy(), config(), "cpu")
    metrics = trainer.train(EpisodeEnv(), 2, transition_callback=observe)
    assert len(observed) == 48 and not pending
    for step, (done, timeout) in enumerate(observed, 1):
        expected_done = torch.tensor([step % 5 == 0, step % 8 == 0, False, step % 48 == 0])
        torch.testing.assert_close(done, expected_done)
        torch.testing.assert_close(timeout, expected_done & torch.tensor([False, True, False, True]))
    assert metrics["controller_mean_transition_reward"] == 2.5
    assert all(reference() is None for reference in reward_refs)


def test_read_only_transition_callback_preserves_native_training_metrics_and_rng():
    control = policy()
    initial = copy.deepcopy(control.state_dict())
    with_callback = NativeControllerTrainer(control, config(), "cpu")
    other_control = policy()
    other_control.load_state_dict(initial)
    without_callback = NativeControllerTrainer(other_control, config(), "cpu")
    total_rewards = torch.zeros(4)

    def observe(rewards, dones, extras):
        total_rewards.add_(rewards)

    torch.manual_seed(911)
    expected = with_callback.train(EpisodeEnv(), 2, transition_callback=observe)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(911)
    actual = without_callback.train(EpisodeEnv(), 2)
    torch.testing.assert_close(total_rewards, torch.tensor([48., 96., 144., 192.]), rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, value in with_callback.control.state_dict().items():
        torch.testing.assert_close(without_callback.control.state_dict()[key], value, rtol=0, atol=0)
    expected_optimizer = with_callback.alg.optimizer.state_dict()
    actual_optimizer = without_callback.alg.optimizer.state_dict()
    assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
    for parameter, state in expected_optimizer["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][parameter][key], value, rtol=0, atol=0)
    for key in expected:
        if key not in ("controller_collect_seconds", "controller_update_seconds"):
            assert actual[key] == expected[key]


@pytest.mark.parametrize("updates", [0, 1])
@pytest.mark.parametrize("missing", ["before_step", "after_step"])
def test_trajectory_observer_methods_validated_before_mutation(updates, missing):
    class Observer:
        def before_step(self, obs, actions, log_prob):
            pass

        def after_step(self, rewards, dones, extras):
            pass

    observer = Observer()
    setattr(observer, missing, 3)
    trainer, env = NativeControllerTrainer(policy(), config(), "cpu"), Env()
    with pytest.raises(TypeError, match=f"trajectory_observer.{missing}"):
        trainer.train(env, updates, elapsed_env_steps=100, trajectory_observer=observer)
    assert env.reset_at is None and env.common_step_counter == 0
    assert trainer.alg is None and trainer.control.training and not trainer._training


def test_trajectory_observer_pairs_sampled_behavior_with_raw_outcomes(monkeypatch):
    cfg = config()
    cfg["num_steps_per_env"] = 4
    trainer = NativeControllerTrainer(policy(), cfg, "cpu")
    events, retained = [], []
    original = PPO.process_env_step

    class CheckedEnv(EpisodeEnv):
        def step(self, actions):
            assert events[-1] == "before"
            events.append("env")
            return super().step(actions)

    class Observer:
        def before_step(self, obs, actions, actual_behavior_log_prob):
            assert torch.is_inference_mode_enabled()
            assert obs is trainer.alg.transition.observations
            assert actions is trainer.alg.transition.actions
            assert actual_behavior_log_prob is trainer.alg.transition.actions_log_prob
            assert trainer.control.actor.obs_normalizer.count.item() == len(retained) * 4
            retained.append((obs["actor"].clone(), actions.clone(), actual_behavior_log_prob.clone()))
            events.append("before")

        def after_step(self, rewards, dones, extras):
            assert events[-1] == "env"
            torch.testing.assert_close(rewards, torch.tensor([1., 2., 3., 4.]), rtol=0, atol=0)
            retained[-1] += (rewards.clone(), dones.clone(), extras["time_outs"].clone())
            events.append("after")

    def process(self, obs, rewards, dones, extras):
        assert events[-1] == "after"
        events.append("ppo")
        original(self, obs, rewards, dones, extras)
        rewards.add_(1000.)

    monkeypatch.setattr(PPO, "process_env_step", process)
    metrics = trainer.train(CheckedEnv(), 2, trajectory_observer=Observer())
    assert events == [item for _ in range(8) for item in ("before", "env", "after", "ppo")]
    assert len(retained) == 8
    assert retained[0][0][0, 0] == 0
    assert retained[1][0][0, 0] == pytest.approx(.2)
    assert all(sample[3].tolist() == [1., 2., 3., 4.] for sample in retained)
    assert metrics["controller_mean_transition_reward"] == 2.5


def test_passive_trajectory_observer_preserves_native_training_metrics_weights_and_rng():
    initial = copy.deepcopy(policy().state_dict())
    observed = NativeControllerTrainer(policy(), config(), "cpu")
    plain = NativeControllerTrainer(policy(), config(), "cpu")
    observed.control.load_state_dict(initial)
    plain.control.load_state_dict(initial)

    class Observer:
        def __init__(self):
            self.before = self.after = 0

        def before_step(self, obs, actions, actual_behavior_log_prob):
            self.before += 1
            assert actual_behavior_log_prob.shape == (4,)

        def after_step(self, rewards, dones, extras):
            self.after += 1

    observer = Observer()
    torch.manual_seed(911)
    expected = observed.train(EpisodeEnv(), 2, trajectory_observer=observer)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(911)
    actual = plain.train(EpisodeEnv(), 2)
    assert (observer.before, observer.after) == (48, 48)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, value in observed.control.state_dict().items():
        torch.testing.assert_close(plain.control.state_dict()[key], value, rtol=0, atol=0)
    expected_optimizer, actual_optimizer = observed.alg.optimizer.state_dict(), plain.alg.optimizer.state_dict()
    assert expected_optimizer["param_groups"] == actual_optimizer["param_groups"]
    for parameter, state in expected_optimizer["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][parameter][key], value, rtol=0, atol=0)
    for key in expected:
        if key not in ("controller_collect_seconds", "controller_update_seconds"):
            assert actual[key] == expected[key]


@pytest.mark.parametrize("failure_method,expected_steps", [("before_step", 0), ("after_step", 1)])
def test_trajectory_observer_exception_releases_rollout_state(failure_method, expected_steps):
    class RaisingObserver:
        def before_step(self, obs, actions, actual_behavior_log_prob):
            if failure_method == "before_step":
                raise RuntimeError("observer failure")

        def after_step(self, rewards, dones, extras):
            if failure_method == "after_step":
                raise RuntimeError("observer failure")

    cfg = config()
    cfg["num_steps_per_env"] = 2
    trainer, env = NativeControllerTrainer(policy(), cfg, "cpu"), Env()
    with pytest.raises(RuntimeError, match="observer failure"):
        trainer.train(env, 1, trajectory_observer=RaisingObserver(), preserve_env_state=True)
    assert env.steps == expected_steps
    assert not trainer._training and not trainer.control.training
    assert trainer._session_env is trainer._session_obs is trainer._session_episodes is None
    assert trainer.alg.storage is None
    assert trainer.train(Env(), 1)["controller_updates"] == 1


def test_preserved_environment_continues_observation_and_partial_episodes_across_calls():
    class LongEpisodeEnv(Env):
        def __init__(self):
            super().__init__()
            self.resets = 0

        def reset(self):
            self.resets += 1
            return super().reset()

        def step(self, actions):
            observation, rewards, _, extras = super().step(actions)
            return observation, torch.ones_like(rewards), torch.zeros_like(rewards, dtype=torch.bool), extras

    cfg = config()
    cfg["num_steps_per_env"] = 4
    trainer, env = NativeControllerTrainer(policy(), cfg, "cpu"), LongEpisodeEnv()
    first = trainer.train(env, 1, preserve_env_state=True)
    second = trainer.train(env, 1, elapsed_env_steps=4, preserve_env_state=True)
    assert env.resets == 1 and env.steps == 8
    assert first["controller_partial_episode_mean_length"] == 4
    assert second["controller_partial_episode_mean_length"] == 8
    assert second["controller_partial_episode_mean_return"] == 8
    assert second["controller_mean_transition_reward"] == 1
    trainer.release_environment(env)
    assert trainer._session_env is trainer._session_obs is trainer._session_episodes is None
    third = trainer.train(env, 1, elapsed_env_steps=8, preserve_env_state=True)
    assert env.resets == 2 and third["controller_partial_episode_mean_length"] == 4


@pytest.mark.parametrize("design_count", [1, 2, 4])
def test_staleness_probes_are_bounded_by_design_and_leave_training_rng_unchanged(design_count):
    cfg = config()
    cfg["num_steps_per_env"] = 4
    cfg["staleness_diagnostics"] = True
    cfg["staleness_probe_samples_per_design"] = 3
    initial = policy().state_dict()
    diagnosed, plain = NativeControllerTrainer(policy(), cfg, "cpu"), NativeControllerTrainer(policy(),
                                                                                             dict(cfg, staleness_diagnostics=False), "cpu")
    diagnosed.control.load_state_dict(initial)
    plain.control.load_state_dict(initial)
    writer = Writer()
    torch.manual_seed(811)
    metrics = diagnosed.train(Env(4), 2, writer=writer, design_count=design_count)
    diagnosed_rng = torch.get_rng_state().clone()
    torch.manual_seed(811)
    plain.train(Env(4), 2, design_count=design_count)
    torch.testing.assert_close(torch.get_rng_state(), diagnosed_rng, rtol=0, atol=0)
    for key, value in diagnosed.control.state_dict().items():
        torch.testing.assert_close(value, plain.control.state_dict()[key], rtol=0, atol=0)
    for design in range(design_count):
        prefix = f"controller_staleness_window_design_{design}_"
        assert metrics[prefix + "probe_count"] == 3
        assert 0 < metrics[prefix + "sampled_ess"] <= 3
        assert 0 < metrics[prefix + "sampled_ess_fraction"] <= 1
        assert metrics[prefix + "kl"] >= 0
        assert 0 <= metrics[prefix + "clip_fraction"] <= 1
        assert (f"Controller/Staleness/Window/Design_{design}/kl", 2) not in writer.scalars
        assert metrics[f"controller_staleness_block_0_design_{design}_probe_count"] > 0
        assert metrics[f"controller_staleness_block_1_design_{design}_probe_count"] > 0
        assert metrics[f"controller_staleness_final_block_0_design_{design}_probe_count"] > 0
        assert metrics[f"controller_staleness_final_block_1_design_{design}_probe_count"] > 0
        assert metrics[prefix + "ratio_mean"] > 0
        assert 0 <= metrics[prefix + "rho_clipped_fraction"] <= 1
        assert (f"Controller/Staleness/FinalBlock_0/Design_{design}/mean_log_ratio", 2) not in writer.scalars
    staleness_tags = {tag for tag, _ in writer.scalars if tag.startswith("Controller/Staleness/")}
    assert staleness_tags == {f"Controller/Staleness/Window/{key}" for key in (
        "kl", "clip_fraction", "sampled_ess_fraction", "early_late_kl_gap")}
    assert writer.scalars["Controller/Staleness/Window/early_late_kl_gap", 2] == pytest.approx(
        metrics["controller_staleness_final_block_0_kl"] - metrics["controller_staleness_final_block_1_kl"])
    controller_tags = {tag for tag, _ in writer.scalars if tag.startswith("Controller/")}
    assert len(controller_tags) <= 10
    assert not any("Design_" in tag or "Block_" in tag for tag in controller_tags)
    assert not any(tag in controller_tags for tag in (
        "Controller/transitions", "Controller/collection_seconds", "Controller/update_seconds"))
