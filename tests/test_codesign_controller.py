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
    assert any(tag.startswith("Controller/Loss/") for tag, _, _ in calls)


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
    assert writer.scalars["Controller/PartialEpisode/count", 1] == 3
    assert writer.scalars["Controller/PartialEpisode/count", 2] == 2
    assert writer.scalars["Controller/PartialEpisode/mean_return", 2] == 73.5
    assert writer.scalars["Controller/PartialEpisode/mean_length", 2] == 25.5
    assert writer.scalars["Controller/PartialEpisode/world_fraction", 2] == .5
    assert metrics["controller_partial_episode_count"] == 2
    assert writer.scalars["Metrics/step", 1] == 12.5
    assert writer.scalars["Metrics/step", 2] == 36.5
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
    assert writer.scalars["Episode_Metrics/height", 1] == 7.
    assert writer.scalars["Episode_Termination/fall", 1] == 2.
    assert writer.scalars["Episode_Termination/timeout", 1] == 1.
    assert writer.scalars["Metrics/speed", 1] == 4.
    assert writer.scalars["Episode/score", 1] == 8.
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
    assert writer.scalars["Controller/PartialEpisode/mean_return", 2] == 48
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
