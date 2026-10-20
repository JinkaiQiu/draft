import copy

import pytest
import torch
from torch import nn

from draft.codesign.vtrace import (HardwareVTraceScorer, WindowTrajectoryRecorder,
                                   finite_window_vtrace)


class _Distribution:
    def update(self, mean):
        self.mean = mean


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.mlp.weight)
        self.distribution = _Distribution()

    def get_latent(self, observation):
        return observation["actor"]

    def get_output_log_prob(self, actions):
        return -.5 * (actions - self.distribution.mean).square().sum(-1)


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def get_latent(self, observation):
        return observation["critic"] + self.anchor * 0


class _Control(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = _Actor()
        self.critic = _Critic()


def _trajectory(count=3, steps=5, terminal_penalty=-10.):
    recorder = WindowTrajectoryRecorder(count, steps, terminal_penalty=terminal_penalty)
    observation = {"actor": torch.zeros(count, 2), "critic": torch.zeros(count, 2)}
    for t in range(steps):
        recorder.before_step(observation, torch.zeros(count, 1), torch.zeros(count))
        rewards = torch.full((count,), float(t + 1))
        done = torch.tensor([t == 1, t == 2, False])[:count]
        recorder.after_step(rewards, done, {"time_outs": torch.tensor([t == 1, False, False])[:count]})
        observation["actor"].fill_(t + 1)
        observation["critic"].fill_(t + 1)
        rewards.fill_(999.)
    return recorder.finalize()


def test_recorder_copies_borrowed_tensors_and_spans_resets():
    trajectory = _trajectory()
    torch.testing.assert_close(trajectory.rewards[:, 0], torch.tensor([1., 2., 3., 4., 5.]))
    torch.testing.assert_close(trajectory.rewards[:, 1], torch.tensor([1., 2., -7., 4., 5.]))
    torch.testing.assert_close(trajectory.observations["actor"][:, 0, 0], torch.arange(5.))
    assert trajectory.storage_bytes == sum(t.numel() * t.element_size() for t in
        (*trajectory.observations.values(), trajectory.actions, trajectory.behavior_logprobs, trajectory.rewards))
    with pytest.raises(RuntimeError, match="complete"):
        trajectory.before_step({"actor": torch.zeros(3, 2)}, torch.zeros(3, 1), torch.zeros(3))


def test_on_policy_identity_for_any_value_and_discount_across_resets():
    trajectory = _trajectory()
    torch.manual_seed(9)
    values = torch.randn_like(trajectory.rewards)
    for gamma in (1., .7):
        corrected = finite_window_vtrace(trajectory.rewards, values,
                                          torch.zeros_like(values), gamma)
        expected = torch.zeros(trajectory.count)
        for t in range(trajectory.steps - 1, -1, -1):
            expected = trajectory.rewards[t] + gamma * expected
            torch.testing.assert_close(corrected[t], expected, atol=1e-5, rtol=1e-5)


def test_two_step_clipped_ratios_follow_vtrace_recursion():
    rewards = torch.tensor([[2.], [3.]])
    values = torch.tensor([[4.], [5.]])
    ratios = torch.tensor([[2.], [.5]]).log()
    result = finite_window_vtrace(rewards, values, ratios)
    # Last: 5 + .5 * (3 - 5) = 4. First: 4 + (2 + 5 - 4) + (4 - 5) = 6.
    torch.testing.assert_close(result[:, 0], torch.tensor([6., 4.]))


def test_scorer_on_policy_matches_window_and_does_not_change_native_or_rng():
    control = _Control()
    control.train()
    native = copy.deepcopy(control.state_dict())
    trajectory = _trajectory()
    rng = torch.get_rng_state().clone()
    scorer = HardwareVTraceScorer(control, dict(value_epochs=1, batch_size=4), seed=23)
    scores, metrics = scorer.score(trajectory)
    torch.testing.assert_close(scores, trajectory.rewards.sum(0), atol=1e-4, rtol=1e-4)
    assert metrics["vtrace_action_ess_fraction"] == pytest.approx(1.)
    assert metrics["vtrace_rho_clipped_fraction"] == 0.
    assert torch.equal(torch.get_rng_state(), rng)
    assert control.actor.training and control.critic.training
    for key, value in native.items():
        torch.testing.assert_close(control.state_dict()[key], value, rtol=0, atol=0)


def test_checkpoint_restores_critic_optimizer_and_sampling_rng():
    cfg = dict(value_epochs=1, batch_size=4)
    trajectory = _trajectory()
    first = HardwareVTraceScorer(_Control(), cfg, seed=7)
    first.score(trajectory)
    state = copy.deepcopy(first.state_dict())
    second = HardwareVTraceScorer(_Control(), cfg, seed=7)
    second.load_state_dict(state)
    left, left_metrics = first.score(trajectory)
    right, right_metrics = second.score(trajectory)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert left_metrics == right_metrics
    assert first.trained_windows == second.trained_windows == 2


def test_pending_checkpoint_survives_resave_before_first_score():
    cfg = dict(value_epochs=1, batch_size=4)
    trajectory = _trajectory()
    original = HardwareVTraceScorer(_Control(), cfg, seed=7)
    original.score(trajectory)
    pending = HardwareVTraceScorer(_Control(), cfg, seed=7)
    pending.load_state_dict(copy.deepcopy(original.state_dict()))
    saved_again = copy.deepcopy(pending.state_dict())
    assert saved_again["input_dimension"] == original.input_dimension
    assert saved_again["value_model"] is not None and saved_again["optimizer"] is not None
    restored = HardwareVTraceScorer(_Control(), cfg, seed=7)
    restored.load_state_dict(saved_again)
    expected, _ = original.score(trajectory)
    actual, _ = restored.score(trajectory)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [dict(extra=1), dict(rho_clip=True), dict(c_clip=float("nan")),
                                  dict(learning_rate=float("inf")), dict(value_epochs=False),
                                  dict(batch_size=0), dict(c_clip=2.)])
def test_rejects_invalid_vtrace_config(bad):
    with pytest.raises(ValueError):
        HardwareVTraceScorer.validate_config(bad)


def test_rejects_nonfinite_vtrace_inputs():
    with pytest.raises(ValueError, match="finite"):
        finite_window_vtrace(torch.tensor([[float("nan")]]), torch.zeros(1, 1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="caps"):
        finite_window_vtrace(torch.ones(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), gamma=float("nan"))


def test_four_by_64_window_is_bounded_and_scores_complete_return():
    count, steps = 4, 4 * 64
    recorder = WindowTrajectoryRecorder(count, steps)
    obs = {"actor": torch.zeros(count, 2), "critic": torch.zeros(count, 2)}
    action = torch.zeros(count, 1)
    likelihood = torch.zeros(count)
    rewards = torch.ones(count)
    done = torch.ones(count, dtype=torch.bool)
    for _ in range(steps):
        recorder.before_step(obs, action, likelihood)
        recorder.after_step(rewards, done, {"time_outs": done})
    scorer = HardwareVTraceScorer(_Control(), dict(value_epochs=1, batch_size=256))
    scores, metrics = scorer.score(recorder)
    torch.testing.assert_close(scores, torch.full((count,), float(steps)), atol=1e-3, rtol=1e-5)
    assert metrics["vtrace_storage_bytes"] == steps * count * (2 * 2 + 1 + 1 + 1) * 4


def test_rejects_partial_window():
    recorder = WindowTrajectoryRecorder(2, 2)
    recorder.before_step({"actor": torch.zeros(2, 1)}, torch.zeros(2, 1), torch.zeros(2))
    with pytest.raises(ValueError, match="complete"):
        recorder.finalize()
