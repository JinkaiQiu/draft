"""Native rollout length controls the complete co-adaptation scoring window."""

import copy

import pytest
import torch

pytest.importorskip("rsl_rl")

from draft.codesign.alternating import AlternatingCodesignRunner
from draft.codesign.coadaptation import CoadaptationCodesignRunner
from draft.codesign.controller import NativeControllerTrainer
from draft.codesign.policy import HardwarePolicy, JointPolicy
from test_codesign_alternating import RecordingCompiler
from test_codesign_controller import Env, config, policy


class WindowEnv(Env):
    horizon_steps = 4
    step_dt = 1.

    def __init__(self, designs, replicas, **kwargs):
        super().__init__(len(designs) * replicas)
        self.closed = False

    def step(self, actions):
        observation, _, _, _ = super().step(actions)
        rewards = torch.tensor([1., 1., 2., 2.])
        dones = torch.tensor([self.steps % 8 == 0, False, False, self.steps % 16 == 0])
        timeouts = torch.tensor([False, False, False, self.steps % 16 == 0])
        return observation, rewards, dones, {"time_outs": timeouts}

    def close(self):
        self.closed = True


def make_runner(steps=64, schedule="coadaptation"):
    native = config()
    native["num_steps_per_env"] = steps
    native["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    cfg = dict(device="cpu", training_schedule=schedule, num_designs=2, replicas=2,
               coadaptation_cohort_lifetime="window", staleness_diagnostics=False,
               horizon_steps=4, episode_length_s=4., gamma=1., terminal_penalty=-10.,
               logger="none", controller_updates_per_cycle=4, controller_train_cfg=native,
               num_learning_epochs=1, hardware_target_kl=100.)
    runner_class = CoadaptationCodesignRunner if schedule == "coadaptation" else AlternatingCodesignRunner
    return runner_class(JointPolicy(policy(), HardwarePolicy(2)), RecordingCompiler(), WindowEnv, cfg)


def test_native_64_by_4_scores_all_256_training_steps_and_preserves_resume(tmp_path):
    runner = make_runner()
    before = copy.deepcopy(runner.policy.control.state_dict())
    batch = runner.collect()
    assert isinstance(runner.controller_trainer, NativeControllerTrainer)
    assert runner.controller_trainer.steps == 64
    assert batch["control"]["scoring_steps"] == 256
    # 32 true terminations cost ten each in world zero; timeout rewards stay raw.
    torch.testing.assert_close(batch["control"]["replica_returns"],
                               torch.tensor([[-64., 256.], [512., 512.]]))
    assert batch["scores"].tolist() == [96., 512.]
    assert batch["control"]["true_terminations"] == 32
    assert batch["control"]["timeouts"] == 16
    assert batch["control"]["score_accumulator_bytes"] == 4 * 4 + 2 * 8
    assert runner.controller_trainer.alg.storage is None
    assert any(not torch.equal(value, runner.policy.control.state_dict()[key])
               for key, value in before.items() if key.startswith("actor.mlp"))
    metrics = runner.update(batch)
    assert metrics["controller_updates"] == metrics["controller_iterations"] == 4
    assert metrics["hardware_scoring_steps"] == runner.elapsed_env_steps == 256
    assert metrics["simulated_transitions"] == metrics["reused_training_transitions"] == runner.global_step == 1024
    assert metrics["hardware_evaluation_transitions"] == 0
    path = tmp_path / "64_step.pt"
    runner.save(path)
    saved = torch.load(path, weights_only=False)
    assert saved["config"]["controller_train_cfg"]["num_steps_per_env"] == 64
    restored = make_runner()
    restored.load(path)
    assert restored.controller_trainer.steps == 64
    assert restored.controller_trainer.iteration == 4
    assert restored.elapsed_env_steps == 256 and restored.global_step == 1024
    with pytest.raises(ValueError, match="protected.*controller_train_cfg"):
        make_runner(24).load(path)
    incompatible_cfg = copy.deepcopy(runner.config["controller_train_cfg"])
    incompatible_cfg["num_steps_per_env"] = 24
    incompatible = NativeControllerTrainer(policy(), incompatible_cfg, "cpu")
    with pytest.raises(ValueError, match="native PPO configuration"):
        incompatible.load_state_dict(runner.controller_trainer.state_dict())


@pytest.mark.parametrize("steps", [True, False, 0, -1, 64., None])
@pytest.mark.parametrize("schedule", ["alternating", "coadaptation"])
def test_native_schedule_rejects_invalid_rollout_lengths(steps, schedule):
    with pytest.raises(ValueError, match="num_steps_per_env.*positive integer"):
        make_runner(steps, schedule)


def test_alternating_runner_accepts_configurable_native_rollouts():
    runner = make_runner(64, "alternating")
    assert runner._ensure_controller_trainer().steps == 64


@pytest.mark.parametrize("schedule", ["alternating", "coadaptation"])
def test_typed_native_configuration_accepts_64_step_rollouts(schedule):
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg

    cfg = CodesignRunnerCfg(training_schedule=schedule, num_steps_per_env=64,
                            controller_updates_per_cycle=4)
    cfg.validate()
    assert cfg.num_steps_per_env * cfg.controller_updates_per_cycle == 256
    assert CodesignRunnerCfg().num_steps_per_env == 24


@pytest.mark.parametrize("steps", [True, False, 0, -1, 64., None])
def test_typed_native_configuration_rejects_invalid_rollout_lengths(steps):
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg

    with pytest.raises(ValueError, match="num_steps_per_env.*positive integer"):
        CodesignRunnerCfg(training_schedule="coadaptation", num_steps_per_env=steps).validate()


def test_legacy_full_window_rollout_constraint_is_unchanged():
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg

    with pytest.raises(ValueError, match="full_window requires num_steps_per_env=24"):
        CodesignRunnerCfg(num_steps_per_env=64).validate()
