"""The topology charge affects design scores after controller task scoring."""

import copy
import json

import pytest
import torch

from draft.codesign.joint_cost import active_joint_counts, apply_joint_cost, validate_joint_cost
from draft.codesign.metrics import tensorboard_scalars


FULL = [1] * 12
ONE_PAIR_OFF = [0, 1, 1, 0, 1, 1, *([1] * 6)]


def test_full_and_frozen_pair_charge_has_small_fixed_scale():
    raw = torch.tensor([[17., 18.], [-3., -4.], [0., 0.]], dtype=torch.float64)
    original = raw.clone()
    net, metrics, records = apply_joint_cost(raw, [12, 10, 0], weight=.025,
                                              discounted_seconds=5.12)
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    torch.testing.assert_close(net, raw - torch.tensor([.128, .128 * 10 / 12, 0.])[:, None])
    assert records[0]["joint_penalty"] == pytest.approx(.128)
    assert records[0]["joint_penalty"] - records[1]["joint_penalty"] == pytest.approx(.0213333333333)
    assert records[2]["joint_penalty"] == 0
    assert metrics["hardware_active_joint_count_mean"] == pytest.approx(22 / 3)
    assert records[1]["task_score"] == -3.5
    json.dumps(records, allow_nan=False)


@pytest.mark.parametrize("weight", [True, -1, float("nan"), float("inf"), "0.025"])
def test_invalid_weight(weight):
    with pytest.raises(ValueError, match="hardware_joint_cost_weight"):
        validate_joint_cost(weight)


@pytest.mark.parametrize("counts", [None, [12], [13, 0], [-1, 0], [1.5, 0], [1, 0],
                                         [float("nan"), 0], torch.tensor([True, False])])
def test_invalid_physical_counts(counts):
    with pytest.raises(ValueError, match="active_counts"):
        apply_joint_cost(torch.ones(2, 2), counts, weight=.025, discounted_seconds=5.12)


def test_action_mask_reduces_identical_replicas_and_rejects_malformed_rows():
    rows = torch.tensor([FULL, FULL, ONE_PAIR_OFF, ONE_PAIR_OFF], dtype=torch.bool)
    torch.testing.assert_close(active_joint_counts(rows, num_designs=2, replicas=2),
                               torch.tensor([12, 10]))
    for invalid in (None, rows[:3], rows[:, :11],
                    torch.tensor([FULL, ONE_PAIR_OFF, ONE_PAIR_OFF, ONE_PAIR_OFF]),
                    torch.tensor([FULL, FULL, [0, *([1] * 11)], [0, *([1] * 11)]]),
                    torch.tensor([FULL, FULL, [2, *([1] * 11)], [2, *([1] * 11)]])):
        with pytest.raises(ValueError, match="hardware_action_mask"):
            active_joint_counts(invalid, num_designs=2, replicas=2)


def _scored_runner(*, mass, vtrace, mask=True, log_dir=None):
    pytest.importorskip("rsl_rl")
    from test_codesign_mass_cost import runner_with_mass

    runner = runner_with_mass(weight=.05 if mass else 0., enabled=vtrace)
    runner.config["hardware_joint_cost_weight"] = .025
    runner.config["joint_topology"] = True
    runner.log_dir = log_dir
    factory = runner.env_factory

    def with_mask(*args, **kwargs):
        env = factory(*args, **kwargs)
        if mask:
            env.hardware_action_mask = torch.tensor([FULL, FULL, ONE_PAIR_OFF, ONE_PAIR_OFF], dtype=torch.bool)
        return env

    runner.env_factory = with_mask
    return runner


@pytest.mark.parametrize("mass", [False, True])
@pytest.mark.parametrize("vtrace", [False, True])
def test_joint_cost_follows_raw_or_vtrace_task_and_mass_without_changing_learner(
        mass, vtrace, tmp_path):
    from test_codesign_mass_cost import runner_with_mass, STOCK_MASS, LEARNED_MASS

    plain = runner_with_mass(weight=0., enabled=vtrace)
    scored = _scored_runner(mass=mass, vtrace=vtrace, log_dir=tmp_path)
    scored.policy.control.load_state_dict(copy.deepcopy(plain.policy.control.state_dict()))
    try:
        for window in range(2):
            torch.manual_seed(512 + window)
            baseline = plain.collect()
            rng = torch.get_rng_state().clone()
            control = copy.deepcopy(plain.policy.control.state_dict())
            torch.manual_seed(512 + window)
            batch = scored.collect()
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, value in scored.policy.control.state_dict().items():
                torch.testing.assert_close(value, control[key], rtol=0, atol=0)
            if vtrace:
                for key, value in scored._vtrace_scorer.value_model.state_dict().items():
                    torch.testing.assert_close(value, plain._vtrace_scorer.value_model.state_dict()[key],
                                               rtol=0, atol=0)
            records = batch["control"]["hardware_objective_records"]
            assert batch["controller_training"]["hardware_task_score_mean"] == pytest.approx(
                baseline["scores"].mean().item())
            for index, record in enumerate(records):
                assert record["task_score"] == pytest.approx(baseline["scores"][index].item())
                assert record["active_joint_count"] == (12, 10)[index]
                assert record["joint_penalty"] == pytest.approx(.025 * .64 * (12, 10)[index] / 12)
                if mass:
                    assert record["mass_penalty"] == pytest.approx(
                        .05 * .64 * (STOCK_MASS, LEARNED_MASS)[index] / STOCK_MASS)
                else:
                    assert "mass_penalty" not in record
                assert batch["scores"][index].item() == pytest.approx(
                    record["task_score"] - record.get("mass_penalty", 0) - record["joint_penalty"])
            metrics = scored.update(batch)
            plain.update(baseline)
            tags = tensorboard_scalars(metrics)
            assert tags["HardwareReward/task"] == pytest.approx(baseline["scores"].mean().item())
            assert tags["HardwareReward/joint_cost"] == pytest.approx(.025 * .64 * 11 / 12)
            assert tags["HardwareTopology/sampled_active_joints"] == 11
            assert {tag for tag in tags if tag.startswith("HardwareReward/")} == (
                {"HardwareReward/task", "HardwareReward/joint_cost", "HardwareReward/mass_cost"}
                if mass else {"HardwareReward/task", "HardwareReward/joint_cost"})
        rows = [json.loads(line) for line in (tmp_path / "design_records.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        for row in rows:
            assert row["score"] == pytest.approx(
                row["task_score"] - row.get("mass_penalty", 0) - row["joint_penalty"])
    finally:
        plain.close()
        scored.close()


def test_enabled_cost_requires_native_mask():
    runner = _scored_runner(mass=False, vtrace=False, mask=False)
    try:
        with pytest.raises(ValueError, match="hardware_action_mask"):
            runner.collect()
    finally:
        runner.close()


def test_rejected_design_keeps_its_original_invalid_return():
    from test_codesign_alternating import RecordingCompiler
    from test_codesign_coadaptation import TrainingEnv, make_runner

    class MaskedTrainingEnv(TrainingEnv):
        def __init__(self, designs, replicas, **kwargs):
            super().__init__(designs, replicas, **kwargs)
            self.hardware_action_mask = torch.tensor([FULL] * self.num_envs, dtype=torch.bool)

    compiler = RecordingCompiler(lambda call, count: [True, False, True])
    runner = make_runner(compiler, env_factory=MaskedTrainingEnv, num_designs=3)
    runner.config["hardware_joint_cost_weight"] = .025
    try:
        batch = runner.collect()
        assert batch["valid_indices"] == [0, 2]
        assert batch["scores"][1].item() == runner.config["invalid_design_return"]
        records = batch["control"]["hardware_objective_records"]
        assert [record["task_score"] for record in records] == [-187., 48.]
        assert [record["active_joint_count"] for record in records] == [12, 12]
        assert [record["joint_penalty"] for record in records] == pytest.approx([1.2, 1.2])
        runner.update(batch)
    finally:
        runner.close()


def test_config_and_resume_protect_joint_weight(tmp_path):
    from draft.tasks.quadruped.codesign.runner_cfg import CodesignRunnerCfg
    from test_codesign_mass_cost import runner_with_mass

    with pytest.raises(ValueError, match="hardware_joint_cost_weight requires"):
        CodesignRunnerCfg(training_schedule="coadaptation", hardware_joint_cost_weight=.025).validate()
    with pytest.raises(ValueError, match="joint_topology currently requires"):
        CodesignRunnerCfg(joint_topology=True, hardware_joint_cost_weight=.025).validate()
    with pytest.raises(ValueError, match="hardware_joint_cost_weight must"):
        CodesignRunnerCfg(hardware_joint_cost_weight=True).validate()
    CodesignRunnerCfg(training_schedule="coadaptation", joint_topology=True,
                     hardware_joint_cost_weight=.025).validate()
    with pytest.raises(ValueError, match="hardware_joint_cost_weight requires"):
        runner_with_mass(weight=0., hardware_joint_cost_weight=.025)

    source, target, changed = (runner_with_mass(weight=0.) for _ in range(3))
    try:
        path = tmp_path / "old.pt"
        source.config["hardware_mass_reference_kg"] = None
        target.config["hardware_mass_reference_kg"] = None
        changed.config["hardware_mass_reference_kg"] = None
        source.save(path)
        payload = torch.load(path, weights_only=False)
        del payload["config"]["hardware_joint_cost_weight"]
        torch.save(payload, path)
        target.load(path)
        changed.config["hardware_joint_cost_weight"] = .025
        with pytest.raises(ValueError, match="protected.*hardware_joint_cost_weight"):
            changed.load(path)
    finally:
        source.close()
        target.close()
        changed.close()
