"""First-trial masking and paired seed-cluster inference, without a GPU."""
from types import SimpleNamespace
import copy
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from draft.codesign.comparison import paired_statistics, summarize, task_contract, compare_protocol
from draft.codesign.evaluation import PairingAudit, evaluate, _tensor_digest
from test_codesign_runner import make_runner, Env


def test_seed_clusters_are_bootstrap_units():
    left = {2: [2., 2.], 3: [6., 6.]}
    right = {2: [1., 1.], 3: [1., 1.]}
    result = paired_statistics(left, right)
    assert result["mean_difference"] == 3.
    assert result["seed_mean_differences"] == [1., 5.]
    assert result["ci95_seed_cluster_bootstrap"] == [1., 5.]
    repeated = paired_statistics({key: value * 100 for key, value in left.items()},
                                 {key: value * 100 for key, value in right.items()})
    assert repeated == result  # Replicating worlds cannot create precision.
    with pytest.raises(ValueError, match="same two"):
        paired_statistics({2: [1.]}, {3: [1.]})
    with pytest.raises(ValueError, match="equal finite"):
        paired_statistics(left, {2: [1.], 3: [1.]})


def test_task_contract_checks_physical_task_and_ignores_sampling():
    config = dict(task="quadruped", max_episode_length=20., env_kwargs=dict(
        robot_xml="stock.xml", hardware_aware=True, morphology_conditioning=dict(mode="fixed")))
    joint = dict(config, episode_length_s=20., env_kwargs=dict(robot_xml="stock.xml", hardware_aware=True))
    assert task_contract(config) == task_contract(joint)
    joint["env_kwargs"]["hardware_aware"] = False
    assert task_contract(config) != task_contract(joint)


def test_per_trial_records_mask_at_first_fall_and_keep_full_horizon():
    runner = make_runner()
    report = evaluate(runner, replicas=2, record_video=False, include_trials=True,
                      physical=runner.compiler.space.center)
    assert report["selection"] == "explicit physical geometry"
    assert report["horizon_steps"] == 4
    assert [trial["return"] for trial in report["trials"]] == [-8., 4.]
    assert [trial["duration_steps"] for trial in report["trials"]] == [2, 4]
    assert [trial["survived"] for trial in report["trials"]] == [False, True]


class PairedEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = SimpleNamespace(auto_reset=True)
        self._manual_reset_pending = torch.zeros(self.count, dtype=torch.bool)
        self.sim = SimpleNamespace(model=SimpleNamespace(hfield_data=torch.arange(3)))
        self.scene = SimpleNamespace(terrain=SimpleNamespace(env_origins=torch.zeros(self.count, 3)))
        term = SimpleNamespace(**{name: torch.zeros(self.count) for name in (
            "time_left", "command_counter", "heading_target", "vel_command_w",
            "is_heading_env", "is_standing_env", "is_world_env", "is_forward_env")})
        self.command_manager = SimpleNamespace(get_term=lambda name: term)

    def step(self, actions):
        assert not self._manual_reset_pending.any()
        result = super().step(actions)
        self._manual_reset_pending |= result[2]
        if self.steps > 2:
            assert actions[0, 0] == 0.
            # Dead-world nonfinites never contaminate its retained first trial.
            result[1][0] = torch.nan
        return result


def test_paired_adapter_continues_dead_worlds_without_resets_or_scoring(monkeypatch):
    runner = make_runner()
    runner.env_factory = PairedEnv
    monkeypatch.setattr("draft.codesign.evaluation._tracking_errors", lambda env: None)
    reports = [evaluate(runner, replicas=2, seed=19, record_video=False, paired=True,
                        include_trials=True) for _ in range(2)]
    assert reports[0]["pairing_audit"] == reports[1]["pairing_audit"]
    assert reports[0]["pairing_audit"]["auto_reset"] is False
    assert reports[0]["trials"][0]["return"] == -8.


def test_paired_adapter_requires_supported_native_environment():
    with pytest.raises(ValueError, match="manual-reset"):
        PairingAudit(SimpleNamespace(unwrapped=SimpleNamespace()))


def test_tensor_digest_accepts_mjlab_torch_proxies():
    class Proxy:
        def detach(self):
            return torch.arange(4)
    assert _tensor_digest([("field", Proxy())]) == _tensor_digest([("field", torch.arange(4))])


def test_summary_consumes_real_trial_records_and_paired_return_keys():
    runner = make_runner()
    records = {}
    for name in ("baseline_stock", "codesign_learned", "codesign_stock", "source_stock"):
        records[name] = {seed: evaluate(runner, replicas=2, seed=seed,
                        record_video=False, include_trials=True) for seed in (31, 32)}
    result = summarize(records)
    assert result["conditions"]["baseline_stock"]["return_mean"] == -2.
    assert result["conditions"]["baseline_stock"]["survival_fraction"] == .5
    assert result["conditions"]["baseline_stock"]["active_duration_fraction"] == .75
    assert result["paired_comparisons"]["primary"]["metrics"]["return"]["mean_difference"] == 0.


@pytest.mark.parametrize("drift", [None, "replicas", "algorithm", "offset", "task"])
def test_protocol_rejects_controller_budget_and_task_drift(tmp_path, monkeypatch, drift):
    source = tmp_path / "source.pt"
    source.write_bytes(b"frozen original controller")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = dict(seed=0, num_designs=16, replicas=256, training_schedule="alternating",
        hardware_sampling_mode="stochastic", optimize_hardware=True,
        controller_updates_per_cycle=20, env_kwargs=dict(hardware_aware=True),
        controller_train_cfg=dict(num_steps_per_env=24, actor={}, critic={}, obs_groups={},
                                  algorithm=dict(schedule="adaptive")))
    data = dict(iteration=226, global_step=681312256, config=config,
        metadata=dict(warmstart=dict(sha256=source_hash)),
        alternating_state=dict(controller_cycles=226, hardware_evaluation_transitions=236978176,
            controller_trainer=dict(iteration=4520, total_transitions=444334080)))
    joint = tmp_path / "joint.pt"
    torch.save(data, joint)
    baseline = copy.deepcopy(data)
    baseline["config"].update(training_schedule="coadaptation", hardware_sampling_mode="fixed",
                                optimize_hardware=False, training_seed_offset=1000003)
    if drift == "replicas":
        baseline["config"].update(num_designs=8, replicas=512)  # Same worlds, different allocation.
    if drift == "algorithm":
        baseline["config"]["controller_train_cfg"]["algorithm"]["schedule"] = "fixed"
    if drift == "offset":
        baseline["config"]["training_seed_offset"] = 0
    if drift == "task":
        baseline["config"]["env_kwargs"]["hardware_aware"] = False
    baseline_path = tmp_path / "baseline.pt"
    torch.save(baseline, baseline_path)
    baseline_config = tmp_path / "config.yaml"
    baseline_config.write_text("task: quadruped\n")
    plan = dict(codesign_checkpoint=str(joint), codesign_sha256=hashlib.sha256(joint.read_bytes()).hexdigest(),
        source_checkpoint=str(source), source_sha256=source_hash,
        baseline_config=str(baseline_config), baseline_config_sha256=hashlib.sha256(baseline_config.read_bytes()).hexdigest(),
        training_seed=0, matched_budget=dict(cycles=226, controller_updates_per_cycle=20,
            controller_updates=4520, controller_rollout_steps=24, worlds=4096, design_rows=16,
            replicas=256, controller_transitions=444334080, hardware_aware=True, training_seed_offset=1000003),
        evaluation=dict(full_trial=True, bootstrap_at_cutoff=False, deterministic_controller=True,
                        seeds=[20, 21], trials_per_condition=4, replicas_per_seed=2))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(plan))
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr("draft.codesign.comparison.compare_checkpoints", lambda **kwargs: dict(ok=True))
    if drift:
        with pytest.raises(ValueError, match="budget|settings|offset"):
            compare_protocol(protocol, baseline_path, output)
    else:
        assert compare_protocol(protocol, baseline_path, output)["ok"]
