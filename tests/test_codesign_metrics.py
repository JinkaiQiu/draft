from collections import Counter
import copy
import json
from types import SimpleNamespace

import pytest

from draft.codesign.metrics import tensorboard_scalars


def alternating_metrics():
    keys = """
        hardware_updates control_updates hardware_records design_score_mean invalid_designs
        hardware_policy_loss hardware_value_loss hardware_kl
        compile_seconds build_seconds rollout_seconds collect_seconds hardware_collect_seconds
        update_seconds iteration_seconds simulated_transitions control_transitions sps
        controller_updates controller_iterations controller_transitions total_controller_transitions
        elapsed_env_steps controller_storage_bytes_peak controller_collect_seconds controller_update_seconds
        controller_learning_rate controller_mean_transition_reward controller_loss_value
        controller_loss_surrogate controller_loss_entropy controller_training_draws controller_training_rejections
        controller_compile_seconds controller_build_seconds controller_stage_seconds controller_cycles
        hardware_evaluation_transitions hardware_active_transitions total_hardware_evaluation_transitions
        hardware_native_horizon_steps hardware_survivors hardware_evaluation_capped
        hardware_observed_return_mean hardware_bootstrap_mean hardware_bootstrap_replicas
        hardware_evaluation_steps hardware_evaluation_cap_steps hardware_bootstrap_enabled hardware_survival_fraction
        controller_completed_episode_count controller_mean_episode_reward controller_mean_episode_length
        controller_partial_episode_mean_return controller_partial_episode_mean_length
        controller_partial_episode_count controller_partial_episode_world_fraction
    """.split()
    metrics = dict.fromkeys(keys, 1.)
    metrics.update(hardware_records=16, invalid_designs=2, controller_storage_bytes_peak=206 * 2**20)
    for name in ("upper_leg", "lower_leg", "torso_x", "torso_y", "torso_z", "shoulder", "hip"):
        metrics["hardware_mean_" + name] = .25
        metrics["hardware_latent_std_" + name] = .5
    return metrics


def test_alternating_cards_are_compact_without_mutating_complete_raw_metrics():
    metrics = alternating_metrics()
    before = copy.deepcopy(metrics)
    scalars = tensorboard_scalars(metrics)
    assert metrics == before and len(metrics) == 71
    assert len(scalars) == 43
    groups = Counter(tag.rsplit("/", 1)[0] for tag in scalars)
    assert groups["codesign"] == 2
    assert groups["HardwareMean"] == groups["HardwareStd"] == 7
    assert groups["HardwareLoss"] == 4 and groups["HardwareEvaluation"] == 3
    assert groups["Perf/CoDesign"] == 6 and groups["Perf/Controller"] == 5
    assert max(groups.values()) == 7
    assert scalars["codesign/valid_fraction"] == .875
    assert scalars["HardwareStd/latent_upper_leg"] == .5
    assert scalars["Memory/controller_rollout_mib"] == 206
    assert scalars["Perf/CoDesign/hardware_collect_seconds"] == metrics["hardware_collect_seconds"]
    assert not any(tag.startswith("Diagnostics/") for tag in scalars)
    assert not any("controller_loss" in tag or "partial_episode" in tag for tag in scalars)


def test_explicit_namespaces_win_collisions_and_unknown_metrics_remain_visible():
    metrics = dict(hardware_policy_loss=1., novel_diagnostic=2., hardware_records=4, invalid_designs=1,
                   **{"HardwareLoss/policy": 9., "codesign/valid_fraction": .1,
                      "Train/mean_reward": 5., "Episode_Reward/velocity": 3.})
    scalars = tensorboard_scalars(metrics)
    assert scalars == {"HardwareLoss/policy": 9., "codesign/valid_fraction": .1,
                       "Train/mean_reward": 5., "Episode_Reward/velocity": 3.,
                       "Diagnostics/novel_diagnostic": 2.}
    # Input order cannot cause a second write or change the winning value.
    assert tensorboard_scalars(dict(reversed(list(metrics.items())))) == scalars
    assert "codesign/valid_fraction" not in tensorboard_scalars(dict(hardware_records=0, invalid_designs=0))


def test_legacy_full_window_control_losses_and_transition_counts_remain_visible():
    scalars = tensorboard_scalars(dict(control_policy_loss=.2, control_value_loss=.3, control_kl=.01,
                                      control_updates=12, control_transitions=30, simulated_transitions=40,
                                      update_seconds=.5, hardware_policy_loss=.1))
    assert scalars == {"ControlLoss/policy": .2, "ControlLoss/value": .3, "ControlLoss/kl": .01,
                       "ControlLoss/updates": 12, "Progress/control_transitions": 30,
                       "Progress/simulated_transitions": 40, "Perf/CoDesign/joint_update_seconds": .5,
                       "HardwareLoss/policy": .1}


def test_runner_routes_writer_only_and_preserves_raw_jsonl(tmp_path):
    pytest.importorskip("torch")
    from test_codesign_runner import make_runner
    runner = make_runner(final_evaluation=False, final_export=False, eval_interval=0)
    runner.log_dir = tmp_path
    writes = []
    runner.writer = SimpleNamespace(add_scalar=lambda *args: writes.append(args), flush=lambda: None)
    metrics = runner.learn(1)[0]
    stored = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert all(stored[key] == value for key, value in metrics.items())
    assert "hardware_mean_a" in stored and "HardwareMean/a" not in stored
    assert "control_policy_loss" in stored and "ControlLoss/policy" not in stored
    assert {tag: value for tag, value, step in writes} == tensorboard_scalars(metrics)
    assert len(writes) == len({tag for tag, _, _ in writes})
    assert all(step == 1 for _, _, step in writes)
