from types import SimpleNamespace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.distributions import Normal

from draft.codesign.policy import HardwarePolicy, JointPolicy
from draft.codesign.runner import CodesignRunner, finite_horizon_advantages


class Control(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(2, 1)
        self.critic = nn.Linear(2, 1)

    def evaluate(self, observation, actions=None):
        dist = Normal(self.actor(observation["actor"]), .3)
        if actions is None:
            actions = dist.sample()
        return actions, dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), self.critic(observation["critic"]).squeeze(-1)

    def deterministic(self, observation):
        return self.actor(observation["actor"])


class Compiler:
    def __init__(self, valid=(True, False)):
        self.valid = valid
        self.space = SimpleNamespace(dim=2, names=("a", "b"), lower=(-1., -1.), upper=(1., 1.),
                                     center=(0., 0.), denormalize=lambda x: np.asarray(x), normalize=lambda x: np.asarray(x))

    def compile_many(self, physical):
        return [SimpleNamespace(design=SimpleNamespace(design_id=str(i)) if valid else None,
                                design_id=str(i), error=None if valid else "infeasible")
                for i, valid in enumerate(self.valid[:len(physical)])]


class Env:
    num_actions = 1
    horizon_steps = 4
    step_dt = 1.

    def __init__(self, designs, replicas, **kwargs):
        self.unwrapped = self
        self.count = len(designs) * replicas
        self.steps = 0
        self.closed = False

    def obs(self):
        return {key: torch.full((self.count, 2), float(self.steps)) for key in ("actor", "critic")}

    def reset(self):
        return self.obs(), {}

    def step(self, actions):
        self.steps += 1
        done = torch.zeros(self.count, dtype=torch.bool)
        done[0] = self.steps >= 2  # Simulates repeated auto-reset terminal signals.
        return self.obs(), torch.ones(self.count), done, {"time_outs": torch.zeros_like(done)}

    def close(self):
        self.closed = True


def make_runner(compiler=None, **overrides):
    config = dict(device="cpu", num_designs=2, replicas=2, horizon_steps=4, episode_length_s=4.,
                  gamma=1., lam=1., num_learning_epochs=3, num_mini_batches=4, desired_kl=100.,
                  hardware_target_kl=100., terminal_penalty=-10., invalid_design_return=-50., logger="none")
    config.update(overrides)
    return CodesignRunner(JointPolicy(Control(), HardwarePolicy(2)), compiler or Compiler(), Env, config)


def test_fixed_window_first_fall_masks_rewards_transitions_and_invalid_rows():
    runner = make_runner()
    before = {key: value.clone() for key, value in runner.policy.state_dict().items()}
    batch = runner.collect()
    for key, value in runner.policy.state_dict().items():
        torch.testing.assert_close(value, before[key])  # Frozen through all four steps.
    assert batch["raw"].shape == (2, 2)
    assert batch["scores"].tolist() == [-2., -50.]
    assert batch["control"]["rewards"][:, 0].tolist() == [1., -9., 0., 0.]
    assert batch["control"]["active"].sum().item() == 6
    metrics = runner.update(batch)
    assert metrics["hardware_records"] == 2
    assert metrics["hardware_updates"] == 3  # Once per epoch, independent of 4 control minibatches.
    assert metrics["control_updates"] == 12
    assert metrics["invalid_designs"] == 1
    assert runner.global_step == 8


def test_zero_bootstrap_at_window_and_fall():
    rewards = torch.tensor([[1., 2.], [3., 4.], [0., 5.]])
    active = torch.tensor([[True, True], [True, True], [False, True]])
    values = torch.full_like(rewards, 999.)
    _, returns = finite_horizon_advantages(rewards, values, active, gamma=1., lam=1.)
    torch.testing.assert_close(returns, torch.tensor([[4., 11.], [3., 9.], [0., 5.]]))


def test_invalid_only_cohort_trains_hardware_and_preserves_draws():
    runner = make_runner(Compiler((False, False)))
    batch = runner.collect()
    assert batch["control"] is None
    assert batch["scores"].tolist() == [-50., -50.]
    metrics = runner.update(batch)
    assert metrics["hardware_updates"] == 3
    assert metrics["control_updates"] == 0
    assert runner.global_step == 0


def test_controller_baseline_has_no_hardware_drift():
    runner = make_runner(optimize_hardware=False)
    before = {key: value.clone() for key, value in runner.policy.hardware.state_dict().items()}
    metrics = runner.update(runner.collect())
    assert metrics["hardware_updates"] == 0
    for key, value in runner.policy.hardware.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_checkpoint_roundtrip_rng_optimizer_schema_and_boundaries(tmp_path):
    runner = make_runner()
    batch = runner.collect()
    with pytest.raises(RuntimeError, match="boundaries"):
        runner.save(tmp_path / "early.pt")
    runner.update(batch)
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    expected = torch.rand(3)
    expected_hardware = runner.policy.hardware.sample(2, runner.hardware_generator)[0]
    resumed = make_runner()
    assert "simulator state not restored" in resumed.load(path)
    torch.testing.assert_close(torch.rand(3), expected)
    torch.testing.assert_close(resumed.policy.hardware.sample(2, resumed.hardware_generator)[0], expected_hardware)
    assert resumed.iteration == 1
    assert resumed.global_step == runner.global_step
    assert len(resumed.optimizer.state) == len(runner.optimizer.state)
    for key, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, runner.policy.state_dict()[key])
    with pytest.raises(ValueError, match="gamma"):
        make_runner(gamma=.5).load(path)


def test_hardware_kl_stop_does_not_stop_control():
    runner = make_runner()
    batch = runner.collect()
    with torch.no_grad():
        runner.policy.hardware.mean.add_(10)
    metrics = runner.update(batch)
    assert metrics["hardware_updates"] == 0
    assert metrics["control_updates"] == 12


def test_evaluation_heldout_seed_and_first_fall_masking_preserves_rng():
    runner = make_runner()
    expected = torch.get_rng_state().clone()
    report = runner.evaluate(replicas=2)
    torch.testing.assert_close(torch.get_rng_state(), expected)
    assert report["seed"] == 1_000_000
    assert report["return_mean"] == -2.
    assert report["survival_fraction"] == .5
    assert report["active_duration_fraction"] == .75
    assert report["velocity_tracking_rmse_m_s"] is None


def test_resume_rejects_changed_compiler_fingerprint(tmp_path):
    runner = make_runner()
    runner.compiler.fingerprint = "original"
    runner.save(tmp_path / "checkpoint.pt")
    other = make_runner()
    other.compiler.fingerprint = "changed"
    with pytest.raises(ValueError, match="compiler"):
        other.load(tmp_path / "checkpoint.pt")


def test_hardware_draws_are_independent_of_simulator_global_reseeding():
    runner = make_runner()
    generator_state = runner.hardware_generator.get_state()
    expected = runner.policy.hardware.sample(2, runner.hardware_generator)[0]
    runner.hardware_generator.set_state(generator_state)
    torch.manual_seed(1_234_567)
    torch.rand(1000)
    actual = runner.policy.hardware.sample(2, runner.hardware_generator)[0]
    torch.testing.assert_close(actual, expected)


def test_export_contains_asset_bundle_parameters_controller_and_joint_state(tmp_path):
    runner = make_runner()
    runner.update(runner.collect())
    assets = tmp_path / "compiled"
    assets.mkdir()
    xml = assets / "quadruped.xml"
    xml.write_text("<mujoco/>")
    (assets / "mesh.obj").write_text("v 0 0 0\n")
    design = SimpleNamespace(design_id="chosen", xml_path=xml, parameters={"a": 1.})
    runner.compiler.compile_many = lambda batch: [SimpleNamespace(design=design, error=None, design_id="chosen")]
    output = runner.export(tmp_path / "export")
    assert (output / "robot/mesh.obj").is_file()
    assert (output / "selected_parameters.yaml").is_file()
    assert (output / "joint_checkpoint.pt").is_file()
    controller = torch.load(output / "controller.pt", weights_only=False)
    assert "actor_state_dict" in controller
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["design_id"] == "chosen"
    assert "controller.pt" in manifest["files"]


def test_progress_and_latest_checkpoint_survive_next_cohort_failure(tmp_path, capsys):
    runner = make_runner(final_evaluation=False, final_export=False, eval_interval=0)
    runner.log_dir = tmp_path
    metrics = runner.learn(1)[0]
    assert (tmp_path / "model_initial.pt").is_file()
    assert torch.load(tmp_path / "model_latest.pt", weights_only=False)["iteration"] == 1
    assert metrics["sps"] > 0
    assert metrics["control_transitions"] == 6
    assert "hardware_mean=" in capsys.readouterr().out
    def fail(*args, **kwargs):
        raise RuntimeError("new cohort build failed")
    runner.env_factory = fail
    with pytest.raises(RuntimeError, match="cohort build"):
        runner.learn(1)
    assert torch.load(tmp_path / "model_latest.pt", weights_only=False)["iteration"] == 1


def test_selected_design_access_renews_compiler_lease_every_time():
    from draft.codesign.evaluation import selected_design
    runner = make_runner()
    original = runner.compiler.compile_many
    calls = []
    def counted(physical):
        calls.append(physical)
        return original(physical)
    runner.compiler.compile_many = counted
    selected_design(runner)
    selected_design(runner)
    assert len(calls) == 2


def test_fixed_nominal_mode_draws_exact_stock_without_hardware_rng_or_policy_drift():
    compiler = Compiler((True,))
    compiler.space.center = (.2, -.4)
    runner = make_runner(compiler, num_designs=1, optimize_hardware=False, hardware_sampling_mode="fixed")
    prior = {key: value.clone() for key, value in runner.policy.hardware.state_dict().items()}
    generator_state = runner.hardware_generator.get_state().clone()
    for _ in range(2):
        batch = runner.collect()
        np.testing.assert_array_equal(batch["physical"], [[.2, -.4]])
        torch.testing.assert_close(batch["normalized"], torch.tensor([[.2, -.4]]))
        assert runner.update(batch)["hardware_updates"] == 0
    torch.testing.assert_close(runner.hardware_generator.get_state(), generator_state)
    for key, value in runner.policy.hardware.state_dict().items():
        torch.testing.assert_close(value, prior[key], rtol=0, atol=0)


@pytest.mark.parametrize("overrides", [dict(num_designs=1), dict(hardware_sampling_mode="fixed"),
                                        dict(hardware_sampling_mode="unknown")])
def test_fixed_mode_and_single_design_validation(overrides):
    with pytest.raises(ValueError):
        make_runner(**overrides)


def test_fixed_to_joint_warmstart_transfers_controller_with_fresh_training_state(tmp_path):
    source = make_runner(Compiler((True,)), num_designs=1, optimize_hardware=False, hardware_sampling_mode="fixed")
    source.update(source.collect())
    checkpoint = tmp_path / "fixed.pt"
    source.save(checkpoint)
    target = make_runner()
    target.update(target.collect())
    assert target.optimizer.state
    target.warmstart(checkpoint)
    assert target.iteration == target.global_step == 0
    assert not target.optimizer.state
    assert target.config["hardware_sampling_mode"] == "stochastic"
    assert target.config["optimize_hardware"]
    assert target.control_schema == source.control_schema
    for key, value in target.policy.state_dict().items():
        torch.testing.assert_close(value, source.policy.state_dict()[key], rtol=0, atol=0)
    assert target.metadata["warmstart"]["source_sampling_mode"] == "fixed"
    # Full-state resume remains strict about switching the training protocol.
    with pytest.raises(ValueError, match="protected"):
        make_runner().load(checkpoint)


def test_native_fixed_to_joint_transfer_keeps_56_features_and_normalization(tmp_path):
    pytest.importorskip("rsl_rl")
    config = dict(device="cpu", num_designs=1, replicas=2, optimize_hardware=False,
                  hardware_sampling_mode="fixed", actor_hidden_dims=(8,), critic_hidden_dims=(8,), logger="none")
    source = CodesignRunner(None, Compiler((True,)), Env, config)
    observation = {key: torch.full((10, 56), 2.) for key in ("actor", "critic")}
    for tensor in observation.values():
        tensor[:, 48:55] = 0.
    source._ensure_control(observation, 1)
    source.policy.control.update_normalization(observation)
    with torch.no_grad():
        next(source.policy.control.actor.mlp.parameters()).fill_(.25)
    source.iteration, source.global_step = 7, 123
    checkpoint = tmp_path / "native_fixed.pt"
    source.save(checkpoint)
    target_config = dict(config, num_designs=2, optimize_hardware=True, hardware_sampling_mode="stochastic")
    target = CodesignRunner(None, Compiler(), Env, target_config)
    target.warmstart(checkpoint)
    assert target.control_schema["observation_shapes"] == {"actor": [56], "critic": [56]}
    assert target.iteration == target.global_step == 0
    assert not target.optimizer.state
    for key, value in target.policy.control.state_dict().items():
        torch.testing.assert_close(value, source.policy.control.state_dict()[key], rtol=0, atol=0)
    novel = {key: tensor.clone() for key, tensor in observation.items()}
    novel["actor"][:, 48:55] = 1.
    latent = target.policy.control.actor.get_latent(novel)
    torch.testing.assert_close(latent[:, 48:55], torch.ones(10, 7))


def test_first_update_evaluation_forces_early_video(tmp_path):
    runner = make_runner(eval_first_update=True, eval_interval=25, eval_video_interval=25,
                         final_evaluation=False, final_export=False)
    runner.log_dir = tmp_path
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        return dict(valid=True, return_mean=0., survival_fraction=1.)
    runner.evaluate = evaluate
    runner.learn(1)
    assert calls == [dict(record_video=True)]


@pytest.mark.parametrize("overrides", [dict(eval_video_width=3), dict(eval_video_height=1),
                                        dict(eval_video_fps=0), dict(eval_video_length_s=0)])
def test_video_configuration_rejected_before_collection(overrides):
    with pytest.raises(ValueError):
        make_runner(**overrides)


def test_native_checkpoint_bridge_requires_verified_matching_schema_and_layout(tmp_path):
    pytest.importorskip("rsl_rl")
    from draft.codesign.policy import NativeControlPolicy
    compiler = Compiler((True,))
    compiler.space.dim = 7
    compiler.space.names = tuple("abcdefg")
    compiler.space.center = (0.,) * 7
    compiler.space.lower = (-1.,) * 7
    compiler.space.upper = (1.,) * 7
    observation = {key: torch.zeros(8, 55) for key in ("actor", "critic")}
    actor = dict(class_name="draft.codesign.models:ConditionedMLPModel", hidden_dims=(8,), activation="elu",
                 obs_normalization=True, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    critic = dict(class_name="draft.codesign.models:ConditionedMLPModel", hidden_dims=(8,), activation="elu", obs_normalization=True)
    source = NativeControlPolicy.from_observation(observation, 12, actor, critic)
    source.update_normalization(observation)
    schema = dict(observation_shapes={key: [55] for key in observation}, num_actions=12)
    groups = {"actor": ["actor"], "critic": ["critic"]}
    config = dict(device="cpu", logger="none", actor_cfg=actor, critic_cfg=critic, obs_groups=groups,
                  normalization_exclusions={role: tuple(range(-7, 0)) for role in groups}, native_transfer_schema=schema)
    target = CodesignRunner(None, compiler, Env, config)
    prior = {key: tensor.clone() for key, tensor in target.policy.hardware.state_dict().items()}
    transfer = dict(version=1, space_schema=target._space_schema(), control_schema=schema,
                    actor_cfg=actor, critic_cfg=critic, obs_groups=groups,
                    conditioning_layout={role: list(range(48, 55)) for role in groups})
    payload = dict(actor_state_dict=source.actor.state_dict(), critic_state_dict=source.critic.state_dict(), iter=13,
                   infos={"codesign_transfer": transfer})
    path = tmp_path / "native.pt"
    torch.save(payload, path)
    target.warmstart(path)
    assert target.iteration == target.global_step == 0
    assert not target.optimizer.state
    assert target.metadata["warmstart"]["source_sampling_mode"] == "native_conditioned"
    for key, tensor in target.policy.control.state_dict().items():
        torch.testing.assert_close(tensor, source.state_dict()[key], rtol=0, atol=0)
    for key, tensor in target.policy.hardware.state_dict().items():
        torch.testing.assert_close(tensor, prior[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="explicitly verified"):
        CodesignRunner(None, compiler, Env, dict(config, native_transfer_schema=None)).warmstart(path)
    wrong_schema = dict(observation_shapes={key: [56] for key in observation}, num_actions=12)
    with pytest.raises(ValueError, match="schema mismatch"):
        CodesignRunner(None, compiler, Env, dict(config, native_transfer_schema=wrong_schema)).warmstart(path)
    with pytest.raises(ValueError, match="feature layout"):
        CodesignRunner(None, compiler, Env, dict(config, normalization_exclusions={role: tuple(range(-8, -1)) for role in groups})).warmstart(path)


def test_control_minibatch_cap_and_chunked_normalization_preserve_hardware_exposure():
    runner = make_runner(max_control_batch_size=2, num_mini_batches=1, rollout_storage_device="cpu")
    evaluate = runner.policy.control.evaluate
    updates, normalization = [], []
    def scored(observation, actions=None):
        if actions is not None:
            updates.append(len(actions))
        return evaluate(observation, actions)
    def normalize(observation):
        assert runner.policy.control.training
        normalization.append(len(observation["actor"]))
    runner.policy.control.evaluate = scored
    runner.policy.control.update_normalization = normalize
    batch = runner.collect()
    assert all(tensor.device.type == "cpu" for tensor in batch["control"]["observations"].values())
    metrics = runner.update(batch)
    assert metrics["hardware_updates"] == 3
    assert metrics["control_updates"] == 9
    assert updates == [2] * 9
    assert normalization == [2] * 3
    assert not runner.policy.control.training


def test_cpu_storage_and_nonbinding_cap_match_uncapped_cpu_update():
    torch.manual_seed(41)
    original = make_runner()
    torch.manual_seed(41)
    capped = make_runner(rollout_storage_device="cpu", max_control_batch_size=2)
    torch.manual_seed(98)
    first = original.collect()
    torch.manual_seed(98)
    second = capped.collect()
    for key in ("actions", "logprob", "values", "rewards", "active"):
        torch.testing.assert_close(first["control"][key], second["control"][key], rtol=0, atol=0)
    torch.manual_seed(45)
    first_metrics = original.update(first)
    torch.manual_seed(45)
    second_metrics = capped.update(second)
    assert first_metrics["hardware_updates"] == second_metrics["hardware_updates"]
    assert first_metrics["control_updates"] == second_metrics["control_updates"]
    for key, value in original.policy.state_dict().items():
        torch.testing.assert_close(value, capped.policy.state_dict()[key], rtol=0, atol=0)


def test_elapsed_native_step_counter_survives_rebuild_and_checkpoint(tmp_path):
    starts = []
    class CounterEnv(Env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.common_step_counter = 0
        def reset(self):
            starts.append(self.common_step_counter)
            return super().reset()
        def step(self, actions):
            self.common_step_counter += 1
            return super().step(actions)
    runner = make_runner()
    runner.env_factory = CounterEnv
    runner.update(runner.collect())
    runner.update(runner.collect())
    assert starts == [0, 4] and runner.elapsed_env_steps == 8
    path = tmp_path / "counter.pt"
    runner.save(path)
    resumed = make_runner()
    resumed.env_factory = CounterEnv
    resumed.load(path)
    resumed.collect()
    assert starts == [0, 4, 8] and resumed.elapsed_env_steps == 12


def test_chunked_native_normalization_actually_updates_after_frozen_collection():
    pytest.importorskip("rsl_rl")
    from draft.codesign.policy import NativeControlPolicy
    obs = {role: torch.zeros(2, 2) for role in ("actor", "critic")}
    actor = dict(hidden_dims=(8,), activation="elu", obs_normalization=True,
                 distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    critic = dict(hidden_dims=(8,), activation="elu", obs_normalization=True)
    control = NativeControlPolicy.from_observation(obs, 1, actor, critic)
    config = dict(make_runner().config, rollout_storage_device="cpu", max_control_batch_size=2)
    runner = CodesignRunner(JointPolicy(control, HardwarePolicy(2)), Compiler(), Env, config)
    batch = runner.collect()
    assert control.actor.obs_normalizer.count.item() == 0
    active = batch["control"]["active"].flatten()
    expected = batch["control"]["observations"]["actor"].flatten(0, 1)[active].mean(0)
    runner.update(batch)
    assert control.actor.obs_normalizer.count.item() == 6
    torch.testing.assert_close(control.actor.obs_normalizer.mean, expected)


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_control_minibatch_cap_rejected(cap):
    with pytest.raises(ValueError, match="max_control_batch_size"):
        make_runner(max_control_batch_size=cap)
