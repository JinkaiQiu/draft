"""CPU integration of topology draws, checkpoints, transfer and native masks."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from draft.codesign.coadaptation import CoadaptationCodesignRunner
from draft.codesign.checkpoint_conversion import expand_topology_checkpoint
from draft.codesign.evaluation import selected_design
from draft.codesign.policy import NativeControlPolicy, TopologyHardwarePolicy
from draft.codesign.space import DesignSpace


class InvalidCompiler:
    def __init__(self, space):
        self.space = space
        self.fingerprint = "topology-test-compiler"
        self.compiled = []

    def compile_many(self, physical):
        self.compiled.append(np.asarray(physical).copy())
        return [SimpleNamespace(design=None, design_id=None, error="infeasible") for _ in physical]


def no_environment(*args, **kwargs):
    raise AssertionError("Invalid-only draws must not create a simulator")


def model_configs(topology):
    common = dict(class_name="draft.codesign.models:ConditionedMLPModel", hidden_dims=(8,),
                  activation="elu", obs_normalization=True,
                  conditioning_dims=19 if topology else 7, joint_topology=topology)
    actor = dict(common, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    return actor, dict(common)


def make_runner(space=None, **overrides):
    space = space or DesignSpace.quadruped(joint_topology=True)
    actor, critic = model_configs(True)
    groups = {role: [role] for role in ("actor", "critic")}
    config = dict(device="cpu", logger="none", joint_topology=True, num_designs=4, replicas=1,
                  coadaptation_cohort_lifetime="window", staleness_diagnostics=False,
                  controller_updates_per_cycle=1, controller_train_cfg=dict(num_steps_per_env=1),
                  num_learning_epochs=2, hardware_target_kl=100., seed=11,
                  actor_cfg=actor, critic_cfg=critic, obs_groups=groups,
                  normalization_exclusions={role: tuple(range(-19, 0)) for role in groups})
    config.update(overrides)
    compiler = InvalidCompiler(space)
    return CoadaptationCodesignRunner(None, compiler, no_environment, config)


def test_coadaptation_draws_real_categorical_ppo_and_selected_design():
    runner = make_runner(hardware_entropy_coef=.01)
    assert isinstance(runner.policy.hardware, TopologyHardwarePolicy)
    batch, designs = runner._prepare_hardware_batch()
    assert designs == []
    assert batch["raw"].shape == (4, 14)
    assert batch["normalized"].shape == (4, 19)
    assert batch["physical"].shape == (4, 19)
    assert np.array_equal(batch["physical"][:, 7:13], batch["normalized"][:, 7:13].numpy())
    assert np.all(batch["physical"][:, 13:19] * batch["physical"][:, 7:13] == 0)
    torch.testing.assert_close(batch["old_logprob"], runner.policy.hardware.evaluate(batch["raw"])[0])
    # The invalid-only path still takes actual hardware optimizer steps.
    before = runner.policy.hardware.topology_logits.detach().clone()
    collected = runner.collect()
    metrics = runner.update(collected)
    assert metrics["hardware_updates"] == 2
    assert metrics["invalid_designs"] == 4
    assert runner.policy.control is None
    assert not torch.equal(runner.policy.hardware.topology_logits, before)
    result, normalized, physical = selected_design(runner)
    assert result.design is None
    np.testing.assert_allclose(normalized, runner.policy.hardware.decode(runner.policy.hardware.mean).numpy())
    np.testing.assert_allclose(physical, runner.compiler.compiled[-1][0])


def test_fixed_draws_use_all_active_reference_geometry():
    runner = make_runner(optimize_hardware=False, hardware_sampling_mode="fixed")
    batch, _ = runner._prepare_hardware_batch()
    assert np.all(batch["raw"][:, 0].numpy() == 63)
    np.testing.assert_allclose(batch["physical"], [runner.compiler.space.center] * 4)
    assert np.all(batch["normalized"][:, 13:].numpy() == 0)
    torch.testing.assert_close(batch["old_logprob"], runner.policy.hardware.evaluate(batch["raw"])[0])


def test_topology_checkpoint_round_trip_and_strict_schema(tmp_path):
    runner = make_runner()
    runner.update(runner.collect())
    path = tmp_path / "topology.pt"
    runner.save(path)
    restored = make_runner()
    restored.load(path)
    assert restored.iteration == runner.iteration
    for key, value in runner.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[key], value, rtol=0, atol=0)
    changed = make_runner(DesignSpace.quadruped(joint_topology=True,
                                                lock_angle_bounds={"front_knee": (-2.7, 0.)}))
    with pytest.raises(ValueError, match="schema mismatch"):
        changed.load(path)


def native_geometry_checkpoint(tmp_path, old_space):
    actor, critic = model_configs(False)
    groups = {role: [role] for role in ("actor", "critic")}
    observation = {role: torch.randn(6, 55) for role in groups}
    source = NativeControlPolicy.from_observation(observation, 12, actor, critic, groups,
                                                  normalization_exclusions={role: tuple(range(-7, 0)) for role in groups})
    source.update_normalization(observation)
    with torch.no_grad():
        for model in (source.actor, source.critic):
            model.mlp[0].weight[:, -7:] = torch.arange(1, 8, dtype=torch.float32)[None, :] * .03
    schema = dict(observation_shapes={role: [55] for role in groups}, num_actions=12)
    transfer = dict(version=1, space_schema=old_space.to_dict(), control_schema=schema,
                    actor_cfg=actor, critic_cfg=critic, obs_groups=groups,
                    conditioning_layout={role: list(range(48, 55)) for role in groups},
                    compiler_fingerprint="topology-test-compiler")
    path = tmp_path / "geometry_native.pt"
    torch.save(dict(actor_state_dict=source.actor.state_dict(), critic_state_dict=source.critic.state_dict(),
                    iter=13, infos={"codesign_transfer": transfer}), path)
    return source, observation, path


def test_offline_expansion_then_warmstart_preserves_function_with_bounds_remap(tmp_path):
    old_space = DesignSpace.quadruped(relative_range=.15)
    target_space = DesignSpace.quadruped(relative_range=.5, joint_topology=True)
    source, observation, path = native_geometry_checkpoint(tmp_path, old_space)
    expanded_path = tmp_path / "topology_native.pt"
    expand_topology_checkpoint(path, expanded_path)
    runner = make_runner(target_space)
    prior = deepcopy(runner.policy.hardware.state_dict())
    # Training authors its target observation shape; checkpoint loading cannot
    # infer a different target shape or expand the checkpoint itself.
    runner._ensure_control({role: torch.zeros(1, 67) for role in ("actor", "critic")}, 12)
    runner.warmstart(expanded_path)
    assert runner.control_schema["observation_shapes"] == {"actor": [67], "critic": [67]}
    assert runner.iteration == runner.global_step == 0 and not runner.optimizer.state
    assert runner.metadata["warmstart"]["checkpoint"] == str(expanded_path.resolve())
    for key, value in prior.items():
        torch.testing.assert_close(runner.policy.hardware.state_dict()[key], value, rtol=0, atol=0)
    physical = np.asarray(old_space.center) * np.asarray([1.04, .96, 1.08, .94, 1.01, .92, 1.05])
    old_geometry = torch.as_tensor(old_space.normalize(physical), dtype=torch.float32)
    new_geometry = torch.as_tensor(target_space.geometry_space().normalize(physical), dtype=torch.float32)
    for role in ("actor", "critic"):
        old_obs = {role: observation[role].clone()}
        old_obs[role][:, -7:] = old_geometry
        target_obs = {role: torch.cat((observation[role][:, :-7], new_geometry.expand(6, -1),
                                       torch.ones(6, 6), torch.zeros(6, 6)), -1)}
        old_model = getattr(source, role)
        new_model = getattr(runner.policy.control, role)
        torch.testing.assert_close(new_model.mlp(new_model.get_latent(target_obs)),
                                   old_model.mlp(old_model.get_latent(old_obs)), rtol=1e-5, atol=1e-5)
        assert not new_model.mlp[0].weight[:, -12:].count_nonzero()
        if new_model.obs_normalization:
            torch.testing.assert_close(new_model.obs_normalizer._mean[:, -19:], torch.zeros(1, 19))
    mask_obs = {"actor": target_obs["critic"].clone(), "critic": target_obs["critic"].clone()}
    mask_obs["actor"][:, -12:-6] = torch.tensor([0., 1., 0., 1., 0., 1.])
    actions, logprob, entropy, _ = runner.policy.control.evaluate(mask_obs)
    expected_mask = torch.tensor([0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1.]).expand(6, -1)
    torch.testing.assert_close(actions * (1 - expected_mask), torch.zeros_like(actions))
    actor_model = runner.policy.control.actor
    changed = actions + (1 - expected_mask) * 5.
    torch.testing.assert_close(actor_model.get_output_log_prob(changed), logprob)
    assert entropy.shape == (6,)
    old_params = actor_model.distribution.params
    new_params = (old_params[0] + (1 - expected_mask) * 5., old_params[1])
    torch.testing.assert_close(actor_model.get_kl_divergence(old_params, new_params), torch.zeros(6))
    # Native PPO calls actor.forward for each minibatch immediately before KL.
    # Verify its mutable action mask follows the newest observation batch.
    next_obs = {"actor": mask_obs["actor"][:2].clone()}
    next_obs["actor"][:, -12:-6] = torch.tensor([1., 0., 1., 0., 1., 0.])
    actor_model(next_obs, stochastic_output=True)
    next_mask = torch.tensor([1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0.]).expand(2, -1)
    before_params = actor_model.output_distribution_params
    after_params = (before_params[0] + (1 - next_mask) * 3., before_params[1])
    torch.testing.assert_close(actor_model.get_kl_divergence(before_params, after_params), torch.zeros(2))


def test_online_warmstart_rejects_unexpanded_geometry_checkpoint_without_mutation(tmp_path):
    _, _, path = native_geometry_checkpoint(tmp_path, DesignSpace.quadruped())
    runner = make_runner()
    before = deepcopy(runner.policy.state_dict())
    with pytest.raises(ValueError, match="Expand it offline.*expand_codesign_checkpoint"):
        runner.warmstart(path)
    assert not runner.control_initialized
    for key, value in before.items():
        torch.testing.assert_close(runner.policy.state_dict()[key], value, rtol=0, atol=0)
    assert runner.iteration == runner.global_step == 0 and not runner.optimizer.state
