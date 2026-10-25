"""Offline geometry-to-topology checkpoint conversion contracts."""

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from draft.codesign.checkpoint_conversion import expand_checkpoint_payload, expand_topology_checkpoint
from draft.codesign.policy import NativeControlPolicy
from draft.codesign.space import DesignSpace


def source_checkpoint(full=False):
    space = DesignSpace.quadruped()
    groups = {"actor": ["actor"], "critic": ["critic"]}
    cfg = dict(class_name="draft.codesign.models:ConditionedMLPModel", hidden_dims=(8,),
               activation="elu", obs_normalization=True, conditioning_dims=7,
               joint_topology=False)
    actor = dict(cfg, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))
    observation = {role: torch.randn(5, 55) for role in groups}
    policy = NativeControlPolicy.from_observation(observation, 12, actor, cfg, groups,
                                                  {role: tuple(range(-7, 0)) for role in groups})
    policy.update_normalization(observation)
    with torch.no_grad():
        policy.actor.mlp[0].weight[:, -7:] = .125
        policy.critic.mlp[0].weight[:, -7:] = -.25
    schema = dict(observation_shapes={role: [55] for role in groups}, num_actions=12)
    transfer = dict(version=1, space_schema=space.to_dict(), control_schema=schema,
                    actor_cfg=actor, critic_cfg=cfg, obs_groups=groups,
                    conditioning_layout={role: list(range(48, 55)) for role in groups},
                    compiler_fingerprint="geometry-only-compiler")
    if full:
        payload = dict(policy={f"control.{role}.{key}": value for role in groups
                               for key, value in getattr(policy, role).state_dict().items()},
                       control_initialized=True, control_schema=schema, space_schema=space.to_dict(),
                       config=dict(actor_cfg=actor, critic_cfg=cfg, obs_groups=groups,
                                   normalization_exclusions={role: tuple(range(-7, 0)) for role in groups}),
                       compiler_fingerprint="geometry-only-compiler", iteration=13,
                       optimizer={"must_not_transfer": True})
    else:
        payload = dict(actor_state_dict=policy.actor.state_dict(),
                       critic_state_dict=policy.critic.state_dict(), iter=13,
                       infos=dict(codesign_transfer=transfer), optimizer={"must_not_transfer": True})
    return payload, policy, observation


@pytest.mark.parametrize("full", [False, True])
def test_expansion_preserves_geometry_function_and_native_load_shape(full):
    payload, source, observation = source_checkpoint(full)
    original = deepcopy(payload)
    converted, report = expand_checkpoint_payload(payload)
    assert report["input_shapes"] == {"actor": 55, "critic": 55}
    assert report["output_shapes"] == {"actor": 67, "critic": 67}
    assert set(converted) == {"actor_state_dict", "critic_state_dict", "iter", "infos"}
    assert converted["iter"] == 13
    transfer = converted["infos"]["codesign_transfer"]
    assert transfer["space_schema"]["version"] == 2
    source_schema = original["space_schema"] if full else original["infos"]["codesign_transfer"]["space_schema"]
    assert transfer["space_schema"]["parameters"] == source_schema["parameters"]
    assert transfer["space_schema"]["reference"] == source_schema["reference"]
    assert "compiler_fingerprint" not in transfer
    target = NativeControlPolicy.from_observation(
        {role: torch.zeros(5, 67) for role in ("actor", "critic")}, 12,
        transfer["actor_cfg"], transfer["critic_cfg"], transfer["obs_groups"],
        {role: tuple(range(-19, 0)) for role in ("actor", "critic")})
    for role in ("actor", "critic"):
        state = converted[role + "_state_dict"]
        original_state = getattr(source, role).state_dict()
        torch.testing.assert_close(state["mlp.0.weight"][:, :55], original_state["mlp.0.weight"])
        assert not state["mlp.0.weight"][:, 55:].count_nonzero()
        for field in ("_mean", "_var", "_std"):
            key = "obs_normalizer." + field
            torch.testing.assert_close(state[key][:, :55], original_state[key])
            torch.testing.assert_close(state[key][:, 55:], torch.zeros(1, 12) if field == "_mean" else torch.ones(1, 12))
        getattr(target, role).load_state_dict(state, strict=True)
        all_active = torch.cat((observation[role], torch.ones(5, 6), torch.zeros(5, 6)), -1)
        old = getattr(source, role).mlp(getattr(source, role).get_latent({role: observation[role]}))
        new = getattr(target, role).mlp(getattr(target, role).get_latent({role: all_active}))
        torch.testing.assert_close(new, old)
        masked = all_active.clone()
        masked[:, -12:-6] = torch.tensor([0., 1., 0., 1., 0., 1.])
        masked[:, -6:] = torch.randn(5, 6)
        torch.testing.assert_close(getattr(target, role).mlp(getattr(target, role).get_latent({role: masked})), old)
    if full:
        assert payload["optimizer"] == original["optimizer"]


def test_rejects_malformed_source_and_double_expansion():
    payload, _, _ = source_checkpoint()
    payload["infos"]["codesign_transfer"]["conditioning_layout"]["actor"] = list(range(47, 54))
    with pytest.raises(ValueError, match="trailing geometry"):
        expand_checkpoint_payload(payload)
    payload, _, _ = source_checkpoint()
    payload["actor_state_dict"]["mlp.0.weight"] = payload["actor_state_dict"]["mlp.0.weight"][:, :-1]
    with pytest.raises(ValueError, match="state does not match"):
        expand_checkpoint_payload(payload)
    payload, _, _ = source_checkpoint()
    payload["infos"]["codesign_transfer"]["critic_cfg"]["hidden_dims"] = (9,)
    with pytest.raises(ValueError, match="critic state does not match"):
        expand_checkpoint_payload(payload)
    payload, _, _ = source_checkpoint()
    converted, _ = expand_checkpoint_payload(payload)
    with pytest.raises(ValueError, match="version-one geometry"):
        expand_checkpoint_payload(converted)
    full_payload, _, _ = source_checkpoint(full=True)
    full_payload["control_initialized"] = False
    with pytest.raises(ValueError, match="no initialized controller"):
        expand_checkpoint_payload(full_payload)


def test_rejects_invalid_bounds_actions_and_shared_group_growth():
    payload, _, _ = source_checkpoint()
    with pytest.raises(ValueError, match="Lock-angle bounds"):
        expand_checkpoint_payload(payload, lock_angle_bounds=False)
    with pytest.raises(ValueError, match="Lock-angle bounds"):
        expand_checkpoint_payload(payload, lock_angle_bounds=[])
    payload["infos"]["codesign_transfer"]["control_schema"]["num_actions"] = 11
    with pytest.raises(ValueError, match="12-action"):
        expand_checkpoint_payload(payload)
    payload, _, _ = source_checkpoint()
    transfer = payload["infos"]["codesign_transfer"]
    transfer["control_schema"]["observation_shapes"] = {"actor": (48,), "critic": (7,)}
    transfer["obs_groups"] = {"actor": ["actor", "critic"], "critic": ["critic", "actor"]}
    with pytest.raises(ValueError, match="more than twelve"):
        expand_checkpoint_payload(payload)


def test_tuple_observation_shape_converts_to_mutable_target_shape():
    payload, _, _ = source_checkpoint()
    shapes = payload["infos"]["codesign_transfer"]["control_schema"]["observation_shapes"]
    shapes.update({role: (55,) for role in shapes})
    converted, _ = expand_checkpoint_payload(payload)
    assert converted["infos"]["codesign_transfer"]["control_schema"]["observation_shapes"] == {
        "actor": [67], "critic": [67]}


def test_file_conversion_is_atomic_no_overwrite_and_cli(tmp_path):
    payload, _, _ = source_checkpoint()
    source = tmp_path / "source.pt"
    target = tmp_path / "target.pt"
    torch.save(payload, source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = expand_topology_checkpoint(source, target, lock_angle_bounds={"front_knee": (-2.5, 0.)})
    assert report["source_sha256"] == before and report["source_path"] == str(source.resolve())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["infos"]["codesign_transfer"]["space_schema"]["joint_topology"]["lock_angle_bounds"][2] == [-2.5, 0.]
    with pytest.raises(FileExistsError):
        expand_topology_checkpoint(source, target)
    cli_target = tmp_path / "cli.pt"
    script = Path(__file__).parents[1] / "scripts" / "expand_codesign_checkpoint.py"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(__file__).parents[1] / "src"),
                                                       os.environ.get("PYTHONPATH", ""))))
    result = subprocess.run([sys.executable, str(script), "--input", str(source), "--output", str(cli_target)],
                            check=True, capture_output=True, text=True, env=env)
    assert cli_target.is_file() and "geometry7_to_topology19" in result.stdout
