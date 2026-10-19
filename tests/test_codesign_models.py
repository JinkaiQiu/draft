import copy

import pytest
import torch

pytest.importorskip("rsl_rl")

from rsl_rl.models import MLPModel
from draft.codesign.models import ConditionedMLPModel
from draft.codesign.policy import NativeControlPolicy


def configuration():
    return dict(hidden_dims=(16, 8), activation="elu", obs_normalization=True,
                distribution_cfg=dict(class_name="GaussianDistribution", init_std=.5, std_type="scalar"))


def test_conditioned_model_preserves_native_initialization_rng_and_stock_behavior():
    groups = {"actor": ["actor"]}
    state = torch.randn(12, 48)
    native_obs = {"actor": state}
    conditioned_obs = {"actor": torch.cat((state, torch.zeros(12, 7)), dim=-1)}
    torch.manual_seed(1234)
    native = MLPModel(native_obs, groups, "actor", 12, **configuration())
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    conditioned = ConditionedMLPModel(conditioned_obs, groups, "actor", 12, **configuration())
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    for key, value in native.state_dict().items():
        actual = conditioned.state_dict()[key]
        if key == "mlp.0.weight" or key in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var"):
            actual = actual[..., :48]
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    assert not conditioned.mlp[0].weight[:, -7:].count_nonzero()
    torch.testing.assert_close(conditioned(conditioned_obs), native(native_obs), rtol=1e-6, atol=1e-7)
    native.update_normalization(native_obs)
    conditioned.update_normalization(conditioned_obs)
    torch.testing.assert_close(conditioned(conditioned_obs), native(native_obs), rtol=1e-6, atol=1e-7)
    novel = {"actor": conditioned_obs["actor"].clone()}
    novel["actor"][:, -7:] = .8
    torch.testing.assert_close(conditioned.get_latent(novel)[:, -7:], torch.full((12, 7), .8))


def test_native_conditioned_checkpoint_load_and_jit_keep_identity_conditioning():
    obs = {"actor": torch.ones(4, 55)}
    model = ConditionedMLPModel(obs, {"actor": ["actor"]}, "actor", 12, **configuration())
    saved = copy.deepcopy(model.state_dict())
    saved["obs_normalizer._mean"][:, -7:] = 100.
    saved["obs_normalizer._std"][:, -7:] = 0.
    model.load_state_dict(saved)
    torch.testing.assert_close(model.get_latent(obs)[:, -7:], torch.ones(4, 7))
    scripted = torch.jit.script(model.as_jit())
    torch.testing.assert_close(scripted(obs["actor"]), model(obs))


@pytest.mark.parametrize("class_name", [ConditionedMLPModel, "draft.codesign.models:ConditionedMLPModel"])
def test_control_adapter_accepts_native_conditioned_model(class_name):
    obs = {"actor": torch.zeros(2, 55), "critic": torch.zeros(2, 55)}
    actor = dict(configuration(), class_name=class_name)
    critic = dict(hidden_dims=(16, 8), activation="elu", obs_normalization=True, class_name=class_name)
    policy = NativeControlPolicy.from_observation(obs, 12, actor, critic)
    assert isinstance(policy.actor, ConditionedMLPModel)
    assert policy.evaluate(obs)[0].shape == (2, 12)


def test_original_runner_defaults_and_opt_in_conditioned_model():
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.runner_cfg import QuadrupedLocomotionRunnerCfg
    cfg = QuadrupedLocomotionRunnerCfg()
    original = cfg.build_rsl_rl_cfg()
    assert cfg.n_envs == 4096
    assert original["num_steps_per_env"] == 24
    assert original["algorithm"]["schedule"] == "adaptive"
    assert original["actor"]["class_name"] == "MLPModel"
    cfg.env_kwargs["morphology_conditioning"] = True
    conditioned = cfg.build_rsl_rl_cfg()
    assert conditioned["actor"]["class_name"] == "draft.codesign.models:ConditionedMLPModel"
    assert conditioned["critic"]["class_name"] == "draft.codesign.models:ConditionedMLPModel"
    assert conditioned["algorithm"] == original["algorithm"]
