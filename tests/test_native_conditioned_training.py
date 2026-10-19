"""Fixed conditioning is the D=1 case of native randomized cohort training."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from mjlab.rl import MjlabOnPolicyRunner
from draft.codesign.sampling import ConditioningSamplingCfg, sample_designs
from draft.codesign.space import DesignSpace
from draft.tasks.native_runner import MetadataOnPolicyRunner
from draft.tasks.quadruped.runner_cfg import QuadrupedLocomotionRunnerCfg


def test_native_learner_and_ppo_configuration_are_inherited_unchanged():
    assert MetadataOnPolicyRunner.learn is MjlabOnPolicyRunner.learn
    original = QuadrupedLocomotionRunnerCfg().build_rsl_rl_cfg()
    conditioned_cfg = QuadrupedLocomotionRunnerCfg(env_kwargs={"morphology_conditioning": {"mode": "fixed"}})
    conditioned = conditioned_cfg.build_rsl_rl_cfg()
    assert conditioned["algorithm"] == original["algorithm"]
    assert conditioned["num_steps_per_env"] == original["num_steps_per_env"] == 24
    for role in ("actor", "critic"):
        model = dict(conditioned[role])
        model["class_name"] = original[role]["class_name"]
        assert model == original[role]
    assert conditioned_cfg.n_envs == 4096
    assert conditioned["algorithm"]["schedule"] == "adaptive"


def test_other_tasks_require_explicit_morphology_integration():
    from draft.tasks import get_task
    task = get_task("quadruped_velocity")
    config = task.make_runner_cfg()
    config.codesign = {"mode": "fixed"}
    with pytest.raises(ValueError, match="has not opted into"):
        task.training_handler(config)
    with pytest.raises(ValueError, match="has not opted into"):
        task.playback_handler(config)


@pytest.mark.parametrize("options", [dict(mode="fixed", num_designs=2), dict(num_designs=0),
                                    dict(mode="randomized", num_designs=16, max_attempts=4),
                                    dict(mode="other"), dict(latent_std=0.)])
def test_sampling_rejects_invalid_protocol(options):
    with pytest.raises(ValueError):
        ConditioningSamplingCfg.parse(options)


def test_randomized_sampling_reproducible_and_renews_complete_accepted_cohort(monkeypatch, tmp_path):
    from draft.codesign import sampling
    from draft.tasks.quadruped import quadruped_entity
    space = DesignSpace.quadruped()
    parameters = tmp_path / "parameters.yaml"
    parameters.write_text("{}\n")
    reference = dict(robot_xml=str(tmp_path / "stock.xml"), parameters_path=str(parameters),
                     robot_xml_sha256="stock", physical=list(space.center), normalized=[0.] * 7)
    monkeypatch.setattr(quadruped_entity, "measure_quadruped_leg_length", lambda path: .48)

    class Compiler:
        instances = []
        def __init__(self, *args, **kwargs):
            self.calls = []
            self.instances.append(self)
        def compile_many(self, physical):
            self.calls.append(np.asarray(physical).copy())
            return [SimpleNamespace(design=SimpleNamespace(physical=tuple(row)), error=None) for row in physical]
        def close(self):
            pass

    monkeypatch.setattr(sampling, "DesignCompiler", Compiler)
    config, _ = ConditioningSamplingCfg.parse(dict(mode="randomized", num_designs=3, seed=19))
    designs, _, rejected = sample_designs(config, space, reference)
    again, _, _ = sample_designs(config, space, reference)
    np.testing.assert_array_equal([d.physical for d in designs], [d.physical for d in again])
    assert len(set(d.physical for d in designs)) == 3 and not rejected
    np.testing.assert_array_equal(Compiler.instances[0].calls[-1], [d.physical for d in designs])
    stock, compiler, rejected = sample_designs(ConditioningSamplingCfg(), space, reference)
    assert len(stock) == 1 and compiler is None and not rejected
    assert stock[0].xml_path == Path(reference["robot_xml"])
    assert stock[0].physical == space.center


def test_metadata_checkpoint_keeps_native_format_and_atomically_publishes_latest(tmp_path):
    runner = MetadataOnPolicyRunner.__new__(MetadataOnPolicyRunner)
    runner.cfg = {"upload_model": False}
    runner.checkpoint_metadata = {"codesign_transfer": {"version": 1}}
    runner.current_learning_iteration = 7
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=192))
    runner.alg = SimpleNamespace(save=lambda: dict(actor_state_dict={"weight": torch.ones(2)},
                                                 critic_state_dict={}, optimizer_state_dict={}))
    runner.save(str(tmp_path / "model_7.pt"))
    saved = torch.load(tmp_path / "model_latest.pt", weights_only=False)
    assert saved["iter"] == 7 and saved["infos"]["env_state"]["common_step_counter"] == 192
    assert saved["infos"]["codesign_transfer"] == {"version": 1}
    assert (tmp_path / "model_7.pt").read_bytes() == (tmp_path / "model_latest.pt").read_bytes()
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_metadata_runner_preserves_native_logger_and_learning(monkeypatch):
    native_logger = SimpleNamespace(writer=None, native_state=object())
    monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", lambda self, *args, **kwargs: setattr(self, "logger", native_logger))
    metadata = {"morphology_conditioning": dict(feature_names=["torso_lx"], physical=[.2])}
    runner = MetadataOnPolicyRunner(checkpoint_metadata=metadata)
    assert runner.logger is native_logger
    assert MetadataOnPolicyRunner.learn is MjlabOnPolicyRunner.learn
