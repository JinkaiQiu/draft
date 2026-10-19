"""Morphology conditioning changes observations only on the ordinary task."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from draft.codesign.space import DesignSpace, GEOMETRY_NAMES
from draft.generation.generator import RobotGenerator
from draft.robots import robot_dir
from draft.tasks.quadruped import env as task
from draft.tasks.quadruped.conditioning import (
    MorphologyObservation, append_morphology_observations, condition_quadruped_cfg,
)

pytestmark = pytest.mark.needs_rl


@pytest.fixture(scope="module")
def native_cfg(tmp_path_factory):
    path = tmp_path_factory.mktemp("conditioned-cheetah")
    space = DesignSpace.quadruped()
    RobotGenerator(robot_dir("quadruped")).generate(path, params_override=space.reference_parameters)
    xml = path / "quadruped.xml"
    return task.make_env_cfg(robot_xml=xml, n_envs=3, episode_length_s=20.0), xml


def test_opt_in_preserves_entire_native_configuration_except_seven_observations(native_cfg, monkeypatch):
    original, xml = native_cfg
    calls = []

    def base_factory(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(original), object()

    monkeypatch.setattr(task, "make_base_env_cfg", base_factory)
    from draft.tasks.quadruped.codesign import native_env
    monkeypatch.setattr(native_env, "make_base_env_cfg", base_factory)
    inputs = dict(robot_xml=xml, n_envs=3, episode_length_s=20.0)
    unchanged = task.make_env_cfg(**inputs)
    conditioned = task.make_env_cfg(**inputs, morphology_conditioning=True)
    assert calls == [inputs, {**inputs, "task_geometry_xml": xml.resolve()}]
    assert not hasattr(unchanged, "morphology_conditioning_metadata")
    # All terrain, sensors, physical randomization, actions, commands, rewards,
    # terminations, curriculum and episode configuration remain exactly equal.
    for group in ("actor", "critic"):
        assert list(conditioned.observations[group].terms) == [*unchanged.observations[group].terms, "morphology"]
        assert "remaining_time" not in conditioned.observations[group].terms
        term = conditioned.observations[group].terms.pop("morphology")
        assert term.noise is None and term.scale is None
        observation = MorphologyObservation(term, SimpleNamespace(device="cpu", num_envs=3))
        value = observation(None, None)
        torch.testing.assert_close(value, torch.zeros((3, 7)))
        assert observation(None, None).data_ptr() == value.data_ptr()
    assert asdict(conditioned) == asdict(unchanged)
    metadata = conditioned.morphology_conditioning_metadata
    assert metadata["feature_names"] == list(GEOMETRY_NAMES)
    assert metadata["space_schema"] == DesignSpace.quadruped().to_dict()
    np.testing.assert_allclose(metadata["physical"], DesignSpace.quadruped().center)
    assert len(metadata["robot_xml_sha256"]) == len(metadata["parameters_sha256"]) == 64


def test_reusable_augmentation_accepts_heterogeneous_rows_without_other_changes(native_cfg):
    original, _ = native_cfg
    cfg = copy.deepcopy(original)
    values = np.asarray([[-1.] * 7, [0.] * 7, [1.] * 7])
    append_morphology_observations(cfg, values)
    term = cfg.observations["actor"].terms["morphology"]
    obs = MorphologyObservation(term, SimpleNamespace(device="cpu", num_envs=3))
    np.testing.assert_array_equal(obs(None, None), values)
    with pytest.raises(ValueError, match="no morphology"):
        append_morphology_observations(cfg, values)


def test_parameter_table_precedence_and_complete_geometry_validation(native_cfg, tmp_path):
    original, xml = native_cfg
    copied = tmp_path / "quadruped.xml"
    copied.write_text(xml.read_text())
    # Only inference reads this XML: geometry remains the native cfg's own robot.
    resolved = yaml.safe_load((xml.parent / "parameters_resolved.yaml").read_text())
    (tmp_path / "parameters_from_gui.yaml").write_text(yaml.safe_dump({name: 9. for name in GEOMETRY_NAMES}))
    resolved_path = tmp_path / "parameters_resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved))
    cfg = condition_quadruped_cfg(copy.deepcopy(original), copied)
    assert cfg.morphology_conditioning_metadata["parameters_path"] == str(resolved_path)
    resolved.pop(GEOMETRY_NAMES[0])
    resolved_path.write_text(yaml.safe_dump(resolved))
    with pytest.raises(ValueError, match="all seven"):
        condition_quadruped_cfg(copy.deepcopy(original), copied)
    resolved[GEOMETRY_NAMES[0]] = 9.
    resolved_path.write_text(yaml.safe_dump(resolved))
    with pytest.raises(ValueError, match="bounds"):
        condition_quadruped_cfg(copy.deepcopy(original), copied)


def test_morphology_option_must_be_boolean(native_cfg):
    _, xml = native_cfg
    with pytest.raises(ValueError, match="boolean"):
        task.make_env_cfg(robot_xml=xml, morphology_conditioning="true")


def test_native_conditioning_keeps_hardware_profile_in_checkpoint_metadata(native_cfg):
    _, xml = native_cfg
    cfg = task.make_env_cfg(robot_xml=xml, n_envs=3, morphology_conditioning=True,
                            hardware_aware=True)
    metadata = cfg.morphology_conditioning_metadata
    assert metadata["hardware_aware"] is True
    assert metadata["nominal_stance_heights"] == list(cfg.hardware_nominal_heights)
    assert metadata["sampling_mode"] == "fixed" and metadata["replicas"] == 3
    assert cfg.rewards["base_height"].params["nominal_heights"] == list(cfg.hardware_nominal_heights) * 3
