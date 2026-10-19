"""Native geometry batching: source-model fidelity and optional GPU parity."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from draft.codesign.batch import prepare_cohort
from draft.generation.generator import RobotGenerator
from draft.robots import robot_dir


@pytest.fixture(scope="module")
def designs(tmp_path_factory):
    root = tmp_path_factory.mktemp("codesign-models")
    result = []
    for i, overrides in enumerate(({}, {"upper_leg_link_length": 0.30,
                                        "lower_leg_link_length": 0.28,
                                        "torso_lx": 0.65, "torso_ly": 0.25,
                                        "torso_lz": 0.18, "shoulder_width": 0.17,
                                        "hip_roll_link_length": 0.14})):
        directory = root / str(i)
        RobotGenerator(robot_dir("quadruped")).generate(directory, params_override=overrides)
        result.append(SimpleNamespace(
            design_id=f"test-{i}", xml_path=directory / "quadruped.xml",
            morphology=tuple([float(i)] * 7), parameters=overrides,
            leg_length=0.44 if i == 0 else 0.58))
    return result


def test_union_mesh_bank_preserves_generated_geometry_and_inertia(designs):
    cohort = prepare_cohort(designs, replicas=3)
    assert cohort.num_envs == 6
    np.testing.assert_array_equal(cohort.design_index, [0, 0, 0, 1, 1, 1])
    models = [cohort.robot_spec(i).compile() for i in range(2)]
    for i, design in enumerate(designs):
        source = mujoco.MjModel.from_xml_path(str(design.xml_path))
        actual = models[i]
        source_geom_ids = [actual.geom(source.geom(i).name).id for i in range(source.ngeom)]
        for name in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos",
                     "body_iquat", "jnt_pos", "jnt_axis", "geom_size", "geom_pos",
                     "geom_quat", "site_pos", "site_quat"):
            value = getattr(actual, name)
            if name.startswith("geom_"):
                value = value[source_geom_ids]
            np.testing.assert_allclose(value, getattr(source, name), atol=1e-10)
        for geom in np.flatnonzero(source.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
            old_id, new_id = source.geom_dataid[geom], actual.geom_dataid[source_geom_ids[geom]]
            old = source.mesh_vert[source.mesh_vertadr[old_id]:
                                   source.mesh_vertadr[old_id] + source.mesh_vertnum[old_id]]
            new = actual.mesh_vert[actual.mesh_vertadr[new_id]:
                                  actual.mesh_vertadr[new_id] + actual.mesh_vertnum[new_id]]
            np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(models[0].mesh_vert, models[1].mesh_vert)
    assert not np.array_equal(models[0].geom_dataid, models[1].geom_dataid)
    assert not np.allclose(models[0].body_mass, models[1].body_mass)
    assert not np.allclose(models[0].site_pos, models[1].site_pos)


@pytest.mark.parametrize("replicas", [0, -1, 1.5, True])
def test_invalid_replica_count(designs, replicas):
    with pytest.raises(ValueError, match="replicas"):
        prepare_cohort(designs, replicas)


def test_invalid_morphology_and_motor_changes_are_rejected(designs, tmp_path):
    invalid = SimpleNamespace(**vars(designs[0]))
    invalid.morphology = (0.0,) * 6
    with pytest.raises(ValueError, match="seven"):
        prepare_cohort([invalid], 1)
    spec = mujoco.MjSpec.from_file(str(designs[0].xml_path))
    spec.joint("fl_knee").actfrcrange = [-1, 1]
    # Export all assets so the motor-only mutation remains independently loadable.
    source = Path(designs[0].xml_path).parent
    invalid.xml_path = tmp_path / "changed.xml"
    spec.compiler.meshdir = str(source / spec.compiler.meshdir)
    invalid.xml_path.write_text(spec.to_xml())
    invalid.morphology = (0.0,) * 7
    with pytest.raises(ValueError, match="same 12 motors"):
        prepare_cohort([designs[0], invalid], 1)


def _rl_available():
    pytest.importorskip("mjlab")
    pytest.importorskip("mujoco_warp")
    return pytest.importorskip("torch")


@pytest.mark.needs_rl
def test_native_field_audit_rejects_unsupported_variation(designs):
    pytest.importorskip("mujoco_warp")
    from draft.codesign.batch import model_field_batches
    cohort = prepare_cohort(designs, 1)
    models = [cohort.robot_spec(i).compile() for i in range(2)]
    batches = model_field_batches(models)
    assert {"body_mass", "body_inertia", "geom_dataid", "geom_aabb", "site_pos"} <= batches.keys()
    models[1].site_size[0, 0] += 0.01
    with pytest.raises(ValueError, match="site_size"):
        model_field_batches(models)


@pytest.mark.needs_rl
def test_common_task_configuration_keeps_fixed_horizon_and_no_heading(designs):
    _rl_available()
    from draft.tasks.quadruped.codesign.env import make_codesign_env_cfg
    cohort = prepare_cohort(designs, 2)
    cfg, _, heights = make_codesign_env_cfg(cohort, episode_length_s=20.0,
                                            env_kwargs={"verbose": True, "render_mode": "rgb_array",
                                                        "render_width": 640, "render_height": 480})
    twist = cfg.commands["twist"]
    assert not twist.heading_command and twist.ranges.heading is None
    assert cfg.episode_length_s == 20.0 and cfg.is_finite_horizon
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.viewer.width == 640 and cfg.viewer.height == 480
    assert cfg.viewer.env_idx == 0 and cfg.viewer.max_extra_envs == 0
    assert heights[0] != heights[1]
    for group in cfg.observations.values():
        assert "morphology" in group.terms and "remaining_time" in group.terms


@pytest.mark.needs_rl
@pytest.mark.parametrize("verbose,fail", [(False, False), (True, False), (False, True)])
def test_cohort_startup_output_and_failure_replay(designs, monkeypatch, capsys, verbose, fail):
    _rl_available()
    import draft.tasks.quadruped.codesign.env as module

    def construct(*args, **kwargs):
        print("native cohort startup")
        if fail:
            raise RuntimeError("startup failure")
        return object()

    monkeypatch.setattr(module, "CoDesignEnv", construct)
    monkeypatch.setattr(module, "CoDesignVecEnv", lambda env, **kwargs: env)
    if fail:
        with pytest.raises(RuntimeError, match="startup failure"):
            module.build_env(designs, 1, device="cpu", env_kwargs={"verbose": verbose})
    else:
        module.build_env(designs, 1, device="cpu", env_kwargs={"verbose": verbose})
    captured = capsys.readouterr()
    assert ("native cohort startup" in captured.out) == verbose
    assert ("native cohort startup" in captured.err) == fail


@pytest.mark.needs_rl
def test_independent_cpu_gpu_kinematics_and_dynamics(designs):
    torch = _rl_available()
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: CPU/GPU native dynamics parity requires a CUDA device")
    from draft.tasks.quadruped.codesign.env import build_env
    env = build_env(designs, 2, device="cuda:0", seed=7,
                    env_kwargs={"enable_corruption": False})
    try:
        raw = env.unwrapped
        obs, _ = env.reset()
        assert obs["actor"].shape[-1] == 56  # native48 + normalized geometry7 + remaining time1
        assert obs["critic"].shape[-1] == 56
        assert raw.horizon_steps == 1000
        np.testing.assert_allclose(raw.scene["robot"].data.default_root_state[:, 2].cpu(),
                                   raw.nominal_height.cpu())
        raw.sim.forward()
        independent = []
        for design in range(2):
            row = design * 2
            model = raw.cpu_models[design]
            data = mujoco.MjData(model)
            data.qpos[:] = raw.sim.data.qpos[row].cpu().numpy()
            data.qvel[:] = raw.sim.data.qvel[row].cpu().numpy()
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(raw.sim.data.site_xpos[row].cpu(), data.site_xpos,
                                       atol=2e-5, rtol=2e-5)
            independent.append((model, data))
        # Advance both independent CPU worlds and the entire GPU batch exactly
        # once from the same snapshots. mj_step kinematics precede integration.
        raw.sim.data.ctrl[:] = 0.2
        for model, data in independent:
            data.ctrl[:] = 0.2
            mujoco.mj_step(model, data)
        raw.sim.step()
        for design, (_, data) in enumerate(independent):
            row = design * 2
            np.testing.assert_allclose(raw.sim.data.qpos[row].cpu(), data.qpos,
                                       atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(raw.sim.data.qvel[row].cpu(), data.qvel,
                                       atol=5e-4, rtol=1e-3)
        for design in range(2):
            rows = slice(design * 2, (design + 1) * 2)
            offsets = raw.action_manager.get_term("joint_pos").offset[rows]
            torch.testing.assert_close(offsets, raw.scene["robot"].data.default_joint_pos[rows])
        _, reward, dones, _ = env.step(torch.zeros((4, 12), device="cuda:0"))
        assert torch.isfinite(reward).all() and dones.shape == (4,)
        # Timeout and actual fall are distinct even with finite-horizon config.
        env.reset()
        raw.episode_length_buf[:] = raw.max_episode_length - 1
        _, _, dones, extras = env.step(torch.zeros((4, 12), device="cuda:0"))
        assert dones.bool().all() and extras["time_outs"].all()
        assert not extras["codesign_terminated"].any()
        env.reset()
        raw.sim.data.qpos[:, 3:7] = torch.tensor([0., 1., 0., 0.], device="cuda:0")
        raw.sim.forward()
        _, _, dones, extras = env.step(torch.zeros((4, 12), device="cuda:0"))
        assert dones.bool().all() and extras["codesign_terminated"].all()
        assert not extras["time_outs"].any()
    finally:
        env.close()
