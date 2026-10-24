"""Original-task cohort parity, shared terrain and independent native dynamics."""
import copy
import math
import random
from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from mjlab.scene import Scene
from draft.codesign.batch import model_field_batches
from draft.codesign.space import DesignSpace
from draft.generation.generator import RobotGenerator
from draft.robots import robot_dir
from draft.tasks.quadruped.base_env import make_base_env_cfg
from draft.tasks.quadruped.codesign.native_env import (
    augment_native_cohort_cfg, balance_cohort_terrain, compile_cohort_models,
    make_native_env, make_native_env_cfg,
)

pytestmark = pytest.mark.needs_rl


@pytest.fixture(scope="module")
def native_designs(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-conditioned-cohort")
    space = DesignSpace.quadruped()
    physical = np.asarray([space.center, space.center])
    physical[1, 0] *= 1.08
    physical[1, 1] *= 0.96
    physical[1, 2] *= 1.06
    physical[1, 5] *= 1.04
    records = []
    for i, row in enumerate(physical):
        directory = root / str(i)
        RobotGenerator(robot_dir("quadruped")).generate(directory, params_override=space.overrides(row))
        records.append(SimpleNamespace(design_id=f"native-{i}", xml_path=directory / "quadruped.xml",
                                       parameters=yaml.safe_load((directory / "parameters_resolved.yaml").read_text()),
                                       physical=tuple(row), morphology=tuple(space.normalize(row)),
                                       leg_length=float(row[0] + row[1])))
    return records


def _structure(value):
    if isinstance(value, dict):
        return {key: _structure(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_structure(item) for item in value)
    if callable(value):
        return value.__module__, value.__qualname__
    return value


def test_fixed_cohort_preserves_full_original_task(native_designs):
    xml = native_designs[0].xml_path
    original, _ = make_base_env_cfg(robot_xml=xml, n_envs=4, episode_length_s=20.0)
    fixed = make_native_env_cfg(native_designs[:1], 4, xml, episode_length_s=20.0)
    assert fixed.auto_reset == original.auto_reset and fixed.is_finite_horizon == original.is_finite_horizon
    assert fixed.episode_length_s == original.episode_length_s
    for group in ("actor", "critic"):
        assert "remaining_time" not in fixed.observations[group].terms
        fixed.observations[group].terms.pop("morphology")
    assert _structure(asdict(fixed)) == _structure(asdict(original))
    original_robot = original.scene.entities["robot"].build().spec.compile()
    fixed_robot = fixed.scene.entities["robot"].build().spec.compile()
    for field in ("body_pos", "body_mass", "body_inertia", "site_pos", "geom_size", "actuator_trnid"):
        np.testing.assert_array_equal(getattr(original_robot, field), getattr(fixed_robot, field))


def test_native_cohort_explicit_step_horizon_sets_task_timeout(native_designs):
    from mjlab.envs import mdp
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    xml = native_designs[0].xml_path
    original = make_native_env_cfg(native_designs[:1], 1, xml, episode_length_s=20.0)
    scaled = make_native_env_cfg(native_designs[:1], 1, xml, episode_length_s=25.6)
    extended = make_native_env_cfg(native_designs[:1], 1, xml,
                                   episode_length_s=25.6, horizon_steps=1280)
    step_dt = extended.sim.mujoco.timestep * extended.decimation
    assert math.ceil(original.episode_length_s / step_dt) == 1045
    assert math.ceil(scaled.episode_length_s / step_dt) > 1280
    assert extended.episode_length_s == pytest.approx(25.6)
    assert math.ceil(extended.episode_length_s / step_dt) == 1280
    native_length = ManagerBasedRlEnv.max_episode_length.fget(SimpleNamespace(
        max_episode_length_s=extended.episode_length_s, step_dt=step_dt))
    assert native_length == 1280
    torch.testing.assert_close(mdp.time_out(SimpleNamespace(
        episode_length_buf=torch.tensor([1279, 1280]), max_episode_length=native_length)),
        torch.tensor([False, True]))
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="horizon_steps"):
            make_native_env_cfg(native_designs[:1], 1, xml, horizon_steps=invalid)


def test_hardware_aware_cohort_uses_per_design_posture_and_keeps_common_demands(native_designs):
    from draft.tasks.quadruped.hardware_mdp import TerrainRelativeHeightReward, TerrainRelativeCollapse
    xml = native_designs[0].xml_path
    original, _ = make_base_env_cfg(robot_xml=xml, n_envs=4)
    cfg = make_native_env_cfg(native_designs, 2, xml, hardware_aware=True)
    heights = np.asarray(cfg.hardware_nominal_heights)
    assert len(heights) == 2 and not np.isclose(heights[0], heights[1])
    expected_masses = tuple(float(model.body_mass[1:].sum())
                            for model in cfg.native_cohort.cohort.robot_models)
    assert cfg.hardware_masses_kg == expected_masses
    assert all(math.isfinite(mass) and mass > 0 for mass in expected_masses)
    assert cfg.morphology_conditioning_metadata["hardware_masses_kg"] == list(expected_masses)
    assert cfg.rewards["base_height"].func is TerrainRelativeHeightReward
    assert cfg.terminations["base_too_low"].func is TerrainRelativeCollapse
    assert "torso_contact" in cfg.terminations
    assert "torso_ground_contact" in {sensor.name for sensor in cfg.scene.sensors}
    for term in (cfg.rewards["base_height"], cfg.terminations["base_too_low"]):
        np.testing.assert_allclose(term.params["nominal_heights"], np.repeat(heights, 2))
    for entity, height in zip(cfg.native_cohort.entities, heights):
        assert entity.init_state.pos[2] == height
    assert cfg.morphology_conditioning_metadata["hardware_aware"] is True
    for name in ("commands", "terrain", "events", "curriculum"):
        before = original.scene.terrain if name == "terrain" else getattr(original, name)
        after = cfg.scene.terrain if name == "terrain" else getattr(cfg, name)
        assert _structure(asdict(before) if name == "terrain" else before) == _structure(
            asdict(after) if name == "terrain" else after)
    assert cfg.episode_length_s == original.episode_length_s
    assert _structure(asdict(cfg.rewards["track_linear_velocity"])) == _structure(
        asdict(original.rewards["track_linear_velocity"]))
    # Added ground-contact sensing leaves the pretrained controller layout intact.
    for group in ("actor", "critic"):
        assert set(cfg.observations[group].terms) == set(original.observations[group].terms) | {"morphology"}


def test_hardware_aware_fixed_is_the_single_design_cohort_case(native_designs):
    from draft.tasks.quadruped.hardware import set_hardware_heights
    xml = native_designs[0].xml_path
    standalone, _ = make_base_env_cfg(robot_xml=xml, n_envs=4, hardware_aware=True)
    cfg = make_native_env_cfg(native_designs[:1], 4, xml, hardware_aware=True)
    assert cfg.hardware_nominal_heights == standalone.hardware_nominal_heights
    assert np.isscalar(standalone.rewards["base_height"].params["nominal_heights"])
    set_hardware_heights(standalone, standalone.hardware_nominal_heights, 4)
    for name in ("rewards", "terminations", "events", "commands"):
        assert _structure(getattr(cfg, name)) == _structure(getattr(standalone, name))
    assert cfg.rewards["base_height"].params["nominal_heights"] == list(cfg.hardware_nominal_heights) * 4
    base_model = standalone.scene.entities["robot"].build().spec.compile()
    fixed_model = cfg.scene.entities["robot"].build().spec.compile()
    for field in ("geom_size", "geom_pos", "geom_contype", "geom_conaffinity", "body_mass", "body_inertia"):
        np.testing.assert_array_equal(getattr(base_model, field), getattr(fixed_model, field))


def test_standalone_hardware_environment_validates_compiled_terrain(monkeypatch):
    from draft.tasks.quadruped.codesign import native_env
    validated = []
    monkeypatch.setattr(native_env, "_validate_hardware_terrain",
                        lambda cfg, models: validated.append((cfg, models)))
    monkeypatch.setattr(native_env.ManagerBasedRlEnv, "load_managers", lambda self: "loaded")
    cfg, model = SimpleNamespace(hardware_aware=True), object()
    env = object.__new__(native_env.HardwareAwareEnv)
    env.cfg, env.sim = cfg, SimpleNamespace(mj_model=model)
    assert native_env.HardwareAwareEnv.load_managers(env) == "loaded"
    assert validated == [(cfg, (model,))]
    monkeypatch.setattr(native_env, "HardwareAwareEnv", lambda *args, **kwargs: "guarded")
    assert make_native_env(cfg, device="cpu") == "guarded"


@pytest.mark.parametrize("options", [dict(hardware_aware="yes"),
                                     dict(hardware_aware=True, terrain_scan_fix="none")])
def test_hardware_aware_rejects_ambiguous_configuration(native_designs, options):
    with pytest.raises(ValueError, match="hardware_aware"):
        make_base_env_cfg(robot_xml=native_designs[0].xml_path, **options)


def test_convenience_factory_keeps_video_options_out_of_native_task_kwargs(native_designs, monkeypatch):
    from draft.tasks.quadruped.codesign import native_env
    monkeypatch.setattr(native_env, "make_native_env", lambda cfg, **kwargs: (cfg, kwargs))
    cfg, kwargs = native_env.build_native_env(
        native_designs[:1], 2, reference_xml=native_designs[0].xml_path, device="cpu",
        env_kwargs={"render_mode": "rgb_array", "render_width": 320, "render_height": 240})
    assert cfg.viewer.width == 320 and cfg.viewer.height == 240
    assert cfg.viewer.max_extra_envs == 0 and kwargs["render_mode"] == "rgb_array"


def test_prebuilt_task_augmentation_and_factory_preserve_extensions(native_designs):
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.curriculum_manager import CurriculumTermCfg
    xml = native_designs[0].xml_path
    cfg, _ = make_base_env_cfg(robot_xml=xml, n_envs=4)
    cfg.commands["twist"].ranges.lin_vel_x = (1.5, 2.5)
    reward = RewardTermCfg(func=lambda env: torch.ones(env.num_envs), weight=7.0)
    curriculum = CurriculumTermCfg(func=lambda env, env_ids: {})
    cfg.rewards["future_task_reward"] = reward
    cfg.curriculum["future_task_curriculum"] = curriculum
    cfg.scene.entities["robot"].init_state.joint_pos = {".*": 0.123}
    cfg.scene.entities["robot"].collisions = ()
    called = []
    cfg.scene.spec_fn = lambda spec: called.append(spec)
    original = {name: getattr(cfg, name) for name in (
        "commands", "rewards", "curriculum", "events", "terminations", "sim")}
    result = augment_native_cohort_cfg(cfg, native_designs, 2, xml)
    assert result is cfg
    assert all(getattr(result, name) is value for name, value in original.items())
    assert result.rewards["future_task_reward"] is reward
    assert result.curriculum["future_task_curriculum"] is curriculum
    assert result.commands["twist"].ranges.lin_vel_x == (1.5, 2.5)
    for entity in result.native_cohort.entities:
        assert entity.init_state.joint_pos == {".*": 0.123} and entity.collisions == ()
    result.scene.spec_fn(mujoco.MjSpec())
    assert len(called) == 1
    calls = []

    def extension_factory(**kwargs):
        calls.append(kwargs)
        authored, _ = make_base_env_cfg(**kwargs)
        authored.rewards["future_task_reward"] = reward
        authored.curriculum["future_task_curriculum"] = curriculum
        return authored

    injected = make_native_env_cfg(native_designs[:1], 4, xml, env_cfg_factory=extension_factory)
    assert len(calls) == 1 and calls[0]["task_geometry_xml"] == xml.resolve()
    assert injected.rewards["future_task_reward"] is reward
    assert injected.curriculum["future_task_curriculum"] is curriculum
    explicit = make_native_env_cfg(native_designs[:1], 4, xml,
                                   task_geometry_xml=native_designs[1].xml_path,
                                   env_cfg_factory=extension_factory)
    assert calls[-1]["task_geometry_xml"] == native_designs[1].xml_path.resolve()
    assert explicit.native_cohort.reference_xml == native_designs[1].xml_path.resolve()


def test_native_terrain_balanced_per_design_preserves_global_assignments_and_rng():
    from mjlab.terrains.terrain_entity import TerrainEntity
    designs, replicas = 16, 256
    terrain = object.__new__(TerrainEntity)
    terrain.cfg = SimpleNamespace(max_init_terrain_level=3)
    terrain._device = "cpu"
    terrain.terrain_origins = torch.arange(4 * 5 * 3).reshape(4, 5, 3).float()
    terrain.env_origins = terrain._compute_env_origins_curriculum(
        designs * replicas, terrain.terrain_origins,
        proportions=np.asarray([.1, .15, .2, .25, .3]))
    scene = object.__new__(Scene)
    scene._terrain = terrain
    assert scene.env_origins is terrain.env_origins
    original = tuple(value.clone() for value in (
        terrain.terrain_types, terrain.terrain_levels, terrain.env_origins))
    pointers = tuple(value.data_ptr() for value in (
        terrain.terrain_types, terrain.terrain_levels, terrain.env_origins))
    rng = torch.get_rng_state().clone()
    assert balance_cohort_terrain(scene, designs, replicas)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    counts = torch.stack([torch.bincount(row, minlength=5) for row in
                          terrain.terrain_types.reshape(designs, replicas)])
    # Native proportions remain weighted; each terrain type's allocation is
    # balanced across designs to within one replica, rather than made uniform.
    assert ((counts.max(dim=0).values - counts.min(dim=0).values) <= 1).all()
    permutation = torch.arange(designs * replicas).reshape(replicas, designs).T.reshape(-1)
    for actual, before, pointer in zip((terrain.terrain_types, terrain.terrain_levels,
                                        terrain.env_origins), original, pointers):
        torch.testing.assert_close(actual, before[permutation])
        assert actual.data_ptr() == pointer
    np.testing.assert_array_equal(np.sort(terrain.terrain_types.numpy()), np.sort(original[0].numpy()))
    np.testing.assert_array_equal(np.sort(terrain.terrain_levels.numpy()), np.sort(original[1].numpy()))
    torch.testing.assert_close(terrain.env_origins,
                               terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
    fixed = tuple(value.clone() for value in (
        terrain.terrain_types, terrain.terrain_levels, terrain.env_origins))
    assert not balance_cohort_terrain(scene, 1, designs * replicas)
    for actual, before in zip((terrain.terrain_types, terrain.terrain_levels, terrain.env_origins), fixed):
        torch.testing.assert_close(actual, before)
    torch.testing.assert_close(torch.get_rng_state(), rng)


def _small_terrain(cfg):
    # Limit test compilation cost while retaining the original terrain/sensors/
    # DR path and native generator semantics. Production cfg is not altered.
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.terrain_generator.sub_terrains = {
        "flat": cfg.scene.terrain.terrain_generator.sub_terrains["flat"]}
    cfg.scene.terrain.max_init_terrain_level = 0


@pytest.mark.parametrize("hardware_aware", [False, True])
def test_cpu_reference_compiles_reuse_one_terrain_without_rng_consumption(
        native_designs, hardware_aware, monkeypatch):
    cfg = make_native_env_cfg(native_designs, 2, native_designs[0].xml_path,
                              cpu_compile_workers=1, hardware_aware=hardware_aware)
    _small_terrain(cfg)
    scene = Scene(cfg.scene, device="cpu")
    def regenerate(*args, **kwargs):
        pytest.fail("CPU references must reuse the realized terrain")
    monkeypatch.setattr("mjlab.scene.scene.TerrainEntity", regenerate)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    models = compile_cohort_models(scene, cfg, cfg.native_cohort)
    cfg.cpu_compile_workers = 4
    parallel = compile_cohort_models(scene, cfg, cfg.native_cohort)
    for serial_model, parallel_model in zip(models, parallel):
        # Compare every exposed native array, including topology, sensors,
        # options, terrain and the common mesh bank, in design order.
        for name in dir(serial_model):
            before = getattr(serial_model, name)
            # Unused entries of MuJoCo's name hash table are uninitialized.
            if isinstance(before, np.ndarray) and name != "names_map":
                np.testing.assert_array_equal(before, getattr(parallel_model, name), err_msg=name)
        for kind, count in (("body", serial_model.nbody), ("joint", serial_model.njnt),
                            ("geom", serial_model.ngeom), ("site", serial_model.nsite),
                            ("sensor", serial_model.nsensor), ("mesh", serial_model.nmesh),
                            ("hfield", serial_model.nhfield), ("actuator", serial_model.nu)):
            for index in range(count):
                name = getattr(serial_model, kind)(index).name
                if name:
                    assert getattr(parallel_model, kind)(name).id == index
    torch.testing.assert_close(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    current = np.random.get_state()
    assert numpy_rng[0] == current[0] and numpy_rng[2:] == current[2:]
    np.testing.assert_array_equal(numpy_rng[1], current[1])
    batches = model_field_batches(models)
    assert not np.array_equal(batches["body_pos"][0], batches["body_pos"][1])
    assert not np.array_equal(batches["site_pos"][0], batches["site_pos"][1])
    assert not np.array_equal(batches["body_mass"][0], batches["body_mass"][1])
    for field in ("hfield_data", "hfield_size", "mesh_vert", "mesh_graph"):
        np.testing.assert_array_equal(getattr(models[0], field), getattr(models[1], field))
    assert {"terrain_scan", "foot_height_scan", "feet_ground_contact", "upper_leg_ground_contact"} <= {
        sensor.name for sensor in cfg.scene.sensors}
    assert {"foot_friction", "base_com", "push_robot"} <= cfg.events.keys()


def test_hardware_aware_cpu_cohort_proxies_keep_shared_native_topology(native_designs):
    from draft.tasks.quadruped.hardware_collision import PROXY_GEOM_NAMES
    cfg = make_native_env_cfg(native_designs, 2, native_designs[0].xml_path, hardware_aware=True)
    _small_terrain(cfg)
    scene = Scene(cfg.scene, device="cpu")
    models = compile_cohort_models(scene, cfg, cfg.native_cohort)
    batches = model_field_batches(models)
    for model in models:
        for name in PROXY_GEOM_NAMES:
            geom = model.geom("robot/" + name)
            assert int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_BOX)
            assert int(geom.contype[0]) == 0 and int(geom.conaffinity[0]) == 1
        bank = [model.geom(i) for i in range(model.ngeom) if "bank_cohort_" in model.geom(i).name]
        assert bank and all(int(geom.contype[0]) == int(geom.conaffinity[0]) == 2 for geom in bank)
    proxy_id = models[0].geom("robot/" + PROXY_GEOM_NAMES[0]).id
    assert not np.array_equal(batches["geom_size"][0, proxy_id], batches["geom_size"][1, proxy_id])


def test_hardware_aware_native_gpu_rollout_preserves_conditioned_layout(native_designs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for hardware-aware native rollout")
    cfg = make_native_env_cfg(native_designs, 2, native_designs[0].xml_path, hardware_aware=True)
    _small_terrain(cfg)
    cfg.seed = 19
    env = make_native_env(cfg, device="cuda:0")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (4, 242) and obs["critic"].shape == (4, 266)
        expected = torch.tensor([d.morphology for d in native_designs], device="cuda:0").repeat_interleave(2, 0).float()
        height_cfg = env.reward_manager.get_term_cfg("base_height")
        collapse_cfg = env.termination_manager.get_term_cfg("base_too_low")
        assert (height_cfg.func(env, **height_cfg.params) > .5).all()
        assert not collapse_cfg.func(env, **collapse_cfg.params).any()
        for _ in range(8):
            obs, reward, terminated, timeout, _ = env.step(torch.zeros(4, 12, device="cuda:0"))
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
            assert terminated.shape == timeout.shape == (4,)
            assert not terminated.any()
        torch.testing.assert_close(obs["actor"][:, -7:], expected)
        term = env.termination_manager.get_term_cfg("base_too_low").func
        torch.testing.assert_close(term.nominal_heights,
                                   torch.tensor(cfg.hardware_nominal_heights, device="cuda:0").repeat_interleave(2))
        # Exercise the actual Warp box/terrain contact and native sensor path.
        env.sim.data.qpos[0, 2] = .07
        env.sim.data.qpos[0, 3:7] = torch.tensor([0., 1., 0., 0.], device="cuda:0")
        env.sim.data.qvel[0] = 0.
        env.sim.forward()
        env.step(torch.zeros(4, 12, device="cuda:0"))
        assert env.termination_manager.get_term("torso_contact")[0]
    finally:
        env.close()


def test_native_two_design_cpu_gpu_parity_and_startup_dr(native_designs):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for native heterogeneous independent CPU/GPU parity")
    cfg = make_native_env_cfg(native_designs, 2, native_designs[0].xml_path)
    _small_terrain(cfg)
    cfg.seed = 12
    env = make_native_env(cfg, device="cuda:0")
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        for group in ("actor", "critic"):
            expected = torch.tensor([d.morphology for d in native_designs], device="cuda:0").repeat_interleave(2, 0).float()
            torch.testing.assert_close(obs[group][:, -7:], expected)
        # Startup DR uses each world's own COM baseline. Other physical fields
        # remain the generated geometry/inertia despite EventManager expansion.
        for i, model in enumerate(env.cpu_models):
            row = i * 2
            for field in ("body_pos", "site_pos", "body_mass", "body_inertia"):
                actual = getattr(env.sim.model, field)[row].cpu().numpy()
                np.testing.assert_allclose(actual, getattr(model, field).reshape(actual.shape), atol=1e-6)
        env.sim.forward()
        independent = []
        # Synchronize every native per-world physical field (including startup
        # friction/COM DR) into each independently compiled C MuJoCo reference.
        for i, source in enumerate(env.cpu_models):
            row = i * 2
            model = copy.deepcopy(source)
            for field in model_field_batches(env.cpu_models):
                target = getattr(model, field)
                target[:] = getattr(env.sim.model, field)[row].cpu().numpy().reshape(target.shape)
            data = mujoco.MjData(model)
            data.qpos[:] = env.sim.data.qpos[row].cpu().numpy()
            data.qvel[:] = env.sim.data.qvel[row].cpu().numpy()
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(env.sim.data.site_xpos[row].cpu(), data.site_xpos, atol=2e-5, rtol=2e-5)
            data.ctrl[:] = .2
            mujoco.mj_step(model, data)
            independent.append(data)
        env.sim.data.ctrl[:] = .2
        env.sim.step()
        for i, data in enumerate(independent):
            np.testing.assert_allclose(env.sim.data.qpos[i * 2].cpu(), data.qpos, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(env.sim.data.qvel[i * 2].cpu(), data.qvel, atol=5e-4, rtol=1e-3)
        torch.testing.assert_close(env.action_manager.get_term("joint_pos").offset,
                                   robot.data.default_joint_pos)
        _, reward, _, _, _ = env.step(torch.zeros((4, 12), device="cuda:0"))
        assert torch.isfinite(reward).all()
    finally:
        env.close()
