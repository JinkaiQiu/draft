"""Actual CPU ground contacts and invariant inertias for hardware-aware proxies."""
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from draft.codesign.batch import prepare_cohort
from draft.codesign.space import DesignSpace
from draft.generation.generator import RobotGenerator
from draft.robots import robot_dir
from draft.tasks.quadruped.hardware_collision import (
    FOOT_GEOM_NAMES, PROXY_BODIES, PROXY_GEOM_NAMES, PROXY_GEOM_PATTERN,
    TORSO_PROXY_NAME, PROXY_PREFIX, MAX_PROXY_DIAGONAL, _body_bounds,
    add_hardware_collision_proxies, collision_subdivisions, configure_hardware_collisions,
    validate_hfield_collision_budget,
)


@pytest.fixture(scope="module")
def hardware_designs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hardware-collision")
    space = DesignSpace.quadruped()
    physical = np.asarray([space.center, space.center])
    physical[1] *= np.asarray([1.10, .90, .90, 1.10, .95, .92, 1.10])
    designs = []
    for index, row in enumerate(physical):
        path = directory / str(index)
        RobotGenerator(robot_dir("quadruped")).generate(path, params_override=space.overrides(row))
        designs.append(SimpleNamespace(design_id=str(index), xml_path=path / "quadruped.xml",
                                       morphology=tuple(space.normalize(row)), physical=tuple(row),
                                       leg_length=float(row[0] + row[1])))
    return designs


@pytest.fixture(scope="module")
def proxy_layout(hardware_designs):
    return collision_subdivisions([design.xml_path for design in hardware_designs])


def _proxy_ids(model):
    return np.asarray([i for i in range(model.ngeom) if model.geom(i).name.startswith(PROXY_PREFIX)])


def _ground_spec(path, aware, terrain, subdivisions=None):
    spec = mujoco.MjSpec.from_file(str(path))
    if aware:
        add_hardware_collision_proxies(spec, subdivisions=subdivisions)
    else:
        # Original native task: four foot spheres only.
        for geom in spec.geoms:
            geom.contype = geom.conaffinity = int(geom.name in FOOT_GEOM_NAMES)
    ground = spec.worldbody.add_body(name="terrain")
    if terrain == "plane":
        ground.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=(2, 2, .1), contype=1, conaffinity=1)
    else:
        spec.add_hfield(name="flat_ground", nrow=41, ncol=41, size=(2, 2, .1, .1),
                        userdata=[0.] * (41 * 41))
        ground.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                        hfieldname="flat_ground", contype=1, conaffinity=1)
    for name, body in (("torso_contact", "torso_link"), ("upper_contact", "fr_upper_leg_link")):
        spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_CONTACT,
                        objtype=mujoco.mjtObj.mjOBJ_BODY, objname=body,
                        reftype=mujoco.mjtObj.mjOBJ_BODY, refname="terrain", intprm=[1, 0, 1])
    return spec


def test_proxies_cover_rotated_torso_and_preserve_generated_physics(hardware_designs, proxy_layout):
    for design in hardware_designs:
        spec = mujoco.MjSpec.from_file(str(design.xml_path))
        original = spec.compile()
        add_hardware_collision_proxies(spec, subdivisions=proxy_layout)
        actual = spec.compile()
        ids = _proxy_ids(actual)
        assert actual.ngeom == original.ngeom + len(ids)
        assert len(ids) == sum(np.prod(counts) for counts in proxy_layout.values())
        assert 21 < len(ids) < 100
        assert len(PROXY_BODIES) == len(PROXY_GEOM_NAMES) == 21
        assert sum((actual.geom_contype != 0) | (actual.geom_conaffinity != 0)) == len(ids) + 4
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_armature"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))
        for name in FOOT_GEOM_NAMES:
            old, new = original.geom(name).id, actual.geom(name).id
            for field in ("geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp"):
                np.testing.assert_array_equal(getattr(actual, field)[new], getattr(original, field)[old])
        for gid in ids:
            name = actual.geom(gid).name
            assert actual.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
            assert actual.geom_contype[gid] == 0 and actual.geom_conaffinity[gid] == 1
            assert spec.geom(name).mass == 0 and spec.geom(name).density == 0
        # The mesh's compiled frame is rotated: using geom_size directly would
        # swap trunk length and height. The body-local envelope must not do so.
        for body in PROXY_BODIES:
            center, half, _ = _body_bounds(original, original.body(body).id)
            body_ids = ids[actual.geom_bodyid[ids] == actual.body(body).id]
            positions, sizes = actual.geom_pos[body_ids], actual.geom_size[body_ids]
            np.testing.assert_allclose(np.min(positions - sizes, axis=0), center - half, atol=1e-12)
            np.testing.assert_allclose(np.max(positions + sizes, axis=0), center + half, atol=1e-12)
            # Equal union volume, exact grid positions and no overlap/gaps.
            np.testing.assert_allclose(np.sum(np.prod(2 * sizes, axis=1)), np.prod(2 * half))
            assert len(np.unique(positions, axis=0)) == len(body_ids)
            np.testing.assert_allclose(sizes, np.broadcast_to(half / proxy_layout[body], sizes.shape))
        _, torso_half, _ = _body_bounds(original, original.body("torso_link").id)
        assert torso_half[0] > torso_half[1] > torso_half[2]
        assert np.max(2 * np.linalg.norm(actual.geom_size[ids], axis=1)
                      + 2 * actual.geom_margin[ids]) <= MAX_PROXY_DIAGONAL + 1e-12
        with pytest.raises(ValueError, match="already present"):
            add_hardware_collision_proxies(spec)


@pytest.mark.parametrize("terrain", ["plane", "hfield"])
def test_actual_torso_fall_hits_ground_and_upper_leg_sensor_fires(hardware_designs, proxy_layout, terrain):
    models = [_ground_spec(hardware_designs[0].xml_path, aware, terrain, proxy_layout).compile() for aware in (False, True)]
    fallen_heights = []
    for aware, model in zip((False, True), models):
        model.opt.timestep = .002
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[2] = .25
        data.qpos[3:7] = (0, 1, 0, 0)  # Actual upside-down fall, feet above trunk.
        torso_hits = 0
        max_contacts = 0
        for _ in range(120):
            mujoco.mj_step(model, data)
            torso_hits += int(data.sensor("torso_contact").data[0] > 0)
            pair_counts = Counter(tuple(contact.geom) for contact in data.contact)
            max_contacts = max(max_contacts, max(pair_counts.values(), default=0))
        fallen_heights.append(data.qpos[2])
        assert (torso_hits > 0) == aware
        assert max_contacts <= 32  # Per-pair prism bound, rather than total scene contacts.
        if aware:
            assert data.qpos[2] > .05
        # Roll sideways: the torso just clears the plane while the right upper
        # leg lies on it. This uses the same body matching as the native sensor.
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[2] = .15
        data.qpos[3:7] = (2**-.5, 2**-.5, 0, 0)
        mujoco.mj_forward(model, data)
        assert (data.sensor("upper_contact").data[0] > 0) == aware
        assert data.sensor("torso_contact").data[0] == 0
    assert fallen_heights[1] > fallen_heights[0] + .05


@pytest.mark.needs_rl
def test_entity_wrapper_is_opt_in_and_rewraps_replaced_cohort_factory(hardware_designs):
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.quadruped_entity import make_quadruped_entity_cfg
    entity = make_quadruped_entity_cfg(hardware_designs[0].xml_path)
    original = entity.build().spec.compile()
    original_spec = entity.spec_fn()
    assert not any(geom.name in PROXY_GEOM_NAMES for geom in original_spec.geoms)
    configure_hardware_collisions(entity)
    wrapped = entity.spec_fn
    configure_hardware_collisions(entity)
    assert entity.spec_fn is wrapped
    actual = entity.build().spec.compile()
    assert actual.ngeom == original.ngeom + len(_proxy_ids(actual))
    assert entity.collisions[0].geom_names_expr[-1] == PROXY_GEOM_PATTERN
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))
    # Cohort augmentation replaces a copied template's spec factory. A previous
    # entity-level flag must not prevent the replacement factory being wrapped.
    entity.spec_fn = lambda: mujoco.MjSpec.from_file(str(hardware_designs[1].xml_path))
    configure_hardware_collisions(entity)
    replacement = entity.spec_fn()
    assert sum(geom.name in PROXY_GEOM_NAMES for geom in replacement.geoms) == 21


@pytest.mark.needs_rl
def test_cohort_proxy_topology_and_native_varying_fields_preserve_bank_isolation(hardware_designs, proxy_layout):
    pytest.importorskip("mujoco_warp")
    from draft.codesign.batch import model_field_batches
    cohort = prepare_cohort(hardware_designs, 2)
    models = []
    for i in range(2):
        spec = add_hardware_collision_proxies(cohort.robot_spec(i), subdivisions=proxy_layout)
        model = spec.compile()
        for geom in spec.geoms:
            if geom.name.startswith("bank_"):
                assert geom.contype == geom.conaffinity == 2
                # No collision-mask intersection with feet or physical proxies.
                assert (geom.contype & 1) == 0
        models.append(model)
    assert [(models[0].geom(name).id, int(models[0].geom_type[models[0].geom(name).id])) for name in PROXY_GEOM_NAMES] == [
        (models[1].geom(name).id, int(models[1].geom_type[models[1].geom(name).id])) for name in PROXY_GEOM_NAMES]
    batches = model_field_batches(models)
    names = [[model.geom(i).name for i in _proxy_ids(model)] for model in models]
    assert names[0] == names[1]
    ids = _proxy_ids(models[0])
    assert not np.array_equal(batches["geom_size"][0, ids], batches["geom_size"][1, ids])
    assert not np.array_equal(batches["geom_pos"][0, ids], batches["geom_pos"][1, ids])
    np.testing.assert_array_equal(models[0].mesh_graph, models[1].mesh_graph)


def test_shared_layout_validation_and_safe_reconfiguration(hardware_designs, proxy_layout):
    with pytest.raises(ValueError, match="At least one"):
        collision_subdivisions([])
    for diagonal in (0, float("nan"), .3):
        with pytest.raises(ValueError, match="max_diagonal"):
            collision_subdivisions([hardware_designs[0].xml_path], max_diagonal=diagonal)
    spec = mujoco.MjSpec.from_file(str(hardware_designs[0].xml_path))
    with pytest.raises(ValueError, match="every generated"):
        add_hardware_collision_proxies(spec, subdivisions={})
    bad = dict.fromkeys(PROXY_BODIES, (1, 1, 1))
    with pytest.raises(ValueError, match="safe proxy diagonal"):
        add_hardware_collision_proxies(spec, subdivisions=bad)
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.quadruped_entity import make_quadruped_entity_cfg
    entity = make_quadruped_entity_cfg(hardware_designs[0].xml_path)
    configure_hardware_collisions(entity)
    configure_hardware_collisions(entity, subdivisions=proxy_layout)
    wrapped = entity.spec_fn
    configure_hardware_collisions(entity, subdivisions=dict(proxy_layout))
    assert entity.spec_fn is wrapped
    model = entity.build().spec.compile()
    assert len(_proxy_ids(model)) == sum(np.prod(counts) for counts in proxy_layout.values())


@pytest.mark.needs_rl
def test_prepared_source_models_reuse_pose_and_proxy_geometry(hardware_designs, monkeypatch):
    """Source-model reuse skips recompiles and survives cohort geom reindexing."""
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.hardware import prepare_hardware_entity
    from draft.tasks.quadruped.quadruped_entity import (
        _stance_height, default_stance, make_quadruped_entity_cfg,
    )

    cohort = prepare_cohort(hardware_designs, 2)
    paths = [design.xml_path for design in hardware_designs]
    expected_layout = collision_subdivisions(paths)
    expected_pose = default_stance(paths[1], paths[0])
    source = cohort.robot_models[1]
    original_mass = source.body_mass.copy()
    original_contacts = source.geom_conaffinity.copy()
    expected_height = _stance_height(source, mujoco.MjData(source), -expected_pose[".*knee"])
    expected = add_hardware_collision_proxies(
        cohort.robot_spec(1), subdivisions=expected_layout).compile()

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("a matching source model was already compiled")

    compile_calls = []
    original_compile = mujoco.MjSpec.compile

    def count_compile(spec, *args, **kwargs):
        compile_calls.append(spec)
        return original_compile(spec, *args, **kwargs)

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", unexpected_compile)
    monkeypatch.setattr(mujoco.MjSpec, "compile", count_compile)
    layout = collision_subdivisions(paths, models=cohort.robot_models)
    assert layout == expected_layout
    entity = make_quadruped_entity_cfg(paths[1], stance_xml=paths[0], model=source,
                                       stance_model=cohort.robot_models[0])
    assert entity.init_state.joint_pos == expected_pose
    entity.spec_fn = lambda: cohort.robot_spec(1)
    height = prepare_hardware_entity(entity, paths[1], paths[0], collision_subdivisions=layout,
                                     model=source, stance_model=cohort.robot_models[0])
    assert height == pytest.approx(expected_height)
    assert not compile_calls
    actual_spec = entity.spec_fn()
    assert len(compile_calls) == 1  # Required post-addition name-index refresh.
    for name in PROXY_GEOM_NAMES:
        assert actual_spec.geom(name).name == name
    actual = actual_spec.compile()
    for field in ("geom_pos", "geom_size", "geom_conaffinity", "geom_contype", "body_mass",
                  "body_inertia", "geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    np.testing.assert_array_equal(source.body_mass, original_mass)
    np.testing.assert_array_equal(source.geom_conaffinity, original_contacts)
    with pytest.raises(ValueError, match="XML count"):
        collision_subdivisions(paths, models=cohort.robot_models[:1])


@pytest.mark.needs_rl
def test_native_mixed_hfields_tilted_falls_stay_below_warp_pair_limit(hardware_designs, proxy_layout):
    """Use every actual task hfield, including uneven noise/slope/waves.

    Count the subgrid Warp would visit (two prisms per cell), even when all
    prisms collide during deep penetration. This is stronger than observing a
    small final CPU contact manifold after MuJoCo's contact reduction.
    """
    pytest.importorskip("mjlab")
    from draft.tasks.quadruped.base_env import make_base_env_cfg
    cfg, _ = make_base_env_cfg(robot_xml=hardware_designs[0].xml_path, n_envs=2)
    generator = cfg.scene.terrain.terrain_generator
    terrain_types = [deepcopy(st) for st in generator.sub_terrains.values()
                     if hasattr(st, "horizontal_scale")]
    assert len(terrain_types) >= 3
    rng = np.random.default_rng(1729)
    rotations = rng.normal(size=(32, 4))
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    observed_contacts, max_triangles = 0, 0
    for terrain in terrain_types:
        terrain.size = generator.size
        for design in hardware_designs:
            spec = mujoco.MjSpec.from_file(str(design.xml_path))
            add_hardware_collision_proxies(spec, subdivisions=proxy_layout)
            spec.worldbody.add_body(name="terrain")
            spec.compile()  # Refresh MuJoCo's post-compile body name index.
            terrain.function(.9, spec, np.random.default_rng(42))
            model = spec.compile()
            assert model.nhfield == 1
            validate_hfield_collision_budget(model)
            spacing = 2 * model.hfield_size[0, :2] / (np.array([model.hfield_ncol[0], model.hfield_nrow[0]]) - 1)
            assert np.min(spacing) >= .1 - 1e-8
            hfield_id = next(i for i in range(model.ngeom)
                             if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD)
            ids = _proxy_ids(model)
            data = mujoco.MjData(model)
            for index, rotation in enumerate(rotations):
                data.qpos[:] = model.qpos0
                data.qpos[:2] = np.asarray(terrain.size) * (.3 if index % 2 else .7)
                data.qpos[:2] += rng.uniform(-.049, .049, 2)  # Non-grid-aligned poses.
                data.qpos[2] = model.geom_pos[hfield_id, 2] + model.hfield_size[0, 2] * .5
                data.qpos[3:7] = rotation
                mujoco.mj_forward(model, data)
                observed_contacts += data.ncon
                pair_counts = Counter(tuple(contact.geom) for contact in data.contact)
                assert max(pair_counts.values(), default=0) < 50
                for gid in ids:
                    matrix = data.geom_xmat[gid].reshape(3, 3)
                    half = np.abs(matrix) @ model.geom_size[gid]
                    center = data.geom_xpos[gid, :2] - data.geom_xpos[hfield_id, :2]
                    lo = np.floor((center - half[:2] + model.hfield_size[0, :2]) / spacing)
                    hi = np.ceil((center + half[:2] + model.hfield_size[0, :2]) / spacing)
                    triangle_count = int(2 * np.prod(hi - lo))
                    max_triangles = max(max_triangles, triangle_count)
                    assert triangle_count <= 32
            assert np.isfinite(data.qpos).all()
    assert observed_contacts > 0 and max_triangles >= 18


def test_compiled_hfield_budget_rejects_fine_cells_and_includes_margins(hardware_designs, proxy_layout):
    path = hardware_designs[0].xml_path
    validate_hfield_collision_budget(_ground_spec(path, True, "plane", proxy_layout).compile())
    spec = _ground_spec(path, True, "hfield", proxy_layout)
    model = spec.compile()
    validate_hfield_collision_budget(model)
    model.hfield_size[0, :2] = .2  # 0.01m cells, no quiet terrain modification.
    with pytest.raises(ValueError, match="heightfield triangles.*cell pitch.*fewer than 50"):
        validate_hfield_collision_budget(model)
    model = spec.compile()
    model.geom_margin[model.geom("ground").id] = .1
    with pytest.raises(ValueError, match="heightfield triangles"):
        validate_hfield_collision_budget(model)
    # The old full-size torso box is unsafe even at stock 0.1m spacing.
    unsplit = spec.compile()
    gid = unsplit.geom(TORSO_PROXY_NAME).id
    unsplit.geom_rbound[gid] = .3
    with pytest.raises(ValueError, match=TORSO_PROXY_NAME):
        validate_hfield_collision_budget(unsplit)


@pytest.mark.needs_rl
@pytest.mark.slow
def test_native_mixed_hfields_gpu_tilted_contacts_have_no_overflow(hardware_designs, proxy_layout, capfd):
    """Explicit GPU gate for every native hfield; run separately before launch."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable for native mixed-heightfield collision stress")
    wp = pytest.importorskip("warp")
    mw = pytest.importorskip("mujoco_warp")
    from draft.tasks.quadruped.base_env import make_base_env_cfg
    cfg, _ = make_base_env_cfg(robot_xml=hardware_designs[0].xml_path, n_envs=2)
    generator = cfg.scene.terrain.terrain_generator
    nworld = 64
    rng = np.random.default_rng(1729)
    contacts = 0
    with wp.ScopedDevice("cuda:0"):
        for source in generator.sub_terrains.values():
            if not hasattr(source, "horizontal_scale"):
                continue
            terrain = deepcopy(source)
            terrain.size = generator.size
            for design in hardware_designs:
                spec = mujoco.MjSpec.from_file(str(design.xml_path))
                add_hardware_collision_proxies(spec, subdivisions=proxy_layout)
                spec.worldbody.add_body(name="terrain")
                spec.compile()
                terrain.function(.9, spec, np.random.default_rng(42))
                model = spec.compile()
                validate_hfield_collision_budget(model)
                hfield_id = next(i for i in range(model.ngeom)
                                 if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD)
                qpos = np.tile(model.qpos0, (nworld, 1)).astype(np.float32)
                qpos[:, :2] = np.asarray(terrain.size) * .3 + rng.uniform(-.2, .2, (nworld, 2))
                qpos[:, 2] = model.geom_pos[hfield_id, 2] + model.hfield_size[0, 2] * .5
                quaternions = rng.normal(size=(nworld, 4))
                qpos[:, 3:7] = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
                gpu_model = mw.put_model(model)
                gpu_data = mw.make_data(model, nworld=nworld, nconmax=512, nccdmax=128)
                wp.copy(gpu_data.qpos, wp.array(qpos, dtype=wp.float32))
                mw.kinematics(gpu_model, gpu_data)
                mw.collision(gpu_model, gpu_data)
                wp.synchronize()
                count = int(gpu_data.nacon.numpy()[0])
                assert 0 < count < gpu_data.naconmax
                assert np.isfinite(gpu_data.contact.dist.numpy()[:count]).all()
                contacts += count
                output = capfd.readouterr()
                assert "overflow" not in (output.out + output.err).lower()
                del gpu_model, gpu_data
    assert contacts > 0
