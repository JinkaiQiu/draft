"""CPU checks of compiled joint locks and motor material replacement."""

import json
import shutil
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from draft.codesign.batch import JOINT_NAMES, prepare_cohort
from draft.codesign.compiler import DesignCompiler
from draft.codesign.space import DesignSpace
from draft.codesign.topology_physics import apply_joint_topology, prepare_exported_topology


def _designs(tmp_path):
    space = DesignSpace.quadruped(joint_topology=True)
    all_active = np.asarray(space.center)
    frozen = all_active.copy()
    frozen[7:13] = [0, 0, 0, 1, 1, 1]
    frozen[13:19] = [.25, -.4, -1.2, 0, 0, 0]
    with DesignCompiler(space, tmp_path) as compiler:
        return space, all_active, frozen, compiler.compile(all_active), compiler.compile(frozen)


def test_topology_material_receipt_mass_and_shared_model(tmp_path):
    _, _, _, stock, frozen = _designs(tmp_path)
    stock_model = mujoco.MjModel.from_xml_path(str(stock.xml_path))
    model = mujoco.MjModel.from_xml_path(str(frozen.xml_path))
    assert (model.nq, model.nv, model.ngeom, model.neq) == (19, 18, stock_model.ngeom, 12)
    np.testing.assert_array_equal(model.eq_active0, 0)
    receipt = yaml.safe_load((frozen.xml_path.parent / "feasibility_report.yaml").read_text())
    rows = receipt["joint_topology"]["joint_materials"]
    assert len(rows) == 12
    assert receipt["joint_topology"]["compiled_mass_kg"] == pytest.approx(model.body_mass.sum())
    assert stock_model.body_mass.sum() == pytest.approx(
        yaml.safe_load((stock.xml_path.parent / "feasibility_report.yaml").read_text())
        ["joint_topology"]["generator_mass_kg"])
    delta = sum(r["effective_mass_kg"] - r["motor_mass_kg"] for r in rows)
    assert model.body_mass.sum() - stock_model.body_mass.sum() == pytest.approx(delta, abs=2e-4)
    for row in rows:
        geom = model.geom(row["motor_geom"])
        original = stock_model.geom(row["motor_geom"])
        np.testing.assert_array_equal(model.geom_size[geom.id], stock_model.geom_size[original.id])
        if row["active"]:
            assert row["effective_mass_kg"] == row["motor_mass_kg"]
        else:
            assert row["effective_density_kg_m3"] != row["motor_density_kg_m3"]
            assert row["effective_mass_kg"] > 0
            dof = model.jnt_dofadr[model.joint(row["joint"]).id]
            assert model.dof_armature[dof] == model.dof_damping[dof] == model.dof_frictionloss[dof] == 0
    # The belt output pulley and visual detail never acquire motor mass.
    assert all(row["motor_geom"] != "fl_upper_leg_link_motor" for row in rows)
    assert "density=\"0\"" in frozen.xml_path.read_text()
    assert prepare_cohort([stock, frozen], 1).num_envs == 2


def test_mirrored_lock_targets_and_constraint_drift(tmp_path):
    _, _, _, _, frozen = _designs(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(frozen.xml_path))
    targets = model.eq_data[:, 0]
    np.testing.assert_allclose(targets[:6], [.25, -.4, -1.2, -.25, -.4, -1.2])
    data = mujoco.MjData(model)
    for i in range(6):
        data.eq_active[i] = 1
        jid = model.eq_obj1id[i]
        data.qpos[model.jnt_qposadr[jid]] = targets[i]
    mujoco.mj_forward(model, data)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    errors = [abs(data.qpos[model.jnt_qposadr[model.eq_obj1id[i]]] - targets[i])
              for i in range(6)]
    assert max(errors) < 1e-3


def test_all_active_matches_legacy_physics_and_bad_angles_reject(tmp_path):
    space, active, frozen, stock, _ = _designs(tmp_path / "topology")
    with DesignCompiler(DesignSpace.quadruped(), tmp_path / "legacy") as compiler:
        legacy = compiler.compile(active[:7])
    top_model = mujoco.MjModel.from_xml_path(str(stock.xml_path))
    legacy_model = mujoco.MjModel.from_xml_path(str(legacy.xml_path))
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "geom_size", "geom_pos", "geom_quat", "jnt_range",
                  "dof_armature", "dof_damping", "dof_frictionloss"):
        np.testing.assert_allclose(getattr(top_model, field), getattr(legacy_model, field))
    assert legacy_model.neq == 0 and top_model.neq == 12
    bad = frozen.copy()
    bad[13] = float(space.upper[13]) + .1
    with DesignCompiler(space, tmp_path / "rejection") as compiler:
        results = compiler.compile_many([bad, frozen])
    assert results[0].design is None and "bounds" in results[0].error
    assert results[1].design is not None


@pytest.mark.needs_rl
def test_native_varying_topology_fields_are_batched(tmp_path):
    pytest.importorskip("mujoco_warp")
    _, _, _, active, frozen = _designs(tmp_path)
    from draft.codesign.batch import model_field_batches
    cohort = prepare_cohort([active, frozen], 1)
    models = [cohort.robot_spec(i).compile() for i in range(2)]
    batches = model_field_batches(models)
    assert {"eq_data", "dof_armature", "dof_damping", "dof_frictionloss",
            "body_mass", "body_inertia"} <= batches.keys()
    assert not np.array_equal(batches["eq_data"][0], batches["eq_data"][1])
    assert not np.array_equal(batches["dof_armature"][0], batches["dof_armature"][1])


def test_active_angles_canonicalize_cache_identity(tmp_path):
    space = DesignSpace.quadruped(joint_topology=True)
    center = np.asarray(space.center)
    changed = center.copy()
    changed[13:19] = [1.5, -1.2, -2.0, .6, .7, -1.0]
    with DesignCompiler(space, tmp_path) as compiler:
        first, second = compiler.compile_many([center, changed])
    assert first.design_id == second.design_id
    assert first.design.xml_path == second.design.xml_path
    assert json.loads((first.design.xml_path.parent / "design.json").read_text())["files"]


def test_all_symmetric_masks_keep_compiled_structure(tmp_path):
    space = DesignSpace.quadruped()
    with DesignCompiler(space, tmp_path / "source") as compiler:
        base = compiler.compile(space.center).xml_path.parent
        signatures = set()
        for bits in range(64):
            target = tmp_path / f"mask-{bits}"
            shutil.copytree(base, target)
            active = [(bits >> i) & 1 for i in range(6)]
            apply_joint_topology(target, {"active": active,
                                          "lock_angles": [0.] * 6})
            model = mujoco.MjModel.from_xml_path(str(target / "quadruped.xml"))
            signatures.add((model.nq, model.nv, model.nbody, model.ngeom,
                            model.nsite, model.neq,
                            tuple(model.joint(i).name for i in range(model.njnt)),
                            tuple(model.equality(i).name for i in range(model.neq))))
        assert len(signatures) == 1


def test_export_activates_locks_and_zeroes_existing_frozen_actuators(tmp_path):
    _, _, _, _, frozen = _designs(tmp_path / "cache")
    destination = tmp_path / "export"
    shutil.copytree(frozen.xml_path.parent, destination)
    exported_xml = destination / "quadruped.xml"
    tree = ET.parse(exported_xml)
    block = ET.SubElement(tree.getroot(), "actuator")
    for name in JOINT_NAMES:
        ET.SubElement(block, "motor", joint=name, name=f"{name}_actuator", gear="1")
    tree.write(exported_xml)
    record = prepare_exported_topology(exported_xml, frozen.parameters["joint_topology"])
    model = mujoco.MjModel.from_xml_path(str(exported_xml))
    cached = mujoco.MjModel.from_xml_path(str(frozen.xml_path))
    assert model.nu == record["actuator_slots"] == 12
    assert record["active_locks"] == record["frozen_actuator_slots_zeroed"] == 6
    np.testing.assert_array_equal(model.eq_active0[:6], 1)
    np.testing.assert_array_equal(model.eq_active0[6:], 0)
    np.testing.assert_array_equal(cached.eq_active0, 0)
    np.testing.assert_array_equal(model.actuator_gear[:6], 0)
    np.testing.assert_array_equal(model.actuator_gear[6:, 0], 1)


def test_folded_locked_posture_is_feasibility_rejection(tmp_path):
    space = DesignSpace.quadruped(joint_topology=True)
    folded = np.asarray(space.center)
    folded[7:13] = 0
    folded[13:19] = [0., -2.5, -2.8, 0., -2.5, -2.8]
    with DesignCompiler(space, tmp_path) as compiler:
        result = compiler.compile_many([folded])[0]
    assert result.design is None
    assert "nominal stance height" in result.error
