"""Post-generation joint locks and honest motor-to-structure mass replacement."""

from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import yaml

from draft.generation.generator import RobotGenerator
from draft.paths import repo_root
from draft.trends.feasibility import FeasibilityViolation
from .topology import JOINT_NAMES, expand_symmetric_topology


_MATERIAL_DESTINATION = {"hip_roll": "torso", "hip_pitch": "hip_link", "knee": "thigh"}


def _motor_hosts(resolved: dict) -> dict[str, str]:
    """Use the expanded generator tree's motor_for tags, not link name guesses."""
    generator = RobotGenerator(repo_root() / "src/draft/robots/quadruped")
    tree = generator._build_tree(resolved)
    hosts = {}

    def visit(node: dict) -> None:
        geom = node.get("geometry") or {}
        role = geom.get("motor_for")
        if role:
            leg = node["name"].split("_", 1)[0]
            joint = f"{leg}_{role}"
            if joint in hosts:
                raise ValueError(f"More than one physical motor housing for {joint}")
            hosts[joint] = f"{node['name']}_motor"
        for child in node.get("children", ()):
            visit(child)

    visit(tree)
    if set(hosts) != set(JOINT_NAMES):
        raise ValueError("Generator tree must identify one physical motor per quadruped joint")
    return hosts


def _check_nominal_stance(model: mujoco.MjModel, active: np.ndarray,
                          angles: np.ndarray) -> float:
    """Reject a lock combination whose nominal native pose cannot stand above feet."""
    # Match quadruped_entity.default_stance for this generated model: the
    # nominal k=1 crouch, clamped to the same 90% soft ranges and .2 rad room.
    def soft_range(name):
        lower, upper = model.jnt_range[model.joint(name).id]
        midpoint = .5 * (lower + upper)
        half = .45 * (upper - lower)
        return midpoint - half, midpoint + half

    knee_lo, knee_hi = soft_range("fl_knee")
    pitch_lo, pitch_hi = soft_range("fl_hip_pitch")
    k_min = max(0., -knee_hi + .2, 2 * (pitch_lo + .2))
    k_max = min(-knee_lo - .2, 2 * (pitch_hi - .2))
    k = max(0., .5 * (k_min + k_max)) if k_max <= k_min else min(max(1., k_min), k_max)
    nominal = np.asarray([0., .5 * k, -k] * 4)
    pose = np.where(active.astype(bool), nominal, angles)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.
    data.qpos[3] = 1.
    for name, value in zip(JOINT_NAMES, pose):
        jid = model.joint(name).id
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    foot_heights = []
    for leg in ("fl", "fr", "rl", "rr"):
        gid = model.geom(f"{leg}_lower_leg_link_sphere").id
        foot_heights.append(data.geom_xpos[gid, 2] - model.geom_size[gid, 0])
    height = float(data.xpos[model.body("torso_link").id, 2] - min(foot_heights))
    if not math.isfinite(height) or height <= 0:
        raise FeasibilityViolation(["Locked posture has no positive finite nominal stance height"])
    return height


def apply_joint_topology(directory: Path, topology: dict) -> None:
    """Modify a strictly generated model, preserving every body and geom slot."""
    active6 = np.asarray(topology["active"], dtype=float)
    angle6 = np.asarray(topology["lock_angles"], dtype=float)
    active12, angles12 = expand_symmetric_topology(active6, angle6)
    if active6.shape != (6,) or angle6.shape != (6,):
        raise ValueError("joint_topology requires six active values and six lock angles")

    xml_path = directory / "quadruped.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    geoms = {g.attrib["name"]: g for g in root.iter("geom") if "name" in g.attrib}
    joints = {j.attrib["name"]: j for j in root.iter("joint") if "name" in j.attrib}
    resolved_path = directory / "parameters_resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    hosts = _motor_hosts(resolved)
    report_path = directory / "feasibility_report.yaml"
    report = yaml.safe_load(report_path.read_text())
    before = mujoco.MjModel.from_xml_path(str(xml_path))
    equalities = ET.SubElement(root, "equality")
    receipts = []
    for name, enabled, angle in zip(JOINT_NAMES, active12, angles12):
        leg, role = name.split("_", 1)
        joint = joints[name]
        destination = _MATERIAL_DESTINATION[role]
        geom_name = hosts[name]
        geom = geoms[geom_name]
        old_rho = float(geom.attrib["density"])
        radius, half_length = map(float, geom.attrib["size"].split()[:2])
        volume = math.pi * radius**2 * 2 * half_length
        if destination == "torso":
            new_rho = float(resolved["torso_rho"])
            source = "torso_rho"
        elif destination == "hip_link":
            source = f"{leg}_roll_link_link.density"
            new_rho = float(geoms[f"{leg}_roll_link_link"].attrib["density"])
        else:
            source = "thigh_rho"
            new_rho = float(resolved["thigh_rho"])
        if not enabled:
            # This is the actual motor housing, not the knee's massless belt pulley.
            geom.set("density", format(new_rho, ".12g"))
            # MuJoCo requires a strictly ordered force range when its shared
            # limit flag is enabled. The runtime masks frozen actuator output.
            joint.set("armature", "0")
            joint.set("frictionloss", "0")
            joint.set("damping", "0")
        reference = float(before.qpos0[before.jnt_qposadr[before.joint(name).id]])
        ET.SubElement(equalities, "joint", {
            "name": f"lock_{name}", "joint1": name, "active": "false",
            "polycoef": f"{format(float(angle - reference), '.12g')} 0 0 0 0",
        })
        receipts.append({
            "joint": name, "active": bool(enabled), "lock_angle_rad": float(angle),
            "motor_geom": geom_name, "destination": destination,
            "density_source": source, "motor_density_kg_m3": old_rho,
            "effective_density_kg_m3": old_rho if enabled else new_rho,
            "envelope_volume_m3": volume,
            "motor_mass_kg": old_rho * volume,
            "effective_mass_kg": (old_rho if enabled else new_rho) * volume,
        })
    tree.write(xml_path, encoding="unicode")
    after = mujoco.MjModel.from_xml_path(str(xml_path))
    if after.nq != before.nq or after.nv != before.nv or after.ngeom != before.ngeom:
        raise ValueError("Topology replacement changed the shared model dimensions")
    if after.neq != 12 or np.any(after.eq_active0):
        raise ValueError("Topology replacement must declare twelve inactive equalities")
    stance_height = _check_nominal_stance(after, active12, angles12)
    mass_before = float(before.body_mass.sum())
    mass_after = float(after.body_mass.sum())
    expected = sum(r["effective_mass_kg"] - r["motor_mass_kg"] for r in receipts)
    if not np.isclose(mass_after - mass_before, expected, rtol=2e-4, atol=2e-4):
        raise ValueError("Model mass does not match the motor material replacement receipt")

    resolved["joint_topology"] = {
        "active": [int(x) for x in active6],
        "lock_angles": [float(x) for x in angle6],
    }
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    report["joint_topology"] = {
        "joint_materials": receipts,
        "generator_mass_kg": mass_before,
        "compiled_mass_kg": mass_after,
        "replaced_motor_mass_kg": sum(r["motor_mass_kg"] for r in receipts if not r["active"]),
        "replacement_structure_mass_kg": sum(r["effective_mass_kg"] for r in receipts if not r["active"]),
        "nominal_stance_height_m": stance_height,
    }
    report["total_mass_kg"] = round(mass_after, 4)
    roles = report.get("mass_kg", {})
    for receipt in receipts:
        if receipt["active"]:
            continue
        roles["motor"] = roles.get("motor", 0.0) - receipt["motor_mass_kg"]
        roles[receipt["destination"]] = roles.get(receipt["destination"], 0.0) + receipt["effective_mass_kg"]
    report["mass_kg"] = {key: round(value, 4) for key, value in roles.items()}
    # The generator's measured-population audit describes its original all-motor model.
    report["mass_composition_check"] = {
        "note": "Generator audit precedes topology material replacement; compiled masses are above.",
        "total_mass_kg": round(mass_after, 4),
    }
    report["generator_actuator_mass_audit"] = report.get("actuator_mass_audit")
    audit = report["actuator_mass_audit"] = {}
    active_rows = [r for r in receipts if r["active"]]
    charged = sum(float(resolved[f"{resolved[r['joint'].split('_', 1)[1] + '_mot']}_motor_mass"])
                  for r in active_rows)
    emitted = sum(r["motor_mass_kg"] for r in active_rows)
    audit.update({"n_joints": len(active_rows),
                  "charged_by_motor_law_kg": round(charged, 4),
                  "emitted_motor_geom_kg": round(emitted, 4),
                  "emitted_over_charged": round(emitted / charged, 4) if charged else None})
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))


def prepare_exported_topology(xml_path: Path, topology: dict) -> dict:
    """Activate physical locks in a copied XML and neutralize frozen actuators.

    Cached training XML deliberately has every equality inactive so a shared
    native model can choose activation per world. Exported XML describes one
    robot and must carry its locks in ``eq_active0`` for standalone MuJoCo use.
    """
    xml_path = Path(xml_path)
    active, _ = expand_symmetric_topology(topology["active"], topology["lock_angles"])
    if np.asarray(topology["active"]).shape != (6,):
        raise ValueError("Export topology requires six symmetric active values")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    equalities = {item.get("name"): item for item in root.findall("./equality/joint")}
    if set(equalities) != {f"lock_{name}" for name in JOINT_NAMES}:
        raise ValueError("Export requires the twelve declared joint lock equalities")
    frozen = {name for name, enabled in zip(JOINT_NAMES, active) if not enabled}
    for name in JOINT_NAMES:
        equalities[f"lock_{name}"].set("active", "true" if name in frozen else "false")

    actuator_count = zeroed = 0
    for element in root.findall("./actuator/*"):
        actuator_count += 1
        joint = element.get("joint")
        if joint in frozen:
            element.set("gear", "0 0 0 0 0 0")
            zeroed += 1
    tree.write(xml_path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.neq != 12 or model.nu != actuator_count:
        raise ValueError("Exported model changed equality or actuator slot count")
    for slot, name in enumerate(JOINT_NAMES):
        if model.equality(slot).name != f"lock_{name}" or bool(model.eq_active0[slot]) != (name in frozen):
            raise ValueError("Exported equality activation does not match topology")
    for slot in range(model.nu):
        trntype = model.actuator_trntype[slot]
        if trntype in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            joint_name = model.joint(model.actuator_trnid[slot, 0]).name
            if joint_name in frozen and np.any(model.actuator_gear[slot]):
                raise ValueError(f"Frozen actuator {slot} retains nonzero gear")
    return {"backend": "joint_equality", "frozen_joints": sorted(frozen),
            "active_locks": len(frozen), "actuator_slots": model.nu,
            "frozen_actuator_slots_zeroed": zeroed}
