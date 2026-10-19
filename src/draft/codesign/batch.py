"""Immutable heterogeneous cohorts backed by native MuJoCo model arrays.

Mesh vertices are shared assets in MuJoCo Warp. A cohort therefore carries a
union mesh bank and selects each world's geometry with batched ``geom_dataid``.
No vertex buffer is incorrectly given a world dimension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

LEGS = ("fl", "fr", "rl", "rr")
JOINT_NAMES = tuple(f"{leg}_{role}" for leg in LEGS
                    for role in ("hip_roll", "hip_pitch", "knee"))
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
_MESH_FIELDS = ("content_type", "refpos", "refquat", "scale", "inertia",
                "smoothnormal", "needsdf", "maxhullvert", "uservert", "usernormal",
                "usertexcoord", "userface", "userfacenormal", "userfacetexcoord")


@dataclass(frozen=True)
class PreparedCohort:
    designs: tuple[Any, ...]
    replicas: int
    mesh_bank: tuple[dict[str, Any], ...]
    mesh_maps: tuple[dict[str, str], ...]
    # Preparation-owned source models, never shared through a global path cache.
    # Consumers must treat these as read-only; scene models compile separately.
    robot_models: tuple[mujoco.MjModel, ...] = field(default=(), repr=False, compare=False)

    @property
    def num_envs(self) -> int:
        return len(self.designs) * self.replicas

    @property
    def design_index(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.designs), dtype=np.int64), self.replicas)

    def robot_spec(self, index: int) -> mujoco.MjSpec:
        """Build a real design with identical shared asset indexing in every scene."""
        spec = mujoco.MjSpec.from_file(str(self.designs[index].xml_path))
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        names = self.mesh_maps[index]
        for geom in spec.geoms:
            if geom.meshname:
                geom.meshname = names[geom.meshname]
            geom.group = 1
        for mesh in list(spec.meshes):
            spec.delete(mesh)
        for asset in self.mesh_bank:
            spec.add_mesh(**asset)
            # MuJoCo builds convex-hull tables only for referenced meshes. A
            # task-isolated invisible world geom makes every bank entry present
            # in every compiled model, keeping the shared hull arrays identical.
            spec.worldbody.add_geom(
                name=f"bank_{asset['name']}", type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=asset["name"], contype=2, conaffinity=2, group=5,
                pos=(0, 0, -1000),
                rgba=(0, 0, 0, 0), mass=0.0)
        # All filenames have been resolved before merging differing mesh dirs.
        spec.compiler.meshdir = ""
        return spec


def prepare_cohort(designs: Sequence[Any], replicas: int) -> PreparedCohort:
    """Validate geometry-only 12-DOF designs and prepare their shared asset bank.

    This path needs only NumPy and CPU MuJoCo; it never initializes CUDA or Torch.
    The environment performs a second, strict topology/array check after native
    entities, motors, terrain and sensors have been compiled into complete scenes.
    """
    if isinstance(replicas, bool) or int(replicas) != replicas or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    designs = tuple(designs)
    if not designs:
        raise ValueError("a cohort must contain at least one design")
    bank, maps, models = [], [], []
    motor_signature = None
    topology = None
    for index, design in enumerate(designs):
        path = Path(design.xml_path).resolve()
        morphology = np.asarray(design.morphology, dtype=float)
        if morphology.shape != (7,) or not np.isfinite(morphology).all():
            raise ValueError("each design requires seven finite morphology values")
        if not np.isfinite(design.leg_length) or design.leg_length <= 0:
            raise ValueError("leg_length must be finite and positive")
        model = mujoco.MjModel.from_xml_path(str(path))
        models.append(model)
        names = tuple(model.joint(i).name for i in range(model.njnt)
                      if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE)
        if set(names) != set(JOINT_NAMES) or model.nq != 19 or model.nv != 18:
            raise ValueError(f"{path}: expected one free root and the fixed 12 quadruped joints")
        if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) < 0
               for name in FOOT_SITES):
            raise ValueError(f"{path}: all four named foot sites are required")
        signature = (model.nbody, model.njnt, model.ngeom, model.nsite, names,
                     tuple(model.jnt_type), tuple(model.geom_type),
                     tuple(model.body(i).name for i in range(model.nbody)),
                     tuple(model.geom(i).name for i in range(model.ngeom)),
                     tuple(model.site(i).name for i in range(model.nsite)))
        if topology is not None and signature != topology:
            raise ValueError("cohort designs must have identical body/joint/geom/site topology")
        topology = signature
        # Motor effort, speed, gear and topology must not become optimization variables.
        speeds = tuple((model.numeric(i).name, tuple(model.numeric(i).data))
                       for i in range(model.nnumeric)
                       if model.numeric(i).name.endswith("_velocity_limit"))
        motors = (model.nu, tuple(model.actuator_trnid.ravel()),
                  tuple(model.actuator_forcerange.ravel()), tuple(model.actuator_gear.ravel()),
                  tuple(model.jnt_actfrcrange.ravel()), tuple(model.dof_armature.ravel()), speeds)
        if model.nu not in (0, 12) or len(speeds) != 12 or (
                motor_signature is not None and motors != motor_signature):
            raise ValueError("all designs must use the same 12 motors, gear ratios and speed limits")
        motor_signature = motors
        spec = mujoco.MjSpec.from_file(str(path))
        mapping = {}
        for mesh in spec.meshes:
            new_name = f"cohort_{index}_{mesh.name}"
            mapping[mesh.name] = new_name
            values = {name: getattr(mesh, name) for name in _MESH_FIELDS}
            values = {name: value.copy() if isinstance(value, np.ndarray) else value
                      for name, value in values.items()}
            values["name"] = new_name
            if mesh.file:
                source = Path(mesh.file)
                if not source.is_absolute():
                    source = path.parent / spec.compiler.meshdir / source
                values["file"] = str(source.resolve())
            bank.append(values)
        maps.append(mapping)
    return PreparedCohort(designs, int(replicas), tuple(bank), tuple(maps), tuple(models))


def model_field_batches(models: Sequence[mujoco.MjModel]) -> dict[str, np.ndarray]:
    """Extract every native world-dependent field, rejecting unsupported variation.

    The pinned Warp model annotations define world support. This also audits
    shared topology and mesh buffers rather than silently dropping varying data.
    """
    import mujoco_warp as mjwarp
    if not models:
        raise ValueError("models cannot be empty")
    batches = {}
    for name, annotation in mjwarp.Model.__annotations__.items():
        shape = getattr(annotation, "shape", ())
        if not shape or not hasattr(models[0], name):
            continue
        arrays = [np.asarray(getattr(model, name)) for model in models]
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError(f"incompatible native model field shape: {name}")
        if shape[0] == "*":
            batches[name] = np.stack(arrays)
        elif any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
            raise ValueError(f"native field {name!r} does not support per-world variation")
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "nsite", "nmesh", "nsensor"):
        if len({getattr(model, name) for model in models}) != 1:
            raise ValueError(f"incompatible native model topology: {name}")
    return batches


class HeterogeneousModelAdapter:
    """Install compiled design fields once; references remain valid for CUDA graphs."""

    @staticmethod
    def apply(env: Any, cohort: PreparedCohort, models: Sequence[mujoco.MjModel],
              *, only_changed: bool = False) -> None:
        import torch
        batches = model_field_batches(models)
        if only_changed:
            batches = {name: values for name, values in batches.items()
                       if any(not np.array_equal(values[0], row) for row in values[1:])}
        # Allocate before obtaining bridge tensors: expansion invalidates their cache.
        env.sim.expand_model_fields(tuple(batches))
        for name, values in batches.items():
            target = getattr(env.sim.model, name)
            source = torch.as_tensor(values, dtype=target.dtype, device=env.device)
            source = source.repeat_interleave(cohort.replicas, dim=0)
            if source.numel() != target.numel():
                raise ValueError(f"native field layout mismatch for {name}: {source.shape} vs {target.shape}")
            target.copy_(source.reshape(target.shape))
        # Every derived constant is copied from its independent CPU compile too.
        # Rebuilding graphs after pointer expansion is handled by mjlab itself.
        env.sim.forward()
