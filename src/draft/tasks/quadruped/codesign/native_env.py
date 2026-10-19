"""Original quadruped task with geometry-conditioned native cohorts.

Fixed stock and randomized geometry enter the same factory. Common demands and
the selected terrain/reward/contact profile comes from ``make_base_env_cfg``
with an explicit stock task-geometry reference. Hardware-aware posture targets
are bound to each design while external demands keep their common reference.
"""
from __future__ import annotations

import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.utils.string import resolve_expr

from draft.codesign.batch import (
    HeterogeneousModelAdapter, PreparedCohort, prepare_cohort,
)
from draft.tasks.quadruped.base_env import make_base_env_cfg
from draft.tasks.quadruped.conditioning import append_morphology_observations


@dataclass(frozen=True)
class NativeCohort:
    cohort: PreparedCohort
    entities: tuple[Any, ...]
    reference_xml: Path


def _robot_spec(cohort: PreparedCohort, index: int, geom_groups: dict[str, int]) -> mujoco.MjSpec:
    spec = cohort.robot_spec(index)
    for geom in spec.geoms:
        if geom.name in geom_groups:
            geom.group = geom_groups[geom.name]
    # Entity options are deliberately owned by the original common task.
    spec.option = mujoco.MjSpec().option
    return spec


def _preserve_mesh_bank(spec: mujoco.MjSpec) -> None:
    for geom in spec.geoms:
        if geom.name.startswith("robot/bank_cohort_"):
            # Required only to compile every bank entry's native convex hull.
            # Task geoms use bit1; static bank refs use bit2, 1km underground.
            geom.contype = geom.conaffinity = 2


def augment_native_cohort_cfg(cfg, designs: Sequence[Any], replicas: int,
                               reference_xml: str | Path):
    """Add physical cohorts and morphology7 to an already authored task config.

    Authored task settings are preserved. The explicit hardware-aware profile
    additionally binds each design's nominal posture and collision envelopes.
    Each entity inherits the task's motor and other physical settings. Other
    tasks may opt into this seam explicitly in future.
    """
    cohort = prepare_cohort(designs, replicas)
    reference_xml = Path(reference_xml).resolve()
    if cfg.scene.num_envs != cohort.num_envs:
        raise ValueError("Prebuilt task environment count must equal designs times replicas")
    template = cfg.scene.entities["robot"]
    entities = [template]
    if len(cohort.designs) > 1:
        template_spec = template.spec_fn()
        groups = {geom.name: geom.group for geom in template_spec.geoms}
        entities = []
        for index in range(len(cohort.designs)):
            entity = copy.deepcopy(template)
            entity.spec_fn = lambda index=index: _robot_spec(cohort, index, groups)
            entities.append(entity)
        cfg.scene.entities["robot"] = entities[0]
        previous = cfg.scene.spec_fn

        def scene_callback(spec):
            if previous is not None:
                previous(spec)
            _preserve_mesh_bank(spec)

        cfg.scene.spec_fn = scene_callback
    if getattr(cfg, "hardware_aware", False):
        from draft.tasks.quadruped.hardware import prepare_hardware_entity, set_hardware_heights
        if len(cohort.designs) > 1:
            from draft.tasks.quadruped.hardware_collision import collision_subdivisions
            layout = collision_subdivisions([design.xml_path for design in cohort.designs],
                                            models=cohort.robot_models)
            reference_model = next((model for design, model in zip(cohort.designs, cohort.robot_models)
                                    if Path(design.xml_path).resolve() == reference_xml), None)
            if reference_model is None:
                reference_model = mujoco.MjModel.from_xml_path(str(reference_xml))
            heights = [prepare_hardware_entity(entity, design.xml_path, reference_xml,
                                              collision_subdivisions=layout, model=model,
                                              stance_model=reference_model)
                       for entity, design, model in zip(entities, cohort.designs, cohort.robot_models)]
        else:
            heights = cfg.hardware_nominal_heights
        set_hardware_heights(cfg, heights, replicas)
    values = np.asarray([design.morphology for design in cohort.designs], dtype=float)
    values = np.repeat(values, cohort.replicas, axis=0)
    metadata = dict(version=1, observation_term="morphology", reference_xml=str(reference_xml),
                    num_designs=len(cohort.designs), replicas=cohort.replicas,
                    ordering="design-major contiguous replicas", design_ids=[d.design_id for d in cohort.designs],
                    physical=[list(d.physical) for d in cohort.designs],
                    normalized=[list(d.morphology) for d in cohort.designs],
                    task="authored task configuration; common explicit task geometry")
    if getattr(cfg, "hardware_aware", False):
        metadata.update(hardware_aware=True, nominal_stance_heights=list(cfg.hardware_nominal_heights))
    append_morphology_observations(cfg, values, metadata=metadata)
    cfg.native_cohort = NativeCohort(cohort, tuple(entities), reference_xml)
    return cfg


def make_native_env_cfg(designs: Sequence[Any], replicas: int,
                        reference_xml: str | Path, *, env_cfg_factory=None,
                        cpu_compile_workers: int = 4, **base):
    """Author the common original task, then apply the reusable cohort wrapper.

    An injectable task factory is an extension seam, not automatic registration
    for other tasks. Factories may return a config or ``(config, context)``.
    ``episode_length_s`` retains the selected task's original episode semantics.
    ``cpu_compile_workers=1`` selects serial independent CPU reference builds.
    """
    designs = tuple(designs)
    if not designs or isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError("A nonempty design cohort and positive integer replicas are required")
    _compile_worker_count(cpu_compile_workers)
    factory = make_base_env_cfg if env_cfg_factory is None else env_cfg_factory
    if not callable(factory):
        raise TypeError("env_cfg_factory must be callable")
    base = dict(base)
    task_reference = Path(base.pop("task_geometry_xml", None) or reference_xml).resolve()
    for key in ("robot_xml", "n_envs"):
        base.pop(key, None)
    result = factory(robot_xml=designs[0].xml_path,
                     task_geometry_xml=task_reference,
                     n_envs=len(designs) * replicas, **base)
    cfg = result[0] if isinstance(result, tuple) else result
    cfg = augment_native_cohort_cfg(cfg, designs, replicas, task_reference)
    cfg.cpu_compile_workers = cpu_compile_workers
    return cfg


class _SharedTerrainScene(Scene):
    """Independent CPU robot/sensor compile using the one realized terrain spec."""

    def __init__(self, cfg, terrain, terrain_spec):
        self._shared_terrain = terrain
        self._shared_terrain_spec = terrain_spec
        super().__init__(cfg, device="cpu")

    def _add_terrain(self):
        terrain = self._shared_terrain
        if terrain is None:
            return
        self._terrain = terrain
        self._entities["terrain"] = terrain
        frame = self._spec.worldbody.add_frame()
        self._spec.attach(self._shared_terrain_spec.copy(), prefix="", frame=frame)


def _terrain_snapshot(scene: Scene) -> mujoco.MjSpec:
    # Copy the ROOT scene, never an attached child spec. MuJoCo retains a
    # child's parent on copy; attaching that copy elsewhere can hang in C++.
    spec = scene.spec.copy()
    for kind in ("sensors", "actuators", "keys", "pairs", "excludes", "equalities", "tendons"):
        for item in list(getattr(spec, kind)):
            spec.delete(item)
    for geom in list(spec.worldbody.geoms):
        if geom.name.startswith("robot/"):
            spec.delete(geom)
    for body in list(spec.worldbody.bodies):
        if body.name.startswith("robot/"):
            spec.delete(body)
    for kind in ("meshes", "materials", "textures", "numerics", "texts", "tuples"):
        for item in list(getattr(spec, kind)):
            if item.name.startswith("robot/"):
                spec.delete(item)
    return spec


def _compile_worker_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("cpu_compile_workers must be a positive integer")
    return value


def compile_cohort_models(scene: Scene, cfg, cohort: NativeCohort):
    """Compile ordered CPU references from one terrain, without consuming RNG.

    Only independent MuJoCo compiles run in threads. Scene construction stays
    on the caller: authored callbacks and sensor builders need not be thread
    safe, and copies of the shared terrain spec are never made concurrently.
    At most ``cpu_compile_workers`` independent scenes are retained in flight;
    set it to one for the serial path.
    """
    workers = min(_compile_worker_count(getattr(cfg, "cpu_compile_workers", 4)),
                  len(cohort.entities))
    models = [None] * len(cohort.entities)
    terrain_spec = _terrain_snapshot(scene)

    def build(entity):
        scene_cfg = copy.deepcopy(cfg.scene)
        scene_cfg.entities = {"robot": entity}
        return _SharedTerrainScene(scene_cfg, scene.terrain, terrain_spec)

    if workers <= 1:
        for index, entity in enumerate(cohort.entities):
            model = build(entity).compile()
            cfg.sim.mujoco.apply(model)
            models[index] = model
        return tuple(models)

    entities = iter(enumerate(cohort.entities))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="draft-cpu-compile") as pool:
        pending = {}

        def submit():
            item = next(entities, None)
            if item is None:
                return
            index, entity = item
            pending[pool.submit(build(entity).compile)] = index

        try:
            for _ in range(workers):
                submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    model = future.result()
                    cfg.sim.mujoco.apply(model)
                    models[index] = model
                for _ in done:
                    submit()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return tuple(models)


def balance_cohort_terrain(scene: Scene, num_designs: int, replicas: int) -> bool:
    """Spread each native terrain column across design-major replica blocks.

    This is a deterministic permutation of the realized native assignments:
    global terrain/level/origin multisets and RNG state remain unchanged. Model
    rows and morphology ordering are never permuted. Single-design tasks retain
    their exact original assignment, including object identity and storage.
    """
    terrain = scene.terrain
    if num_designs <= 1 or terrain is None or terrain.terrain_origins is None:
        return False
    count = num_designs * replicas
    fields = (terrain.terrain_types, terrain.terrain_levels, terrain.env_origins)
    if any(value.shape[0] != count for value in fields):
        raise ValueError("Native terrain assignment count must match the cohort")
    if scene.env_origins.data_ptr() != terrain.env_origins.data_ptr():
        raise RuntimeError("Scene and native terrain must share the environment-origin buffer")
    permutation = torch.arange(count, device=terrain.env_origins.device).reshape(
        replicas, num_designs).T.reshape(-1)
    for value in fields:
        value.copy_(value.index_select(0, permutation))
    return True


def _validate_hardware_terrain(cfg, models):
    if getattr(cfg, "hardware_aware", False):
        from draft.tasks.quadruped.hardware_collision import validate_hfield_collision_budget
        for model in models:
            validate_hfield_collision_budget(model)


class HardwareAwareEnv(ManagerBasedRlEnv):
    def load_managers(self):
        _validate_hardware_terrain(self.cfg, (self.sim.mj_model,))
        return super().load_managers()


class NativeCohortEnv(ManagerBasedRlEnv):
    def load_managers(self):
        info = self.cfg.native_cohort
        self.cohort = info.cohort
        self.design_index = torch.as_tensor(info.cohort.design_index, device=self.device)
        if len(info.cohort.designs) == 1:
            # Exact original native setup, startup DR and cached model defaults.
            self.cpu_models = (self.sim.mj_model,)
            _validate_hardware_terrain(self.cfg, self.cpu_models)
            return super().load_managers()
        balance_cohort_terrain(self.scene, len(info.cohort.designs), info.cohort.replicas)
        self.cpu_models = compile_cohort_models(self.scene, self.cfg, info)
        _validate_hardware_terrain(self.cfg, self.cpu_models)
        HeterogeneousModelAdapter.apply(self, info.cohort, self.cpu_models, only_changed=True)
        # Native DR must add/scale each design's own physical baseline rather
        # than replacing heterogeneous fields with the first design's defaults.
        if not hasattr(self.sim, "per_world_default_fields"):
            raise RuntimeError("Native heterogeneous DR requires mjlab per-world model defaults support")
        randomized = {name for term in self.cfg.events.values() if term is not None
                      for name in getattr(term.func, "model_fields", ())}
        defaults = self.sim.expanded_fields & randomized
        for name in defaults:
            self.sim.default_model_fields[name] = getattr(self.sim.model, name).clone()
        self.sim.per_world_default_fields.update(defaults)
        robot = self.scene["robot"]
        for index, entity in enumerate(info.entities):
            rows = slice(index * info.cohort.replicas, (index + 1) * info.cohort.replicas)
            robot.data.default_root_state[rows] = torch.tensor(
                (*entity.init_state.pos, *entity.init_state.rot,
                 *entity.init_state.lin_vel, *entity.init_state.ang_vel), device=self.device)
            robot.data.default_joint_pos[rows] = torch.tensor(
                resolve_expr(entity.init_state.joint_pos, robot.joint_names, 0.0), device=self.device)
        joint_ids = [self.sim.mj_model.joint(f"robot/{name}").id for name in robot.joint_names]
        limits = self.sim.model.jnt_range[:, joint_ids].clone()
        if limits.shape[0] == 1:
            limits = limits.expand(self.num_envs, -1, -1).clone()
        robot.data.default_joint_pos_limits = limits.clone()
        robot.data.joint_pos_limits = limits.clone()
        center = limits.mean(dim=-1)
        articulation = self.cfg.scene.entities["robot"].articulation
        factor = articulation.soft_joint_pos_limit_factor if articulation is not None else 1.0
        half = 0.5 * factor * (limits[..., 1] - limits[..., 0])
        robot.data.soft_joint_pos_limits = torch.stack((center - half, center + half), dim=-1)
        # mjlab's expansion helper retile-copies even already expanded fields.
        # EventManager performs its normal expansion during load_managers; only
        # newly requested fields need allocation after our cohort installation.
        expand = self.sim.expand_model_fields
        self.sim.expand_model_fields = lambda fields: expand(tuple(
            name for name in fields if name not in self.sim.expanded_fields))
        try:
            super().load_managers()
        finally:
            self.sim.expand_model_fields = expand

    def close(self):
        try:
            super().close()
        finally:
            compiler = getattr(self.cfg, "_conditioning_compiler", None)
            if compiler is not None:
                compiler.close()
                self.cfg._conditioning_compiler = None


def make_native_env(cfg, device: str, render_mode: str | None = None):
    """Return a raw native environment, ready for the usual training wrappers."""
    if not hasattr(cfg, "native_cohort"):
        factory = HardwareAwareEnv if getattr(cfg, "hardware_aware", False) else ManagerBasedRlEnv
        return factory(cfg, device=device, render_mode=render_mode)
    try:
        return NativeCohortEnv(cfg, device=device, render_mode=render_mode)
    except BaseException:
        compiler = getattr(cfg, "_conditioning_compiler", None)
        if compiler is not None:
            compiler.close()
            cfg._conditioning_compiler = None
        raise


make_env = make_native_env


def build_native_env(designs, replicas, *, reference_xml, device, episode_length_s=20.0,
                     seed=0, env_kwargs=None, reward_overrides=None, render_mode=None):
    """Convenience raw-env factory; generic train/play may use the two-stage API."""
    options = dict(env_kwargs or {})
    render_mode = options.pop("render_mode", render_mode)
    render_size = {name: options.pop(f"render_{name}", None) for name in ("width", "height")}
    if render_mode not in (None, "rgb_array"):
        raise ValueError("render_mode must be None or 'rgb_array'")
    for name, value in render_size.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2):
            raise ValueError(f"render_{name} must be a positive even integer")
    cfg = make_native_env_cfg(designs, replicas, reference_xml,
                              episode_length_s=episode_length_s, **options)
    cfg.seed = seed
    for name, value in render_size.items():
        if value is not None:
            setattr(cfg.viewer, name, value)
    if render_mode is not None:
        cfg.viewer.max_extra_envs = 0
    for name, weight in (reward_overrides or {}).items():
        if name not in cfg.rewards:
            raise ValueError(f"Unknown native reward override: {name}")
        cfg.rewards[name].weight = float(weight)
    return make_native_env(cfg, device=device, render_mode=render_mode)
