"""Low-cost, terrain-only collision envelopes for generated quadruped hardware.

The stock task keeps its original feet-only collisions. Opt-in callers add
massless subdivided boxes enclosing each body's generated physical geometry.
These are conservative collision proxies, not exact chamfered or curved solids.
"""
from __future__ import annotations

from functools import wraps
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

_LEGS = ("fl", "fr", "rl", "rr")
_BODY_SUFFIXES = ("roll_motor_link", "roll_link", "pitch_link", "upper_leg_link", "lower_leg_link")
PROXY_BODIES = ("torso_link", *(f"{leg}_{suffix}" for leg in _LEGS for suffix in _BODY_SUFFIXES))
PROXY_PREFIX = "hardware_collision_"
PROXY_GEOM_PATTERN = rf"{PROXY_PREFIX}.*"
PROXY_GEOM_NAMES = tuple(f"{PROXY_PREFIX}{body}" for body in PROXY_BODIES)
TORSO_PROXY_NAME = f"{PROXY_PREFIX}torso_link"
FOOT_GEOM_NAMES = tuple(f"{leg}_lower_leg_link_sphere" for leg in _LEGS)
_FOOT_PATTERN = r".*_lower_leg_link_sphere"
# At the task's minimum 0.1m hfield spacing, a <=0.25m box diagonal
# spans at most four cells per horizontal axis at every orientation/offset:
# ceil(width / spacing) + 1 <= 4, or <=32 triangular prisms per pair.
# Warp's fixed MJ_MAXCONPAIR is 50; a single long torso box exceeds it.
MAX_PROXY_DIAGONAL = 0.25
CollisionSubdivisions = Mapping[str, tuple[int, int, int]]


def _body_bounds(model: mujoco.MjModel, body_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return the exact body-local AABB of its nondecorative, non-foot geoms."""
    lower, upper = [], []
    representative = None
    for gid in range(int(model.body_geomadr[body_id]),
                     int(model.body_geomadr[body_id] + model.body_geomnum[body_id])):
        name = model.geom(gid).name
        if name in FOOT_GEOM_NAMES or name.endswith("_detail"):
            continue
        center, half = model.geom_aabb[gid].reshape(2, 3)
        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        rotation = rotation.reshape(3, 3)
        center = model.geom_pos[gid] + rotation @ center
        half = np.abs(rotation) @ half
        lower.append(center - half)
        upper.append(center + half)
        if representative is None:
            representative = gid
    if representative is None:
        raise ValueError(f"No physical geometry on required robot body {model.body(body_id).name!r}")
    lo, hi = np.min(lower, axis=0), np.max(upper, axis=0)
    if not np.isfinite((lo, hi)).all() or (hi <= lo).any():
        raise ValueError(f"Invalid collision envelope on robot body {model.body(body_id).name!r}")
    return (lo + hi) / 2, (hi - lo) / 2, representative


def _envelopes(model: mujoco.MjModel):
    envelopes = []
    for name in PROXY_BODIES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Hardware collision proxies require generated quadruped body {name!r}")
        envelopes.append((name, *_body_bounds(model, body_id)))
    return envelopes


def _subdivide(extents: np.ndarray, margin: float, max_diagonal: float) -> tuple[int, int, int]:
    counts = np.ones(3, dtype=int)
    if 2 * margin >= max_diagonal:
        raise ValueError("Collision margin leaves no room for a bounded hardware proxy")
    while np.linalg.norm(extents / counts) + 2 * margin > max_diagonal:
        counts[np.argmax(extents / counts)] += 1
    return tuple(map(int, counts))


def collision_subdivisions(xml_paths: Sequence[str | Path], *,
                           max_diagonal: float = MAX_PROXY_DIAGONAL,
                           models: Sequence[mujoco.MjModel] | None = None) -> dict[str, tuple[int, int, int]]:
    """Choose one bounded topology for all designs, with O(D) CPU preparation.

    Cell sizes/positions vary physically per design; only subdivision counts are
    shared. Pass this layout to every cohort entity so all native array shapes
    agree. No terrain, geometry dimensions, or inertias are changed.
    ``models`` can reuse matching, read-only source models in XML order.
    """
    if not xml_paths:
        raise ValueError("At least one robot XML is required")
    if not np.isfinite(max_diagonal) or not 0 < max_diagonal <= MAX_PROXY_DIAGONAL:
        raise ValueError(f"max_diagonal must be positive and <= {MAX_PROXY_DIAGONAL}")
    if models is not None and len(models) != len(xml_paths):
        raise ValueError("Source models must match the robot XML count")
    extents = {name: np.zeros(3) for name in PROXY_BODIES}
    margins = dict.fromkeys(PROXY_BODIES, 0.0)
    sources = (mujoco.MjModel.from_xml_path(str(path)) for path in xml_paths) if models is None else models
    for model in sources:
        for name, _, half, representative in _envelopes(model):
            extents[name] = np.maximum(extents[name], 2 * half)
            margins[name] = max(margins[name], float(model.geom_margin[representative]))
    return {name: _subdivide(extents[name], margins[name], max_diagonal) for name in PROXY_BODIES}


def _validated_layout(envelopes, model, subdivisions):
    if subdivisions is not None and set(subdivisions) != set(PROXY_BODIES):
        raise ValueError("Collision subdivisions must specify every generated robot body")
    layout = {}
    for name, _, half, representative in envelopes:
        margin = float(model.geom_margin[representative])
        counts = (_subdivide(2 * half, margin, MAX_PROXY_DIAGONAL) if subdivisions is None
                  else subdivisions[name])
        if (len(counts) != 3 or any(isinstance(n, (bool, np.bool_))
                or not isinstance(n, (int, np.integer)) or n < 1 for n in counts)):
            raise ValueError(f"Expected three positive integer subdivision counts for {name!r}")
        if np.linalg.norm(2 * half / counts) + 2 * margin > MAX_PROXY_DIAGONAL + 1e-12:
            raise ValueError(f"Collision subdivisions for {name!r} exceed the safe proxy diagonal")
        layout[name] = tuple(map(int, counts))
    return layout


def add_hardware_collision_proxies(spec: mujoco.MjSpec, *,
                                   subdivisions: CollisionSubdivisions | None = None,
                                   model: mujoco.MjModel | None = None) -> mujoco.MjSpec:
    """Tile physical envelopes with massless boxes bounded for native hfields.

    Call on an unattached generated robot spec, before entity collision setup.
    For heterogeneous cohorts supply the same ``collision_subdivisions`` layout
    to every design. Bank geoms on worldbody retain their separate channel.
    An optional read-only model must match the spec's original robot geometry;
    added shared mesh-bank world geoms and removed actuators are allowed.
    """
    if any(geom.name.startswith(PROXY_PREFIX) for geom in spec.geoms):
        raise ValueError("Hardware collision proxies are already present")
    if model is None:
        model = spec.compile()
    envelopes = _envelopes(model)
    layout = _validated_layout(envelopes, model, subdivisions)
    for name in FOOT_GEOM_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            raise ValueError(f"Hardware collision proxies require foot geom {name!r}")
    target_bodies = set(PROXY_BODIES)
    for geom in spec.geoms:
        if geom.parent.name in target_bodies:
            geom.contype = 0
            geom.conaffinity = 1 if geom.name in FOOT_GEOM_NAMES else 0
    for name, center, half, representative in envelopes:
        counts = layout[name]
        cell_half = half / counts
        for index, cell in enumerate(product(*(range(n) for n in counts))):
            suffix = f"_part_{index}" if index else ""
            pos = center - half + (2 * np.asarray(cell) + 1) * cell_half
            spec.body(name).add_geom(
                name=f"{PROXY_PREFIX}{name}{suffix}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos, size=cell_half, mass=0.0, density=0.0, group=3, rgba=(0, 0, 0, 0),
                contype=0, conaffinity=1, condim=3, priority=1,
                friction=model.geom_friction[representative],
                solref=model.geom_solref[representative], solimp=model.geom_solimp[representative],
                solmix=float(model.geom_solmix[representative]),
                margin=float(model.geom_margin[representative]), gap=float(model.geom_gap[representative]))
    # MuJoCo 3.8 retains the pre-addition name index after the first compile.
    # Refresh it so native CollisionCfg can resolve every newly created proxy.
    spec.compile()
    return spec


def validate_hfield_collision_budget(model: mujoco.MjModel) -> None:
    """Reject unsupported fine terrains before allocating/launching GPU worlds.

    Conservatively bound all triangle visits, even for deep penetration at any
    body orientation. The bound needs only O(geoms + hfields) work per design;
    it does not depend on replica count and never modifies the common terrain.
    """
    hfields = np.flatnonzero((model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)
                             & (model.geom_contype != 0))
    if not len(hfields):
        return
    field_ids = model.geom_dataid[hfields]
    pitch = np.min(2 * model.hfield_size[field_ids, :2]
                   / (np.column_stack((model.hfield_ncol[field_ids],
                                       model.hfield_nrow[field_ids])) - 1), axis=0)
    if not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Hardware collision heightfields require a positive finite cell pitch")
    ground_margin = float(np.max(model.geom_margin[hfields]))
    ground_channels = int(np.bitwise_or.reduce(model.geom_contype[hfields]))
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        local_name = name.rsplit("/", 1)[-1]
        if (not (local_name.startswith(PROXY_PREFIX) or local_name in FOOT_GEOM_NAMES)
                or not (int(model.geom_conaffinity[gid]) & ground_channels)):
            continue
        # rbound is the sphere radius for feet, the half-diagonal for boxes,
        # and a conservative enclosing radius for any future proxy primitive.
        span = 2 * (float(model.geom_rbound[gid]) + float(model.geom_margin[gid]) + ground_margin)
        cells = np.ceil(span / pitch).astype(np.int64) + 1
        triangles = 2 * int(cells[0]) * int(cells[1])
        if triangles >= 50:
            raise ValueError(
                f"Hardware collision geom {name!r} can visit {triangles} heightfield triangles "
                f"at cell pitch {tuple(map(float, pitch))}; MuJoCo Warp requires fewer than 50. "
                "Use a finer shared collision subdivision layout for this terrain.")


def configure_hardware_collisions(entity: Any, *, subdivisions: CollisionSubdivisions | None = None,
                                  model: mujoco.MjModel | None = None):
    """Wrap an EntityCfg's current spec factory; leave the baseline opt-in.

    Calling twice on the same factory is harmless. Replacing ``spec_fn`` (as the
    heterogeneous cohort adapter does) deliberately permits wrapping the new
    per-design factory again. No mutable global model cache is kept.
    """
    from mjlab.utils.spec_config import CollisionCfg

    source = entity.spec_fn
    layout = None if subdivisions is None else {name: tuple(counts) for name, counts in subdivisions.items()}
    if (not getattr(source, "_hardware_collision_proxies", False)
            or getattr(source, "_hardware_collision_layout", None) != layout
            or (model is not None and getattr(source, "_hardware_collision_model", None) is not model)):
        source = getattr(source, "_hardware_collision_source", source)
        @wraps(source)
        def hardware_spec():
            return add_hardware_collision_proxies(source(), subdivisions=layout, model=model)

        hardware_spec._hardware_collision_proxies = True
        hardware_spec._hardware_collision_source = source
        hardware_spec._hardware_collision_layout = layout
        hardware_spec._hardware_collision_model = model
        entity.spec_fn = hardware_spec
    entity.collisions = (CollisionCfg(
        geom_names_expr=(_FOOT_PATTERN, PROXY_GEOM_PATTERN),
        contype=0, conaffinity=1, condim=3, priority=1, disable_other_geoms=True),)
    return entity
