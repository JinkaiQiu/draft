"""Offline conversion of geometry-conditioned controllers to topology inputs."""

from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import torch

from .models import ConditionedMLPModel
from .space import DesignSpace
from .topology import JOINT_GROUP_NAMES


MODEL_CLASS = "draft.codesign.models:ConditionedMLPModel"
ROLES = ("actor", "critic")


def _source_parts(payload):
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a mapping")
    if "policy" in payload:
        if payload.get("control_initialized") is not True:
            raise ValueError("Co-design checkpoint has no initialized controller")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("Co-design checkpoint lacks controller configuration")
        policy = payload["policy"]
        if not isinstance(policy, dict):
            raise ValueError("Co-design checkpoint lacks policy state")
        metadata = {key: config.get(key) for key in ("actor_cfg", "critic_cfg", "obs_groups")}
        metadata.update(space_schema=payload.get("space_schema"),
                        control_schema=payload.get("control_schema"),
                        compiler_fingerprint=payload.get("compiler_fingerprint"))
        exclusions = config.get("normalization_exclusions")
        if not isinstance(exclusions, dict):
            raise ValueError("Co-design checkpoint lacks normalization exclusions")
        metadata["conditioning_layout"] = exclusions
        states = {role: {key.removeprefix(f"control.{role}."): value for key, value in policy.items()
                         if key.startswith(f"control.{role}.")} for role in ROLES}
        return metadata, states, payload.get("iteration"), "codesign"
    infos = payload.get("infos")
    transfer = infos.get("codesign_transfer") if isinstance(infos, dict) else None
    if not isinstance(transfer, dict) or transfer.get("version") != 1:
        raise ValueError("Native checkpoint requires version-one infos.codesign_transfer")
    states = {role: payload.get(role + "_state_dict") for role in ROLES}
    return transfer, states, payload.get("iter"), "native"


def _target_space(source, lock_angle_bounds):
    if not isinstance(source, dict) or source.get("version") != 1:
        raise ValueError("Expected an explicit version-one geometry design-space schema")
    old = DesignSpace.from_dict(source)
    reference = old.reference_parameters
    supplied = {} if lock_angle_bounds is None else lock_angle_bounds
    if not isinstance(supplied, Mapping) or set(supplied) - set(JOINT_GROUP_NAMES):
        raise ValueError("Lock-angle bounds must map known joint group names to pairs")
    bounds = []
    for name, role in zip(JOINT_GROUP_NAMES, ("hip_roll", "hip_pitch", "knee") * 2):
        limits = reference.get(role + "_range")
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(f"Source reference lacks physical {role} range")
        lower, upper = limits
        if role == "hip_roll":
            lower, upper = max(lower, -upper), min(upper, -lower)
        bounds.append(supplied.get(name, (lower, upper)))
    target = DesignSpace(old.parameters, old.reference_json, version=2,
                         lock_angle_bounds=tuple(bounds))
    return target


def _validate_and_expand_state(state, cfg, groups, schema, role, layout):
    if not isinstance(cfg, dict) or cfg.get("class_name") != MODEL_CLASS or cfg.get("conditioning_dims", 7) != 7 or cfg.get("joint_topology", False) is not False:
        raise ValueError(f"{role} requires a seven-input ConditionedMLPModel")
    if not isinstance(state, dict):
        raise ValueError(f"Missing {role} controller state")
    width = sum(schema["observation_shapes"][group][0] for group in groups[role])
    if width <= 7:
        raise ValueError(f"{role} requires ordinary observations before geometry")
    indices = [i if i >= 0 else width + i for i in layout]
    if indices != list(range(width - 7, width)):
        raise ValueError(f"{role} requires seven trailing geometry inputs")
    observation = {group: torch.zeros((1, *shape)) for group, shape in schema["observation_shapes"].items()}
    model_cfg = copy.deepcopy(cfg)
    model_cfg.pop("class_name")
    try:
        source = ConditionedMLPModel(observation, groups, role, 12 if role == "actor" else 1,
                                     **model_cfg)
        source.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} state does not match declared model and observation widths") from exc
    weight = state["mlp.0.weight"]
    if weight.ndim != 2 or weight.shape[1] != width:
        raise ValueError(f"{role} first-layer input width differs from schema")
    expanded = dict(state)
    expanded["mlp.0.weight"] = torch.cat((weight, weight.new_zeros((weight.shape[0], 12))), -1)
    if cfg.get("obs_normalization"):
        for field in ("_mean", "_var", "_std"):
            key = "obs_normalizer." + field
            value = state[key]
            if value.shape != (1, width):
                raise ValueError(f"{role} normalizer width differs from schema")
            extra = value.new_zeros((1, 12)) if field == "_mean" else value.new_ones((1, 12))
            expanded[key] = torch.cat((value, extra), -1)
    return expanded, width


def expand_checkpoint_payload(payload, *, lock_angle_bounds=None, source_provenance=None):
    """Return a native warm-start payload and conversion report, without mutation."""
    transfer, states, source_iteration, source_kind = _source_parts(payload)
    target_space = _target_space(transfer.get("space_schema"), lock_angle_bounds)
    schema = transfer.get("control_schema")
    groups = transfer.get("obs_groups")
    layouts = transfer.get("conditioning_layout")
    if (not isinstance(schema, dict) or schema.get("num_actions") != 12
            or not isinstance(schema.get("observation_shapes"), dict)):
        raise ValueError("Source requires explicit 12-action controller schema")
    if (not isinstance(groups, dict) or set(groups) != set(ROLES)
            or not isinstance(layouts, dict) or set(layouts) != set(ROLES)):
        raise ValueError("Source requires explicit actor/critic observation groups and layouts")
    shapes = schema["observation_shapes"]
    if not all(isinstance(shape, (list, tuple)) and len(shape) == 1 and
               type(shape[0]) is int and shape[0] > 0 for shape in shapes.values()):
        raise ValueError("Controller observations must have flat positive widths")
    if not all(isinstance(groups[role], (list, tuple)) and groups[role] and
               all(isinstance(group, str) and group in shapes for group in groups[role]) and
               isinstance(layouts[role], (list, tuple)) and
               all(type(index) is int for index in layouts[role]) for role in ROLES):
        raise ValueError("Invalid actor/critic observation groups or layouts")
    if source_iteration is not None and (type(source_iteration) is not int or source_iteration < 0):
        raise ValueError("Source iteration must be a nonnegative integer")
    expanded = {}
    widths = {}
    for role in ROLES:
        expanded[role], widths[role] = _validate_and_expand_state(
            states[role], transfer.get(role + "_cfg"), groups, schema, role, layouts[role])
    target_schema = copy.deepcopy(schema)
    target_schema["observation_shapes"] = {group: list(shape) for group, shape in shapes.items()}
    for group in {groups[role][-1] for role in ROLES}:
        target_schema["observation_shapes"][group][0] += 12
    for role in ROLES:
        target_width = sum(target_schema["observation_shapes"][group][0] for group in groups[role])
        if target_width != widths[role] + 12:
            raise ValueError(f"{role} observation groups expand by more than twelve topology inputs")
    target_transfer = {key: copy.deepcopy(transfer[key]) for key in
                       ("actor_cfg", "critic_cfg", "obs_groups")}
    target_transfer.update(version=1, space_schema=target_space.to_dict(),
                           control_schema=target_schema,
                           conditioning_layout={role: list(range(widths[role] - 7, widths[role] + 12))
                                                for role in ROLES})
    for role in ROLES:
        target_transfer[role + "_cfg"].update(conditioning_dims=19, joint_topology=True)
    target_observation = {group: torch.zeros((1, *shape))
                          for group, shape in target_schema["observation_shapes"].items()}
    for role in ROLES:
        model_cfg = copy.deepcopy(target_transfer[role + "_cfg"])
        model_cfg.pop("class_name")
        try:
            target_model = ConditionedMLPModel(target_observation, groups, role,
                                               12 if role == "actor" else 1, **model_cfg)
            target_model.load_state_dict(expanded[role], strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"Expanded {role} state does not match target model and schema") from exc
    report = dict(conversion_type="geometry7_to_topology19_controller_warmstart",
                  source_format=source_kind, source_iteration=source_iteration,
                  input_shapes={role: widths[role] for role in ROLES},
                  output_shapes={role: widths[role] + 12 for role in ROLES},
                  source_compiler_fingerprint=transfer.get("compiler_fingerprint"))
    if source_provenance:
        report.update(source_provenance)
    result = dict(actor_state_dict=expanded["actor"], critic_state_dict=expanded["critic"],
                  iter=source_iteration or 0,
                  infos=dict(codesign_transfer=target_transfer,
                             checkpoint_conversion=copy.deepcopy(report)))
    # Compiler fingerprints refer to geometry-only compilation. The topology
    # compiler must be checked by the target run, rather than inferred here.
    return result, report


def expand_topology_checkpoint(input_path, output_path, *, lock_angle_bounds=None):
    """Convert a CPU-loaded checkpoint atomically, refusing existing outputs."""
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source == destination or destination.exists():
        raise FileExistsError(f"Conversion output already exists: {destination}")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    converted, report = expand_checkpoint_payload(
        payload, lock_angle_bounds=lock_angle_bounds,
        source_provenance=dict(source_path=str(source), source_sha256=digest))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".topology-conversion-", suffix=".pt", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(converted, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report
