"""Fixed and randomized design cohorts for the native quadruped learner."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .compiler import CompiledDesign, DesignCompiler
from .space import DesignSpace


@dataclass(frozen=True)
class ConditioningSamplingCfg:
    mode: str = "fixed"
    num_designs: int = 1
    seed: int = 0
    latent_std: float = 0.5
    cache_dir: str = "generated/codesign"
    compiler_workers: int = 0
    max_attempts: int = 128

    @classmethod
    def parse(cls, value):
        if value is True:
            return cls(), DesignSpace.quadruped()
        if not isinstance(value, dict):
            raise ValueError("morphology_conditioning must be true, false, or a sampling mapping")
        options = dict(value)
        schema = options.pop("space_schema", None)
        if options.get("mode", "fixed") == "randomized":
            options.setdefault("num_designs", 16)
        config = cls(**options)
        if config.mode not in ("fixed", "randomized"):
            raise ValueError("Conditioning mode must be fixed or randomized")
        for name, minimum in (("num_designs", 1), ("compiler_workers", 0), ("max_attempts", 1)):
            number = getattr(config, name)
            if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
            raise ValueError("Conditioning seed must be a nonnegative integer")
        if not np.isfinite(config.latent_std) or config.latent_std <= 0:
            raise ValueError("latent_std must be finite and positive")
        if config.mode == "fixed" and config.num_designs != 1:
            raise ValueError("Fixed conditioning is the one-design special case")
        if config.max_attempts < config.num_designs:
            raise ValueError("max_attempts must be at least num_designs")
        space = DesignSpace.from_dict(schema) if schema is not None else DesignSpace.quadruped()
        return config, space


def sample_designs(config, space, reference):
    """Keep the supplied stock robot exactly; draw independent feasible variants.

    Randomized conditioning trains a controller over an immutable D-design
    cohort. Feasibility rejection is logged and bounded; this sampler does not
    estimate hardware-policy gradients. Joint hardware optimization retains its
    separate rejected-draw accounting in CodesignRunner.
    """
    from draft.tasks.quadruped.quadruped_entity import measure_quadruped_leg_length

    path = Path(reference["robot_xml"])
    parameters = yaml.safe_load(Path(reference["parameters_path"]).read_text())
    stock = CompiledDesign(reference["robot_xml_sha256"], path, parameters,
                           tuple(reference["normalized"]), tuple(reference["physical"]),
                           measure_quadruped_leg_length(path))
    if config.mode == "fixed":
        return [stock], None, []
    compiler = DesignCompiler(space, config.cache_dir, workers=config.compiler_workers,
                              max_cache_entries=max(128, config.num_designs))
    generator = np.random.default_rng(config.seed)
    normalized_reference = space.normalize(reference["physical"])
    if (np.abs(normalized_reference) >= 1.).any():
        compiler.close()
        raise ValueError("The reference design must lie strictly inside the sampling space")
    mean = np.arctanh(normalized_reference)
    designs, rejected, attempts = [], [], 0
    try:
        while len(designs) < config.num_designs and attempts < config.max_attempts:
            count = min(config.num_designs - len(designs), config.max_attempts - attempts)
            raw = generator.normal(mean, config.latent_std, size=(count, space.dim))
            candidates = [d.physical for d in designs] + list(space.denormalize(np.tanh(raw)))
            results = compiler.compile_many(candidates)[len(designs):]
            attempts += count
            for result in results:
                if result.design is None:
                    rejected.append(dict(design_id=result.design_id, error=result.error))
                else:
                    designs.append(result.design)
        if len(designs) != config.num_designs:
            raise RuntimeError(f"Only {len(designs)}/{config.num_designs} feasible designs after {attempts} draws")
        # Renew the cache lease for the COMPLETE cohort after rejection rounds.
        designs = [result.design for result in compiler.compile_many([d.physical for d in designs])]
        return designs, compiler, rejected
    except BaseException:
        compiler.close()
        raise


def make_conditioned_cfg(morphology_conditioning, *, env_cfg_factory=None, **base):
    """Sample designs independently of a task's environment-authoring factory."""
    from draft.tasks.quadruped.conditioning import describe_morphology
    from draft.tasks.quadruped.codesign.native_env import make_native_env_cfg

    config, space = ConditioningSamplingCfg.parse(morphology_conditioning)
    reference = describe_morphology(base["robot_xml"], space=space)
    count = base.get("n_envs", 4096)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count % config.num_designs:
        raise ValueError("n_envs must be positive and divisible by num_designs (D times R)")
    replicas = count // config.num_designs
    designs, compiler, rejected = sample_designs(config, space, reference)
    try:
        options = {key: value for key, value in base.items() if key not in ("robot_xml", "n_envs")}
        task_reference = str(Path(base.get("task_geometry_xml") or reference["robot_xml"]).resolve())
        cfg = make_native_env_cfg(designs, replicas, reference_xml=task_reference,
                                  env_cfg_factory=env_cfg_factory, **options)
        cfg.morphology_conditioning_metadata = dict(
            **reference, sampling_mode=config.mode, sampling_seed=config.seed,
            num_designs=len(designs), replicas=replicas, layout="design-major D x R",
            resampling="one immutable cohort per training run", reference_task_geometry=task_reference,
            designs=[dict(design_id=d.design_id, xml_path=str(d.xml_path),
                          physical=list(d.physical), normalized=list(d.morphology)) for d in designs],
            rejected_draws=rejected)
        if getattr(cfg, "hardware_aware", False):
            cfg.morphology_conditioning_metadata.update(
                hardware_aware=True, nominal_stance_heights=list(cfg.hardware_nominal_heights))
        cfg._conditioning_compiler = compiler
        return cfg
    except BaseException:
        if compiler is not None:
            compiler.close()
        raise
