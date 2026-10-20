"""Frozen-checkpoint, seed-paired full-episode comparisons on the native task."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .evaluation import evaluate


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def task_contract(config):
    """Exclude training-only morphology sampling from the common task contract."""
    codesign = config.get("codesign") or {}
    return dict(task=config.get("task", "quadruped"),
                episode_length_s=config.get("episode_length_s", config.get("max_episode_length", 20.)),
                horizon_steps=config.get("horizon_steps", codesign.get("horizon_steps")),
                env_kwargs={key: value for key, value in config.get("env_kwargs", {}).items()
                            if key != "morphology_conditioning"},
                reward_overrides=config.get("reward_overrides", {}),
                gamma=config.get("gamma", .99),
                terminal_penalty=config.get("terminal_penalty", codesign.get("terminal_penalty", -10.)))


def controller_contract(config):
    native = config.get("controller_train_cfg", {})
    return {key: native.get(key) for key in ("actor", "critic", "algorithm", "obs_groups", "num_steps_per_env")} | {
        "normalization_exclusions": config.get("normalization_exclusions")}


def paired_statistics(left, right, *, bootstrap_samples=10000, bootstrap_seed=731):
    """Cluster-bootstrap seed means, preserving all condition/replica pairs.

    Replicas share terrain generation and RNG streams within a seed. Treating
    every replica as an independent bootstrap unit would overstate precision.
    """
    if set(left) != set(right) or len(left) < 2:
        raise ValueError("Paired statistics require the same two or more seed clusters")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    differences = []
    for seed in sorted(left):
        a, b = np.asarray(left[seed], dtype=float), np.asarray(right[seed], dtype=float)
        if a.shape != b.shape or a.ndim != 1 or not a.size or not np.isfinite([a, b]).all():
            raise ValueError("Paired trials must have equal finite one-dimensional replica arrays")
        differences.append(float((a - b).mean()))
    differences = np.asarray(differences)
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.choice(differences, size=(bootstrap_samples, len(differences))).mean(axis=1)
    return dict(mean_difference=float(differences.mean()),
                ci95_seed_cluster_bootstrap=np.quantile(draws, [.025, .975]).tolist(),
                seed_mean_differences=differences.tolist(), seed_clusters=len(differences),
                bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
                uncertainty="conditional on these trained checkpoints; seed-cluster percentile bootstrap")


def _frozen_state(policy):
    return {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}


def _verify_frozen(policy, before):
    if any(not torch.equal(before[key], value.detach().cpu()) for key, value in policy.state_dict().items()):
        raise RuntimeError("Evaluation changed policy parameters or normalization buffers")


def _native_runner(checkpoint, reference, compiler, factory, device):
    from .runner import CodesignRunner
    config = dict(reference, device=device, logger="none")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "policy" in data:
        from . import get_runner_class
        if task_contract(data["config"]) != task_contract(reference):
            raise ValueError("Joint baseline checkpoint changes the evaluation task")
        runner = get_runner_class(data["config"])(None, compiler, factory,
                    dict(data["config"], device=device, logger="none"))
        runner.load(checkpoint)
        return runner
    transfer = data.get("infos", {}).get("codesign_transfer", {})
    schema = transfer.get("control_schema")
    if not schema:
        raise ValueError("Native baseline checkpoint lacks an explicit controller transfer schema")
    runner = CodesignRunner(None, compiler, factory, config)
    runner._ensure_control({key: torch.zeros((1, *shape), device=device)
                            for key, shape in schema["observation_shapes"].items()}, schema["num_actions"])
    runner.warmstart(checkpoint)
    return runner


def compare_checkpoints(*, joint_checkpoint, baseline_checkpoint, baseline_config, output,
                        source_checkpoint=None, seeds=tuple(range(2_000_000, 2_000_008)),
                        replicas=128, device="cuda:0", expected_horizon=1045, frozen_protocol=None):
    """Write the immutable protocol first, then individual records and summary.

    Paths are explicit final checkpoints; this function never ranks/selects a
    checkpoint. Existing output directories are rejected to preserve evidence.
    """
    from . import get_runner_class
    from .compiler import DesignCompiler
    from .runner import CodesignRunner
    from .space import DesignSpace
    from draft.tasks.quadruped.dispatch import native_codesign_factory

    if len(set(seeds)) != len(seeds) or len(seeds) < 2 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Use at least two unique nonnegative integer held-out seeds")
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = dict(joint=Path(joint_checkpoint).resolve(), baseline=Path(baseline_checkpoint).resolve())
    if source_checkpoint:
        paths["source"] = Path(source_checkpoint).resolve()
    hashes = {name: dict(path=str(path), sha256=_sha(path)) for name, path in paths.items()}
    data = torch.load(paths["joint"], map_location="cpu", weights_only=False)
    if data.get("version") != CodesignRunner.CHECKPOINT_VERSION or not data.get("control_initialized"):
        raise ValueError("Comparison requires a trained joint co-design checkpoint")
    config = data["config"]
    if config.get("environment_family") != "quadruped_native":
        raise ValueError("Comparison requires the authored native quadruped task")
    baseline_config = Path(baseline_config).resolve()
    baseline_yaml = yaml.safe_load(baseline_config.read_text())
    if task_contract(baseline_yaml) != task_contract(config):
        raise ValueError("Baseline run configuration changes the evaluation task")
    space = DesignSpace.from_dict(data["space_schema"])
    conditions = dict(baseline_stock=("baseline", space.center),
                      codesign_learned=("joint", None), codesign_stock=("joint", space.center))
    if source_checkpoint:
        conditions["source_stock"] = ("source", space.center)
    protocol = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoints=hashes, checkpoint_selection="explicit frozen final checkpoint; no evaluation-based selection",
        seeds=list(seeds), replicas_per_seed=replicas, trials_per_condition=len(seeds) * replicas,
        conditions={name: dict(controller=controller, geometry="decoded hardware mean" if physical is None else "stock center")
                    for name, (controller, physical) in conditions.items()},
        task=task_contract(config), baseline_config=dict(path=str(baseline_config), sha256=_sha(baseline_config)),
        deterministic_actions=True, frozen_normalizers=True, full_episode=True,
        expected_horizon_steps=expected_horizon, terminal_penalty=config["terminal_penalty"], gamma=config["gamma"],
        value_bootstrap=False, first_done_mask=True, auto_reset=False,
        uncertainty_unit="seed cluster; paired differences within matching seed and replica",
        primary_comparison="codesign_learned minus baseline_stock",
        geometry_ablation="codesign_learned minus codesign_stock",
        evaluation_source_sha256={p.name: _sha(p) for p in (Path(__file__), Path(__file__).with_name("evaluation.py"))},
        training_metadata=data.get("metadata", {}))
    source_root = Path(__file__).resolve().parents[1]
    sources = [*Path(__file__).parent.glob("*.py"),
               *(source_root / "tasks/quadruped").rglob("*.py")]
    protocol["current_source_sha256"] = {str(path.relative_to(source_root)): _sha(path) for path in sorted(sources)}
    protocol["runtime_versions"] = {"torch": torch.__version__}
    for package in ("mjlab", "rsl-rl-lib", "mujoco", "mujoco-warp", "warp-lang", "numpy"):
        try:
            protocol["runtime_versions"][package] = version(package)
        except PackageNotFoundError:
            protocol["runtime_versions"][package] = None
    if frozen_protocol:
        protocol["frozen_protocol"] = dict(path=str(Path(frozen_protocol).resolve()), sha256=_sha(frozen_protocol))
    _write(output / "protocol.json", protocol)
    try:
        with ExitStack() as stack:
            compiler = stack.enter_context(DesignCompiler(space, output / "compiled", workers=0, max_cache_entries=8))
            factory = native_codesign_factory(config["env_kwargs"]["robot_xml"],
                                              horizon_steps=config.get("horizon_steps"))
            runners = {}
            runners["joint"] = get_runner_class(config)(None, compiler, factory, dict(config, device=device, logger="none"))
            stack.callback(runners["joint"].close)
            runners["joint"].load(paths["joint"])
            for name in paths.keys() - {"joint"}:
                runners[name] = _native_runner(paths[name], config, compiler, factory, device)
                stack.callback(runners[name].close)
            states = {name: _frozen_state(runner.policy) for name, runner in runners.items()}
            records = {name: {} for name in conditions}
            for seed in seeds:
                expected_audit = None
                for name, (controller, physical) in conditions.items():
                    print(f"Evaluating {name}: seed={seed}, replicas={replicas}", flush=True)
                    report = evaluate(runners[controller], seed=seed, replicas=replicas, physical=physical,
                                      include_trials=True, paired=True, record_video=False)
                    if not report["valid"] or report["horizon_steps"] != expected_horizon:
                        raise ValueError("Invalid geometry or full-trial horizon differs from the frozen protocol")
                    report["role"] = "held-out final comparison; no checkpoint selection"
                    _write(output / f"{name}_seed_{seed}.json", report)
                    audit = report["pairing_audit"]
                    if expected_audit is not None and audit != expected_audit:
                        raise ValueError(f"Seed {seed}: terrain, command latents or RNG stream differs across conditions")
                    expected_audit = audit
                    records[name][seed] = report
                    _verify_frozen(runners[controller].policy, states[controller])
            summary = summarize(records)
            summary["pairing_audit_passed"] = True
            summary["normalizers_and_parameters_unchanged"] = True
            _write(output / "summary.json", summary)
            _write(output / "comparison.json", summary)
        if any(_sha(paths[name]) != item["sha256"] for name, item in hashes.items()):
            raise RuntimeError("A checkpoint changed during evaluation")
        _write(output / "status.json", dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat()))
        return summary
    except BaseException as exc:
        _write(output / "status.json", dict(status="failed", error=f"{type(exc).__name__}: {exc}"))
        raise


def compare_protocol(protocol, baseline_checkpoint, output, *, device="cuda:0"):
    """Validate the user's frozen protocol before allocating an evaluation env."""
    protocol = Path(protocol).resolve()
    plan = json.loads(protocol.read_text())
    for path_key, hash_key in (("codesign_checkpoint", "codesign_sha256"),
                               ("source_checkpoint", "source_sha256"),
                               ("baseline_config", "baseline_config_sha256")):
        if _sha(plan[path_key]) != plan[hash_key]:
            raise ValueError(f"Frozen protocol hash mismatch: {path_key}")
    evaluation = plan["evaluation"]
    required = dict(full_trial=True, bootstrap_at_cutoff=False, deterministic_controller=True)
    if any(evaluation.get(key) != value for key, value in required.items()):
        raise ValueError("Protocol must require deterministic full trials without value bootstrap")
    if evaluation["trials_per_condition"] != len(evaluation["seeds"]) * evaluation["replicas_per_seed"]:
        raise ValueError("Frozen protocol trial count is inconsistent")
    budget = plan["matched_budget"]
    checkpoints = {"codesign": plan["codesign_checkpoint"], "baseline": baseline_checkpoint}
    receipts = {}
    reference_controller = reference_task = None
    for name, checkpoint in checkpoints.items():
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = data["config"]
        cycles = data.get("alternating_state", {}).get("controller_cycles")
        trainer = data.get("alternating_state", {}).get("controller_trainer", {})
        steps = config.get("controller_updates_per_cycle")
        checks = {"cycles": cycles, "controller_updates_per_cycle": steps,
                  "controller_updates": trainer.get("iteration"),
                  "controller_rollout_steps": config.get("controller_train_cfg", {}).get("num_steps_per_env"),
                  "design_rows": config.get("num_designs"), "replicas": config.get("replicas"),
                  "worlds": config.get("num_designs", 0) * config.get("replicas", 0),
                  "controller_transitions": trainer.get("total_transitions"),
                  "hardware_aware": config.get("env_kwargs", {}).get("hardware_aware", False)}
        if any(checks[key] != budget[key] for key in checks):
            raise ValueError(f"{name} checkpoint differs from frozen matched controller budget: {checks}")
        if data["iteration"] != budget["cycles"] or config["seed"] != plan["training_seed"]:
            raise ValueError(f"{name} checkpoint is not the final matched-budget training seed")
        if name == "baseline" and (config["hardware_sampling_mode"] != "fixed" or config["optimize_hardware"]):
            raise ValueError("Baseline must have trained only fixed stock hardware")
        offset = 1_000_003 if config.get("training_schedule") == "alternating" else config.get("training_seed_offset")
        if offset != budget["training_seed_offset"]:
            raise ValueError(f"{name} controller training environment seed offset differs")
        if reference_controller is None:
            reference_controller, reference_task = controller_contract(config), task_contract(config)
        elif controller_contract(config) != reference_controller or task_contract(config) != reference_task:
            raise ValueError("Baseline and co-design native controller settings or task differ")
        warmstart = data.get("metadata", {}).get("warmstart", {})
        if warmstart.get("sha256") != plan["source_sha256"]:
            raise ValueError(f"{name} did not start from the frozen source checkpoint")
        receipts[name] = dict(checks,
            hardware_evaluation_transitions=data.get("alternating_state", {}).get("hardware_evaluation_transitions"),
            all_transitions=data["global_step"], hardware_bootstrap=config.get("hardware_bootstrap"),
            hardware_evaluation_steps=config.get("hardware_evaluation_steps"))
    result = compare_checkpoints(joint_checkpoint=plan["codesign_checkpoint"],
        baseline_checkpoint=baseline_checkpoint, baseline_config=plan["baseline_config"],
        source_checkpoint=plan["source_checkpoint"], output=output,
        seeds=evaluation["seeds"], replicas=evaluation["replicas_per_seed"], device=device,
        expected_horizon=evaluation.get("horizon_steps", 1045), frozen_protocol=protocol)
    result["matched_budget_verified"] = receipts
    result["budget_interpretation"] = "Matched controller training updates and transitions; co-design additionally consumed hardware-evaluation transitions"
    result["condition_labels"] = dict(baseline_stock="stock_continuation_on_stock",
        codesign_learned="codesign_controller_on_learned_mean",
        codesign_stock="codesign_controller_on_stock", source_stock="source_999_on_stock")
    _write(Path(output) / "comparison.json", result)
    return result


def summarize(records):
    metrics = ("return", "discounted_return", "survived", "duration_steps")
    arrays = {name: {metric: {seed: [trial[metric] for trial in report["trials"]]
                    for seed, report in seeds.items()} for metric in metrics}
              for name, seeds in records.items()}
    summaries = {name: {metric: dict(mean=float(np.concatenate(list(values.values())).mean()),
                                    std=float(np.concatenate(list(values.values())).std()))
                       for metric, values in data.items()} for name, data in arrays.items()}
    for name, values in records.items():
        reports = list(values.values())
        durations = sum(trial["duration_steps"] for report in reports for trial in report["trials"])
        count = sum(len(report["trials"]) for report in reports)
        summaries[name].update(return_mean=summaries[name]["return"]["mean"],
            return_std=summaries[name]["return"]["std"],
            survival_fraction=summaries[name]["survived"]["mean"],
            active_duration_fraction=durations / (count * reports[0]["horizon_steps"]))
        for key, metric in (("velocity_tracking_squared_error", "velocity_tracking_rmse_m_s"),
                            ("yaw_tracking_squared_error", "yaw_tracking_rmse_rad_s")):
            errors = [trial[key] for report in reports for trial in report["trials"]]
            summaries[name][metric] = float(np.sqrt(sum(errors) / durations)) if all(x is not None for x in errors) else None
    pairs = dict(primary=("codesign_learned", "baseline_stock"),
                 geometry_ablation=("codesign_learned", "codesign_stock"))
    if "source_stock" in records:
        pairs["fixed_training_gain"] = ("baseline_stock", "source_stock")
    comparisons = {name: dict(left=left, right=right, metrics={metric: paired_statistics(
                    arrays[left][metric], arrays[right][metric]) for metric in metrics})
                   for name, (left, right) in pairs.items()}
    return dict(conditions=summaries, paired_comparisons=comparisons,
                caveat="Simulation-seed uncertainty only; one training run per method does not estimate training-seed variability")
