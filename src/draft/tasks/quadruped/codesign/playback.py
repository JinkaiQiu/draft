"""Reconstruct and play the selected design from a native co-design checkpoint."""

from pathlib import Path

import torch

from draft.codesign.compiler import DesignCompiler
from draft.codesign import get_runner_class
from draft.codesign.evaluation import selected_design
from draft.codesign.runner import CodesignRunner
from draft.codesign.space import DesignSpace

from .env import build_env


def _checkpoint_path(cfg) -> Path:
    if cfg.checkpoint:
        if cfg.checkpoint == "zeros":
            raise ValueError("Co-design playback requires a joint checkpoint with a controller")
        path = Path(cfg.checkpoint)
    else:
        root = Path(cfg.log_dir) / cfg.run_name
        runs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime) if root.exists() else []
        if not runs:
            raise FileNotFoundError(f"No co-design runs found under {root}")
        final = runs[-1] / "model_final.pt"
        latest = runs[-1] / "model_latest.pt"
        checkpoints = list(runs[-1].glob("model_[0-9]*.pt"))
        checkpoints.extend(p for p in (final, latest) if p.is_file())
        path = max(checkpoints, key=lambda p: p.stat().st_mtime, default=None)
        if path is None:
            raise FileNotFoundError(f"No co-design checkpoints found under {runs[-1]}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def play_checkpoint(cfg) -> None:
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    path = _checkpoint_path(cfg)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != CodesignRunner.CHECKPOINT_VERSION:
        raise ValueError("Playback requires a native co-design joint checkpoint")
    if not checkpoint.get("control_initialized"):
        raise ValueError("The checkpoint contains no trained locomotion controller")
    space = DesignSpace.from_dict(checkpoint["space_schema"])
    config = {**checkpoint["config"], "device": cfg.device, "logger": "none"}
    factory = build_env
    if config.get("environment_family") == "quadruped_native":
        from draft.tasks.quadruped.dispatch import native_codesign_factory
        factory = native_codesign_factory(config["env_kwargs"]["robot_xml"])
    with DesignCompiler(space, cfg.cache_dir, workers=cfg.compiler_workers,
                        max_cache_entries=cfg.max_cache_entries) as compiler:
        runner = get_runner_class(config)(None, compiler, factory, config)
        runner.load(path)
        result, _, physical = selected_design(runner)
        if result.design is None:
            raise ValueError(f"Selected design is infeasible: {result.error}")
        print(f"Checkpoint: {path}\nSelected geometry: {dict(zip(space.names, physical.tolist()))}")
        env = factory([result.design], cfg.play_n_envs, device=cfg.device,
                        episode_length_s=config["episode_length_s"], seed=cfg.seed,
                        env_kwargs=config["env_kwargs"], reward_overrides=config["reward_overrides"])
        try:
            runner.policy.eval()
            viewer = {"native": NativeMujocoViewer, "viser": ViserPlayViewer}[cfg.viewer]
            viewer(env, runner.policy.control.deterministic).run()
        finally:
            env.close()
            runner.close()
