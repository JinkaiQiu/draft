"""Train a PPO policy for a task in ``src/draft/tasks/tasks.yaml``.

The robot comes from the run config's ``env_kwargs.robot_xml``, not a flag.

  python scripts/train.py --task quadruped
  python scripts/train.py --task quadruped_terrain
  python scripts/train.py --config generated/lineup_configs/stage1/cheetah_flat_s0.yaml
"""

from __future__ import annotations

import os
import sys

# Set before any mujoco import: headless Linux needs EGL (absent on macOS).
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Friendly error when the `rl` extra (mjlab, torch) is missing.
def _need_rl_extra(exc: ModuleNotFoundError) -> "NoReturn":   # noqa: F821
    raise SystemExit(
        f"{exc.name} is not installed.\n\n"
        "Training and playback need the RL extra, on a Linux machine with a "
        "CUDA GPU:\n    pip install -e \".[rl]\" -c constraints.txt\n\n"
        "Everything else in this repository — generating a design, the editor, "
        "the twins,\nevery figure and every number the paper reports — runs "
        "without it.")


import draft  # noqa: E402,F401  before mjlab/torch: loads the environment's libstdc++

try:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
except ModuleNotFoundError as exc:
    _need_rl_extra(exc)

from draft.tasks import get_task, list_tasks
from draft.tasks.config_loader import load_run_config, require_robot_xml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GPU-accelerated RL policy from a registered task."
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help=f"Task name. Available: {', '.join(list_tasks())}",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Per-run YAML. Defaults to <task_path>/config.yaml.",
    )
    parser.add_argument(
        "--warmstart", type=str, default=None,
        help="Path to a .pt checkpoint to warm-start actor+critic weights from "
             "(overrides warmstart_checkpoint in the config). Used to seed the "
             "quadruped challenge tasks from their flat-base policy.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a .pt checkpoint to RESUME from — restores the FULL training "
             "state (actor+critic+optimizer+obs-normalizer), so training continues "
             "without the warm-start normalizer dip. Use to top-up an existing run.",
    )
    parser.add_argument("--video", action="store_true", help="Record native training rollouts headlessly.")
    parser.add_argument("--video-length", type=int, default=500, help="Environment steps per video.")
    parser.add_argument("--video-interval", type=int, default=2400, help="Environment steps between video starts.")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    return parser.parse_args()


def _apply_reward_overrides(env_cfg, overrides: dict) -> None:
    if not overrides:
        return
    valid = set(env_cfg.rewards.keys())
    for name, weight in overrides.items():
        if name not in valid:
            raise KeyError(
                f"Unknown reward term '{name}'. "
                f"Available for this task: {sorted(valid)}"
            )
        env_cfg.rewards[name].weight = float(weight)


def main() -> None:
    args = _parse_args()
    if args.video:
        if args.video_length < 1 or args.video_interval < args.video_length:
            raise SystemExit("Video length must be positive and no greater than its interval.")
        if any(value < 2 or value % 2 for value in (args.video_width, args.video_height)):
            raise SystemExit("Video dimensions must be positive even integers.")

    # ── Resolve task + config ────────────────────────────────────────────────
    if args.config is not None:
        run_cfg = load_run_config(args.config)
        task_name = args.task or run_cfg["task"]
    elif args.task is not None:
        task = get_task(args.task)
        run_cfg = load_run_config(task.default_config_yaml())
        task_name = args.task
    else:
        raise SystemExit("Must pass --task and/or --config.")

    task = get_task(task_name)

    # ── Build runner cfg with overrides ──────────────────────────────────────
    runner_cfg = task.make_runner_cfg()
    runner_cfg.apply_overrides(run_cfg["runner_overrides"])
    if args.warmstart is not None:
        runner_cfg.warmstart_checkpoint = args.warmstart
    runner_cfg.env_kwargs = {**runner_cfg.env_kwargs, **run_cfg["env_kwargs"]}
    require_robot_xml(runner_cfg.env_kwargs)
    runner_cfg.reward_overrides = {
        **runner_cfg.reward_overrides, **run_cfg["reward_overrides"]
    }

    import torch
    if runner_cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("Training needs a CUDA GPU and none was found. "
                         "It does not run on macOS or on a CPU.")

    # ── Log dir ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(runner_cfg.log_dir) / runner_cfg.run_name / f"{timestamp}_{runner_cfg.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training run : {runner_cfg.run_name}")
    print(f"  Task       : {task_name}")
    if runner_cfg.env_kwargs:
        for k, v in runner_cfg.env_kwargs.items():
            print(f"  {k:<10} : {v}")
    print(f"  Envs       : {runner_cfg.n_envs}")
    print(f"  Device     : {runner_cfg.device}")
    print(f"  Log dir    : {log_dir}")

    # ── Run metadata (for easy sweep aggregation) ─────────────────────────────
    # Sidecar describing the run; written first so it survives interruption.
    import subprocess
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_sha = None
    run_meta = {
        "task": task_name,
        "run_name": runner_cfg.run_name,
        "config": args.config,
        "env_kwargs": runner_cfg.env_kwargs,
        "reward_overrides": runner_cfg.reward_overrides,
        "n_envs": runner_cfg.n_envs,
        "max_iterations": runner_cfg.max_iterations,
        "num_steps_per_env": runner_cfg.num_steps_per_env,
        "seed": runner_cfg.seed,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "codesign": getattr(runner_cfg, "codesign", None),
        "video": {"enabled": args.video, "length_steps": args.video_length,
                  "interval_steps": args.video_interval, "width": args.video_width, "height": args.video_height},
    }
    with open(log_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Some tasks own multiple successive robot cohorts rather than one fixed
    # environment. The ordinary script still owns task/config/device/log setup.
    training_handler = task.training_handler(runner_cfg)
    if training_handler is not None:
        training_handler(runner_cfg, log_dir=log_dir, resume=args.resume, metadata=run_meta)
        return

    # ── Build env cfg ────────────────────────────────────────────────────────
    env_cfg = task.make_env_cfg(
        n_envs=runner_cfg.n_envs,
        episode_length_s=runner_cfg.max_episode_length,
        play=False,
        **runner_cfg.env_kwargs,
    )
    _apply_reward_overrides(env_cfg, runner_cfg.reward_overrides)

    # Seed before construction so scene and policy initialization are seeded.
    env_cfg.seed = runner_cfg.seed
    if args.video:
        env_cfg.viewer.width = args.video_width
        env_cfg.viewer.height = args.video_height
        env_cfg.viewer.env_idx = 0
        env_cfg.viewer.max_extra_envs = 0

    # ── Build env + runner ───────────────────────────────────────────────────
    env = task.make_env(env_cfg, device=runner_cfg.device,
                        render_mode="rgb_array" if args.video else None)
    if args.video:
        import shutil
        import mediapy
        from mjlab.utils.wrappers import VideoRecorder
        if shutil.which("ffmpeg") is None:
            try:
                import imageio_ffmpeg
            except ImportError as exc:
                raise RuntimeError("--video requires ffmpeg on PATH or the imageio-ffmpeg package") from exc
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        env = VideoRecorder(env, video_folder=log_dir / "videos",
                            step_trigger=lambda step: step % args.video_interval == 0,
                            video_length=args.video_length)
    vec_env = RslRlVecEnvWrapper(env)

    train_cfg = runner_cfg.build_rsl_rl_cfg()
    from draft.tasks.native_runner import MetadataOnPolicyRunner, conditioning_transfer_metadata
    checkpoint_metadata = conditioning_transfer_metadata(vec_env, train_cfg)
    if checkpoint_metadata:
        (log_dir / "conditioning.json").write_text(json.dumps(checkpoint_metadata, indent=2) + "\n")
    runner = MetadataOnPolicyRunner(
        vec_env, train_cfg, log_dir=str(log_dir), device=runner_cfg.device,
        checkpoint_metadata=checkpoint_metadata)

    if args.resume:
        # Full-state resume (actor, critic, optimizer, obs normalizer).
        print(f"  Resume     : {args.resume}")
        runner.load(args.resume, load_cfg=None)
    else:
        warmstart = getattr(runner_cfg, "warmstart_checkpoint", None)
        if warmstart:
            print(f"  Warmstart  : {warmstart}")
            runner.load(warmstart, load_cfg={"actor": True, "critic": True}, strict=False)
            runner.current_learning_iteration = 0
            # runner.load restores the env step counter; a warm start enters a
            # new task whose curriculum must start from zero, else it opens
            # past the end of its ramp. (--resume keeps the counter.)
            vec_env.unwrapped.common_step_counter = 0
            import hashlib
            source = Path(warmstart).resolve()
            runner.checkpoint_metadata["warmstart"] = dict(
                checkpoint=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                scope="actor, critic and normalization; fresh optimizer and training counters")

    if checkpoint_metadata:
        runner.save(str(log_dir / "model_initial.pt"), infos={"stage": "before learning"})

    try:
        runner.learn(num_learning_iterations=runner_cfg.max_iterations)
    finally:
        vec_env.close()


if __name__ == "__main__":
    main()
