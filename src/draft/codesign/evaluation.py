"""Deterministic selected-design trials and self-contained controller exports."""

from contextlib import contextmanager, ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch
import yaml


@contextmanager
def preserved_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3]:
            torch.cuda.set_rng_state_all(state[3])


def selected_design(runner):
    normalized = runner.policy.hardware.decode(runner.policy.hardware.mean).cpu().numpy()
    physical = runner.compiler.space.denormalize(normalized)
    with preserved_rng():
        # The compiler verifies cache hashes and renews its live asset lease.
        result = runner.compiler.compile_many([physical])[0]
    return result, normalized, physical


def _tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        # mjlab TorchArray proxies delegate detach()/cpu() to the actual tensor.
        if isinstance(value, torch.Tensor) or callable(getattr(value, "detach", None)):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class PairingAudit:
    """Keep terminated worlds outside RNG-consuming resets during paired trials.

    mjlab normally requires a manual reset before the next step when auto-reset
    is disabled. Here only first episodes are scored; terminated worlds continue
    with zero actions and their observations/rewards are excluded. Clearing the
    pending-reset guard permits this evaluation-only absorbing-trial adapter.
    """

    def __init__(self, env):
        self.env = env.unwrapped
        if not hasattr(self.env, "_manual_reset_pending"):
            raise ValueError("Paired evaluation requires mjlab manual-reset support")
        self.env.cfg.auto_reset = False
        self.commands = hashlib.sha256()
        self.rng = hashlib.sha256()
        model = self.env.sim.model
        terrain = self.env.scene.terrain
        self.terrain = _tensor_digest([
            (name, getattr(model, name)) for name in ("hfield_data", "hfield_size")
            if hasattr(model, name)] + [
            (name, getattr(terrain, name)) for name in ("env_origins", "terrain_levels", "terrain_types")
            if hasattr(terrain, name)])

    def record(self):
        term = self.env.command_manager.get_term("twist")
        # Body-frame heading/world transforms depend on robot motion. Audit the
        # sampled latent targets and flags, which are the external demands.
        names = ("time_left", "command_counter", "heading_target", "vel_command_w",
                 "is_heading_env", "is_standing_env", "is_world_env", "is_forward_env")
        self.commands.update(_tensor_digest([(name, getattr(term, name)) for name in names]).encode())
        states = [("cpu", torch.get_rng_state())]
        if torch.cuda.is_available():
            states.extend((f"cuda_{i}", state) for i, state in enumerate(torch.cuda.get_rng_state_all()))
        self.rng.update(_tensor_digest(states).encode())

    def before_step(self):
        self.env._manual_reset_pending.zero_()

    def report(self):
        return dict(terrain_sha256=self.terrain, command_latents_sha256=self.commands.hexdigest(),
                    torch_rng_stream_sha256=self.rng.hexdigest(), auto_reset=False,
                    posttermination="zero actions; continue simulation; exclude all later samples")


def _tracking_errors(env):
    native = env.unwrapped
    if not hasattr(native, "scene") or not hasattr(native, "command_manager"):
        return None
    data = native.scene["robot"].data
    command = native.command_manager.get_command("twist")
    return ((data.root_link_lin_vel_b[:, :2] - command[:, :2]).square().sum(-1),
            (data.root_link_ang_vel_b[:, 2] - command[:, 2]).square())


def _video_options(runner, record_video):
    config = runner.config
    interval = config.get("eval_video_interval", 0)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError("eval_video_interval must be a nonnegative integer")
    if record_video is None:
        record_video = interval > 0 and runner.iteration % interval == 0
    if not isinstance(record_video, bool):
        raise ValueError("record_video must be a boolean or None")
    if not record_video:
        return None
    if runner.log_dir is None:
        raise ValueError("Evaluation video requires a log directory")
    options = dict(fps=config.get("eval_video_fps", 25),
                   width=config.get("eval_video_width", 640),
                   height=config.get("eval_video_height", 480),
                   length_s=config.get("eval_video_length_s", 10.0),
                   path=Path(runner.log_dir) / "videos" / f"eval_{runner.iteration:06d}.mp4")
    for name in ("fps", "width", "height"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "fps" else 2):
            raise ValueError(f"eval_video_{name} must be a positive integer")
        if name != "fps" and value % 2:
            raise ValueError(f"eval_video_{name} must be even")
    if not math.isfinite(options["length_s"]) or options["length_s"] <= 0:
        raise ValueError("eval_video_length_s must be finite and positive")
    return options


@torch.no_grad()
def evaluate(runner, *, replicas=None, seed=None, record_video=None, physical=None,
             include_trials=False, paired=False):
    from .runner import _observation

    if runner.phase != "boundary":
        raise RuntimeError("Evaluation requires a completed update boundary")
    c = runner.config
    replicas = c["eval_replicas"] if replicas is None else replicas
    seed = c["seed"] + 1_000_000 if seed is None else seed
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError("Evaluation replicas must be a positive integer")
    video_options = _video_options(runner, record_video)
    if paired and video_options is not None:
        raise ValueError("Paired first-trial evaluation cannot record auto-reset video")
    if physical is None:
        result, normalized, physical = selected_design(runner)
        selection = "decoded hardware mean"
    else:
        physical = np.asarray(physical, dtype=float)
        normalized = runner.compiler.space.normalize(physical)
        with preserved_rng():
            result = runner.compiler.compile_many([physical])[0]
        selection = "explicit physical geometry"
    report = dict(iteration=runner.iteration, seed=seed, replicas=replicas,
                  selection=selection, role="development evaluation on held-out simulation seeds",
                  normalized=normalized.tolist(), physical=physical.tolist(), design_id=result.design_id,
                  valid=result.design is not None, error=result.error)
    if result.design is None:
        return dict(report, discounted_return_mean=c["invalid_design_return"], survival_fraction=0.)
    with preserved_rng():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env_options = dict(c["env_kwargs"] or {})
        if video_options is not None:
            env_options.update(render_mode="rgb_array", render_width=video_options["width"],
                               render_height=video_options["height"])
        env = runner.env_factory([result.design], replicas, device=str(runner.device),
                                 episode_length_s=c["episode_length_s"], seed=seed,
                                 env_kwargs=env_options, reward_overrides=c["reward_overrides"])
        recording = ExitStack()
        try:
            observation = _observation(env.reset())
            audit = PairingAudit(env) if paired else None
            if audit:
                audit.record()
            runner._ensure_control(observation, env.num_actions)
            runner.policy.eval()
            active = torch.ones(replicas, dtype=torch.bool, device=runner.device)
            fell = torch.zeros_like(active)
            returns = torch.zeros(replicas, device=runner.device)
            discounted = torch.zeros_like(returns)
            durations = torch.zeros_like(returns)
            tracking = torch.zeros(2, device=runner.device)
            tracking_count = 0
            trial_tracking = torch.zeros((2, replicas), device=runner.device)
            horizon = int(env.unwrapped.horizon_steps)
            writer = None
            if video_options is not None:
                import mediapy as media
                if shutil.which("ffmpeg") is None:
                    try:
                        import imageio_ffmpeg
                    except ImportError as exc:
                        raise RuntimeError("Evaluation video requires ffmpeg on PATH or the imageio-ffmpeg package") from exc
                    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
                stride = max(1, round(1.0 / (video_options["fps"] * env.unwrapped.step_dt)))
                video_fps = 1.0 / (stride * env.unwrapped.step_dt)
                video_steps = min(horizon, math.ceil(video_options["length_s"] / env.unwrapped.step_dt))
                video_path = video_options["path"]
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = recording.enter_context(media.VideoWriter(
                    str(video_path), shape=(video_options["height"], video_options["width"]), fps=video_fps))
                report.update(video_path=str(video_path.resolve()), video_fps=video_fps, video_frames=0,
                              video_env_index=0, video_includes_autoresets=True,
                              video_trial_rewards_masked_after_first_done=True)
            for step in range(horizon):
                # One representative world shows complete repeated policy
                # attempts; trial rewards remain masked after its first done.
                # Streaming keeps video memory bounded independently of length.
                if writer is not None and step < video_steps and step % stride == 0:
                    frame = env.unwrapped.render()
                    if frame is None:
                        raise RuntimeError("Evaluation video renderer returned no frame")
                    writer.add_image(frame)
                    report["video_frames"] += 1
                errors = _tracking_errors(env)
                if errors is not None:
                    tracking += torch.stack([error[active].sum() for error in errors])
                    trial_tracking += torch.stack([torch.where(active, error, 0.) for error in errors])
                    tracking_count += int(active.sum())
                observing = active
                if writer is not None and step < video_steps:
                    observing = active.clone()
                    observing[0] = True
                safe_observation = {key: torch.where(observing.reshape(-1, *([1] * (value.ndim - 1))),
                                                      value, 0.) for key, value in observation.items()}
                if any(not value.isfinite().all() for value in safe_observation.values()):
                    raise ValueError("Nonfinite active deterministic evaluation observation")
                actions = runner.policy.control.deterministic(safe_observation)
                stepping_actions = torch.where(active[:, None], actions, torch.zeros_like(actions))
                if writer is not None and step < video_steps:
                    stepping_actions[0] = actions[0]
                if audit:
                    audit.before_step()
                observation, rewards, done, extras = env.step(stepping_actions)
                if audit:
                    audit.record()
                observation = _observation(observation)
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool().reshape(-1)
                terminated = extras.get("codesign_terminated", done.bool().reshape(-1) & ~timeout).bool().reshape(-1)
                done = done.bool().reshape(-1) | terminated
                rewards = torch.where(active, rewards.reshape(-1) + terminated * c["terminal_penalty"], 0.)
                if not torch.isfinite(rewards).all():
                    raise ValueError("Nonfinite deterministic evaluation reward")
                returns += rewards
                discounted += c["gamma"] ** step * rewards
                durations += active
                fell |= terminated & active
                active &= ~done
            report.update(return_mean=returns.mean().item(), return_std=returns.std(unbiased=False).item(),
                          discounted_return_mean=discounted.mean().item(), survival_fraction=(~fell).float().mean().item(),
                          active_duration_fraction=(durations / horizon).mean().item(), horizon_steps=horizon,
                          velocity_tracking_rmse_m_s=(tracking[0] / tracking_count).sqrt().item() if tracking_count else None,
                          yaw_tracking_rmse_rad_s=(tracking[1] / tracking_count).sqrt().item() if tracking_count else None)
            if audit:
                report["pairing_audit"] = audit.report()
            if include_trials:
                report["trials"] = [dict(replica=i, **{"return": float(returns[i])},
                    discounted_return=float(discounted[i]), survived=not bool(fell[i]),
                    duration_steps=int(durations[i]),
                    velocity_tracking_squared_error=float(trial_tracking[0, i]) if tracking_count else None,
                    yaw_tracking_squared_error=float(trial_tracking[1, i]) if tracking_count else None)
                    for i in range(replicas)]
        finally:
            try:
                recording.close()
            finally:
                env.close()
                del env
                gc.collect()
    return report


def export(runner, destination):
    if runner.phase != "boundary" or not runner.control_initialized:
        raise RuntimeError("Export requires an initialized controller at a completed update boundary")
    result, normalized, physical = selected_design(runner)
    if result.design is None:
        raise ValueError(f"Decoded hardware mean is infeasible: {result.error}")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        shutil.copytree(result.design.xml_path.parent, stage / "robot")
        (stage / "selected_parameters.yaml").write_text(yaml.safe_dump(result.design.parameters, sort_keys=False))
        runner.save(stage / "joint_checkpoint.pt")
        torch.save(dict(actor_state_dict=runner.policy.control.actor.state_dict(),
                        critic_state_dict=runner.policy.control.critic.state_dict(),
                        iter=runner.iteration, infos=dict(runner.metadata, purpose="selected-design controller inference"),
                        control_schema=runner.control_schema, config=runner.config,
                        space_schema=runner._space_schema(), morphology=normalized.tolist(), physical=physical.tolist()),
                   stage / "controller.pt")
        manifest = dict(version=1, selection="decoded hardware mean", iteration=runner.iteration,
                        design_id=result.design_id, xml=str(Path("robot") / result.design.xml_path.name),
                        physical=physical.tolist(), normalized=normalized.tolist(),
                        compiler_fingerprint=getattr(runner.compiler, "fingerprint", None),
                        files={str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(stage.rglob("*")) if path.is_file()})
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(stage, destination)
    return destination
