"""Headless evaluation video scheduling, streaming and unchanged trial scoring."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from draft.codesign import evaluation


class VideoEnv:
    num_actions = 1
    horizon_steps = 8
    step_dt = 0.1
    instances = []

    def __init__(self, designs, replicas, **kwargs):
        self.unwrapped = self
        self.count = replicas
        self.steps = 0
        self.closed = False
        self.actions = []
        self.options = kwargs["env_kwargs"]
        self.instances.append(self)

    def obs(self):
        return {key: torch.zeros((self.count, 1)) for key in ("actor", "critic")}

    def reset(self):
        return self.obs(), {}

    def step(self, actions):
        self.actions.append(actions.clone())
        self.steps += 1
        fallen = torch.zeros(self.count, dtype=torch.bool)
        fallen[0] = self.steps == 2
        timeout = torch.full_like(fallen, self.steps == self.horizon_steps)
        return self.obs(), torch.ones(self.count), fallen | timeout, {
            "codesign_terminated": fallen, "time_outs": timeout}

    def render(self):
        return np.full((self.options["render_height"], self.options["render_width"], 3),
                       self.steps * 25, dtype=np.uint8)

    def close(self):
        self.closed = True


def _runner(tmp_path, **options):
    config = dict(eval_replicas=2, seed=5, episode_length_s=.8, env_kwargs={},
                  reward_overrides={}, invalid_design_return=-100., gamma=1.,
                  terminal_penalty=-10., eval_video_interval=10, eval_video_length_s=.8,
                  eval_video_fps=5, eval_video_width=16, eval_video_height=16)
    config.update(options)
    policy = SimpleNamespace(eval=lambda: None, control=SimpleNamespace(
        deterministic=lambda observation: torch.full((len(observation["actor"]), 1), .7)))
    return SimpleNamespace(config=config, phase="boundary", log_dir=tmp_path,
                           iteration=10, device=torch.device("cpu"), policy=policy,
                           env_factory=VideoEnv, _ensure_control=lambda *args: None)


def test_cpu_mp4_stream_preserves_score_rng_and_postfall_attempts(tmp_path, monkeypatch):
    pytest.importorskip("mediapy")
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    result = SimpleNamespace(design=object(), design_id="chosen", error=None)
    monkeypatch.setattr(evaluation, "selected_design", lambda runner: (result, np.zeros(7), np.ones(7)))
    # Exercise the packaged ffmpeg fallback independently of host PATH.
    original_which = evaluation.shutil.which
    monkeypatch.setattr(evaluation.shutil, "which",
                        lambda name: None if name == "ffmpeg" else original_which(name))
    runner = _runner(tmp_path, hardware_evaluation_steps=2, hardware_bootstrap=True)
    rng = torch.get_rng_state().clone()
    report = evaluation.evaluate(runner)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    path = Path(report["video_path"])
    assert path == tmp_path / "videos/eval_000010.mp4" and path.stat().st_size > 0
    assert report["video_frames"] == 4 and report["video_fps"] == 5
    assert report["video_includes_autoresets"]
    assert report["video_trial_rewards_masked_after_first_done"]
    recorded = VideoEnv.instances[-1]
    assert recorded.closed and recorded.options["render_mode"] == "rgb_array"
    assert recorded.steps == report["horizon_steps"] == VideoEnv.horizon_steps == 8
    assert recorded.actions[-1][0, 0] == pytest.approx(.7)
    frames = ffmpeg.read_frames(str(path))
    metadata = next(frames)
    assert metadata["size"] == (16, 16) and metadata["fps"] == 5
    assert sum(1 for _ in frames) == 4
    baseline = evaluation.evaluate(runner, record_video=False)
    assert VideoEnv.instances[-1].steps == baseline["horizon_steps"] == 8
    for name in ("return_mean", "discounted_return_mean", "survival_fraction", "active_duration_fraction"):
        assert report[name] == baseline[name]


def test_video_schedule_override_and_validation(tmp_path):
    runner = _runner(tmp_path)
    runner.iteration = 11
    assert evaluation._video_options(runner, None) is None
    assert evaluation._video_options(runner, True)["path"].name == "eval_000011.mp4"
    runner.config["eval_video_width"] = 15
    with pytest.raises(ValueError, match="even"):
        evaluation._video_options(runner, True)
    runner.config["eval_video_width"] = 16
    runner.log_dir = None
    with pytest.raises(ValueError, match="log directory"):
        evaluation._video_options(runner, True)


def test_hardware_cutoff_does_not_limit_full_evaluation(tmp_path, monkeypatch):
    result = SimpleNamespace(design=object(), design_id="chosen", error=None)
    monkeypatch.setattr(evaluation, "selected_design", lambda runner: (result, np.zeros(7), np.ones(7)))
    runner = _runner(tmp_path, hardware_evaluation_steps=2, hardware_bootstrap=True)
    report = evaluation.evaluate(runner, record_video=False)
    assert VideoEnv.instances[-1].steps == report["horizon_steps"] == 8
    assert report["return_mean"] == pytest.approx(0.)
