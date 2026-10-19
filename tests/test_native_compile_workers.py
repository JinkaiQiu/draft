"""Bounded independent CPU compilation, ordering, and failure cleanup."""
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")

from draft.tasks.quadruped.codesign import native_env

pytestmark = pytest.mark.needs_rl


def _fake_compile(monkeypatch, workers, compile_fn, count=7, build_fn=None):
    caller = threading.get_ident()
    snapshot, terrain = object(), object()
    snapshots = []
    built, applied = [], []

    def terrain_snapshot(scene):
        snapshots.append(scene)
        return snapshot

    class Scene:
        def __init__(self, cfg, shared_terrain, terrain_spec):
            assert threading.get_ident() == caller
            assert shared_terrain is terrain and terrain_spec is snapshot
            self.index = cfg.entities["robot"]
            built.append(self.index)
            if build_fn is not None:
                build_fn(self.index)

        def compile(self):
            return compile_fn(self.index)

    def apply(model):
        assert threading.get_ident() == caller
        applied.append(model)

    monkeypatch.setattr(native_env, "_terrain_snapshot", terrain_snapshot)
    monkeypatch.setattr(native_env, "_SharedTerrainScene", Scene)
    cfg = SimpleNamespace(cpu_compile_workers=workers, scene=SimpleNamespace(),
                          sim=SimpleNamespace(mujoco=SimpleNamespace(apply=apply)))
    scene = SimpleNamespace(terrain=terrain)
    cohort = SimpleNamespace(entities=tuple(range(count)))
    return cfg, scene, cohort, snapshots, built, applied


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_compile_workers_bound_concurrency_and_preserve_design_order(monkeypatch, workers):
    lock = threading.Lock()
    barrier = threading.Barrier(workers)
    active = peak = 0

    def compile_model(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if index < workers:
                barrier.wait(timeout=5)
            time.sleep(.002 * (7 - index))
            return index
        finally:
            with lock:
                active -= 1

    cfg, scene, cohort, snapshots, built, applied = _fake_compile(
        monkeypatch, workers, compile_model)
    assert native_env.compile_cohort_models(scene, cfg, cohort) == tuple(range(7))
    assert peak == workers and active == 0
    assert snapshots == [scene] and built == list(range(7))
    assert sorted(applied) == list(range(7))
    assert not any(thread.name.startswith("draft-cpu-compile") for thread in threading.enumerate())


@pytest.mark.parametrize("failure", ["compile", "build", "apply"])
def test_compile_failure_propagates_and_joins_workers(monkeypatch, failure):
    active = set()
    lock = threading.Lock()

    def compile_model(index):
        with lock:
            active.add(index)
        try:
            time.sleep(.01)
            if failure == "compile" and index == 1:
                raise RuntimeError("compile failed")
            return index
        finally:
            with lock:
                active.remove(index)

    def build(index):
        if failure == "build" and index == 1:
            raise RuntimeError("build failed")

    cfg, scene, cohort, *_ = _fake_compile(monkeypatch, 2, compile_model, build_fn=build)
    if failure == "apply":
        def apply(model):
            raise RuntimeError("apply failed")
        cfg.sim.mujoco.apply = apply
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        native_env.compile_cohort_models(scene, cfg, cohort)
    assert not active
    assert not any(thread.name.startswith("draft-cpu-compile") for thread in threading.enumerate())


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5, "2", None])
def test_invalid_worker_count_rejected_before_task_factory(monkeypatch, workers):
    def factory(**kwargs):
        pytest.fail("invalid worker count reached task factory")
    with pytest.raises(ValueError, match="cpu_compile_workers"):
        native_env.make_native_env_cfg([object()], 1, "robot.xml", env_cfg_factory=factory,
                                       cpu_compile_workers=workers)


def test_worker_option_consumed_before_authored_task_factory(monkeypatch):
    cfg = SimpleNamespace()
    forwarded = []

    def factory(**kwargs):
        forwarded.append(kwargs)
        return cfg

    monkeypatch.setattr(native_env, "augment_native_cohort_cfg", lambda cfg, *args: cfg)
    actual = native_env.make_native_env_cfg(
        [SimpleNamespace(xml_path="robot.xml")], 1, "robot.xml",
        env_cfg_factory=factory, cpu_compile_workers=1)
    assert actual is cfg and cfg.cpu_compile_workers == 1
    assert "cpu_compile_workers" not in forwarded[0]
