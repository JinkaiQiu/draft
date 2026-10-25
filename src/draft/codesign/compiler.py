"""Content-addressed Draft compilation with bounded, process-safe asset storage."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import fcntl
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import yaml

from draft.paths import repo_root
from .space import DesignSpace


@dataclass(frozen=True)
class CompiledDesign:
    design_id: str
    xml_path: Path
    parameters: dict[str, Any]
    morphology: tuple[float, ...]
    physical: tuple[float, ...]
    leg_length: float


@dataclass(frozen=True)
class CompilationResult:
    """One result per hardware draw, including rejected designs."""

    design: CompiledDesign | None
    error: str | None = None
    design_id: str = ""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@contextmanager
def _cache_lock(root: Path):
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _compile_job(spec: dict, cache_dir: str, fingerprint: str,
                 physical: tuple[float, ...]) -> CompilationResult:
    """Spawn-safe worker; only anticipated feasibility failures become rejections."""
    from draft.generation.generator import RobotGenerator
    from draft.trends.feasibility import FeasibilityViolation

    space = DesignSpace.from_dict(spec)
    root = Path(cache_dir)
    parameters = space.overrides(physical)
    topology = parameters.pop("joint_topology", None)
    if topology is not None:
        from .topology_physics import apply_joint_topology
        topology = dict(topology)
        # An active joint has no lock angle. Canonicalization also unifies cache IDs.
        topology["lock_angles"] = [0.0 if enabled else float(angle)
                                   for enabled, angle in zip(topology["active"], topology["lock_angles"])]
        parameters["joint_topology"] = topology
    design_id = _digest({"compiler": fingerprint, "parameters": parameters})
    destination = root / design_id

    def cached():
        receipt = destination / "design.json"
        if not receipt.exists():
            return None
        data = json.loads(receipt.read_text())
        if data.get("compiler") != fingerprint or data.get("design_id") != design_id:
            raise ValueError(f"Inconsistent design-cache receipt: {receipt}")
        for name, want in data.get("files", {}).items():
            path = destination / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != want:
                raise ValueError(f"Design-cache artifact failed validation: {path}")
        os.utime(receipt, None)
        if data["error"] is not None:
            return CompilationResult(None, data["error"], design_id)
        resolved = yaml.safe_load((destination / "parameters_resolved.yaml").read_text())
        return CompilationResult(CompiledDesign(
            design_id, destination / "quadruped.xml", resolved,
            tuple(map(float, space.normalize(physical))), tuple(map(float, physical)),
            float(resolved["upper_leg_link_length"] + resolved["lower_leg_link_length"])),
            design_id=design_id)

    with _cache_lock(root):
        result = cached()
        if result is not None:
            return result

    # Stage on the same filesystem; publishing a complete cohort asset is atomic.
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as staging:
        stage = Path(staging)
        error = None
        with (stage / "generation.log").open("w") as log, redirect_stdout(log):
            try:
                RobotGenerator(repo_root() / "src/draft/robots/quadruped").generate(
                    stage, params_override={k: v for k, v in parameters.items()
                                             if k != "joint_topology"}, strict=True)
                if topology is not None:
                    apply_joint_topology(stage, topology)
            except FeasibilityViolation as exc:
                error = str(exc)
        if error is not None:
            # Keep the reason, not half-built assets that can never be simulated.
            for path in stage.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        else:
            (stage / "parameters_from_gui.yaml").write_text(yaml.safe_dump(parameters, sort_keys=False))
        files = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(stage.rglob("*")) if p.is_file()}
        (stage / "design.json").write_text(json.dumps({
            "version": 1, "design_id": design_id, "compiler": fingerprint,
            "physical": list(physical), "error": error, "files": files,
        }, sort_keys=True, indent=2) + "\n")
        with _cache_lock(root):
            result = cached()
            if result is not None:
                return result
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(stage, destination)
            # TemporaryDirectory tolerates its directory having been moved.
            return cached()


class DesignCompiler:
    """Compile unique hardware actions once, retaining result order and failures.

    CPU generation uses a persistent spawn pool when ``workers > 0``. Assets
    belonging to the current batch are pinned during pruning; other least-
    recently-used designs are removed. No assets are stored in checkpoints:
    their space and source fingerprint make them reproducible after resume.
    """

    def __init__(self, space: DesignSpace, cache_dir: str | Path, workers: int = 0,
                 max_cache_entries: int = 128):
        for name, value, minimum in (("workers", workers, 0), ("max_cache_entries", max_cache_entries, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.space, self.workers = space, workers
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lease_dir = self.cache_dir / ".leases"
        self._lease_dir.mkdir(exist_ok=True)
        self._lease_path = self._lease_dir / f"{os.getpid()}-{uuid4().hex}.json"
        self.max_cache_entries = max_cache_entries
        self.fingerprint = self.source_fingerprint()
        self._pool = None

    @staticmethod
    def source_fingerprint() -> str:
        root = repo_root()
        inputs = [root / "src/draft/codesign/compiler.py",
                  root / "src/draft/codesign/topology_physics.py",
                  root / "src/draft/codesign/topology.py"]
        for name in ("generation", "trends", "robots/quadruped"):
            inputs.extend(p for p in (root / "src/draft" / name).rglob("*")
                          if p.is_file() and p.suffix in {".py", ".yaml", ".json", ".xml"})
        return _digest({
            "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(inputs)},
            "versions": {name: version(name) for name in (
                "mujoco", "numpy", "trimesh", "scipy", "pyyaml", "mapbox-earcut", "rtree")},
        })

    def compile_many(self, physical: Sequence[Sequence[float]]) -> list[CompilationResult]:
        values = np.asarray(physical, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.space.dim or not len(values)
                or not np.isfinite(values).all()):
            raise ValueError("A nonempty [designs, dimensions] physical batch is required")
        # Validate before any worker or filesystem side effects.
        rejected = {}
        if self.space.topology_enabled:
            if not np.isin(values[:, 7:13], (0., 1.)).all():
                raise ValueError("Topology active masks must be binary")
            # Out-of-range lock requests are ordinary rejected hardware draws.
            # Active angle coordinates are marginalized by the design space.
            lo, hi = np.asarray(self.space.lower)[13:], np.asarray(self.space.upper)[13:]
            outside = ((values[:, 13:] < lo - 1e-5) | (values[:, 13:] > hi + 1e-5))
            bad = np.any(outside & (values[:, 7:13] == 0), axis=1)
            for index in np.flatnonzero(bad):
                row = tuple(map(float, values[index]))
                rejected[index] = CompilationResult(
                    None, "Frozen joint lock angle lies outside its declared bounds",
                    _digest({"compiler": self.fingerprint, "rejected_physical": row}))
            # Validate geometry on every row, including rejected topology draws.
            self.space.geometry_space().normalize(values[:, :7])
            values = values[~bad]
        if not len(values):
            return [rejected[i] for i in range(len(physical))]
        normalized = self.space.normalize(values)
        if self.space.topology_enabled:
            values = self.space.denormalize(normalized)
        rows = [tuple(map(float, row)) for row in values]
        unique = list(dict.fromkeys(rows))
        spec = self.space.to_dict()
        # Lease the *next* cohort before workers read or publish its assets.
        # A second training/playback process may prune this same shared cache.
        ids = {_digest({"compiler": self.fingerprint, "parameters": self.space.overrides(row)})
               for row in unique}
        with _cache_lock(self.cache_dir):
            temporary = self._lease_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"pid": os.getpid(), "designs": sorted(ids)}))
            os.replace(temporary, self._lease_path)
        if self.workers:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"))
            futures = [self._pool.submit(_compile_job, spec, str(self.cache_dir), self.fingerprint, row)
                       for row in unique]
            results = [f.result() for f in futures]
        else:
            results = [_compile_job(spec, str(self.cache_dir), self.fingerprint, row) for row in unique]
        by_row = dict(zip(unique, results))
        self.prune({r.design_id for r in results})
        valid_results = iter(by_row[row] for row in rows)
        return [rejected[i] if i in rejected else next(valid_results)
                for i in range(len(physical))]

    def compile(self, physical: Sequence[float]) -> CompiledDesign:
        result = self.compile_many([physical])[0]
        if result.design is None:
            raise ValueError(f"Rejected design {result.design_id}: {result.error}")
        return result.design

    @contextmanager
    def retain_assets(self, design_ids: Sequence[str]):
        """Pin assets while another compile may replace this compiler's batch lease."""
        ids = set(design_ids)
        if not ids or any(not isinstance(design_id, str) or not design_id for design_id in ids):
            raise ValueError("retain_assets requires nonempty design IDs")
        lease = self._lease_dir / f"{os.getpid()}-{uuid4().hex}.json"
        with _cache_lock(self.cache_dir):
            temporary = lease.with_suffix(".tmp")
            temporary.write_text(json.dumps({"pid": os.getpid(), "designs": sorted(ids)}))
            os.replace(temporary, lease)
        try:
            yield
        finally:
            with _cache_lock(self.cache_dir):
                lease.unlink(missing_ok=True)

    def prune(self, protected: set[str] = frozenset()) -> None:
        with _cache_lock(self.cache_dir):
            entries = sorted((p for p in self.cache_dir.iterdir()
                              if p.is_dir() and not p.name.startswith(".")),
                             key=lambda p: (p / "design.json").stat().st_mtime
                             if (p / "design.json").exists() else 0, reverse=True)
            keep = set(protected)
            for lease in self._lease_dir.glob("*.json"):
                data = json.loads(lease.read_text())
                try:
                    os.kill(data["pid"], 0)
                except ProcessLookupError:
                    lease.unlink()
                    continue
                except PermissionError:
                    pass  # Another user's live process still owns its assets.
                keep.update(data["designs"])
            for entry in entries:
                if entry.name in keep:
                    continue
                if len(keep) < self.max_cache_entries:
                    keep.add(entry.name)
                else:
                    shutil.rmtree(entry)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        with _cache_lock(self.cache_dir):
            self._lease_path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
