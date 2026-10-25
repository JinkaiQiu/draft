"""Project a live TensorBoard event stream into a compact topology view.

The source event files are opened only for reading. The projection has its own
fresh event directory and can be polled while the source writer is running.
"""

from __future__ import annotations

import math
import re
from collections import deque
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util import tensor_util


_PREFIX = "HardwareTopology/"
_ALIASES = {
    "mean_active_joints": "expected_active_joints",
    "active_joint_count": "sampled_active_joints",
}
_KEEP = frozenset(("expected_active_joints", "sampled_active_joints", "entropy", "max_probability"))
_PROBABILITY = re.compile(r"probability_(0[0-9]|[1-5][0-9]|6[0-3])\Z")


def _scalar(value: summary_pb2.Summary.Value) -> float | None:
    if value.HasField("simple_value"):
        return float(value.simple_value)
    if value.HasField("tensor"):
        try:
            array = tensor_util.make_ndarray(value.tensor)
            if array.size == 1:
                return float(array.reshape(-1)[0])
        except (TypeError, ValueError):
            pass
    return None


class EventProjector:
    """Filter one Event at a time; retain probability state across poll batches."""

    def __init__(self, max_pending_steps: int = 32):
        if max_pending_steps < 1:
            raise ValueError("max_pending_steps must be positive")
        self.max_pending_steps = max_pending_steps
        self._probabilities: dict[int, dict[int, float]] = {}
        self._completed: set[int] = set()
        self._recently_completed: deque[int] = deque(maxlen=max_pending_steps)

    def project(self, event: event_pb2.Event) -> list[event_pb2.Event]:
        if not event.HasField("summary"):
            return [event]

        kept = []
        maxima = []
        changed = False
        for value in event.summary.value:
            tag = value.tag
            if not tag.startswith(_PREFIX):
                kept.append(value)
                continue
            suffix = tag[len(_PREFIX):]
            match = _PROBABILITY.fullmatch(suffix)
            if match:
                changed = True
                probability = _scalar(value)
                if probability is not None and math.isfinite(probability):
                    maximum = self._add_probability(int(event.step), int(suffix[-2:]), probability)
                    if maximum is not None:
                        maxima.append(maximum)
                continue
            renamed = _ALIASES.get(suffix, suffix)
            if renamed in _KEEP:
                if renamed != suffix:
                    changed = True
                    clone = summary_pb2.Summary.Value()
                    clone.CopyFrom(value)
                    clone.tag = _PREFIX + renamed
                    kept.append(clone)
                else:
                    kept.append(value)
            else:
                changed = True

        if not changed:
            return [event]
        projected = []
        if kept:
            clone = event_pb2.Event()
            clone.CopyFrom(event)
            del clone.summary.value[:]
            clone.summary.value.extend(kept)
            projected.append(clone)
        for maximum in maxima:
            projected.append(event_pb2.Event(
                step=event.step,
                wall_time=event.wall_time,
                summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(
                    tag=_PREFIX + "max_probability", simple_value=maximum,
                )]),
            ))
        return projected

    def _add_probability(self, step: int, index: int, value: float) -> float | None:
        if step in self._completed:
            return None
        values = self._probabilities.setdefault(step, {})
        values[index] = value
        if len(values) == 64:
            maximum = max(values.values())
            del self._probabilities[step]
            if len(self._recently_completed) == self.max_pending_steps:
                self._completed.discard(self._recently_completed.popleft())
            self._recently_completed.append(step)
            self._completed.add(step)
            return maximum
        while len(self._probabilities) > self.max_pending_steps:
            self._probabilities.pop(next(iter(self._probabilities)))
        return None


class EventMirror:
    """Incrementally copy source event files to an independent event directory."""

    def __init__(self, source: str | Path, output: str | Path):
        self.source = Path(source).resolve()
        self.output = Path(output).resolve()
        if not self.source.is_dir():
            raise ValueError(f"source event directory does not exist: {self.source}")
        if self.source == self.output or self.source in self.output.parents or self.output in self.source.parents:
            raise ValueError("source and output event directories must not overlap")
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError(f"output event directory must be empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.projector = EventProjector()
        self._loaders: dict[Path, LegacyEventFileLoader] = {}
        self._file_state: dict[Path, tuple[int, int]] = {}
        self._writer = EventFileWriter(str(self.output))

    def sync(self) -> int:
        """Copy newly completed source records; return source events consumed."""
        count = 0
        for path in sorted(self.source.rglob("events.out.tfevents.*")):
            if not path.is_file():
                continue
            stat = path.stat()
            previous = self._file_state.get(path)
            if previous and (stat.st_ino != previous[0] or stat.st_size < previous[1]):
                raise RuntimeError(f"source event file was replaced or truncated: {path}")
            if path not in self._loaders:
                self._loaders[path] = LegacyEventFileLoader(str(path))
            loader = self._loaders[path]
            for event in loader.Load():
                for projected in self.projector.project(event):
                    self._writer.add_event(projected)
                count += 1
            self._file_state[path] = (stat.st_ino, stat.st_size)
        if count:
            self._writer.flush()
        return count

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> EventMirror:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
