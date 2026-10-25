"""Native event projection retains unrelated TensorBoard data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tensorboard")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util import tensor_util

from draft.codesign.tensorboard_view import EventMirror, EventProjector


def _value(tag, number):
    return summary_pb2.Summary.Value(tag=tag, simple_value=number)


def _event(step, *values):
    return event_pb2.Event(step=step, wall_time=1000 + step, summary=summary_pb2.Summary(value=values))


def _read_events(directory):
    return [event for path in sorted(directory.glob("events.out.tfevents.*"))
            for event in LegacyEventFileLoader(str(path)).Load()]


def _values(events):
    return [value for event in events if event.HasField("summary") for value in event.summary.value]


def test_projector_preserves_event_fields_and_non_topology_value_bytes():
    image = summary_pb2.Summary.Value(
        tag="Video/rollout", image=summary_pb2.Summary.Image(
            height=1, width=1, colorspace=3, encoded_image_string=b"opaque-image-payload"))
    text = summary_pb2.Summary.Value(
        tag="Text/config", tensor=tensor_util.make_tensor_proto([b"hello"]),
        metadata=summary_pb2.SummaryMetadata(plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name="text", content=b"opaque-plugin-metadata")))
    scalar = _value("Controller/reward", 2.5)
    event = _event(17, image, text, scalar,
                   _value("HardwareTopology/mean_active_joints", 9.5),
                   _value("HardwareTopology/active_joint_count", 10),
                   _value("HardwareTopology/entropy", 1.5),
                   _value("HardwareTopology/obsolete_detail", 42))
    before = event.SerializeToString()

    projected = EventProjector().project(event)
    assert len(projected) == 1
    assert projected[0].step == 17 and projected[0].wall_time == 1017
    assert [v.tag for v in projected[0].summary.value] == [
        "Video/rollout", "Text/config", "Controller/reward",
        "HardwareTopology/expected_active_joints",
        "HardwareTopology/sampled_active_joints", "HardwareTopology/entropy",
    ]
    assert [v.SerializeToString() for v in projected[0].summary.value[:3]] == [
        v.SerializeToString() for v in (image, text, scalar)]
    assert event.SerializeToString() == before


def test_probabilities_complete_across_batches_and_compact_input_is_idempotent():
    projector = EventProjector()
    for index in range(31):
        event = _event(4, _value(f"HardwareTopology/probability_{index:02d}", index / 100))
        assert projector.project(event) == []
    for index in range(31, 63):
        value = summary_pb2.Summary.Value(
            tag=f"HardwareTopology/probability_{index:02d}",
            tensor=tensor_util.make_tensor_proto(index / 100))
        assert projector.project(_event(4, value)) == []
    result = projector.project(_event(4, _value("HardwareTopology/probability_63", .99)))
    assert len(result) == 1
    assert result[0].summary.value[0].tag == "HardwareTopology/max_probability"
    assert result[0].summary.value[0].simple_value == pytest.approx(.99)
    assert projector.project(_event(4, _value("HardwareTopology/probability_63", .99))) == []
    compact = _event(5, *[_value("HardwareTopology/" + tag, .5) for tag in (
        "expected_active_joints", "sampled_active_joints", "entropy", "max_probability")])
    assert projector.project(compact) == [compact]


def test_mirror_tails_native_events_preserves_raw_source_and_rotates(tmp_path):
    source, output = tmp_path / "source", tmp_path / "view"
    source.mkdir()
    writer = EventFileWriter(str(source), filename_suffix=".first")
    writer.add_event(_event(1, _value("Controller/policy_loss", .1),
                            _value("HardwareTopology/mean_active_joints", 9),
                            _value("HardwareTopology/active_joint_count", 10),
                            _value("HardwareTopology/entropy", .8)))
    for index in range(30):
        writer.add_event(_event(1, _value(f"HardwareTopology/probability_{index:02d}", index / 100)))
    writer.flush()
    raw_before = {p.name: p.read_bytes() for p in source.iterdir()}

    with EventMirror(source, output) as mirror:
        assert mirror.sync() == 32  # Writer header, first event, 30 probabilities.
        assert mirror.sync() == 0
        first_view = _read_events(output)
        assert len([v for v in _values(first_view) if v.tag == "Controller/policy_loss"]) == 1
        assert not any(v.tag == "HardwareTopology/max_probability" for v in _values(first_view))
        assert {p.name: p.read_bytes() for p in source.iterdir()} == raw_before

        for index in range(30, 64):
            writer.add_event(_event(1, _value(f"HardwareTopology/probability_{index:02d}", index / 100)))
        writer.flush()
        assert mirror.sync() == 34
        assert mirror.sync() == 0
        writer.close()

        rotated = EventFileWriter(str(source), filename_suffix=".second")
        rotated.add_event(_event(2, _value("Reward/episode_total", 3),
                                 _value("HardwareTopology/expected_active_joints", 8),
                                 _value("HardwareTopology/sampled_active_joints", 10),
                                 _value("HardwareTopology/entropy", .7),
                                 _value("HardwareTopology/max_probability", .6)))
        rotated.close()
        raw_final = {p.name: p.read_bytes() for p in source.iterdir()}
        assert mirror.sync() == 2
        assert mirror.sync() == 0

    assert {p.name: p.read_bytes() for p in source.iterdir()} == raw_final
    events = _read_events(output)
    tags = [v.tag for v in _values(events)]
    assert set(t for t in tags if t.startswith("HardwareTopology/")) == {
        "HardwareTopology/expected_active_joints", "HardwareTopology/sampled_active_joints",
        "HardwareTopology/entropy", "HardwareTopology/max_probability",
    }
    assert tags.count("HardwareTopology/max_probability") == 2
    assert tags.count("Controller/policy_loss") == 1
    assert tags.count("Reward/episode_total") == 1
    accumulator = EventAccumulator(str(output)).Reload()
    assert set(accumulator.Tags()["scalars"]) >= {
        "Controller/policy_loss", "Reward/episode_total",
        "HardwareTopology/expected_active_joints", "HardwareTopology/sampled_active_joints",
        "HardwareTopology/entropy", "HardwareTopology/max_probability",
    }


def test_mirror_rejects_overlap_and_nonempty_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        EventMirror(source, source / "view")
    with pytest.raises(ValueError, match="overlap"):
        EventMirror(source, tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "events.out.tfevents.old").touch()
    with pytest.raises(ValueError, match="empty"):
        EventMirror(source, output)


def test_mirror_waits_for_incomplete_record(tmp_path):
    staging, source, output = (tmp_path / name for name in ("staging", "source", "view"))
    staging.mkdir()
    source.mkdir()
    writer = EventFileWriter(str(staging))
    writer.add_event(_event(7, _value("Reward/episode_total", 9)))
    writer.close()
    raw = next(staging.iterdir()).read_bytes()
    target = source / "events.out.tfevents.partial"
    target.write_bytes(raw[:-5])
    with EventMirror(source, output) as mirror:
        assert mirror.sync() == 1  # Complete writer header only.
        assert mirror.sync() == 0
        with target.open("ab") as stream:
            stream.write(raw[-5:])
        assert mirror.sync() == 1
        assert mirror.sync() == 0
    assert [_v.simple_value for _v in _values(_read_events(output))] == [9]


def test_once_cli_backfills_events(tmp_path):
    source, output = tmp_path / "source", tmp_path / "view"
    source.mkdir()
    writer = EventFileWriter(str(source))
    writer.add_event(_event(3, _value("HardwareTopology/mean_active_joints", 8),
                            _value("Controller/reward", 12)))
    writer.close()
    script = Path(__file__).resolve().parents[1] / "scripts" / "compact_tensorboard.py"
    result = subprocess.run([sys.executable, str(script), "--source", str(source),
                             "--output", str(output), "--once"], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert result.returncode == 0, result.stderr
    assert {v.tag for v in _values(_read_events(output))} == {
        "HardwareTopology/expected_active_joints", "Controller/reward"}
