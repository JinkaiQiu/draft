#!/usr/bin/env python3
"""Expand a seven-geometry controller checkpoint for topology warm starting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from draft.codesign.checkpoint_conversion import expand_topology_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Geometry-conditioned checkpoint")
    parser.add_argument("--output", required=True, type=Path, help="New native controller checkpoint")
    parser.add_argument("--lock-angle-bounds", help="YAML mapping, or @path to a YAML file")
    args = parser.parse_args(argv)
    bounds = None
    if args.lock_angle_bounds:
        value = args.lock_angle_bounds
        bounds = yaml.safe_load(Path(value[1:]).read_text() if value.startswith("@") else value)
    report = expand_topology_checkpoint(args.input, args.output, lock_angle_bounds=bounds)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
