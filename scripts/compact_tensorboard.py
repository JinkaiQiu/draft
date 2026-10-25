#!/usr/bin/env python3
"""Follow native TensorBoard events into a compact topology view."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from draft.codesign.tensorboard_view import EventMirror


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original run event directory")
    parser.add_argument("--output", required=True, type=Path, help="Fresh projection event directory")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    parser.add_argument("--once", action="store_true", help="Backfill available events and exit")
    args = parser.parse_args(argv)
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    with EventMirror(args.source, args.output) as mirror:
        while True:
            mirror.sync()
            if args.once:
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
