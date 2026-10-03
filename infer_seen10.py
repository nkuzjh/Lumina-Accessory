"""CSGO inference entrypoint; shared parser and layout live in csgo_seen10.cli."""

import sys

from csgo_seen10.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["infer", *sys.argv[1:]]))
