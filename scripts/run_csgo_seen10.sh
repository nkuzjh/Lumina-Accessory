#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MODEL_PYTHON="$("${CSGO_CONFIG_PYTHON:-python3}" - "$@" <<'PY'
import argparse, sys
from csgo_seen10.config import load_config
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--model-python')
args, _ = p.parse_known_args(sys.argv[1:])
print(load_config(overrides={'model_python': args.model_python})['paths']['model_python'])
PY
)"
export PYTHONDONTWRITEBYTECODE=1
exec "$MODEL_PYTHON" -m csgo_seen10.cli "$@"
