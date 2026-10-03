#!/usr/bin/env bash
# Create an independent Python environment using an explicit accelerator profile.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" != "--env-only" || $# -gt 3 ]]; then
  echo "Usage: bash scripts/setup_csgo_seen10.sh --env-only [--profile blackwell-cu128|custom]" >&2
  exit 2
fi
PROFILE="auto"
if [[ $# -gt 1 ]]; then
  if [[ "${2:-}" != "--profile" || $# -ne 3 ]]; then
    echo "Expected --profile blackwell-cu128|custom" >&2; exit 2
  fi
  PROFILE="$3"
fi
if [[ "$PROFILE" == "auto" || "$PROFILE" == "blackwell-cu128" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA driver query available. Select --profile custom with pinned package and FlashAttention wheel settings." >&2; exit 1
  fi
  CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' ' | sort -u)"
  DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  if [[ ! "$CAPS" =~ ^(10\.0|12\.0)$ ]] || ! awk -v version="$DRIVER_CUDA" 'BEGIN {split(version, p, "."); exit !(p[1]>12 || (p[1]==12 && p[2]>=8))}'; then
    echo "Validated default requires one Blackwell GPU (SM 10.0/12.0) and driver CUDA >=12.8; found capability=$CAPS driver_cuda=$DRIVER_CUDA. Select --profile custom." >&2
    exit 1
  fi
  PROFILE="blackwell-cu128"
elif [[ "$PROFILE" != "custom" ]]; then
  echo "Unknown profile: $PROFILE" >&2; exit 2
fi

if [[ "$PROFILE" == "blackwell-cu128" ]]; then
  TORCH_VERSION="2.8.0+cu128"
  TORCHVISION_VERSION="0.23.0+cu128"
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
  EXPECTED_CUDA="12.8"
else
  : "${CSGO_TORCH_VERSION:?custom profile requires CSGO_TORCH_VERSION}"
  : "${CSGO_TORCHVISION_VERSION:?custom profile requires CSGO_TORCHVISION_VERSION}"
  : "${CSGO_TORCH_INDEX_URL:?custom profile requires CSGO_TORCH_INDEX_URL}"
  : "${CSGO_EXPECTED_TORCH_CUDA:?custom profile requires CSGO_EXPECTED_TORCH_CUDA}"
  : "${CSGO_FLASH_WHEEL_URL:?custom profile requires CSGO_FLASH_WHEEL_URL}"
  : "${CSGO_FLASH_WHEEL_SHA256:?custom profile requires CSGO_FLASH_WHEEL_SHA256}"
  TORCH_VERSION="$CSGO_TORCH_VERSION"
  TORCHVISION_VERSION="$CSGO_TORCHVISION_VERSION"
  TORCH_INDEX_URL="$CSGO_TORCH_INDEX_URL"
  EXPECTED_CUDA="$CSGO_EXPECTED_TORCH_CUDA"
fi

if [[ -n "${CSGO_CONSTRAINTS_FILE:-}" ]]; then
  CONSTRAINTS_PATH="$CSGO_CONSTRAINTS_FILE"
  if [[ "$CONSTRAINTS_PATH" != /* ]]; then CONSTRAINTS_PATH="$PROJECT_ROOT/$CONSTRAINTS_PATH"; fi
elif [[ "$PROFILE" == "blackwell-cu128" ]]; then
  CONSTRAINTS_PATH="$PROJECT_ROOT/requirements-csgo-seen10-cu128.lock.txt"
else
  CONSTRAINTS_PATH=""
fi
PIP_CONSTRAINTS=()
if [[ -n "$CONSTRAINTS_PATH" ]]; then
  if [[ ! -f "$CONSTRAINTS_PATH" ]]; then
    echo "Dependency constraints file is missing: $CONSTRAINTS_PATH" >&2; exit 1
  fi
  PIP_CONSTRAINTS=(-c "$CONSTRAINTS_PATH")
  CONSTRAINTS_SHA="$(sha256sum "$CONSTRAINTS_PATH" | cut -d ' ' -f1)"
else
  CONSTRAINTS_SHA=""
  echo "Custom accelerator profile: no default dependency lock; validate and pin this host's resolved environment with CSGO_CONSTRAINTS_FILE." >&2
fi

PYTHON_BIN="${CSGO_PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$(command -v "$candidate")"; break; fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11 or 3.12 is required; set CSGO_PYTHON_BIN to one." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  if ! "$PYTHON_BIN" -m venv .venv 2>/dev/null; then
    if python3 -m virtualenv --version >/dev/null 2>&1; then
      python3 -m virtualenv --clear --python "$PYTHON_BIN" .venv
    else
      echo "venv/ensurepip is unavailable. Install python3-venv or virtualenv and rerun." >&2
      exit 1
    fi
  fi
fi

.venv/bin/python -m pip install --upgrade 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1' 'packaging==25.0'
.venv/bin/python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "${PIP_CONSTRAINTS[@]}" --index-url "$TORCH_INDEX_URL"
.venv/bin/python -m pip install -r requirements-csgo-seen10.txt "${PIP_CONSTRAINTS[@]}"

# Official FlashAttention 2.8.3 release wheel matching torch 2.8, Python 3.12,
# CUDA 12 and PyTorch's CXX11 ABI. Python 3.11 uses the corresponding wheel.
ABI="$(.venv/bin/python - <<'PY'
import torch
print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')
PY
)"
PY_TAG="$(.venv/bin/python - <<'PY'
import sys
print(f'cp{sys.version_info.major}{sys.version_info.minor}')
PY
)"
if [[ "$PROFILE" == "blackwell-cu128" ]]; then
  WHEEL_NAME="flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
  WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME/+/%2B}"
else
  WHEEL_URL="$CSGO_FLASH_WHEEL_URL"
  WHEEL_NAME="${WHEEL_URL##*/}"
  WHEEL_NAME="${WHEEL_NAME%%\?*}"
  WHEEL_NAME="${WHEEL_NAME//%2B/+}"
fi
WHEEL_DIR="$PROJECT_ROOT/.venv/asset_wheels"
mkdir -p "$WHEEL_DIR"
if [[ ! -s "$WHEEL_DIR/$WHEEL_NAME" ]]; then
  curl --fail --location --retry 3 --silent --show-error "$WHEEL_URL" -o "$WHEEL_DIR/$WHEEL_NAME.part"
  mv "$WHEEL_DIR/$WHEEL_NAME.part" "$WHEEL_DIR/$WHEEL_NAME"
fi
# SHA256 values published on the official v2.8.3 GitHub release.
if [[ "$PROFILE" == "blackwell-cu128" ]]; then
  case "$ABI/$PY_TAG" in
    FALSE/cp311) WHEEL_SHA='f1a9e6cb4dfbd1647e56235d81fd6b56e6cd01c7ea3249968ca4aa36c389371a' ;;
    FALSE/cp312) WHEEL_SHA='2605b2653e7b9d6615bbb154896ae4d3f0b95d3120af7a6374ee97092e61df02' ;;
    TRUE/cp311) WHEEL_SHA='3d41b2fc55753faa7f45d6568ea73a96b96afb48b82994ab9b49bcbcb6c87588' ;;
    TRUE/cp312) WHEEL_SHA='f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb' ;;
    *) echo "No fixed FlashAttention wheel for ABI=$ABI Python=$PY_TAG" >&2; exit 1 ;;
  esac
else
  WHEEL_SHA="$CSGO_FLASH_WHEEL_SHA256"
fi
printf '%s  %s\n' "$WHEEL_SHA" "$WHEEL_DIR/$WHEEL_NAME" | sha256sum --check --status
.venv/bin/python -m pip install --no-deps "$WHEEL_DIR/$WHEEL_NAME"
.venv/bin/python -m pip check
mkdir -p outputs/implementation_audit
.venv/bin/python scripts/check_csgo_environment.py --prepare-headers > outputs/implementation_audit/environment_python_headers.json
CSGO_SETUP_PROFILE="$PROFILE" CSGO_SETUP_TORCH="$TORCH_VERSION" CSGO_SETUP_TORCHVISION="$TORCHVISION_VERSION" CSGO_SETUP_CUDA="$EXPECTED_CUDA" CSGO_SETUP_FLASH_SHA="$WHEEL_SHA" CSGO_SETUP_CONSTRAINTS_SHA="$CONSTRAINTS_SHA" .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
profile = {
    "profile": os.environ["CSGO_SETUP_PROFILE"],
    "torch": os.environ["CSGO_SETUP_TORCH"],
    "torchvision": os.environ["CSGO_SETUP_TORCHVISION"],
    "expected_torch_cuda": os.environ["CSGO_SETUP_CUDA"],
    "flash_wheel_sha256": os.environ["CSGO_SETUP_FLASH_SHA"],
    "constraints_sha256": os.environ["CSGO_SETUP_CONSTRAINTS_SHA"] or None,
}
Path(".venv/csgo_environment_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
PY
CSGO_EXPECTED_TORCH_CUDA="$EXPECTED_CUDA" .venv/bin/python scripts/check_csgo_environment.py
.venv/bin/python -m pip freeze > outputs/implementation_audit/environment_lock.txt
echo "Independent environment ready: $PROJECT_ROOT/.venv"
