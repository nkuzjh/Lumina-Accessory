"""Fixed experiment semantics, relocatable machine paths and content identity."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "csgo_seen10_exp32gen_aligned"
PATH_ENV = {
    "data_root": "DATA_ROOT", "shared_eval_dir": "SHARED_EVAL_DIR",
    "model_python": "MODEL_PYTHON", "eval_python": "EVAL_PYTHON",
    "base_checkpoint": "OFFICIAL_BASE_CHECKPOINT", "gemma_path": "GEMMA_PATH",
    "tokenizer_path": "TOKENIZER_PATH", "vae_path": "VAE_PATH", "run_root": "RUN_ROOT",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def scientific_config(cfg):
    return {k: v for k, v in cfg.items() if k not in {"paths", "runtime", "run_formal", "identity"}}


def resolve_path(value, base=PROJECT_ROOT):
    p = Path(value).expanduser()
    # Preserve the venv/bin/python symlink: executing its resolved system target
    # loses Python's virtual-environment discovery and installed dependencies.
    return os.path.abspath(p if p.is_absolute() else Path(base) / p)


def load_config(experiment=EXPERIMENT, *, seed=42, overrides=None, machine_config=None):
    if experiment != EXPERIMENT:
        raise ValueError(f"Unknown experiment: {experiment}")
    cfg = json.loads((PROJECT_ROOT / "configs" / f"{experiment}.json").read_text())
    cfg["seed"] = int(seed)
    cfg["paths"]["run_root"] = f"outputs/{experiment}/Lumina-Accessory/seed_{seed}"
    machine = Path(machine_config or os.environ.get("CSGO_MACHINE_CONFIG", PROJECT_ROOT / "configs/machine.local.json"))
    if machine.is_file():
        cfg["paths"].update(json.loads(machine.read_text()).get("paths", {}))
    for name, env in PATH_ENV.items():
        if os.environ.get(env):
            cfg["paths"][name] = os.environ[env]
    for name, value in (overrides or {}).items():
        if value is not None:
            if name not in PATH_ENV:
                raise ValueError(f"Unknown path override: {name}")
            cfg["paths"][name] = value
    cfg["paths"] = {name: resolve_path(value) for name, value in cfg["paths"].items()}
    # RUN_FORMAL is a record of this implementation session, never an auto-run switch.
    cfg["run_formal"] = False
    cfg["identity"] = {"science_sha256": content_hash(scientific_config(cfg))}
    return cfg


def batch_configuration(world_size, micro_batch_size, accumulation=None, effective=128):
    if world_size < 1 or micro_batch_size < 1:
        raise ValueError("world_size and micro_batch_size must be positive")
    divisor = world_size * micro_batch_size
    if accumulation is None:
        if effective % divisor:
            raise ValueError(f"128 is not divisible by world×micro={divisor}")
        accumulation = effective // divisor
    if accumulation < 1 or divisor * accumulation != effective:
        raise ValueError(f"world×micro×accum must equal {effective}; got {world_size}×{micro_batch_size}×{accumulation}")
    return int(accumulation)


def code_identity():
    tracked = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    # Include newly created source as well as tracked source; exclude machine paths/artifacts.
    files = sorted({*PROJECT_ROOT.glob("csgo_seen10/*.py"), *PROJECT_ROOT.glob("models_accessory/*.py"),
                    *PROJECT_ROOT.glob("transport/*.py"), *PROJECT_ROOT.glob("scripts/*.py"),
                    *PROJECT_ROOT.glob("scripts/*.sh"), *PROJECT_ROOT.glob("*seen10.py"),
                    *PROJECT_ROOT.glob("requirements-csgo-seen10*.txt"),
                    PROJECT_ROOT / "scripts/csgo_seen10_assets.json",
                    PROJECT_ROOT / "configs" / f"{EXPERIMENT}.json"})
    config_path = PROJECT_ROOT / "configs" / f"{EXPERIMENT}.json"
    # Machine paths must not re-enter identity through a raw config-file digest.
    hashes = {str(p.relative_to(PROJECT_ROOT)):
              (content_hash(scientific_config(json.loads(p.read_text()))) if p == config_path else sha256_file(p))
              for p in files if p.is_file()}
    return {"upstream_revision": tracked, "source_sha256": content_hash(hashes), "files": hashes}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
