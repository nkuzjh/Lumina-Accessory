"""Read-only dataset checks and explicitly isolated evaluator I/O fixtures."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import builtins
import io

from .config import atomic_json, code_identity, PROJECT_ROOT
from .data import Seen10Dataset, protocol_identity, make_prompt


def check_data(cfg, output):
    started = time.monotonic()
    identity = protocol_identity(cfg)
    counts = {}
    max_tokens = 0
    tokenizer = None
    if Path(cfg["paths"]["tokenizer_path"]).is_dir():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["tokenizer_path"], local_files_only=True)
    for split, expected in [("train", 50000), ("validation", 5000), ("discrete", 20000), ("continuous", 12800)]:
        ds = Seen10Dataset(cfg, split, load_targets=False)
        if len(ds) != expected:
            raise ValueError(f"{split}: {len(ds)} != {expected}")
        counts[split] = len(ds)
        if tokenizer:
            for i in range(0, len(ds), 512):
                ids = tokenizer([make_prompt(r) for r in ds.rows[i:i+512]], truncation=False)["input_ids"]
                max_tokens = max(max_tokens, max(map(len, ids)))
    if max_tokens > cfg["model"]["max_text_length"]:
        raise ValueError(f"Prompt would truncate: {max_tokens}")
    result = {"status": "passed", "counts": counts, "data_identity": identity,
              "prompt_tokens_max": max_tokens if tokenizer else None,
              "prompt_token_check": "passed" if tokenizer else "untested: tokenizer unavailable",
              "code": code_identity(), "seconds": time.monotonic()-started, "gpu_compute": False}
    atomic_json(output, result)
    return result


def evaluator_fixture_smoke(cfg, root):
    """Checks actual shared metric CLI only; fixture images are NOT model predictions."""
    from PIL import Image
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "fixture_marker.json", {"fixture_only": True, "model_generated": False, "formal": False})
    results = {}
    for task in ("discrete", "continuous"):
        row = Seen10Dataset(cfg, task, load_targets=False, limit=1).rows[0]
        pred = root / task / "gen_imgs"
        path = pred / row["map_name"] / (row["file_frame"] + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (448, 448), (100, 115, 130)).save(path, quality=75, optimize=False, progressive=False)
        command = [cfg["paths"]["eval_python"], str(Path(cfg["paths"]["shared_eval_dir"]) / "run_eval.py"),
                   "smoke", task, "--pred-root", str(pred), "--data-root", cfg["paths"]["data_root"],
                   "--config", str(Path(cfg["paths"]["shared_eval_dir"]) / "benchmark_v2.yaml"), "--device", "cpu"]
        command += ["--limit", "1"] if task == "discrete" else ["--frame-only"]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        # Temporary metric loader links/caches belong to this new project.
        # Unix multiprocessing sockets have a short path limit; keep TMPDIR short.
        temp = PROJECT_ROOT / ".tmp"
        temp.mkdir(exist_ok=True)
        env["TMPDIR"] = str(temp)
        with (root / f"{task}.log").open("w") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        results[task] = {"command": command, "returncode": proc.returncode, "log": str(root / f"{task}.log")}
    atomic_json(root / "result.json", {"fixture_only": True, "results": results,
                "not_tested": ["model quality", "FID", "FVD", "TWE", "TDE", "formal coverage"]})
    return results
