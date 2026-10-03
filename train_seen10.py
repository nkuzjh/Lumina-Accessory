"""Explicit entry point for Seen-10 training and bounded smoke runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from csgo_seen10.config import EXPERIMENT, load_config
from csgo_seen10.training import run_training


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--micro-batch-size", type=int, required=True)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--resume", default=None, help="latest, best, late, step name, or complete directory")
    p.add_argument("--run-root", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--shared-eval-dir", type=Path)
    p.add_argument("--model-python", type=Path)
    p.add_argument("--eval-python", type=Path)
    p.add_argument("--base-checkpoint", type=Path)
    p.add_argument("--gemma-path", type=Path)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--vae-path", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=2)
    p.add_argument("--smoke-limit", type=int, default=128)
    p.add_argument("--validation-limit", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    overrides = {name: str(getattr(args, name)) if getattr(args, name) is not None else None
                 for name in ("run_root", "data_root", "shared_eval_dir", "model_python", "eval_python", "base_checkpoint",
                              "gemma_path", "tokenizer_path", "vae_path")}
    if args.smoke and overrides["run_root"] is None:
        overrides["run_root"] = "outputs/smoke_csgo_seen10_exp32gen_aligned"
    cfg = load_config(args.experiment, seed=args.seed, overrides=overrides)
    result = run_training(cfg, micro_batch_size=args.micro_batch_size,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          resume=args.resume, smoke=args.smoke, smoke_steps=args.smoke_steps,
                          smoke_limit=args.smoke_limit, validation_limit=args.validation_limit)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
