"""Explicit commands; no chaining into formal experiments or background queues."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import EXPERIMENT, PROJECT_ROOT, PATH_ENV, atomic_json, load_config, sha256_file


def parser():
    p = argparse.ArgumentParser(description="Lumina-Accessory CSGO aligned manual runner")
    p.add_argument("command", choices=["check", "smoke", "train", "infer", "eval", "coverage", "benchmark", "compare", "plot"])
    p.add_argument("--experiment", default=EXPERIMENT)
    p.add_argument("--seed", type=int, default=42)
    for path in PATH_ENV:
        p.add_argument("--"+path.replace("_", "-"), default=None)
    p.add_argument("--checkpoint", default="late")
    p.add_argument("--resume", default=None)
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--task", choices=["all", "discrete", "continuous"], default="all")
    p.add_argument("--inference-engine", "--engine", choices=["eager", "compiled"], default="eager")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead"], default="default")
    p.add_argument("--batch-size", type=int, default=1, help="Inference batch; compare requires >=2")
    p.add_argument("--vae-batch-size", type=int, default=1)
    p.add_argument("--benchmark-batches", type=int)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--pred-root", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-only", action="store_true", help="Run CPU semantic checks and evaluator fixtures only")
    p.add_argument("--smoke-steps", type=int, default=2)
    p.add_argument("--limit", type=int, help="Isolated partial inference only; never formal coverage")
    p.add_argument("--frame-only", action="store_true")
    p.add_argument("--eval-smoke", action="store_true")
    return p


def inference_profile(args):
    return f"{args.inference_engine}-{args.compile_mode if args.inference_engine == 'compiled' else 'native'}-b{args.batch_size}-vae{args.vae_batch_size}"


def prediction_root(cfg, args, task):
    return Path(cfg["paths"]["run_root"]) / "predictions" / args.checkpoint / inference_profile(args) / task


def require_isolated_output(cfg, output):
    if output is None:
        raise ValueError("A separate --output-root is required")
    output = Path(output).resolve()
    formal_tree = PROJECT_ROOT / "outputs" / cfg["experiment"]
    for protected in (Path(cfg["paths"]["run_root"]).resolve(), formal_tree.resolve()):
        if output == protected or protected in output.parents:
            raise ValueError("Smoke/partial/benchmark outputs must be outside the formal run tree")
    return output


def gpu_resource_check(min_free_mib=40000):
    # Inspect metadata without allocating a CUDA context or touching running tasks.
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "reason": str(exc)}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    rows = [line.split(", ") for line in result.strip().splitlines()]
    if visible == "":
        return {"available": False, "reason": "CUDA_VISIBLE_DEVICES is empty", "snapshot": result}
    if visible:
        wanted = visible.split(",")[0]
        rows = [r for r in rows if r[0] == wanted or r[1].startswith(wanted)]
    elif len(rows) != 1:
        return {"available": False, "reason": "Choose CUDA_VISIBLE_DEVICES explicitly on multi-GPU hosts", "snapshot": result}
    okay = bool(rows) and int(rows[0][2]) >= min_free_mib
    return {"available": okay, "minimum_free_mib": min_free_mib, "snapshot": result,
            "reason": "resource preflight only; not a capacity guarantee" if okay else "Insufficient unoccupied GPU memory"}


def check(cfg, root):
    from .checks import check_data
    root.mkdir(parents=True, exist_ok=True)
    result = check_data(cfg, root / "data_check.json")
    command = [cfg["paths"]["model_python"], str(PROJECT_ROOT / "scripts/download_csgo_seen10_assets.py"),
               "--experiment", cfg["experiment"], "--check"]
    for name in ("base_checkpoint", "gemma_path", "tokenizer_path", "vae_path"):
        command += ["--" + name.replace("_", "-"), cfg["paths"][name]]
    with (root / "asset_check.log").open("w") as log:
        proc = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    result["asset_check_returncode"] = proc.returncode
    result["resources"] = gpu_resource_check()
    atomic_json(root / "check_summary.json", result)
    if proc.returncode:
        raise RuntimeError(f"Asset check failed; see {root / 'asset_check.log'}")
    return result


def smoke(cfg, args):
    if not args.run_root:
        raise ValueError("smoke requires --run-root in an independent smoke directory")
    root = Path(cfg["paths"]["run_root"])
    if "smoke" not in str(root).lower():
        raise ValueError("Use a visibly isolated smoke run root")
    formal_tree = (PROJECT_ROOT / "outputs" / cfg["experiment"]).resolve()
    if root.resolve() == formal_tree or formal_tree in root.resolve().parents:
        raise ValueError("Smoke outputs must be outside the formal experiment tree")
    root.mkdir(parents=True, exist_ok=True)
    from .checks import evaluator_fixture_smoke
    check(cfg, root / "checks")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    with (root / "cpu_tests.log").open("w") as log:
        proc = subprocess.run([sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"),
                               "-q"], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    fixture = evaluator_fixture_smoke(cfg, root / "evaluator_fixture")
    resources = gpu_resource_check()
    result = {"cpu_tests_returncode": proc.returncode, "evaluator_fixture": fixture,
              "gpu_resources": resources, "gpu": "untested", "formal": False}
    if proc.returncode or any(r["returncode"] for r in fixture.values()):
        atomic_json(root / "smoke_summary.json", result)
        raise RuntimeError(f"Smoke check failed; inspect {root}")
    if not args.cpu_only and resources["available"]:
        from .training import run_training
        run_training(cfg, micro_batch_size=args.micro_batch_size,
                     gradient_accumulation_steps=args.gradient_accumulation_steps,
                     smoke=True, smoke_steps=args.smoke_steps)
        result["gpu"] = "training smoke completed; engine parity is a separate validation command"
    atomic_json(root / "smoke_summary.json", result)
    return result


def infer(cfg, args):
    from .checkpoint import resolve_checkpoint, checkpoint_metadata
    from .data import Seen10Dataset, protocol_identity
    from .model import load_bundle
    from .inference import run_inference
    root = Path(cfg["paths"]["run_root"])
    ckpt = resolve_checkpoint(root / "train/checkpoints", args.checkpoint)
    metadata = checkpoint_metadata(ckpt)
    if metadata.get("smoke") and not args.output_root:
        raise ValueError("Smoke checkpoint inference requires an explicit isolated --output-root")
    if args.limit is not None and not args.output_root:
        raise ValueError("Partial inference requires independent --output-root")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.benchmark_batches is not None and (args.benchmark_batches < 1 or args.output_root is None):
        raise ValueError("Benchmark requires positive --benchmark-batches and a new --output-root")
    if args.benchmark_batches is not None or args.limit is not None or metadata.get("smoke"):
        require_isolated_output(cfg, args.output_root)
    if args.output_root and args.output_root.exists() and any(args.output_root.iterdir()) and args.benchmark_batches:
        raise ValueError("Benchmark output must be empty")
    # Validate scientific base/protocol identities before allocating model memory.
    data_identity = protocol_identity(cfg)
    saved = metadata["identity"]
    if saved.get("science_sha256") != cfg["identity"]["science_sha256"]:
        raise ValueError("Checkpoint scientific configuration differs")
    if saved.get("protocol") != data_identity:
        raise ValueError("Checkpoint data/prompt/protocol identity differs")
    bundle = load_bundle(cfg, device=args.device, for_training=False, checkpoint=ckpt / "adapter.pt")
    if saved.get("base") != bundle.base_identity:
        raise ValueError("Checkpoint official base/component identity differs")
    if saved.get("components") != bundle.component_identity:
        raise ValueError("Checkpoint frozen Gemma/tokenizer/VAE identity differs")
    checkpoint_id = {"metadata": metadata, "adapter_sha256": sha256_file(ckpt / "adapter.pt"),
                     "base": bundle.base_identity, "protocol": data_identity}
    results = {}
    for task in (["discrete", "continuous"] if args.task == "all" else [args.task]):
        output = args.output_root / task if args.output_root and args.task == "all" else args.output_root
        output = output or prediction_root(cfg, args, task)
        dataset = Seen10Dataset(cfg, task, load_targets=False, limit=args.limit)
        results[task] = run_inference(bundle, dataset, output, task=task, checkpoint_identity=checkpoint_id,
                                      cfg=cfg, engine=args.inference_engine, compile_mode=args.compile_mode,
                                      batch_size=args.batch_size, vae_batch_size=args.vae_batch_size,
                                      benchmark_batches=args.benchmark_batches)
    return results


def compare(cfg, args):
    from .data import Seen10Dataset, protocol_identity
    from .model import load_bundle
    from .inference import compare_engines
    if args.output_root is None:
        raise ValueError("compare requires a new isolated --output-root")
    if args.batch_size < 2:
        raise ValueError("compare requires --batch-size >=2 to test multiple samples and a tail")
    require_isolated_output(cfg, args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("compare output must be empty")
    if not gpu_resource_check()["available"]:
        raise RuntimeError("No selected GPU with sufficient free memory for bounded engine comparison")
    adapter = None
    checkpoint_id = {"base_only": True, "zero_initialized_lora": True}
    if args.checkpoint != "official":
        from .checkpoint import resolve_checkpoint, checkpoint_metadata
        ckpt = resolve_checkpoint(Path(cfg["paths"]["run_root"]) / "train/checkpoints", args.checkpoint)
        meta = checkpoint_metadata(ckpt)
        if meta["identity"]["science_sha256"] != cfg["identity"]["science_sha256"] or meta["identity"]["protocol"] != protocol_identity(cfg):
            raise ValueError("Comparison checkpoint scientific/protocol identity mismatch")
        adapter = ckpt / "adapter.pt"
        checkpoint_id = {"metadata": meta, "adapter_sha256": sha256_file(adapter)}
    bundle = load_bundle(cfg, device=args.device, for_training=False, checkpoint=adapter)
    report = {}
    for task in (["discrete", "continuous"] if args.task == "all" else [args.task]):
        dataset = Seen10Dataset(cfg, task, load_targets=False, limit=args.batch_size * 3 + 1)
        report[task] = compare_engines(bundle, dataset, args.output_root / task,
                                     task=task, checkpoint_identity=checkpoint_id, cfg=cfg,
                                     batch_size=args.batch_size, vae_batch_size=args.vae_batch_size, full_batches=3)
    return report


def validate_prediction_identity(cfg, args, task, pred_root):
    from .checkpoint import checkpoint_metadata, resolve_checkpoint
    from .data import protocol_identity
    from .inference import MARKER_NAME
    pred_root = Path(pred_root)
    task_root = pred_root.parent if pred_root.name == "gen_imgs" else pred_root
    marker = task_root / MARKER_NAME
    if not marker.is_file():
        raise ValueError(f"Missing prediction provenance: {marker}")
    identity = json.loads(marker.read_text())
    ckpt = resolve_checkpoint(Path(cfg["paths"]["run_root"]) / "train/checkpoints", args.checkpoint)
    metadata = checkpoint_metadata(ckpt)
    current_protocol = protocol_identity(cfg)
    recorded = identity.get("checkpoint", {})
    if recorded.get("metadata") != metadata or recorded.get("protocol") != current_protocol:
        raise ValueError("Prediction checkpoint metadata/protocol identity differs from the selected checkpoint and data")
    if metadata.get("identity", {}).get("protocol") != current_protocol:
        raise ValueError("Checkpoint protocol differs from the current released data")
    if identity.get("checkpoint", {}).get("adapter_sha256") != sha256_file(ckpt / "adapter.pt"):
        raise ValueError("Prediction checkpoint differs from the requested alias")
    if identity.get("science_sha256") != cfg["identity"]["science_sha256"] or identity.get("task") != task:
        raise ValueError("Prediction experiment/seed/task identity mismatch")
    expected_mode = args.compile_mode if args.inference_engine == "compiled" else None
    for key, expected in {"engine": args.inference_engine, "compile_mode": expected_mode,
                          "batch_size": args.batch_size, "vae_batch_size": args.vae_batch_size}.items():
        if identity.get(key) != expected:
            raise ValueError(f"Prediction {key} differs; pass the matching profile parameters")
    if not args.eval_smoke and (identity.get("benchmark_only") or identity.get("selection", {}).get("partial_debug") or metadata.get("smoke")):
        raise ValueError("Partial/benchmark/smoke predictions cannot be evaluated as formal results")
    return identity


def evaluate(cfg, args):
    if args.task == "all" and args.pred_root:
        raise ValueError("--pred-root requires one explicit task")
    tasks = ["discrete", "continuous"] if args.task == "all" else [args.task]
    for task in tasks:
        pred = args.pred_root or prediction_root(cfg, args, task) / "gen_imgs"
        validate_prediction_identity(cfg, args, task, pred)
        command = [cfg["paths"]["eval_python"], str(Path(cfg["paths"]["shared_eval_dir"]) / "run_eval.py")]
        command += ["smoke", task] if args.eval_smoke else [task]
        command += ["--pred-root", str(pred), "--data-root", cfg["paths"]["data_root"],
                    "--config", str(Path(cfg["paths"]["shared_eval_dir"]) / "benchmark_v2.yaml"), "--device", args.device]
        if args.eval_smoke:
            command += ["--limit", str(args.limit or 1)]
            if args.frame_only:
                command.append("--frame-only")
        else:
            out = (args.output_root / task if args.output_root and args.task == "all" else args.output_root)
            out = out or Path(cfg["paths"]["run_root"]) / "evaluation" / args.checkpoint / inference_profile(args) / task
            command += ["--output", str(out)]
        temp = PROJECT_ROOT / ".tmp"
        temp.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temp))
        subprocess.run(command, check=True, env=env)


def coverage(cfg, args):
    from PIL import Image
    from .data import Seen10Dataset
    result = {}
    if args.task == "all" and args.pred_root:
        raise ValueError("--pred-root requires one explicit task")
    for task in (["discrete", "continuous"] if args.task == "all" else [args.task]):
        root = args.pred_root or prediction_root(cfg, args, task) / "gen_imgs"
        validate_prediction_identity(cfg, args, task, root)
        ds = Seen10Dataset(cfg, task, load_targets=False)
        expected = {f'{r["map_name"]}/{r["file_frame"]}.jpg' for r in ds.rows}
        actual = {str(p.relative_to(root)) for p in root.glob("*/*.jpg")}
        bad = []
        for rel in sorted(expected & actual):
            try:
                with Image.open(root / rel) as image:
                    image.load()
                    if image.size != (448, 448) or image.mode != "RGB":
                        bad.append(rel)
            except (OSError, ValueError):
                bad.append(rel)
        result[task] = {"expected": len(expected), "present": len(actual), "valid": len(expected & actual)-len(bad),
                        "missing": sorted(expected-actual), "extra": sorted(actual-expected), "bad": bad,
                        "complete": expected == actual and not bad}
    output = args.output_root or Path(cfg["paths"]["run_root"]) / "logs" / f"coverage_{args.checkpoint}_{inference_profile(args)}.json"
    atomic_json(output, result)
    return {task: {k: v for k, v in report.items() if k not in {"missing", "extra", "bad"}} for task, report in result.items()}


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_config(args.experiment, seed=args.seed, overrides={p: getattr(args, p) for p in PATH_ENV})
    if args.command == "check":
        result = check(cfg, args.output_root or PROJECT_ROOT / "outputs/implementation_audit/check")
    elif args.command == "smoke":
        result = smoke(cfg, args)
    elif args.command == "train":
        from .config import batch_configuration
        batch_configuration(args.nproc_per_node, args.micro_batch_size, args.gradient_accumulation_steps)
        command = [cfg["paths"]["model_python"], "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node", str(args.nproc_per_node), str(PROJECT_ROOT / "train_seen10.py"),
                   "--experiment", args.experiment, "--seed", str(args.seed), "--micro-batch-size", str(args.micro_batch_size)]
        for key in PATH_ENV:
            command += ["--"+key.replace("_", "-"), cfg["paths"][key]]
        if args.gradient_accumulation_steps is not None:
            command += ["--gradient-accumulation-steps", str(args.gradient_accumulation_steps)]
        if args.resume:
            command += ["--resume", args.resume]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        result = {"training": "completed"}
    elif args.command in {"infer", "benchmark"}:
        if args.command == "benchmark" and not args.benchmark_batches:
            raise ValueError("benchmark requires --benchmark-batches")
        result = infer(cfg, args)
    elif args.command == "eval":
        evaluate(cfg, args)
        result = {"evaluation": "completed"}
    elif args.command == "compare":
        result = compare(cfg, args)
    elif args.command == "coverage":
        result = coverage(cfg, args)
    else:
        subprocess.run([cfg["paths"]["model_python"], str(PROJECT_ROOT / "scripts/plot_csgo_seen10_loss.py"),
                        "--run-root", cfg["paths"]["run_root"]], check=True)
        result = {"plot": "completed"}
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
