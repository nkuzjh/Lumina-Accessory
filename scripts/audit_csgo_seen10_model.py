#!/usr/bin/env python3
"""Bounded CPU audit of real official assets; never starts a formal run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from csgo_seen10.config import atomic_json, load_config
from csgo_seen10.cli import require_isolated_output
from csgo_seen10.checkpoint import load_checkpoint, save_checkpoint, trainable_state
from csgo_seen10.lora_audit import audit_trainable
from csgo_seen10.model import load_bundle
from csgo_seen10.training import _checkpoint_activations, make_optimizer, make_transport


def apply_probe_optimizer_update(parameters, optimizer, scheduler, *, gradient_clip: float) -> dict:
    """Use the production clipping/step order on an already-backpropagated microbatch."""
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite LoRA gradient norm in CPU structural probe")
    optimizer.step()
    scheduler.step()
    group = optimizer.param_groups[0]
    if "d" not in group:
        raise AssertionError("Prodigy did not expose adaptive d after optimizer step")
    lr, d = float(group["lr"]), float(group["d"])
    if not math.isfinite(lr) or not math.isfinite(d):
        raise FloatingPointError("Nonfinite Prodigy effective learning rate in CPU structural probe")
    return {"grad_norm": float(grad_norm), "gradient_clip": float(gradient_clip),
            "lr": lr, "prodigy_d": d, "effective_lr": lr * d,
            "scheduler_last_epoch": scheduler.last_epoch}


def write_probe_events(path: Path, steps: list[dict]) -> None:
    lines = [json.dumps({"event": "train", "structural_cpu_probe": True, **step}, sort_keys=True)
             for step in steps]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def assert_same_state(expected, actual, name="state") -> None:
    """Exact, recursive CPU-state comparison for same-process checkpoint replay."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(expected, actual):
            raise AssertionError(f"Resume mismatch at {name}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"Resume keys differ at {name}")
        for key in expected:
            assert_same_state(expected[key], actual[key], f"{name}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"Resume sequence differs at {name}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_same_state(left, right, f"{name}[{index}]")
    elif expected != actual:
        raise AssertionError(f"Resume value differs at {name}: {expected!r} != {actual!r}")


def structural_micro_step(bundle, sample, transport, optimizer, scheduler, params, recipe, step: int) -> dict:
    optimizer.zero_grad(set_to_none=True)
    before = bundle.dit.final_layer.linear.lora_B.bias.detach().clone()
    step_started = time.perf_counter()
    loss_dict = bundle.training_loss(sample["target"].unsqueeze(0), sample["radar"].unsqueeze(0),
                                     [sample["prompt"]], transport, caption_dropout=0.0)
    loss = loss_dict["loss"].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite native flow loss at structural step {step}")
    loss.backward()
    frozen_grads = [name for name, param in bundle.dit.named_parameters()
                    if not param.requires_grad and param.grad is not None]
    if frozen_grads:
        raise AssertionError(f"Frozen DiT tensors received gradients: {frozen_grads[:10]}")
    for component_name in ("text_encoder", "vae"):
        component = getattr(bundle, component_name)
        if any(p.grad is not None for p in component.parameters()):
            raise AssertionError(f"Frozen {component_name} received gradients")
    bias_grads = {name: float(param.grad.abs().sum()) for name, param in bundle.dit.named_parameters()
                  if ".lora_B.bias" in name and param.grad is not None and param.grad.abs().sum().item() > 0}
    if not bias_grads:
        raise AssertionError("No native LoRA B bias received a gradient")
    nonzero_A = sum(".lora_A." in name and parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                    for name, parameter in bundle.dit.named_parameters())
    nonzero_B_weight = sum(".lora_B.weight" in name and parameter.grad is not None and parameter.grad.abs().sum().item() > 0
                           for name, parameter in bundle.dit.named_parameters())
    update = apply_probe_optimizer_update(params, optimizer, scheduler,
                                          gradient_clip=float(recipe["gradient_clip"]))
    final_bias_update = float((bundle.dit.final_layer.linear.lora_B.bias.detach() - before).abs().sum())
    if final_bias_update == 0:
        raise AssertionError("Final native B bias did not update")
    return {"step": step, "sample_id": sample["sample_id"],
            "loss": float(loss.detach()), "seconds": time.perf_counter() - step_started,
            "nonzero_B_bias_gradient_count": len(bias_grads),
            "nonzero_A_gradient_count": nonzero_A,
            "nonzero_B_weight_gradient_count": nonzero_B_weight,
            "radar_B_bias_gradient_count": sum(n.startswith(("cond_embedder", "cond_refiner")) for n in bias_grads),
            "final_B_bias_update_l1": final_bias_update,
            "frozen_gradient_count": 0, **update}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="csgo_seen10_exp32gen_aligned")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/implementation_audit/real_model",
                        help="Fresh empty directory outside the formal run tree; use a new directory for every attempt")
    parser.add_argument("--smoke-steps", type=int, choices=(0, 1, 2), default=0,
                        help="Optional 1-2 CPU structural updates on published seen_train records; never formal training")
    parser.add_argument("--verify-resume", action="store_true",
                        help="Save production-format structural checkpoint after step 1 and replay step 2; requires --smoke-steps 2")
    parser.add_argument("--no-probe", action="store_true", help="Skip two synthetic forward calls around zero-LoRA injection")
    args = parser.parse_args(argv)
    if args.verify_resume and args.smoke_steps != 2:
        parser.error("--verify-resume requires --smoke-steps 2")
    cfg = load_config(args.experiment, seed=args.seed)
    output = require_isolated_output(cfg, args.output_root.expanduser())
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError(f"CPU audit requires a fresh empty output root: {output}")
    else:
        output.mkdir(parents=True)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    bundle = load_bundle(cfg, device="cpu", for_training=args.smoke_steps > 0,
                         audit_initial_output=not args.no_probe)
    loaded_seconds = time.perf_counter() - started
    params = [p for p in bundle.dit.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, cfg)
    recipe = cfg["training"]
    if args.smoke_steps and recipe["activation_checkpointing"]:
        _checkpoint_activations(bundle.dit)
    if recipe["scheduler"] != "constant" or recipe["warmup_steps"] != 0 or float(recipe["gradient_clip"]) != 2.0:
        raise ValueError("CPU probe requires the aligned constant/no-warmup scheduler and gradient clip 2")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    audit = audit_trainable(bundle, optimizer=optimizer, output_dir=output)
    if audit["summary"]["meta_only"]:
        raise AssertionError("Real asset audit unexpectedly used meta tensors")
    result = {"status": "real_official_cpu_audit", "formal_training": False,
              "device": "cpu", "threads": torch.get_num_threads(),
              "base_identity": bundle.base_identity, "component_identity": bundle.component_identity,
              "zero_lora_probe": bundle.initial_probe, "summary": audit["summary"],
              "condition_injection_audit": bundle.condition_injection_audit,
              "load_seconds": loaded_seconds, "optimizer_for_structural_audit": "Prodigy; exact aligned recipe",
              "structural_probe_config": {"micro_batch": 1, "effective_batch": 1, "caption_dropout": 0.0,
                                          "gradient_clip": 2.0, "scheduler": "constant", "warmup_steps": 0,
                                          "activation_checkpointing": bool(args.smoke_steps and recipe["activation_checkpointing"]),
                                          "exposure_accounting": "not part of aligned 128-sample training budget"},
              "smoke_steps_requested": args.smoke_steps, "steps": []}
    if args.verify_resume:
        result["resume_verification"] = {"status": "pending", "scope": "same-process, same frozen bundle; not cold-process resume",
                                          "checkpoint_use": "structural CPU acceptance only; never formal training resume"}
    atomic_json(output / "model_real_audit.json", result)
    if args.smoke_steps:
        from csgo_seen10.data import Seen10Dataset

        dataset = Seen10Dataset(cfg, "train", load_targets=True, limit=args.smoke_steps)
        transport = make_transport(cfg)
        bundle.dit.train()
        checkpoint_root = output / "structural_resume" / "checkpoints"
        probe_identity = {"purpose": "cpu_structural_probe_only", "effective_batch": 1,
                          "seed": cfg["seed"], "science_sha256": cfg["identity"]["science_sha256"],
                          "base_identity": bundle.base_identity, "component_identity": bundle.component_identity,
                          "micro_batch": 1, "caption_dropout": 0.0}
        probe_topology = {"world_size": 1, "micro_batch_size": 1, "gradient_accumulation_steps": 1,
                          "device": "cpu", "purpose": "structural_probe"}
        checkpoint_path = None
        for step in range(args.smoke_steps):
            sample = dataset[step]
            event = structural_micro_step(bundle, sample, transport, optimizer, scheduler, params, recipe, step + 1)
            result["steps"].append(event)
            atomic_json(output / "model_real_audit.json", result)
            write_probe_events(output / "cpu_probe_events.jsonl", result["steps"])
            if args.verify_resume and step == 0:
                checkpoint_path = save_checkpoint(checkpoint_root, model=bundle.dit, optimizer=optimizer,
                                                  scheduler=scheduler, step=1, exposures=1,
                                                  sampler_state={"sample_ids": [sample["sample_id"]], "next_index": 1},
                                                  identity=probe_identity, topology=probe_topology,
                                                  validation_loss=None, smoke=True)
                result["resume_verification"]["checkpoint"] = str(checkpoint_path)
                atomic_json(output / "model_real_audit.json", result)
            if args.verify_resume and step == 1:
                try:
                    expected_trainable = trainable_state(bundle.dit)
                    expected_optimizer = copy.deepcopy(optimizer.state_dict())
                    expected_scheduler = copy.deepcopy(scheduler.state_dict())
                    recovered = load_checkpoint(checkpoint_path, model=bundle.dit, optimizer=optimizer,
                                                scheduler=scheduler, identity=probe_identity,
                                                topology=probe_topology, allow_topology_change=False)
                    if recovered["step"] != 1 or recovered["exposures"] != 1 or recovered["topology_changed"]:
                        raise AssertionError("Structural checkpoint did not restore the first optimizer boundary")
                    if recovered["sampler"] != {"sample_ids": [result["steps"][0]["sample_id"]], "next_index": 1}:
                        raise AssertionError("Structural checkpoint sampler identity differed")
                    replay = structural_micro_step(bundle, dataset[1], transport, optimizer, scheduler,
                                                   params, recipe, 2)
                    assert_same_state(event["sample_id"], replay["sample_id"], "sample_id")
                    assert_same_state(event["loss"], replay["loss"], "loss")
                    assert_same_state(event["prodigy_d"], replay["prodigy_d"], "prodigy_d")
                    assert_same_state(expected_trainable, trainable_state(bundle.dit), "adapter")
                    assert_same_state(expected_optimizer, optimizer.state_dict(), "optimizer")
                    assert_same_state(expected_scheduler, scheduler.state_dict(), "scheduler")
                except Exception as error:
                    result["resume_verification"].update({"status": "failed", "error":
                                                          {"type": type(error).__name__, "message": str(error)}})
                    atomic_json(output / "model_real_audit.json", result)
                    raise
                result["resume_verification"].update({"status": "passed", "same_process": True,
                                                      "sample_id": replay["sample_id"], "loss": replay["loss"],
                                                      "prodigy_d": replay["prodigy_d"],
                                                      "adapter_tensors_equal": len(expected_trainable),
                                                      "optimizer_state_equal": True,
                                                      "scheduler_state_equal": True,
                                                      "micro_updates_total": 3})
                atomic_json(output / "model_real_audit.json", result)
    result["total_seconds"] = time.perf_counter() - started
    atomic_json(output / "model_real_audit.json", result)
    print(result["summary"])
    print(f"Real CPU audit saved to {output / 'model_real_audit.json'}")


if __name__ == "__main__":
    main()
