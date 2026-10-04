"""Fixed-exposure CSGO training, distributed validation and exact-boundary resume."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .checkpoint import MILESTONES, checkpoint_metadata, load_checkpoint, resolve_checkpoint, save_checkpoint
from .config import PROJECT_ROOT, atomic_json, batch_configuration, code_identity
from .data import Seen10Dataset, collate_samples, protocol_identity


def distributed_context():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world, device


class GlobalSourceStream:
    """One seed-defined permutation stream; the last partial epoch joins the next."""

    def __init__(self, length, seed, effective_batch=128):
        if length < 1 or effective_batch < 1:
            raise ValueError("Source stream requires positive dataset length and batch")
        self.length, self.seed, self.effective_batch = int(length), int(seed), int(effective_batch)
        self._epoch = None
        self._permutation = None

    def index(self, absolute_position):
        epoch, offset = divmod(int(absolute_position), self.length)
        if epoch != self._epoch:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
            self._permutation = torch.randperm(self.length, generator=generator).tolist()
            self._epoch = epoch
        return self._permutation[offset]

    def micro_indices(self, step, rank, world, micro_batch, accumulation, micro_index):
        if world * micro_batch * accumulation != self.effective_batch:
            raise ValueError("World/micro/accum no longer matches global batch")
        start = step * self.effective_batch + (micro_index * world + rank) * micro_batch
        return [self.index(start + i) for i in range(micro_batch)]

    def state_after(self, step):
        offset = int(step) * self.effective_batch
        epoch, position = divmod(offset, self.length)
        return {"seed": self.seed, "dataset_length": self.length,
                "global_offset": offset, "epoch": epoch, "position_in_epoch": position,
                "effective_batch": self.effective_batch}

    def check_resume(self, state, step):
        if state != self.state_after(step):
            raise ValueError("Sampler seed, length, or global offset changed")


def make_optimizer(parameters, cfg):
    import prodigyopt

    parameters = list(parameters)
    if not parameters or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters):
        raise ValueError("Optimizer must contain only FP32 trainable LoRA parameters")
    recipe = cfg["optimizer"]
    if recipe["name"] != "Prodigy":
        raise ValueError("Aligned recipe requires Prodigy")
    return prodigyopt.Prodigy(parameters, lr=recipe["lr"], betas=tuple(recipe["betas"]),
                             eps=recipe["eps"], weight_decay=recipe["weight_decay"],
                             decouple=recipe["decouple"], use_bias_correction=recipe["use_bias_correction"],
                             safeguard_warmup=recipe["safeguard_warmup"], slice_p=recipe["slice_p"],
                             d0=recipe["d0"], d_coef=recipe["d_coef"], beta3=recipe["beta3"],
                             growth_rate=float(recipe["growth_rate"]), fsdp_in_use=recipe["fsdp_in_use"])


def make_transport(cfg):
    from transport import create_transport

    train = cfg["training"]
    return create_transport(train["path_type"], train["prediction"], snr_type=train["snr_type"],
                            do_shift=train["do_shift"], seq_len=train["seq_len"])


def _checkpoint_activations(model):
    from torch.utils.checkpoint import checkpoint

    targets = list(model.get_checkpointing_wrap_module_list())
    if not targets:
        raise ValueError("Activation checkpointing requested, but model has no target blocks")
    for block in targets:
        original = block.forward

        def recompute(*args, _original=original, _block=block, **kwargs):
            if not _block.training or not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint(_original, *args, use_reentrant=False, **kwargs)

        # Keep the original module tree and LoRA parameter names stable for inference.
        block.forward = recompute


def _fixed_validation_transport(transport, sample_indices, seed):
    """Generate t and x0 per sample identity, invariant to rank and batching."""
    fixed = copy.copy(transport)

    def sample(x1):
        if len(x1) != len(sample_indices):
            raise ValueError("Validation index/batch mismatch")
        x0, u = [], []
        for index, value in zip(sample_indices, x1):
            def domain_seed(domain):
                payload = json.dumps([int(seed), int(index), domain], separators=(",", ":")).encode()
                return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

            # Separate deterministic streams keep the timestep draw independent
            # of latent noise even when both generators run on the CPU.
            generator = torch.Generator(device=value.device).manual_seed(domain_seed("noise"))
            x0.append(torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator))
            cpu_generator = torch.Generator(device="cpu").manual_seed(domain_seed("timestep"))
            u.append(torch.randn((), generator=cpu_generator))
        x0 = torch.stack(x0)
        u = torch.stack(u)
        t0, t1 = fixed.check_interval(fixed.train_eps, fixed.sample_eps)
        if fixed.snr_type != "lognorm":
            raise ValueError("Fixed validation implemented for the reviewed lognorm recipe")
        t = torch.sigmoid(u) * (t1 - t0) + t0
        if fixed.do_shift:
            mu = fixed.get_lin_function(y1=0.5, y2=1.15)(fixed.seq_len)
            t = fixed.time_shift(mu, 1.0, t)
        return t.to(x1), x0, x1

    fixed.sample = sample
    return fixed


@torch.no_grad()
def validate(bundle, dataset, transport, *, rank, world, micro_batch_size, device, seed):
    wrapped = bundle.dit
    # Validation shards can have unequal batch counts. Avoid DDP forward collectives.
    bundle.dit = wrapped.module if isinstance(wrapped, DDP) else wrapped
    was_training = bundle.dit.training
    bundle.dit.eval()
    total = torch.zeros(2, dtype=torch.float64, device=device)
    progress = None
    try:
        indices = list(range(rank, len(dataset), world))
        progress = tqdm(total=math.ceil(len(indices) / micro_batch_size), desc="Validation (rank 0 shard)",
                        unit="batch", leave=False, disable=rank != 0)
        for start in range(0, len(indices), micro_batch_size):
            subset = indices[start:start + micro_batch_size]
            batch = collate_samples([dataset[i] for i in subset])
            fixed = _fixed_validation_transport(transport, subset, seed)
            losses = bundle.training_loss(batch["target"], batch["radar"], batch["prompt"], fixed,
                                          caption_dropout=0.0)["loss"].detach().double()
            if losses.numel() != len(subset):
                raise ValueError("Validation loss is not per sample")
            total += torch.stack((losses.sum(), torch.tensor(len(subset), device=device, dtype=torch.float64)))
            progress.update(1)
        if world > 1:
            dist.all_reduce(total)
        if int(total[1].item()) != len(dataset):
            raise AssertionError(f"Validation counted {total[1].item()} of {len(dataset)} source records")
        return float((total[0] / total[1]).item()), int(total[1].item())
    finally:
        if progress is not None:
            progress.close()
        bundle.dit.train(was_training)
        bundle.dit = wrapped


def _identity(cfg, bundle, protocol, effective_batch):
    return {"experiment": cfg["experiment"], "seed": cfg["seed"],
            "science_sha256": cfg["identity"]["science_sha256"],
            "base": bundle.base_identity, "components": bundle.component_identity,
            "protocol": protocol,
            "code": code_identity(), "effective_batch": effective_batch}


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _append_log(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def run_training(cfg, *, micro_batch_size, gradient_accumulation_steps=None,
                 resume=None, smoke=False, smoke_steps=2, smoke_limit=128,
                 validation_limit=8):
    """Run only on explicit CLI invocation. A smoke run uses an isolated root."""
    rank, world, device = distributed_context()
    recipe = cfg["training"]
    effective = int(recipe["effective_batch"])
    accumulation = batch_configuration(world, micro_batch_size, gradient_accumulation_steps, effective)
    if smoke and (smoke_steps < 1 or smoke_steps > 3):
        raise ValueError("Smoke is limited to one through three optimizer updates")
    root = Path(cfg["paths"]["run_root"])
    if smoke and "smoke" not in root.parts and "smoke" not in root.name:
        raise ValueError("Smoke requires an isolated run root containing 'smoke'")
    formal_tree = (PROJECT_ROOT / "outputs" / cfg["experiment"]).resolve()
    if smoke and (root.resolve() == formal_tree or formal_tree in root.resolve().parents):
        raise ValueError("Smoke outputs must be outside the formal experiment tree")
    if not smoke and any("smoke" in part.lower() for part in root.parts):
        raise ValueError("Formal training cannot write in a smoke root")
    ckpt_root = root / "train/checkpoints"
    if not smoke and resume is None and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Nonempty training run requires --resume or a new run root: {root}")
    if resume:
        selected = resolve_checkpoint(ckpt_root, resume)
        selected_metadata = checkpoint_metadata(selected)
        if bool(selected_metadata["smoke"]) != bool(smoke):
            raise ValueError("Smoke and formal checkpoint histories must remain separate")
    seed = int(cfg["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(recipe["tf32"])
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    from .model import load_bundle

    bundle = load_bundle(cfg, device=device, for_training=True)
    raw_model = bundle.dit
    if recipe["activation_checkpointing"]:
        _checkpoint_activations(raw_model)
    train_set = Seen10Dataset(cfg, "seen_train", limit=smoke_limit if smoke else None)
    val_set = Seen10Dataset(cfg, "seen_validation", limit=validation_limit if smoke else None)
    if not smoke and (len(train_set) != 50000 or len(val_set) != 5000):
        raise ValueError("Released Seen-10 train/validation count mismatch")
    stream = GlobalSourceStream(len(train_set), cfg["seed"], effective)
    protocol = protocol_identity(cfg)
    identity = _identity(cfg, bundle, protocol, effective)
    if resume:
        if selected_metadata["identity"] != identity:
            raise ValueError("Checkpoint experiment/base/component/protocol/recipe/source identity mismatch")
        if selected.parent != ckpt_root.resolve():
            # Branching from an older checkpoint keeps its immutable history visible
            # to best selection without copying multi-GB optimizer state.
            branch_error = None
            if rank == 0:
                try:
                    ckpt_root.mkdir(parents=True, exist_ok=True)
                    if any(ckpt_root.iterdir()):
                        raise FileExistsError("External checkpoint branch requires an empty checkpoint root")
                    inherited = {selected_metadata["step"], selected_metadata["best_step"]}
                    for inherited_step in sorted(s for s in inherited if s is not None):
                        source = selected.parent / f"step_{inherited_step:08d}"
                        checkpoint_metadata(source)
                        (ckpt_root / source.name).symlink_to(source)
                    atomic_json(root / "provenance/branch.json", {
                        "source_checkpoint": str(selected), "source_step": selected_metadata["step"],
                        "inherited_steps": sorted(s for s in inherited if s is not None),
                        "external_checkpoint_dependency": True,
                    })
                except Exception as exc:
                    branch_error = f"{type(exc).__name__}: {exc}"
            if world > 1:
                notice = [branch_error]
                dist.broadcast_object_list(notice, src=0)
                branch_error = notice[0]
            if branch_error:
                raise RuntimeError(f"Checkpoint branch setup failed: {branch_error}")
    topology = {"world_size": world, "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation, "device_type": device.type,
                "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0}
    optimizer = make_optimizer((p for p in raw_model.parameters() if p.requires_grad), cfg)
    from .lora_audit import audit_trainable

    audit_trainable(bundle, optimizer, root / "provenance" if rank == 0 else None)
    if rank == 0:
        atomic_json(root / "train/resolved_config.json", cfg)
        atomic_json(root / "config_resolved.json", cfg)
        atomic_json(root / "provenance/identity.json", identity)
        atomic_json(root / "provenance/environment.json", {
            "python": sys.version, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "prodigyopt": importlib.metadata.version("prodigyopt"),
            "topology": topology, "optimizer_defaults": _jsonable(optimizer.defaults),
            "optimizer_groups": _jsonable([{k: v for k, v in group.items() if k != "params"}
                                            for group in optimizer.param_groups]),
            "tf32": bool(recipe["tf32"]), "amp": "bf16 CUDA autocast; disabled CPU; no scaler",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })
    if world > 1:
        bundle.dit = DDP(raw_model, device_ids=[device.index] if device.type == "cuda" else None,
                         broadcast_buffers=False, find_unused_parameters=False)
    if recipe["scheduler"] != "constant" or recipe["warmup_steps"] != 0:
        raise ValueError("Aligned scheduler must be constant without external warmup")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    transport = make_transport(cfg)
    start_step = 0
    if resume:
        checkpoint_path = resolve_checkpoint(ckpt_root, resume)
        if not smoke and any(p.is_dir() and (p / "COMPLETE").is_file() and
                             checkpoint_metadata(p)["step"] > checkpoint_metadata(checkpoint_path)["step"]
                             for p in ckpt_root.glob("step_*")):
            raise ValueError("Older-step rollback requires a new run root")
        resumed = load_checkpoint(checkpoint_path, model=raw_model, optimizer=optimizer,
                                  scheduler=scheduler, identity=identity, topology=topology)
        start_step = resumed["step"]
        stream.check_resume(resumed["sampler"], start_step)
        if rank == 0:
            _append_log(root / "train/events.jsonl", {"event": "resume", **resumed})
    limit = smoke_steps if smoke else int(recipe["max_steps"])
    if not smoke and (limit != 19500 or tuple(recipe["save_steps"]) != MILESTONES):
        raise ValueError("Aligned training budget or milestones changed")
    if start_step >= limit:
        return {"step": start_step, "status": "already_complete"}
    progress = tqdm(total=limit, initial=start_step, desc="Training", unit="step", disable=rank != 0)
    try:
        for step_index in range(start_step, limit):
            started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            local_sum = torch.zeros((), dtype=torch.float64, device=device)
            ids = []
            for micro_index in range(accumulation):
                indices = stream.micro_indices(step_index, rank, world, micro_batch_size, accumulation, micro_index)
                batch = collate_samples([train_set[i] for i in indices])
                ids.extend(batch["sample_id"])
                sync = micro_index == accumulation - 1
                context = contextlib.nullcontext() if sync or world == 1 else bundle.dit.no_sync()
                with context:
                    loss_vector = bundle.training_loss(batch["target"], batch["radar"], batch["prompt"], transport,
                                                       caption_dropout=recipe["caption_dropout"])["loss"]
                    if loss_vector.numel() != micro_batch_size or not torch.isfinite(loss_vector).all():
                        raise FloatingPointError("Nonfinite or incorrectly sized training loss")
                    local_sum += loss_vector.detach().double().sum()
                    (loss_vector.float().sum() / (micro_batch_size * accumulation)).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_((p for p in raw_model.parameters() if p.requires_grad),
                                                       float(recipe["gradient_clip"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite LoRA gradient norm")
            optimizer.step()
            scheduler.step()
            step = step_index + 1
            exposures = step * effective
            if world > 1:
                dist.all_reduce(local_sum)
            event = {"event": "train", "step": step, "exposures": exposures,
                     "loss": float((local_sum / effective).item()), "grad_norm": float(grad_norm),
                     "lr": optimizer.param_groups[0]["lr"],
                     "prodigy_d": (float(optimizer.param_groups[0]["d"])
                                   if "d" in optimizer.param_groups[0] else None),
                     "effective_lr": (float(optimizer.param_groups[0]["lr"] * optimizer.param_groups[0]["d"])
                                      if "d" in optimizer.param_groups[0] else None),
                     "seconds": time.monotonic() - started,
                     "samples_this_rank": ids if smoke else None,
                     "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
            event["samples_per_second"] = effective / event["seconds"]
            if rank == 0:
                _append_log(root / "train/events.jsonl", event)
                progress.set_postfix(loss=f"{event['loss']:.4g}", grad=f"{event['grad_norm']:.4g}",
                                     lr=f"{event['lr']:.3g}", refresh=False)
            progress.update(1)
            if (smoke and step == limit) or (not smoke and step in MILESTONES):
                validation_started = time.monotonic()
                validation_loss, validation_count = validate(bundle, val_set, transport, rank=rank, world=world,
                                                             micro_batch_size=micro_batch_size, device=device,
                                                             seed=recipe["validation_seed"])
                validation_seconds = time.monotonic() - validation_started
                if rank == 0:
                    _append_log(root / "train/events.jsonl", {"event": "validation", "step": step,
                                "loss": validation_loss, "count": validation_count,
                                "validation_seed": recipe["validation_seed"], "seconds": validation_seconds})
                    tqdm.write(f"Validation step={step} loss={validation_loss:.6g} count={validation_count} "
                               f"seconds={validation_seconds:.1f}", file=sys.stderr)
                save_started = time.monotonic()
                saved_path = save_checkpoint(ckpt_root, model=raw_model, optimizer=optimizer, scheduler=scheduler,
                                             step=step, exposures=exposures, sampler_state=stream.state_after(step),
                                             identity=identity, topology=topology, validation_loss=validation_loss,
                                             smoke=smoke)
                save_seconds = time.monotonic() - save_started
                if rank == 0:
                    _append_log(root / "train/events.jsonl", {"event": "checkpoint", "step": step,
                                "path": str(saved_path), "seconds": save_seconds})
                    tqdm.write(f"Checkpoint step={step} path={saved_path} seconds={save_seconds:.1f}",
                               file=sys.stderr)
    finally:
        progress.close()
    return {"step": limit, "exposures": limit * effective, "checkpoint_root": str(ckpt_root)}
