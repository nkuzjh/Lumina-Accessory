"""Complete, atomic optimizer-boundary checkpoints for the Seen-10 experiment."""
from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import atomic_json, sha256_file

MILESTONES = (4000, 8000, 12000, 16000, 19500)


def _distributed():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _distributed() else 0


def _world():
    return dist.get_world_size() if _distributed() else 1


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def trainable_state(model):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_trainable_state(model, state):
    trainables = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(trainables):
        raise ValueError(f"LoRA key mismatch: missing={sorted(set(trainables)-set(state))[:8]}, extra={sorted(set(state)-set(trainables))[:8]}")
    with torch.no_grad():
        for name, parameter in trainables.items():
            if state[name].shape != parameter.shape:
                raise ValueError(f"LoRA shape mismatch: {name}")
            parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))


def _atomic_alias(root, name, target):
    link = root / name
    temp = root / f".{name}.{uuid.uuid4().hex}.tmp"
    os.symlink(os.path.relpath(target, root), temp)
    try:
        os.replace(temp, link)
    finally:
        temp.unlink(missing_ok=True)


def resolve_checkpoint(root, name="latest"):
    root = Path(root)
    path = Path(name)
    if not path.is_absolute() and len(path.parts) == 1:
        path = root / path
    if not path.is_dir() or not (path / "COMPLETE").is_file():
        raise FileNotFoundError(f"Complete checkpoint missing: {path}")
    return path.resolve()


def checkpoint_metadata(path):
    path = Path(path)
    if not (path / "COMPLETE").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    metadata = json.loads((path / "metadata.json").read_text())
    for name, expected in metadata["payload_sha256"].items():
        actual = sha256_file(path / name)
        if actual != expected:
            raise ValueError(f"Checkpoint payload hash mismatch: {path / name}")
    return metadata


def best_saved(root):
    root = Path(root)
    candidates = [checkpoint_metadata(path) for path in root.glob("step_*")
                  if (path / "COMPLETE").is_file()]
    candidates = [m for m in candidates if m["validation_loss"] is not None]
    return min(candidates, key=lambda m: (m["validation_loss"], m["step"])) if candidates else None


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(root, *, model, optimizer, scheduler, step, exposures,
                    sampler_state, identity, topology, validation_loss,
                    best=None, smoke=False):
    """All ranks enter; rank zero publishes a complete directory then aliases."""
    if step < 1 or exposures != step * int(identity["effective_batch"]):
        raise ValueError("Save must occur at a completed optimizer boundary")
    if not smoke and step not in MILESTONES:
        raise ValueError("Formal checkpoint outside the five milestones")
    if not smoke and validation_loss is None:
        raise ValueError("Formal milestone requires complete validation")
    rank_states = [None] * _world()
    state = {"rank": _rank(), "rng": capture_rng()}
    if _distributed():
        dist.all_gather_object(rank_states, state)
    else:
        rank_states[0] = state
    root = Path(root)
    path = root / f"step_{step:08d}"
    error = None
    if _rank() == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Checkpoint is immutable: {path}")
            temp = root / f".{path.name}.{uuid.uuid4().hex}.tmp"
            temp.mkdir()
            try:
                torch.save(trainable_state(model), temp / "adapter.pt")
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "rank_states": rank_states, "scaler": {"enabled": False},
                            "global_step": step, "exposures": exposures,
                            "sampler": sampler_state, "topology": topology}, temp / "state.pt")
                prior = best if best is not None else best_saved(root)
                is_best = validation_loss is not None and (prior is None or
                          float(validation_loss) < float(prior["validation_loss"]))
                best_step = step if is_best else (prior["step"] if prior else None)
                best_loss = float(validation_loss) if is_best else (prior["validation_loss"] if prior else None)
                metadata = {"step": step, "exposures": exposures, "validation_loss": validation_loss,
                            "best_step": best_step, "best_loss": best_loss,
                            "identity": identity, "topology": topology, "sampler": sampler_state,
                            "smoke": bool(smoke), "rng_resume": "bitwise only under identical software/hardware/topology",
                            "payload_sha256": {name: sha256_file(temp / name) for name in ("adapter.pt", "state.pt")}}
                for name in ("adapter.pt", "state.pt"):
                    with (temp / name).open("rb") as handle:
                        os.fsync(handle.fileno())
                atomic_json(temp / "metadata.json", metadata)
                with (temp / "COMPLETE").open("w") as handle:
                    handle.write("complete\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                _sync_directory(temp)
                os.replace(temp, path)
                _sync_directory(root)
                chosen_best = root / f"step_{best_step:08d}" if best_step is not None else None
                for directory in (root, root.parent):
                    _atomic_alias(directory, "latest", path)
                    if chosen_best is not None:
                        _atomic_alias(directory, "best", chosen_best)
                    if step == MILESTONES[-1] and not smoke:
                        _atomic_alias(directory, "late", path)
                    _sync_directory(directory)
            finally:
                if temp.exists():
                    shutil.rmtree(temp)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if _distributed():
        notice = [error]
        dist.broadcast_object_list(notice, src=0)
        error = notice[0]
    if error:
        raise RuntimeError(f"Checkpoint publication failed: {error}")
    return path


def load_checkpoint(path, *, model, optimizer, scheduler, identity, topology,
                    allow_topology_change=True):
    path = resolve_checkpoint(Path(path).parent, Path(path).name)
    metadata = checkpoint_metadata(path)
    if metadata["identity"] != identity:
        raise ValueError("Checkpoint experiment/base/protocol/recipe/source identity mismatch")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    saved_topology = state["topology"]
    changed = saved_topology != topology
    if changed and not allow_topology_change:
        raise ValueError("Checkpoint topology differs")
    if not changed and len(state["rank_states"]) != _world():
        raise ValueError("Checkpoint rank RNG count differs")
    load_trainable_state(model, torch.load(path / "adapter.pt", map_location="cpu", weights_only=True))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if changed:
        # Same global stream offset; each new rank gets a documented fresh RNG.
        seed = int(identity["seed"]) + 1000003 * int(state["global_step"]) + _rank()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        restore_rng(state["rank_states"][_rank()]["rng"])
    return {"step": int(state["global_step"]), "exposures": int(state["exposures"]),
            "sampler": state["sampler"], "topology_changed": changed,
            "resume_precision": "non-bitwise: changed rank partition/reduction/RNG" if changed else
                                "same-topology RNG state restored"}
