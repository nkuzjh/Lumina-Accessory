"""Strict official Accessory initialization and frozen component encoders."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from models_accessory import NextDiT_2B_GQA_patch2_Adaln_Refiner
from models_accessory.lora import LinearLora, replace_linear_with_lora

OFFICIAL_SHA256 = "b787a35ab72e8fe14b908e3795c08baa81556dd6b219b0bdc8c495c18386e700"
OFFICIAL_REVISION = "711d5d6656c62957e8625b02ea53cc74f2c5589d"
OFFICIAL_CODE_REVISION = "f260ba28dcd76fd6176036e9657de1bce1081a68"
MODEL_NAME = "NextDiT_2B_GQA_patch2_Adaln_Refiner"
VAE_SCALE = 0.3611
VAE_SHIFT = 0.1159


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_official_components(paths: dict) -> dict:
    manifest = json.loads((Path(__file__).parents[1] / "scripts/csgo_seen10_assets.json").read_text())
    repository = next(r for r in manifest["repositories"] if r["target"] == "components")
    roots = {"text_encoder": Path(paths["gemma_path"]), "tokenizer": Path(paths["tokenizer_path"]), "vae": Path(paths["vae_path"])}
    identities = {}
    for record in repository["files"]:
        component, filename = record["path"].split("/", 1)
        path = roots[component] / filename
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise FileNotFoundError(f"Pinned official component missing or wrong size: {path}")
        digest = sha256_file(path) if record["digest_type"] == "sha256" else _git_blob_sha1(path)
        if digest != record["digest"]:
            raise ValueError(f"Pinned official component digest mismatch: {path}")
        identities[record["path"]] = {"digest_type": record["digest_type"], "digest": digest}
    return {"repo": repository["repo_id"], "revision": repository["revision"], "files": identities}


def _model_config(cfg: dict) -> dict:
    config = cfg.get("model", {})
    name = config.get("name", config.get("architecture", MODEL_NAME))
    if name != MODEL_NAME:
        raise ValueError(f"Unsupported official Accessory architecture: {name}")
    if config.get("qk_norm", True) is not True:
        raise ValueError("Official normal Accessory checkpoint requires qk_norm=True")
    return {"name": name, "in_channels": 16, "qk_norm": True, "cap_feat_dim": 2304}


def construct_dit(cfg: dict, *, meta: bool = False) -> nn.Module:
    """Use explicit reviewed architecture, never unpickle model_args.pth."""
    spec = _model_config(cfg)
    with torch.device("meta" if meta else "cpu"):
        return NextDiT_2B_GQA_patch2_Adaln_Refiner(
            in_channels=spec["in_channels"],
            qk_norm=spec["qk_norm"],
            cap_feat_dim=spec["cap_feat_dim"],
        )


def _unwrap_official_state(state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError("Official checkpoint must be a tensor state dictionary")
    if len(state) == 1 and next(iter(state)) in {"model", "state_dict"}:
        state = next(iter(state.values()))
    if not isinstance(state, dict) or not state:
        raise ValueError("Empty or unsupported official checkpoint")
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise TypeError("Official checkpoint contains non-tensor values")
    if not any(k.startswith("cond_embedder.") for k in state):
        raise ValueError("Official checkpoint lacks trained cond_embedder")
    if not any(k.startswith("cond_refiner.") for k in state):
        raise ValueError("Official checkpoint lacks trained cond_refiner")
    return state


def load_official_dit(path: str | Path, cfg: dict, *, device: str | torch.device = "cpu", verify_hash: bool = True) -> tuple[nn.Module, dict]:
    path = Path(path)
    base_cfg = cfg.get("base", {})
    if base_cfg and (base_cfg.get("repo") != "Alpha-VLLM/Lumina-Accessory" or base_cfg.get("revision") != OFFICIAL_REVISION or base_cfg.get("sha256") != OFFICIAL_SHA256 or base_cfg.get("variant") != "normal"):
        raise ValueError("Resolved science config does not identify the reviewed official normal base")
    if path.name != "consolidated.00-of-01.pth":
        raise ValueError("Expected the official normal consolidated.00-of-01.pth; EMA is a separate experiment")
    actual_hash = sha256_file(path) if verify_hash else None
    if verify_hash and actual_hash != OFFICIAL_SHA256:
        raise ValueError(f"Official normal Accessory SHA256 mismatch: {actual_hash}")
    model = construct_dit(cfg, meta=True)
    # weights_only forbids arbitrary pickle execution. model_args.pth is never loaded.
    state = _unwrap_official_state(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"Official key mismatch: missing={sorted(set(expected)-set(state))[:20]}, unexpected={sorted(set(state)-set(expected))[:20]}")
    mismatches = {k: (tuple(state[k].shape), tuple(expected[k].shape)) for k in state if state[k].shape != expected[k].shape}
    if mismatches:
        raise ValueError(f"Official shape mismatch: {list(mismatches.items())[:20]}")
    # Asset identity belongs to the published FP32 tensors. Compute it before
    # any CPU/GPU precision conversion; compute-dtype hashes are checked later
    # only for the local LoRA injection invariant.
    condition_source_hash = _condition_state_sha256(state, require_fp32=True)
    model.load_state_dict(state, strict=True, assign=True)
    # The released checkpoint is FP32. Frozen compute uses BF16 on CUDA.
    compute_dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = model.to(device=device, dtype=compute_dtype).eval()
    model.requires_grad_(False)
    identity = {
        "repository": "Alpha-VLLM/Lumina-Accessory",
        "revision": OFFICIAL_REVISION,
        "code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint": path.name,
        "sha256": actual_hash or OFFICIAL_SHA256,
        "selection": "normal",
        "model_config": _model_config(cfg),
        "condition_sha256": condition_source_hash,
    }
    return model, identity


def inject_native_lora(model: nn.Module, *, rank: int = 128, scale: float = 1.0) -> list[str]:
    if rank != 128 or scale != 1.0:
        raise ValueError("Aligned LoRA contract is rank 128, scale 1.0")
    if any(isinstance(m, LinearLora) for m in model.modules()):
        raise ValueError("LoRA already injected; resume must load the adapter into the existing structure")
    before = {name: id(p) for name, p in model.named_parameters()}
    replace_linear_with_lora(model, max_rank=rank, scale=scale, lora_dtype=torch.float32)
    targets = [name for name, m in model.named_modules() if isinstance(m, LinearLora)]
    for name, param in model.named_parameters():
        param.requires_grad_(".lora_A." in name or ".lora_B." in name)
        if name in before and id(param) != before[name]:
            raise AssertionError(f"Base parameter replaced during LoRA injection: {name}")
        if param.requires_grad and param.dtype != torch.float32:
            raise AssertionError(f"LoRA is not FP32: {name}")
    if len(targets) != 197:
        raise AssertionError(f"Expected 197 native Linear targets, found {len(targets)}")
    return targets


def _condition_state_sha256(state: dict[str, torch.Tensor], *, require_fp32: bool = False) -> str:
    """Hash named condition tensors in their current representation."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if not name.startswith(("cond_embedder.", "cond_refiner.")) or ".lora_" in name:
            continue
        tensor = state[name]
        if require_fp32 and tensor.dtype != torch.float32:
            raise ValueError(f"Published normal Accessory condition tensor is not FP32: {name}: {tensor.dtype}")
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode())
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def condition_sha256(model: nn.Module) -> str:
    """Hash current compute-dtype condition weights to check LoRA injection."""
    return _condition_state_sha256(model.state_dict())


def merge_native_lora(model: nn.Module) -> nn.Module:
    """Merge both B weight and B bias; preserve bias even for biasless base Linear."""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if not isinstance(child, LinearLora):
                continue
            weight = child.weight.detach().float() + child.scale * (child.lora_B.weight.detach().float() @ child.lora_A.weight.detach().float())
            base_bias = child.bias.detach().float() if child.bias is not None else torch.zeros(child.out_features, device=weight.device)
            bias = base_bias + child.scale * child.lora_B.bias.detach().float()
            merged = nn.Linear(child.in_features, child.out_features, bias=True, device=weight.device, dtype=child.weight.dtype)
            merged.weight.data.copy_(weight.to(merged.weight.dtype))
            merged.bias.data.copy_(bias.to(merged.bias.dtype))
            setattr(parent, name, merged)
    return model


@dataclass
class AccessoryBundle:
    dit: nn.Module
    text_encoder: nn.Module
    tokenizer: Any
    vae: nn.Module
    base_identity: dict
    component_identity: dict | None = None
    initial_probe: dict | None = None
    condition_injection_audit: dict | None = None

    @property
    def device(self) -> torch.device:
        return next(self.dit.parameters()).device

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts or any(not isinstance(p, str) for p in prompts):
            raise ValueError("encode_text requires a nonempty list of strings")
        tokens = self.tokenizer(list(prompts), padding=True, pad_to_multiple_of=8, truncation=False, return_tensors="pt")
        mask = tokens["attention_mask"]
        if mask.shape[1] > 256 or mask.sum(dim=1).max().item() > 256:
            raise ValueError(f"Prompt exceeds official 256-token context: length={mask.shape[1]}")
        output = self.text_encoder(
            input_ids=tokens["input_ids"].to(self.device),
            attention_mask=mask.to(self.device),
            output_hidden_states=True,
        )
        return output.hidden_states[-2], mask.to(self.device).bool()

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-1] % 16 or images.shape[-2] % 16:
            raise ValueError("Image tensor must be [N,3,H,W] with H,W divisible by 16")
        if not torch.isfinite(images).all() or images.amin() < -1.001 or images.amax() > 1.001:
            raise ValueError("VAE input must be finite RGB normalized to [-1,1]")
        result = []
        batch_size = batch_size or len(images)
        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        for chunk in images.split(batch_size):
            latent = self.vae.encode(chunk.to(vae_device, dtype=vae_dtype)).latent_dist.mode()
            result.append(((latent - VAE_SHIFT) * VAE_SCALE).float())
        return torch.cat(result, dim=0)

    def latent_loss(self, target_latent: torch.Tensor, radar_latent: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, transport: Any) -> dict:
        if target_latent.shape[-2:] != (56, 56) or radar_latent.shape[-2:] != (28, 28):
            raise ValueError("Expected FPV448 latent 56x56 and radar224 latent 28x28")
        if target_latent.shape[0] != radar_latent.shape[0] or target_latent.shape[0] != cap_feats.shape[0]:
            raise ValueError("Target, radar, and text batch sizes differ")
        kwargs = {
            "cond": [[c] for c in radar_latent],
            "cap_feats": cap_feats,
            "cap_mask": cap_mask,
            "position_type": [["offset"] for _ in radar_latent],
        }
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            return transport.training_losses(self.dit, target_latent, kwargs)

    def training_loss(self, target: torch.Tensor, radar: torch.Tensor, prompts: Sequence[str], transport: Any, *, caption_dropout: float = 0.1) -> dict:
        if not 0 <= caption_dropout <= 1:
            raise ValueError("caption_dropout must be in [0,1]")
        keep = torch.rand(len(prompts)) >= caption_dropout if caption_dropout else torch.ones(len(prompts), dtype=torch.bool)
        texts = [p if keep[i] else "" for i, p in enumerate(prompts)]
        cap_feats, cap_mask = self.encode_text(texts)
        return self.latent_loss(self.encode_images(target).to(self.device), self.encode_images(radar).to(self.device), cap_feats, cap_mask, transport)


def load_bundle(cfg: dict, *, device: str | torch.device, for_training: bool, checkpoint: str | Path | None = None,
                verify_hash: bool = True, audit_initial_output: bool = False) -> AccessoryBundle:
    """Load full normal official DiT, then inject CSGO LoRA, then optional adapter.

    ``checkpoint`` is a CSGO adapter state file, not a substitute base checkpoint.
    """
    from diffusers import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer

    paths = cfg.get("paths", {})
    base = paths.get("base_checkpoint") or paths.get("accessory_checkpoint")
    if not base:
        raise ValueError("paths.base_checkpoint is required")
    text_path = paths.get("gemma_path") or paths.get("text_encoder") or "google/gemma-2-2b"
    tokenizer_path = paths.get("tokenizer_path") or paths.get("tokenizer") or text_path
    vae_path = paths.get("vae_path") or paths.get("vae") or "black-forest-labs/FLUX.1-dev"
    components = verify_official_components({"gemma_path": text_path, "tokenizer_path": tokenizer_path, "vae_path": vae_path})
    model, identity = load_official_dit(base, cfg, device=device, verify_hash=verify_hash)
    probe_inputs = None
    if audit_initial_output:
        generator = torch.Generator(device="cpu").manual_seed(20260930)
        probe_x = torch.randn(1, 16, 56, 56, generator=generator).to(device)
        probe_cond = torch.randn(1, 16, 28, 28, generator=generator).to(device)
        probe_cap = torch.randn(1, 8, 2304, generator=generator).to(device)
        probe_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool, device=device)
        probe_time = torch.tensor([0.4], device=device)
        probe_inputs = (probe_x, probe_time, [[probe_cond[0]]], probe_cap, probe_mask, [["offset"]])
        probe_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            output_before = model(*probe_inputs).float().cpu()
        before_seconds = time.perf_counter() - probe_started
        if output_before.abs().max().item() < 1e-7:
            raise AssertionError("Official loaded model produced a degenerate zero output in initialization probe")
    condition_before = condition_sha256(model)
    inject_native_lora(model, rank=cfg.get("lora", {}).get("rank", 128), scale=cfg.get("lora", {}).get("scale", 1.0))
    condition_after = condition_sha256(model)
    if condition_after != condition_before:
        raise AssertionError("Official trained condition branch changed during LoRA injection")
    injection_audit = {"compute_device": str(next(model.parameters()).device),
                       "compute_dtype": str(next(model.parameters()).dtype),
                       "before_sha256": condition_before, "after_sha256": condition_after,
                       "unchanged": True}
    probe = None
    if probe_inputs is not None:
        probe_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            output_after = model(*probe_inputs).float().cpu()
        after_seconds = time.perf_counter() - probe_started
        difference = (output_after - output_before).abs().max().item()
        if difference != 0.0:
            raise AssertionError(f"Zero-initialized native LoRA changed official model output: {difference}")
        probe = {"input": "seeded synthetic 1x target56/radar28/caption8", "output_max_abs": output_before.abs().max().item(),
                 "zero_lora_max_abs_difference": difference,
                 "before_injection_forward_seconds": before_seconds,
                 "after_injection_forward_seconds": after_seconds}
    if checkpoint is not None:
        adapter = torch.load(checkpoint, map_location="cpu", weights_only=True)
        trainables = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
        if set(adapter) != set(trainables):
            raise ValueError("Adapter checkpoint keys do not exactly match LoRA trainables")
        with torch.no_grad():
            for name, parameter in trainables.items():
                if adapter[name].shape != parameter.shape:
                    raise ValueError(f"Adapter shape mismatch: {name}")
                parameter.copy_(adapter[name].to(parameter.device, dtype=parameter.dtype))
    model.train(for_training)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    text_encoder = AutoModel.from_pretrained(text_path, torch_dtype=dtype, local_files_only=True).to(device).eval().requires_grad_(False)
    if text_encoder.config.hidden_size != 2304:
        raise ValueError(f"Gemma hidden size must be 2304, got {text_encoder.config.hidden_size}")
    vae_args = {} if Path(str(vae_path)).is_dir() and (Path(vae_path) / "config.json").is_file() else {"subfolder": "vae"}
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype, local_files_only=True, **vae_args).to(device).eval().requires_grad_(False)
    if abs(float(vae.config.scaling_factor) - VAE_SCALE) > 1e-6 or abs(float(vae.config.shift_factor) - VAE_SHIFT) > 1e-6:
        raise ValueError("FLUX VAE scale/shift differs from reviewed 0.3611/0.1159")
    return AccessoryBundle(model, text_encoder, tokenizer, vae, identity, components, probe, injection_audit)
