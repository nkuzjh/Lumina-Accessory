"""Target-independent CSGO inference, resumable atomic JPEG output."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from .config import atomic_json, code_identity, content_hash, scientific_config
from .fast_inference import build_engine, prepare_condition
from .model import VAE_SCALE, VAE_SHIFT

MARKER_NAME = "prediction_identity.json"


def seed_for_sample(seed: int, task: str, sample_id: str) -> int:
    payload = json.dumps([int(seed), str(task), str(sample_id)], separators=(",", ":"), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def initial_noise(seed: int, task: str, sample_id: str, *, size: int = 56) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed_for_sample(seed, task, sample_id))
    return torch.randn((16, size, size), generator=generator, dtype=torch.float32)


def shifted_euler_grid(steps: int = 50, shift: float = 6.0) -> torch.Tensor:
    if steps != 50 or shift != 6.0:
        raise ValueError("Aligned primary sampler uses 50 points and time shift 6")
    t = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)
    return t / (t + shift - shift * t)


def _guided_velocity(model, x, t, cap_feats, cap_mask, radar_latent, *, cfg_scale, cfg_trunc, renorm_cfg, prepared, denoiser, engine):
    batch = x.shape[0]
    input_x = torch.cat((x, x), dim=0)
    input_t = torch.full((2 * batch,), t, device=x.device, dtype=torch.float32)
    if engine == "eager":
        # Official Accessory CFG implementation, including corrected per-sample renorm.
        radar = [[c] for c in radar_latent]
        return model.forward_with_cfg(input_x, input_t, cap_feats, cap_mask, cfg_scale,
                                      cond=radar, position_type=[["offset"] for _ in radar],
                                      cfg_trunc=cfg_trunc, renorm_cfg=renorm_cfg)[:batch].float()
    # A CUDA graph may reuse its output storage at the next timestep or batch.
    # Clone immediately outside the compiled region before the ODE update retains it.
    velocity = denoiser(input_x, input_t, prepared).clone()
    pos, neg = velocity.chunk(2, dim=0)
    if t >= cfg_trunc:
        return pos.float()
    guided = neg + cfg_scale * (pos - neg)
    if renorm_cfg > 0:
        axes = tuple(range(1, guided.ndim))
        maximum = torch.linalg.vector_norm(pos, dim=axes, keepdim=True) * renorm_cfg
        norm = torch.linalg.vector_norm(guided, dim=axes, keepdim=True)
        guided = guided * torch.minimum(torch.ones_like(norm), maximum / norm.clamp_min(torch.finfo(norm.dtype).tiny))
    return guided.float()


@torch.no_grad()
def sample_latents(bundle, noise: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor,
                   radar_latent: torch.Tensor, *, sampling: dict, engine: str = "eager", compile_mode: str = "default",
                   denoiser=None) -> tuple[torch.Tensor, int]:
    if noise.ndim != 4 or noise.shape[1:] != (16, 56, 56):
        raise ValueError("Sampling requires FP32 target noise [N,16,56,56]")
    if sampling.get("sampler", "euler").lower() != "euler":
        raise ValueError("Only the reviewed Euler sampler is supported")
    batch = noise.shape[0]
    if cap_feats.shape[0] != 2 * batch or radar_latent.shape[0] != 2 * batch:
        raise ValueError("CFG expects positive then negative text and duplicated radar")
    if engine == "compiled":
        with torch.autocast(bundle.device.type, dtype=torch.bfloat16, enabled=bundle.device.type == "cuda"):
            prepared = prepare_condition(bundle.dit, cap_feats, cap_mask, radar_latent)
        denoiser = denoiser or build_engine(bundle.dit, engine, compile_mode)
    elif engine == "eager":
        prepared = None
    else:
        raise ValueError("Unknown inference engine")
    grid = shifted_euler_grid(int(sampling.get("num_sampling_steps", 50)), float(sampling.get("time_shift", 6.0)))
    cfg_scale = float(sampling.get("cfg_scale", 4.0))
    cfg_trunc = float(sampling.get("cfg_trunc", 100.0))
    renorm_cfg = float(sampling.get("renorm_cfg", 1.0))
    state = noise.to(bundle.device, dtype=torch.float32)
    nfe = 0
    for t0, t1 in zip(grid[:-1].tolist(), grid[1:].tolist()):
        with torch.autocast(bundle.device.type, dtype=torch.bfloat16, enabled=bundle.device.type == "cuda"):
            velocity = _guided_velocity(bundle.dit, state, t0, cap_feats, cap_mask, radar_latent,
                                        cfg_scale=cfg_scale, cfg_trunc=cfg_trunc, renorm_cfg=renorm_cfg,
                                        prepared=prepared, denoiser=denoiser, engine=engine)
        nfe += 1
        state = state + (t1 - t0) * velocity.float()
    if not torch.isfinite(state).all():
        raise FloatingPointError("Euler sampling produced nonfinite latents; refusing to encode predictions")
    return state, nfe


@torch.no_grad()
def decode_latents(bundle, latents: torch.Tensor, *, batch_size: int = 4) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("VAE batch size must be positive")
    vae = bundle.vae
    dtype = next(vae.parameters()).dtype
    device = next(vae.parameters()).device
    pixels = []
    for chunk in latents.split(batch_size):
        decoded = vae.decode((chunk / VAE_SCALE + VAE_SHIFT).to(device=device, dtype=dtype)).sample
        pixels.append(decoded.float().add(1).div(2).clamp(0, 1).cpu())
    return torch.cat(pixels, dim=0)


def prediction_identity(cfg: dict, *, task: str, checkpoint_identity: dict, engine: str,
                        compile_mode: str, batch_size: int, vae_batch_size: int, benchmark_only: bool,
                        bundle=None, selection=None) -> dict:
    package_versions = {}
    for name in ("torch", "transformers", "diffusers", "flash-attn"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    environment = {"python": sys.version.split()[0], "platform": platform.platform(),
                   "packages": package_versions, "torch_cuda_runtime": torch.version.cuda}
    parameter = next(bundle.dit.parameters(), None) if bundle is not None else None
    runtime_precision = {"device_type": parameter.device.type, "model_dtype": str(parameter.dtype),
                         "ode_state_dtype": "torch.float32"} if parameter is not None else None
    return {
        "experiment": cfg["experiment"], "science_sha256": content_hash(scientific_config(cfg)),
        "base": cfg["base"], "loaded_base": getattr(bundle, "base_identity", None),
        "components": getattr(bundle, "component_identity", None), "checkpoint": checkpoint_identity,
        "code": code_identity(), "environment": environment, "selection": selection,
        "data": {k: cfg["data"][k] for k in ("manifest_sha256", "calibration_file_sha256", "image_size", "radar_size")},
        "seed": cfg["seed"], "seed_algorithm": "SHA256(JSON([seed,task,sample_id])) first 8 bytes little-endian -> CPU torch.Generator",
        "task": task, "sampling": cfg["sampling"], "encoding": cfg["encoding"],
        "precision": cfg["model"]["precision"],
        "runtime_precision": runtime_precision,
        "engine": engine, "compile_mode": compile_mode if engine == "compiled" else None,
        "batch_size": batch_size, "vae_batch_size": vae_batch_size, "benchmark_only": benchmark_only,
    }


def _selection_identity(dataset, task: str, limit: int) -> dict:
    rows = getattr(dataset, "rows", None)
    chosen = rows[:limit] if rows is not None else [dataset[i] for i in range(limit)]
    identities = [{"sample_id": r["sample_id"], "map": r.get("map", r.get("map_name")),
                   "file_frame": r["file_frame"], "clip_id": r.get("clip_id"),
                   "frame_index": r.get("frame_index")} for r in chosen]
    expected = {"discrete": 20000, "continuous": 12800}[task]
    return {"sha256": content_hash(identities), "count": len(identities),
            "official_expected_count": expected, "partial_debug": len(identities) != expected}


def _valid_jpeg(path: Path, size: int) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == "JPEG" and image.mode == "RGB" and image.size == (size, size)
    except (OSError, ValueError):
        return False


def _save_jpeg(path: Path, pixels: torch.Tensor, quality: int) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    array = pixels.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(np.asarray(array), mode="RGB")
    name = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
            name = handle.name
            image.save(handle, format="JPEG", quality=quality, optimize=False, progressive=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


class AtomicJpegWriter:
    def __init__(self, *, workers: int = 4, max_pending: int = 8, quality: int = 75):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.max_pending = max_pending
        self.quality = quality

    def submit(self, path: Path, pixels: torch.Tensor) -> None:
        if len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(_save_jpeg, path, pixels.detach().cpu().clone(), self.quality))

    def close(self) -> None:
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@torch.no_grad()
def run_inference(bundle, dataset, output_root: str | Path, *, task: str, checkpoint_identity: dict,
                  cfg: dict, engine: str = "eager", compile_mode: str = "default", batch_size: int = 16,
                  vae_batch_size: int = 4, benchmark_batches: int | None = None) -> dict:
    if getattr(dataset, "load_targets", True):
        raise ValueError("Inference dataset must be constructed with load_targets=False")
    if task not in {"discrete", "continuous"} or batch_size < 1:
        raise ValueError("Inference requires discrete/continuous task and positive batch")
    if benchmark_batches is not None and benchmark_batches < 1:
        raise ValueError("benchmark_batches must be positive")
    root = Path(output_root)
    marker = root / MARKER_NAME
    limit = min(len(dataset), benchmark_batches * batch_size) if benchmark_batches else len(dataset)
    selection = _selection_identity(dataset, task, limit)
    identity = prediction_identity(cfg, task=task, checkpoint_identity=checkpoint_identity,
                                   engine=engine, compile_mode=compile_mode, batch_size=batch_size,
                                   vae_batch_size=vae_batch_size, benchmark_only=benchmark_batches is not None,
                                   bundle=bundle, selection=selection)
    if benchmark_batches is not None and root.exists() and any(root.iterdir()):
        raise ValueError("Benchmark requires a fresh empty output root")
    if marker.is_file():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"Prediction identity mismatch at {root}")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Output root exists without identity marker: {root}")
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(marker, identity)
    bundle.dit.eval()
    radar_cache: dict[str, torch.Tensor] = {}
    denoiser = build_engine(bundle.dit, engine, compile_mode) if engine == "compiled" else None
    generated = skipped = repaired = 0
    batch_times = []
    started = time.perf_counter()
    if bundle.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bundle.device)
    with tqdm(total=limit, desc=f"{task} processed", unit="image") as progress, \
            AtomicJpegWriter(quality=cfg["encoding"]["quality"]) as writer:
        for start in range(0, limit, batch_size):
            batch_started = time.perf_counter()
            batch = [dataset[i] for i in range(start, min(start + batch_size, limit))]
            paths = [root / "gen_imgs" / row["map"] / f"{row['file_frame']}.jpg" for row in batch]
            valid = [_valid_jpeg(path, cfg["data"]["image_size"]) for path in paths]
            skipped += sum(valid)
            pending = [(row, path) for row, path, okay in zip(batch, paths, valid) if not okay]
            repaired += sum(path.exists() for _, path in pending)
            if not pending:
                progress.set_postfix(generated=generated, skipped=skipped, repaired=repaired, refresh=False)
                progress.update(len(batch))
                continue
            rows, paths = zip(*pending)
            # Compile a constant CFG batch shape, even after resume or at the tail.
            # Only the real rows below are decoded and written.
            actual_count = len(rows)
            if engine == "compiled" and actual_count < batch_size:
                rows = tuple(rows) + (rows[-1],) * (batch_size - actual_count)
            for row in rows:
                if row["map"] not in radar_cache:
                    radar_cache[row["map"]] = bundle.encode_images(row["radar"].unsqueeze(0)).to(bundle.device)[0]
            radar = torch.stack([radar_cache[row["map"]] for row in rows])
            prompts = [row["prompt"] for row in rows]
            negatives = [cfg["sampling"].get("negative_prompt", "")] * len(rows)
            feats, mask = bundle.encode_text(prompts + negatives)
            condition = torch.cat((radar, radar), dim=0)
            noise = torch.stack([initial_noise(cfg["seed"], task, row["sample_id"]) for row in rows])
            latents, nfe = sample_latents(bundle, noise, feats, mask, condition,
                                          sampling=cfg["sampling"], engine=engine,
                                          compile_mode=compile_mode, denoiser=denoiser)
            if nfe != cfg["sampling"]["expected_nfe"]:
                raise AssertionError(f"Unexpected velocity NFE: {nfe}")
            pixels = decode_latents(bundle, latents[:actual_count], batch_size=vae_batch_size)
            for path, pixel in zip(paths, pixels):
                writer.submit(path, pixel)
                generated += 1
            if bundle.device.type == "cuda":
                torch.cuda.synchronize(bundle.device)
            batch_times.append({"batch_index": start // batch_size, "real_samples": actual_count,
                                "seconds": time.perf_counter() - batch_started})
            progress.set_postfix(generated=generated, skipped=skipped, repaired=repaired, refresh=False)
            progress.update(len(batch))
    elapsed = time.perf_counter() - started
    report = {"generated": generated, "skipped": skipped, "repaired": repaired,
            "processed": limit, "nfe_per_image": cfg["sampling"]["expected_nfe"], "output_root": str(root),
            "benchmark_only": benchmark_batches is not None,
            "partial_debug": selection["partial_debug"], "wall_seconds": elapsed,
            "batch_times": batch_times,
            "cold_batch_seconds": batch_times[0]["seconds"] if batch_times else None,
            "steady_batch_seconds": [b["seconds"] for b in batch_times[1:]],
            "images_per_second": generated / elapsed if elapsed else None}
    if bundle.device.type == "cuda":
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(bundle.device)
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(bundle.device)
    atomic_json(root / ("benchmark_report.json" if benchmark_batches else "inference_report.json"), report)
    return report


@torch.no_grad()
def _compare_engines_impl(bundle, dataset, output_root: str | Path, *, task: str, checkpoint_identity: dict,
                          cfg: dict, batch_size: int, vae_batch_size: int, full_batches: int = 3) -> dict:
    """GPU validation harness: cold + steady full batches, tail, and eager parity.

    This deliberately requires an isolated empty root and never emits formal coverage.
    It records first-batch latents/pixels for later numerical inspection.
    """
    if bundle.device.type != "cuda":
        raise RuntimeError("Three-engine performance comparison requires an explicitly available CUDA device")
    if getattr(dataset, "load_targets", True):
        raise ValueError("Benchmark must use target-free dataset")
    if full_batches < 3 or batch_size < 2 or len(dataset) < full_batches * batch_size + 1:
        raise ValueError("Benchmark needs three complete batches and one tail sample")
    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Benchmark comparison requires a fresh empty output root")
    root.mkdir(parents=True, exist_ok=True)
    limit = full_batches * batch_size + 1
    selection = _selection_identity(dataset, task, limit)
    marker = {"benchmark_only": True, "checkpoint": checkpoint_identity,
              "loaded_base": bundle.base_identity, "components": bundle.component_identity,
              "science_sha256": content_hash(scientific_config(cfg)), "selection": selection,
              "code": code_identity(), "engines": ["eager", "compiled/default", "compiled/reduce-overhead"]}
    atomic_json(root / MARKER_NAME, marker)
    bundle.dit.eval()
    rows = [dataset[i] for i in range(limit)]
    radar_cache = {}
    for row in rows:
        if row["map"] not in radar_cache:
            radar_cache[row["map"]] = bundle.encode_images(row["radar"].unsqueeze(0))[0].to(bundle.device)
    runs = {}
    references = {}
    calibration = {}
    thresholds = {}
    calibration_seconds = None

    def eager_reference(work):
        radar = torch.stack([radar_cache[r["map"]] for r in work])
        captions = [r["prompt"] for r in work]
        negatives = [cfg["sampling"].get("negative_prompt", "")] * len(work)
        cap_feats, cap_mask = bundle.encode_text(captions + negatives)
        noise = torch.stack([initial_noise(cfg["seed"], task, r["sample_id"]) for r in work])
        latents, nfe = sample_latents(bundle, noise, cap_feats, cap_mask, torch.cat((radar, radar)),
                                      sampling=cfg["sampling"], engine="eager")
        if nfe != 49:
            raise AssertionError(f"Eager repeat NFE differed: {nfe}")
        pixels = decode_latents(bundle, latents, batch_size=vae_batch_size)
        return {"latents": latents.float().cpu(), "pixels": pixels}

    for engine, mode in (("eager", "default"), ("compiled", "default"), ("compiled", "reduce-overhead")):
        label = engine if engine == "eager" else f"compiled_{mode}"
        references[label] = []
        denoiser = build_engine(bundle.dit, engine, mode) if engine == "compiled" else None
        torch.cuda.reset_peak_memory_stats(bundle.device)
        batch_timings = []
        total_started = time.perf_counter()
        with AtomicJpegWriter(quality=cfg["encoding"]["quality"]) as writer:
            for batch_index in range(full_batches + 1):
                real = rows[batch_index * batch_size:(batch_index + 1) * batch_size]
                actual = len(real)
                if engine == "compiled" and actual < batch_size:
                    work = real + [real[-1]] * (batch_size - actual)
                else:
                    work = real
                torch.cuda.synchronize(bundle.device)
                started = time.perf_counter()
                radar = torch.stack([radar_cache[r["map"]] for r in work])
                captions = [r["prompt"] for r in work]
                negatives = [cfg["sampling"].get("negative_prompt", "")] * len(work)
                cap_feats, cap_mask = bundle.encode_text(captions + negatives)
                conditions = torch.cat((radar, radar))
                noise = torch.stack([initial_noise(cfg["seed"], task, r["sample_id"]) for r in work])
                latents, nfe = sample_latents(bundle, noise, cap_feats, cap_mask, conditions,
                                              sampling=cfg["sampling"], engine=engine,
                                              compile_mode=mode, denoiser=denoiser)
                if nfe != 49:
                    raise AssertionError(f"Expected 49 actual velocity calls, got {nfe}")
                latents = latents[:actual]
                pixels = decode_latents(bundle, latents, batch_size=vae_batch_size)
                reference = {"latents": latents.float().cpu().clone(), "pixels": pixels.clone()}
                references[label].append(reference)
                if batch_index == 0:
                    torch.save(reference["latents"], root / f"{label}_first_latents.pt")
                    torch.save(reference["pixels"], root / f"{label}_first_pixels.pt")
                for row, pixels_one in zip(real, pixels):
                    writer.submit(root / label / row["map"] / f"{row['file_frame']}.jpg", pixels_one)
                torch.cuda.synchronize(bundle.device)
                batch_timings.append({"batch": batch_index, "real_samples": actual,
                                      "seconds": time.perf_counter() - started})
        elapsed = time.perf_counter() - total_started
        runs[label] = {"cold_batch_seconds": batch_timings[0]["seconds"],
                       "steady_full_batch_seconds": [r["seconds"] for r in batch_timings[1:full_batches]],
                       "tail_batch_seconds": batch_timings[-1]["seconds"],
                       "end_to_end_seconds": elapsed, "images_per_second": limit / elapsed,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(bundle.device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(bundle.device)}
        if label == "eager":
            calibration_started = time.perf_counter()
            repeated = eager_reference(rows[:batch_size])
            single = eager_reference(rows[:1])
            for domain, floor in (("latents", 1e-3), ("pixels", 2.0 / 255)):
                repeat_max = (repeated[domain] - references["eager"][0][domain]).abs().max().item()
                batch_max = (single[domain][0] - references["eager"][0][domain][0]).abs().max().item()
                calibration[domain] = {"repeat_max_abs": repeat_max, "single_vs_batch_max_abs": batch_max}
                thresholds[domain] = max(floor, 10.0 * max(repeat_max, batch_max))
            calibration_seconds = time.perf_counter() - calibration_started
    differences = {}
    for label, value in references.items():
        if label == "eager":
            continue
        differences[label] = {}
        for domain in ("latents", "pixels"):
            by_batch = []
            for batch_index, (candidate, baseline) in enumerate(zip(value, references["eager"])):
                diff = (candidate[domain] - baseline[domain]).abs()
                by_batch.append({"batch_index": batch_index, "max_abs": diff.max().item(),
                                 "mean_abs": diff.mean().item(), "samples": len(candidate[domain])})
            differences[label][domain] = {"max_abs": max(x["max_abs"] for x in by_batch),
                                          "mean_abs": sum(x["mean_abs"] * x["samples"] for x in by_batch) / limit,
                                          "by_batch": by_batch}
    parity_pass = all(differences[label][domain]["max_abs"] <= thresholds[domain]
                      for label in differences for domain in thresholds)
    class TwoSampleDataset:
        load_targets = False

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return dataset[index]

    resume_root = root / "resume_check"
    resume_dataset = TwoSampleDataset()
    first = run_inference(bundle, resume_dataset, resume_root, task=task,
                          checkpoint_identity=checkpoint_identity, cfg=cfg, engine="eager",
                          batch_size=2, vae_batch_size=vae_batch_size)
    second = run_inference(bundle, resume_dataset, resume_root, task=task,
                           checkpoint_identity=checkpoint_identity, cfg=cfg, engine="eager",
                           batch_size=2, vae_batch_size=vae_batch_size)
    corrupt_path = resume_root / "gen_imgs" / rows[0]["map"] / f"{rows[0]['file_frame']}.jpg"
    corrupt_path.write_bytes(b"incomplete JPEG")
    third = run_inference(bundle, resume_dataset, resume_root, task=task,
                          checkpoint_identity=checkpoint_identity, cfg=cfg, engine="eager",
                          batch_size=2, vae_batch_size=vae_batch_size)
    mismatch_rejected = False
    try:
        run_inference(bundle, resume_dataset, resume_root, task=task,
                      checkpoint_identity={"mismatch_test": True}, cfg=cfg, engine="eager",
                      batch_size=2, vae_batch_size=vae_batch_size)
    except ValueError as error:
        mismatch_rejected = "identity mismatch" in str(error)
    resume_pass = (first["generated"] == 2 and second["skipped"] == 2 and
                   third["repaired"] == 1 and third["skipped"] == 1 and mismatch_rejected)
    report = {"benchmark_only": True, "task": task, "samples": limit, "full_batches": full_batches,
              "batch_size": batch_size, "vae_batch_size": vae_batch_size, "runs": runs,
              "vs_official_eager": differences, "selection": selection,
              "eager_calibration": calibration, "calibration_seconds": calibration_seconds,
              "parity_threshold_max_abs": thresholds, "parity_pass": parity_pass,
              "resume_check": {"pass": resume_pass, "first_generated": first["generated"],
                               "second_skipped": second["skipped"], "corrupt_repaired": third["repaired"],
                               "checkpoint_mismatch_rejected": mismatch_rejected}}
    atomic_json(root / "benchmark_report.json", report)
    if not parity_pass:
        raise AssertionError(f"Compiled output exceeded eager-calibrated parity threshold; see {root / 'benchmark_report.json'}")
    if not resume_pass:
        raise AssertionError(f"Inference restart/repair check failed; see {root / 'benchmark_report.json'}")
    return report


@torch.no_grad()
def compare_engines(bundle, dataset, output_root: str | Path, *, task: str, checkpoint_identity: dict,
                    cfg: dict, batch_size: int, vae_batch_size: int, full_batches: int = 3) -> dict:
    """Persist a failure report once a benchmark identity has been established."""
    root = Path(output_root)
    try:
        return _compare_engines_impl(bundle, dataset, root, task=task,
                                     checkpoint_identity=checkpoint_identity, cfg=cfg,
                                     batch_size=batch_size, vae_batch_size=vae_batch_size,
                                     full_batches=full_batches)
    except Exception as error:
        marker = root / MARKER_NAME
        if marker.is_file():
            report_path = root / "benchmark_report.json"
            prior = json.loads(report_path.read_text()) if report_path.is_file() else {}
            prior.update({"status": "failed" if prior else "incomplete",
                          "benchmark_only": True,
                          "error": {"type": type(error).__name__, "message": str(error)},
                          "identity": json.loads(marker.read_text())})
            atomic_json(report_path, prior)
        raise
