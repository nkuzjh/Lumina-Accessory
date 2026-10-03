"""Read the shared released protocol; test images are never opened by inference."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import math
import sys

from .config import content_hash, sha256_file

SYSTEM_PROMPT = ("You are an assistant designed to generate a first-person view from a radar map "
                 "and a camera pose. <Prompt Start> ")
PROMPT_TEMPLATE = ("Generate a CS2 first-person view from the supplied radar map.\n"
                   "Map: {map_name}.\n"
                   "Camera pose: x={x}, y={y}, z={z}, pitch={pitch} rad, yaw={yaw} rad.\n"
                   "Coordinates x and y use the published 1024x1024 map coordinate system.")
PROMPT_HASH = content_hash({"system": SYSTEM_PROMPT, "template": PROMPT_TEMPLATE, "number": "python_repr_float"})
SPLITS = {"train": "seen_train", "validation": "seen_validation", "discrete": "seen_discrete_test", "continuous": "seen_continuous"}


def load_protocol(shared_eval_dir):
    path = Path(shared_eval_dir) / "protocol.py"
    name = "lumina_shared_protocol_" + sha256_file(path)[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_prompt(row):
    values = {k: float(row["pose_raw"][k]) for k in ("x", "y", "z", "pitch", "yaw")}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Non-finite physical pose")
    # repr preserves the published float value; never reconstruct from normalized tensors.
    return SYSTEM_PROMPT + PROMPT_TEMPLATE.format(map_name=row["map_name"], **{k: repr(v) for k, v in values.items()})


def protocol_identity(cfg):
    root = Path(cfg["paths"]["data_root"])
    files = [root / "benchmark_manifest.json", root / "minimal_dataset_report.json", root / "calibration/z_calibration.json"]
    files += sorted((root / "splits/seen").glob("*/*.json"))
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in files}
    for rel, key in [("benchmark_manifest.json", "manifest_sha256"), ("calibration/z_calibration.json", "calibration_file_sha256")]:
        if hashes[rel] != cfg["data"][key]:
            raise ValueError(f"Published {rel} hash mismatch: {hashes[rel]}")
    protocol = load_protocol(cfg["paths"]["shared_eval_dir"])
    benchmark = protocol.BenchmarkData(root)
    for map_name, rel in benchmark._radar_targets.items():
        p = root / benchmark._radar_root / rel
        hashes[str(p.relative_to(root))] = sha256_file(p)
    return {"files": hashes, "data_sha256": content_hash(hashes), "prompt_sha256": PROMPT_HASH,
            "protocol_source_sha256": sha256_file(Path(cfg["paths"]["shared_eval_dir"]) / "protocol.py"),
            "calibration_fingerprint": benchmark.manifest["calibration"]["fingerprint"],
            "maps": list(benchmark.maps)}


def image_tensor(path, size):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as im:
        array = np.array(im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1)


class Seen10Dataset:
    def __init__(self, cfg, split, *, load_targets=None, limit=None):
        self.cfg = cfg
        self.split = SPLITS.get(split, split)
        allowed = self.split in ("seen_train", "seen_validation")
        self.load_targets = allowed if load_targets is None else bool(load_targets)
        if self.load_targets and not allowed:
            raise ValueError(f"Target reads prohibited for inference split {self.split}")
        protocol = load_protocol(cfg["paths"]["shared_eval_dir"])
        self.benchmark = protocol.BenchmarkData(cfg["paths"]["data_root"])
        self.rows = self.benchmark.rows(self.split, max_samples=limit)
        # Retain protocol order, clip identity and raw physical pose, never directory-scan.
        if len({r["sample_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate sample identity")
        self._radars = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        radar_path = row["radar_path"]
        if radar_path not in self._radars:
            self._radars[radar_path] = image_tensor(radar_path, self.cfg["data"]["radar_size"])
        result = {k: row[k] for k in ("sample_id", "map_name", "file_frame", "pose_raw", "clip_id", "frame_index")}
        result.update(map=row["map_name"], prompt=make_prompt(row), radar=self._radars[radar_path].clone())
        if self.load_targets:
            result["target"] = image_tensor(row["image_path"], self.cfg["data"]["image_size"])
        return result


def collate_samples(samples):
    import torch
    batch = {k: [r[k] for r in samples] for k in samples[0]}
    for key in ("radar", "target"):
        if key in batch:
            batch[key] = torch.stack(batch[key])
    return batch
