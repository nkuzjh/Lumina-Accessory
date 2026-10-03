import tempfile
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from csgo_seen10.config import load_config
from csgo_seen10.inference import (AtomicJpegWriter, initial_noise, run_inference,
                                   compare_engines, seed_for_sample, shifted_euler_grid)


def test_seed_is_stable_and_sample_specific():
    assert seed_for_sample(42, "discrete", "a") == seed_for_sample(42, "discrete", "a")
    assert seed_for_sample(42, "discrete", "a") != seed_for_sample(42, "continuous", "a")
    assert seed_for_sample(42, "discrete", "a") != seed_for_sample(42, "discrete", "b")
    assert torch.equal(initial_noise(42, "discrete", "a"), initial_noise(42, "discrete", "a"))
    grid = shifted_euler_grid()
    assert len(grid) == 50 and len(grid[:-1]) == 49
    assert grid[0] == 0 and grid[-1] == 1


def test_euler_grid_matches_official_torchdiffeq_49_calls():
    try:
        from transport.integrators import ode
    except ImportError:
        # The isolated model environment installs torchdiffeq; bare system Python may not.
        return
    calls = []

    def drift(x, t, model, **kwargs):
        calls.append(t[0].item())
        return x * 0.1 + t[:, None]

    initial = torch.tensor([[0.3], [-0.4]])
    official = ode(drift, t0=0, t1=1, sampler_type="euler", num_steps=50,
                   atol=1e-6, rtol=1e-3, time_shifting_factor=6.0).sample(initial, None)[-1]
    assert len(calls) == 49
    ours = initial.clone()
    grid = shifted_euler_grid()
    for t0, t1 in zip(grid[:-1], grid[1:]):
        ours = ours + (t1 - t0) * (ours * 0.1 + t0)
    torch.testing.assert_close(ours, official, atol=1e-6, rtol=1e-6)


class FakeDataset:
    load_targets = False

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return {"sample_id": f"sample{i}", "map": "de_nuke", "file_frame": f"{i:06d}",
                "prompt": f"pose {i}", "radar": torch.zeros(3, 224, 224)}


class FakeBundle:
    def __init__(self):
        self.dit = nn.Linear(1, 1)
        self.device = torch.device("cpu")

    def encode_images(self, images):
        return torch.zeros(len(images), 16, 28, 28)

    def encode_text(self, texts):
        return torch.zeros(len(texts), 1, 4), torch.ones(len(texts), 1, dtype=torch.bool)


def test_nonfinite_sampling_cannot_become_a_valid_jpeg(monkeypatch):
    import pytest
    import csgo_seen10.inference as inference
    monkeypatch.setattr(inference, "_guided_velocity", lambda model, state, *a, **kw: torch.full_like(state, float("nan")))
    bundle = FakeBundle()
    with pytest.raises(FloatingPointError, match="nonfinite latents"):
        inference.sample_latents(bundle, torch.zeros(1, 16, 56, 56), torch.zeros(2, 1, 4),
                                 torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 16, 28, 28),
                                 sampling=load_config()["sampling"])


def test_atomic_writer_resume_repair_and_identity(monkeypatch):
    import csgo_seen10.inference as inference

    def fake_sample(bundle, noise, feats, mask, radar, **kwargs):
        return torch.zeros(len(noise), 16, 56, 56), 49

    monkeypatch.setattr(inference, "sample_latents", fake_sample)
    monkeypatch.setattr(inference, "decode_latents", lambda bundle, latents, **kw: torch.ones(len(latents), 3, 448, 448) / 2)
    cfg = load_config()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "pred"
        kwargs = dict(task="discrete", checkpoint_identity={"sha256": "abc"}, cfg=cfg,
                      engine="eager", batch_size=2, vae_batch_size=1)
        first = run_inference(FakeBundle(), FakeDataset(), root, **kwargs)
        assert first["generated"] == 2 and first["skipped"] == 0
        image = root / "gen_imgs/de_nuke/000000.jpg"
        with Image.open(image) as loaded:
            loaded.load()
            assert loaded.format == "JPEG" and loaded.mode == "RGB" and loaded.size == (448, 448)
        second = run_inference(FakeBundle(), FakeDataset(), root, **kwargs)
        assert second["skipped"] == 2 and second["generated"] == 0
        image.write_bytes(b"broken")
        third = run_inference(FakeBundle(), FakeDataset(), root, **kwargs)
        assert third["repaired"] == 1 and third["skipped"] == 1
        try:
            run_inference(FakeBundle(), FakeDataset(), root, **(kwargs | {"checkpoint_identity": {"sha256": "other"}}))
        except ValueError as error:
            assert "identity mismatch" in str(error)
        else:
            raise AssertionError("Mixed checkpoint output was accepted")


def test_writer_propagates_async_error():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "occupied"
        path.write_bytes(b"file")
        try:
            with AtomicJpegWriter(workers=1, max_pending=1) as writer:
                writer.submit(path / "frame.jpg", torch.zeros(3, 8, 8))
        except (FileExistsError, NotADirectoryError):
            pass
        else:
            raise AssertionError("Asynchronous JPEG failure was swallowed")


def test_inference_rejects_target_loading_before_dataset_access():
    class ForbiddenDataset:
        load_targets = True

        def __getitem__(self, index):
            raise AssertionError("Target path was accessed")

    with tempfile.TemporaryDirectory() as temporary:
        try:
            run_inference(FakeBundle(), ForbiddenDataset(), Path(temporary) / "pred",
                          task="discrete", checkpoint_identity={"sha256": "a"}, cfg=load_config())
        except ValueError as error:
            assert "load_targets=False" in str(error)
        else:
            raise AssertionError("Target-reading dataset was accepted")


def test_compiled_cfg_keeps_bf16_math_until_ode_cast():
    from csgo_seen10.inference import _guided_velocity
    from models_accessory.model import NextDiT

    class FakeBF16Model(nn.Module):
        in_channels = 2
        forward_with_cfg = NextDiT.forward_with_cfg

        def forward(self, x, t, cond, cap_feats, cap_mask, position_type):
            return cap_feats[:, :1, :2].reshape(-1, 2, 1, 1).expand_as(x).to(torch.bfloat16)

    model = FakeBF16Model()
    state = torch.zeros(2, 2, 2, 2, dtype=torch.float32)
    feats = torch.tensor([[[1.0625, 0.25]], [[2.375, 0.125]],
                          [[0.125, 1.5]], [[0.25, 1.125]]], dtype=torch.bfloat16)
    mask = torch.ones(4, 1, dtype=torch.bool)
    radar = torch.zeros(4, 16, 28, 28)
    args = dict(cfg_scale=4.0, cfg_trunc=100.0, renorm_cfg=1.0)
    eager = _guided_velocity(model, state, 0.25, feats, mask, radar,
                             prepared=None, denoiser=None, engine="eager", **args)
    compiled = _guided_velocity(model, state, 0.25, feats, mask, radar,
                                prepared=None,
                                denoiser=lambda x, t, prepared: model.forward(x, t, [], feats, mask, []),
                                engine="compiled", **args)
    assert compiled.dtype == torch.float32
    torch.testing.assert_close(compiled, eager, atol=0, rtol=0)


def test_compare_failure_persists_incomplete_report(monkeypatch):
    import csgo_seen10.inference as inference

    def fail_after_marker(bundle, dataset, output_root, **kwargs):
        output_root.mkdir()
        (output_root / inference.MARKER_NAME).write_text(json.dumps({"checkpoint": "fixture"}))
        raise RuntimeError("compile failed")

    monkeypatch.setattr(inference, "_compare_engines_impl", fail_after_marker)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "benchmark"
        try:
            compare_engines(None, None, root, task="discrete", checkpoint_identity={},
                            cfg={}, batch_size=2, vae_batch_size=1)
        except RuntimeError as error:
            assert "compile failed" in str(error)
        else:
            raise AssertionError("Benchmark failure was swallowed")
        report = json.loads((root / "benchmark_report.json").read_text())
        assert report["status"] == "incomplete"
        assert report["error"] == {"type": "RuntimeError", "message": "compile failed"}
        assert report["identity"]["checkpoint"] == "fixture"
