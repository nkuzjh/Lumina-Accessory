import torch
from torch import nn
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from csgo_seen10.fast_inference import build_position_ids, prepare_condition, static_denoise
from csgo_seen10.lora_audit import audit_trainable
from csgo_seen10.model import AccessoryBundle, condition_sha256, construct_dit, inject_native_lora, load_official_dit, merge_native_lora, sha256_file
from models_accessory.lora import LinearLora, replace_linear_with_lora
from models_accessory.model import NextDiT


def test_official_meta_architecture_and_lora_audit():
    model = construct_dit({}, meta=True)
    assert sum(p.numel() for p in model.parameters()) == 2784546368
    model.requires_grad_(False)
    targets = inject_native_lora(model)
    assert len(targets) == 197
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    summary = audit_trainable(model, optimizer=optimizer)["summary"]
    assert summary["trainable_parameters"] == 227963968
    assert summary["dit_parameters"] == 3012510336
    assert summary["optimizer_verified"]


def test_native_lora_zero_injection_fp32_and_merge_including_b_bias():
    torch.manual_seed(3)
    base = nn.Sequential(nn.Linear(4, 5, bias=False), nn.Linear(5, 3, bias=True)).to(dtype=torch.bfloat16)
    base.requires_grad_(False)
    x = torch.randn(7, 4).to(torch.bfloat16)
    original = base(x).float()
    weights = [p.detach().clone() for p in base.parameters()]
    replace_linear_with_lora(base, max_rank=128, scale=1.0, lora_dtype=torch.float32)
    for name, parameter in base.named_parameters():
        parameter.requires_grad_("lora_" in name)
    assert isinstance(base[0], LinearLora) and base[0].rank == 4
    assert base[0].bias is None and base[0].lora_B.bias is not None
    assert all(p.dtype == torch.float32 for n, p in base.named_parameters() if "lora_" in n)
    assert torch.equal(base(x).float(), original)
    assert all(torch.equal(a, b) for a, b in zip(weights, (base[0].weight, base[1].weight, base[1].bias)))
    with torch.no_grad():
        base[0].lora_B.bias.fill_(0.25)
        base[1].lora_B.weight.fill_(0.02)
    adapted = base(x).float()
    merge_native_lora(base)
    assert base[0].bias is not None
    torch.testing.assert_close(base(x).float(), adapted, atol=0.02, rtol=0.02)


def test_optimizer_audit_rejects_frozen_parameter():
    model = nn.Sequential(nn.Linear(4, 4))
    model.requires_grad_(False)
    replace_linear_with_lora(model, max_rank=2, lora_dtype=torch.float32)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name)
    bad_optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    try:
        audit_trainable(model, optimizer=bad_optimizer, expect_official=False)
    except AssertionError as error:
        assert "coverage" in str(error)
    else:
        raise AssertionError("Frozen parameter silently entered optimizer")


def test_audit_markdown_lists_every_tensor_and_optimizer_hyperparameters():
    model = nn.Sequential(nn.Linear(4, 3))
    model.requires_grad_(False)
    replace_linear_with_lora(model, max_rank=2, lora_dtype=torch.float32)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name)
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.02, weight_decay=0.1)
    with TemporaryDirectory() as temporary:
        report = audit_trainable(model, optimizer=optimizer, output_dir=temporary, expect_official=False)
        markdown = (Path(temporary) / "trainable_parameter_audit.md").read_text()
        assert markdown.count("| `dit.") == len(report["parameters"])
        assert "| `dit.0.lora_B.bias` | 3 | 3 | lora |" in markdown
        assert "| 0.02 | 0.1 | yes |" in markdown
        assert "| — | — | no |" in markdown
        assert "Trainable / DiT total:" in markdown and "trainable / bundle total:" in markdown
        assert report["summary"]["base_dit_parameters"] == sum(
            row["numel"] for row in report["parameters"] if row["category"] == "frozen")


def test_cpu_probe_clips_before_constant_scheduler_step():
    from scripts.audit_csgo_seen10_model import apply_probe_optimizer_update

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.param_groups[0]["d"] = 0.25  # Exercise Prodigy diagnostic field.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    loss = model(torch.ones(8, 2)).square().mean() * 1000
    loss.backward()
    update = apply_probe_optimizer_update(list(model.parameters()), optimizer, scheduler, gradient_clip=2.0)
    post_clip_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert update["grad_norm"] > 2.0
    assert post_clip_norm <= 2.0001
    assert update["scheduler_last_epoch"] == 1
    assert update["lr"] == 1.0 and update["prodigy_d"] == 0.25 and update["effective_lr"] == 0.25


def test_cpu_resume_state_comparison_checks_nested_optimizer_tensors():
    from scripts.audit_csgo_seen10_model import assert_same_state

    expected = {"groups": [{"d": torch.tensor(0.25), "steps": [1, 2]}]}
    assert_same_state(expected, {"groups": [{"d": torch.tensor(0.25), "steps": [1, 2]}]})
    try:
        assert_same_state(expected, {"groups": [{"d": torch.tensor(0.5), "steps": [1, 2]}]})
    except AssertionError as error:
        assert "groups[0].d" in str(error)
    else:
        raise AssertionError("Changed optimizer state was accepted as exact resume")


def test_cpu_structural_probe_replays_prodigy_step_from_production_checkpoint():
    import copy

    from csgo_seen10.checkpoint import load_checkpoint, save_checkpoint, trainable_state
    from csgo_seen10.config import load_config
    from csgo_seen10.training import make_optimizer
    from scripts.audit_csgo_seen10_model import assert_same_state, structural_micro_step

    class TinyDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.final_layer = nn.Module()
            self.final_layer.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.final_layer.linear(x)

    class TinyBundle:
        def __init__(self):
            self.dit = TinyDiT()
            self.dit.requires_grad_(False)
            replace_linear_with_lora(self.dit, max_rank=2, scale=1.0, lora_dtype=torch.float32)
            for name, parameter in self.dit.named_parameters():
                parameter.requires_grad_("lora_" in name)
            self.text_encoder = nn.Linear(1, 1).requires_grad_(False)
            self.vae = nn.Linear(1, 1).requires_grad_(False)

        def training_loss(self, target, radar, prompts, transport, caption_dropout):
            assert caption_dropout == 0.0 and len(prompts) == 1
            features = torch.stack((target.mean(dim=(1, 2, 3)), radar.mean(dim=(1, 2, 3))), dim=1)
            features = features + torch.randn_like(features) * 0.1
            return {"loss": (self.dit(features) - 0.3).square().flatten(1).mean(1)}

    torch.manual_seed(120)
    cfg = load_config()
    bundle = TinyBundle()
    params = [p for p in bundle.dit.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    samples = [{"sample_id": f"s{i}", "target": torch.full((3, 4, 4), float(i + 1)),
                "radar": torch.full((3, 4, 4), float(i + 2)), "prompt": "pose"} for i in range(2)]
    identity = {"purpose": "tiny_cpu_structural_test", "effective_batch": 1, "seed": 120}
    topology = {"world_size": 1, "device": "cpu"}
    first = structural_micro_step(bundle, samples[0], None, optimizer, scheduler, params, cfg["training"], 1)
    assert first["sample_id"] == "s0" and first["nonzero_B_bias_gradient_count"] == 1
    with TemporaryDirectory() as temporary:
        saved = save_checkpoint(Path(temporary) / "checkpoints", model=bundle.dit,
                                optimizer=optimizer, scheduler=scheduler, step=1, exposures=1,
                                sampler_state={"next_index": 1}, identity=identity,
                                topology=topology, validation_loss=None, smoke=True)
        expected = structural_micro_step(bundle, samples[1], None, optimizer, scheduler,
                                         params, cfg["training"], 2)
        trainable = trainable_state(bundle.dit)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        restored = load_checkpoint(saved, model=bundle.dit, optimizer=optimizer,
                                   scheduler=scheduler, identity=identity, topology=topology,
                                   allow_topology_change=False)
        assert restored["step"] == 1 and restored["exposures"] == 1
        assert restored["sampler"] == {"next_index": 1}
        replay = structural_micro_step(bundle, samples[1], None, optimizer, scheduler,
                                       params, cfg["training"], 2)
        assert_same_state(expected["sample_id"], replay["sample_id"], "sample_id")
        assert_same_state(expected["loss"], replay["loss"], "loss")
        assert_same_state(expected["prodigy_d"], replay["prodigy_d"], "prodigy_d")
        assert_same_state(trainable, trainable_state(bundle.dit), "adapter")
        assert_same_state(optimizer_state, optimizer.state_dict(), "optimizer")
        assert_same_state(scheduler_state, scheduler.state_dict(), "scheduler")


def test_real_cpu_audit_requires_fresh_isolated_root_before_loading():
    from csgo_seen10.config import load_config
    from scripts import audit_csgo_seen10_model as audit_script

    with TemporaryDirectory() as temporary, patch.object(audit_script, "load_bundle", side_effect=AssertionError("loaded")):
        existing = Path(temporary) / "old"
        existing.mkdir()
        (existing / "prior_evidence.json").write_text("{}")
        try:
            audit_script.main(["--output-root", str(existing)])
        except ValueError as error:
            assert "fresh empty" in str(error)
        else:
            raise AssertionError("Existing audit evidence was overwritten")
        formal = Path(load_config()["paths"]["run_root"]) / "structural_probe"
        try:
            audit_script.main(["--output-root", str(formal)])
        except ValueError as error:
            assert "outside the formal run tree" in str(error)
        else:
            raise AssertionError("CPU structural probe entered the formal run tree")


def _tiny_model():
    torch.manual_seed(4)
    model = NextDiT(patch_size=2, in_channels=16, dim=32, n_layers=1, n_refiner_layers=1,
                    n_heads=2, n_kv_heads=1, multiple_of=16, qk_norm=True,
                    cap_feat_dim=24, axes_dims=(4, 4, 8), axes_lens=(300, 512, 512))
    # Native constructors deliberately zero the output and AdaLN gates. Activate
    # these paths or attention/condition parity would compare only zero images.
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and (name.endswith("adaLN_modulation.1") or name == "final_layer.linear"):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.normal_(module.bias, std=0.02)
    return model.eval()


def test_condition_branch_is_unchanged_by_injection():
    model = _tiny_model()
    before = condition_sha256(model)
    model.requires_grad_(False)
    replace_linear_with_lora(model, max_rank=4, scale=1.0, lora_dtype=torch.float32)
    assert condition_sha256(model) == before


@torch.no_grad()
def test_zero_native_lora_preserves_nondegenerate_output():
    model = _tiny_model()
    x = torch.randn(1, 16, 56, 56)
    radar = torch.randn(1, 16, 28, 28)
    features = torch.randn(1, 8, 24)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    args = (x, torch.tensor([0.4]), [[radar[0]]], features, mask, [["offset"]])
    original = model(*args)
    assert original.abs().max().item() > 1e-4
    changed_radar = model(x, args[1], [[radar[0] + 0.3]], features, mask, [["offset"]])
    changed_text = model(x, args[1], [[radar[0]]], features + 0.3, mask, [["offset"]])
    assert (changed_radar - original).abs().max().item() > 1e-6
    assert (changed_text - original).abs().max().item() > 1e-6
    before = condition_sha256(model)
    model.requires_grad_(False)
    replace_linear_with_lora(model, max_rank=4, scale=1.0, lora_dtype=torch.float32)
    assert condition_sha256(model) == before
    torch.testing.assert_close(model(*args), original, atol=0, rtol=0)


def test_strict_loader_keeps_trained_condition_and_rejects_missing_keys():
    model = _tiny_model()
    with torch.no_grad():
        model.cond_embedder.weight.fill_(0.17)
        model.cond_refiner[0].attention.qkv.weight.fill_(0.23)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "consolidated.00-of-01.pth"
        torch.save(model.state_dict(), path)
        digest = sha256_file(path)
        with patch("csgo_seen10.model.construct_dit", side_effect=lambda cfg, meta: _tiny_model().to("meta")), \
             patch("csgo_seen10.model.OFFICIAL_SHA256", digest):
            loaded, identity = load_official_dit(path, {}, device="cpu")
            assert identity["sha256"] == digest
            assert identity["condition_sha256"] == condition_sha256(model)
            assert torch.equal(loaded.cond_embedder.weight, model.cond_embedder.weight)
            assert torch.equal(loaded.cond_refiner[0].attention.qkv.weight, model.cond_refiner[0].attention.qkv.weight)
            # A BF16 compute copy must not rewrite the asset identity.
            loaded.to(dtype=torch.bfloat16)
            assert condition_sha256(loaded) != identity["condition_sha256"]
            assert identity["condition_sha256"] == condition_sha256(model)
            corrupted = dict(model.state_dict())
            corrupted.pop("cond_refiner.0.attention.qkv.weight")
            torch.save(corrupted, path)
            with patch("csgo_seen10.model.OFFICIAL_SHA256", sha256_file(path)):
                try:
                    load_official_dit(path, {}, device="cpu")
                except ValueError as error:
                    assert "missing" in str(error) or "lacks trained" in str(error)
                else:
                    raise AssertionError("Missing official condition tensor accepted")


@torch.no_grad()
def test_static_true_mask_and_positions_match_native_cpu():
    model = _tiny_model()
    x = torch.randn(2, 16, 56, 56)
    radar = torch.randn(2, 16, 28, 28)
    feats = torch.randn(2, 8, 24)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    time = torch.tensor([0.3, 0.7])
    native = model(x, time, [[r] for r in radar], feats, mask, [["offset"], ["offset"]])
    assert native.abs().max().item() > 1e-4
    prepared = prepare_condition(model, feats, mask, radar)
    static = static_denoise(model, x, time, prepared)
    torch.testing.assert_close(static, native, atol=2e-5, rtol=2e-5)
    assert prepared.full_mask.shape[1] == 256 + 196 + 784
    assert prepared.full_mask[:, :256].sum(dim=1).tolist() == [5, 3]
    ids = build_position_ids(torch.tensor([5, 3]))
    assert ids[0, 0].tolist() == [0, 0, 0]
    assert ids[0, 256].tolist() == [5, 28, 28]
    assert ids[0, 256 + 195].tolist() == [5, 41, 41]
    assert ids[0, 256 + 196].tolist() == [6, 0, 0]
    assert ids[0, -1].tolist() == [6, 27, 27]
    assert ids[1, 256].tolist() == [3, 28, 28]


@torch.no_grad()
def test_native_cfg_renorm_is_per_sample():
    class Fake(nn.Module):
        in_channels = 2

        def forward(self, x, t, cond, cap_feats, cap_mask, position_type):
            # Two distinct norms, so scalar tensor boolean would be invalid.
            return cap_feats[:, :1, :2].reshape(-1, 2, 1, 1).expand_as(x)

    from models_accessory.model import NextDiT
    fake = Fake()
    x = torch.zeros(4, 2, 2, 2)
    feats = torch.tensor([[[1., 0.]], [[2., 0.]], [[0., 1.]], [[0., 2.]]])
    result = NextDiT.forward_with_cfg(fake, x, torch.zeros(4), feats, torch.ones(4, 1, dtype=torch.bool),
                                      4.0, cond=[[]] * 4, position_type=[[]] * 4, renorm_cfg=1.0)
    assert result.shape == x.shape
    torch.testing.assert_close(result[:2].flatten(1).norm(dim=1), torch.tensor([2., 4.]))


@torch.no_grad()
def test_static_cfg_matches_official_eager_for_batch_two():
    from csgo_seen10.inference import _guided_velocity

    model = _tiny_model()
    state = torch.randn(2, 16, 56, 56)
    radar = torch.randn(2, 16, 28, 28)
    radar_cfg = torch.cat((radar, radar))
    feats = torch.randn(4, 8, 24)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    prepared = prepare_condition(model, feats, mask, radar_cfg)
    common = dict(cfg_scale=4.0, cfg_trunc=100.0, renorm_cfg=1.0)
    eager = _guided_velocity(model, state, 0.25, feats, mask, radar_cfg,
                             prepared=None, denoiser=None, engine="eager", **common)
    static = _guided_velocity(model, state, 0.25, feats, mask, radar_cfg,
                              prepared=prepared, denoiser=lambda x, t, p: static_denoise(model, x, t, p),
                              engine="compiled", **common)
    torch.testing.assert_close(static, eager, atol=3e-5, rtol=3e-5)


def test_native_single_resolution_latent_loss_contract():
    class NativeLoss:
        def training_losses(self, model, target, kwargs):
            predicted = model(target, torch.full((len(target),), 0.3), **kwargs)
            return {"loss": (predicted - target).square().flatten(1).mean(1)}

    bundle = AccessoryBundle(_tiny_model(), nn.Linear(1, 1), None, nn.Linear(1, 1), {})
    target = torch.randn(1, 16, 56, 56)
    radar = torch.randn(1, 16, 28, 28)
    feats = torch.randn(1, 8, 24)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    loss = bundle.latent_loss(target, radar, feats, mask, NativeLoss())["loss"]
    assert loss.shape == (1,) and loss.item() > 0


def test_text_encoder_rejects_overlength_before_gemma_call():
    class LongTokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": torch.zeros(1, 257, dtype=torch.long),
                    "attention_mask": torch.ones(1, 257, dtype=torch.long)}

    class ForbiddenEncoder(nn.Module):
        def forward(self, **kwargs):
            raise AssertionError("Overlength prompt reached Gemma")

    bundle = AccessoryBundle(nn.Linear(1, 1), ForbiddenEncoder(), LongTokenizer(), nn.Linear(1, 1), {})
    try:
        bundle.encode_text(["too long"])
    except ValueError as error:
        assert "256-token" in str(error)
    else:
        raise AssertionError("Overlength prompt was silently truncated")
