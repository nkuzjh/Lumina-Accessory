"""CPU checks for the exact source stream and distributed accumulation."""
from __future__ import annotations

import os
import tempfile
import unittest
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from csgo_seen10.config import batch_configuration
from csgo_seen10.training import GlobalSourceStream, _checkpoint_activations, _fixed_validation_transport, validate


class TinySeenDataset:
    def __init__(self, cfg, split, *, limit=None, **_):
        self.split = split
        self.size = min(5 if split == "seen_train" else 2, limit or 99)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"sample_id": f"sample_{index}", "prompt": f"pose_{index}",
                "radar": torch.tensor([float(index + 1)]),
                "target": torch.tensor([float(index % 3)])}


class TinyBundle:
    def __init__(self):
        from models_accessory.lora import LinearLora
        self.dit = torch.nn.Sequential(LinearLora(1, 1, bias=True, rank=1,
                                                   dtype=torch.float32, device=torch.device("cpu"),
                                                   lora_dtype=torch.float32))
        with torch.no_grad():
            layer = self.dit[0]
            layer.weight.fill_(0.5)
            layer.bias.zero_()
            layer.lora_A.weight.fill_(0.25)
            layer.lora_B.weight.zero_()
            layer.lora_B.bias.zero_()
        self.dit[0].weight.requires_grad_(False)
        self.dit[0].bias.requires_grad_(False)
        self.base_identity = {"sha256": "tiny-base"}
        self.component_identity = {"sha256": "tiny-components"}

    def training_loss(self, target, radar, prompts, transport, *, caption_dropout):
        # All three RNG families are part of the actual training loop's state.
        import random
        import numpy as np
        noise = torch.randn_like(target) * 0.02
        shift = random.random() * 0.01 + np.random.random() * 0.01
        keep = (torch.rand(len(prompts), 1) >= caption_dropout).float()
        prediction = self.dit(radar) * keep
        return {"loss": (prediction - target - noise - shift).square().flatten(1).mean(1)}


class RecordingTqdm:
    instances = []
    messages = []

    def __init__(self, *, total, initial=0, desc, disable=False, **kwargs):
        self.total, self.initial, self.n = total, initial, initial
        self.desc, self.disable, self.closed = desc, disable, False
        self.postfix = None
        self.instances.append(self)

    def set_postfix(self, **kwargs):
        self.postfix = kwargs

    def update(self, count):
        self.n += count

    def close(self):
        self.closed = True

    @classmethod
    def write(cls, message, *, file):
        cls.messages.append((message, file))


def _tree_equal(testcase, left, right):
    if isinstance(left, torch.Tensor):
        testcase.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        testcase.assertEqual(set(left), set(right))
        for key in left:
            _tree_equal(testcase, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        testcase.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            _tree_equal(testcase, a, b)
    else:
        testcase.assertEqual(left, right)


def _ddp_worker(rank, init_file, result_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = DDP(torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(1.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        stream = GlobalSourceStream(5, seed=42, effective_batch=4)
        all_indices = []
        for micro in range(2):
            indices = stream.micro_indices(1, rank, 2, 1, 2, micro)
            all_indices += indices
            x = torch.tensor(indices, dtype=torch.float32).reshape(-1, 1)
            context = model.no_sync() if micro == 0 else __import__("contextlib").nullcontext()
            with context:
                loss = model(x).square().mean() / 2
                loss.backward()
        gradient = model.module.weight.grad.item()
        gathered = [None, None]
        dist.all_gather_object(gathered, all_indices)
        if rank == 0:
            torch.save({"indices": gathered, "gradient": gradient}, result_file)
        optimizer.step()
    finally:
        dist.destroy_process_group()


def _validation_worker(rank, init_file, result_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        class Bundle:
            def __init__(self):
                self.dit = torch.nn.Linear(1, 1)
                self.seen = []

            def training_loss(self, target, radar, prompts, transport, *, caption_dropout):
                indices = [int(prompt.split("_")[-1]) for prompt in prompts]
                self.seen.extend(indices)
                return {"loss": torch.tensor(indices, dtype=torch.float32)}

        bundle = Bundle()
        loss, count = validate(bundle, TinySeenDataset({}, "seen_train"), SimpleNamespace(),
                               rank=rank, world=2, micro_batch_size=2, device=torch.device("cpu"), seed=424242)
        gathered = [None, None]
        dist.all_gather_object(gathered, bundle.seen)
        if rank == 0:
            torch.save({"loss": loss, "count": count, "seen": gathered}, result_file)
    finally:
        dist.destroy_process_group()


class TrainingSemanticsTest(unittest.TestCase):
    def test_validation_progress_counts_batches_and_closes_on_error(self):
        class Bundle:
            def __init__(self):
                self.dit = torch.nn.Linear(1, 1)
                self.fail = False

            def training_loss(self, target, radar, prompts, transport, *, caption_dropout):
                if self.fail and prompts[0] == "pose_2":
                    raise RuntimeError("validation failed")
                return {"loss": torch.tensor([int(prompt[5:]) for prompt in prompts], dtype=torch.float32)}

        bundle = Bundle()
        RecordingTqdm.instances.clear()
        with patch("csgo_seen10.training.tqdm", RecordingTqdm):
            loss, count = validate(bundle, TinySeenDataset({}, "seen_train"), SimpleNamespace(),
                                   rank=0, world=1, micro_batch_size=2, device=torch.device("cpu"), seed=42)
            self.assertEqual((loss, count), (2.0, 5))
            self.assertEqual((RecordingTqdm.instances[-1].total, RecordingTqdm.instances[-1].n), (3, 3))
            self.assertTrue(RecordingTqdm.instances[-1].closed)
            bundle.fail = True
            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                validate(bundle, TinySeenDataset({}, "seen_train"), SimpleNamespace(),
                         rank=0, world=1, micro_batch_size=2, device=torch.device("cpu"), seed=42)
            self.assertTrue(RecordingTqdm.instances[-1].closed)
            self.assertTrue(bundle.dit.training)

    def test_fixed_validation_noise_is_sample_stable(self):
        class Transport:
            train_eps = sample_eps = 0.0
            snr_type = "lognorm"
            do_shift = False

            def check_interval(self, *_):
                return 0.0, 1.0

        original = Transport()
        batch_a = _fixed_validation_transport(original, [1, 2], 424242)
        batch_b = _fixed_validation_transport(original, [2, 3], 424242)
        ta, noise_a, _ = batch_a.sample(torch.zeros(2, 1, 4, 4))
        tb, noise_b, _ = batch_b.sample(torch.zeros(2, 1, 4, 4))
        self.assertTrue(torch.equal(noise_a[1], noise_b[0]))
        self.assertTrue(torch.equal(ta[1], tb[0]))
        # A scalar fixture exposes accidental reuse of the same normal draw for
        # timestep and noise; separate random domains must not share that value.
        scalar = _fixed_validation_transport(original, [2], 424242)
        timestep, noise, _ = scalar.sample(torch.zeros(1, 1))
        self.assertFalse(torch.equal(timestep, noise.flatten().sigmoid()))

    def test_validation_two_rank_unbiased_count(self):
        with tempfile.TemporaryDirectory() as td:
            result_file = os.path.join(td, "result.pt")
            mp.spawn(_validation_worker, args=(os.path.join(td, "init"), result_file), nprocs=2, join=True)
            result = torch.load(result_file, weights_only=True)
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["loss"], 2.0)
        self.assertCountEqual(result["seen"][0] + result["seen"][1], list(range(5)))

    def test_run_training_resume_and_run_root_guard(self):
        try:
            import prodigyopt  # noqa: F401
        except ImportError:
            self.skipTest("Prodigy is not installed in the current CPU environment")
        from csgo_seen10.config import load_config
        from csgo_seen10.checkpoint import load_checkpoint
        from csgo_seen10.training import make_optimizer, run_training
        bundles = []

        def bundle_factory(*_args, **_kwargs):
            bundle = TinyBundle()
            bundles.append(bundle)
            return bundle

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            RecordingTqdm.instances.clear()
            RecordingTqdm.messages.clear()
            stack.enter_context(patch("csgo_seen10.training.tqdm", RecordingTqdm))
            stack.enter_context(patch("csgo_seen10.training.distributed_context",
                                      return_value=(0, 1, torch.device("cpu"))))
            stack.enter_context(patch("csgo_seen10.model.load_bundle", side_effect=bundle_factory))
            stack.enter_context(patch("csgo_seen10.training.Seen10Dataset", TinySeenDataset))
            stack.enter_context(patch("csgo_seen10.training.protocol_identity",
                                      return_value={"data_sha256": "tiny-protocol"}))
            stack.enter_context(patch("csgo_seen10.training.make_transport", return_value=object()))
            stack.enter_context(patch("csgo_seen10.lora_audit.audit_trainable", return_value={"ok": True}))
            stack.enter_context(patch("csgo_seen10.training.validate",
                                      side_effect=lambda bundle, dataset, *_args, **_kwargs: (0.5, len(dataset))))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            full_root, resume_root = Path(td) / "smoke_full", Path(td) / "smoke_resume"
            cfg = load_config(overrides={"run_root": str(full_root)})
            cfg["training"]["effective_batch"] = 4
            cfg["training"]["activation_checkpointing"] = False
            run_training(cfg, micro_batch_size=2, smoke=True, smoke_steps=2, smoke_limit=5,
                         validation_limit=2)
            full_bundle = bundles[-1]
            cfg["paths"]["run_root"] = str(resume_root)
            run_training(cfg, micro_batch_size=2, smoke=True, smoke_steps=1, smoke_limit=5,
                         validation_limit=2)
            run_training(cfg, micro_batch_size=2, smoke=True, smoke_steps=2, smoke_limit=5,
                         validation_limit=2, resume="latest")
            train_bars = [bar for bar in RecordingTqdm.instances if bar.desc == "Training"]
            self.assertEqual([(bar.initial, bar.total, bar.n) for bar in train_bars],
                             [(0, 2, 2), (0, 1, 1), (1, 2, 2)])
            self.assertTrue(all(bar.closed and not bar.disable for bar in train_bars))
            self.assertIn("loss", train_bars[-1].postfix)
            self.assertFalse(train_bars[-1].postfix["refresh"])
            resumed_bundle = bundles[-1]
            def events(root):
                return [json.loads(line) for line in (root / "train/events.jsonl").read_text().splitlines()
                        if json.loads(line).get("event") == "train"]
            full_events, resumed_events = events(full_root), events(resume_root)
            all_events = [json.loads(line) for line in (resume_root / "train/events.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in all_events if row["event"] == "checkpoint"], [1, 2])
            self.assertEqual([row["path"] for row in all_events if row["event"] == "checkpoint"],
                             [str(resume_root / "train/checkpoints" / f"step_{step:08d}") for step in (1, 2)])
            self.assertTrue(all(row["seconds"] >= 0 for row in all_events
                                if row["event"] in {"validation", "checkpoint"}))
            self.assertTrue(all(file is sys.stderr for _, file in RecordingTqdm.messages))
            self.assertTrue(any("Validation step=2 loss=0.5 count=2" in message
                                for message, _ in RecordingTqdm.messages))
            self.assertTrue(any(f"Checkpoint step=2 path={resume_root / 'train/checkpoints/step_00000002'}"
                                in message for message, _ in RecordingTqdm.messages))
            self.assertEqual([r["samples_this_rank"] for r in full_events],
                             [r["samples_this_rank"] for r in resumed_events])
            self.assertEqual([r["loss"] for r in full_events], [r["loss"] for r in resumed_events])
            self.assertTrue(torch.equal(full_bundle.dit[0].weight, resumed_bundle.dit[0].weight))
            self.assertIsNone(resumed_bundle.dit[0].weight.grad)
            self.assertNotEqual(float(resumed_bundle.dit[0].lora_B.bias.detach()), 0.0)
            full_state = torch.load(full_root / "train/checkpoints/step_00000002/state.pt",
                                    map_location="cpu", weights_only=False)
            resumed_state = torch.load(resume_root / "train/checkpoints/step_00000002/state.pt",
                                       map_location="cpu", weights_only=False)
            _tree_equal(self, full_state["optimizer"], resumed_state["optimizer"])
            _tree_equal(self, full_state["scheduler"], resumed_state["scheduler"])
            for a, b in zip(full_bundle.dit.parameters(), resumed_bundle.dit.parameters()):
                self.assertTrue(torch.equal(a, b))
            branch_root = Path(td) / "smoke_branch"
            cfg["paths"]["run_root"] = str(branch_root)
            run_training(cfg, micro_batch_size=2, smoke=True, smoke_steps=2, smoke_limit=5,
                         validation_limit=2,
                         resume=str(resume_root / "train/checkpoints/step_00000001"))
            self.assertEqual(events(branch_root)[0]["loss"], full_events[1]["loss"])
            self.assertEqual((branch_root / "train/checkpoints/best").resolve(),
                             (resume_root / "train/checkpoints/step_00000001").resolve())
            identity = json.loads((resume_root / "train/checkpoints/step_00000002/metadata.json").read_text())["identity"]
            changed_model = TinyBundle().dit
            changed_optimizer = make_optimizer((p for p in changed_model.parameters() if p.requires_grad), cfg)
            changed_scheduler = torch.optim.lr_scheduler.LambdaLR(changed_optimizer, lambda _: 1.0)
            changed = load_checkpoint(resume_root / "train/checkpoints/step_00000002",
                                      model=changed_model, optimizer=changed_optimizer,
                                      scheduler=changed_scheduler, identity=identity,
                                      topology={"world_size": 2, "micro_batch_size": 1,
                                                "gradient_accumulation_steps": 2,
                                                "device_type": "cpu", "cuda_device_count": 0})
            self.assertTrue(changed["topology_changed"])
            self.assertIn("non-bitwise", changed["resume_precision"])
            formal_root = Path(td) / "formal_existing"
            formal_root.mkdir()
            (formal_root / "existing.txt").write_text("preserve")
            cfg["paths"]["run_root"] = str(formal_root)
            with self.assertRaises(FileExistsError):
                run_training(cfg, micro_batch_size=2, smoke=False)
            self.assertEqual((formal_root / "existing.txt").read_text(), "preserve")
            cfg["paths"]["run_root"] = str(Path(td) / "formal_new")
            with self.assertRaisesRegex(ValueError, "histories must remain separate"):
                run_training(cfg, micro_batch_size=2, smoke=False,
                             resume=str(resume_root / "train/checkpoints/step_00000001"))
            cfg["paths"]["run_root"] = str(Path(td) / "smoke_run" / "new_branch")
            with self.assertRaisesRegex(ValueError, "Formal training cannot write"):
                run_training(cfg, micro_batch_size=2, smoke=False)

    def test_activation_checkpoint_keeps_adapter_names(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

            def get_checkpointing_wrap_module_list(self):
                return list(self.layers)

            def forward(self, x):
                return self.layers[0](x).square().sum()

        model = Tiny()
        names = list(model.state_dict())
        _checkpoint_activations(model)
        self.assertEqual(names, list(model.state_dict()))
        model(torch.ones(1, 2)).backward()
        self.assertIsNotNone(model.layers[0].weight.grad)

    def test_progress_closes_on_save_failure_and_nonzero_rank_is_quiet(self):
        try:
            import prodigyopt  # noqa: F401
        except ImportError:
            self.skipTest("Prodigy is not installed in the current CPU environment")
        from csgo_seen10.config import load_config
        from csgo_seen10.training import run_training

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            RecordingTqdm.instances.clear()
            RecordingTqdm.messages.clear()
            stack.enter_context(patch("csgo_seen10.training.tqdm", RecordingTqdm))
            rank = stack.enter_context(patch("csgo_seen10.training.distributed_context",
                                             return_value=(0, 1, torch.device("cpu"))))
            stack.enter_context(patch("csgo_seen10.model.load_bundle", side_effect=lambda *_a, **_kw: TinyBundle()))
            stack.enter_context(patch("csgo_seen10.training.Seen10Dataset", TinySeenDataset))
            stack.enter_context(patch("csgo_seen10.training.protocol_identity",
                                      return_value={"data_sha256": "tiny-protocol"}))
            stack.enter_context(patch("csgo_seen10.training.make_transport", return_value=object()))
            stack.enter_context(patch("csgo_seen10.lora_audit.audit_trainable", return_value={"ok": True}))
            stack.enter_context(patch("csgo_seen10.training.validate", return_value=(0.5, 2)))
            stack.enter_context(patch("torch.cuda.is_available", return_value=False))
            saved = stack.enter_context(patch("csgo_seen10.training.save_checkpoint",
                                              side_effect=RuntimeError("disk failed")))
            failed_root = Path(td) / "smoke_failed"
            cfg = load_config(overrides={"run_root": str(failed_root)})
            cfg["training"]["effective_batch"] = 4
            cfg["training"]["activation_checkpointing"] = False
            with self.assertRaisesRegex(RuntimeError, "disk failed"):
                run_training(cfg, micro_batch_size=2, smoke=True, smoke_steps=1,
                             smoke_limit=5, validation_limit=2)
            saved.assert_called_once()
            failed_events = [json.loads(line) for line in (failed_root / "train/events.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in failed_events], ["train", "validation"])
            self.assertTrue(RecordingTqdm.instances[-1].closed)
            self.assertFalse(any(message.startswith("Checkpoint") for message, _ in RecordingTqdm.messages))

            # Isolate the rank output gate without introducing a process group or DDP.
            rank.return_value = (1, 1, torch.device("cpu"))
            saved.side_effect = None
            quiet_root = Path(td) / "smoke_quiet"
            saved.return_value = quiet_root / "train/checkpoints/step_00000001"
            cfg["paths"]["run_root"] = str(quiet_root)
            RecordingTqdm.messages.clear()
            run_training(cfg, micro_batch_size=2, smoke=True, smoke_steps=1,
                         smoke_limit=5, validation_limit=2)
            self.assertTrue(RecordingTqdm.instances[-1].disable)
            self.assertTrue(RecordingTqdm.instances[-1].closed)
            self.assertEqual(RecordingTqdm.messages, [])
            self.assertFalse((quiet_root / "train/events.jsonl").exists())

    def test_batch_contract_and_epoch_crossing(self):
        self.assertEqual(batch_configuration(2, 4, None, 128), 16)
        for args in ((3, 4, None), (2, 4, 15), (0, 4, None)):
            with self.assertRaises(ValueError):
                batch_configuration(*args, effective=128)
        stream = GlobalSourceStream(5, 42, effective_batch=4)
        first = [stream.index(i) for i in range(4)]
        second = [stream.index(i) for i in range(4, 8)]
        self.assertEqual(len(set(first + second[:1])), 5)
        self.assertEqual(stream.state_after(2)["position_in_epoch"], 3)
        self.assertEqual(stream.state_after(2)["epoch"], 1)
        self.assertEqual(first + second, [stream.index(i) for i in range(8)])

    def test_two_rank_no_sync_scaling(self):
        with tempfile.TemporaryDirectory() as td:
            init_file = os.path.join(td, "init")
            result_file = os.path.join(td, "result.pt")
            mp.spawn(_ddp_worker, args=(init_file, result_file), nprocs=2, join=True)
            result = torch.load(result_file, weights_only=True)
        stream = GlobalSourceStream(5, 42, effective_batch=4)
        expected = [stream.index(i) for i in range(4, 8)]
        flattened = result["indices"][0] + result["indices"][1]
        self.assertCountEqual(flattened, expected)
        # d/dw mean((w*x)^2) at w=1, with DDP mean across two ranks.
        self.assertAlmostEqual(result["gradient"], sum(2 * i * i for i in expected) / 4, places=5)


if __name__ == "__main__":
    unittest.main()
