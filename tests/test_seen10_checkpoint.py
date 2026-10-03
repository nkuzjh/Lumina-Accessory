"""CPU checks for atomic selection, payload integrity and exact optimizer resume."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from csgo_seen10.checkpoint import (checkpoint_metadata, load_checkpoint, resolve_checkpoint,
                                     save_checkpoint)


def _fixture(optimizer_type="adam"):
    model = torch.nn.Linear(2, 1)
    if optimizer_type == "prodigy":
        from csgo_seen10.config import load_config
        from csgo_seen10.training import make_optimizer
        optimizer = make_optimizer(model.parameters(), load_config())
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def _step(model, optimizer, scheduler, sample_index=0):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(3, 2) + torch.tensor(np.random.random()) + sample_index
    x = x + random.random()
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


class CheckpointSemanticsTest(unittest.TestCase):
    def test_milestones_tie_late_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "train/checkpoints"
            model, optimizer, scheduler = _fixture()
            identity = {"effective_batch": 1, "seed": 42}
            topology = {"world_size": 1}
            sampler = {"global_offset": 4000}
            with self.assertRaises(ValueError):
                save_checkpoint(root, model=model, optimizer=optimizer, scheduler=scheduler,
                                step=1, exposures=1, sampler_state=sampler, identity=identity,
                                topology=topology, validation_loss=1.0)
            for step, loss in ((4000, 0.4), (8000, 0.4), (12000, 0.6),
                               (16000, 0.5), (19500, 0.3)):
                save_checkpoint(root, model=model, optimizer=optimizer, scheduler=scheduler,
                                step=step, exposures=step, sampler_state={"global_offset": step},
                                identity=identity, topology=topology, validation_loss=loss)
                if step == 8000:
                    self.assertEqual(checkpoint_metadata(resolve_checkpoint(root, "best"))["step"], 4000)
            self.assertEqual(checkpoint_metadata(resolve_checkpoint(root, "late"))["step"], 19500)
            self.assertEqual(checkpoint_metadata(resolve_checkpoint(root.parent, "best"))["step"], 19500)
            adapter = resolve_checkpoint(root, "latest") / "adapter.pt"
            with adapter.open("ab") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                checkpoint_metadata(adapter.parent)

    def test_prodigy_rng_and_state_resume(self):
        try:
            import prodigyopt  # noqa: F401
        except ImportError:
            self.skipTest("Prodigy is not installed in the current CPU environment")
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        from csgo_seen10.training import GlobalSourceStream
        stream = GlobalSourceStream(5, seed=3, effective_batch=1)
        first_id, next_id = stream.index(0), stream.index(1)
        model, optimizer, scheduler = _fixture("prodigy")
        _step(model, optimizer, scheduler, first_id)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "train/checkpoints"
            identity = {"effective_batch": 1, "seed": 3}
            topology = {"world_size": 1, "micro_batch_size": 1, "gradient_accumulation_steps": 1}
            save_checkpoint(root, model=model, optimizer=optimizer, scheduler=scheduler,
                            step=1, exposures=1, sampler_state=stream.state_after(1),
                            identity=identity, topology=topology, validation_loss=0.5, smoke=True)
            uninterrupted_loss = _step(model, optimizer, scheduler, next_id)
            uninterrupted_weights = {k: v.clone() for k, v in model.state_dict().items()}
            uninterrupted_opt = optimizer.state_dict()
            uninterrupted_scheduler = scheduler.state_dict()
            fresh_model, fresh_optimizer, fresh_scheduler = _fixture("prodigy")
            result = load_checkpoint(resolve_checkpoint(root, "latest"), model=fresh_model,
                                     optimizer=fresh_optimizer, scheduler=fresh_scheduler,
                                     identity=identity, topology=topology)
            self.assertFalse(result["topology_changed"])
            restored_stream = GlobalSourceStream(5, seed=3, effective_batch=1)
            restored_stream.check_resume(result["sampler"], 1)
            self.assertEqual(next_id, restored_stream.index(1))
            resumed_loss = _step(fresh_model, fresh_optimizer, fresh_scheduler, restored_stream.index(1))
            self.assertEqual(uninterrupted_loss, resumed_loss)
            for key, value in uninterrupted_weights.items():
                self.assertTrue(torch.equal(value, fresh_model.state_dict()[key]), key)
            self.assertEqual(uninterrupted_scheduler, fresh_scheduler.state_dict())
            for parameter_id, state in uninterrupted_opt["state"].items():
                other = fresh_optimizer.state_dict()["state"][parameter_id]
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, other[key]) if isinstance(value, torch.Tensor)
                                    else value == other[key], (parameter_id, key))


if __name__ == "__main__":
    unittest.main()
