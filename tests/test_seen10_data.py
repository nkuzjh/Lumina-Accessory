import builtins
import io
import os
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from csgo_seen10.config import batch_configuration, load_config, scientific_config, content_hash
from csgo_seen10.data import Seen10Dataset, make_prompt, protocol_identity


class ConfigurationTests(unittest.TestCase):
    def test_batch_products(self):
        for w, m, a in [(1, 1, 128), (1, 4, 32), (2, 4, 16), (4, 4, 8), (8, 4, 4)]:
            self.assertEqual(batch_configuration(w, m), a)
            self.assertEqual(batch_configuration(w, m, a), a)
        for args in [(3, 4), (2, 4, 8), (0, 1), (1, 0), (1, 256)]:
            with self.assertRaises(ValueError):
                batch_configuration(*args)

    def test_paths_do_not_change_science_identity(self):
        a = load_config()
        b = load_config(overrides={"data_root": "/new/machine/data", "run_root": "/another/run"})
        self.assertEqual(content_hash(scientific_config(a)), content_hash(scientific_config(b)))
        with patch.dict(os.environ, {"DATA_ROOT": "/env/path"}):
            self.assertEqual(load_config(overrides={"data_root": "/cli/path"})["paths"]["data_root"], "/cli/path")
        # Resolving venv/bin/python through its symlink loses installed packages.
        self.assertTrue(load_config()["paths"]["model_python"].endswith(".venv/bin/python"))

    def test_prompt_uses_raw_radians(self):
        row = {"map_name": "de_train", "pose_raw": dict(x=12.3456789, y=0, z=-2.5, pitch=0.1, yaw=6.2)}
        prompt = make_prompt(row)
        self.assertIn("x=12.3456789", prompt)
        self.assertIn("pitch=0.1 rad, yaw=6.2 rad", prompt)


class ReleasedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        if not Path(cls.cfg["paths"]["data_root"]).is_dir():
            raise unittest.SkipTest("Released bundle unavailable")

    def test_hashes_counts_and_shapes(self):
        identity = protocol_identity(self.cfg)
        self.assertEqual(identity["maps"], self.cfg["data"]["maps"])
        for split, count in [("train", 50000), ("validation", 5000), ("discrete", 20000), ("continuous", 12800)]:
            ds = Seen10Dataset(self.cfg, split)
            self.assertEqual(len(ds), count)
        row = Seen10Dataset(self.cfg, "train", limit=1)[0]
        self.assertEqual(tuple(row["radar"].shape), (3, 224, 224))
        self.assertEqual(tuple(row["target"].shape), (3, 448, 448))

    def test_inference_rejects_actual_target_opens(self):
        images = Path(self.cfg["paths"]["data_root"]) / "images"
        attempts = []
        def guard(original):
            def opened(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)):
                    p = Path(os.fsdecode(file)).resolve()
                    if images in p.parents:
                        attempts.append(str(p))
                        raise AssertionError(f"Inference attempted target read {p}")
                return original(file, *args, **kwargs)
            return opened
        # Intercept both Path.open and PIL/open. stat/existence checks are permitted by shared protocol.
        with patch("builtins.open", guard(builtins.open)), patch("io.open", guard(io.open)):
            for split in ("discrete", "continuous"):
                dataset = Seen10Dataset(self.cfg, split, load_targets=False, limit=2)
                for row in dataset:
                    self.assertNotIn("target", row)
                    self.assertEqual(tuple(row["radar"].shape), (3, 224, 224))
        self.assertFalse(attempts)
        with self.assertRaises(ValueError):
            Seen10Dataset(self.cfg, "discrete", load_targets=True)

    def test_full_inference_io_denies_target_reads(self):
        """Execute production inference I/O with stub neural compute and real radar rows."""
        import torch
        from csgo_seen10.inference import run_inference
        images = Path(self.cfg["paths"]["data_root"]) / "images"
        builtin_open, io_open = builtins.open, io.open
        def guard(original):
            def checked(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)) and images in Path(os.fsdecode(file)).resolve().parents:
                    raise AssertionError(f"Target read during production inference I/O: {file}")
                return original(file, *args, **kwargs)
            return checked
        bundle = SimpleNamespace(
            device=torch.device('cpu'), dit=torch.nn.Linear(1,1), base_identity={}, component_identity={},
            encode_images=lambda image: torch.zeros(len(image),16,image.shape[-2]//8,image.shape[-1]//8),
            encode_text=lambda prompts: (torch.zeros(len(prompts),8,24),torch.ones(len(prompts),8,dtype=torch.bool)))
        with tempfile.TemporaryDirectory() as td, \
             patch('builtins.open',guard(builtin_open)), patch('io.open',guard(io_open)), \
             patch('csgo_seen10.inference.sample_latents',side_effect=lambda bundle,noise,*a,**kw:(noise,49)), \
             patch('csgo_seen10.inference.decode_latents',side_effect=lambda bundle,latent,**kw:torch.full((len(latent),3,448,448),0.5)):
            for task in ('discrete','continuous'):
                dataset = Seen10Dataset(self.cfg,task,load_targets=False,limit=2)
                report = run_inference(bundle,dataset,Path(td)/task,task=task,checkpoint_identity={'stub_only':True},
                                       cfg=self.cfg,engine='eager',compile_mode='default',batch_size=2,vae_batch_size=1)
                self.assertEqual(report['generated'],2)


if __name__ == "__main__":
    unittest.main()
