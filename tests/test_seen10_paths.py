"""Small filesystem contracts for relocating the released Seen-10 bundle."""

import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest, TestCase
from unittest.mock import patch

from csgo_seen10.config import PROJECT_ROOT, code_identity, content_hash, load_config, scientific_config, sha256_file
from csgo_seen10.data import Seen10Dataset, load_protocol, protocol_identity


SHARED_EVAL = Path(load_config()["paths"]["shared_eval_dir"])
FILE_FRAME = "file_num1_frame_1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def tiny_bundle(root, maps):
    ranges = {name: {"z_min": 0.0, "z_max": 10.0} for name in maps}
    calibration = {"calibration_sha256": "tiny-fingerprint", "z_ranges": ranges}
    write_json(root / "calibration/z_calibration.json", calibration)
    write_json(root / "benchmark_manifest.json", {
        "benchmark_id": "csgo_benchmark_v2",
        "protocol": {"seen_maps": list(maps)},
        "calibration": {"file": "calibration/z_calibration.json", "fingerprint": "tiny-fingerprint",
                        "z_ranges": ranges},
        "counts": {"seen": {maps[0]: {"train": 1}}},
    })
    write_json(root / "minimal_dataset_report.json", {
        "benchmark_id": "csgo_benchmark_v2", "status": "verified",
        "images": {"target_template": "images/{map}/{file_frame}.jpg"},
        "radars": {"root": "radars", "entries": [
            {"map": name, "target": f"{name}.jpg"} for name in maps]},
    })
    for name in maps:
        path = root / "radars" / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny radar bytes")
    image = root / "images" / maps[0] / f"{FILE_FRAME}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"tiny image bytes")
    write_json(root / "splits/seen" / maps[0] / "train.json", [
        {"map": maps[0], "file_frame": FILE_FRAME, "x": 1, "y": 2, "z": 3,
         "angle_h": 0.5, "angle_v": 0.25},
    ])


class PathCompatibilityTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SHARED_EVAL / "protocol.py").is_file():
            raise SkipTest(f"Shared protocol unavailable: {SHARED_EVAL / 'protocol.py'}")
        cls.protocol = load_protocol(SHARED_EVAL)

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        tiny_bundle(self.bundle, self.protocol.SEEN_MAPS)

    def config(self, root):
        cfg = load_config(overrides={"data_root": str(root), "shared_eval_dir": str(SHARED_EVAL)})
        cfg["data"]["manifest_sha256"] = sha256_file(root / "benchmark_manifest.json")
        cfg["data"]["calibration_file_sha256"] = sha256_file(root / "calibration/z_calibration.json")
        cfg["identity"]["science_sha256"] = content_hash(scientific_config(cfg))
        return cfg

    def first_row(self, root):
        return self.protocol.BenchmarkData(root).rows("seen_train", maps=[self.protocol.SEEN_MAPS[0]])[0]

    def test_relative_root_and_whole_bundle_symlink(self):
        relative = os.path.relpath(self.bundle, PROJECT_ROOT)
        cfg = load_config(overrides={"data_root": relative})
        self.assertEqual(Path(cfg["paths"]["data_root"]), self.bundle)
        self.assertEqual(Path(self.first_row(cfg["paths"]["data_root"])["image_path"]),
                         self.bundle / "images" / self.protocol.SEEN_MAPS[0] / f"{FILE_FRAME}.jpg")
        alias = self.root / "bundle_alias"
        alias.symlink_to(self.bundle, target_is_directory=True)
        self.assertEqual(self.first_row(alias)["sample_id"], self.first_row(self.bundle)["sample_id"])

        physical_cfg, alias_cfg = self.config(self.bundle), self.config(alias)
        self.assertEqual(physical_cfg["identity"], alias_cfg["identity"])
        self.assertEqual(protocol_identity(physical_cfg), protocol_identity(alias_cfg))
        self.assertEqual(len(Seen10Dataset(alias_cfg, "train", load_targets=False, limit=1)), 1)

    def test_external_image_and_radar_links_are_rejected(self):
        for leaf, expected in (("images", "image .* escapes data root"),
                               ("radars", "radar .* escapes data root")):
            with self.subTest(leaf=leaf):
                bundle = self.root / f"external_{leaf}"
                shutil.copytree(self.bundle, bundle)
                outside = self.root / f"outside_{leaf}"
                shutil.move(str(bundle / leaf), outside)
                (bundle / leaf).symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(self.protocol.BenchmarkDataError, expected):
                    self.first_row(bundle)

    def test_missing_image_is_rejected(self):
        (self.bundle / "images" / self.protocol.SEEN_MAPS[0] / f"{FILE_FRAME}.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing image"):
            self.first_row(self.bundle)

    def test_machine_env_cli_precedence_and_content_identity(self):
        moved = self.root / "moved_bundle"
        shutil.copytree(self.bundle, moved)
        machine = self.root / "machine.json"
        write_json(machine, {"paths": {"data_root": os.path.relpath(self.bundle, PROJECT_ROOT)}})
        with patch.dict(os.environ, {"DATA_ROOT": ""}):
            from_machine = load_config(machine_config=machine)
            from_default = load_config(machine_config=self.root / "absent.json")
            with patch.dict(os.environ, {"DATA_ROOT": str(moved)}):
                from_env = load_config(machine_config=machine)
                from_cli = load_config(machine_config=machine, overrides={"data_root": str(self.bundle)})
        self.assertEqual(from_default["paths"]["data_root"],
                         os.path.abspath(PROJECT_ROOT / "../UniLIP/data/csgo_benchmark_v2"))
        self.assertEqual(from_machine["paths"]["data_root"], str(self.bundle))
        self.assertEqual(from_env["paths"]["data_root"], str(moved))
        self.assertEqual(from_cli["paths"]["data_root"], str(self.bundle))
        self.assertEqual(from_machine["identity"], from_env["identity"])
        self.assertEqual(from_machine["identity"], from_cli["identity"])
        self.assertEqual(protocol_identity(self.config(self.bundle)), protocol_identity(self.config(moved)))
        before = code_identity()
        write_json(machine, {"paths": {"data_root": os.path.relpath(moved, PROJECT_ROOT)}})
        self.assertEqual(code_identity(), before)
