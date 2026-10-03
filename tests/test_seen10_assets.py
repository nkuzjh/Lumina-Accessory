"""Offline integrity and provenance checks for the fixed official assets."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import pickle
import threading
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "seen10_assets", PROJECT / "scripts/download_csgo_seen10_assets.py")
assert SPEC and SPEC.loader
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def test_manifest_selects_only_accessory_normal_and_frozen_components():
    data = assets.manifest()
    assert data["repositories"][0]["revision"] == "711d5d6656c62957e8625b02ea53cc74f2c5589d"
    base = data["repositories"][0]["files"]
    assert {file["path"] for file in base} == {"consolidated.00-of-01.pth", "model_args.pth"}
    assert base[0]["digest"] == "b787a35ab72e8fe14b908e3795c08baa81556dd6b219b0bdc8c495c18386e700"
    components = {file["path"] for file in data["repositories"][1]["files"]}
    assert any(path.startswith("text_encoder/") for path in components)
    assert any(path.startswith("tokenizer/") for path in components)
    assert any(path.startswith("vae/") for path in components)
    assert not any(path.startswith("transformer/") for path in components)
    assert data["static_model_args"]["verified_without_unpickle"]["qk_norm"] is True


def test_hash_checker_rejects_same_size_corruption(tmp_path: Path):
    repo = {"repo_id": "official/test", "revision": "a" * 40,
            "target": "target", "files": [{"path": "weight.bin", "size": 4,
            "digest_type": "sha256", "digest": hashlib.sha256(b"good").hexdigest(),
            "sha256": hashlib.sha256(b"good").hexdigest()}]}
    path = assets.destination(tmp_path, repo, repo["files"][0])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"good")
    assert assets.inspect(tmp_path, {"repositories": [repo]})[0]["status"] == "ok"
    path.write_bytes(b"evil")
    assert assets.inspect(tmp_path, {"repositories": [repo]})[0]["status"] == "hash_mismatch"


def test_offline_missing_and_canonical_paths(tmp_path: Path, capsys):
    data = assets.manifest()
    statuses = assets.inspect(tmp_path, data)
    assert statuses and all(row["status"] == "missing" for row in statuses)
    assets.print_env(tmp_path)
    exports = capsys.readouterr().out
    assert f"OFFICIAL_BASE_CHECKPOINT={tmp_path}/Lumina-Accessory/consolidated.00-of-01.pth" in exports
    assert f"GEMMA_PATH={tmp_path}/components/text_encoder" in exports
    assert f"TOKENIZER_PATH={tmp_path}/components/tokenizer" in exports
    assert f"VAE_PATH={tmp_path}/components/vae" in exports


def test_ranged_download_discards_untrusted_part_then_recovers(tmp_path: Path, monkeypatch):
    content = bytes(range(100))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            byte_range = self.headers["Range"].removeprefix("bytes=")
            start, end = map(int, byte_range.split("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(content[start:end + 1])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        monkeypatch.setattr(assets, "RANGE_BYTES", 16)
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("CSGO_ASSET_BYPASS_PROXY", "1")
        repo = {"repo_id": "official/test", "revision": "a" * 40, "target": "target"}
        item = {"path": "test.bin", "size": len(content), "digest_type": "sha256",
                "digest": hashlib.sha256(content).hexdigest(), "sha256": hashlib.sha256(content).hexdigest()}
        target = assets.destination(tmp_path, repo, item)
        parts = target.parent / ".test.bin.parts"
        parts.mkdir(parents=True)
        (parts / "00000.part").write_bytes(b"x" * 16)
        try:
            assets.download_large_by_ranges(tmp_path, repo, item)
        except RuntimeError as exc:
            assert "discarded untrusted temporary ranges" in str(exc)
        else:
            raise AssertionError("Corrupt same-size range was accepted")
        assert not parts.exists()
        assets.download_large_by_ranges(tmp_path, repo, item)
        assert target.read_bytes() == content
    finally:
        server.shutdown()
        worker.join(timeout=2)


def test_model_args_static_reader_does_not_execute_pickle(tmp_path: Path):
    marker = tmp_path / "should_not_exist"

    class Unsafe:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    archive = tmp_path / "model_args.pth"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("model_args/data.pkl", pickle.dumps(Unsafe()))
    assert assets.inspect_model_args_without_unpickle(archive, {}) == {}
    assert not marker.exists()


def test_asset_path_precedence_matches_experiment_config(tmp_path: Path, monkeypatch, capsys):
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps({"paths": {"gemma_path": str(tmp_path / "machine-gemma")}}))
    monkeypatch.setenv("GEMMA_PATH", str(tmp_path / "env-gemma"))
    assert assets.main(["--dry-run", "--json", "--machine-config", str(machine)]) == 0
    assert json.loads(capsys.readouterr().out)["paths"]["gemma_path"] == str(tmp_path / "env-gemma")
    assert assets.main(["--dry-run", "--json", "--machine-config", str(machine),
                        "--gemma-path", str(tmp_path / "cli-gemma")]) == 0
    assert json.loads(capsys.readouterr().out)["paths"]["gemma_path"] == str(tmp_path / "cli-gemma")
