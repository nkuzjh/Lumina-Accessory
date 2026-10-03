#!/usr/bin/env python3
"""Fetch and verify the fixed official Accessory and frozen component files.

This never deserializes the published model_args.pth.  The official normal
checkpoint is the only DiT weight selected by this manifest.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickletools
import shlex
import shutil
import sys
import time
import zipfile

RANGE_BYTES = 32 * 1024 * 1024
LARGE_FILE_BYTES = 128 * 1024 * 1024

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("csgo_seen10_assets.json")
EXPERIMENT = "csgo_seen10_exp32gen_aligned"
sys.path.insert(0, str(ROOT))
from csgo_seen10.config import load_config  # noqa: E402 (stdlib-only module)


def manifest() -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported asset manifest schema")
    for repo in data["repositories"]:
        if not repo["revision"] or len(repo["revision"]) != 40:
            raise ValueError("Every repository requires a full commit revision")
        for item in repo["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe manifest path")
    return data


def destination(root: Path, repo: dict, item: dict, paths: dict | None = None) -> Path:
    if paths:
        relative = Path(item["path"])
        if repo["target"] == "Lumina-Accessory":
            return Path(paths["base_checkpoint"]) if relative.name == "consolidated.00-of-01.pth" else Path(paths["base_checkpoint"]).parent / relative.name
        component = relative.parts[0]
        prefix = {"text_encoder": "gemma_path", "tokenizer": "tokenizer_path", "vae": "vae_path"}[component]
        return Path(paths[prefix]).joinpath(*relative.parts[1:])
    return root / repo["target"] / item["path"]


def hub_cache_root(explicit: Path | None) -> Path:
    if explicit:
        path = explicit.expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve() / "hub"
    return Path.home() / ".cache/huggingface/hub"


def cached_snapshot(cache: Path, repo: dict, item: dict) -> Path:
    return cache / ("models--" + repo["repo_id"].replace("/", "--")) / "snapshots" / repo["revision"] / item["path"]


def copy_verified(source: Path, target: Path, item: dict) -> bool:
    if not source.is_file() or source.stat().st_size != item["size"] or digest_file(source, "sha256") != item["sha256"]:
        return False
    if source.resolve() == target.resolve():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copyfile(source, temporary)
    if digest_file(temporary, "sha256") != item["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Verified cache copy changed unexpectedly: {target}")
    temporary.replace(target)
    return True


def digest_file(path: Path, kind: str) -> str:
    size = path.stat().st_size
    if kind == "sha256":
        digest = hashlib.sha256()
    elif kind == "git-blob-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode("ascii"))
    else:
        raise ValueError(f"Unknown manifest digest type: {kind}")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model_args_without_unpickle(path: Path, expected: dict) -> dict:
    """Read literal scalar pickle opcodes in the authenticated tiny Torch ZIP.

    Unknown Python objects are never instantiated or imported. The full file
    SHA256 must be checked separately before using the extracted values.
    """
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError("Official model_args archive has no unique data.pkl")
        operations = list(pickletools.genops(archive.read(members[0])))
    scalar = {"BINUNICODE": lambda value: value, "SHORT_BINUNICODE": lambda value: value,
              "BININT": lambda value: value, "BININT1": lambda value: value,
              "BININT2": lambda value: value, "BINFLOAT": lambda value: value,
              "NEWTRUE": lambda _value: True, "NEWFALSE": lambda _value: False,
              "NONE": lambda _value: None}
    found = {}
    for index, (op, value, _offset) in enumerate(operations):
        if op.name not in ("BINUNICODE", "SHORT_BINUNICODE") or value not in expected:
            continue
        following = index + 1
        while following < len(operations) and operations[following][0].name in ("BINPUT", "LONG_BINPUT", "MEMOIZE"):
            following += 1
        if following >= len(operations) or operations[following][0].name not in scalar:
            raise ValueError(f"Unsupported scalar opcode for official model_args field {value}")
        value_op, value_arg, _ = operations[following]
        found[value] = scalar[value_op.name](value_arg)
    if found != expected:
        raise ValueError(f"Official model_args static metadata mismatch: {found}")
    return found


def inspect(root: Path, data: dict, paths: dict | None = None) -> list[dict]:
    results = []
    for repo in data["repositories"]:
        for item in repo["files"]:
            path = destination(root, repo, item, paths)
            status = "missing"
            actual = None
            actual_sha256 = None
            if path.is_file():
                if path.stat().st_size != item["size"]:
                    status = "size_mismatch"
                else:
                    actual = digest_file(path, item["digest_type"])
                    status = "ok" if actual == item["digest"] else "hash_mismatch"
                    actual_sha256 = actual if item["digest_type"] == "sha256" else digest_file(path, "sha256")
                    if actual_sha256 != item["sha256"]:
                        status = "hash_mismatch"
            results.append({"repo_id": repo["repo_id"], "revision": repo["revision"],
                            "path": item["path"], "local_path": str(path),
                            "bytes": item["size"], "digest_type": item["digest_type"],
                            "expected_digest": item["digest"], "actual_digest": actual,
                            "expected_sha256": item["sha256"], "actual_sha256": actual_sha256,
                            "status": status})
    return results


def download_large_by_ranges(root: Path, repo: dict, item: dict, paths: dict | None = None) -> None:
    """Resume independent HTTP ranges and accept only the pinned full-file hash."""
    import requests

    path = destination(root, repo, item, paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = path.parent / f".{path.name}.parts"
    parts.mkdir(exist_ok=True)
    count = (item["size"] + RANGE_BYTES - 1) // RANGE_BYTES
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo['repo_id']}/resolve/{repo['revision']}/{item['path']}"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}

    def one(index: int) -> tuple[int, int]:
        start = index * RANGE_BYTES
        end = min(item["size"], start + RANGE_BYTES) - 1
        expected = end - start + 1
        final = parts / f"{index:05d}.part"
        if final.is_file() and final.stat().st_size == expected:
            return index, 0
        temporary = parts / f"{index:05d}.partial"
        session = requests.Session()
        session.trust_env = os.environ.get("CSGO_ASSET_BYPASS_PROXY") != "1"
        for attempt in range(5):
            try:
                have = temporary.stat().st_size if temporary.is_file() else 0
                if have > expected:
                    temporary.unlink()
                    have = 0
                if have == expected:
                    temporary.replace(final)
                    return index, 0
                requested_start = start + have
                with session.get(url, headers={**headers, "Range": f"bytes={requested_start}-{end}"},
                                 stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {requested_start}-{end}/{item['size']}":
                        raise RuntimeError("Server did not honor the requested exact byte range")
                    total = have
                    with temporary.open("ab") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                output.write(chunk)
                                total += len(chunk)
                    if total != expected:
                        raise RuntimeError("Incomplete byte range")
                temporary.replace(final)
                return index, expected
            except (requests.RequestException, OSError, RuntimeError):
                if attempt == 4:
                    raise RuntimeError(f"Byte range {index + 1}/{count} failed after retries") from None
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    workers = max(1, min(32, int(os.environ.get("CSGO_ASSET_DOWNLOAD_WORKERS", "8"))))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, i) for i in range(count)]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == count or completed % max(1, count // 10) == 0:
                print(f"  verified ranges {completed}/{count}", file=sys.stderr, flush=True)

    temporary = path.with_name(path.name + ".assembling")
    digest = hashlib.sha256() if item["digest_type"] == "sha256" else hashlib.sha1()
    if item["digest_type"] == "git-blob-sha1":
        digest.update(f"blob {item['size']}\0".encode("ascii"))
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:05d}.part").open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
    if temporary.stat().st_size != item["size"] or digest.hexdigest() != item["digest"]:
        temporary.unlink(missing_ok=True)
        for part in parts.glob("*.part"):
            part.unlink()
        for part in parts.glob("*.partial"):
            part.unlink()
        parts.rmdir()
        raise RuntimeError(f"Assembled asset failed pinned SHA/hash; discarded untrusted temporary ranges for retry: {path}")
    temporary.replace(path)
    for part in parts.glob("*.part"):
        part.unlink()
    parts.rmdir()


def download_missing(root: Path, data: dict, statuses: list[dict], cache_dir: Path | None,
                     paths: dict | None = None) -> None:
    broken = [s for s in statuses if s["status"] not in ("ok", "missing")]
    if broken:
        raise RuntimeError("Existing asset failed integrity; inspect it or choose a fresh --root. No file was overwritten.")
    missing = {(s["repo_id"], s["path"]) for s in statuses if s["status"] == "missing"}
    if not missing:
        return
    cache = hub_cache_root(cache_dir)
    # Regular HTTP supports resuming the Hub .incomplete file, including on
    # hosts where hf-xet's CAS route is unavailable through a proxy.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    hf_hub_download = None
    for repo in data["repositories"]:
        for item in repo["files"]:
            if (repo["repo_id"], item["path"]) not in missing:
                continue
            existing = destination(root, repo, item, paths)
            if existing.is_file() and existing.stat().st_size == item["size"] and digest_file(existing, "sha256") == item["sha256"]:
                continue
            if copy_verified(cached_snapshot(cache, repo, item), existing, item):
                print(f"Reused verified Hub cache: {repo['repo_id']}/{item['path']}", file=sys.stderr)
                continue
            print(f"Fetching {repo['repo_id']}@{repo['revision'][:12]}/{item['path']} ({item['size']:,} bytes)",
                  file=sys.stderr, flush=True)
            try:
                if item["size"] >= LARGE_FILE_BYTES:
                    download_large_by_ranges(root, repo, item, paths)
                else:
                    if hf_hub_download is None:
                        from huggingface_hub import hf_hub_download
                    snapshot_file = hf_hub_download(repo_id=repo["repo_id"], revision=repo["revision"],
                                                    filename=item["path"], cache_dir=str(cache),
                                                    token=os.environ.get("HF_TOKEN") or None)
                    if not copy_verified(Path(snapshot_file), existing, item):
                        raise RuntimeError(f"Hub cache file failed pinned hash: {item['path']}")
            except RuntimeError as exc:
                raise RuntimeError(f"Official asset transfer failed: {exc}") from None
            except Exception as exc:
                raise RuntimeError(
                    f"Official download failed for {repo['repo_id']}/{item['path']} "
                    f"({type(exc).__name__}); check network, HF_ENDPOINT and account access"
                ) from None
            path = destination(root, repo, item, paths)
            if not path.is_file() or path.stat().st_size != item["size"] or digest_file(path, item["digest_type"]) != item["digest"] or digest_file(path, "sha256") != item["sha256"]:
                raise RuntimeError(f"Downloaded asset failed pinned size/hash: {path}")


def print_env(root: Path, paths: dict | None = None) -> None:
    names = {
        "OFFICIAL_BASE_CHECKPOINT": Path(paths["base_checkpoint"]) if paths else root / "Lumina-Accessory/consolidated.00-of-01.pth",
        "GEMMA_PATH": Path(paths["gemma_path"]) if paths else root / "components/text_encoder",
        "TOKENIZER_PATH": Path(paths["tokenizer_path"]) if paths else root / "components/tokenizer",
        "VAE_PATH": Path(paths["vae_path"]) if paths else root / "components/vae",
    }
    for name, path in names.items():
        print(f"export {name}={shlex.quote(str(path))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline, read-only full hash check")
    mode.add_argument("--dry-run", action="store_true", help="List pinned files without reading or downloading")
    mode.add_argument("--print-env", action="store_true", help="Print shell exports for pinned local paths")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--root", type=Path, help="Override all four asset paths under this root")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache root")
    parser.add_argument("--machine-config", type=Path, help="Machine path config; defaults to CSGO_MACHINE_CONFIG or configs/machine.local.json")
    parser.add_argument("--base-checkpoint", type=Path, help="Override OFFICIAL_BASE_CHECKPOINT")
    parser.add_argument("--gemma-path", type=Path, help="Override GEMMA_PATH")
    parser.add_argument("--tokenizer-path", type=Path, help="Override TOKENIZER_PATH")
    parser.add_argument("--vae-path", type=Path, help="Override VAE_PATH")
    parser.add_argument("--components-only", action="store_true", help="Prepare only Gemma/tokenizer/VAE components")
    parser.add_argument("--base-only", action="store_true", help="Prepare only the official normal Accessory checkpoint and metadata")
    parser.add_argument("--small-only", action="store_true", help="Prepare only files below 128 MiB; useful while large files transfer")
    parser.add_argument("--json", action="store_true", help="Print machine-readable audit")
    args = parser.parse_args(argv)
    if args.experiment != EXPERIMENT:
        parser.error(f"Only {EXPERIMENT} is supported")
    if args.base_only and args.components_only:
        parser.error("--base-only and --components-only cannot be combined")
    root = (args.root or ROOT / "checkpoints").expanduser()
    root = (root if root.is_absolute() else ROOT / root).resolve()
    overrides = {}
    if args.root:
        overrides = {"base_checkpoint": str(root / "Lumina-Accessory/consolidated.00-of-01.pth"),
                     "gemma_path": str(root / "components/text_encoder"),
                     "tokenizer_path": str(root / "components/tokenizer"),
                     "vae_path": str(root / "components/vae")}
    overrides.update({name: str(value) for name, value in {
        "base_checkpoint": args.base_checkpoint, "gemma_path": args.gemma_path,
        "tokenizer_path": args.tokenizer_path, "vae_path": args.vae_path,
    }.items() if value is not None})
    machine_config = args.machine_config.expanduser() if args.machine_config else None
    if machine_config and not machine_config.is_absolute():
        machine_config = ROOT / machine_config
    paths = load_config(args.experiment, overrides=overrides, machine_config=machine_config)["paths"]
    data = manifest()
    if args.components_only or args.base_only or args.small_only:
        data = {**data, "repositories": [
            {**repo, "files": [item for item in repo["files"] if not args.small_only or item["size"] < LARGE_FILE_BYTES]}
            for repo in data["repositories"]
            if (not args.components_only or repo["target"] == "components")
            and (not args.base_only or repo["target"] == "Lumina-Accessory")
        ]}
    if args.print_env:
        print_env(root, paths)
        return 0
    if args.dry_run:
        statuses = [{"repo_id": r["repo_id"], "revision": r["revision"],
                     "path": f["path"], "local_path": str(destination(root, r, f, paths)),
                     "bytes": f["size"], "status": "not_checked"}
                    for r in data["repositories"] for f in r["files"]]
    else:
        statuses = inspect(root, data, paths)
        if not args.check:
            download_missing(root, data, statuses, args.cache_dir, paths)
            statuses = inspect(root, data, paths)
    model_args_static = None
    metadata_status = next((row for row in statuses if row["path"] == "model_args.pth" and row["status"] == "ok"), None)
    if metadata_status:
        model_args_static = inspect_model_args_without_unpickle(
            Path(metadata_status["local_path"]), data["static_model_args"]["verified_without_unpickle"])
    ready = not args.dry_run and all(s["status"] == "ok" for s in statuses)
    result = {"ready": ready, "root": str(root), "paths": {k: paths[k] for k in ("base_checkpoint", "gemma_path", "tokenizer_path", "vae_path")}, "normal_checkpoint_only": True,
              "model_args_static_verified": model_args_static,
              "total_bytes": sum(s["bytes"] for s in statuses), "files": statuses}
    if args.json or args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        for s in statuses:
            print(f"{s['status'].upper()}: {s['local_path']}")
        print(f"{len(statuses)} files, {result['total_bytes']:,} bytes, ready={ready}")
        if ready:
            print_env(root, paths)
    return 0 if args.dry_run or ready else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
