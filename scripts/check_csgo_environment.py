#!/usr/bin/env python3
"""CPU-side dependency/import audit; no model weights or CUDA kernels loaded."""
from __future__ import annotations

import importlib
from importlib import metadata
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

PROJECT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torchvision", "flash-attn", "transformers", "diffusers",
            "accelerate", "fairscale", "torchdiffeq", "prodigyopt",
            "huggingface-hub", "safetensors", "sentencepiece")
MODULES = ("torch", "torchvision", "flash_attn", "transformers", "diffusers",
           "accelerate", "fairscale", "torchdiffeq", "prodigyopt",
           "huggingface_hub", "safetensors", "sentencepiece")
HEADER_ROOT = PROJECT / ".venv/python_headers"
PROVENANCE = HEADER_ROOT / "provenance.json"
HEADER_STARTUP_MARKER = "# CSGO_SEEN10_LOCAL_PYTHON_HEADERS_V2"
OLD_SITE_CUSTOMIZE_MARKER = "# CSGO_SEEN10_LOCAL_PYTHON_HEADERS_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiler_probe(extra_includes: list[Path] | None = None) -> tuple[bool, str]:
    env = os.environ.copy()
    if extra_includes:
        prefix = ":".join(str(path) for path in extra_includes)
        env["CPATH"] = prefix + (":" + env["CPATH"] if env.get("CPATH") else "")
    source = (f"#include <Python.h>\n#if PY_MAJOR_VERSION != {sys.version_info.major} "
              f"|| PY_MINOR_VERSION != {sys.version_info.minor}\n#error Wrong Python headers\n#endif\n"
              "int main(void) { return 0; }\n")
    process = subprocess.run(["cc", "-fsyntax-only", "-x", "c", "-"], input=source,
                             text=True, capture_output=True, env=env)
    return process.returncode == 0, process.stderr.strip()


def write_header_startup() -> Path:
    site_packages = PROJECT / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    module = site_packages / "csgo_seen10_python_headers_startup.py"
    hook = site_packages / "csgo_seen10_python_headers.pth"
    code = HEADER_STARTUP_MARKER + "\n" + '''from pathlib import Path
import json
import os
import sys

_provenance = Path(sys.prefix) / "python_headers/provenance.json"
if _provenance.is_file():
    _source = json.loads(_provenance.read_text(encoding="utf-8"))
    _include = Path(_source["include"])
    _paths = [_include, Path(_source.get("multiarch_include_root", _include.parent))]
    if (_include / "Python.h").is_file():
        _existing = [p for p in os.environ.get("CPATH", "").split(":") if p]
        os.environ["CPATH"] = ":".join([str(p) for p in _paths if str(p) not in _existing] + _existing)
'''
    site_packages.mkdir(parents=True, exist_ok=True)
    for target in (module, hook):
        if target.is_file() and not target.read_text(encoding="utf-8").startswith(HEADER_STARTUP_MARKER):
            raise RuntimeError(f"Refusing to overwrite existing environment hook: {target}")
    module.write_text(code, encoding="utf-8")
    hook.write_text(HEADER_STARTUP_MARKER + "\nimport csgo_seen10_python_headers_startup\n", encoding="utf-8")
    old_hook = site_packages / "sitecustomize.py"
    if old_hook.is_file() and old_hook.read_text(encoding="utf-8").startswith(OLD_SITE_CUSTOMIZE_MARKER):
        old_hook.unlink()
    return hook


def apt_package_metadata(package: str, version: str) -> dict:
    output = subprocess.check_output(["apt-cache", "show", package], text=True)
    for stanza in output.split("\n\n"):
        fields = {}
        for line in stanza.splitlines():
            if ": " in line and not line.startswith(" "):
                key, value = line.split(": ", 1)
                fields[key] = value
        if fields.get("Version") == version and fields.get("SHA256") and fields.get("Filename"):
            return fields
    raise RuntimeError(f"No signed apt package index entry for {package}={version}")


def prepare_headers() -> dict:
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError("Prepare headers with this project's .venv/bin/python")
    version_short = f"{sys.version_info.major}.{sys.version_info.minor}"
    local_include_root = HEADER_ROOT / "root/usr/include"
    local_include = local_include_root / f"python{version_short}"
    if PROVENANCE.is_file() and (local_include / "Python.h").is_file():
        evidence = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        ok, error = compiler_probe([local_include, local_include_root])
        if not ok:
            raise RuntimeError(f"Existing local Python headers fail compiler probe: {error}")
        write_header_startup()
        return evidence
    system_include = Path(sysconfig.get_config_var("INCLUDEPY"))
    if (system_include / "Python.h").is_file():
        ok, error = compiler_probe([system_include, system_include.parent])
        if not ok:
            raise RuntimeError(f"Installed Python development headers fail compiler probe: {error}")
        evidence = {"source": "system", "include": str(system_include),
                    "multiarch_include_root": str(system_include.parent), "python": sys.version.split()[0]}
        PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        write_header_startup()
        return evidence

    custom = os.environ.get("CSGO_PYTHON_HEADERS")
    if custom:
        candidate = Path(custom).expanduser().resolve()
        include = candidate if (candidate / "Python.h").is_file() else candidate / f"python{version_short}"
        if not (include / "Python.h").is_file():
            raise RuntimeError(f"CSGO_PYTHON_HEADERS has no Python.h for {version_short}: {candidate}")
        ok, error = compiler_probe([include, include.parent])
        if not ok:
            raise RuntimeError(f"CSGO_PYTHON_HEADERS fail compiler probe: {error}")
        evidence = {"source": "user-provided", "include": str(include),
                    "multiarch_include_root": str(include.parent), "python": sys.version.split()[0]}
        PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        write_header_startup()
        return evidence

    release = {}
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        for line in os_release.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                release[key] = value.strip('"')
    if release.get("ID") != "ubuntu":
        raise RuntimeError("Python.h is absent. Install matching Python development headers or set CSGO_PYTHON_HEADERS to an independently obtained matching include tree.")
    package = f"libpython{version_short}-dev"
    interpreter_package = f"python{version_short}"
    installed_version = subprocess.check_output(["dpkg-query", "-W", "-f=${Version}", interpreter_package], text=True).strip()
    fields = apt_package_metadata(package, installed_version)
    packages = HEADER_ROOT / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    archive = packages / Path(fields["Filename"]).name
    if not archive.is_file() or sha256_file(archive) != fields["SHA256"]:
        archive.unlink(missing_ok=True)
        subprocess.run(["apt", "download", f"{package}={installed_version}"], cwd=packages, check=True,
                       stdout=subprocess.DEVNULL)
    if sha256_file(archive) != fields["SHA256"] or archive.stat().st_size != int(fields["Size"]):
        raise RuntimeError(f"Downloaded {package} does not match signed apt size/SHA256")
    extracted = HEADER_ROOT / "root"
    if extracted.is_dir():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    subprocess.run(["dpkg-deb", "-x", str(archive), str(extracted)], check=True)
    ok, error = compiler_probe([local_include, local_include_root])
    if not ok:
        raise RuntimeError(f"Extracted {package} headers fail compiler probe: {error}")
    evidence = {"source": "ubuntu-signed-apt-index", "distribution": release.get("VERSION_ID"),
                "python": sys.version.split()[0], "installed_python_package": installed_version,
                "package": package, "package_version": installed_version,
                "package_archive": str(archive), "package_sha256": fields["SHA256"],
                "package_size": int(fields["Size"]), "apt_filename": fields["Filename"],
                "python_h_sha256": sha256_file(local_include / "Python.h"),
                "pyconfig_h_sha256": sha256_file(local_include / "pyconfig.h"),
                "include": str(local_include), "multiarch_include_root": str(local_include_root)}
    PROVENANCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    write_header_startup()
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-headers", action="store_true", help="Install exact matching Python headers into this .venv only")
    args = parser.parse_args()
    if args.prepare_headers:
        print(json.dumps(prepare_headers(), indent=2, sort_keys=True))
        return 0
    versions = {name: metadata.version(name) for name in PACKAGES}
    for name in MODULES:
        importlib.import_module(name)
    import torch
    profile_path = PROJECT / ".venv/csgo_environment_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
    expected_cuda = os.environ.get("CSGO_EXPECTED_TORCH_CUDA") or profile.get("expected_torch_cuda")
    header_ok, header_error = compiler_probe()
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "packages": versions, "torch_cuda_build": torch.version.cuda,
              "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
              "torch_compiled_arches": torch.cuda.get_arch_list(),
              "profile": profile,
              "python_headers": json.loads(PROVENANCE.read_text(encoding="utf-8")) if PROVENANCE.is_file() else None,
              "python_header_compiler_probe": {"passed": header_ok, "error": header_error},
              "cuda_operation_tested": False,
              "project": str(PROJECT)}
    print(json.dumps(report, indent=2, sort_keys=True))
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError("Expected project-local independent .venv")
    if expected_cuda and torch.version.cuda != expected_cuda:
        raise RuntimeError(f"Expected CUDA {expected_cuda} build, found {torch.version.cuda}")
    if not header_ok:
        raise RuntimeError("Python development headers are unavailable to C compiler; rerun setup --env-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
