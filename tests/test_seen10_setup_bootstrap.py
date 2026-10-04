"""CPU-only checks for setup's recovery of a venv without pip."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]
FAKE_PIP = '''import sys
if "--version" in sys.argv:
    print("pip 25.3 (test stub)")
    raise SystemExit(0)
raise SystemExit(71)
'''


class SetupBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(SOURCE / "scripts/setup_csgo_seen10.sh", self.root / "scripts/setup_csgo_seen10.sh")
        shutil.copyfile(SOURCE / "requirements-csgo-seen10-cu128.lock.txt",
                        self.root / "requirements-csgo-seen10-cu128.lock.txt")
        self.python = next((path for name in ("python3.12", "python3.11")
                            if (path := shutil.which(name))), None)
        if self.python is None:
            self.skipTest("Python 3.11/3.12 is unavailable")
        subprocess.run([self.python, "-m", "venv", "--without-pip", str(self.root / ".venv")], check=True)
        self.venv = self.root / ".venv"
        self.bin = self.root / "fake-bin"
        self.bin.mkdir()
        smi = self.bin / "nvidia-smi"
        smi.write_text('#!/bin/sh\ncase "$1" in --query-gpu=compute_cap) echo 12.0;; *) echo "CUDA Version: 13.0";; esac\n')
        smi.chmod(0o755)
        self.bootstrap = self.bin / "bootstrap-python"
        self.bootstrap.write_text(f'''#!{sys.executable}
import pathlib, subprocess, sys
args = sys.argv[1:]
if args[:1] == ["-c"]:
    raise SystemExit(0)
if args[:3] == ["-m", "pip", "--python"] and args[4:] == ["install", "pip==25.3"]:
    target = pathlib.Path(args[3])
    site = subprocess.check_output([str(target / "bin/python"), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
    (pathlib.Path(site) / "pip.py").write_text({FAKE_PIP!r})
    (target / "bootstrap.marker").write_text("target only")
    raise SystemExit(0)
raise SystemExit(72)
''')
        self.bootstrap.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env["PATH"]
        self.env["CSGO_PYTHON_BIN"] = self.python
        self.env["CSGO_BOOTSTRAP_PYTHON"] = str(self.bootstrap)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def run_setup(self):
        return subprocess.run(["bash", "scripts/setup_csgo_seen10.sh", "--env-only"],
                              cwd=self.root, env=self.env, text=True, capture_output=True)

    def install_fake_pip(self):
        site = subprocess.check_output([str(self.venv / "bin/python"), "-c",
                                        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
        (Path(site) / "pip.py").write_text(FAKE_PIP)

    def test_incomplete_venv_bootstraps_pip_into_target(self):
        before = subprocess.run([str(self.venv / "bin/python"), "-m", "pip", "--version"], capture_output=True)
        self.assertNotEqual(before.returncode, 0)
        result = self.run_setup()
        self.assertEqual(result.returncode, 71, result.stderr)
        self.assertEqual((self.venv / "bootstrap.marker").read_text(), "target only")
        self.assertEqual(subprocess.run([str(self.venv / "bin/python"), "-m", "pip", "--version"],
                                        capture_output=True).returncode, 0)
        self.assertFalse((self.root / "bootstrap.marker").exists())

    def test_existing_pip_does_not_bootstrap(self):
        self.install_fake_pip()
        result = self.run_setup()
        self.assertEqual(result.returncode, 71, result.stderr)
        self.assertFalse((self.venv / "bootstrap.marker").exists())

    def test_missing_bootstrap_tool_fails_with_repair_instruction(self):
        self.env["CSGO_BOOTSTRAP_PYTHON"] = str(self.bin / "missing-python")
        result = self.run_setup()
        self.assertEqual(result.returncode, 1)
        self.assertIn("CSGO_BOOTSTRAP_PYTHON", result.stderr)
        self.assertIn("pip >=22.3", result.stderr)
        self.assertFalse((self.venv / "bootstrap.marker").exists())


if __name__ == "__main__":
    unittest.main()
