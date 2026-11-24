"""Regression checks for release packaging without compiling native binaries."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import package_release


class PackageReleaseTest(unittest.TestCase):
    def test_windows_checksum_is_portable_to_linux(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = "x86_64-pc-windows-msvc"
            binaries = root / "target" / target / "release"
            binaries.mkdir(parents=True)
            for name in package_release.BINS:
                (binaries / (name + ".exe")).write_bytes(b"test binary")
            examples = root / "assets/example"
            examples.mkdir(parents=True)
            with zipfile.ZipFile(examples / "example.zip", "w") as archive:
                for name in ("hpo.toml", "train.txt", "test.txt", "valid.txt"):
                    archive.writestr(name, "fixture\n")
            (root / "docs").mkdir()
            for name in ("BINARIES.md", "CONFIG.md", "Preparation.md", "session-knn-validation.md"):
                (root / "docs" / name).write_text("fixture\n")
            (root / "LICENSE.md").write_text("fixture\n")
            (root / "docs/third-party").mkdir()
            (root / "docs/third-party/wsknn-LICENSE.md").write_text("fixture\n")
            (root / "docs/validation").mkdir()
            (root / "docs/validation/python-metrics.json").write_text("{}\n")
            argv = ["package_release.py", "--target", target,
                    "--platform", "windows-x86_64", "--version", "v0.3.2"]
            with patch.object(package_release, "ROOT", root), \
                    patch.object(package_release, "smoke"), \
                    patch("sys.argv", argv):
                package_release.main()
            archive = root / "dist/serenade-v0.3.2-windows-x86_64.zip"
            checksum = archive.with_name(archive.name + ".sha256").read_bytes()
            expected = f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
            self.assertEqual(checksum, expected.encode("ascii"))


if __name__ == "__main__":
    unittest.main()
