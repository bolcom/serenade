"""Regression checks for release packaging without compiling native binaries."""
import hashlib
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import zipfile

import package_release


class PackageReleaseTest(unittest.TestCase):
    def test_release_archives_preserve_licenses_and_portable_checksums(self):
        for platform, target in (
            ("windows-x86_64", "x86_64-pc-windows-msvc"),
            ("linux-x86_64", "x86_64-unknown-linux-gnu"),
            ("macos-x86_64", "x86_64-apple-darwin"),
            ("macos-aarch64", "aarch64-apple-darwin"),
        ):
            with self.subTest(platform=platform):
                self.check_archive(platform, target)

    def check_archive(self, platform, target):
        licenses = {name: (package_release.ROOT / name).read_bytes()
                    for name in ("LICENSE.md", "docs/third-party/wsknn-LICENSE.md")}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suffix = ".exe" if platform.startswith("windows-") else ""
            binaries = root / "target" / target / "release"
            binaries.mkdir(parents=True)
            for name in package_release.BINS:
                (binaries / (name + suffix)).write_bytes(b"test binary")
            examples = root / "assets/example"
            examples.mkdir(parents=True)
            with zipfile.ZipFile(examples / "example.zip", "w") as archive:
                for name in ("hpo.toml", "train.txt", "test.txt", "valid.txt"):
                    archive.writestr(name, "fixture\n")
            (root / "docs").mkdir()
            for name in ("BINARIES.md", "CONFIG.md", "Preparation.md"):
                (root / "docs" / name).write_text("fixture\n")
            for name, content in licenses.items():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            argv = ["package_release.py", "--target", target,
                    "--platform", platform, "--version", "v0.3.2"]
            with patch.object(package_release, "ROOT", root), \
                    patch.object(package_release, "smoke"), \
                    patch("sys.argv", argv):
                package_release.main()
            package_name = f"serenade-v0.3.2-{platform}"
            extension = ".zip" if suffix else ".tar.gz"
            archive = root / "dist" / (package_name + extension)
            if suffix:
                with zipfile.ZipFile(archive) as release:
                    names = release.namelist() if suffix else release.getnames()
                    self.assertEqual(sorted(Path(n).name for n in names if Path(n).name in package_release.BINS or n.endswith(".exe")), sorted(name + suffix for name in package_release.BINS))
                    self.assertFalse(any("knn_benchmark" in n for n in names))
                    for name, content in licenses.items():
                        self.assertEqual(release.read(f"{package_name}/{name}"), content)
            else:
                with tarfile.open(archive) as release:
                    names = release.namelist() if suffix else release.getnames()
                    self.assertEqual(sorted(Path(n).name for n in names if Path(n).name in package_release.BINS or n.endswith(".exe")), sorted(name + suffix for name in package_release.BINS))
                    self.assertFalse(any("knn_benchmark" in n for n in names))
                    for name, content in licenses.items():
                        with release.extractfile(f"{package_name}/{name}") as license_file:
                            self.assertEqual(license_file.read(), content)
            checksum = archive.with_name(archive.name + ".sha256").read_bytes()
            expected = f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
            self.assertEqual(checksum, expected.encode("ascii"))


if __name__ == "__main__":
    unittest.main()
