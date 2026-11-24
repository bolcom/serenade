import unittest
from pathlib import Path
import tempfile

from release_version import bump_version, release_tag


class ReleaseVersionTests(unittest.TestCase):
    def fixture(self, root, version="0.3.5"):
        (root / "Cargo.toml").write_text(f'[package]\nname = "serenade"\nversion = "{version}"\n')
        (root / "Cargo.lock").write_text(f'[[package]]\nname = "serenade"\nversion = "{version}"\n\n[[package]]\nname = "other"\nversion = "0.3.5"\n')

    def test_patch_bump_updates_both_files_only_for_root_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            self.assertEqual(bump_version("0.3.5", root), "v0.3.6")
            self.assertIn('version = "0.3.6"', (root / "Cargo.toml").read_text())
            self.assertIn('name = "serenade"\nversion = "0.3.6"', (root / "Cargo.lock").read_text())
            self.assertIn('name = "other"\nversion = "0.3.5"', (root / "Cargo.lock").read_text())

    def test_preserves_explicit_increase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, "0.4.0")
            before = (root / "Cargo.toml").read_bytes()
            self.assertEqual(bump_version("0.3.5", root), "v0.4.0")
            self.assertEqual((root / "Cargo.toml").read_bytes(), before)

    def test_decrease_and_missing_lock_entry_leave_files_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            before = (root / "Cargo.toml").read_bytes()
            with self.assertRaises(ValueError):
                bump_version("0.3.6", root)
            (root / "Cargo.lock").write_text("")
            with self.assertRaises(ValueError):
                bump_version("0.3.5", root)
            self.assertEqual((root / "Cargo.toml").read_bytes(), before)

    def test_explicit_version_requires_matching_lockfile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, "0.4.0")
            lock = root / "Cargo.lock"
            lock.write_text(lock.read_text().replace('version = "0.4.0"', 'version = "0.3.5"'))
            with self.assertRaisesRegex(ValueError, "must match"):
                bump_version("0.3.5", root)

    def test_increases(self):
        for previous, current in [("0.3.3", "0.3.4"), ("0.3.9", "0.3.10"), ("0.9.9", "1.0.0")]:
            with self.subTest(current=current):
                self.assertEqual(release_tag(previous, current), f"v{current}")

    def test_no_increase(self):
        self.assertEqual(release_tag("0.3.4", "0.3.4"), "")
        self.assertEqual(release_tag("0.3.4", "0.3.3"), "")

    def test_invalid_version(self):
        for version in ["0.3", "0.3.4-rc1", "00.3.4"]:
            with self.assertRaises(ValueError):
                release_tag("0.3.3", version)
