import unittest

from release_version import release_tag


class ReleaseVersionTests(unittest.TestCase):
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
