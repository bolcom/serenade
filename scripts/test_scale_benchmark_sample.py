"""Verify scaled datasets retain session structure and empirical marginals."""
import collections
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import scale_benchmark_sample


class ScalingTest(unittest.TestCase):
    def test_replication_preserves_popularity_lengths_and_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("train.txt", "SessionId\tItemId\tTime\n1\t10\t100\n1\t20\t110\n2\t10\t200\n2\t10\t210\n2\t30\t220\n")
                stream.writestr("test.txt", "SessionId\tItemId\tTime\n3\t10\t1\n3\t20\t2\n3\t30\t3\n")
                stream.writestr("valid.txt", "SessionId\tItemId\tTime\n4\t30\t1\n4\t10\t2\n")
            output = root / "output"
            command = ["scale", "--archive", str(archive), "--output", str(output), "--factor", "3"]
            with patch.object(sys, "argv", command), contextlib.redirect_stdout(io.StringIO()):
                scale_benchmark_sample.main()
            first = (output / "train.tsv").read_bytes()
            with patch.object(sys, "argv", command), contextlib.redirect_stdout(io.StringIO()):
                scale_benchmark_sample.main()
            self.assertEqual(first, (output / "train.tsv").read_bytes())
            sessions = collections.defaultdict(list)
            with (output / "train.tsv").open() as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    sessions[row["SessionId"]].append((int(row["ItemId"]), int(row["Time"])))
            self.assertEqual(collections.Counter(map(len, sessions.values())), {2: 3, 3: 3})
            self.assertEqual(collections.Counter(item for events in sessions.values() for item, _ in events), {10: 9, 20: 3, 30: 3})
            self.assertEqual(collections.Counter(tuple(t-events[0][1] for _, t in events) for events in sessions.values()), {(0, 10): 3, (0, 10, 20): 3})
            self.assertCountEqual((output / "queries.tsv").read_text().splitlines(), ["10", "10\t20", "30"])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["scaled_interactions"], 15)
            self.assertEqual(manifest["catalog_items"], 3)


if __name__ == "__main__":
    unittest.main()
