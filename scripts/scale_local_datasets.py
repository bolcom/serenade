"""Expand sessions and item catalogs while preserving empirical session patterns."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_sessions(path):
    sessions = defaultdict(list)
    with path.open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        for row in reader:
            sessions[int(row["SessionId"])].append((int(row["ItemId"]), row["Time"]))
    if not sessions:
        raise ValueError(f"No sessions in {path}")
    return list(sessions.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=100)
    parser.add_argument("--catalog-factor", type=int, default=1)
    args = parser.parse_args()
    if args.catalog_factor <= 0 or args.factor % args.catalog_factor != 0:
        parser.error("catalog-factor must be positive and divide factor")
    if args.factor <= 0:
        parser.error("factor must be positive")
    if args.input_dir.resolve() == args.output.resolve():
        parser.error("output must differ from input directory")
    splits = {name: read_sessions(args.input_dir / f"{name}.txt")
              for name in ("train", "valid", "test")}
    item_stride = 1 + max(item for sessions in splits.values()
                          for events in sessions for item, _ in events)
    if item_stride * args.catalog_factor - 1 > 2**64 - 1:
        parser.error("expanded item IDs would exceed u64")
    total_sessions = sum(map(len, splits.values())) * args.factor
    if total_sessions > 2**32:
        parser.error("session IDs would exceed u32")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(args.input_dir.resolve()), "factor": args.factor, "catalog_factor": args.catalog_factor,
              "method": "Whole-session replication with globally unique session IDs and disjoint copies of the item catalog. Remapping is consistent across train/valid/test. Event order and timestamps unchanged. Catalog copies repeat the empirical popularity shape; no new session patterns.",
              "splits": {}}
    next_id = 0
    for name, sessions in splits.items():
        popularity = Counter(item for events in sessions for item, _ in events)
        lengths = Counter(map(len, sessions))
        path = args.output / f"{name}.tsv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["SessionId", "ItemId", "Time"])
            for copy in range(args.factor):
                item_offset = (copy % args.catalog_factor) * item_stride
                for events in sessions:
                    writer.writerows((next_id, item + item_offset, timestamp) for item, timestamp in events)
                    next_id += 1
        # Independently scan the written output to verify both exact marginals.
        actual_popularity = Counter()
        actual_lengths = Counter()
        current_id = None
        length = 0
        with path.open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                session_id = int(row["SessionId"])
                if session_id != current_id:
                    if current_id is not None:
                        actual_lengths[length] += 1
                    current_id, length = session_id, 0
                length += 1
                actual_popularity[int(row["ItemId"])] += 1
        actual_lengths[length] += 1
        assert actual_popularity == Counter({item + shard * item_stride: count * (args.factor // args.catalog_factor)
                                            for shard in range(args.catalog_factor)
                                            for item, count in popularity.items()})
        assert actual_lengths == Counter({length: count * args.factor for length, count in lengths.items()})
        report["splits"][name] = {
            "source_sessions": len(sessions), "sessions": len(sessions) * args.factor,
            "source_interactions": sum(popularity.values()),
            "interactions": sum(actual_popularity.values()), "source_distinct_items": len(popularity), "distinct_items": len(actual_popularity),
            "session_length_counts": dict(sorted(actual_lengths.items())),
            "bytes": path.stat().st_size, "verified_exact_marginals": True}
        print(f"Verified {path}: {sum(actual_lengths.values()):,} sessions, {sum(actual_popularity.values()):,} interactions", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
