"""Scale the bundled session sample without flattening its empirical heavy tails."""
import argparse
import collections
import csv
import io
import json
import random
import zipfile
from pathlib import Path


def read_sessions(archive, member):
    sessions = collections.defaultdict(list)
    with archive.open(member) as stream:
        for row in csv.DictReader(io.TextIOWrapper(stream), delimiter="\t"):
            sessions[int(row["SessionId"])].append((int(row["ItemId"]), round(float(row["Time"]))))
    return [sorted(events, key=lambda event: event[1]) for events in sessions.values()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("assets/example/example.zip"))
    parser.add_argument("--output", type=Path, default=Path("target/scaled-sample"))
    parser.add_argument("--factor", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.factor <= 0:
        parser.error("--factor must be positive")
    with zipfile.ZipFile(args.archive) as archive:
        training = read_sessions(archive, "train.txt")
        heldout = read_sessions(archive, "test.txt") + read_sessions(archive, "valid.txt")
    if not training:
        parser.error("training data must contain sessions")
    max_duration = max(events[-1][1] - events[0][1] for events in training)
    if 1_600_000_000 + (len(training) * args.factor - 1) * 10 + max_duration > 2**32 - 1:
        parser.error("scaled timestamps would exceed the index's u32 timestamp limit")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    order = list(range(len(training)))
    session_id = 0
    with (args.output / "train.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["SessionId", "ItemId", "Time"])
        for _ in range(args.factor):
            rng.shuffle(order)
            for index in order:
                events = training[index]
                start = events[0][1]
                for item, timestamp in events:
                    writer.writerow([session_id, item, 1_600_000_000 + session_id * 10 + timestamp - start])
                session_id += 1
    # Every held-out prediction prefix, weighted as evaluation events rather than
    # giving each length equal probability. The final item remains unseen.
    queries = [events[:end] for events in heldout for end in range(1, len(events))]
    rng.shuffle(queries)
    with (args.output / "queries.tsv").open("w") as stream:
        for events in queries:
            stream.write("\t".join(str(item) for item, _ in events) + "\n")
    lengths = sorted(map(len, training))
    popularity = collections.Counter(item for events in training for item, _ in events)
    report = {
        "archive": str(args.archive), "seed": args.seed, "factor": args.factor,
        "source_sessions": len(training), "source_interactions": sum(lengths),
        "scaled_sessions": session_id, "scaled_interactions": sum(lengths) * args.factor,
        "catalog_items": len(popularity), "heldout_prediction_prefixes": len(queries),
        "session_length_percentiles": {str(p): lengths[int((len(lengths)-1)*p/100)] for p in [50,90,95,99,99.5,100]},
        "top_1_percent_item_interaction_share": sum(n for _,n in popularity.most_common(max(1,len(popularity)//100))) / sum(lengths),
        "method": "Exact whole-session replication, shuffled per copy; fixed observed catalog. Empirical marginals and co-occurrence preserved; no unseen tail extrapolated."
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
