"""Check median prediction performance against the original-code benchmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--throughput-tolerance', type=float, default=5.0, help='Allowed throughput regression in percent (default: 5).')
    parser.add_argument('--latency-tolerance', type=float, default=10.0, help='Allowed p95/p99 regression in percent (default: 10).')
    args = parser.parse_args()
    if not (0 <= args.throughput_tolerance < 100 and args.latency_tolerance >= 0):
        parser.error('tolerances must be nonnegative; throughput tolerance must be below 100')
    report = json.loads(args.results.read_text())
    groups = defaultdict(lambda: defaultdict(list))
    for row in report['results']:
        groups[(int(row['history']), int(row['concurrency']))][row['version']].append(row)
    failed = False
    if not groups:
        raise ValueError('empty benchmark results')
    for (history, clients), versions in sorted(groups.items()):
        values = {}
        for version in ['original', 'current']:
            rows = versions[version]
            if len(rows) != report['repeats'] or len({int(row['repeat']) for row in rows}) != report['repeats']:
                raise ValueError(f'missing/duplicate runs for {version}, history={history}, clients={clients}')
            values[version] = {key: median(float(row[key]) for row in rows) for key in ['successful_per_second', 'p95_us', 'p99_us']}
            if any(value <= 0 for value in values[version].values()):
                raise ValueError('benchmark metrics must be positive')
        changes = {key: 100 * (values['current'][key] / values['original'][key] - 1) for key in values['original']}
        passed = changes['successful_per_second'] >= -args.throughput_tolerance and all(changes[key] <= args.latency_tolerance for key in ['p95_us', 'p99_us'])
        failed |= not passed
        print(f"{'PASS' if passed else 'FAIL'} history={history}, clients={clients}: throughput {changes['successful_per_second']:+.1f}%, p95 {changes['p95_us']:+.1f}%, p99 {changes['p99_us']:+.1f}%")
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
