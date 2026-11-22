"""Build and compare a fixed original commit with the working tree using one harness."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a142ab81dd49e774075f98094409eb103747eebe")
    parser.add_argument("--output", type=Path, default=Path("target/capacity-comparison-results.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    baseline = root / 'target/original-baseline'
    baseline.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', args.baseline], text=True).strip()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'src'], text=True)
    for name in names.splitlines():
        path = baseline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(['git', 'show', f'{commit}:{name}']))
    for name in ['Cargo.toml', 'Cargo.lock', 'examples/original_comparison.rs']:
        destination = baseline / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    # Leave the working-tree build in the shared target directory.
    for version, manifest in [('original', baseline / 'Cargo.toml'), ('current', root / 'Cargo.toml')]:
        # These identical package names share artifacts on this Cargo version.
        # Clean only Serenade, retaining dependency builds, before switching sources.
        subprocess.run(['cargo', 'clean', '--release', '-p', 'serenade', '--manifest-path', str(manifest), '--target-dir', str(root / 'target')], check=True)
        with (root / f'target/comparison-{version}-build.log').open('w') as log:
            subprocess.run(['cargo', 'build', '--release', '--offline', '--manifest-path', str(manifest), '--target-dir', str(root / 'target'), '--example', 'original_comparison'], stdout=log, stderr=log, check=True)
        dependencies = (root / 'target/release/examples/original_comparison.d').read_text()
        expected_source = str(manifest.parent / 'src/vmisknn/mod.rs')
        if expected_source not in dependencies:
            raise RuntimeError(f'Unexpected build source for {version}: {dependencies}')
        shutil.copyfile(root / 'target/release/examples/original_comparison.exe', root / f'target/comparison-{version}.exe')
    env = os.environ.copy()
    env['BENCH_TRAINING_DATA'] = str(root / 'target/scaled-sample/train.tsv')
    env['BENCH_QUERIES'] = str(root / 'target/scaled-sample/queries.tsv')
    results = []
    for repeat in range(3):
        for version in (['original', 'current'] if repeat % 2 == 0 else ['current', 'original']):
            process = subprocess.run([str(root / f'target/comparison-{version}.exe')], env=env, capture_output=True, text=True, check=True)
            (root / f'target/comparison-{version}-{repeat}.log').write_text(process.stdout + '\nSTDERR\n' + process.stderr)
            lines = process.stdout.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith('history,concurrency,'))
            results.extend(dict(version=version, repeat=repeat, **row) for row in csv.DictReader(lines[start:]))
            print(version, repeat, 'complete', flush=True)
    report = dict(baseline_commit=commit, warmup_seconds=1, measurement_seconds=3, repeats=3, training_manifest='scaled-sample-manifest.json', current_mode='default inline predict, fresh scratch', dependency_note='Same current Cargo.toml and Cargo.lock for both; original src from the fixed baseline commit. No original prediction or index source changes.', results=results)
    (root / args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
