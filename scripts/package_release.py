"""Package only verified native applications. Uses Python's standard library."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

BINS = ("serving", "evaluator", "tpe_hyperparameter_optm")
LICENSES = ("LICENSE.md", "docs/third-party/wsknn-LICENSE.md")
ROOT = Path(__file__).resolve().parents[1]


def verify_release_binaries(directory, suffix):
    actual = {p.name for p in directory.iterdir() if p.is_file() and
              (p.suffix == ".exe" if suffix else os.access(p, os.X_OK))}
    expected = {name + suffix for name in BINS}
    assert actual == expected, f"Expected exactly {sorted(expected)}, found {sorted(actual)}"


def smoke(root, iterations=3):
    suffix = ".exe" if os.name == "nt" else ""
    def run(name, *args):
        subprocess.run([str(root / (name + suffix)), *map(str, args)], cwd=root,
                       check=True, timeout=120)
    workspace = root / "workspace"
    config = workspace / "hpo.toml"
    config.write_text(config.read_text().replace("num_trials = 5", f"num_trials = {iterations}"))
    run("tpe_hyperparameter_optm", "--config", config)
    models = sorted(workspace.glob("best-*.toml"))
    assert len(models) == 7
    options = [arg for model in models for arg in ("--model", model)]
    run("evaluator", *options)
    run("evaluator", *options, "--benchmark", "--iterations", "2", "--workers", "2",
        "--snapshot-dir", root / "snapshots")
    for filename in ("evaluation.csv", "benchmark.csv"):
        with (workspace / filename).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 7 and all(int(row["predictions"]) > 0 for row in rows)
    snapshot = root / "standalone.vmis"
    run("serving", "--model", models[0], "--config", workspace / "serenade.toml",
        "--build-index", snapshot, "--training-statistics", root / "training-stats.bin")
    assert snapshot.is_file() and (root / "training-stats.bin").is_file()
    predictions = root / "predictions.txt"
    predictions.write_text("13598,;13598\n")
    run("evaluator", "--train", workspace / "train.txt", "--predictions", predictions)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    deployment = root / "deployment"
    deployment.mkdir()
    config = deployment / "serenade.toml"
    config.write_text((workspace / "serenade.toml").read_text().replace("port = 8081", f"port = {port}"))
    launches = [(model, []) for model in models]
    launches.append((models[0], ["--index", str(snapshot)]))
    for model, overrides in launches:
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen([str(root / ("serving" + suffix)), "--model", str(model), "--config", str(config), *overrides],
                cwd=root, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError((root / "server.log").read_text())
                    try:
                        url = f"http://127.0.0.1:{port}/v1/recommend?item_id=13598&session_id=smoke&user_consent=true"
                        with urllib.request.urlopen(url, timeout=1) as response:
                            items = json.load(response)
                        assert isinstance(items, list) and items
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError((root / "server.log").read_text())
                        time.sleep(0.1)
            finally:
                process.terminate()
                process.wait(timeout=15)
    print("Extracted workspace: HPO, seven portable models, quality, benchmarking, snapshots and serving passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    name = f"serenade-{args.version}-{args.platform}"
    assert all(c.isalnum() or c in "-._" for c in name)
    suffix = ".exe" if args.platform.startswith("windows-") else ""
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="serenade-package-") as temporary:
        staging = Path(temporary) / name
        staging.mkdir()
        verify_release_binaries(ROOT / "target" / args.target / "release", suffix)
        for binary in BINS:
            source = ROOT / "target" / args.target / "release" / (binary + suffix)
            assert source.is_file(), source
            shutil.copy2(source, staging / source.name)
        workspace = staging / "workspace"
        workspace.mkdir()
        with zipfile.ZipFile(ROOT / "assets/example/example.zip") as example:
            for filename in ("hpo.toml", "serenade.toml", "train.txt", "validation.txt", "test.txt"):
                (workspace / filename).write_bytes(example.read(filename))
        shutil.copy2(ROOT / "docs/BINARIES.md", staging / "README.md")
        for guide in ("CONFIG.md", "Preparation.md"):
            (staging / guide).write_text(
                (ROOT / "docs" / guide).read_text(encoding="utf-8-sig").replace("(BINARIES.md)", "(README.md)"),
                encoding="utf-8",
            )
        for license_path in LICENSES:
            destination = staging / license_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / license_path, destination)
        expected = {p.relative_to(staging).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in staging.rglob("*") if p.is_file()}
        archive = dist / (name + (".zip" if suffix else ".tar.gz"))
        if suffix:
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                for path in sorted(staging.rglob("*")):
                    if path.is_file():
                        output.write(path, path.relative_to(staging.parent))
        else:
            with tarfile.open(archive, "w:gz") as output:
                output.add(staging, arcname=name)
        extracted = Path(temporary) / "extracted"
        if suffix:
            with zipfile.ZipFile(archive) as output:
                output.extractall(extracted)
        else:
            with tarfile.open(archive) as output:
                output.extractall(extracted, filter="data")
        package = extracted / name
        actual = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in package.rglob("*") if p.is_file()}
        assert actual == expected, "Archive contents differ from package manifest"
        smoke(package)
        checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
        # Linux sha256sum treats a CR from Windows text mode as part of the filename.
        (dist / (archive.name + ".sha256")).write_bytes(
            f"{checksum}  {archive.name}\n".encode("ascii")
        )
        print(f"Verified {archive.name}: {sorted(actual)}")


if __name__ == "__main__":
    main()
