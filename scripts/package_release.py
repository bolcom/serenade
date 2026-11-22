"""Package only verified native applications. Uses Python's standard library."""
import argparse
import hashlib
import json
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
ROOT = Path(__file__).resolve().parents[1]


def smoke(root):
    suffix = ".exe" if os.name == "nt" else ""
    def run(name, *args):
        subprocess.run([str(root / (name + suffix)), *args], cwd=root,
                       check=True, timeout=120)
    config = "assets/example/workflows.toml"
    run("evaluator", config, "vmis", "evaluation.json")
    result = json.loads((root / "evaluation.json").read_text())["results"][0]
    assert result["predictions"] > 0 and result["mrr_at_20"] > 0, result
    run("tpe_hyperparameter_optm", config, "vmis", "hpo")
    best = json.loads((root / "hpo/vmis.json").read_text())
    assert len(best["trials"]) == 3, best
    generated = root / "hpo/vmis-serving.toml"
    assert generated.is_file()
    # Reload exactly the generated model in evaluator as well as serving.
    (root / "reload.toml").write_text(generated.read_text() +
        '\n[hyperparam]\ntest_data_path = "assets/example/valid.txt"\n')
    run("evaluator", "reload.toml", "vmis", "reload.json")
    reloaded = json.loads((root / "reload.json").read_text())["results"][0]
    assert reloaded["mrr_at_20"] == best["best_validation_mrr_at_20"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    generated.write_text(generated.read_text().replace('host = "0.0.0.0"',
        'host = "127.0.0.1"').replace("port = 8081", f"port = {port}"))
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen([str(root / ("serving" + suffix)), str(generated)],
                                   cwd=root, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError((root / "server.log").read_text())
                try:
                    url = f"http://127.0.0.1:{port}/v1/recommend?item_id=13598&session_id=release-smoke&user_consent=true"
                    with urllib.request.urlopen(url, timeout=1) as response:
                        items = json.load(response)
                    assert isinstance(items, list) and items
                    assert all(isinstance(item, int) for item in items)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError((root / "server.log").read_text())
                    time.sleep(0.1)
        finally:
            process.terminate()
            process.wait(timeout=15)
    print("Extracted package: evaluation, three TPE trials, config reload, serving and recommendation passed")


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
        for binary in BINS:
            source = ROOT / "target" / args.target / "release" / (binary + suffix)
            assert source.is_file(), source
            shutil.copy2(source, staging / source.name)
        data = staging / "assets/example"
        data.mkdir(parents=True)
        shutil.copy2(ROOT / "assets/example/workflows.toml", data)
        with zipfile.ZipFile(ROOT / "assets/example/example.zip") as example:
            for filename in ("train.txt", "test.txt", "valid.txt"):
                (data / filename).write_bytes(example.read(filename))
        shutil.copy2(ROOT / "docs/BINARIES.md", staging / "README.md")
        shutil.copy2(ROOT / "LICENSE.md", staging / "LICENSE.md")
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
        (dist / (archive.name + ".sha256")).write_text(f"{checksum}  {archive.name}\n")
        print(f"Verified {archive.name}: {sorted(actual)}")


if __name__ == "__main__":
    main()
