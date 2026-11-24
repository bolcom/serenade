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


def smoke(root, iterations=3):
    suffix = ".exe" if os.name == "nt" else ""
    def run(name, *args):
        subprocess.run([str(root / (name + suffix)), *args], cwd=root,
                       check=True, timeout=120)
    algorithms = ["vmis", "sknn", "ssknn", "wsknn", "vsknn", "stan", "vsknn_stan"]
    config = "configs/hpo.toml"
    run("evaluator", config, "vmis", "evaluation.json")
    result = json.loads((root / "evaluation.json").read_text())["results"][0]
    assert result["predictions"] > 0 and result["mrr_at_20"] > 0, result
    import re
    legacy = re.sub(r"(?ms)^\[hyperparam\.algorithms\.[^\]]+\]\n.*?(?=^\[|\Z)", "", (root / config).read_text())
    legacy = legacy.replace('algorithm = "vmis"\n', '').replace('num_trials = 50', 'num_trials = 3')
    legacy = legacy.replace('output_dir = "hpo-results"\n', '')
    (root / "legacy.toml").write_text(legacy)
    run("evaluator", "legacy.toml")
    legacy_result = json.loads((root / "evaluation-results.json").read_text())["results"][0]
    for key in ["algorithm", "predictions", "mrr_at_20", "ndcg_at_20", "hitrate_at_20", "coverage_at_20"]:
        assert legacy_result[key] == result[key]
    run("tpe_hyperparameter_optm", "legacy.toml")
    assert [p.name for p in (root / "hpo-results").glob("*.toml")] == ["vmis.toml"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Set the runtime environment before HPO; both applications receive the exact export.
    text = (root / "configs/hpo.toml").read_text().replace("num_trials = 50", f"num_trials = {iterations}")
    text = text.replace('host = "0.0.0.0"', 'host = "127.0.0.1"').replace("port = 8081", f"port = {port}")
    text = text.replace('output_dir = "hpo-results"', 'output_dir = "hpo"')
    (root / "smoke.toml").write_text(text)
    run("tpe_hyperparameter_optm", "smoke.toml")
    assert sorted(p.stem for p in (root / "hpo").glob("*.toml")) == sorted(algorithms)
    run("evaluator", "hpo")
    for algorithm in algorithms:
        best = json.loads((root / f"hpo/{algorithm}.json").read_text())
        assert len(best["trials"]) == iterations, best
        generated = root / f"hpo/{algorithm}.toml"
        run("evaluator", str(generated))
        reloaded = json.loads((root / "evaluation-results.json").read_text())
        combined = json.loads((root / f"hpo/{algorithm}.evaluation.json").read_text())
        for key in ["mrr_at_20", "ndcg_at_20", "hitrate_at_20", "coverage_at_20", "predictions"]:
            assert reloaded["results"][0][key] == combined["results"][0][key]
        assert reloaded["model_configurations"][0]["knn"]["algorithm"].lower().replace("_", "") == algorithm.replace("_", "")
        model = reloaded["model_configurations"][0]
        for parameter, value in best["best_hyperparameters"].items():
            section = model if parameter in ["m_most_recent_sessions", "neighborhood_size_k", "max_items_in_session"] or (parameter == "idf_weighting" and algorithm == "vmis") else model["knn"]
            assert section[parameter] == value, (algorithm, parameter, section, value)
        # Independently reload a fresh index and reproduce the validation objective.
        validation = generated.read_text().replace('test_data_path = "assets/example/test.txt"',
                                                   'test_data_path = "assets/example/valid.txt"')
        (root / "validation.toml").write_text(validation)
        run("evaluator", "validation.toml", algorithm, "validation.json")
        assert json.loads((root / "validation.json").read_text())["results"][0]["mrr_at_20"] == best["best_validation_mrr_at_20"]
        # The unchanged generated file serves the same selected model.
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen([str(root / ("serving" + suffix)), str(generated)],
                cwd=root, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError((root / "server.log").read_text())
                    try:
                        url = f"http://127.0.0.1:{port}/v1/recommend?item_id=13598&session_id=smoke-{algorithm}&user_consent=true"
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
        run("tpe_hyperparameter_optm", "smoke.toml", algorithm, "repeat")
        assert best == json.loads((root / f"repeat/{algorithm}.json").read_text())
    disabled = text.replace("[hyperparam.algorithms.sknn]\nenabled = true", "[hyperparam.algorithms.sknn]\nenabled = false")
    disabled = disabled.replace("[hyperparam.algorithms.stan]\nenabled = true", "[hyperparam.algorithms.stan]\nenabled = false")
    (root / "disabled.toml").write_text(disabled.replace('output_dir = "hpo"', 'output_dir = "subset"'))
    run("tpe_hyperparameter_optm", "disabled.toml")
    assert sorted(p.stem for p in (root / "subset").glob("*.toml")) == sorted(a for a in algorithms if a not in ["sknn", "stan"])
    comparison = json.loads((root / "subset/comparison.json").read_text())
    assert [r["algorithm"] for r in comparison] == [a for a in algorithms if a not in ["sknn", "stan"]]
    for algorithm in [a for a in algorithms if a not in ["sknn", "stan"]]:
        assert json.loads((root / f"subset/{algorithm}.json").read_text()) == json.loads((root / f"hpo/{algorithm}.json").read_text())
    for name, content, override in [
        ("none", disabled.replace("enabled = true", "enabled = false"), []),
        ("disabled-selection", disabled, ["stan"]),
    ]:
        (root / f"{name}.toml").write_text(content)
        result = subprocess.run([str(root / ("tpe_hyperparameter_optm" + suffix)), f"{name}.toml", *override], cwd=root)
        assert result.returncode != 0
    # Test data is metadata during optimization and must never be opened or affect selection.
    (root / "holdout.toml").write_text(text.replace("assets/example/test.txt", "missing-test.txt"))
    run("tpe_hyperparameter_optm", "holdout.toml", "vmis", "holdout")
    original = json.loads((root / "hpo/vmis.json").read_text())
    heldout = json.loads((root / "holdout/vmis.json").read_text())
    assert original["trials"] == heldout["trials"]
    assert original["best_hyperparameters"] == heldout["best_hyperparameters"]
    print("Extracted package: seven algorithms, exact exports, evaluation, validation scores, serving, seeds, disabled algorithms and legacy config passed")


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
        (staging / "configs").mkdir()
        with zipfile.ZipFile(ROOT / "assets/example/example.zip") as example:
            (staging / "configs/hpo.toml").write_bytes(example.read("hpo.toml"))
            for filename in ("train.txt", "test.txt", "valid.txt"):
                (data / filename).write_bytes(example.read(filename))
        shutil.copy2(ROOT / "docs/BINARIES.md", staging / "README.md")
        for guide in ("CONFIG.md", "Preparation.md"):
            (staging / guide).write_text(
                (ROOT / "docs" / guide).read_text(encoding="utf-8-sig").replace("(BINARIES.md)", "(README.md)"),
                encoding="utf-8",
            )
        shutil.copy2(ROOT / "LICENSE.md", staging / "LICENSE.md")
        shutil.copytree(ROOT / "docs/third-party", staging / "docs/third-party")
        shutil.copytree(ROOT / "docs/validation", staging / "docs/validation")
        (staging / "session-knn-validation.md").write_text(
            (ROOT / "docs/session-knn-validation.md").read_text(encoding="utf-8")
            .replace("(third-party/", "(docs/third-party/")
            .replace("(validation/", "(docs/validation/"), encoding="utf-8")
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
