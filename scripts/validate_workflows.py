"""End-to-end checks on the bundled corpus; run after building release binaries."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BIN = ROOT / "target" / "release"
SUFFIX = ".exe" if os.name == "nt" else ""
ALGORITHMS = ["vmis", "sknn", "vsknn", "stan", "vsknn_stan"]
CONFIG = "assets/example/workflows.toml"


def run(binary, *args):
    result = subprocess.run([str(BIN / (binary + SUFFIX)), *map(str, args)],
                            check=True, capture_output=True, text=True)
    return result.stdout


def read(path):
    return json.loads(path.read_text())


def metrics(result):
    return {key: result[key] for key in
            ["algorithm", "predictions", "mrr_at_20", "ndcg_at_20", "hitrate_at_20", "coverage_at_20"]}


def main():
    with zipfile.ZipFile("assets/example/example.zip") as archive:
        archive.extractall("assets/example")
    with tempfile.TemporaryDirectory(prefix="serenade-e2e-") as temp:
        output = Path(temp)
        evaluation = output / "evaluation.json"
        run("evaluator", CONFIG, "all", evaluation)
        baseline = read(evaluation)["results"]
        assert [r["algorithm"] for r in baseline] == ALGORITHMS
        assert {r["predictions"] for r in baseline} == {931}
        hpo = output / "hpo"
        run("tpe_hyperparameter_optm", CONFIG, "all", hpo)
        for algorithm, expected in zip(ALGORITHMS, baseline):
            # Each standalone selection must exactly match the combined execution.
            run("evaluator", CONFIG, algorithm, evaluation)
            assert metrics(read(evaluation)["results"][0]) == metrics(expected)
            single = output / "single"
            run("tpe_hyperparameter_optm", CONFIG, algorithm, single)
            best = read(hpo / (algorithm + ".json"))
            assert best == read(single / (algorithm + ".json")), algorithm
            assert len(best["trials"]) == 3
            # Reload the generated config with a fresh index; reproduce validation score.
            serving = hpo / (algorithm + "-serving.toml")
            validation = output / "validation.toml"
            validation.write_text(serving.read_text() + '\n[hyperparam]\ntest_data_path = "assets/example/valid.txt"\n')
            run("evaluator", validation, algorithm, evaluation)
            actual = read(evaluation)["results"][0]
            assert actual["mrr_at_20"] == best["best_validation_mrr_at_20"], (algorithm, actual, best)
            # Isolate the session database and use an available ephemeral port.
            import socket
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            server_dir = output / algorithm
            server_dir.mkdir()
            server_config = server_dir / "serving.toml"
            text = serving.read_text().replace('host = "0.0.0.0"', 'host = "127.0.0.1"')
            text = text.replace("port = 8081", "port = " + str(port))
            text = text.replace("assets/example/train.txt", (ROOT / "assets/example/train.txt").as_posix())
            server_config.write_text(text)
            with (server_dir / "server.log").open("w") as log:
                process = subprocess.Popen([str(BIN / ("serving" + SUFFIX)), str(server_config)],
                    cwd=server_dir, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    deadline = time.monotonic() + 30
                    url = f"http://127.0.0.1:{port}/v1/recommend?item_id=13598&session_id=e2e-{algorithm}&user_consent=true"
                    while True:
                        assert process.poll() is None, (server_dir / "server.log").read_text()
                        try:
                            with urllib.request.urlopen(url, timeout=1) as response:
                                recommendations = json.load(response)
                            break
                        except (urllib.error.URLError, TimeoutError):
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.1)
                    assert isinstance(recommendations, list) and recommendations, recommendations
                    assert all(isinstance(item, int) for item in recommendations)
                    print(f"{algorithm}: evaluation, seeded TPE, config reload, fresh-index score, serving OK ({len(recommendations)} items)")
                finally:
                    process.terminate()
                    process.wait(timeout=10)
        # Alias follows the existing canonical identifier.
        run("evaluator", CONFIG, "vstan", evaluation)
        assert read(evaluation)["results"][0]["algorithm"] == "vsknn_stan"
    print("All bundled-data workflow checks passed.")


if __name__ == "__main__":
    main()
