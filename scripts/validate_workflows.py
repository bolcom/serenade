"""Exercise locally built release executables on the bundled corpus."""
from pathlib import Path
import shutil
import tempfile
import zipfile
from package_release import BINS, ROOT, smoke
import os

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="serenade-e2e-") as temporary:
        root = Path(temporary)
        suffix = ".exe" if os.name == "nt" else ""
        for binary in BINS:
            shutil.copy2(ROOT / "target/release" / (binary + suffix), root)
        (root / "configs").mkdir()
        shutil.copy2(ROOT / "configs/hpo.toml", root / "configs/hpo.toml")
        data = root / "assets/example"
        data.mkdir(parents=True)
        shutil.copy2(ROOT / "assets/example/workflows.toml", data)
        with zipfile.ZipFile(ROOT / "assets/example/example.zip") as archive:
            for name in ("train.txt", "test.txt", "valid.txt"):
                (data / name).write_bytes(archive.read(name))
        smoke(root, iterations=50)
