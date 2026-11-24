"""Validate or bump the package version on a release PR."""
from pathlib import Path
import re
import subprocess
import sys
import tomllib


def version_at(ref):
    manifest = subprocess.check_output(["git", "show", f"{ref}:Cargo.toml"], text=True)
    return tomllib.loads(manifest)["package"]["version"]


def release_tag(previous, current):
    def parse(version):
        if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
            raise ValueError(f"Expected a stable major.minor.patch version: {version}")
        return tuple(map(int, version.split(".")))

    return f"v{current}" if parse(current) > parse(previous) else ""


def bump_version(previous, root=Path(".")):
    manifest = root / "Cargo.toml"
    lockfile = root / "Cargo.lock"
    manifest_text = manifest.read_text()
    lock_text = lockfile.read_text()
    package = tomllib.loads(manifest_text)["package"]
    current = package["version"]
    locked = [entry["version"] for entry in tomllib.loads(lock_text).get("package", [])
              if entry["name"] == package["name"]]
    if locked != [current]:
        raise ValueError("Root package versions in Cargo.toml and Cargo.lock must match")
    tag = release_tag(previous, current)
    if tag:
        return tag  # Preserve an explicit version increase.
    if current != previous:
        raise ValueError("Refusing to release a decreased package version")
    major, minor, patch = map(int, current.split("."))
    version = f"{major}.{minor}.{patch + 1}"
    # Restrict edits to the root package and its lockfile entry.
    manifest_pattern = r'(?ms)(^\[package\]\s*\n.*?^version\s*=\s*")' + re.escape(current) + r'(")'
    lock_pattern = r'(?m)(^name = "' + re.escape(package["name"]) + r'"\nversion = ")' + re.escape(current) + r'(")'
    manifest_text, manifests = re.subn(manifest_pattern, lambda m: m[1] + version + m[2], manifest_text, count=1)
    lock_text, locks = re.subn(lock_pattern, lambda m: m[1] + version + m[2], lock_text)
    if manifests != 1 or locks != 1:
        raise ValueError("Expected one root package version in Cargo.toml and Cargo.lock")
    manifest.write_text(manifest_text)
    lockfile.write_text(lock_text)
    return f"v{version}"


if __name__ == "__main__":
    if sys.argv[1] == "--bump":
        print(bump_version(version_at(sys.argv[2])))
    else:
        print(release_tag(version_at(sys.argv[1]), version_at(sys.argv[2])))
