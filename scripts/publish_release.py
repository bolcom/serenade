"""Publish/resume one Cargo version from one validated commit (CI entry point)."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from release_version import release_tag

PLATFORMS = ("windows-x86_64", "linux-x86_64", "macos-x86_64", "macos-aarch64")


def gh(*args):
    subprocess.run(["gh", *args], check=True)


def github(path):
    request = Request(f"https://api.github.com/repos/{os.environ['GH_REPO']}/{path}",
                      headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                               "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28"})
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise


def published_crate(name, version, sha):
    # Only 404 means absent. Authentication, rate limits and network errors fail closed.
    request = Request(f"https://static.crates.io/crates/{name}/{name}-{version}.crate",
                      headers={"User-Agent": "serenade-release (github.com/bolcom/serenade)"})
    try:
        with urlopen(request, timeout=60) as response:
            data = response.read()
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        prefix = f"{name}-{version}/"
        vcs = json.load(archive.extractfile(prefix + ".cargo_vcs_info.json"))
        manifest = tomllib.loads(archive.extractfile(prefix + "Cargo.toml").read().decode())
    if (vcs["git"]["sha1"] != sha or vcs["git"].get("dirty", False)
            or manifest["package"]["version"] != version
            or manifest["package"]["name"] != name):
        raise ValueError("Published crate version identifies different source; refusing to release")
    return True


def archive_names(tag):
    return [f"serenade-{tag}-{platform}.{'zip' if platform.startswith('windows') else 'tar.gz'}"
            for platform in PLATFORMS]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archives(directory, tag):
    hashes = {}
    for name in archive_names(tag):
        archive = directory / name
        checksum = directory / (name + ".sha256")
        if archive.stat().st_size == 0:
            raise ValueError(f"Empty release archive: {name}")
        actual = digest(archive)
        if checksum.read_text().strip() != f"{actual}  {name}":
            raise ValueError(f"Release checksum mismatch: {name}")
        hashes[name] = actual
    return hashes


def source_name(tag):
    return f"serenade-{tag}-source.json"


def verify_release(directory, tag, version, sha):
    hashes = verify_archives(directory, tag)
    provenance = json.loads((directory / source_name(tag)).read_text())
    if provenance != {"version": version, "commit": sha, "archives": hashes}:
        raise ValueError("GitHub release provenance does not match its version, commit and archives")


def verify_tag(tag, sha):
    ref = github(f"git/ref/tags/{tag}")
    if ref is None:
        return False
    obj = ref["object"]
    # Annotated tags identify a tag object, not the underlying commit.
    while obj["type"] == "tag":
        obj = github(f"git/tags/{obj['sha']}")["object"]
    if obj["type"] != "commit" or obj["sha"] != sha:
        raise ValueError(f"Tag {tag} identifies a different source commit")
    return True


def verify_remote_release(tag, version, sha):
    # Use the already published binaries on retries: rebuilds need not be byte-identical.
    with tempfile.TemporaryDirectory(prefix="serenade-release-check-") as temporary:
        directory = Path(temporary)
        gh("release", "download", tag, "--dir", str(directory))
        verify_release(directory, tag, version, sha)


def publish(root=Path(".")):
    package = tomllib.loads((root / "Cargo.toml").read_text())["package"]
    name, version = package["name"], package["version"]
    tag = release_tag("0.0.0", version)
    if not tag:
        raise ValueError("Release version must be greater than 0.0.0")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if sha != os.environ["RELEASE_SHA"]:
        raise ValueError("Checkout does not match the validated release commit")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Release checkout must be clean")
    if not os.environ.get("CARGO_REGISTRY_TOKEN"):
        raise ValueError("CARGO_REGISTRY_TOKEN secret is required")
    distribution = root / "dist"
    hashes = verify_archives(distribution, tag)
    # Check immutable destinations before mutating either service.
    crate_exists = published_crate(name, version, sha)
    tag_exists = verify_tag(tag, sha)
    release = github(f"releases/tags/{tag}")
    if release and not tag_exists:
        raise ValueError("Existing release has no matching source tag")
    if release and not release["draft"]:
        verify_remote_release(tag, version, sha)
    else:
        if not tag_exists:
            gh("api", f"repos/{os.environ['GH_REPO']}/git/refs", "--method", "POST",
               "-f", f"ref=refs/tags/{tag}", "-f", f"sha={sha}")
        if not release:
            gh("release", "create", tag, "--verify-tag", "--target", sha, "--draft",
               "--title", f"Serenade {tag}", "--generate-notes")
        provenance = distribution / source_name(tag)
        provenance.write_text(json.dumps({"version": version, "commit": sha,
                                          "archives": hashes}, indent=2) + "\n")
        files = [distribution / filename for name in archive_names(tag)
                 for filename in (name, name + ".sha256")]
        gh("release", "upload", tag, *map(str, [*files, provenance]), "--clobber")
        verify_remote_release(tag, version, sha)
    if not crate_exists:
        # Cargo can time out after a successful upload. Recheck the immutable crate
        # before deciding whether a failed command requires a workflow retry.
        result = subprocess.run(["cargo", "+stable", "publish", "--locked"], cwd=root)
        for attempt in range(6):
            if published_crate(name, version, sha):
                break
            if attempt == 5:
                raise RuntimeError(f"Crate publication not visible (cargo exit {result.returncode}); rerun workflow")
            time.sleep(10)
    if not release or release["draft"]:
        gh("release", "edit", tag, "--draft=false")
    print(f"Published/verified {name} {version} from {sha} on GitHub and crates.io")


if __name__ == "__main__":
    publish()
