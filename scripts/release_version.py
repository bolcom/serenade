"""Print a release tag only when the Cargo package version increases."""
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


if __name__ == "__main__":
    print(release_tag(version_at(sys.argv[1]), version_at(sys.argv[2])))
