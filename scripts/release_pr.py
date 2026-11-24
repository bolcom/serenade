"""Prepare and merge a release PR around the binary build jobs. Run from main."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from release_version import bump_version, version_at


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def api(path, method="GET", **fields):
    command = ["gh", "api", path, "--method", method]
    for name, value in fields.items():
        command.extend(["-f", f"{name}={value}"])
    return json.loads(subprocess.check_output(command, text=True))


def validate_pr(pr, repository, expected_head=None):
    if (pr["state"] != "open" or pr["draft"]
            or pr["base"]["ref"] != "main"
            or (pr["head"]["repo"] or {}).get("full_name") != repository
            or not pr["head"]["ref"].startswith("release/")):
        raise ValueError("Use an open, non-draft release/* PR from this repository into main")
    if expected_head and pr["head"]["sha"] != expected_head:
        raise ValueError("PR changed during the build; restart the release")


def output(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def prepare(repository, number):
    pr = api(f"repos/{repository}/pulls/{number}")
    validate_pr(pr, repository)
    git("fetch", "origin", "main", f"refs/heads/{pr['head']['ref']}")
    base = git("rev-parse", "origin/main")
    head = pr["head"]["sha"]
    if git("merge-base", base, head) != base:
        raise ValueError("Update the release branch with main before starting the release")
    with tempfile.TemporaryDirectory(prefix="serenade-release-") as directory:
        worktree = Path(directory) / "checkout"
        git("worktree", "add", "--detach", str(worktree), head)
        try:
            tag = bump_version(version_at(base), worktree)
            if git("status", "--porcelain", cwd=worktree):
                git("add", "Cargo.toml", "Cargo.lock", cwd=worktree)
                git("-c", "user.name=github-actions[bot]", "-c",
                    "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                    "commit", "-m", f"Release {tag}", cwd=worktree)
                head = git("rev-parse", "HEAD", cwd=worktree)
                # Normal push rejects concurrent branch changes. Never force-push.
                git("push", "origin", f"HEAD:refs/heads/{pr['head']['ref']}", cwd=worktree)
        finally:
            git("worktree", "remove", "--force", str(worktree))
    validate_pr(api(f"repos/{repository}/pulls/{number}"), repository, head)
    output(ref=head, base=base, version=tag, pr=number)
    status(repository, head, "pending", "Building release binaries")


def status(repository, head, state, description):
    api(f"repos/{repository}/statuses/{head}", "POST", state=state,
        context="release-binaries", description=description,
        target_url=f"{os.environ['GITHUB_SERVER_URL']}/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}")


def merge(repository, number, head, base, result):
    if result != "success":
        status(repository, head, "failure", "Release builds did not pass")
        raise ValueError("Build failed or was cancelled; PR remains unmerged")
    pr = api(f"repos/{repository}/pulls/{number}")
    validate_pr(pr, repository, head)
    current_base = api(f"repos/{repository}/git/ref/heads/main")["object"]["sha"]
    if current_base != base:
        status(repository, head, "failure", "Main changed; update branch and restart release")
        raise ValueError("Main changed during the build; update branch and restart release")
    status(repository, head, "success", "All release binaries built and tested")
    # GitHub enforces required reviews/checks; sha rejects concurrent PR changes.
    merged = api(f"repos/{repository}/pulls/{number}/merge", "PUT",
                 sha=head, merge_method="squash")
    if not merged.get("merged"):
        raise ValueError(f"PR was not merged: {merged.get('message')}")
    merged_sha = merged["sha"]
    tested_tree = api(f"repos/{repository}/git/commits/{head}")["tree"]["sha"]
    merged_tree = api(f"repos/{repository}/git/commits/{merged_sha}")["tree"]["sha"]
    if tested_tree != merged_tree:
        raise ValueError("Merged tree differs from tested tree; refusing to publish")
    output(ref=merged_sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "merge"])
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head")
    parser.add_argument("--base")
    parser.add_argument("--result")
    args = parser.parse_args()
    if args.pr <= 0:
        parser.error("PR number must be positive")
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "prepare":
        prepare(repository, args.pr)
    else:
        if not all([args.head, args.base, args.result]):
            parser.error("merge requires --head, --base and --result")
        merge(repository, args.pr, args.head, args.base, args.result)
