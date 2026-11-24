"""Prepare and merge a release PR around the binary build jobs. Run from main."""
import argparse
import json
import os
import re
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
            or (pr["head"]["repo"] or {}).get("full_name") != repository):
        raise ValueError("Use an open, non-draft PR from this repository into main")
    if expected_head and pr["head"]["sha"] != expected_head:
        raise ValueError("PR changed during the build; restart the release")


def output(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for name, value in values.items():
            stream.write(f"{name}={value}\n")


def require_auto_merge(pr):
    if not pr.get("auto_merge"):
        raise ValueError("Auto-merge was disabled; release cancelled")


SKIP_RELEASE = "Skip version bump, binary builds and release; merge changes only"


def skip_release(pr):
    # Only the actual task-list control counts, not a quoted example or code block.
    body = re.sub(r"(?ms)^```.*?^```[^\n]*", "", pr.get("body") or "")
    return bool(re.search(r"(?mi)^- \[x\] " + re.escape(SKIP_RELEASE) + r"\s*$", body))


def gate(repository, number):
    pr = api(f"repos/{repository}/pulls/{number}")
    validate_pr(pr, repository)
    if not pr.get("auto_merge"):
        status(repository, pr["head"]["sha"], "pending",
               "Enable auto-merge to increment version, build and deploy binaries")


def prepare(repository, number):
    pr = api(f"repos/{repository}/pulls/{number}")
    validate_pr(pr, repository)
    require_auto_merge(pr)
    git("fetch", "origin", "main", f"refs/heads/{pr['head']['ref']}")
    base = git("rev-parse", "origin/main")
    head = pr["head"]["sha"]
    if git("merge-base", base, head) != base:
        raise ValueError("Update the release branch with main before starting the release")
    if skip_release(pr):
        output(ref=head, base=base, version="", pr=number, release="false")
        status(repository, head, "pending", "Quick merge requested; skipping version bump and binary release")
        return
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
    updated = api(f"repos/{repository}/pulls/{number}")
    validate_pr(updated, repository, head)
    require_auto_merge(updated)
    if skip_release(updated):
        raise ValueError("Release checkbox changed; restart the workflow")
    output(ref=head, base=base, version=tag, pr=number, release="true")
    status(repository, head, "pending", "Building release binaries")


def status(repository, head, state, description):
    api(f"repos/{repository}/statuses/{head}", "POST", state=state,
        context="release-binaries", description=description,
        target_url=f"{os.environ['GITHUB_SERVER_URL']}/{repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}")


def merge(repository, number, head, base, result, quick_merge=False):
    if result != "success" and not (quick_merge and result == "skipped"):
        status(repository, head, "failure", "Release builds did not pass")
        raise ValueError("Build failed or was cancelled; PR remains unmerged")
    pr = api(f"repos/{repository}/pulls/{number}")
    if not pr.get("merged"):
        validate_pr(pr, repository, head)
        require_auto_merge(pr)
        if skip_release(pr) != quick_merge:
            raise ValueError("Release checkbox changed; restart the workflow")
        current_base = api(f"repos/{repository}/git/ref/heads/main")["object"]["sha"]
        if current_base != base:
            status(repository, head, "failure", "Main changed; update branch and restart release")
            raise ValueError("Main changed during the build; update branch and restart release")
        status(repository, head, "success",
               "Quick merge; version bump and binary release skipped" if quick_merge
               else "All release binaries built and tested")
        # Native auto-merge can win the race once the required status passes.
        pr = api(f"repos/{repository}/pulls/{number}")
    if not pr.get("merged"):
        validate_pr(pr, repository, head)
        require_auto_merge(pr)
        if skip_release(pr) != quick_merge:
            raise ValueError("Release checkbox changed; restart the workflow")
        try:
            merged = api(f"repos/{repository}/pulls/{number}/merge", "PUT",
                         sha=head, merge_method=pr["auto_merge"]["merge_method"].lower())
        except subprocess.CalledProcessError:
            pr = api(f"repos/{repository}/pulls/{number}")
            if not pr.get("merged"):
                raise  # Required reviews/checks still apply.
        else:
            if not merged.get("merged"):
                raise ValueError(f"PR was not merged: {merged.get('message')}")
            pr = api(f"repos/{repository}/pulls/{number}")
    if not pr.get("merged") or pr["head"]["sha"] != head:
        raise ValueError("PR did not merge the tested commit; refusing to publish")
    merged_sha = pr["merge_commit_sha"]
    tested_tree = api(f"repos/{repository}/git/commits/{head}")["tree"]["sha"]
    merged_tree = api(f"repos/{repository}/git/commits/{merged_sha}")["tree"]["sha"]
    if tested_tree != merged_tree:
        raise ValueError("Merged tree differs from tested tree; refusing to publish")
    output(ref=merged_sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gate", "prepare", "merge"])
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head")
    parser.add_argument("--base")
    parser.add_argument("--result")
    parser.add_argument("--quick-merge", choices=["true", "false"], default="false")
    args = parser.parse_args()
    if args.pr <= 0:
        parser.error("PR number must be positive")
    repository = os.environ["GITHUB_REPOSITORY"]
    if args.mode == "gate":
        gate(repository, args.pr)
    elif args.mode == "prepare":
        prepare(repository, args.pr)
    else:
        if not all([args.head, args.base, args.result]):
            parser.error("merge requires --head, --base and --result")
        merge(repository, args.pr, args.head, args.base, args.result, args.quick_merge == "true")
