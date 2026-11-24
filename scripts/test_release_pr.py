import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import release_pr


class ReleasePrTests(unittest.TestCase):
    def setUp(self):
        self.pr = {"state": "open", "draft": False,
                   "auto_merge": {"merge_method": "SQUASH"}, "body": "",
                   "base": {"ref": "main"},
                   "head": {"ref": "release/0.3.6", "sha": "tested",
                            "repo": {"full_name": "bolcom/serenade"}}}
        self.repository = "bolcom/serenade"

    def test_only_open_same_repository_release_prs(self):
        release_pr.validate_pr(self.pr, self.repository, "tested")
        for field, value in [("state", "closed"), ("draft", True),
                             ("base", {"ref": "other"}),
                             ("head", {"ref": "feature", "sha": "tested", "repo": None})]:
            pr = copy.deepcopy(self.pr)
            pr[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                release_pr.validate_pr(pr, self.repository)
        with self.assertRaises(ValueError):
            release_pr.validate_pr(self.pr, self.repository, "different")

    def run_merge(self, result="success", base="base", tree="tree", head="tested", quick=False):
        merged_pr = dict(self.pr, state="closed", merged=True, merge_commit_sha="merged")
        responses = [self.pr, {"object": {"sha": base}}, self.pr,
                     {"merged": True, "sha": "merged"}, merged_pr,
                     {"tree": {"sha": "tree"}}, {"tree": {"sha": tree}}]
        with patch.object(release_pr, "api", side_effect=responses) as api, \
             patch.object(release_pr, "status") as status, \
             patch.object(release_pr, "output") as output:
            try:
                release_pr.merge(self.repository, 27, head, "base", result, quick)
            except ValueError:
                self.assertFalse(output.called)
                raise
            return api, status, output

    def test_all_builds_must_pass(self):
        for result in ["failure", "cancelled", "skipped"]:
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, "Build failed"):
                self.run_merge(result=result)

    def test_changed_base_or_pr_prevents_merge(self):
        with self.assertRaisesRegex(ValueError, "Main changed"):
            self.run_merge(base="new-main")
        with self.assertRaisesRegex(ValueError, "PR changed"):
            self.run_merge(head="outdated")

    def test_merges_exact_tested_commit_and_outputs_merged_commit(self):
        api, status, output = self.run_merge()
        self.assertEqual(api.call_args_list[3].kwargs,
                         {"sha": "tested", "merge_method": "squash"})
        status.assert_called_once_with(self.repository, "tested", "success",
                                       "All release binaries built and tested")
        output.assert_called_once_with(ref="merged")

    def test_different_merged_tree_prevents_publication(self):
        with self.assertRaisesRegex(ValueError, "Merged tree differs"):
            self.run_merge(tree="changed")

    def test_prepare_pushes_version_bump_and_retry_keeps_same_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote = root / "origin.git"
            checkout = root / "repo"

            def git(*args, cwd=None):
                return subprocess.check_output(
                    ["git", *args], cwd=cwd or checkout, text=True,
                    stderr=subprocess.PIPE).strip()

            git("init", "--bare", str(remote), cwd=root)
            git("clone", str(remote), str(checkout), cwd=root)
            git("config", "user.email", "test@example.com")
            git("config", "user.name", "Release Test")
            git("checkout", "-b", "main")
            (checkout / "Cargo.toml").write_text(
                '[package]\nname = "serenade"\nversion = "0.3.5"\n')
            (checkout / "Cargo.lock").write_text(
                '[[package]]\nname = "serenade"\nversion = "0.3.5"\n')
            git("add", ".")
            git("commit", "-m", "Initial")
            git("push", "origin", "main")
            git("checkout", "-b", "release/0.3.6")
            (checkout / "README.md").write_text("Release changes")
            git("add", ".")
            git("commit", "-m", "Changes")
            git("push", "origin", "HEAD")

            def pr_response(*args):
                pr = copy.deepcopy(self.pr)
                pr["head"]["sha"] = git("rev-parse", "refs/heads/release/0.3.6", cwd=remote)
                return pr

            with patch.object(release_pr, "git", side_effect=git), \
                 patch.object(release_pr, "api", side_effect=pr_response), \
                 patch.object(release_pr, "version_at", return_value="0.3.5"), \
                 patch.object(release_pr, "status"), \
                 patch.object(release_pr, "output") as output:
                release_pr.prepare(self.repository, 27)
                first = output.call_args.kwargs
                self.assertEqual(first["version"], "v0.3.6")
                self.assertIn('version = "0.3.6"', git("show", f"{first['ref']}:Cargo.lock"))
                release_pr.prepare(self.repository, 27)
                self.assertEqual(first, output.call_args.kwargs)
                self.assertEqual(git("rev-list", "--count", "main.." + first["ref"]), "2")

    def test_checkbox_is_explicit_and_ignores_documentation_examples(self):
        for body, expected in [
                ("- [ ] " + release_pr.SKIP_RELEASE, False),
                ("- [x] " + release_pr.SKIP_RELEASE, True),
                ("- [X] " + release_pr.SKIP_RELEASE, True),
                ("> - [x] " + release_pr.SKIP_RELEASE, False),
                ("```markdown\n- [x] " + release_pr.SKIP_RELEASE + "\n```", False),
                (None, False)]:
            with self.subTest(body=body):
                self.assertEqual(release_pr.skip_release({"body": body}), expected)

    def test_quick_merge_accepts_skipped_builds_only_when_requested(self):
        self.pr["body"] = "- [x] " + release_pr.SKIP_RELEASE
        _, status, output = self.run_merge(result="skipped", quick=True)
        self.assertIn("Quick merge", status.call_args.args[3])
        output.assert_called_once_with(ref="merged")
        with self.assertRaisesRegex(ValueError, "checkbox changed"):
            self.run_merge()
        with self.assertRaisesRegex(ValueError, "Build failed"):
            self.run_merge(result="failure", quick=True)

    def test_quick_prepare_does_not_bump_or_push(self):
        self.pr["body"] = "- [x] " + release_pr.SKIP_RELEASE
        with patch.object(release_pr, "api", return_value=self.pr), \
             patch.object(release_pr, "git", side_effect=["", "base", "base"]) as git, \
             patch.object(release_pr, "bump_version") as bump, \
             patch.object(release_pr, "status"), \
             patch.object(release_pr, "output") as output:
            release_pr.prepare(self.repository, 27)
            bump.assert_not_called()
            self.assertFalse(any("push" in call.args for call in git.call_args_list))
            output.assert_called_once_with(ref="tested", base="base", version="",
                                           pr=27, release="false")

    def test_disabling_auto_merge_cancels_release(self):
        self.pr["auto_merge"] = None
        with patch.object(release_pr, "api", return_value=self.pr), \
             patch.object(release_pr, "git") as git:
            with self.assertRaisesRegex(ValueError, "Auto-merge was disabled"):
                release_pr.prepare(self.repository, 27)
            git.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Auto-merge was disabled"):
            self.run_merge()

    def test_native_auto_merge_can_finish_before_workflow_merge(self):
        merged_pr = dict(self.pr, state="closed", merged=True, merge_commit_sha="merged")
        with patch.object(release_pr, "api", side_effect=[
                self.pr, {"object": {"sha": "base"}}, merged_pr,
                {"tree": {"sha": "tree"}}, {"tree": {"sha": "tree"}}]) as api, \
             patch.object(release_pr, "status"), \
             patch.object(release_pr, "output") as output:
            release_pr.merge(self.repository, 27, "tested", "base", "success")
            self.assertFalse(any(call.args[1:] == ("PUT",) for call in api.call_args_list))
            output.assert_called_once_with(ref="merged")

    def test_open_pr_gets_pending_status_without_building(self):
        self.pr["auto_merge"] = None
        with patch.object(release_pr, "api", return_value=self.pr), \
             patch.object(release_pr, "status") as status:
            release_pr.gate(self.repository, 27)
            self.assertEqual(status.call_args.args[2], "pending")

    def test_rejected_merge_has_no_publish_output(self):
        with patch.object(release_pr, "api", side_effect=[
                self.pr, {"object": {"sha": "base"}}, self.pr,
                {"merged": False, "message": "Required checks missing"}]), \
             patch.object(release_pr, "status"), \
             patch.object(release_pr, "output") as output:
            with self.assertRaisesRegex(ValueError, "Required checks missing"):
                release_pr.merge(self.repository, 27, "tested", "base", "success")
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
