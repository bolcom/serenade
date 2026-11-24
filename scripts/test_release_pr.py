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

    def run_merge(self, result="success", base="base", tree="tree", head="tested"):
        responses = [self.pr, {"object": {"sha": base}},
                     {"merged": True, "sha": "merged"},
                     {"tree": {"sha": "tree"}}, {"tree": {"sha": tree}}]
        with patch.object(release_pr, "api", side_effect=responses) as api, \
             patch.object(release_pr, "status") as status, \
             patch.object(release_pr, "output") as output:
            try:
                release_pr.merge(self.repository, 27, head, "base", result)
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
        self.assertEqual(api.call_args_list[2].kwargs,
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

    def test_rejected_merge_has_no_publish_output(self):
        with patch.object(release_pr, "api", side_effect=[
                self.pr, {"object": {"sha": "base"}},
                {"merged": False, "message": "Required checks missing"}]), \
             patch.object(release_pr, "status"), \
             patch.object(release_pr, "output") as output:
            with self.assertRaisesRegex(ValueError, "Required checks missing"):
                release_pr.merge(self.repository, 27, "tested", "base", "success")
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
