import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import publish_release as release


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'Cargo.toml').write_text('[package]\nname="serenade"\nversion="1.2.3"\n')
        self.dist = self.root / 'dist'
        self.dist.mkdir()
        for name in release.archive_names('v1.2.3'):
            archive = self.dist / name
            archive.write_bytes(name.encode())
            (self.dist / (name + '.sha256')).write_text(f'{release.digest(archive)}  {name}\n')
        self.remote = self.root / 'remote'
        self.remote.mkdir()
        self.tag = False
        self.draft = None
        self.crate = False
        self.commands = []
        self.cargo_exit = 0
        self.upload_succeeds = True
        self.fail_finalize = False
        for mock in [patch.dict(os.environ, GH_REPO='bolcom/serenade', GH_TOKEN='test',
                                CARGO_REGISTRY_TOKEN='test', RELEASE_SHA='abc'),
                     patch.object(release.subprocess, 'check_output', side_effect=self.git),
                     patch.object(release.subprocess, 'run', side_effect=self.cargo),
                     patch.object(release, 'gh', side_effect=self.gh),
                     patch.object(release, 'github', side_effect=self.github),
                     patch.object(release, 'published_crate', side_effect=lambda *args: self.crate),
                     patch.object(release.time, 'sleep')]:
            mock.start()
            self.addCleanup(mock.stop)

    def git(self, args, **kwargs):
        return 'abc\n' if args[1] == 'rev-parse' else ''

    def cargo(self, args, **kwargs):
        self.assertEqual(args, ['cargo', '+stable', 'publish', '--locked'])
        self.commands.append('cargo')
        self.crate = self.upload_succeeds
        return subprocess.CompletedProcess(args, self.cargo_exit)

    def github(self, path):
        if path.startswith('git/ref/'):
            return {'object': {'type': 'commit', 'sha': 'abc'}} if self.tag else None
        return None if self.draft is None else {'draft': self.draft}

    def gh(self, *args):
        self.commands.append(args[:2])
        if args[0] == 'api':
            self.tag = True
        elif args[1] == 'create':
            self.draft = True
        elif args[1] == 'upload':
            self.assertTrue(self.draft, 'Published assets must never be overwritten')
            for name in args[3:-1]:
                shutil.copyfile(name, self.remote / Path(name).name)
        elif args[1] == 'download':
            for source in self.remote.iterdir():
                shutil.copyfile(source, Path(args[4]) / source.name)
        elif args[1] == 'edit':
            self.assertTrue(self.crate, 'Crate must exist before making GitHub release public')
            if self.fail_finalize:
                raise RuntimeError('GitHub unavailable')
            self.draft = False

    def test_fresh_release_then_idempotent_retry(self):
        release.publish(self.root)
        self.assertFalse(self.draft)
        self.assertTrue(self.tag)
        self.assertTrue(self.crate)
        self.commands.clear()
        # Rebuilt binaries can differ; a retry verifies the original public assets.
        archive = self.dist / release.archive_names('v1.2.3')[0]
        archive.write_bytes(b'rebuilt')
        archive.with_name(archive.name + '.sha256').write_text(f'{release.digest(archive)}  {archive.name}')
        release.publish(self.root)
        self.assertEqual(self.commands, [('release', 'download')])

    def test_retry_after_crate_upload_before_github_finalization(self):
        self.fail_finalize = True
        with self.assertRaisesRegex(RuntimeError, 'GitHub unavailable'):
            release.publish(self.root)
        self.assertTrue(self.crate)
        self.assertTrue(self.draft)
        self.fail_finalize = False
        self.commands.clear()
        release.publish(self.root)
        self.assertFalse(self.draft)
        self.assertNotIn('cargo', self.commands)

    def test_failed_upload_leaves_resumable_draft(self):
        self.upload_succeeds = False
        self.cargo_exit = 1
        with self.assertRaisesRegex(RuntimeError, 'rerun workflow'):
            release.publish(self.root)
        self.assertTrue(self.draft)
        self.upload_succeeds = True
        self.cargo_exit = 0
        release.publish(self.root)
        self.assertFalse(self.draft)

    def test_timeout_after_successful_crate_upload_is_recovered(self):
        self.cargo_exit = 1
        release.publish(self.root)
        self.assertFalse(self.draft)

    def test_existing_public_release_can_complete_missing_crate(self):
        release.publish(self.root)
        self.crate = False
        self.commands.clear()
        release.publish(self.root)
        self.assertEqual(self.commands, [('release', 'download'), 'cargo'])

    def test_conflicting_destinations_fail_before_mutation(self):
        with patch.object(release, 'published_crate', side_effect=ValueError('different source')):
            with self.assertRaisesRegex(ValueError, 'different source'):
                release.publish(self.root)
        with patch.object(release, 'verify_tag', side_effect=ValueError('different commit')):
            with self.assertRaisesRegex(ValueError, 'different commit'):
                release.publish(self.root)
        self.assertEqual(self.commands, [])

    def test_corrupt_public_assets_and_provenance_are_rejected(self):
        release.publish(self.root)
        provenance = self.remote / release.source_name('v1.2.3')
        original = provenance.read_text()
        provenance.write_text(original.replace('abc', 'def'))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            release.publish(self.root)
        provenance.write_text(original)
        (self.remote / release.archive_names('v1.2.3')[0]).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            release.publish(self.root)

    def test_missing_artifacts_token_or_wrong_checkout_fail_before_mutation(self):
        for environment, message in [({'RELEASE_SHA': 'def'}, 'validated'),
                                     ({'CARGO_REGISTRY_TOKEN': ''}, 'secret')]:
            with patch.dict(os.environ, environment):
                with self.assertRaisesRegex(ValueError, message):
                    release.publish(self.root)
        with patch.object(release.subprocess, 'check_output', side_effect=['abc', ' M Cargo.toml']):
            with self.assertRaisesRegex(ValueError, 'clean'):
                release.publish(self.root)
        (self.dist / release.archive_names('v1.2.3')[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            release.publish(self.root)
        self.assertEqual(self.commands, [])


class RemoteIdentityTests(unittest.TestCase):
    def crate(self, sha='abc', dirty=False):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode='w:gz') as archive:
            for name, content in {
                '.cargo_vcs_info.json': json.dumps({'git': {'sha1': sha, 'dirty': dirty}}),
                'Cargo.toml': '[package]\nname="serenade"\nversion="1.2.3"\n',
            }.items():
                entry = tarfile.TarInfo('serenade-1.2.3/' + name)
                entry.size = len(content.encode())
                archive.addfile(entry, io.BytesIO(content.encode()))
        data.seek(0)
        return data

    def test_checks_actual_published_source_metadata(self):
        for sha, dirty in [('def', False), ('abc', True), ('abc', False)]:
            with self.subTest(sha=sha, dirty=dirty), patch.object(release, 'urlopen', return_value=self.crate(sha, dirty)):
                if sha != 'abc' or dirty:
                    with self.assertRaisesRegex(ValueError, 'different source'):
                        release.published_crate('serenade', '1.2.3', 'abc')
                else:
                    self.assertTrue(release.published_crate('serenade', '1.2.3', 'abc'))

    def test_only_404_is_absent(self):
        with patch.dict(os.environ, GH_REPO='bolcom/serenade', GH_TOKEN='test'):
            for code in (404, 403, 429, 500):
                for call in (lambda: release.published_crate('serenade', '1.2.3', 'abc'),
                             lambda: release.github('releases/tags/v1.2.3')):
                    with patch.object(release, 'urlopen', side_effect=HTTPError('url', code, 'error', {}, None)):
                        if code == 404:
                            self.assertFalse(call())
                        else:
                            with self.assertRaises(HTTPError):
                                call()

    def test_annotated_tag_resolves_to_commit_and_rejects_conflicts(self):
        for sha in ('abc', 'def'):
            with patch.object(release, 'github', side_effect=[
                {'object': {'type': 'tag', 'sha': 'tag-object'}},
                {'object': {'type': 'commit', 'sha': sha}},
            ]):
                if sha == 'abc':
                    self.assertTrue(release.verify_tag('v1.2.3', 'abc'))
                else:
                    with self.assertRaisesRegex(ValueError, 'different source'):
                        release.verify_tag('v1.2.3', 'abc')
