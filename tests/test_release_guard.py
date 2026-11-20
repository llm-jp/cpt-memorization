import gzip
import io
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.verify_release import check_contents, check_release_inputs


class ReleaseGuardTests(unittest.TestCase):
    def test_text_formats_and_credentials(self):
        home_path = "/" + "/".join(["home", "private-account", "work"])
        token = "ghp_" + "aB012345" * 5
        for filename, payload in [
            ("drawing.svg", home_path.encode()),
            ("config.toml", token.encode()),
            ("manifest.lock", home_path.encode("utf-16")),
            ("pages.jsonl.gz", gzip.compress(token.encode())),
            ("local.ini", ("file://" + home_path).encode()),
            ("build.ps1", ("C:" + "\\".join(["", "Users", "private-account", "work"])).encode()),
        ]:
            with self.subTest(filename=filename), self.assertRaises(ValueError) as caught:
                check_contents(filename, io.BytesIO(payload))
            self.assertNotIn(token, str(caught.exception))
            self.assertNotIn(home_path, str(caught.exception))
        for prefix in ["sk-proj-", "hf_", "github_pat_", "xoxb-"]:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                check_contents("code", io.BytesIO((prefix + "aB012345" * 5).encode()))
        for key in ["AKIA" + "A1" * 8, "-----BEGIN " + "OPENSSH PRIVATE KEY-----"]:
            with self.assertRaisesRegex(ValueError, "Credential-like literal"):
                check_contents("config", io.BytesIO(key.encode()))

    def test_placeholders_urls_and_fixtures(self):
        home = "/" + "home"
        for snippet in (
            "https://example.org" + home + "/documentation",
            home + "/<user>/checkpoints",
            home + "/username/project",
            "data/lid.176.bin",
            '{"id":"synthetic", "text":"Synthetic source text."}',
            'OPENAI_API_KEY="${YOUR_API_KEY}"',
            "hf_<token>",
        ):
            check_contents("fixture", io.BytesIO(snippet.encode()))

    def test_git_candidates_and_built_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            ignored = root / "private-results"
            ignored.mkdir()
            (root / ".gitignore").write_text("private-results/\ndist/\n", encoding="utf-8")
            dist = root / "dist"
            dist.mkdir()
            wheel, source = dist / "sample.whl", dist / "sample.tar.gz"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("package/__init__.py", "")
            required = [
                "README.md",
                "LICENSE_STATUS.md",
                "THIRD_PARTY_NOTICES.md",
                "uv.lock",
                "docs/REPRODUCING.md",
                "examples/qa_manifest.json",
                "scripts/check.sh",
            ]

            def write_sdist(extra=None, link=None):
                with tarfile.open(source, "w:gz") as archive:
                    for filename, content in {
                        **dict.fromkeys(required, b""),
                        **(extra or {}),
                    }.items():
                        member = tarfile.TarInfo("package-1/" + filename)
                        member.size = len(content)
                        archive.addfile(member, io.BytesIO(content))
                    if link is not None:
                        member = tarfile.TarInfo("package-1/docs/linked")
                        member.type, member.linkname = link
                        archive.addfile(member)

            def validate():
                return check_release_inputs(root, wheel, source, dist / "extracted", ["git"])

            token = ("hf_" + "ABcd12" * 6).encode()
            raw_file = ignored / "raw.txt"
            raw_file.write_bytes(token)
            write_sdist()
            self.assertTrue(validate().is_dir())
            candidate = root / "example.toml"
            candidate.write_bytes(token)
            with self.assertRaisesRegex(ValueError, "Credential-like literal in example.toml"):
                validate()
            candidate.unlink()
            subprocess.run(["git", "add", "-f", "private-results/raw.txt"], cwd=root, check=True)
            with self.assertRaisesRegex(ValueError, "Ignored content in release"):
                validate()
            subprocess.run(
                ["git", "rm", "--cached", "-q", "private-results/raw.txt"], cwd=root, check=True
            )
            for name in ["notes.tsv", "data/corpus.jsonl", "private-results/stats.csv"]:
                with self.subTest(name=name):
                    write_sdist({name: token})
                    with self.assertRaises(ValueError):
                        validate()
            absolute_target = str(root / "unpublished")
            for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                with self.subTest(link_type=kind):
                    write_sdist(link=(kind, absolute_target))
                    with self.assertRaisesRegex(ValueError, "Unsafe link target") as caught:
                        validate()
                    self.assertNotIn(absolute_target, str(caught.exception))
                    self.assertFalse((dist / "extracted/package-1/docs/linked").is_symlink())
            write_sdist(link=(tarfile.SYMTYPE, "../README.md"))
            self.assertTrue((validate() / "docs/linked").is_file())
            write_sdist()
            with zipfile.ZipFile(wheel, "a") as archive:
                entry = zipfile.ZipInfo("package/linked")
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(entry, absolute_target)
            with self.assertRaisesRegex(ValueError, "Unsafe link target") as caught:
                validate()
            self.assertNotIn(absolute_target, str(caught.exception))
            write_sdist()
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("package/figure.svg", token)
            with self.assertRaisesRegex(
                ValueError, "Credential-like literal in package/figure.svg"
            ):
                validate()
