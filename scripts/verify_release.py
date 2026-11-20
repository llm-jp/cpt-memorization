"""Check source candidates and distribution archives before testing the built package."""

import gzip
import io
import os
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


LOCAL_PATH = re.compile(
    r"(?<![\w./\\:-])(?:file://)?"
    r"(?:/(?:home|Users|root|mnt|scratch|workspace|shared|gpfs|lustre|nfs|Volumes|model)/"
    r"|[A-Za-z]:[/\\]+Users[/\\]+)"
    r"(?!<|\$|\{|(?:user(?:name)?|example)(?:[/\\]|\b))[\w.-]+"
)
CREDENTIAL = re.compile(
    r"(?<![\w-])(?:(?:sk-(?:proj-|svcacct-)?|hf_|gh[pousr]_|github_pat_)[\w-]{24,}"
    r"|(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])|xox[baprs]-[\w-]{16,}"
    r"|-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----)"
)
PRIVATE_DIRS = {
    ".git",
    ".work",
    ".venv",
    "__pycache__",
    "data",
    "data_",
    "checkpoints",
    "artifacts",
}


def check_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"Unsafe archive path: {name!r}")
    if PRIVATE_DIRS.intersection(path.parts):
        raise ValueError(f"Unexpected private/generated content: {name}")


def check_link(root, git, name, target, *, hardlink=False):
    # Symbolic targets are relative to their entry. Tar hardlinks use the archive root.
    parent = "" if hardlink else posixpath.dirname(name)
    resolved = posixpath.normpath(posixpath.join(parent, target))
    try:
        if not target or "\0" in target or re.match(r"^[A-Za-z]:", target):
            raise ValueError
        check_name(resolved)
        check_git_paths(root, git, [resolved])
    except ValueError:
        raise ValueError(f"Unsafe link target in {name}") from None


def check_contents(name, stream):
    """Inspect textual files independently of their extension, including gzip-compressed text."""
    if name.endswith(".gz"):
        stream = gzip.GzipFile(fileobj=stream)
    with io.BufferedReader(stream) as buffered:
        prefix = buffered.peek(4)
        if prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        elif b"\0" in prefix:
            return
        else:
            encoding = "utf-8"
        with io.TextIOWrapper(buffered, encoding=encoding, errors="replace") as reader:
            for line in reader:
                if LOCAL_PATH.search(line):
                    raise ValueError(f"Machine-specific path in {name}")
                if CREDENTIAL.search(line):
                    raise ValueError(f"Credential-like literal in {name}")


def check_git_paths(root, git, names):
    """Use the repository's ignore syntax, including directory rules and negations."""
    if not names:
        return
    result = subprocess.run(
        [*git, "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=root,
        input=b"\0".join(map(os.fsencode, names)) + b"\0",
        stdout=subprocess.PIPE,
    )
    if result.returncode == 0:
        ignored = os.fsdecode(result.stdout.split(b"\0", 1)[0])
        raise ValueError(f"Ignored content in release: {ignored}")
    if result.returncode != 1:
        result.check_returncode()


def check_release_inputs(root, wheel, source, standalone, git):
    candidates = subprocess.check_output(
        [*git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    names = [os.fsdecode(name) for name in candidates.split(b"\0") if name]
    check_git_paths(root, git, names)
    for name in names:
        check_name(name)
        path = root / name
        if path.is_symlink():
            target = path.resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"External symlink in public files: {name}")
            relative_target = target.relative_to(root).as_posix()
            check_name(relative_target)
            check_git_paths(root, git, [relative_target])
        if path.is_file():
            with path.open("rb") as stream:
                check_contents(name, stream)

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        for name in wheel_names:
            check_name(name)
        check_git_paths(root, git, wheel_names)
        for entry in archive.infolist():
            if stat.S_ISLNK(entry.external_attr >> 16):
                with archive.open(entry) as stream:
                    check_link(root, git, entry.filename, os.fsdecode(stream.read()))
            elif not entry.is_dir():
                with archive.open(entry) as stream:
                    check_contents(entry.filename, stream)
        expected = {
            p.relative_to(root / "src").as_posix()
            for p in (root / "src").rglob("*")
            if p.suffix in (".py", ".yaml")
        }
        missing = expected.difference(wheel_names)
        if missing:
            raise ValueError(f"Wheel is missing package files: {sorted(missing)}")

    with tarfile.open(source) as archive:
        members = archive.getmembers()
        for member in members:
            check_name(member.name)
        roots = {PurePosixPath(member.name).parts[0] for member in members}
        if len(roots) != 1:
            raise ValueError("Expected one top-level directory in the source distribution")
        source_dir = roots.pop()
        names = [
            member.name.removeprefix(source_dir + "/") + ("/" if member.isdir() else "")
            for member in members
            if member.name != source_dir
        ]
        check_git_paths(root, git, names)
        for member in members:
            if member.issym() or member.islnk():
                name = member.name.removeprefix(source_dir + "/")
                target = member.linkname
                if member.islnk():
                    if not target.startswith(source_dir + "/"):
                        raise ValueError(f"Unsafe link target in {name}")
                    target = target.removeprefix(source_dir + "/")
                check_link(root, git, name, target, hardlink=member.islnk())
            elif member.isfile():
                with archive.extractfile(member) as stream:
                    check_contents(member.name, stream)
        required = {
            "README.md",
            "LICENSE_STATUS.md",
            "THIRD_PARTY_NOTICES.md",
            "uv.lock",
            "docs/REPRODUCING.md",
            "examples/qa_manifest.json",
            "scripts/check.sh",
        }
        missing = required.difference(names)
        if missing:
            raise ValueError(f"Source distribution is missing {sorted(missing)}")
        archive.extractall(standalone, filter="data")
    return standalone / source_dir


def main():
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("*.whl"))
    sources = list((root / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("Expected exactly one wheel and one source distribution in dist/")
    scratch = root / ".work/tmp"
    scratch.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        git = ["git", "-c", f"core.excludesFile={os.devnull}"]
        if not (root / ".git").exists():
            # A standalone export still has .gitignore but no Git metadata.
            git_dir = Path(temporary) / "git"
            subprocess.run(["git", "init", "--bare", "-q", str(git_dir)], check=True)
            git += [f"--git-dir={git_dir}", f"--work-tree={root}"]
        source_root = check_release_inputs(
            root, wheels[0], sources[0], Path(temporary) / "sdist", git
        )
        for path in [*root.glob("*.md"), *(root / "docs").rglob("*.md")]:
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0]
                if not (path.parent / target).exists():
                    raise ValueError(f"Broken documentation link in {path.name}: {target}")
        probe = """
from pathlib import Path
import cpt_memorization
from cpt_memorization.qa.common import load_prompt
assert '.whl/' in cpt_memorization.__file__, cpt_memorization.__file__
assert '{{ document }}' in load_prompt('open')['user_prompt']
assert '{{ document }}' not in load_prompt('closed')['user_prompt']
assert load_prompt('evaluate')['output_schema']['properties']['is_correct']['type'] == 'boolean'
print('Built wheel imports and bundled prompts verified')
"""
        subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=scratch,
            env={**os.environ, "PYTHONPATH": str(wheels[0])},
            check=True,
        )
        subprocess.run(
            [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
            cwd=source_root,
            env={
                **os.environ,
                "PYTHONPATH": str(source_root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=True,
        )
        print("Distribution contents, documentation links, and public-file checks passed")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
