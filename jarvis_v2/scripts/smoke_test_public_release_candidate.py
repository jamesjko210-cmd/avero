"""Focused offline tests for the history-free public candidate builder."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import unicodedata
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.scripts import public_release_candidate as candidate_module
from jarvis_v2.scripts.public_release_candidate import (
    ALLOWED_TOP_LEVEL_FILES,
    CANDIDATE_PROFILE_PUBLICATION,
    CandidateBuildError,
    FINAL_DIRECTORY_MODE,
    FINAL_EXECUTABLE_FILE_MODE,
    FINAL_REGULAR_FILE_MODE,
    MANIFEST_DIGEST_VERSION,
    PUBLIC_CANDIDATE_MARKER_KEYS,
    PUBLIC_CANDIDATE_MARKER_NAME,
    PUBLIC_CANDIDATE_MARKER_VERSION,
    _candidate_tree_has_finalized_modes,
    _parse_public_candidate_marker,
    build_public_candidate,
    main as candidate_main,
    structural_public_candidate_profile,
    valid_public_candidate_marker,
)


ROOT = Path(__file__).resolve().parents[2]
from jarvis_v2.scripts.public_release_preflight import EXTRA_DENY_ENV, PreflightConfigError


_PROJECT_DOCUMENT_REFERENCE_RE = re.compile(r"`(?P<name>[A-Z][A-Z0-9_]*\.md)`")
_EXCLUDED_REFERENCE_MARKERS = (
    "intentionally excluded",
    "not included",
    "omitted",
)
def _git(root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), *args],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode:
        raise SystemExit(f"synthetic Git fixture failed: {args!r}")


def _write(root: Path, relative: str, content: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _fixture(root: Path, *, unsafe_source: bool = False) -> Path:
    # Keep every destructive synthetic fixture inside its own temporary root
    # while exercising the production immediate-child destination rule.
    candidate_module.PRIVATE_TMP_ROOT = root
    source = root / "source"
    source.mkdir()
    _git(source, "init", "--quiet")
    _write(source, "README.md", "Safe synthetic release documentation.\n")
    _write(
        source,
        "RELEASE_NOTES.md",
        "Unreleased macOS single-owner functional preview.\n",
    )
    _write(
        source,
        "SECURITY.md",
        "Use GitHub private vulnerability reporting.\n",
    )
    _write(source, "PUBLIC_RELEASE_PRECHECK.md", "Run the offline privacy preflight.\n")
    _write(
        source,
        "V3_SUPERVISED_PROOF_RUNBOOK.md",
        "Run supervised proofs only with the operator present.\n",
    )
    _write(source, "requirements.txt", "\n")
    _write(source, ".env.example", "SYNTHETIC_TOKEN=\n")
    module_content = "VALUE = 'offline'\n"
    if unsafe_source:
        module_content += "PRIVATE = '/" + "Users/example/private'\n"
    _write(source, "jarvis_v2/__init__.py", "\n")
    _write(source, "jarvis_v2/module.py", module_content)
    _write(source, "jarvis_v2/scripts/v3_scheduler_bootstrap.py", "VALUE = 'bootstrap'\n")
    _write(
        source,
        "jarvis_v2/scripts/smoke_test_v3_scheduler_bootstrap.py",
        "VALUE = 'bootstrap-smoke'\n",
    )
    _write(source, "launch_jarvis_v3.py", "#!/usr/bin/env python3\nprint('synthetic launcher')\n")
    (source / "launch_jarvis_v3.py").chmod(0o755)
    _write(source, "CODEX_TASKS.md", "private operational history\n")
    _write(source, "com.jarvis-v3.synthetic.plist", "service template\n")
    _write(source, "state.sqlite", "runtime state\n")
    _write(source, "unreviewed.txt", "not allowlisted\n")
    _git(source, "add", ".")
    _git(
        source,
        "-c",
        "user.name=Synthetic Maintainer",
        "-c",
        "user.email=maintainer@example.com",
        "commit",
        "--quiet",
        "-m",
        "synthetic fixture",
    )
    return source


def test_run_git_uses_a_bounded_command_and_environment() -> None:
    completed = subprocess.CompletedProcess([], 0, stdout=b"bounded-output")
    with mock.patch.object(candidate_module.subprocess, "run", return_value=completed) as run:
        output = candidate_module._run_git(
            Path("/private/tmp/synthetic-repository"),
            "status",
            "--porcelain=v1",
        )
    if output != b"bounded-output":
        raise SystemExit("bounded Git wrapper lost stdout")
    expected_command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        "/private/tmp/synthetic-repository",
        "status",
        "--porcelain=v1",
    ]
    expected_environment = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    call = run.call_args
    if call.args != (expected_command,):
        raise SystemExit(f"bounded Git wrapper used unexpected argv: {call.args!r}")
    if call.kwargs != {
        "check": False,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
        "env": expected_environment,
    }:
        raise SystemExit("bounded Git wrapper used unexpected process options or environment")


def test_run_git_disables_local_fsmonitor_and_optional_index_writes() -> None:
    with TemporaryDirectory(prefix="jarvis-public-git-boundary-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        sentinel = root / "fsmonitor-invoked"
        hook = root / "fsmonitor-hook"
        _write(
            root,
            hook.name,
            f"#!/bin/sh\nprintf invoked > '{sentinel}'\nexit 0\n",
        )
        hook.chmod(0o700)
        _git(source, "config", "core.fsmonitor", os.fspath(hook))
        _git(source, "config", "core.untrackedCache", "true")

        tracked = source / "README.md"
        tracked_stat = tracked.stat()
        os.utime(
            tracked,
            ns=(tracked_stat.st_atime_ns, tracked_stat.st_mtime_ns + 1_000_000_000),
        )
        index = source / ".git" / "index"
        index_bytes_before = index.read_bytes()
        index_stat_before = index.stat()
        index_metadata_before = (
            index_stat_before.st_ino,
            index_stat_before.st_size,
            index_stat_before.st_mtime_ns,
            index_stat_before.st_ctime_ns,
        )

        output = candidate_module._run_git(
            source,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        index_stat_after = index.stat()
        index_metadata_after = (
            index_stat_after.st_ino,
            index_stat_after.st_size,
            index_stat_after.st_mtime_ns,
            index_stat_after.st_ctime_ns,
        )
        if output:
            raise SystemExit("bounded Git status misclassified an unchanged tracked file")
        if sentinel.exists():
            raise SystemExit("bounded Git status invoked a repository-local fsmonitor hook")
        if index.read_bytes() != index_bytes_before or index_metadata_after != index_metadata_before:
            raise SystemExit("bounded Git status refreshed or rewrote the repository index")


def _add_publication_metadata(
    source: Path,
    *,
    version: str = "3.0.0-rc.1",
    license_text: str = "Synthetic reviewed license terms for offline publication tests.\n",
    license_executable: bool = False,
) -> None:
    _write(source, "LICENSE", license_text)
    _write(source, "VERSION", version)
    if license_executable:
        (source / "LICENSE").chmod(0o755)
    _git(source, "add", "LICENSE", "VERSION")
    _git(
        source,
        "-c",
        "user.name=Synthetic Maintainer",
        "-c",
        "user.email=maintainer@example.com",
        "commit",
        "--quiet",
        "-m",
        "add synthetic publication metadata",
    )


def test_builder_copies_only_allowlisted_head_blobs_and_passes_preflight() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        report = build_public_candidate(
            source,
            destination,
            environ={
                EXTRA_DENY_ENV: json.dumps(
                    ["absent-owner-marker", "absent-contact-marker"]
                )
            },
        )
        if not report.ok or not report.candidate_created or report.preflight is None:
            raise SystemExit(f"safe synthetic candidate did not pass: {report}")
        if len(report.source_commit) not in {40, 64} or len(report.manifest_digest) != 64:
            raise SystemExit(f"candidate report missed source/content binding: {report}")
        expected = {
            ".env.example",
            PUBLIC_CANDIDATE_MARKER_NAME,
            "PUBLIC_RELEASE_PRECHECK.md",
            "README.md",
            "RELEASE_NOTES.md",
            "SECURITY.md",
            "V3_SUPERVISED_PROOF_RUNBOOK.md",
            "jarvis_v2/__init__.py",
            "jarvis_v2/module.py",
            "jarvis_v2/scripts/smoke_test_v3_scheduler_bootstrap.py",
            "jarvis_v2/scripts/v3_scheduler_bootstrap.py",
            "launch_jarvis_v3.py",
            "requirements.txt",
        }
        actual = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual != expected:
            raise SystemExit(f"candidate allowlist drifted: {actual}")
        if not valid_public_candidate_marker(destination):
            raise SystemExit("generated public candidate marker did not validate")
        if structural_public_candidate_profile(destination) != "preview":
            raise SystemExit("structural candidate classifier missed the preview profile")
        if not report.candidate_finalized_immutable:
            raise SystemExit("successful candidate was not reported as immutable")
        summary = report.summary_json_payload()
        expected_modes = {
            "finalized_directory_mode": "0500",
            "finalized_regular_file_mode": "0400",
            "finalized_executable_file_mode": "0500",
        }
        for key, value in expected_modes.items():
            if summary.get(key) != value:
                raise SystemExit(f"candidate summary missed final mode {key}: {summary}")
        if (
            summary.get("extra_private_deny_literals_supplied") is not True
            or summary.get("extra_private_deny_literal_count") != 2
        ):
            raise SystemExit("candidate summary lost content-free private deny-list evidence")
        if (
            summary.get("candidate_profile") != "preview"
            or summary.get("publication_requirements_checked") is not False
            or summary.get("publication_ready") is not False
            or summary.get("authorizes_publication") is not False
            or summary.get("creates_repository") is not False
            or summary.get("license_choice_made_by_tool") is not False
            or summary.get("version_choice_made_by_tool") is not False
            or summary.get("publication_date_authorized") is not False
            or summary.get("private_deny_literals_retained") is not False
        ):
            raise SystemExit("ordinary preview candidate gained publication authority")
        for path in (destination, destination / "jarvis_v2"):
            if stat.S_IMODE(path.stat().st_mode) != FINAL_DIRECTORY_MODE:
                raise SystemExit(f"candidate directory was writable: {path.name}")
        for path in destination.rglob("*"):
            if not path.is_file():
                continue
            expected_mode = (
                FINAL_EXECUTABLE_FILE_MODE
                if path.name == "launch_jarvis_v3.py"
                else FINAL_REGULAR_FILE_MODE
            )
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                raise SystemExit(f"candidate file mode drifted: {path.name}")
        launcher = destination / "launch_jarvis_v3.py"
        if "synthetic launcher" not in launcher.read_text(encoding="utf-8"):
            raise SystemExit("immutable launcher was not readable")
        launched = subprocess.run(
            [os.fspath(launcher)],
            cwd=destination,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if launched.returncode or launched.stdout.strip() != "synthetic launcher":
            raise SystemExit(f"immutable launcher was not executable: {launched}")
        imported = subprocess.run(
            ["python3", "-c", "import jarvis_v2.module; print(jarvis_v2.module.VALUE)"],
            cwd=destination,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": os.fspath(destination)},
        )
        if imported.returncode or imported.stdout.strip() != "offline":
            raise SystemExit(f"ordinary immutable-tree import failed: {imported}")
        if tuple(destination.rglob("__pycache__")) or tuple(destination.rglob("*.pyc")):
            raise SystemExit("ordinary Python import wrote bytecode into immutable candidate")
        if not valid_public_candidate_marker(destination):
            raise SystemExit("ordinary Python import invalidated immutable candidate marker")
        normalized = root / "normalized-public-checkout"
        shutil.copytree(destination, normalized)
        for path in normalized.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
            elif path.is_file():
                path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        normalized.chmod(0o755)
        if valid_public_candidate_marker(normalized):
            raise SystemExit("writable normalized checkout retained immutable candidate validity")
        marker_path = destination / PUBLIC_CANDIDATE_MARKER_NAME
        marker_path.chmod(FINAL_EXECUTABLE_FILE_MODE)
        if _candidate_tree_has_finalized_modes(destination):
            raise SystemExit("final mode validator accepted an executable marker")
        if valid_public_candidate_marker(destination):
            raise SystemExit("executable marker retained immutable candidate validity")
        marker_path.chmod(FINAL_REGULAR_FILE_MODE)
        if valid_public_candidate_marker(destination / "jarvis_v2"):
            raise SystemExit("public candidate marker helper searched outside the supplied root")
        marker_payload = json.loads((destination / PUBLIC_CANDIDATE_MARKER_NAME).read_text(encoding="utf-8"))
        if frozenset(marker_payload) != PUBLIC_CANDIDATE_MARKER_KEYS:
            raise SystemExit(f"generated public candidate marker schema drifted: {marker_payload}")
        for forbidden in (
            ".git",
            "CODEX_TASKS.md",
            "com.jarvis-v3.synthetic.plist",
            "state.sqlite",
            "unreviewed.txt",
        ):
            if (destination / forbidden).exists():
                raise SystemExit(f"candidate included forbidden tracked content: {forbidden}")


def test_marker_helper_rejects_tampering_and_extra_entries() -> None:
    def built_candidate(root: Path) -> Path:
        source = _fixture(root)
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if not report.ok or not valid_public_candidate_marker(destination):
            raise SystemExit("synthetic candidate fixture did not start valid")
        return destination

    def rewrite_file(path: Path, content: str) -> None:
        original_mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(0o600)
        path.write_text(content, encoding="utf-8")
        path.chmod(original_mode)

    def add_file(path: Path, content: str) -> None:
        parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
        path.parent.chmod(0o700)
        path.write_text(content, encoding="utf-8")
        path.chmod(FINAL_REGULAR_FILE_MODE)
        path.parent.chmod(parent_mode)

    def add_directory(path: Path) -> None:
        parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
        path.parent.chmod(0o700)
        path.mkdir(mode=FINAL_DIRECTORY_MODE)
        path.parent.chmod(parent_mode)

    def finalize_fixture_modes(root: Path) -> None:
        directories: list[Path] = []
        for current, _, file_names in os.walk(root):
            current_path = Path(current)
            directories.append(current_path)
            for name in file_names:
                target = current_path / name
                target.chmod(
                    FINAL_EXECUTABLE_FILE_MODE
                    if target.stat().st_mode & 0o111
                    else FINAL_REGULAR_FILE_MODE
                )
        for directory in reversed(directories):
            directory.chmod(FINAL_DIRECTORY_MODE)

    with TemporaryDirectory(prefix="jarvis-public-marker-schema-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        marker_path = destination / PUBLIC_CANDIDATE_MARKER_NAME
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        payload["unexpected"] = True
        rewrite_file(marker_path, json.dumps(payload))
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted an extra schema field")
        if structural_public_candidate_profile(destination) is not None:
            raise SystemExit("structural classifier accepted an extra marker field")

    with TemporaryDirectory(prefix="jarvis-public-marker-content-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        module = destination / "jarvis_v2/module.py"
        rewrite_file(module, "VALUE = 'tampered'\n")
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted source-content tampering")
        if structural_public_candidate_profile(destination) is not None:
            raise SystemExit("structural classifier accepted source-content tampering")

    with TemporaryDirectory(prefix="jarvis-public-marker-file-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        add_file(destination / "unexpected.txt", "unexpected\n")
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted an extra file")
        if structural_public_candidate_profile(destination) is not None:
            raise SystemExit("structural classifier accepted an extra file")

    with TemporaryDirectory(prefix="jarvis-public-marker-git-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        add_directory(destination / ".git")
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted an empty Git directory")

    with TemporaryDirectory(prefix="jarvis-public-marker-dir-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        add_directory(destination / "unexpected-directory")
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted an empty top-level directory")

    with TemporaryDirectory(prefix="jarvis-public-marker-nested-dir-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        add_directory(destination / "jarvis_v2/untracked-empty")
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted an empty nested directory")

    with TemporaryDirectory(prefix="jarvis-public-marker-link-", dir="/private/tmp") as temp:
        destination = built_candidate(Path(temp))
        parent = destination / "jarvis_v2"
        parent.chmod(0o700)
        (parent / "linked.py").symlink_to("module.py")
        parent.chmod(FINAL_DIRECTORY_MODE)
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted a symlink")

    with TemporaryDirectory(prefix="jarvis-public-marker-hardlink-", dir="/private/tmp") as temp:
        root = Path(temp)
        destination = built_candidate(root)
        module = destination / "jarvis_v2/module.py"
        external = root / "external.py"
        external.write_bytes(module.read_bytes())
        external.chmod(FINAL_REGULAR_FILE_MODE)
        module.parent.chmod(0o700)
        module.unlink()
        os.link(external, module)
        module.parent.chmod(FINAL_DIRECTORY_MODE)
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted a hard-linked candidate file")

    with TemporaryDirectory(prefix="jarvis-public-marker-preflight-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root, unsafe_source=True)
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if report.ok or report.preflight is None or report.preflight.ok:
            raise SystemExit("unsafe synthetic candidate did not preserve its preflight failure")
        if report.candidate_finalized_immutable:
            raise SystemExit("preflight-unsafe candidate was finalized as immutable")
        finalize_fixture_modes(destination)
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted a digest-bound but preflight-unsafe candidate")
        if structural_public_candidate_profile(destination) is not None:
            raise SystemExit("structural classifier accepted a preflight-unsafe candidate")


def test_marker_validator_requires_exact_finalized_modes_for_both_profiles() -> None:
    for profile in ("preview", CANDIDATE_PROFILE_PUBLICATION):
        with TemporaryDirectory(
            prefix=f"jarvis-public-marker-modes-{profile}-",
            dir="/private/tmp",
        ) as temp:
            root = Path(temp)
            source = _fixture(root)
            environment: dict[str, str] = {}
            if profile == CANDIDATE_PROFILE_PUBLICATION:
                _add_publication_metadata(source)
                environment = {
                    EXTRA_DENY_ENV: json.dumps(["absent-private-review-marker"])
                }
            destination = root / "candidate"
            report = build_public_candidate(
                source,
                destination,
                environ=environment,
                profile=profile,
            )

            def validates() -> bool:
                return valid_public_candidate_marker(
                    destination,
                    environ=environment,
                    profile=profile,
                )

            if not report.ok or not validates():
                raise SystemExit(f"{profile} mode fixture did not start valid")
            targets = (
                (destination, 0o700),
                (destination / "README.md", 0o600),
                (destination / "jarvis_v2", 0o700),
                (destination / "launch_jarvis_v3.py", 0o700),
            )
            for target, unsafe_mode in targets:
                original_mode = stat.S_IMODE(target.stat().st_mode)
                target.chmod(unsafe_mode)
                try:
                    if validates():
                        raise SystemExit(
                            f"{profile} validator accepted writable {target.name}"
                        )
                finally:
                    target.chmod(original_mode)
                if not validates():
                    raise SystemExit(f"{profile} validator did not recover exact modes")


def test_marker_helpers_bound_directory_and_total_enumeration() -> None:
    with TemporaryDirectory(prefix="jarvis-public-marker-entry-bounds-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if not report.ok or not valid_public_candidate_marker(destination):
            raise SystemExit("entry-bound marker fixture did not start valid")

        with mock.patch.object(candidate_module, "MAX_CANDIDATE_DIRECTORY_ENTRIES", 2):
            if valid_public_candidate_marker(destination):
                raise SystemExit("marker validator accepted an over-limit directory")
            root_fd = os.open(destination, candidate_module._OPEN_DIRECTORY_FLAGS)
            try:
                if candidate_module._candidate_tree_has_finalized_modes_fd(root_fd):
                    raise SystemExit("mode validator accepted over-limit enumeration")
                try:
                    candidate_module._finalize_candidate_immutable_fd(root_fd)
                except CandidateBuildError:
                    pass
                else:
                    raise SystemExit("finalizer accepted over-limit enumeration")
            finally:
                os.close(root_fd)

        with mock.patch.object(candidate_module, "MAX_CANDIDATE_ENTRIES", 2):
            if valid_public_candidate_marker(destination):
                raise SystemExit("marker validator accepted an over-limit total entry count")


def test_marker_binds_exact_source_commit_and_rejects_legacy_schema() -> None:
    with TemporaryDirectory(
        prefix="jarvis-public-marker-origin-binding-",
        dir="/private/tmp",
    ) as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if not report.ok or not valid_public_candidate_marker(destination):
            raise SystemExit("origin-binding fixture did not start valid")

        marker_path = destination / PUBLIC_CANDIDATE_MARKER_NAME
        original_mode = stat.S_IMODE(marker_path.stat().st_mode)
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != PUBLIC_CANDIDATE_MARKER_VERSION
            or payload.get("manifest_digest_version") != MANIFEST_DIGEST_VERSION
        ):
            raise SystemExit(f"generated marker did not use version 2 semantics: {payload}")

        for invalid_length in (41, 63):
            malformed = dict(payload)
            malformed["source_commit"] = "a" * invalid_length
            content = json.dumps(malformed, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            if _parse_public_candidate_marker(content) is not None:
                raise SystemExit(
                    f"marker parser accepted a {invalid_length}-character source commit"
                )

        legacy = dict(payload)
        legacy["version"] = 1
        legacy["manifest_digest_version"] = "1"
        legacy_content = json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        if _parse_public_candidate_marker(legacy_content) is not None:
            raise SystemExit("marker parser accepted obsolete version 1 semantics")

        changed_commit = ("a" if payload["source_commit"][0] != "a" else "b") * len(
            payload["source_commit"]
        )
        tampered = dict(payload)
        tampered["source_commit"] = changed_commit
        marker_path.chmod(0o600)
        marker_path.write_text(
            json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        marker_path.chmod(original_mode)
        if valid_public_candidate_marker(destination):
            raise SystemExit("marker helper accepted a source-commit-only edit")


def test_dry_run_is_deterministic_and_creates_nothing() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-dry-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        first = build_public_candidate(source, destination, dry_run=True)
        second = build_public_candidate(source, destination, dry_run=True)
        if first != second or not first.ok or first.candidate_created or destination.exists():
            raise SystemExit(f"candidate dry-run was not deterministic and inert: {first} / {second}")
        if first.source_commit != second.source_commit or first.manifest_digest != second.manifest_digest:
            raise SystemExit("identical HEAD content produced unstable report bindings")


def test_release_guide_preserves_exact_candidate_tree_during_verification() -> None:
    guide = " ".join((ROOT / "PUBLIC_RELEASE_PRECHECK.md").read_text(encoding="utf-8").split())
    for expected in (
        "python3 -B -m jarvis_v2.scripts.public_release_preflight",
        "python3 -B -m jarvis_v2.scripts.smoke_test_all",
        "immutable staging tree",
        "candidate_finalized_immutable:true",
        "defense-in-depth",
        "bytecode caches",
        "discard that staging directory and build a fresh one",
        "do not relax marker validation",
        "unsigned self-declared origin binding",
        "not authenticated provenance",
        "version 1 candidates are obsolete and must be rebuilt",
        "--profile publication",
        "publication_preconditions_passed:true",
        "publication_license_included:false",
        "publication_ready:true",
        "runtime-only deny list is empty",
        "builder does not choose either value",
        "does not initialize a repository",
        "authorize publication",
        "Unicode NFKC normalization plus case folding",
        "private_variant_scan_limit",
        "never prints the deny literal or matching text",
        "V3_SUPERVISED_PROOF_RUNBOOK.md",
    ):
        if expected not in guide:
            raise SystemExit(f"public release guide missed immutable-candidate guidance: {expected}")


def test_selected_public_documents_have_closed_project_document_references() -> None:
    """Keep the history-free public guide usable without private operational files."""

    selected_documents = tuple(
        sorted(name for name in ALLOWED_TOP_LEVEL_FILES if name.endswith(".md"))
    )
    for document_name in selected_documents:
        document = ROOT / document_name
        text = document.read_text(encoding="utf-8")
        for match in _PROJECT_DOCUMENT_REFERENCE_RE.finditer(text):
            referenced_name = match.group("name")
            if (
                referenced_name in ALLOWED_TOP_LEVEL_FILES
                and (ROOT / referenced_name).is_file()
            ):
                continue
            paragraph_start = text.rfind("\n\n", 0, match.start())
            paragraph_end = text.find("\n\n", match.end())
            start_index = 0 if paragraph_start < 0 else paragraph_start + 2
            end_index = len(text) if paragraph_end < 0 else paragraph_end
            paragraph = text[start_index:end_index].lower()
            if "private" not in paragraph or not any(
                marker in paragraph for marker in _EXCLUDED_REFERENCE_MARKERS
            ):
                raise SystemExit(
                    "selected public document has an unresolved project-document "
                    f"reference without an explicit private/excluded label: "
                    f"{document_name} -> {referenced_name}"
                )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    development = (ROOT / "V3_DEVELOPMENT.md").read_text(encoding="utf-8")
    for document_name, text in (("README.md", readme), ("V3_DEVELOPMENT.md", development)):
        for authority in ("`QUICKSTART.md`", "`CAPABILITIES.md`"):
            if authority not in text:
                raise SystemExit(
                    f"{document_name} missed self-contained public authority {authority}"
                )


def test_public_security_and_release_notes_are_sanitized_and_truthful() -> None:
    required_documents = {"SECURITY.md", "RELEASE_NOTES.md"}
    if not required_documents <= ALLOWED_TOP_LEVEL_FILES:
        raise SystemExit("public hygiene documents are not both allowlisted")

    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    security_flat = " ".join(security.split())
    release_notes_flat = " ".join(release_notes.split())
    for phrase in (
        "GitHub's private vulnerability reporting",
        "Do not open a public issue",
        "single-owner functional preview",
        "no response-time",
    ):
        if phrase not in security_flat:
            raise SystemExit(f"security policy missed required boundary: {phrase}")
    for phrase in (
        "4.0.0-rc.1",
        "GitHub prerelease",
        "MIT License",
        "macOS",
        "single trusted local owner",
        "not enabled by the source package",
        "never retry automatically",
        "does not activate services",
    ):
        if phrase not in release_notes_flat:
            raise SystemExit(f"release notes missed required preview limitation: {phrase}")

    combined = security + "\n" + release_notes
    forbidden_patterns = (
        re.compile("/" + "Users" + "/", re.IGNORECASE),
        re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_.-]+"),
        re.compile(r"\b(?:proof|approval|receipt)\s*#?\d+\b", re.IGNORECASE),
        re.compile(r"\b[0-9a-f]{40,64}\b", re.IGNORECASE),
        re.compile(r"\b(?:[A-Za-z0-9._%+-]+)@(?:[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),
    )
    if any(pattern.search(combined) for pattern in forbidden_patterns):
        raise SystemExit("public hygiene documents contain private operational identifiers")


def test_manifest_digest_binds_source_commit_and_selected_content() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-digest-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        before = build_public_candidate(source, destination, dry_run=True)

        _write(source, "unreviewed.txt", "changed excluded documentation\n")
        _git(source, "add", "unreviewed.txt")
        _git(
            source,
            "-c",
            "user.name=Synthetic Maintainer",
            "-c",
            "user.email=maintainer@example.com",
            "commit",
            "--quiet",
            "-m",
            "change excluded documentation",
        )
        excluded_change = build_public_candidate(source, destination, dry_run=True)
        if before.source_commit == excluded_change.source_commit:
            raise SystemExit("excluded-only commit did not change the source commit")
        if (
            before.files_selected != excluded_change.files_selected
            or before.bytes_selected != excluded_change.bytes_selected
        ):
            raise SystemExit("excluded-only commit changed the selected source payload")
        if before.manifest_digest == excluded_change.manifest_digest:
            raise SystemExit("version 2 manifest did not bind the changed source commit")

        _write(source, "jarvis_v2/module.py", "VALUE = 'changed offline content'\n")
        _git(source, "add", "jarvis_v2/module.py")
        _git(
            source,
            "-c",
            "user.name=Synthetic Maintainer",
            "-c",
            "user.email=maintainer@example.com",
            "commit",
            "--quiet",
            "-m",
            "change selected content",
        )
        after = build_public_candidate(source, destination, dry_run=True)
        repeated = build_public_candidate(source, destination, dry_run=True)
        if excluded_change.source_commit == after.source_commit:
            raise SystemExit("source commit binding did not change after a commit")
        if excluded_change.manifest_digest == after.manifest_digest:
            raise SystemExit("manifest digest did not change with selected content")
        if after.manifest_digest != repeated.manifest_digest:
            raise SystemExit("changed selected content produced an unstable manifest digest")


def test_invalid_hidden_deny_configuration_creates_nothing() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-deny-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        try:
            build_public_candidate(
                source,
                destination,
                environ={EXTRA_DENY_ENV: '{"not":"a-list"}'},
            )
        except PreflightConfigError:
            pass
        else:
            raise SystemExit("invalid hidden deny-list configuration was accepted")
        if destination.exists():
            raise SystemExit("invalid hidden deny-list configuration created a candidate")


def test_publication_profile_requires_opt_in_license_version_and_private_review() -> None:
    deny_environment = {EXTRA_DENY_ENV: json.dumps(["absent-private-review-marker"])}

    with TemporaryDirectory(prefix="jarvis-public-profile-license-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        try:
            build_public_candidate(
                source,
                destination,
                dry_run=True,
                environ=deny_environment,
                profile=CANDIDATE_PROFILE_PUBLICATION,
            )
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("publication profile accepted a missing LICENSE/VERSION")
        if destination.exists():
            raise SystemExit("missing publication metadata created a candidate")

    for version in (
        "3.0.0-dev.1",
        "3.0.0-SNAPSHOT",
        "v3.0.0",
        "3.0",
        "3.0.0+local",
        "03.0.0",
        "3.0.0-rc.01",
        "3.0.0\nextra",
        "3." + ("0" * 200),
    ):
        with TemporaryDirectory(
            prefix="jarvis-public-profile-version-", dir="/private/tmp"
        ) as temp:
            root = Path(temp)
            source = _fixture(root)
            _add_publication_metadata(source, version=version)
            destination = root / "candidate"
            try:
                build_public_candidate(
                    source,
                    destination,
                    dry_run=True,
                    environ=deny_environment,
                    profile=CANDIDATE_PROFILE_PUBLICATION,
                )
            except CandidateBuildError:
                pass
            else:
                raise SystemExit(f"publication profile accepted unsafe version class: {version!r}")
            if destination.exists():
                raise SystemExit("invalid publication version created a candidate")

    for license_text, executable in (
        ("short\n", False),
        ("valid text\x00hidden\n", False),
        ("Synthetic reviewed license\u202eterms with hidden direction.\n", False),
        ("L" * (candidate_module.MAX_PUBLICATION_LICENSE_BYTES + 1), False),
        ("Synthetic reviewed executable license terms for testing.\n", True),
    ):
        with TemporaryDirectory(
            prefix="jarvis-public-profile-license-invalid-", dir="/private/tmp"
        ) as temp:
            root = Path(temp)
            source = _fixture(root)
            _add_publication_metadata(
                source,
                license_text=license_text,
                license_executable=executable,
            )
            destination = root / "candidate"
            try:
                build_public_candidate(
                    source,
                    destination,
                    dry_run=True,
                    environ=deny_environment,
                    profile=CANDIDATE_PROFILE_PUBLICATION,
                )
            except CandidateBuildError:
                pass
            else:
                raise SystemExit("publication profile accepted an unsafe LICENSE artifact")
            if destination.exists():
                raise SystemExit("unsafe publication license created a candidate")

    with TemporaryDirectory(prefix="jarvis-public-profile-deny-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        _add_publication_metadata(source)
        destination = root / "candidate"
        try:
            build_public_candidate(
                source,
                destination,
                dry_run=True,
                environ={},
                profile=CANDIDATE_PROFILE_PUBLICATION,
            )
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("publication profile accepted an empty private deny-list")
        if destination.exists():
            raise SystemExit("missing private publication review created a candidate")


def test_default_preview_excludes_even_an_unsafe_tracked_license() -> None:
    with TemporaryDirectory(prefix="jarvis-preview-license-boundary-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        _add_publication_metadata(
            source,
            version="3.0.0-dev.1",
            license_text="Synthetic unsafe license\u202ewith hidden direction.\n",
            license_executable=True,
        )
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if not report.ok or not valid_public_candidate_marker(destination):
            raise SystemExit("ordinary preview behavior changed after a tracked LICENSE appeared")
        if (destination / "LICENSE").exists():
            raise SystemExit("default preview selected the publication-only LICENSE artifact")
        if not (destination / "VERSION").is_file():
            raise SystemExit("default preview unexpectedly dropped its existing VERSION allowlist")
        summary = report.summary_json_payload()
        if (
            summary.get("candidate_profile") != "preview"
            or summary.get("publication_license_validated") is not False
            or summary.get("publication_license_included") is not False
            or summary.get("publication_ready") is not False
        ):
            raise SystemExit("default preview overstated publication metadata")


def test_publication_profile_build_and_revalidation_are_fail_closed() -> None:
    for version, expected_kind in (("3.0.0", "release"), ("3.0.0-beta.2", "prerelease")):
        with TemporaryDirectory(
            prefix="jarvis-public-profile-version-pass-", dir="/private/tmp"
        ) as temp:
            root = Path(temp)
            source = _fixture(root)
            _add_publication_metadata(source, version=version)
            report = build_public_candidate(
                source,
                root / "candidate",
                dry_run=True,
                environ={EXTRA_DENY_ENV: json.dumps(["absent-private-review-marker"])},
                profile=CANDIDATE_PROFILE_PUBLICATION,
            )
            summary = report.summary_json_payload()
            if (
                summary.get("publication_version_kind") != expected_kind
                or summary.get("publication_preconditions_passed") is not True
                or summary.get("publication_ready") is not False
            ):
                raise SystemExit(f"valid publication version was misclassified: {summary}")

    with TemporaryDirectory(prefix="jarvis-public-profile-pass-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        _add_publication_metadata(source)
        destination = root / "candidate"
        private_marker = "absent-runtime-only-private-marker"
        environment = {EXTRA_DENY_ENV: json.dumps([private_marker])}
        dry_run = build_public_candidate(
            source,
            destination,
            dry_run=True,
            environ=environment,
            profile=CANDIDATE_PROFILE_PUBLICATION,
        )
        dry_summary = dry_run.summary_json_payload()
        if (
            not dry_run.ok
            or dry_summary.get("publication_preconditions_passed") is not True
            or dry_summary.get("publication_ready") is not False
            or dry_summary.get("publication_version_kind") != "prerelease"
            or dry_summary.get("publication_license_validated") is not True
            or dry_summary.get("publication_license_included") is not False
        ):
            raise SystemExit(f"publication dry-run overstated or lost readiness: {dry_summary}")

        report = build_public_candidate(
            source,
            destination,
            environ=environment,
            profile=CANDIDATE_PROFILE_PUBLICATION,
        )
        summary = report.summary_json_payload()
        if (
            not report.ok
            or summary.get("candidate_profile") != "publication"
            or summary.get("publication_requirements_checked") is not True
            or summary.get("publication_preconditions_passed") is not True
            or summary.get("publication_ready") is not True
            or summary.get("publication_license_validated") is not True
            or summary.get("publication_license_included") is not True
            or summary.get("extra_private_deny_literal_count") != 1
            or summary.get("private_deny_literals_retained") is not False
            or any(
                summary.get(key) is not False
                for key in (
                    "publishes",
                    "authorizes_publication",
                    "creates_repository",
                    "license_choice_made_by_tool",
                    "version_choice_made_by_tool",
                    "publication_date_authorized",
                )
            )
        ):
            raise SystemExit(f"publication report boundary drifted: {summary}")
        if not (destination / "LICENSE").is_file() or not (destination / "VERSION").is_file():
            raise SystemExit("publication candidate omitted reviewed metadata")
        if private_marker in "\n".join(
            path.read_text(encoding="utf-8")
            for path in destination.rglob("*")
            if path.is_file()
        ):
            raise SystemExit("runtime-only private deny literal was retained")
        if valid_public_candidate_marker(
            destination,
            profile=CANDIDATE_PROFILE_PUBLICATION,
        ):
            raise SystemExit("publication validator accepted a missing runtime deny-list")
        if not valid_public_candidate_marker(
            destination,
            environ=environment,
            profile=CANDIDATE_PROFILE_PUBLICATION,
        ):
            raise SystemExit("publication validator rejected exact reviewed evidence")
        if valid_public_candidate_marker(destination):
            raise SystemExit("default preview validator accepted publication-only content")
        if structural_public_candidate_profile(destination) != CANDIDATE_PROFILE_PUBLICATION:
            raise SystemExit("structural classifier missed a valid publication candidate")
        license_path = destination / "LICENSE"
        license_path.chmod(0o600)
        try:
            if structural_public_candidate_profile(destination) is not None:
                raise SystemExit("structural classifier accepted candidate mode tampering")
        finally:
            license_path.chmod(FINAL_REGULAR_FILE_MODE)


def test_publication_cli_is_explicit_and_errors_are_bounded() -> None:
    with TemporaryDirectory(prefix="jarvis-public-profile-cli-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        private_invalid_profile = "private-profile/should-not-echo"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = candidate_main(
                [
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                    "--profile",
                    private_invalid_profile,
                    "--dry-run",
                ]
            )
        rendered = output.getvalue()
        if code != 2 or rendered != "ERROR public-release-candidate configuration or source state is unsafe\n":
            raise SystemExit(f"invalid profile refusal was not bounded: {code} {rendered!r}")
        if private_invalid_profile in rendered or destination.exists():
            raise SystemExit("invalid profile was echoed or created a destination")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = candidate_main(
                [
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                    "--profile",
                    "publication",
                    "--dry-run",
                ]
            )
        rendered = output.getvalue()
        if code != 2 or rendered != "ERROR public-release-candidate configuration or source state is unsafe\n":
            raise SystemExit(f"publication CLI refusal was not bounded: {code} {rendered!r}")
        if destination.exists() or str(source) in rendered or str(destination) in rendered:
            raise SystemExit("publication CLI refusal leaked a path or created output")

        default_destination = root / "preview-candidate"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = candidate_main(
                [
                    "--source",
                    str(source),
                    "--destination",
                    str(default_destination),
                    "--dry-run",
                ]
            )
        payload = json.loads(output.getvalue())
        if code != 0 or payload.get("candidate_profile") != "preview":
            raise SystemExit("default CLI no longer preserved preview behavior")


def test_private_deny_scans_selected_modules_and_top_level_launchers() -> None:
    with TemporaryDirectory(
        prefix="jarvis-public-candidate-private-deny-",
        dir="/private/tmp",
    ) as temp:
        root = Path(temp)
        source = _fixture(root)
        selected_paths = (
            "jarvis_v2/module.py",
            "launch_jarvis_v3.py",
            "launch_jarvis_v3_calendar_auth.py",
            "launch_jarvis_v3_chat.py",
            "launch_jarvis_v3_dashboard.py",
            "launch_jarvis_v3_voice.py",
        )
        hangul_nfc = "가나다"
        hangul_nfd = unicodedata.normalize("NFD", hangul_nfc)
        deny_and_content = (
            ("SyntheticPrivateLatin", "sYnThEtIcPrIvAtElAtIn"),
            (hangul_nfc, hangul_nfd),
            (hangul_nfd, hangul_nfc),
            ("synthetic-private-deny-marker-4", "synthetic-private-deny-marker-4"),
            ("synthetic-private-deny-marker-5", "synthetic-private-deny-marker-5"),
            ("synthetic-private-deny-marker-6", "synthetic-private-deny-marker-6"),
        )
        deny_literals: list[str] = []
        private_variants: list[str] = []
        for relative, (marker, content_variant) in zip(selected_paths, deny_and_content):
            deny_literals.append(marker)
            private_variants.append(content_variant)
            path = source / relative
            existing = (
                path.read_text(encoding="utf-8")
                if path.exists()
                else "#!/usr/bin/env python3\n"
            )
            _write(
                source,
                relative,
                existing + f"SYNTHETIC_PRIVATE = {content_variant!r}\n",
            )
            if relative.startswith("launch_"):
                path.chmod(0o755)
        _git(source, "add", *selected_paths)
        _git(
            source,
            "-c",
            "user.name=Synthetic Maintainer",
            "-c",
            "user.email=maintainer@example.com",
            "commit",
            "--quiet",
            "-m",
            "add synthetic private deny fixtures",
        )
        destination = root / "candidate"
        report = build_public_candidate(
            source,
            destination,
            environ={EXTRA_DENY_ENV: json.dumps(deny_literals)},
        )
        if report.ok or report.preflight is None or report.preflight.ok:
            raise SystemExit("private deny literals did not fail the candidate build")
        rendered_report = json.dumps(
            report.summary_json_payload(), ensure_ascii=False, sort_keys=True
        )
        if any(value in rendered_report for value in (*deny_literals, *private_variants)):
            raise SystemExit("private deny literal or Unicode/case variant leaked in summary")
        denied_paths = {
            finding.path
            for finding in report.preflight.findings
            if finding.code == "extra_private_literal"
        }
        if denied_paths != set(selected_paths):
            raise SystemExit(
                "private deny scan missed selected module or top-level launcher blobs: "
                f"{denied_paths}"
            )
        summary = report.summary_json_payload()
        if (
            summary.get("extra_private_deny_literals_supplied") is not True
            or summary.get("extra_private_deny_literal_count") != len(deny_literals)
        ):
            raise SystemExit("failed candidate summary lost private deny-list evidence")
        if not destination.is_dir() or any(
            not (destination / relative).is_file() for relative in selected_paths
        ):
            raise SystemExit("private-deny failure did not preserve the review candidate")
        if (destination / PUBLIC_CANDIDATE_MARKER_NAME).exists():
            raise SystemExit("private-deny failure left a public candidate marker")
        if valid_public_candidate_marker(destination, environ={}):
            raise SystemExit("private-deny failure became valid after deny configuration removal")


def test_builder_refuses_dirty_source_existing_or_unsafe_destination() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-refuse-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        _write(source, "untracked.txt", "dirty\n")
        destination = root / "candidate"
        try:
            build_public_candidate(source, destination)
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("dirty source was accepted")
        if destination.exists():
            raise SystemExit("dirty-source refusal created a candidate")

        (source / "untracked.txt").unlink()
        destination.mkdir()
        try:
            build_public_candidate(source, destination)
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("existing destination was accepted")

        outside = Path("/var/tmp/jarvis-public-candidate-outside")
        try:
            build_public_candidate(source, outside)
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("destination outside private tmp boundary was accepted")

        nested = root / "nested" / "candidate"
        nested.parent.mkdir()
        try:
            build_public_candidate(source, nested, dry_run=True)
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("nested private-tmp destination was accepted")


def test_atomic_destination_creation_refuses_race_without_redirecting_writes() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-destination-race-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        outside = root / "outside"
        outside.mkdir()
        real_mkdir = os.mkdir
        raced = False

        def race_mkdir(path, mode=0o777, *, dir_fd=None):
            nonlocal raced
            if path == destination.name and dir_fd is not None and not raced:
                raced = True
                os.symlink(outside, destination.name, dir_fd=dir_fd)
            return real_mkdir(path, mode, dir_fd=dir_fd)

        with mock.patch.object(candidate_module.os, "mkdir", side_effect=race_mkdir):
            try:
                build_public_candidate(source, destination)
            except CandidateBuildError:
                pass
            else:
                raise SystemExit("destination creation race was accepted")
        if tuple(outside.iterdir()):
            raise SystemExit("destination creation race redirected candidate writes")


def test_validator_rejects_tree_changed_during_preflight() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-aba-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if not report.ok:
            raise SystemExit("A-B-A fixture candidate did not start valid")
        module = destination / "jarvis_v2/module.py"
        real_scan = candidate_module._scan_release_tree_fd
        changed = False

        def mutate_after_scan(*args, **kwargs):
            nonlocal changed
            result = real_scan(*args, **kwargs)
            if not changed:
                changed = True
                module.parent.chmod(0o700)
                module.chmod(0o600)
                module.write_bytes(b"VALUE = 'changed-during-preflight'\n")
                module.chmod(FINAL_REGULAR_FILE_MODE)
                module.parent.chmod(FINAL_DIRECTORY_MODE)
            return result

        with mock.patch.object(
            candidate_module,
            "_scan_release_tree_fd",
            side_effect=mutate_after_scan,
        ):
            if valid_public_candidate_marker(destination):
                raise SystemExit("A-B-A validation accepted a tree changed during preflight")

    with TemporaryDirectory(
        prefix="jarvis-public-candidate-structural-aba-",
        dir="/private/tmp",
    ) as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        report = build_public_candidate(source, destination)
        if not report.ok:
            raise SystemExit("structural A-B-A fixture candidate did not start valid")
        module = destination / "jarvis_v2/module.py"
        real_scan = candidate_module._scan_release_tree_fd
        changed = False

        def mutate_during_structural_scan(*args, **kwargs):
            nonlocal changed
            result = real_scan(*args, **kwargs)
            if not changed:
                changed = True
                module.parent.chmod(0o700)
                module.chmod(0o600)
                module.write_bytes(b"VALUE = 'changed-during-structural-preflight'\n")
                module.chmod(FINAL_REGULAR_FILE_MODE)
                module.parent.chmod(FINAL_DIRECTORY_MODE)
            return result

        with mock.patch.object(
            candidate_module,
            "_scan_release_tree_fd",
            side_effect=mutate_during_structural_scan,
        ):
            if structural_public_candidate_profile(destination) is not None:
                raise SystemExit(
                    "structural classification accepted a tree changed during preflight"
                )


def test_builder_aba_mismatch_is_a_configuration_failure_not_success() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-builder-aba-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        destination = root / "candidate"
        module = None
        real_scan = candidate_module._scan_release_tree_fd
        scan_count = 0

        def mutate_after_bound_preflight(*args, **kwargs):
            nonlocal module, scan_count
            result = real_scan(*args, **kwargs)
            scan_count += 1
            if scan_count == 2:
                module = destination / "jarvis_v2/module.py"
                module.write_bytes(b"VALUE = 'changed-during-builder-preflight'\n")
            return result

        output = io.StringIO()
        with mock.patch.object(
            candidate_module,
            "_scan_release_tree_fd",
            side_effect=mutate_after_bound_preflight,
        ), contextlib.redirect_stdout(output):
            code = candidate_main(
                ["--source", str(source), "--destination", str(destination)]
            )
        if code != 2 or "ERROR" not in output.getvalue():
            raise SystemExit(
                f"builder A-B-A mismatch returned a successful/reportable result: {code} {output.getvalue()!r}"
            )
        if (destination / PUBLIC_CANDIDATE_MARKER_NAME).exists():
            raise SystemExit("builder A-B-A mismatch left a reusable candidate marker")


def test_open_candidate_root_closes_fd_when_post_open_validation_raises() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-open-close-") as temp:
        root = Path(temp)
        real_close = os.close
        closed: list[int] = []

        def record_close(fd: int) -> None:
            closed.append(fd)
            real_close(fd)

        with mock.patch.object(
            candidate_module.os,
            "stat",
            side_effect=OSError("synthetic post-open stat failure"),
        ), mock.patch.object(candidate_module.os, "close", side_effect=record_close):
            opened = candidate_module._open_candidate_root(root)
        if opened is not None or len(closed) != 1:
            raise SystemExit(f"candidate root post-open failure leaked its fd: {closed}")
        try:
            os.fstat(closed[0])
        except OSError:
            pass
        else:
            raise SystemExit("candidate root post-open failure left its fd open")


def test_allowed_symlink_is_refused_before_destination_creation() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-link-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root)
        link = source / "jarvis_v2" / "linked.py"
        link.symlink_to("module.py")
        _git(source, "add", "jarvis_v2/linked.py")
        _git(
            source,
            "-c",
            "user.name=Synthetic Maintainer",
            "-c",
            "user.email=maintainer@example.com",
            "commit",
            "--quiet",
            "-m",
            "tracked symlink",
        )
        destination = root / "candidate"
        try:
            build_public_candidate(source, destination)
        except CandidateBuildError:
            pass
        else:
            raise SystemExit("tracked symlink in the source allowlist was copied")
        if destination.exists():
            raise SystemExit("symlink refusal created a candidate")


def test_preflight_failure_is_content_free_and_never_creates_history() -> None:
    with TemporaryDirectory(prefix="jarvis-public-candidate-fail-", dir="/private/tmp") as temp:
        root = Path(temp)
        source = _fixture(root, unsafe_source=True)
        destination = root / "candidate"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = candidate_main(
                ["--source", str(source), "--destination", str(destination)]
            )
        payload = json.loads(output.getvalue())
        if code != 1 or payload.get("preflight_ok") is not False:
            raise SystemExit(f"builder did not preserve the preflight failure: {payload}")
        if payload.get("paths_included") is not False or str(destination) in output.getvalue():
            raise SystemExit("candidate summary leaked a path")
        if (
            len(str(payload.get("source_commit") or "")) not in {40, 64}
            or len(str(payload.get("manifest_digest") or "")) != 64
            or str(source) in output.getvalue()
        ):
            raise SystemExit("candidate summary missed path-free source/content bindings")
        if (destination / ".git").exists():
            raise SystemExit("candidate builder created repository history")
        if not (destination / "jarvis_v2/module.py").exists():
            raise SystemExit("failed candidate was not retained for private review")
        if payload.get("candidate_finalized_immutable") is not False:
            raise SystemExit("failed candidate summary claimed immutable finalization")
        for path in (destination, destination / "jarvis_v2"):
            if stat.S_IMODE(path.stat().st_mode) != 0o700:
                raise SystemExit("failed candidate directory was not owner-only reviewable")
        if stat.S_IMODE((destination / "jarvis_v2/module.py").stat().st_mode) != 0o600:
            raise SystemExit("failed candidate file was not owner-only reviewable")


def main() -> None:
    test_run_git_uses_a_bounded_command_and_environment()
    test_run_git_disables_local_fsmonitor_and_optional_index_writes()
    test_builder_copies_only_allowlisted_head_blobs_and_passes_preflight()
    test_marker_helper_rejects_tampering_and_extra_entries()
    test_marker_validator_requires_exact_finalized_modes_for_both_profiles()
    test_marker_helpers_bound_directory_and_total_enumeration()
    test_marker_binds_exact_source_commit_and_rejects_legacy_schema()
    test_dry_run_is_deterministic_and_creates_nothing()
    test_release_guide_preserves_exact_candidate_tree_during_verification()
    test_selected_public_documents_have_closed_project_document_references()
    test_public_security_and_release_notes_are_sanitized_and_truthful()
    test_manifest_digest_binds_source_commit_and_selected_content()
    test_invalid_hidden_deny_configuration_creates_nothing()
    test_publication_profile_requires_opt_in_license_version_and_private_review()
    test_default_preview_excludes_even_an_unsafe_tracked_license()
    test_publication_profile_build_and_revalidation_are_fail_closed()
    test_publication_cli_is_explicit_and_errors_are_bounded()
    test_private_deny_scans_selected_modules_and_top_level_launchers()
    test_builder_refuses_dirty_source_existing_or_unsafe_destination()
    test_atomic_destination_creation_refuses_race_without_redirecting_writes()
    test_validator_rejects_tree_changed_during_preflight()
    test_builder_aba_mismatch_is_a_configuration_failure_not_success()
    test_open_candidate_root_closes_fd_when_post_open_validation_raises()
    test_allowed_symlink_is_refused_before_destination_creation()
    test_preflight_failure_is_content_free_and_never_creates_history()
    print("public release candidate smoke passed")


if __name__ == "__main__":
    main()
