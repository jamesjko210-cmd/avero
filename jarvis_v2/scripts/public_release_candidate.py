"""Build a history-free, source-only public candidate from a clean Git HEAD.

The builder is deliberately narrow: it reads regular blobs from the current
commit, copies only an explicit source/document allowlist into a new directory
under ``/private/tmp``, and runs the existing privacy preflight.  It never
initializes Git, publishes, overwrites, deletes, or reads runtime state.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import unicodedata
from typing import Iterable

from jarvis_v2.scripts.public_release_preflight import (
    FORBIDDEN_DIRECTORY_NAMES,
    PreflightConfigError,
    ReleasePreflightReport,
    _OPEN_DIRECTORY_FLAGS,
    _OPEN_FILE_FLAGS,
    _StableTreeError,
    _open_directory_at,
    _read_regular_at,
    _same_entry,
    _scan_release_tree_fd,
    extra_deny_literals_from_env,
)


PRIVATE_TMP_ROOT = Path("/private/tmp")
MAX_CANDIDATE_FILES = 2_000
MAX_CANDIDATE_BYTES = 64 * 1024 * 1024
MAX_CANDIDATE_ENTRIES = 4_096
MAX_CANDIDATE_DIRECTORY_ENTRIES = 2_048
MAX_CANDIDATE_TREE_FILES = MAX_CANDIDATE_FILES + 1
DESTINATION_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
SOURCE_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
PUBLICATION_VERSION_RE = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?P<prerelease>(?:alpha|beta|rc)(?:\.(?:0|[1-9][0-9]*))?))?"
)
MANIFEST_DIGEST_VERSION = "2"
PUBLIC_CANDIDATE_MARKER_NAME = "PUBLIC_CANDIDATE_MANIFEST.json"
PUBLIC_CANDIDATE_MARKER_SCHEMA = "jarvis-public-source-candidate"
PUBLIC_CANDIDATE_MARKER_VERSION = 2
MAX_PUBLIC_CANDIDATE_MARKER_BYTES = 2_048
REVIEW_DIRECTORY_MODE = 0o700
REVIEW_REGULAR_FILE_MODE = 0o600
REVIEW_EXECUTABLE_FILE_MODE = 0o700
FINAL_DIRECTORY_MODE = 0o500
FINAL_REGULAR_FILE_MODE = 0o400
FINAL_EXECUTABLE_FILE_MODE = 0o500
CANDIDATE_PROFILE_PREVIEW = "preview"
CANDIDATE_PROFILE_PUBLICATION = "publication"
CANDIDATE_PROFILES = frozenset(
    {CANDIDATE_PROFILE_PREVIEW, CANDIDATE_PROFILE_PUBLICATION}
)
PUBLICATION_LICENSE_PATH = "LICENSE"
PUBLICATION_VERSION_PATH = "VERSION"
MAX_PUBLICATION_LICENSE_BYTES = 128 * 1024
MAX_PUBLICATION_VERSION_BYTES = 128
PUBLIC_CANDIDATE_MARKER_KEYS = frozenset(
    {
        "schema",
        "version",
        "source_commit",
        "manifest_digest",
        "manifest_digest_algorithm",
        "manifest_digest_version",
        "history_included",
        "service_plists_included",
    }
)

ALLOWED_TOP_LEVEL_FILES = frozenset(
    {
        ".env.example",
        ".gitignore",
        "CAPABILITIES.md",
        "MIGRATION_MANIFEST.md",
        "PUBLIC_RELEASE_PRECHECK.md",
        "QUICKSTART.md",
        "README.md",
        "RELEASE_NOTES.md",
        "SECURITY.md",
        "V3_SUPERVISED_PROOF_RUNBOOK.md",
        "V3_DEVELOPMENT.md",
        "VERSION",
        "launch_jarvis_v3.py",
        "launch_jarvis_v3_calendar_auth.py",
        "launch_jarvis_v3_chat.py",
        "launch_jarvis_v3_dashboard.py",
        "launch_jarvis_v3_voice.py",
        "requirements.txt",
    }
)
PUBLICATION_ALLOWED_TOP_LEVEL_FILES = ALLOWED_TOP_LEVEL_FILES | frozenset(
    {PUBLICATION_LICENSE_PATH}
)


class CandidateBuildError(ValueError):
    """Raised when a candidate cannot be built without weakening boundaries."""


@dataclass(frozen=True)
class CandidateBlob:
    path: str
    oid: str
    executable: bool
    content: bytes


@dataclass(frozen=True)
class CandidateBuildReport:
    ok: bool
    dry_run: bool
    candidate_created: bool
    files_selected: int
    bytes_selected: int
    source_commit: str
    manifest_digest: str
    preflight: ReleasePreflightReport | None
    candidate_finalized_immutable: bool
    extra_deny_literal_count: int = 0
    profile: str = CANDIDATE_PROFILE_PREVIEW
    publication_license_validated: bool = False
    publication_version_kind: str | None = None

    def summary_json_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "candidate_created": self.candidate_created,
            "files_selected": self.files_selected,
            "bytes_selected": self.bytes_selected,
            "source_commit": self.source_commit,
            "manifest_digest": self.manifest_digest,
            "manifest_digest_algorithm": "sha256",
            "manifest_digest_version": MANIFEST_DIGEST_VERSION,
            "history_included": False,
            "initializes_git": False,
            "publishes": False,
            "authorizes_publication": False,
            "creates_repository": False,
            "license_choice_made_by_tool": False,
            "version_choice_made_by_tool": False,
            "publication_date_authorized": False,
            "paths_included": False,
            "candidate_profile": self.profile,
            "publication_requirements_checked": (
                self.profile == CANDIDATE_PROFILE_PUBLICATION
            ),
            "publication_license_validated": self.publication_license_validated,
            "publication_license_included": (
                self.publication_license_validated
                and self.candidate_created
                and self.preflight is not None
                and self.preflight.ok
                and self.candidate_finalized_immutable
            ),
            "publication_version_kind": self.publication_version_kind,
            "private_deny_literals_retained": False,
            "publication_preconditions_passed": (
                self.profile == CANDIDATE_PROFILE_PUBLICATION
                and self.publication_license_validated
                and self.publication_version_kind in {"prerelease", "release"}
                and self.extra_deny_literal_count > 0
            ),
            "publication_ready": (
                self.profile == CANDIDATE_PROFILE_PUBLICATION
                and self.publication_license_validated
                and self.publication_version_kind in {"prerelease", "release"}
                and self.extra_deny_literal_count > 0
                and self.candidate_created
                and self.preflight is not None
                and self.preflight.ok
                and self.candidate_finalized_immutable
            ),
            "candidate_finalized_immutable": self.candidate_finalized_immutable,
            "extra_private_deny_literals_supplied": self.extra_deny_literal_count > 0,
            "extra_private_deny_literal_count": self.extra_deny_literal_count,
            "finalized_directory_mode": (
                f"{FINAL_DIRECTORY_MODE:04o}" if self.candidate_finalized_immutable else None
            ),
            "finalized_regular_file_mode": (
                f"{FINAL_REGULAR_FILE_MODE:04o}"
                if self.candidate_finalized_immutable
                else None
            ),
            "finalized_executable_file_mode": (
                f"{FINAL_EXECUTABLE_FILE_MODE:04o}"
                if self.candidate_finalized_immutable
                else None
            ),
        }
        if self.preflight is None:
            payload["preflight_run"] = False
            payload["preflight_ok"] = None
            payload["preflight_finding_count"] = None
        else:
            summary = self.preflight.summary_json_payload()
            payload["preflight_run"] = True
            payload["preflight_ok"] = self.preflight.ok
            payload["preflight_finding_count"] = summary["finding_count"]
            payload["preflight_finding_counts_by_code"] = summary[
                "finding_counts_by_code"
            ]
        return payload


@dataclass(frozen=True)
class PublicCandidateMarker:
    source_commit: str
    manifest_digest: str


@dataclass(frozen=True)
class _CandidateState:
    marker_content: bytes
    marker: PublicCandidateMarker
    blobs: tuple[CandidateBlob, ...]


def _public_candidate_marker_payload(
    source_commit: str,
    manifest_digest: str,
) -> dict[str, object]:
    return {
        "schema": PUBLIC_CANDIDATE_MARKER_SCHEMA,
        "version": PUBLIC_CANDIDATE_MARKER_VERSION,
        "source_commit": source_commit,
        "manifest_digest": manifest_digest,
        "manifest_digest_algorithm": "sha256",
        "manifest_digest_version": MANIFEST_DIGEST_VERSION,
        "history_included": False,
        "service_plists_included": False,
    }


def _parse_public_candidate_marker(content: bytes) -> PublicCandidateMarker | None:
    if not content or len(content) > MAX_PUBLIC_CANDIDATE_MARKER_BYTES or b"\x00" in content:
        return None

    duplicate_key = False

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate_key
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key = True
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, ValueError):
        return None
    if duplicate_key or type(payload) is not dict or frozenset(payload) != PUBLIC_CANDIDATE_MARKER_KEYS:
        return None
    if payload.get("schema") != PUBLIC_CANDIDATE_MARKER_SCHEMA:
        return None
    if type(payload.get("version")) is not int or payload["version"] != PUBLIC_CANDIDATE_MARKER_VERSION:
        return None
    source_commit = payload.get("source_commit")
    manifest_digest = payload.get("manifest_digest")
    if type(source_commit) is not str or not SOURCE_COMMIT_RE.fullmatch(source_commit):
        return None
    if type(manifest_digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
        return None
    if payload.get("manifest_digest_algorithm") != "sha256":
        return None
    if payload.get("manifest_digest_version") != MANIFEST_DIGEST_VERSION:
        return None
    if type(payload.get("history_included")) is not bool or payload["history_included"] is not False:
        return None
    if (
        type(payload.get("service_plists_included")) is not bool
        or payload["service_plists_included"] is not False
    ):
        return None
    return PublicCandidateMarker(source_commit, manifest_digest)


def _run_git(root: Path, *args: str) -> bytes:
    git_environment = {
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
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        os.fspath(root),
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=git_environment,
        )
    except OSError as exc:
        raise CandidateBuildError("Git is unavailable") from exc
    if completed.returncode != 0:
        raise CandidateBuildError("Git repository inspection failed")
    return completed.stdout


def _validated_source_root(source_root: str | Path) -> Path:
    source = Path(source_root)
    if not source.is_dir() or source.is_symlink():
        raise CandidateBuildError("source must be a real directory")
    source = source.resolve()
    top_level_raw = _run_git(source, "rev-parse", "--show-toplevel")
    try:
        top_level = Path(top_level_raw.decode("utf-8").strip()).resolve()
    except UnicodeDecodeError as exc:
        raise CandidateBuildError("Git repository path is not valid UTF-8") from exc
    if top_level != source:
        raise CandidateBuildError("source must be the Git worktree root")
    if _run_git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise CandidateBuildError("source worktree must be clean, including untracked files")
    return source


def _source_commit(source: Path) -> str:
    raw = _run_git(source, "rev-parse", "--verify", "HEAD^{commit}")
    try:
        commit = raw.decode("ascii").strip().lower()
    except UnicodeDecodeError as exc:
        raise CandidateBuildError("source commit identifier is invalid") from exc
    if not SOURCE_COMMIT_RE.fullmatch(commit):
        raise CandidateBuildError("source commit identifier is invalid")
    return commit


def _confirm_source_unchanged(source: Path, source_commit: str) -> None:
    if _source_commit(source) != source_commit:
        raise CandidateBuildError("source HEAD changed during candidate inspection")
    if _run_git(source, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise CandidateBuildError("source worktree changed during candidate inspection")


def _validated_destination(destination: str | Path, *, source: Path) -> Path:
    raw = Path(destination)
    if not raw.is_absolute():
        raise CandidateBuildError("destination must be an absolute immediate child of private tmp")
    if not DESTINATION_NAME_RE.fullmatch(raw.name):
        raise CandidateBuildError("destination name is invalid")
    temp_root = PRIVATE_TMP_ROOT.resolve(strict=True)
    if raw.parent != temp_root:
        raise CandidateBuildError("destination must be an immediate child of private tmp")
    destination_path = temp_root / raw.name
    if destination_path == source or source in destination_path.parents:
        raise CandidateBuildError("destination cannot be inside the source worktree")
    temp_fd: int | None = None
    try:
        temp_fd = os.open(temp_root, _OPEN_DIRECTORY_FLAGS)
        try:
            os.stat(raw.name, dir_fd=temp_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise CandidateBuildError("destination must not already exist")
    except CandidateBuildError:
        raise
    except OSError as exc:
        raise CandidateBuildError("destination could not be inspected safely") from exc
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
    return destination_path


def _allowed_path(
    path_text: str,
    *,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> bool:
    if type(profile) is not str or profile not in CANDIDATE_PROFILES:
        return False
    path = PurePosixPath(path_text)
    parts = path.parts
    if not parts or path.is_absolute() or ".." in parts or "." in parts:
        return False
    if len(parts) == 1:
        allowed = (
            PUBLICATION_ALLOWED_TOP_LEVEL_FILES
            if profile == CANDIDATE_PROFILE_PUBLICATION
            else ALLOWED_TOP_LEVEL_FILES
        )
        return parts[0] in allowed
    return parts[0] == "jarvis_v2" and path.suffix == ".py"


def _bounded_candidate_directory_names(
    directory_fd: int,
) -> tuple[str, ...] | None:
    names: list[str] = []
    try:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > MAX_CANDIDATE_DIRECTORY_ENTRIES:
                    return None
    except OSError:
        return None
    return tuple(sorted(names))


def _validated_profile(profile: str) -> str:
    if type(profile) is not str or profile not in CANDIDATE_PROFILES:
        raise CandidateBuildError("candidate profile is invalid")
    return profile


def _publication_artifacts(blobs: tuple[CandidateBlob, ...]) -> tuple[bool, str]:
    """Validate the tracked publication artifacts without claiming private review."""

    by_path = {blob.path: blob for blob in blobs}
    if len(by_path) != len(blobs):
        raise CandidateBuildError("candidate contains duplicate selected paths")

    license_blob = by_path.get(PUBLICATION_LICENSE_PATH)
    if (
        license_blob is None
        or license_blob.executable
        or not license_blob.content
        or len(license_blob.content) > MAX_PUBLICATION_LICENSE_BYTES
        or b"\x00" in license_blob.content
    ):
        raise CandidateBuildError("publication license artifact is invalid")
    try:
        license_text = license_blob.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CandidateBuildError("publication license artifact is invalid") from exc
    if len(license_text.strip()) < 20 or any(
        character not in "\t\n\r"
        and unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in license_text
    ):
        raise CandidateBuildError("publication license artifact is invalid")

    version_blob = by_path.get(PUBLICATION_VERSION_PATH)
    if (
        version_blob is None
        or version_blob.executable
        or not version_blob.content
        or len(version_blob.content) > MAX_PUBLICATION_VERSION_BYTES
        or b"\x00" in version_blob.content
    ):
        raise CandidateBuildError("publication version artifact is invalid")
    try:
        version_text = version_blob.content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CandidateBuildError("publication version artifact is invalid") from exc
    if version_text.endswith("\n"):
        version_value = version_text[:-1]
    else:
        version_value = version_text
    if not version_value or version_text not in {version_value, version_value + "\n"}:
        raise CandidateBuildError("publication version artifact is invalid")
    match = PUBLICATION_VERSION_RE.fullmatch(version_value)
    if match is None:
        raise CandidateBuildError("publication version artifact is invalid")
    return True, "prerelease" if match.group("prerelease") else "release"


def _publication_metadata(
    blobs: tuple[CandidateBlob, ...],
    extra_deny_literals: tuple[bytes, ...],
) -> tuple[bool, str]:
    """Validate only operator-supplied publication prerequisites.

    This does not choose a license or version, authorize a date, initialize a
    repository, or publish. The private deny literals remain runtime-only and
    are represented outside this helper only by their count/presence.
    """

    if not extra_deny_literals:
        raise CandidateBuildError("publication profile requires private deny review")
    return _publication_artifacts(blobs)


def _candidate_blobs(
    source: Path,
    source_commit: str,
    *,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> tuple[CandidateBlob, ...]:
    tree = _run_git(source, "ls-tree", "-r", "-z", "--full-tree", source_commit)
    selected: list[tuple[str, str, bool]] = []
    for raw_record in tree.split(b"\0"):
        if not raw_record:
            continue
        try:
            header, raw_path = raw_record.split(b"\t", 1)
            mode_raw, kind_raw, oid_raw = header.split(b" ", 2)
            path_text = raw_path.decode("utf-8")
            mode = mode_raw.decode("ascii")
            kind = kind_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise CandidateBuildError("Git tree contains an invalid entry") from exc
        if not _allowed_path(path_text, profile=profile):
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise CandidateBuildError("allowed candidate path is not a regular tracked file")
        selected.append((path_text, oid, mode == "100755"))

    selected.sort(key=lambda item: item[0])
    if not selected:
        raise CandidateBuildError("candidate allowlist selected no files")
    if len(selected) > MAX_CANDIDATE_FILES:
        raise CandidateBuildError("candidate file count exceeds safety limit")

    blobs: list[CandidateBlob] = []
    total_bytes = 0
    for path_text, oid, executable in selected:
        raw_size = _run_git(source, "cat-file", "-s", oid)
        try:
            blob_size = int(raw_size.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise CandidateBuildError("Git blob size is invalid") from exc
        if blob_size < 0 or total_bytes + blob_size > MAX_CANDIDATE_BYTES:
            raise CandidateBuildError("candidate size exceeds safety limit")
        content = _run_git(source, "cat-file", "blob", oid)
        if len(content) != blob_size:
            raise CandidateBuildError("Git blob size changed during inspection")
        total_bytes += len(content)
        blobs.append(CandidateBlob(path_text, oid, executable, content))
    return tuple(blobs)


def _manifest_digest(
    blobs: tuple[CandidateBlob, ...],
    source_commit: str,
) -> str:
    if not SOURCE_COMMIT_RE.fullmatch(source_commit):
        raise CandidateBuildError("source commit identifier is invalid")
    digest = hashlib.sha256()
    digest.update(f"jarvis-public-candidate-manifest-v{MANIFEST_DIGEST_VERSION}\0".encode("ascii"))
    digest.update(b"source-commit\0")
    digest.update(source_commit.encode("ascii"))
    digest.update(b"\0")
    for blob in blobs:
        record = {
            "content_sha256": hashlib.sha256(blob.content).hexdigest(),
            "executable": blob.executable,
            "path": blob.path,
            "size": len(blob.content),
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _candidate_tree_blobs_fd(
    root_fd: int,
    *,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> tuple[CandidateBlob, ...] | None:
    blobs: list[CandidateBlob] = []
    directories: set[str] = set()
    total_bytes = 0
    entries_seen = 0
    files_seen = 0

    def walk(directory_fd: int, relative_parent: str) -> bool:
        nonlocal total_bytes, entries_seen, files_seen
        try:
            before_directory = os.fstat(directory_fd)
            names = _bounded_candidate_directory_names(directory_fd)
        except OSError:
            return False
        if names is None:
            return False
        for name in names:
            entries_seen += 1
            if entries_seen > MAX_CANDIDATE_ENTRIES:
                return False
            relative = f"{relative_parent}/{name}" if relative_parent else name
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                return False
            if stat.S_ISDIR(entry_stat.st_mode):
                if (
                    not relative.startswith("jarvis_v2/") and relative != "jarvis_v2"
                ) or name in FORBIDDEN_DIRECTORY_NAMES:
                    return False
                directories.add(relative)
                child_fd: int | None = None
                try:
                    child_fd = _open_directory_at(directory_fd, name, entry_stat)
                    if not walk(child_fd, relative):
                        return False
                    child_after = os.fstat(child_fd)
                    named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not _same_entry(entry_stat, child_after) or not _same_entry(
                        child_after, named_after
                    ):
                        return False
                except (OSError, _StableTreeError):
                    return False
                finally:
                    if child_fd is not None:
                        os.close(child_fd)
                continue
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                return False
            files_seen += 1
            if files_seen > MAX_CANDIDATE_TREE_FILES:
                return False
            if relative == PUBLIC_CANDIDATE_MARKER_NAME:
                continue
            if not _allowed_path(relative, profile=profile):
                return False
            remaining = MAX_CANDIDATE_BYTES - total_bytes
            if remaining < 0:
                return False
            try:
                content, stable_stat = _read_regular_at(
                    directory_fd,
                    name,
                    max_bytes=remaining,
                    require_single_link=True,
                )
            except (OSError, OverflowError, _StableTreeError):
                return False
            total_bytes += len(content)
            blobs.append(
                CandidateBlob(
                    relative,
                    "",
                    bool(stable_stat.st_mode & 0o111),
                    content,
                )
            )
        try:
            return _same_entry(before_directory, os.fstat(directory_fd))
        except OSError:
            return False

    try:
        root_before = os.fstat(root_fd)
        if not stat.S_ISDIR(root_before.st_mode) or not walk(root_fd, ""):
            return None
        if not _same_entry(root_before, os.fstat(root_fd)):
            return None
    except OSError:
        return None
    blobs.sort(key=lambda blob: blob.path)
    if not blobs or len(blobs) > MAX_CANDIDATE_FILES:
        return None
    implied_directories: set[str] = set()
    for blob in blobs:
        parent = PurePosixPath(blob.path).parent
        while parent != PurePosixPath("."):
            implied_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != implied_directories:
        return None
    return tuple(blobs)


def _candidate_state_fd(
    root_fd: int,
    *,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> _CandidateState | None:
    try:
        marker_content, _ = _read_regular_at(
            root_fd,
            PUBLIC_CANDIDATE_MARKER_NAME,
            max_bytes=MAX_PUBLIC_CANDIDATE_MARKER_BYTES,
            require_single_link=True,
        )
    except (OSError, OverflowError, _StableTreeError):
        return None
    marker = _parse_public_candidate_marker(marker_content)
    if marker is None:
        return None
    blobs = _candidate_tree_blobs_fd(root_fd, profile=profile)
    if blobs is None or _manifest_digest(blobs, marker.source_commit) != marker.manifest_digest:
        return None
    return _CandidateState(marker_content, marker, blobs)


def _open_candidate_root(candidate_root: str | Path) -> tuple[Path, int, os.stat_result] | None:
    root = Path(candidate_root)
    fd: int | None = None
    try:
        fd = os.open(root, _OPEN_DIRECTORY_FLAGS)
        opened = os.fstat(fd)
        named = os.stat(root, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or not _same_entry(opened, named):
            return None
        result = fd
        fd = None
        return root, result, opened
    except OSError:
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _candidate_root_path_matches_fd(path: Path, root_fd: int, opened: os.stat_result) -> bool:
    """Keep an anchored candidate bound to the same named directory identity."""

    try:
        current_fd = os.fstat(root_fd)
        current_path = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(current_fd.st_mode)
        and stat.S_ISDIR(current_path.st_mode)
        and (opened.st_dev, opened.st_ino) == (current_fd.st_dev, current_fd.st_ino)
        and _same_entry(current_fd, current_path)
    )


def _marker_matches_candidate_tree(
    candidate_root: str | Path,
    *,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> bool:
    opened = _open_candidate_root(candidate_root)
    if opened is None:
        return False
    root, root_fd, root_stat = opened
    try:
        first = _candidate_state_fd(root_fd, profile=profile)
        second = _candidate_state_fd(root_fd, profile=profile)
        return (
            first is not None
            and first == second
            and _candidate_root_path_matches_fd(root, root_fd, root_stat)
        )
    finally:
        os.close(root_fd)


def valid_public_candidate_marker(
    candidate_root: str | Path,
    *,
    environ: dict[str, str] | None = None,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> bool:
    """Recognize only an exact, bound, preflight-clean candidate-root marker.

    The helper never searches parent directories. A development checkout with
    service plists or any other non-allowlisted file therefore cannot acquire
    candidate semantics merely by containing marker-shaped JSON elsewhere.
    The opt-in publication profile additionally revalidates exact publication
    metadata and requires the runtime-only private deny list on every call.
    """
    opened = _open_candidate_root(candidate_root)
    if opened is None:
        return False
    root, root_fd, root_stat = opened
    try:
        selected_profile = _validated_profile(profile)
        extra = extra_deny_literals_from_env(environ)
        before = _candidate_state_fd(root_fd, profile=selected_profile)
        if before is None:
            return False
        if selected_profile == CANDIDATE_PROFILE_PUBLICATION:
            _publication_metadata(before.blobs, extra)
        preflight = _scan_release_tree_fd(root_fd, extra_deny_literals=extra)
        after = _candidate_state_fd(root_fd, profile=selected_profile)
        return (
            preflight.ok
            and before == after
            and _candidate_tree_has_finalized_modes_fd(root_fd)
            and _candidate_root_path_matches_fd(root, root_fd, root_stat)
        )
    except (CandidateBuildError, OSError, PreflightConfigError):
        return False
    finally:
        os.close(root_fd)


def structural_public_candidate_profile(candidate_root: str | Path) -> str | None:
    """Return the structural profile of an exact finalized candidate, if any.

    This helper exists for candidate-aware offline tests and unavailable-feature
    guards. It verifies the marker-bound tree, finalized modes, stable root, and
    ordinary privacy policy for either profile. For a publication tree it also
    validates LICENSE and VERSION, but deliberately does not accept or attest a
    runtime-only private deny review. Publication readiness must continue to use
    ``valid_public_candidate_marker(..., profile="publication")`` with the real
    reviewed deny list.
    """

    opened = _open_candidate_root(candidate_root)
    if opened is None:
        return None
    root, root_fd, root_stat = opened
    try:
        for profile in (CANDIDATE_PROFILE_PREVIEW, CANDIDATE_PROFILE_PUBLICATION):
            before = _candidate_state_fd(root_fd, profile=profile)
            if before is None:
                continue
            if profile == CANDIDATE_PROFILE_PUBLICATION:
                _publication_artifacts(before.blobs)
            preflight = _scan_release_tree_fd(root_fd, extra_deny_literals=())
            after = _candidate_state_fd(root_fd, profile=profile)
            if (
                preflight.ok
                and before == after
                and _candidate_tree_has_finalized_modes_fd(root_fd)
                and _candidate_root_path_matches_fd(root, root_fd, root_stat)
            ):
                return profile
        return None
    except (CandidateBuildError, OSError, PreflightConfigError):
        return None
    finally:
        os.close(root_fd)


def _create_candidate_root(destination: Path) -> tuple[int, os.stat_result]:
    temp_fd: int | None = None
    root_fd: int | None = None
    try:
        temp_fd = os.open(PRIVATE_TMP_ROOT, _OPEN_DIRECTORY_FLAGS)
        os.mkdir(destination.name, REVIEW_DIRECTORY_MODE, dir_fd=temp_fd)
        root_fd = os.open(destination.name, _OPEN_DIRECTORY_FLAGS, dir_fd=temp_fd)
        opened = os.fstat(root_fd)
        named = os.stat(destination.name, dir_fd=temp_fd, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or not _same_entry(opened, named):
            raise CandidateBuildError("candidate destination changed during creation")
        result = root_fd
        root_fd = None
        return result, opened
    except FileExistsError as exc:
        raise CandidateBuildError("destination must not already exist") from exc
    except CandidateBuildError:
        raise
    except OSError as exc:
        raise CandidateBuildError("candidate destination could not be created safely") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        if temp_fd is not None:
            os.close(temp_fd)


def _open_or_create_directory_at(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, REVIEW_DIRECTORY_MODE, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise CandidateBuildError("candidate directory could not be created safely") from exc
    try:
        expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(expected.st_mode):
            raise CandidateBuildError("candidate directory target unexpectedly exists")
        return _open_directory_at(parent_fd, name, expected)
    except (OSError, _StableTreeError) as exc:
        raise CandidateBuildError("candidate directory changed during creation") from exc


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short candidate write")
        view = view[written:]


def _write_file_at(parent_fd: int, name: str, content: bytes, mode: int) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd: int | None = None
    try:
        fd = os.open(name, flags, REVIEW_REGULAR_FILE_MODE, dir_fd=parent_fd)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise CandidateBuildError("candidate file is not a private regular file")
        _write_all(fd, content)
        os.fchmod(fd, mode)
        after = os.fstat(fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not _same_entry(after, named) or after.st_size != len(content):
            raise CandidateBuildError("candidate file changed during creation")
    except CandidateBuildError:
        raise
    except OSError as exc:
        raise CandidateBuildError("candidate file could not be created safely") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _write_candidate(root_fd: int, blobs: tuple[CandidateBlob, ...]) -> None:
    for blob in blobs:
        parts = PurePosixPath(blob.path).parts
        parent_fd = os.dup(root_fd)
        try:
            for part in parts[:-1]:
                child_fd = _open_or_create_directory_at(parent_fd, part)
                os.close(parent_fd)
                parent_fd = child_fd
            _write_file_at(
                parent_fd,
                parts[-1],
                blob.content,
                REVIEW_EXECUTABLE_FILE_MODE if blob.executable else REVIEW_REGULAR_FILE_MODE,
            )
        finally:
            os.close(parent_fd)


def _write_public_candidate_marker(
    root_fd: int,
    *,
    source_commit: str,
    manifest_digest: str,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> None:
    payload = _public_candidate_marker_payload(source_commit, manifest_digest)
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if _parse_public_candidate_marker(content) != PublicCandidateMarker(source_commit, manifest_digest):
        raise CandidateBuildError("generated public candidate marker failed validation")
    _write_file_at(
        root_fd,
        PUBLIC_CANDIDATE_MARKER_NAME,
        content,
        REVIEW_REGULAR_FILE_MODE,
    )
    if _candidate_state_fd(root_fd, profile=profile) is None:
        raise CandidateBuildError("written public candidate marker failed tree binding")


def _candidate_tree_has_finalized_modes_fd(root_fd: int) -> bool:
    """Require the exact non-writable modes used by a successful build."""
    entries_seen = 0
    files_seen = 0

    def walk(directory_fd: int, *, is_root: bool = False) -> bool:
        nonlocal entries_seen, files_seen
        try:
            directory_stat = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory_stat.st_mode)
                or stat.S_IMODE(directory_stat.st_mode) != FINAL_DIRECTORY_MODE
            ):
                return False
            names = _bounded_candidate_directory_names(directory_fd)
            if names is None:
                return False
            for name in names:
                entries_seen += 1
                if entries_seen > MAX_CANDIDATE_ENTRIES:
                    return False
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(entry_stat.st_mode):
                    child_fd = _open_directory_at(directory_fd, name, entry_stat)
                    try:
                        if not walk(child_fd):
                            return False
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                    return False
                files_seen += 1
                if files_seen > MAX_CANDIDATE_TREE_FILES:
                    return False
                expected = (
                    FINAL_REGULAR_FILE_MODE
                    if is_root and name == PUBLIC_CANDIDATE_MARKER_NAME
                    else (
                        FINAL_EXECUTABLE_FILE_MODE
                        if entry_stat.st_mode & 0o111
                        else FINAL_REGULAR_FILE_MODE
                    )
                )
                if stat.S_IMODE(entry_stat.st_mode) != expected:
                    return False
            return True
        except (OSError, _StableTreeError):
            return False

    return walk(root_fd, is_root=True)


def _candidate_tree_has_finalized_modes(candidate_root: Path) -> bool:
    opened = _open_candidate_root(candidate_root)
    if opened is None:
        return False
    root, root_fd, root_stat = opened
    try:
        return _candidate_tree_has_finalized_modes_fd(root_fd) and _candidate_root_path_matches_fd(
            root, root_fd, root_stat
        )
    finally:
        os.close(root_fd)


def _finalize_candidate_immutable_fd(root_fd: int) -> None:
    """Make a preflight-clean staging tree owner-readable but non-writable."""
    entries_seen = 0
    files_seen = 0

    def finalize(directory_fd: int, *, is_root: bool = False) -> None:
        nonlocal entries_seen, files_seen
        before = os.fstat(directory_fd)
        names = _bounded_candidate_directory_names(directory_fd)
        if names is None:
            raise CandidateBuildError("candidate entry count exceeds safety limit")
        for name in names:
            entries_seen += 1
            if entries_seen > MAX_CANDIDATE_ENTRIES:
                raise CandidateBuildError("candidate entry count exceeds safety limit")
            entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(entry_stat.st_mode):
                child_fd = _open_directory_at(directory_fd, name, entry_stat)
                try:
                    finalize(child_fd)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                raise CandidateBuildError("candidate file changed before finalization")
            files_seen += 1
            if files_seen > MAX_CANDIDATE_TREE_FILES:
                raise CandidateBuildError("candidate file count exceeds safety limit")
            file_fd = os.open(name, _OPEN_FILE_FLAGS, dir_fd=directory_fd)
            try:
                opened = os.fstat(file_fd)
                if not _same_entry(entry_stat, opened):
                    raise CandidateBuildError("candidate file changed before finalization")
                mode = (
                    FINAL_REGULAR_FILE_MODE
                    if is_root and name == PUBLIC_CANDIDATE_MARKER_NAME
                    else (
                        FINAL_EXECUTABLE_FILE_MODE
                        if opened.st_mode & 0o111
                        else FINAL_REGULAR_FILE_MODE
                    )
                )
                os.fchmod(file_fd, mode)
            finally:
                os.close(file_fd)
        after_children = os.fstat(directory_fd)
        if before.st_dev != after_children.st_dev or before.st_ino != after_children.st_ino:
            raise CandidateBuildError("candidate directory changed before finalization")
        os.fchmod(directory_fd, FINAL_DIRECTORY_MODE)

    try:
        finalize(root_fd, is_root=True)
    except CandidateBuildError:
        raise
    except (OSError, _StableTreeError) as exc:
        raise CandidateBuildError("candidate could not be finalized immutable") from exc
    if not _candidate_tree_has_finalized_modes_fd(root_fd):
        raise CandidateBuildError("candidate immutable finalization failed validation")


def build_public_candidate(
    source_root: str | Path,
    destination: str | Path,
    *,
    dry_run: bool = False,
    environ: dict[str, str] | None = None,
    profile: str = CANDIDATE_PROFILE_PREVIEW,
) -> CandidateBuildReport:
    selected_profile = _validated_profile(profile)
    source = _validated_source_root(source_root)
    source_commit = _source_commit(source)
    destination_path = _validated_destination(destination, source=source)
    blobs = _candidate_blobs(source, source_commit, profile=selected_profile)
    file_count = len(blobs)
    byte_count = sum(len(blob.content) for blob in blobs)
    manifest_digest = _manifest_digest(blobs, source_commit)
    # Hidden scan configuration is part of build readiness. Validate it before
    # creating a directory so malformed private input cannot leave a candidate.
    extra = extra_deny_literals_from_env(environ)
    publication_license_validated = False
    publication_version_kind: str | None = None
    if selected_profile == CANDIDATE_PROFILE_PUBLICATION:
        publication_license_validated, publication_version_kind = _publication_metadata(
            blobs, extra
        )
    _confirm_source_unchanged(source, source_commit)
    if dry_run:
        return CandidateBuildReport(
            True,
            True,
            False,
            file_count,
            byte_count,
            source_commit,
            manifest_digest,
            None,
            False,
            len(extra),
            selected_profile,
            publication_license_validated,
            publication_version_kind,
        )

    root_fd, root_stat = _create_candidate_root(destination_path)
    try:
        _write_candidate(root_fd, blobs)
        preflight = _scan_release_tree_fd(root_fd, extra_deny_literals=extra)
        finalized = False
        if preflight.ok:
            _write_public_candidate_marker(
                root_fd,
                source_commit=source_commit,
                manifest_digest=manifest_digest,
                profile=selected_profile,
            )
            # A-B-A: bind the marker/tree before the privacy scan, require the
            # scan itself to be stable, then require the same bound tree after.
            before = _candidate_state_fd(root_fd, profile=selected_profile)
            preflight = _scan_release_tree_fd(root_fd, extra_deny_literals=extra)
            after = _candidate_state_fd(root_fd, profile=selected_profile)
            if not preflight.ok or before is None or before != after:
                try:
                    os.unlink(PUBLIC_CANDIDATE_MARKER_NAME, dir_fd=root_fd)
                except OSError as exc:
                    raise CandidateBuildError(
                        "failed candidate marker could not be invalidated"
                    ) from exc
                raise CandidateBuildError(
                    "candidate tree changed during final preflight validation"
                )
            else:
                _finalize_candidate_immutable_fd(root_fd)
                root_stat = os.fstat(root_fd)
                finalized_state = _candidate_state_fd(root_fd, profile=selected_profile)
                finalized = (
                    finalized_state == after
                    and _candidate_tree_has_finalized_modes_fd(root_fd)
                    and _candidate_root_path_matches_fd(destination_path, root_fd, root_stat)
                )
                if finalized:
                    final_preflight = _scan_release_tree_fd(root_fd, extra_deny_literals=extra)
                    final_state = _candidate_state_fd(root_fd, profile=selected_profile)
                    finalized = (
                        final_preflight.ok
                        and final_state == finalized_state
                        and _candidate_tree_has_finalized_modes_fd(root_fd)
                    )
                if not finalized:
                    raise CandidateBuildError("finalized candidate failed marker validation")
        if not _candidate_root_path_matches_fd(destination_path, root_fd, root_stat):
            raise CandidateBuildError("candidate destination changed during build")
    finally:
        os.close(root_fd)
    return CandidateBuildReport(
        preflight.ok,
        False,
        True,
        file_count,
        byte_count,
        source_commit,
        manifest_digest,
        preflight,
        finalized,
        len(extra),
        selected_profile,
        publication_license_validated,
        publication_version_kind,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and scan a history-free source-only public release candidate."
    )
    parser.add_argument("--source", required=True, help="clean Git worktree root")
    parser.add_argument(
        "--destination",
        required=True,
        help="new immediate-child candidate directory in /private/tmp",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the source, destination, and allowlist without creating files",
    )
    parser.add_argument(
        "--profile",
        default=CANDIDATE_PROFILE_PREVIEW,
        metavar="PROFILE",
        help=(
            "candidate profile; publication additionally requires an operator-selected "
            "LICENSE, non-development VERSION, and runtime-only private deny list"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = build_public_candidate(
            args.source,
            args.destination,
            dry_run=args.dry_run,
            profile=args.profile,
        )
    except (CandidateBuildError, PreflightConfigError):
        print("ERROR public-release-candidate configuration or source state is unsafe")
        return 2
    print(json.dumps(report.summary_json_payload(), sort_keys=True, separators=(",", ":")))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
