"""Read-only proof that a prepared public Git commit matches one release candidate.

This verifier deliberately does not initialize Git, stage files, create commits,
change permissions, write receipts, or publish.  It compares an already-finalized
publication candidate with one explicitly named, clean Git ``HEAD`` after reducing
filesystem modes to Git's regular/executable distinction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time
from typing import Iterable, Mapping, Sequence

from jarvis_v2.scripts.public_release_candidate import (
    CANDIDATE_PROFILE_PUBLICATION,
    CandidateBuildError,
    PUBLIC_CANDIDATE_MARKER_NAME,
    SOURCE_COMMIT_RE,
    _candidate_root_path_matches_fd,
    _candidate_state_fd,
    _candidate_tree_has_finalized_modes_fd,
    _open_candidate_root,
    _publication_metadata,
)
from jarvis_v2.scripts.public_release_preflight import (
    PreflightConfigError,
    _OPEN_DIRECTORY_FLAGS,
    _open_directory_at,
    _read_regular_at,
    _scan_release_tree_fd,
    _same_entry,
    extra_deny_literals_from_env,
)


MAX_GIT_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_GIT_TREE_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_GIT_TREE_RECORDS = 2_001
MAX_GIT_PATH_BYTES = 4_096
MAX_WORKTREE_ENTRIES = 4_096
MAX_WORKTREE_FILES = 2_001
MAX_WORKTREE_BYTES = 64 * 1024 * 1024
MAX_GIT_METADATA_ENTRIES = 16_384
MAX_GIT_OBJECT_RECORDS = 8_192
GIT_TIMEOUT_SECONDS = 20.0
# macOS can delay process-group reaping under memory or CPU pressure even after
# SIGKILL. Keep this bounded, but allow enough time to attest group absence
# before reporting an outcome-unknown cleanup failure.
GIT_TERMINATE_GRACE_SECONDS = 5.0
_GIT_OBJECT_RE = re.compile(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class HandoffVerificationError(ValueError):
    """A content-free public handoff failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PublicReleaseHandoffReport:
    ok: bool
    public_commit: str
    source_commit: str
    manifest_digest: str
    files_compared: int
    bytes_compared: int

    def summary_json_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "candidate_profile": CANDIDATE_PROFILE_PUBLICATION,
            "candidate_valid": self.ok,
            "candidate_finalized_immutable": self.ok,
            "repository_clean": self.ok,
            "head_matches_explicit_commit": self.ok,
            "single_root_commit": self.ok,
            "history_free": self.ok,
            "refs_isolated": self.ok,
            "remotes_absent": self.ok,
            "replace_refs_absent": self.ok,
            "promisor_disabled": self.ok,
            "repository_descriptor_anchored": self.ok,
            "git_metadata_no_follow": self.ok,
            "common_git_directory_absent": self.ok,
            "reflogs_absent": self.ok,
            "object_database_exact": self.ok,
            "git_process_groups_cleaned": self.ok,
            "index_and_worktree_match_commit": self.ok,
            "worktree_bytes_match_candidate": self.ok,
            "git_tree_matches_candidate": self.ok,
            "marker_matches_candidate": self.ok,
            "public_commit": self.public_commit,
            "source_commit": self.source_commit,
            "manifest_digest": self.manifest_digest,
            "files_compared": self.files_compared,
            "bytes_compared": self.bytes_compared,
            "paths_included": False,
            "private_content_included": False,
            "private_deny_literals_retained": False,
            "writes_files": False,
            "changes_permissions": False,
            "initializes_git": False,
            "stages_files": False,
            "creates_commits": False,
            "publishes": False,
            "authorizes_publication": False,
        }


@dataclass(frozen=True)
class _GitTreeBlob:
    path: bytes
    oid: str
    executable: bool


@dataclass(frozen=True)
class _AnchoredRepository:
    path: Path
    root_fd: int
    root_stat: os.stat_result
    git_directory_fd: int
    git_directory_stat: os.stat_result


def _open_repository(repository_root: str | Path) -> _AnchoredRepository:
    raw = Path(repository_root)
    root_fd: int | None = None
    git_directory_fd: int | None = None
    try:
        named = os.stat(raw, follow_symlinks=False)
        if not stat.S_ISDIR(named.st_mode) or raw.is_symlink():
            raise HandoffVerificationError("repository_invalid")
        path = raw.resolve(strict=True)
        root_fd = os.open(path, _OPEN_DIRECTORY_FLAGS)
        opened = os.fstat(root_fd)
        current = os.stat(path, follow_symlinks=False)
        if not _same_entry(named, opened) or not _same_entry(opened, current):
            raise HandoffVerificationError("repository_invalid")
        git_named = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(git_named.st_mode):
            raise HandoffVerificationError("repository_invalid")
        git_directory_fd = os.open(".git", _OPEN_DIRECTORY_FLAGS, dir_fd=root_fd)
        git_opened = os.fstat(git_directory_fd)
        git_current = os.stat(".git", dir_fd=root_fd, follow_symlinks=False)
        if not _same_entry(git_named, git_opened) or not _same_entry(
            git_opened, git_current
        ):
            raise HandoffVerificationError("repository_invalid")
        result = _AnchoredRepository(
            path,
            root_fd,
            opened,
            git_directory_fd,
            git_opened,
        )
        root_fd = None
        git_directory_fd = None
        return result
    except HandoffVerificationError:
        raise
    except OSError:
        raise HandoffVerificationError("repository_invalid") from None
    finally:
        if git_directory_fd is not None:
            os.close(git_directory_fd)
        if root_fd is not None:
            os.close(root_fd)


def _repository_anchor_valid(repository: _AnchoredRepository) -> bool:
    try:
        root_opened = os.fstat(repository.root_fd)
        root_named = os.stat(repository.path, follow_symlinks=False)
        git_opened = os.fstat(repository.git_directory_fd)
        git_named = os.stat(".git", dir_fd=repository.root_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        _same_entry(repository.root_stat, root_opened)
        and _same_entry(root_opened, root_named)
        and _same_entry(repository.git_directory_stat, git_opened)
        and _same_entry(git_opened, git_named)
    )


def _anchored_git_output(
    repository: _AnchoredRepository,
    *arguments: str,
    limit: int = MAX_GIT_OUTPUT_BYTES,
) -> bytes:
    if not _repository_anchor_valid(repository):
        raise HandoffVerificationError("repository_changed")
    output = _git_output(
        repository.path,
        *arguments,
        limit=limit,
        repository_fd=repository.root_fd,
        git_directory_fd=repository.git_directory_fd,
    )
    if not _repository_anchor_valid(repository):
        raise HandoffVerificationError("repository_changed")
    return output


def _git_environment(
    repository_fd: int | None = None,
    git_directory_fd: int | None = None,
) -> dict[str, str]:
    """Use Git without optional index writes, hooks, prompts, or user config."""

    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    if repository_fd is not None and git_directory_fd is not None:
        # macOS Git cannot traverse directory descriptors through /dev/fd.
        # The parent retains and revalidates both descriptors around every Git
        # call; Git itself reads relative to the already-resolved path.  The
        # exact worktree comparison never falls back to that path.
        environment["GIT_DIR"] = ".git"
        environment["GIT_WORK_TREE"] = "."
    return environment


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _wait_process_group_exit(
    process: subprocess.Popen[bytes], deadline: float
) -> bool:
    while time.monotonic() < deadline:
        try:
            process.poll()
        except OSError:
            return False
        if not _process_group_exists(process.pid):
            return True
        time.sleep(0.01)
    try:
        process.poll()
    except OSError:
        return False
    return not _process_group_exists(process.pid)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Stop and attest absence of the isolated Git process group."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            if process.poll() is None:
                process.terminate()
        except OSError:
            pass
    if not _wait_process_group_exit(
        process, time.monotonic() + GIT_TERMINATE_GRACE_SECONDS
    ):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass
        _wait_process_group_exit(
            process, time.monotonic() + GIT_TERMINATE_GRACE_SECONDS
        )
    try:
        process.wait(timeout=0)
    except (OSError, subprocess.SubprocessError):
        pass
    return not _process_group_exists(process.pid)


def _run_git_bounded(
    repository: Path,
    arguments: Sequence[str],
    *,
    max_output_bytes: int = MAX_GIT_OUTPUT_BYTES,
    repository_fd: int | None = None,
    git_directory_fd: int | None = None,
) -> tuple[int, bytes]:
    if max_output_bytes < 0 or max_output_bytes > MAX_GIT_OUTPUT_BYTES:
        raise HandoffVerificationError("git_inspection_failed")
    anchored = repository_fd is not None or git_directory_fd is not None
    if anchored and (repository_fd is None or git_directory_fd is None):
        raise HandoffVerificationError("repository_invalid")
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
    ]
    if not anchored:
        command.extend(("-C", os.fspath(repository)))
    command.extend(arguments)
    popen_options: dict[str, object] = {}
    if repository_fd is not None and git_directory_fd is not None:
        popen_options["pass_fds"] = (repository_fd, git_directory_fd)
        popen_options["cwd"] = os.fspath(repository)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_git_environment(repository_fd, git_directory_fd),
            start_new_session=True,
            **popen_options,
        )
    except OSError:
        raise HandoffVerificationError("git_unavailable") from None
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    output = bytearray()
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    completed = False
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HandoffVerificationError("git_inspection_failed")
            events = selector.select(min(0.25, remaining))
            if not events:
                continue
            chunk = os.read(process.stdout.fileno(), min(65_536, max_output_bytes + 1 - len(output)))
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > max_output_bytes:
                raise HandoffVerificationError("git_inspection_limit")
        try:
            return_code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise HandoffVerificationError("git_inspection_failed") from None
        completed = True
        if _process_group_exists(process.pid) and not _wait_process_group_exit(
            process,
            time.monotonic() + GIT_TERMINATE_GRACE_SECONDS,
        ):
            if not _terminate_process_group(process):
                raise HandoffVerificationError("git_process_cleanup_unknown") from None
            raise HandoffVerificationError("git_inspection_failed") from None
        return return_code, bytes(output)
    finally:
        try:
            selector.close()
        except OSError:
            pass
        if not completed:
            if not _terminate_process_group(process):
                raise HandoffVerificationError("git_process_cleanup_unknown") from None
        try:
            process.stdout.close()
        except OSError:
            pass


def _git_output(
    repository: Path,
    *arguments: str,
    limit: int = MAX_GIT_OUTPUT_BYTES,
    repository_fd: int | None = None,
    git_directory_fd: int | None = None,
) -> bytes:
    code, output = _run_git_bounded(
        repository,
        arguments,
        max_output_bytes=limit,
        repository_fd=repository_fd,
        git_directory_fd=git_directory_fd,
    )
    if code != 0:
        raise HandoffVerificationError("git_inspection_failed")
    return output


def _optional_git_output(
    repository: _AnchoredRepository,
    *arguments: str,
    limit: int = MAX_GIT_OUTPUT_BYTES,
) -> bytes | None:
    if not _repository_anchor_valid(repository):
        raise HandoffVerificationError("repository_changed")
    code, output = _run_git_bounded(
        repository.path,
        arguments,
        max_output_bytes=limit,
        repository_fd=repository.root_fd,
        git_directory_fd=repository.git_directory_fd,
    )
    if not _repository_anchor_valid(repository):
        raise HandoffVerificationError("repository_changed")
    if code == 1:
        return None
    if code != 0:
        raise HandoffVerificationError("git_inspection_failed")
    return output


def _git_metadata_entry(
    repository: _AnchoredRepository,
    name: str,
) -> os.stat_result | None:
    try:
        return os.stat(
            name,
            dir_fd=repository.git_directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError:
        raise HandoffVerificationError("repository_metadata_invalid") from None


def _scan_git_metadata_directory(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    prefix: str,
) -> tuple[int, tuple[str, ...]]:
    """Reject symlinked, special, hard-linked, or unstable Git metadata."""

    directory_fd: int | None = None
    entries_seen = 0
    regular_files: list[str] = []
    try:
        directory_fd = _open_directory_at(parent_fd, name, expected)

        def walk(current_fd: int, relative: str) -> None:
            nonlocal entries_seen
            before = os.fstat(current_fd)
            try:
                with os.scandir(current_fd) as iterator:
                    names = tuple(sorted(entry.name for entry in iterator))
            except OSError:
                raise HandoffVerificationError("repository_metadata_invalid") from None
            for child_name in names:
                entries_seen += 1
                if entries_seen > MAX_GIT_METADATA_ENTRIES:
                    raise HandoffVerificationError("repository_metadata_limit")
                if child_name in {"", ".", ".."} or "/" in child_name or "\0" in child_name:
                    raise HandoffVerificationError("repository_metadata_invalid")
                child_relative = f"{relative}/{child_name}"
                if child_relative.startswith("objects/info/"):
                    raise HandoffVerificationError("repository_metadata_invalid")
                if child_relative in {"info/attributes", "info/grafts"}:
                    raise HandoffVerificationError("repository_metadata_invalid")
                try:
                    child = os.stat(
                        child_name,
                        dir_fd=current_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    raise HandoffVerificationError("repository_metadata_invalid") from None
                if stat.S_ISDIR(child.st_mode):
                    child_fd: int | None = None
                    try:
                        child_fd = _open_directory_at(current_fd, child_name, child)
                        walk(child_fd, child_relative)
                        if not _same_entry(child, os.fstat(child_fd)):
                            raise HandoffVerificationError("repository_metadata_invalid")
                    except HandoffVerificationError:
                        raise
                    except OSError:
                        raise HandoffVerificationError("repository_metadata_invalid") from None
                    finally:
                        if child_fd is not None:
                            os.close(child_fd)
                    continue
                if not stat.S_ISREG(child.st_mode) or child.st_nlink != 1:
                    raise HandoffVerificationError("repository_metadata_invalid")
                regular_files.append(child_relative)
            try:
                after = os.fstat(current_fd)
            except OSError:
                raise HandoffVerificationError("repository_metadata_invalid") from None
            if not _same_entry(before, after):
                raise HandoffVerificationError("repository_metadata_invalid")

        walk(directory_fd, prefix)
        if not _same_entry(expected, os.fstat(directory_fd)):
            raise HandoffVerificationError("repository_metadata_invalid")
        return entries_seen, tuple(regular_files)
    except HandoffVerificationError:
        raise
    except OSError:
        raise HandoffVerificationError("repository_metadata_invalid") from None
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _validate_required_git_metadata(repository: _AnchoredRepository) -> None:
    for forbidden, code in (
        ("commondir", "repository_common_directory_present"),
        ("gitdir", "repository_common_directory_present"),
        ("worktrees", "repository_common_directory_present"),
        ("modules", "repository_metadata_invalid"),
        ("logs", "repository_reflog_present"),
    ):
        if _git_metadata_entry(repository, forbidden) is not None:
            raise HandoffVerificationError(code)

    for name in ("HEAD", "config", "index"):
        entry = _git_metadata_entry(repository, name)
        if entry is None or not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            raise HandoffVerificationError("repository_metadata_invalid")
    packed_refs = _git_metadata_entry(repository, "packed-refs")
    if packed_refs is not None:
        raise HandoffVerificationError("repository_metadata_invalid")

    total = 0
    for name in ("objects", "refs", "info"):
        entry = _git_metadata_entry(repository, name)
        if entry is None or not stat.S_ISDIR(entry.st_mode):
            raise HandoffVerificationError("repository_metadata_invalid")
        entries, regular_files = _scan_git_metadata_directory(
            repository.git_directory_fd,
            name,
            entry,
            prefix=name,
        )
        total += entries
        if total > MAX_GIT_METADATA_ENTRIES:
            raise HandoffVerificationError("repository_metadata_limit")
        if name == "refs" and (
            len(regular_files) != 1
            or not regular_files[0].startswith("refs/heads/")
        ):
            raise HandoffVerificationError("repository_refs_invalid")


def _reject_git_storage_indirection(repository: _AnchoredRepository) -> None:
    """Reject local configuration that can redirect or lazily obtain objects."""

    _validate_required_git_metadata(repository)
    forbidden_config = _optional_git_output(
        repository,
        "config",
        "--local",
        "--no-includes",
        "--get-regexp",
        (
            r"^(extensions\..*|core\.worktree|include\.path|"
            r"core\.(attributesfile|excludesfile)|filter\..*|includeif\..*\.path|"
            r"remote\..*)$"
        ),
        limit=16_384,
    )
    if forbidden_config:
        lowered = forbidden_config.lower()
        allowed_extension = b"extensions.objectformat sha256"
        lines = tuple(line.strip() for line in lowered.splitlines() if line.strip())
        rejected_lines = tuple(line for line in lines if line != allowed_extension)
        if rejected_lines:
            rejected = b"\n".join(rejected_lines)
            if b"promisor" in rejected or b"partialclone" in rejected:
                raise HandoffVerificationError("repository_promisor_configured")
            if b"remote." in rejected:
                raise HandoffVerificationError("repository_remote_present")
            raise HandoffVerificationError("repository_metadata_invalid")
    for relative in ("shallow",):
        try:
            os.stat(relative, dir_fd=repository.git_directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError:
            raise HandoffVerificationError("repository_metadata_invalid") from None
        raise HandoffVerificationError("repository_metadata_invalid")


def _reject_nonstandalone_git_state(repository: _AnchoredRepository, commit: str) -> None:
    shallow = _anchored_git_output(
        repository,
        "rev-parse",
        "--is-shallow-repository",
        limit=16,
    ).strip()
    if shallow != b"false":
        raise HandoffVerificationError("repository_history_not_isolated")
    parents = _anchored_git_output(
        repository,
        "rev-list",
        "--parents",
        "-n",
        "1",
        commit,
        limit=512,
    )
    try:
        parent_fields = parents.decode("ascii", errors="strict").strip().split()
    except UnicodeDecodeError:
        raise HandoffVerificationError("repository_history_not_isolated") from None
    if parent_fields != [commit]:
        raise HandoffVerificationError("repository_history_not_isolated")
    commit_count = _anchored_git_output(
        repository,
        "rev-list",
        "--count",
        commit,
        limit=64,
    ).strip()
    if commit_count != b"1":
        raise HandoffVerificationError("repository_history_not_isolated")

    refs_raw = _anchored_git_output(
        repository,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs",
        limit=MAX_GIT_TREE_OUTPUT_BYTES,
    )
    try:
        ref_lines = tuple(line for line in refs_raw.decode("ascii").splitlines() if line)
    except UnicodeDecodeError:
        raise HandoffVerificationError("repository_refs_invalid") from None
    if len(ref_lines) != 1:
        raise HandoffVerificationError("repository_refs_invalid")
    parts = ref_lines[0].split(" ")
    if (
        len(parts) != 2
        or not parts[0].startswith("refs/heads/")
        or parts[1] != commit
    ):
        raise HandoffVerificationError("repository_refs_invalid")


def _require_exact_object_database(
    repository: _AnchoredRepository,
    commit: str,
) -> None:
    reachable_raw = _anchored_git_output(
        repository,
        "rev-list",
        "--objects",
        "--no-object-names",
        commit,
        limit=MAX_GIT_TREE_OUTPUT_BYTES,
    )
    reachable_lines = tuple(line for line in reachable_raw.splitlines() if line)
    if (
        not reachable_lines
        or len(reachable_lines) > MAX_GIT_OBJECT_RECORDS
        or len(set(reachable_lines)) != len(reachable_lines)
        or any(_GIT_OBJECT_RE.fullmatch(line) is None for line in reachable_lines)
    ):
        raise HandoffVerificationError("repository_object_inventory_invalid")

    inventory_raw = _anchored_git_output(
        repository,
        "cat-file",
        "--batch-all-objects",
        "--batch-check=%(objectname) %(objecttype)",
        limit=MAX_GIT_TREE_OUTPUT_BYTES,
    )
    inventory: dict[bytes, bytes] = {}
    for line in inventory_raw.splitlines():
        if not line:
            continue
        if len(inventory) >= MAX_GIT_OBJECT_RECORDS:
            raise HandoffVerificationError("repository_object_inventory_invalid")
        fields = line.split(b" ")
        if (
            len(fields) != 2
            or _GIT_OBJECT_RE.fullmatch(fields[0]) is None
            or fields[1] not in {b"blob", b"tree", b"commit"}
            or fields[0] in inventory
        ):
            raise HandoffVerificationError("repository_object_inventory_invalid")
        inventory[fields[0]] = fields[1]
    reachable = frozenset(reachable_lines)
    if (
        frozenset(inventory) != reachable
        or inventory.get(commit.encode("ascii")) != b"commit"
        or sum(kind == b"commit" for kind in inventory.values()) != 1
        or not any(kind == b"tree" for kind in inventory.values())
        or not any(kind == b"blob" for kind in inventory.values())
    ):
        raise HandoffVerificationError("repository_object_inventory_invalid")


def _validated_repository(repository: _AnchoredRepository, commit: str) -> Path:
    if type(commit) is not str or not SOURCE_COMMIT_RE.fullmatch(commit):
        raise HandoffVerificationError("explicit_commit_invalid")
    _reject_git_storage_indirection(repository)
    head_raw = _anchored_git_output(
        repository,
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
        limit=256,
    )
    try:
        head = head_raw.decode("ascii").strip().lower()
    except UnicodeDecodeError:
        raise HandoffVerificationError("repository_invalid") from None
    if head != commit:
        raise HandoffVerificationError("explicit_commit_not_head")
    _reject_nonstandalone_git_state(repository, commit)
    _require_exact_object_database(repository, commit)
    return repository.path


def _git_tree(
    repository: _AnchoredRepository, commit: str
) -> tuple[_GitTreeBlob, ...]:
    raw = _anchored_git_output(
        repository,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
        limit=MAX_GIT_TREE_OUTPUT_BYTES,
    )
    blobs: list[_GitTreeBlob] = []
    seen: set[bytes] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        if len(blobs) >= MAX_GIT_TREE_RECORDS:
            raise HandoffVerificationError("git_tree_limit")
        try:
            header, path = record.split(b"\t", 1)
            mode, kind, oid = header.split(b" ", 2)
        except ValueError:
            raise HandoffVerificationError("git_tree_invalid") from None
        if (
            mode not in {b"100644", b"100755"}
            or kind != b"blob"
            or not _GIT_OBJECT_RE.fullmatch(oid)
            or not path
            or len(path) > MAX_GIT_PATH_BYTES
            or b"\n" in path
            or b"\r" in path
            or path in seen
        ):
            raise HandoffVerificationError("git_tree_invalid")
        seen.add(path)
        blobs.append(_GitTreeBlob(path, oid.decode("ascii"), mode == b"100755"))
    if not blobs:
        raise HandoffVerificationError("git_tree_invalid")
    return tuple(blobs)


def _index_matches_tree(
    repository: _AnchoredRepository, tree: tuple[_GitTreeBlob, ...]
) -> bool:
    raw = _anchored_git_output(
        repository,
        "ls-files",
        "--cached",
        "--stage",
        "-z",
        limit=MAX_GIT_TREE_OUTPUT_BYTES,
    )
    index_entries: list[tuple[bytes, str, bool]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path = record.split(b"\t", 1)
            mode, oid, stage = header.split(b" ", 2)
        except ValueError:
            return False
        if (
            mode not in {b"100644", b"100755"}
            or not _GIT_OBJECT_RE.fullmatch(oid)
            or stage != b"0"
            or not path
            or len(path) > MAX_GIT_PATH_BYTES
        ):
            return False
        index_entries.append((path, oid.decode("ascii"), mode == b"100755"))
    expected = tuple((blob.path, blob.oid, blob.executable) for blob in tree)
    if tuple(index_entries) != expected:
        return False

    flags = _anchored_git_output(
        repository,
        "ls-files",
        "--cached",
        "-v",
        "-z",
        limit=MAX_GIT_TREE_OUTPUT_BYTES,
    )
    flagged_paths: list[bytes] = []
    for record in flags.split(b"\0"):
        if not record:
            continue
        if not record.startswith(b"H "):
            return False
        flagged_paths.append(record[2:])
    return tuple(flagged_paths) == tuple(blob.path for blob in tree)


def _git_blob(
    repository: _AnchoredRepository, oid: str, expected_size: int
) -> bytes:
    size_raw = _anchored_git_output(repository, "cat-file", "-s", oid, limit=64)
    try:
        size = int(size_raw.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        raise HandoffVerificationError("git_tree_invalid") from None
    if size != expected_size:
        raise HandoffVerificationError("git_tree_mismatch")
    return _anchored_git_output(
        repository, "cat-file", "blob", oid, limit=expected_size
    )


def _expected_worktree_directories(paths: Iterable[bytes]) -> frozenset[bytes]:
    directories: set[bytes] = set()
    for path in paths:
        parts = path.split(b"/")
        for index in range(1, len(parts)):
            directories.add(b"/".join(parts[:index]))
    return frozenset(directories)


def _worktree_snapshot(
    repository: _AnchoredRepository,
) -> tuple[dict[bytes, tuple[bool, bytes]], frozenset[bytes]]:
    """Read the worktree independently through anchored, no-follow descriptors."""

    files: dict[bytes, tuple[bool, bytes]] = {}
    directories: set[bytes] = set()
    entries_seen = 0
    bytes_seen = 0

    def walk(directory_fd: int, prefix: bytes) -> None:
        nonlocal entries_seen, bytes_seen
        before = os.fstat(directory_fd)
        try:
            with os.scandir(directory_fd) as iterator:
                names = tuple(sorted(entry.name for entry in iterator))
        except OSError:
            raise HandoffVerificationError("worktree_invalid") from None
        for name in names:
            if not prefix and name == ".git":
                continue
            entries_seen += 1
            if entries_seen > MAX_WORKTREE_ENTRIES:
                raise HandoffVerificationError("worktree_limit")
            if name in {"", ".", "..", ".git"} or "/" in name or "\0" in name:
                raise HandoffVerificationError("worktree_invalid")
            try:
                encoded_name = name.encode("utf-8", errors="strict")
                named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except (OSError, UnicodeError):
                raise HandoffVerificationError("worktree_invalid") from None
            relative = encoded_name if not prefix else prefix + b"/" + encoded_name
            if not relative or len(relative) > MAX_GIT_PATH_BYTES:
                raise HandoffVerificationError("worktree_invalid")
            if stat.S_ISDIR(named.st_mode):
                directories.add(relative)
                child_fd: int | None = None
                try:
                    child_fd = _open_directory_at(directory_fd, name, named)
                    walk(child_fd, relative)
                    if not _same_entry(named, os.fstat(child_fd)):
                        raise HandoffVerificationError("worktree_changed")
                except HandoffVerificationError:
                    raise
                except OSError:
                    raise HandoffVerificationError("worktree_changed") from None
                finally:
                    if child_fd is not None:
                        os.close(child_fd)
                continue
            if not stat.S_ISREG(named.st_mode) or named.st_nlink != 1:
                raise HandoffVerificationError("worktree_invalid")
            mode = stat.S_IMODE(named.st_mode)
            if mode not in {0o644, 0o755}:
                raise HandoffVerificationError("worktree_invalid")
            if len(files) >= MAX_WORKTREE_FILES:
                raise HandoffVerificationError("worktree_limit")
            try:
                content, stable = _read_regular_at(
                    directory_fd,
                    name,
                    max_bytes=MAX_WORKTREE_BYTES - bytes_seen,
                    require_single_link=True,
                )
            except OverflowError:
                raise HandoffVerificationError("worktree_limit") from None
            except OSError:
                raise HandoffVerificationError("worktree_changed") from None
            if stat.S_IMODE(stable.st_mode) != mode or relative in files:
                raise HandoffVerificationError("worktree_changed")
            bytes_seen += len(content)
            files[relative] = (mode == 0o755, content)
        try:
            after = os.fstat(directory_fd)
        except OSError:
            raise HandoffVerificationError("worktree_changed") from None
        if not _same_entry(before, after):
            raise HandoffVerificationError("worktree_changed")

    if not _repository_anchor_valid(repository):
        raise HandoffVerificationError("repository_changed")
    walk(repository.root_fd, b"")
    if not _repository_anchor_valid(repository):
        raise HandoffVerificationError("repository_changed")
    return files, frozenset(directories)


def _worktree_matches_expected(
    repository: _AnchoredRepository,
    expected: Mapping[bytes, tuple[bool, bytes]],
) -> bool:
    files, directories = _worktree_snapshot(repository)
    return files == expected and directories == _expected_worktree_directories(expected)


def verify_public_release_handoff(
    candidate_root: str | Path,
    repository_root: str | Path,
    commit: str,
    *,
    environ: dict[str, str] | None = None,
) -> PublicReleaseHandoffReport:
    """Compare one immutable publication candidate with one clean public HEAD."""

    try:
        extra_deny_literals = extra_deny_literals_from_env(environ)
    except PreflightConfigError:
        raise HandoffVerificationError("private_review_invalid") from None
    try:
        opened = _open_candidate_root(candidate_root)
    except (OSError, UnicodeError, ValueError):
        raise HandoffVerificationError("candidate_invalid") from None
    if opened is None:
        raise HandoffVerificationError("candidate_invalid")
    candidate_path, candidate_fd, candidate_stat = opened
    try:
        before = _candidate_state_fd(candidate_fd, profile=CANDIDATE_PROFILE_PUBLICATION)
        if before is None:
            raise HandoffVerificationError("candidate_invalid")
        try:
            _publication_metadata(before.blobs, extra_deny_literals)
        except CandidateBuildError:
            raise HandoffVerificationError("candidate_not_publication_ready") from None
        first_preflight = _scan_release_tree_fd(
            candidate_fd,
            extra_deny_literals=extra_deny_literals,
        )
        if (
            not first_preflight.ok
            or not _candidate_tree_has_finalized_modes_fd(candidate_fd)
            or not _candidate_root_path_matches_fd(candidate_path, candidate_fd, candidate_stat)
        ):
            raise HandoffVerificationError("candidate_invalid")

        repository: _AnchoredRepository | None = None
        try:
            repository = _open_repository(repository_root)
            _validated_repository(repository, commit)
            actual_tree = _git_tree(repository, commit)
            if not _index_matches_tree(repository, actual_tree):
                raise HandoffVerificationError("repository_not_clean")
            expected: dict[bytes, tuple[bool, bytes]] = {
                blob.path.encode("utf-8"): (blob.executable, blob.content)
                for blob in before.blobs
            }
            expected[PUBLIC_CANDIDATE_MARKER_NAME.encode("ascii")] = (
                False,
                before.marker_content,
            )
            actual = {blob.path: blob for blob in actual_tree}
            if frozenset(actual) != frozenset(expected):
                raise HandoffVerificationError("git_tree_mismatch")

            total_bytes = 0
            for path in sorted(expected):
                expected_executable, expected_content = expected[path]
                tree_blob = actual[path]
                if tree_blob.executable is not expected_executable:
                    raise HandoffVerificationError("git_tree_mismatch")
                content = _git_blob(repository, tree_blob.oid, len(expected_content))
                if content != expected_content:
                    raise HandoffVerificationError("git_tree_mismatch")
                total_bytes += len(content)
            if not _worktree_matches_expected(repository, expected):
                raise HandoffVerificationError("worktree_mismatch")

            _validated_repository(repository, commit)
            tree_after = _git_tree(repository, commit)
            if (
                tree_after != actual_tree
                or not _index_matches_tree(repository, tree_after)
                or not _worktree_matches_expected(repository, expected)
                or not _repository_anchor_valid(repository)
            ):
                raise HandoffVerificationError("repository_changed")
        finally:
            if repository is not None:
                for descriptor in (
                    repository.git_directory_fd,
                    repository.root_fd,
                ):
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass

        after = _candidate_state_fd(candidate_fd, profile=CANDIDATE_PROFILE_PUBLICATION)
        second_preflight = _scan_release_tree_fd(
            candidate_fd,
            extra_deny_literals=extra_deny_literals,
        )
        if (
            after is None
            or before != after
            or not second_preflight.ok
            or not _candidate_tree_has_finalized_modes_fd(candidate_fd)
            or not _candidate_root_path_matches_fd(candidate_path, candidate_fd, candidate_stat)
        ):
            raise HandoffVerificationError("candidate_changed")

        return PublicReleaseHandoffReport(
            ok=True,
            public_commit=commit,
            source_commit=before.marker.source_commit,
            manifest_digest=before.marker.manifest_digest,
            files_compared=len(expected),
            bytes_compared=total_bytes,
        )
    except HandoffVerificationError:
        raise
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        raise HandoffVerificationError("handoff_verification_failed") from None
    finally:
        try:
            os.close(candidate_fd)
        except OSError:
            pass


def _failure_payload(code: str) -> dict[str, object]:
    return {
        "ok": False,
        "error_code": code,
        "paths_included": False,
        "private_content_included": False,
        "writes_files": False,
        "changes_permissions": False,
        "initializes_git": False,
        "stages_files": False,
        "creates_commits": False,
        "publishes": False,
        "authorizes_publication": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only verification of an immutable candidate against a clean public Git HEAD."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_public_release_handoff(
            args.candidate,
            args.repository,
            args.commit,
        )
    except HandoffVerificationError as exc:
        print(json.dumps(_failure_payload(exc.code), sort_keys=True, separators=(",", ":")))
        return 2
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError):
        print(
            json.dumps(
                _failure_payload("handoff_verification_failed"),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(report.summary_json_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
