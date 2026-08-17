from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from unittest import mock
import uuid

from jarvis_v2.scripts.public_release_candidate import (
    CANDIDATE_PROFILE_PUBLICATION,
    build_public_candidate,
)
from jarvis_v2.scripts.public_release_handoff import (
    HandoffVerificationError,
    _run_git_bounded,
    main as handoff_main,
    verify_public_release_handoff,
)
from jarvis_v2.scripts import public_release_handoff as handoff_module
from jarvis_v2.scripts.public_release_preflight import EXTRA_DENY_ENV


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_MARKER = "synthetic-runtime-only-owner-review-marker"
DENY_ENVIRONMENT = {EXTRA_DENY_ENV: json.dumps([PRIVATE_MARKER])}
PROCESS_CLEANUP_TEST_DEADLINE_SECONDS = 10.0
PROCESS_PROGRESS_OBSERVATION_SECONDS = 0.5


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", os.fspath(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise SystemExit("synthetic Git fixture failed")
    return result.stdout


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Synthetic Maintainer",
        "-c",
        "user.email=maintainer@example.com",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD").decode("ascii").strip()


def _prune_unreachable(root: Path) -> None:
    _git(root, "reflog", "expire", "--expire=now", "--all")
    _git(root, "prune", "--expire=now")


def _amend(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Synthetic Maintainer",
        "-c",
        "user.email=maintainer@example.com",
        "commit",
        "--quiet",
        "--amend",
        "-m",
        message,
    )
    commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    _prune_unreachable(root)
    return commit


def _source_fixture(root: Path) -> None:
    (root / "jarvis_v2").mkdir()
    (root / "jarvis_v2/__init__.py").write_text("", encoding="utf-8")
    (root / "jarvis_v2/module.py").write_text("VALUE = 'public fixture'\n", encoding="utf-8")
    launcher = root / "launch_jarvis_v3.py"
    launcher.write_text("#!/usr/bin/env python3\nprint('synthetic')\n", encoding="utf-8")
    launcher.chmod(0o755)
    (root / "LICENSE").write_text(
        "Synthetic reviewed license terms used only by the handoff smoke fixture.\n",
        encoding="utf-8",
    )
    (root / "VERSION").write_text("3.0.0-rc.1\n", encoding="ascii")
    (root / ".gitignore").write_text(".env\n", encoding="utf-8")
    _git(root, "init", "--quiet")
    _git(root, "config", "core.logAllRefUpdates", "false")
    _commit(root, "synthetic publication source")


def _remove_candidate(candidate: Path) -> None:
    if not candidate.exists():
        return
    for path in sorted(candidate.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)
    candidate.chmod(0o700)
    shutil.rmtree(candidate)


@contextlib.contextmanager
def _publication_candidate():
    with TemporaryDirectory(prefix="jarvis-handoff-source-", dir="/private/tmp") as temp:
        source = Path(temp)
        _source_fixture(source)
        candidate = Path("/private/tmp") / f"jarvis-handoff-candidate-{uuid.uuid4().hex}"
        try:
            report = build_public_candidate(
                source,
                candidate,
                environ=DENY_ENVIRONMENT,
                profile=CANDIDATE_PROFILE_PUBLICATION,
            )
            if not report.ok or not report.candidate_finalized_immutable:
                raise SystemExit("synthetic publication candidate was not finalized")
            yield candidate
        finally:
            _remove_candidate(candidate)


def _copy_candidate_to_repository(candidate: Path, repository: Path) -> None:
    repository.mkdir()
    for source in sorted(candidate.rglob("*")):
        relative = source.relative_to(candidate)
        destination = repository / relative
        if source.is_dir():
            destination.mkdir()
            destination.chmod(0o700)
            continue
        content = source.read_bytes()
        destination.write_bytes(content)
        destination.chmod(0o755 if source.stat().st_mode & 0o111 else 0o644)
    _git(repository, "init", "--quiet")
    _git(repository, "config", "core.logAllRefUpdates", "false")


def _prepared_repository(candidate: Path, root: Path, name: str) -> tuple[Path, str]:
    repository = root / name
    _copy_candidate_to_repository(candidate, repository)
    return repository, _commit(repository, "synthetic public handoff")


def _snapshot(root: Path) -> tuple[tuple[str, int, int, int, str], ...]:
    rows: list[tuple[str, int, int, int, str]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: os.fspath(item)):
        info = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        rows.append(
            (
                "." if path == root else path.relative_to(root).as_posix(),
                stat.S_IMODE(info.st_mode),
                info.st_size,
                info.st_mtime_ns,
                digest,
            )
        )
    return tuple(rows)


def _expect_failure(candidate: Path, repository: Path, commit: str, code: str) -> None:
    try:
        verify_public_release_handoff(
            candidate,
            repository,
            commit,
            environ=DENY_ENVIRONMENT,
        )
    except HandoffVerificationError as exc:
        if exc.code != code:
            raise SystemExit(f"handoff failure code drifted: {exc.code}")
    else:
        raise SystemExit(f"handoff accepted synthetic {code} fixture")


def test_exact_clean_commit_passes_without_mutation_or_disclosure() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-public-", dir="/private/tmp"
    ) as temp:
        repository, commit = _prepared_repository(candidate, Path(temp), "public")
        candidate_before = _snapshot(candidate)
        repository_before = _snapshot(repository)
        report = verify_public_release_handoff(
            candidate,
            repository,
            commit,
            environ=DENY_ENVIRONMENT,
        )
        if not report.ok or report.files_compared < 5 or report.bytes_compared <= 0:
            raise SystemExit(f"exact public handoff did not pass: {report}")
        summary = report.summary_json_payload()
        required_true = {
            "candidate_valid",
            "candidate_finalized_immutable",
            "repository_clean",
            "head_matches_explicit_commit",
            "single_root_commit",
            "history_free",
            "refs_isolated",
            "remotes_absent",
            "replace_refs_absent",
            "promisor_disabled",
            "repository_descriptor_anchored",
            "git_metadata_no_follow",
            "common_git_directory_absent",
            "reflogs_absent",
            "object_database_exact",
            "git_process_groups_cleaned",
            "index_and_worktree_match_commit",
            "worktree_bytes_match_candidate",
            "git_tree_matches_candidate",
            "marker_matches_candidate",
        }
        required_false = {
            "paths_included",
            "private_content_included",
            "private_deny_literals_retained",
            "writes_files",
            "changes_permissions",
            "initializes_git",
            "stages_files",
            "creates_commits",
            "publishes",
            "authorizes_publication",
        }
        if any(summary.get(key) is not True for key in required_true) or any(
            summary.get(key) is not False for key in required_false
        ):
            raise SystemExit(f"handoff summary boundary drifted: {summary}")
        if _snapshot(candidate) != candidate_before or _snapshot(repository) != repository_before:
            raise SystemExit("read-only handoff verification mutated candidate or repository")


def test_tree_mismatches_and_marker_drift_fail_closed() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-mismatch-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)

        extra_repo, _ = _prepared_repository(candidate, root, "extra")
        (extra_repo / "EXTRA.txt").write_text("unexpected\n", encoding="utf-8")
        extra_commit = _amend(extra_repo, "extra file")
        _expect_failure(candidate, extra_repo, extra_commit, "git_tree_mismatch")

        missing_repo, _ = _prepared_repository(candidate, root, "missing")
        (missing_repo / "jarvis_v2/module.py").unlink()
        missing_commit = _amend(missing_repo, "missing file")
        _expect_failure(candidate, missing_repo, missing_commit, "git_tree_mismatch")

        content_repo, _ = _prepared_repository(candidate, root, "content")
        (content_repo / "jarvis_v2/module.py").write_text("VALUE = 'changed'\n", encoding="utf-8")
        content_commit = _amend(content_repo, "changed content")
        _expect_failure(candidate, content_repo, content_commit, "git_tree_mismatch")

        mode_repo, _ = _prepared_repository(candidate, root, "mode")
        (mode_repo / "jarvis_v2/module.py").chmod(0o755)
        mode_commit = _amend(mode_repo, "changed executable bit")
        _expect_failure(candidate, mode_repo, mode_commit, "git_tree_mismatch")

        marker_repo, _ = _prepared_repository(candidate, root, "marker")
        marker = marker_repo / "PUBLIC_CANDIDATE_MANIFEST.json"
        marker.write_bytes(marker.read_bytes() + b" ")
        marker_commit = _amend(marker_repo, "changed candidate marker")
        _expect_failure(candidate, marker_repo, marker_commit, "git_tree_mismatch")


def test_dirty_repository_and_non_head_commit_are_rejected() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-dirty-", dir="/private/tmp"
    ) as temp:
        repository, commit = _prepared_repository(candidate, Path(temp), "public")
        (repository / ".env").write_text("PRIVATE=hidden\n", encoding="utf-8")
        (repository / ".env").chmod(0o644)
        _expect_failure(candidate, repository, commit, "worktree_mismatch")
        (repository / ".env").unlink()

        module = repository / "jarvis_v2/module.py"
        module.write_text("VALUE = 'dirty'\n", encoding="utf-8")
        _git(repository, "add", "jarvis_v2/module.py")
        _expect_failure(
            candidate,
            repository,
            commit,
            "repository_object_inventory_invalid",
        )
        _git(repository, "restore", "--staged", "--worktree", "jarvis_v2/module.py")
        module.chmod(0o644)
        _prune_unreachable(repository)

        _git(repository, "update-index", "--assume-unchanged", "jarvis_v2/module.py")
        module.write_text("VALUE = 'stat-hidden-dirty'\n", encoding="utf-8")
        _expect_failure(candidate, repository, commit, "repository_not_clean")
        _git(repository, "update-index", "--no-assume-unchanged", "jarvis_v2/module.py")
        _git(repository, "restore", "--worktree", "jarvis_v2/module.py")
        module.chmod(0o644)

        _expect_failure(candidate, repository, "a" * len(commit), "explicit_commit_not_head")

        real_git_output = handoff_module._anchored_git_output
        commands: list[str] = []

        def record_git_commands(repository_anchor, *arguments, **kwargs):
            if arguments:
                commands.append(arguments[0])
            return real_git_output(repository_anchor, *arguments, **kwargs)

        with mock.patch.object(
            handoff_module,
            "_anchored_git_output",
            side_effect=record_git_commands,
        ):
            result = verify_public_release_handoff(
                candidate,
                repository,
                commit,
                environ=DENY_ENVIRONMENT,
            )
        if not result.ok or "status" in commands:
            raise SystemExit("handoff invoked git status instead of exact no-follow checks")


def test_nonstandalone_git_state_is_rejected() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-isolation-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)

        history_repo, history_commit = _prepared_repository(candidate, root, "history")
        _git(
            history_repo,
            "-c",
            "user.name=Synthetic Maintainer",
            "-c",
            "user.email=maintainer@example.com",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "second commit with the same tree",
        )
        second_commit = _git(history_repo, "rev-parse", "HEAD").decode("ascii").strip()
        if second_commit == history_commit:
            raise SystemExit("history fixture did not create a second commit")
        _expect_failure(
            candidate,
            history_repo,
            second_commit,
            "repository_history_not_isolated",
        )

        refs_repo, refs_commit = _prepared_repository(candidate, root, "refs")
        _git(refs_repo, "branch", "extra-ref", refs_commit)
        _expect_failure(candidate, refs_repo, refs_commit, "repository_refs_invalid")

        replace_repo, replace_commit = _prepared_repository(candidate, root, "replace")
        tree = _git(replace_repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
        replacement = _git(
            replace_repo,
            "-c",
            "user.name=Synthetic Maintainer",
            "-c",
            "user.email=maintainer@example.com",
            "commit-tree",
            tree,
            "-m",
            "synthetic replacement",
        ).decode("ascii").strip()
        _git(replace_repo, "replace", replace_commit, replacement)
        _expect_failure(candidate, replace_repo, replace_commit, "repository_refs_invalid")

        promisor_repo, promisor_commit = _prepared_repository(candidate, root, "promisor")
        _git(promisor_repo, "config", "remote.synthetic.promisor", "true")
        _expect_failure(
            candidate,
            promisor_repo,
            promisor_commit,
            "repository_promisor_configured",
        )

        remote_repo, remote_commit = _prepared_repository(candidate, root, "remote")
        _git(remote_repo, "remote", "add", "origin", "https://example.invalid/public.git")
        _expect_failure(candidate, remote_repo, remote_commit, "repository_remote_present")

        shallow_repo, shallow_commit = _prepared_repository(candidate, root, "shallow")
        (shallow_repo / ".git/shallow").write_text(shallow_commit + "\n", encoding="ascii")
        _expect_failure(
            candidate,
            shallow_repo,
            shallow_commit,
            "repository_metadata_invalid",
        )

        alternate_repo, alternate_commit = _prepared_repository(candidate, root, "alternate")
        (alternate_repo / ".git/objects/info/alternates").write_text(
            "/private/tmp/synthetic-object-store\n",
            encoding="utf-8",
        )
        _expect_failure(
            candidate,
            alternate_repo,
            alternate_commit,
            "repository_metadata_invalid",
        )


def test_independent_worktree_read_rejects_git_invisible_state() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-worktree-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)

        hardlink_repo, hardlink_commit = _prepared_repository(candidate, root, "hardlink")
        module = hardlink_repo / "jarvis_v2/module.py"
        outside = root / "same-content-outside-repository"
        outside.write_bytes(module.read_bytes())
        outside.chmod(0o644)
        module.unlink()
        os.link(outside, module)
        _expect_failure(candidate, hardlink_repo, hardlink_commit, "worktree_invalid")

        empty_repo, empty_commit = _prepared_repository(candidate, root, "empty-directory")
        (empty_repo / "unexpected-empty-directory").mkdir()
        _expect_failure(candidate, empty_repo, empty_commit, "worktree_mismatch")

        mode_repo, mode_commit = _prepared_repository(candidate, root, "group-mode")
        (mode_repo / "jarvis_v2/module.py").chmod(0o640)
        _expect_failure(candidate, mode_repo, mode_commit, "worktree_invalid")


def test_redirected_git_metadata_reflogs_and_unreachable_objects_are_rejected() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-metadata-", dir="/private/tmp"
    ) as temp:
        root = Path(temp)

        common_repo, common_commit = _prepared_repository(candidate, root, "commondir")
        (common_repo / ".git/commondir").write_text(
            "../external-common-directory\n",
            encoding="utf-8",
        )
        _expect_failure(
            candidate,
            common_repo,
            common_commit,
            "repository_common_directory_present",
        )

        objects_repo, objects_commit = _prepared_repository(candidate, root, "objects-link")
        external_objects = root / "external-objects"
        (objects_repo / ".git/objects").rename(external_objects)
        (objects_repo / ".git/objects").symlink_to(external_objects, target_is_directory=True)
        _expect_failure(
            candidate,
            objects_repo,
            objects_commit,
            "repository_metadata_invalid",
        )

        refs_repo, refs_commit = _prepared_repository(candidate, root, "refs-link")
        external_refs = root / "external-refs"
        (refs_repo / ".git/refs").rename(external_refs)
        (refs_repo / ".git/refs").symlink_to(external_refs, target_is_directory=True)
        _expect_failure(
            candidate,
            refs_repo,
            refs_commit,
            "repository_metadata_invalid",
        )

        head_repo, head_commit = _prepared_repository(candidate, root, "head-link")
        external_head = root / "external-head"
        (head_repo / ".git/HEAD").rename(external_head)
        (head_repo / ".git/HEAD").symlink_to(external_head)
        _expect_failure(
            candidate,
            head_repo,
            head_commit,
            "repository_metadata_invalid",
        )

        info_repo, info_commit = _prepared_repository(candidate, root, "info-link")
        external_info = root / "external-info-directory"
        (info_repo / ".git/info").rename(external_info)
        (info_repo / ".git/info").symlink_to(external_info, target_is_directory=True)
        _expect_failure(
            candidate,
            info_repo,
            info_commit,
            "repository_metadata_invalid",
        )

        info_file_repo, info_file_commit = _prepared_repository(
            candidate, root, "info-file-link"
        )
        exclude = info_file_repo / ".git/info/exclude"
        external_exclude = root / "external-info-exclude"
        exclude.rename(external_exclude)
        exclude.symlink_to(external_exclude)
        _expect_failure(
            candidate,
            info_file_repo,
            info_file_commit,
            "repository_metadata_invalid",
        )

        info_hardlink_repo, info_hardlink_commit = _prepared_repository(
            candidate, root, "info-file-hardlink"
        )
        hardlinked_exclude = info_hardlink_repo / ".git/info/exclude"
        external_hardlink = root / "external-info-hardlink"
        external_hardlink.write_bytes(hardlinked_exclude.read_bytes())
        hardlinked_exclude.unlink()
        os.link(external_hardlink, hardlinked_exclude)
        _expect_failure(
            candidate,
            info_hardlink_repo,
            info_hardlink_commit,
            "repository_metadata_invalid",
        )

        info_special_repo, info_special_commit = _prepared_repository(
            candidate, root, "info-special-entry"
        )
        os.mkfifo(info_special_repo / ".git/info/synthetic-fifo")
        _expect_failure(
            candidate,
            info_special_repo,
            info_special_commit,
            "repository_metadata_invalid",
        )

        attributes_repo, attributes_commit = _prepared_repository(
            candidate, root, "info-attributes"
        )
        (attributes_repo / ".git/info/attributes").write_text(
            "* filter=synthetic-proof\n",
            encoding="utf-8",
        )
        _expect_failure(
            candidate,
            attributes_repo,
            attributes_commit,
            "repository_metadata_invalid",
        )

        filter_config_repo, filter_config_commit = _prepared_repository(
            candidate, root, "filter-config"
        )
        _git(
            filter_config_repo,
            "config",
            "filter.synthetic-clean.clean",
            "/private/tmp/synthetic-filter-command-that-must-not-run",
        )
        _git(
            filter_config_repo,
            "config",
            "filter.synthetic-process.process",
            "/private/tmp/synthetic-filter-process-that-must-not-run",
        )
        _expect_failure(
            candidate,
            filter_config_repo,
            filter_config_commit,
            "repository_metadata_invalid",
        )

        filter_repo, filter_commit = _prepared_repository(
            candidate, root, "configured-filter"
        )
        filter_sentinel = root / "configured-filter-command-ran"
        filter_program = root / "synthetic-filter-command.py"
        filter_program.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            f"Path({os.fspath(filter_sentinel)!r}).write_text('ran', encoding='ascii')\n"
            "import sys\n"
            "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
            encoding="utf-8",
        )
        filter_program.chmod(0o700)
        _git(
            filter_repo,
            "config",
            "filter.synthetic-proof.clean",
            os.fspath(filter_program),
        )
        (filter_repo / ".git/info/attributes").write_text(
            "* filter=synthetic-proof\n",
            encoding="utf-8",
        )
        _expect_failure(
            candidate,
            filter_repo,
            filter_commit,
            "repository_metadata_invalid",
        )
        if filter_sentinel.exists():
            raise SystemExit("configured Git content filter command was executed")

        reflog_repo, reflog_commit = _prepared_repository(candidate, root, "reflog")
        (reflog_repo / ".git/logs").mkdir()
        (reflog_repo / ".git/logs/HEAD").write_text(
            f"{'0' * len(reflog_commit)} {reflog_commit} synthetic private reflog\n",
            encoding="ascii",
        )
        _expect_failure(
            candidate,
            reflog_repo,
            reflog_commit,
            "repository_reflog_present",
        )

        unreachable_repo, unreachable_commit = _prepared_repository(
            candidate, root, "unreachable"
        )
        private_unreachable = root / "private-unreachable-object-source"
        private_unreachable.write_text(
            "synthetic private unreachable object\n",
            encoding="utf-8",
        )
        _git(
            unreachable_repo,
            "hash-object",
            "-w",
            os.fspath(private_unreachable),
        )
        _expect_failure(
            candidate,
            unreachable_repo,
            unreachable_commit,
            "repository_object_inventory_invalid",
        )


def test_anchored_git_environment_disables_replacement_and_lazy_fetch() -> None:
    environment = handoff_module._git_environment(7, 8)
    if (
        environment.get("GIT_NO_REPLACE_OBJECTS") != "1"
        or environment.get("GIT_NO_LAZY_FETCH") != "1"
        or environment.get("GIT_DIR") != ".git"
        or environment.get("GIT_WORK_TREE") != "."
    ):
        raise SystemExit(f"anchored Git environment drifted: {environment}")


def test_cli_errors_are_content_free_and_inert() -> None:
    hidden_candidate = "/private/tmp/private-candidate-owner-marker"
    hidden_repository = "/private/tmp/private-repository-contact-marker"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = handoff_main(
            [
                "--candidate",
                hidden_candidate,
                "--repository",
                hidden_repository,
                "--commit",
                "not-a-commit",
            ]
        )
    rendered = output.getvalue()
    payload = json.loads(rendered)
    if code != 2 or payload.get("ok") is not False:
        raise SystemExit(f"handoff CLI did not fail safely: {payload}")
    if hidden_candidate in rendered or hidden_repository in rendered or PRIVATE_MARKER in rendered:
        raise SystemExit("handoff CLI leaked a supplied path or private deny literal")
    for key in (
        "paths_included",
        "private_content_included",
        "writes_files",
        "changes_permissions",
        "initializes_git",
        "stages_files",
        "creates_commits",
        "publishes",
        "authorizes_publication",
    ):
        if payload.get(key) is not False:
            raise SystemExit(f"handoff CLI failure overstated boundary: {payload}")
    if Path(hidden_candidate).exists() or Path(hidden_repository).exists():
        raise SystemExit("handoff CLI created a supplied path")


def test_public_api_suppresses_path_bearing_exception_chains() -> None:
    with _publication_candidate() as candidate, TemporaryDirectory(
        prefix="jarvis-handoff-private-trace-", dir="/private/tmp"
    ) as temp:
        hidden_repository = Path(temp) / "owner-contact-private-repository-marker"
        try:
            verify_public_release_handoff(
                candidate,
                hidden_repository,
                "a" * 40,
                environ=DENY_ENVIRONMENT,
            )
        except HandoffVerificationError as exc:
            rendered = "".join(traceback.format_exception(exc))
            if (
                exc.__cause__ is not None
                or not exc.__suppress_context__
                or os.fspath(hidden_repository) in rendered
                or PRIVATE_MARKER in rendered
            ):
                raise SystemExit("public handoff exception retained a private cause or path")
        else:
            raise SystemExit("missing private repository did not fail closed")


def _progress_snapshot(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None


def _assert_process_progress_stopped(
    progress_path: Path,
    *,
    process_group_id: int,
    label: str,
) -> None:
    before = _progress_snapshot(progress_path)
    if before is None:
        raise SystemExit(f"{label} never published progress")
    time.sleep(PROCESS_PROGRESS_OBSERVATION_SECONDS)
    after = _progress_snapshot(progress_path)
    if before == after:
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise SystemExit(f"{label} continued executing after cleanup")


def _wait_for_process_progress(
    progress_path: Path,
    *,
    process_group_id: int,
    label: str,
) -> None:
    deadline = time.monotonic() + PROCESS_CLEANUP_TEST_DEADLINE_SECONDS
    while _progress_snapshot(progress_path) is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if _progress_snapshot(progress_path) is not None:
        return
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise SystemExit(f"{label} never published progress")


def test_bounded_git_failures_stop_the_entire_child_process_group() -> None:
    parent_script = """
import pathlib
import signal
import subprocess
import sys
import time

child = None

def stop_parent(_signum, _frame):
    if child is None:
        raise SystemExit(0)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise SystemExit(1)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop_parent)
progress = pathlib.Path(sys.argv[1])
child_code = (
    "import pathlib,sys,time\\n"
    "progress=pathlib.Path(sys.argv[1])\\n"
    "for index in range(1200):\\n"
    "    progress.write_text(str(index),encoding='ascii')\\n"
    "    time.sleep(0.05)\\n"
)
child = subprocess.Popen(
    [sys.executable, "-B", "-c", child_code, str(progress)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
deadline = time.monotonic() + 5
while not progress.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
if not progress.exists():
    raise SystemExit("descendant failed to start")
if sys.argv[2] == "overflow":
    print("X" * 4096, flush=True)
time.sleep(60)
""".strip()
    real_popen = subprocess.Popen
    for mode in ("overflow", "timeout"):
        with TemporaryDirectory(prefix=f"jarvis-handoff-group-{mode}-") as temp:
            progress_file = Path(temp) / "child.progress"
            preexec_seen = False
            synthetic_process: subprocess.Popen[bytes] | None = None

            def synthetic_popen(_command, **kwargs):
                nonlocal preexec_seen, synthetic_process
                preexec_seen = "preexec_fn" in kwargs
                synthetic_process = real_popen(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        parent_script,
                        str(progress_file),
                        mode,
                    ],
                    **kwargs,
                )
                _wait_for_process_progress(
                    progress_file,
                    process_group_id=synthetic_process.pid,
                    label=f"bounded Git {mode} descendant",
                )
                return synthetic_process

            timeout = 2.0 if mode == "overflow" else 0.2
            with mock.patch.object(
                handoff_module.subprocess,
                "Popen",
                side_effect=synthetic_popen,
            ), mock.patch.object(handoff_module, "GIT_TIMEOUT_SECONDS", timeout):
                repository_fd = os.open(temp, os.O_RDONLY)
                git_directory_fd = os.open(temp, os.O_RDONLY)
                try:
                    try:
                        _run_git_bounded(
                            Path(temp),
                            ("status",),
                            max_output_bytes=1,
                            repository_fd=repository_fd,
                            git_directory_fd=git_directory_fd,
                        )
                    except HandoffVerificationError as exc:
                        expected = (
                            "git_inspection_limit"
                            if mode == "overflow"
                            else "git_inspection_failed"
                        )
                        if exc.code != expected:
                            raise SystemExit(
                                f"bounded Git cleanup failure drifted: {exc.code}"
                            )
                    else:
                        raise SystemExit(f"bounded Git {mode} fixture unexpectedly completed")
                finally:
                    os.close(git_directory_fd)
                    os.close(repository_fd)

            if preexec_seen:
                raise SystemExit("bounded Git launch retained preexec_fn deadlock risk")
            if synthetic_process is None:
                raise SystemExit("bounded Git cleanup fixture did not launch")
            _assert_process_progress_stopped(
                progress_file,
                process_group_id=synthetic_process.pid,
                label=f"bounded Git {mode} descendant",
            )

    with TemporaryDirectory(prefix="jarvis-handoff-cleanup-unknown-") as temp:
        progress_file = Path(temp) / "child.progress"
        cleanup_process: subprocess.Popen[bytes] | None = None

        def synthetic_popen(_command, **kwargs):
            nonlocal cleanup_process
            cleanup_process = real_popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    parent_script,
                    str(progress_file),
                    "overflow",
                ],
                **kwargs,
            )
            _wait_for_process_progress(
                progress_file,
                process_group_id=cleanup_process.pid,
                label="cleanup-unknown descendant",
            )
            return cleanup_process

        real_cleanup = handoff_module._terminate_process_group

        def cleanup_without_attestation(process):
            real_cleanup(process)
            return False

        with (
            mock.patch.object(
                handoff_module.subprocess,
                "Popen",
                side_effect=synthetic_popen,
            ),
            mock.patch.object(
                handoff_module,
                "_terminate_process_group",
                side_effect=cleanup_without_attestation,
            ),
        ):
            try:
                _run_git_bounded(Path(temp), ("status",), max_output_bytes=1)
            except HandoffVerificationError as exc:
                if exc.code != "git_process_cleanup_unknown":
                    raise SystemExit(
                        f"unknown Git cleanup attestation drifted: {exc.code}"
                    )
            else:
                raise SystemExit("unattested Git cleanup was accepted")
        if cleanup_process is None:
            raise SystemExit("cleanup-unknown fixture did not launch")
        _assert_process_progress_stopped(
            progress_file,
            process_group_id=cleanup_process.pid,
            label="cleanup-unknown descendant",
        )


def test_bounded_git_cleanup_escalates_to_sigkill() -> None:
    process = mock.Mock()
    process.pid = 424242
    process.wait.return_value = 0
    with (
        mock.patch.object(
            handoff_module,
            "_wait_process_group_exit",
            side_effect=(False, True),
        ),
        mock.patch.object(
            handoff_module,
            "_process_group_exists",
            return_value=False,
        ),
        mock.patch.object(handoff_module.os, "killpg") as killpg,
    ):
        cleaned = handoff_module._terminate_process_group(process)
    if not cleaned:
        raise SystemExit("bounded Git cleanup did not attest SIGKILL escalation")
    if killpg.call_args_list != [
        mock.call(process.pid, signal.SIGTERM),
        mock.call(process.pid, signal.SIGKILL),
    ]:
        raise SystemExit("bounded Git cleanup lost TERM-then-KILL escalation order")


def test_completed_git_process_group_settle_is_bounded_and_fail_closed() -> None:
    with (
        mock.patch.object(
            handoff_module,
            "_process_group_exists",
            side_effect=(True, False),
        ) as process_group_exists,
        mock.patch.object(handoff_module, "_terminate_process_group") as terminate,
    ):
        code, output = _run_git_bounded(Path("/private/tmp"), ("--version",))
    if code != 0 or not output.startswith(b"git version "):
        raise SystemExit("completed Git command lost its successful result while settling")
    if process_group_exists.call_count != 2 or terminate.called:
        raise SystemExit("completed Git process group was not allowed to settle naturally")

    with (
        mock.patch.object(
            handoff_module,
            "_process_group_exists",
            return_value=True,
        ),
        mock.patch.object(
            handoff_module,
            "_wait_process_group_exit",
            return_value=False,
        ) as wait_for_exit,
        mock.patch.object(
            handoff_module,
            "_terminate_process_group",
            return_value=True,
        ) as terminate,
    ):
        try:
            _run_git_bounded(Path("/private/tmp"), ("--version",))
        except HandoffVerificationError as exc:
            if exc.code != "git_inspection_failed":
                raise SystemExit(
                    f"persistent completed Git descendant failure drifted: {exc.code}"
                )
        else:
            raise SystemExit("persistent completed Git descendants were accepted")
    if wait_for_exit.call_count != 1 or terminate.call_count != 1:
        raise SystemExit("persistent completed Git descendants bypassed bounded cleanup")


def test_public_release_docs_state_one_truthful_handoff_boundary() -> None:
    precheck = (ROOT / "PUBLIC_RELEASE_PRECHECK.md").read_text(encoding="utf-8")
    development = (ROOT / "V3_DEVELOPMENT.md").read_text(encoding="utf-8")
    development_flat = " ".join(development.split())
    for expected in (
        "jarvis_v2.scripts.public_release_handoff",
        "--candidate /path/to/immutable-candidate",
        "--repository /path/to/prepared-public-repository",
        "--commit <full-public-commit-id>",
        "git_tree_matches_candidate:true",
        "does not initialize Git",
        "does not write",
        "normalizes only the executable bit",
    ):
        if expected not in precheck:
            raise SystemExit(f"public release handoff guide missed boundary: {expected}")
    if (
        "`V3_SUPERVISED_PROOF_RUNBOOK.md` is a sanitized public runbook included"
        not in development_flat
        or "runbook is intentionally excluded" in development_flat
    ):
        raise SystemExit("V3 development boundary contradicts the candidate runbook allowlist")


def main() -> None:
    test_exact_clean_commit_passes_without_mutation_or_disclosure()
    test_tree_mismatches_and_marker_drift_fail_closed()
    test_dirty_repository_and_non_head_commit_are_rejected()
    test_nonstandalone_git_state_is_rejected()
    test_independent_worktree_read_rejects_git_invisible_state()
    test_redirected_git_metadata_reflogs_and_unreachable_objects_are_rejected()
    test_anchored_git_environment_disables_replacement_and_lazy_fetch()
    test_cli_errors_are_content_free_and_inert()
    test_public_api_suppresses_path_bearing_exception_chains()
    test_bounded_git_failures_stop_the_entire_child_process_group()
    test_bounded_git_cleanup_escalates_to_sigkill()
    test_completed_git_process_group_settle_is_bounded_and_fail_closed()
    test_public_release_docs_state_one_truthful_handoff_boundary()
    print("public release handoff smoke passed")


if __name__ == "__main__":
    main()
