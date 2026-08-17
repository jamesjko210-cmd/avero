from __future__ import annotations

import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
)
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.tools.files import _file_handoff_metadata, _metadata_bool, find_files, list_files, read_text_file, write_text_file


def assert_file_metadata_safe(metadata: dict, label: str) -> None:
    for key in [
        "calls_model",
        "executes_tools",
        "queues_approval",
        "approves_request",
        "dismisses_request",
        "executes_side_effect",
        "writes_files",
        "writes_memory",
        "writes_notes",
        "controls_computer",
    ]:
        if metadata.get(key):
            raise SystemExit(f"{label} should not mark {key}: {metadata}")


def assert_local_productivity_read_recovery(result, label: str) -> None:
    action = LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid the canonical local recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"{label} local recovery declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "next_command": "setup check",
        "recovery_commands": ["setup check"],
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} local recovery field {key} drifted: {result.metadata}")


def assert_resource_not_found_recovery(result, label: str) -> None:
    action = RESOURCE_NOT_FOUND_RECOVERY_ACTION
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid canonical missing-resource recovery: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": [],
    }:
        raise SystemExit(f"{label} missing-resource declaration drifted: {result.metadata}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) != value:
            raise SystemExit(f"{label} missing-resource field {key} drifted: {result.metadata}")


def assert_path_raw_number_redaction(result, key: str, label: str) -> None:
    if result.metadata.get(key) != "<local-path>":
        raise SystemExit(f"{label} should redact path-shaped raw {key}: {result.metadata}")
    assert_file_metadata_safe(result.metadata, label)


def assert_file_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"file metadata bool should fail closed for {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("file metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("file metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("file metadata bool should honor the explicit default")


def assert_file_malformed_handoff_flags() -> None:
    handoff = {
        "source": "list_files",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "boundaries": {"read_only": True},
    }
    metadata = _file_handoff_metadata("list_files_handoff", handoff)
    if metadata.get("state_changed") is not False:
        raise SystemExit(f"malformed file state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False:
        raise SystemExit(f"malformed file content_in_handoff should fail closed: {metadata}")


def assert_file_planner_routes() -> None:
    planner = RuleBasedPlanner()
    expected = {
        "read README.md": ("read_text_file", {"path": "README.md"}),
        "read README.md please": ("read_text_file", {"path": "README.md"}),
        "read file README.md please": ("read_text_file", {"path": "README.md"}),
        "open package.json": ("read_text_file", {"path": "package.json"}),
        "open package.json please": ("read_text_file", {"path": "package.json"}),
        "list files please": ("list_files", {"directory": "."}),
        "show files please": ("list_files", {"directory": "."}),
        "list files in jarvis_v2 please": ("list_files", {"directory": "jarvis_v2"}),
        # Real gap found live 2026-07-09: "show me files in X" fell through to
        # chat because list_match only recognized "list"/"show" directly
        # followed by "files"/"folder"/"directory", with no "me"/"my" lead-in.
        "show me files in jarvis_v2 please": ("list_files", {"directory": "jarvis_v2"}),
        "show my files in jarvis_v2 please": ("list_files", {"directory": "jarvis_v2"}),
        # Real gap found live 2026-07-10, same class as the round-39/40/41
        # list-command bugs (a working verb-prefixed phrasing didn't imply
        # its bare, verb-less sibling worked): "list files"/"show my files"
        # worked, but bare "files"/"my files" (no "list"/"show" verb at all)
        # fell through to chat.
        "files": ("list_files", {"directory": "."}),
        "my files": ("list_files", {"directory": "."}),
        # Real gap found live 2026-07-09: "create a file X with Y" fell through
        # to chat because write_match only recognized the verb "write", not
        # "create"/"make".
        "create a file test.txt with hello world": ("write_text_file", {"path": "test.txt", "content": "hello world", "overwrite": False}),
        "make a file test.txt with hello world": ("write_text_file", {"path": "test.txt", "content": "hello world", "overwrite": False}),
        # Real gap found live 2026-07-10, same privacy-relevant misroute class
        # as the round-21/26 notes-search and round-27 task-search word-order
        # fixes: "search for the/a file about X" / "find the/a file about X"
        # (article BEFORE the noun) fell through to the generic public
        # web-search fallback instead of searching local files.
        "find the file about budget": ("find_files", {"pattern": "budget", "root": "."}),
        "search for the file about budget": ("find_files", {"pattern": "budget", "root": "."}),
        "search for a file about budget": ("find_files", {"pattern": "budget", "root": "."}),
        # Real gap found live 2026-07-10: "find file budget and then send it
        # to john" (a compound sentence) swallowed the whole second clause
        # into the search pattern instead of stopping at "budget" -- and
        # silently dropped the "send it to john" intent.
        "find file budget and then send it to john": ("find_files", {"pattern": "budget", "root": "."}),
        "find my receipt file": ("find_files", {"pattern": "receipt", "root": "."}),
        "find my receipt file please": ("find_files", {"pattern": "receipt", "root": "."}),
        "find receipt files": ("find_files", {"pattern": "receipt", "root": "."}),
        "find receipt files please": ("find_files", {"pattern": "receipt", "root": "."}),
        "find files receipt please": ("find_files", {"pattern": "receipt", "root": "."}),
        "search for files named receipt": ("find_files", {"pattern": "receipt", "root": "."}),
        "search for files named receipt please": ("find_files", {"pattern": "receipt", "root": "."}),
        "search files for receipt": ("find_files", {"pattern": "receipt", "root": "."}),
        "search files for receipt please": ("find_files", {"pattern": "receipt", "root": "."}),
        "search files for receipt in jarvis_v2 please": ("find_files", {"pattern": "receipt", "root": "jarvis_v2"}),
    }
    for command, (tool_name, args) in expected.items():
        plan = planner.plan(command)
        actual = [(action.tool_name, action.args) for action in plan.actions]
        if actual != [(tool_name, args)]:
            raise SystemExit(f"{command!r} should route to {tool_name}: {actual}")

    web_plan = planner.plan("search for local cafes")
    if [(action.tool_name, action.args) for action in web_plan.actions] != [("web_lookup", {"query": "local cafes"})]:
        raise SystemExit(f"ordinary web search should not route to file search: {web_plan.actions}")

    page_plan = planner.plan("read https://example.com")
    if [(action.tool_name, action.args) for action in page_plan.actions] != [("fetch_page", {"url": "https://example.com"})]:
        raise SystemExit(f"URL read should still route to fetch_page: {page_plan.actions}")


def assert_no_local_paths(value, label: str) -> None:
    text = repr(value)
    if "/private/" in text or "/\x55sers/" in text or "/var/folders/" in text or "/tmp/" in text:
        raise SystemExit(f"{label} handoff should not expose raw local paths: {text[:1000]}")


def assert_common_boundaries(boundaries: dict, *, label: str, read_only: bool, reads_private_data: bool, writes_files: bool) -> None:
    expected = {
        "read_only": read_only,
        "reads_private_data": reads_private_data,
        "writes_files": writes_files,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "controls_computer": False,
        "external_side_effect": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} boundary {key} should be {value}: {boundaries}")


def assert_file_contract(result, handoff: dict, *, label: str, state_changed: bool, changed: list[str], content_in_handoff: bool) -> None:
    metadata = result.metadata
    for key, value in [
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ]:
        if handoff.get(key) != value or metadata.get(key) != value:
            raise SystemExit(f"{label} file handoff {key} parity failed: {handoff} / {metadata}")


def assert_list_files_handoff(result, *, label: str) -> None:
    handoff = result.metadata.get("list_files_handoff")
    if not result.metadata.get("list_files_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready list_files_handoff: {result.metadata}")
    if handoff.get("source") != "list_files" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong handoff source/readiness: {handoff}")
    assert_file_contract(result, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    if handoff.get("count") != result.metadata.get("count") or handoff.get("total_entries") != result.metadata.get("total_entries"):
        raise SystemExit(f"{label} list count metadata diverged: {handoff} vs {result.metadata}")
    if handoff.get("limit") != result.metadata.get("limit") or handoff.get("truncated") != result.metadata.get("truncated"):
        raise SystemExit(f"{label} list limit/truncation metadata diverged: {handoff} vs {result.metadata}")
    if len(handoff.get("entries") or []) != result.metadata.get("count"):
        raise SystemExit(f"{label} list entries should match count: {handoff}")
    assert_common_boundaries(handoff.get("boundaries") or {}, label=label, read_only=True, reads_private_data=True, writes_files=False)
    assert_no_local_paths(handoff, label)


def assert_read_text_file_handoff(result, *, label: str) -> None:
    handoff = result.metadata.get("read_text_file_handoff")
    if not result.metadata.get("read_text_file_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready read_text_file_handoff: {result.metadata}")
    if handoff.get("source") != "read_text_file" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong handoff source/readiness: {handoff}")
    assert_file_contract(result, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    for key in ("size_bytes", "truncated", "max_chars"):
        if handoff.get(key) != result.metadata.get(key):
            raise SystemExit(f"{label} read handoff {key} diverged: {handoff} vs {result.metadata}")
    if handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} read handoff should be content-free: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    assert_common_boundaries(boundaries, label=label, read_only=True, reads_private_data=True, writes_files=False)
    if boundaries.get("read_file_contents") is not True or boundaries.get("write_file_contents"):
        raise SystemExit(f"{label} read content boundary is wrong: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_write_text_file_handoff(result, *, label: str) -> None:
    handoff = result.metadata.get("write_text_file_handoff")
    if not result.metadata.get("write_text_file_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready write_text_file_handoff: {result.metadata}")
    if handoff.get("source") != "write_text_file" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong handoff source/readiness: {handoff}")
    assert_file_contract(result, handoff, label=label, state_changed=True, changed=["file_write"], content_in_handoff=False)
    if handoff.get("chars") != result.metadata.get("chars"):
        raise SystemExit(f"{label} write chars diverged: {handoff} vs {result.metadata}")
    if handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} write handoff should not include file content: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    assert_common_boundaries(boundaries, label=label, read_only=False, reads_private_data=False, writes_files=True)
    if boundaries.get("write_file_contents") is not True or boundaries.get("read_file_contents"):
        raise SystemExit(f"{label} write content boundary is wrong: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_find_files_handoff(result, *, label: str) -> None:
    handoff = result.metadata.get("find_files_handoff")
    if not result.metadata.get("find_files_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready find_files_handoff: {result.metadata}")
    if handoff.get("source") != "find_files" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong handoff source/readiness: {handoff}")
    assert_file_contract(result, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    if handoff.get("count") != result.metadata.get("count"):
        raise SystemExit(f"{label} find count metadata diverged: {handoff} vs {result.metadata}")
    if handoff.get("limit") != result.metadata.get("limit") or handoff.get("truncated") != result.metadata.get("truncated"):
        raise SystemExit(f"{label} find limit/truncation metadata diverged: {handoff} vs {result.metadata}")
    if len(handoff.get("matches") or []) != result.metadata.get("count"):
        raise SystemExit(f"{label} match rows should match count: {handoff}")
    assert_common_boundaries(handoff.get("boundaries") or {}, label=label, read_only=True, reads_private_data=True, writes_files=False)
    assert_no_local_paths(handoff, label)


def assert_file_refusal_handoff(result, *, label: str, source: str, mutation: str, reason: str, reads_private_data: bool) -> None:
    metadata = result.metadata
    handoff = metadata.get("file_refusal_handoff")
    if not metadata.get("file_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready file_refusal_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} refusal source/mutation mismatch: {handoff}")
    if handoff.get("reason") != reason or metadata.get("refusal_reason") != reason:
        raise SystemExit(f"{label} refusal reason mismatch: {metadata}")
    if not handoff.get("refused") or handoff.get("changed") != []:
        raise SystemExit(f"{label} should report refused/no-change state: {handoff}")
    assert_file_contract(result, handoff, label=label, state_changed=False, changed=[], content_in_handoff=False)
    if not any(str(command).startswith(("find files", "list files", "read text file")) for command in handoff.get("next_commands", [])):
        raise SystemExit(f"{label} should include command-first recovery hints: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    assert_common_boundaries(boundaries, label=label, read_only=True, reads_private_data=reads_private_data, writes_files=False)
    if boundaries.get("read_file_contents") or boundaries.get("write_file_contents"):
        raise SystemExit(f"{label} refusal should not claim file content read/write: {boundaries}")
    assert_no_local_paths(handoff, label)
    assert_no_local_paths(metadata, f"{label} metadata")


def main() -> None:
    assert_file_exact_metadata_bool()
    assert_file_malformed_handoff_flags()
    assert_file_planner_routes()
    with TemporaryDirectory(prefix="jarvis-files-smoke-") as temp:
        root = Path(temp)
        note = root / "note.md"
        note.write_text("Jarvis file smoke test\n", encoding="utf-8")

        listed = list_files({"directory": str(root), "limit": 10})
        if not listed.ok or "note.md" not in listed.output or listed.metadata.get("count") != 1:
            raise SystemExit(f"list_files should list a normal temp file: {listed.output} {listed.metadata}")
        assert_file_metadata_safe(listed.metadata, "list_files")
        assert_list_files_handoff(listed, label="list_files")

        read = read_text_file({"path": str(note), "max_chars": 8})
        if not read.ok or not read.output.startswith("Jarvis f") or not read.metadata.get("truncated"):
            raise SystemExit(f"read_text_file should read and truncate text files: {read.output} {read.metadata}")
        assert_file_metadata_safe(read.metadata, "read_text_file")
        assert_read_text_file_handoff(read, label="read_text_file")

        path_type = type(note)
        original_read_text = path_type.read_text

        def fail_target_read(path: Path, *args, **kwargs):
            if path.name == note.name:
                raise OSError(
                    "synthetic read failure near /\x55sers/example/private/file"
                )
            return original_read_text(path, *args, **kwargs)

        with patch.object(path_type, "read_text", fail_target_read):
            failed_read = read_text_file({"path": str(note)})
        assert_local_productivity_read_recovery(
            failed_read,
            "read_text_file I/O failure",
        )
        if (
            failed_read.metadata.get("refusal_reason") != "read_failed"
            or failed_read.metadata.get("exception_type") != "OSError"
            or "/\x55sers/" in failed_read.output
        ):
            raise SystemExit(
                f"read_text_file I/O failure lost bounded diagnostics: "
                f"{failed_read.output} {failed_read.metadata}"
            )

        write_target = root / "written.md"
        written = write_text_file({"path": str(write_target), "content": "new local text", "overwrite": False})
        if not written.ok or not write_target.exists() or written.metadata.get("chars") != 14:
            raise SystemExit(f"write_text_file should write text files: {written.output} {written.metadata}")
        if not written.metadata.get("writes_files") or not written.metadata.get("executes_side_effect"):
            raise SystemExit(f"write_text_file should mark local file side effects: {written.metadata}")
        assert_write_text_file_handoff(written, label="write_text_file")

        overwrite_target = root / "overwrite.md"
        overwrite_target.write_text("old complete text", encoding="utf-8")
        overwritten = write_text_file(
            {"path": str(overwrite_target), "content": "new complete text", "overwrite": True}
        )
        if not overwritten.ok or overwrite_target.read_text(encoding="utf-8") != "new complete text":
            raise SystemExit(
                f"write_text_file overwrite should atomically publish complete content: "
                f"{overwritten.output} {overwritten.metadata}"
            )
        assert_write_text_file_handoff(overwritten, label="write_text_file overwrite")

        failed_replace_target = root / "failed-replace.md"
        failed_replace_target.write_text("preserve these bytes", encoding="utf-8")
        with patch("jarvis_v2.tools.files.os.replace", side_effect=OSError("injected pre-commit failure")):
            failed_replace = write_text_file(
                {"path": str(failed_replace_target), "content": "must not appear", "overwrite": True}
            )
        if failed_replace.ok or failed_replace_target.read_bytes() != b"preserve these bytes":
            raise SystemExit(
                f"write_text_file replace failure should preserve existing bytes: "
                f"{failed_replace.output} {failed_replace.metadata}"
            )
        if list(root.glob(f".{failed_replace_target.name}.*.tmp")):
            raise SystemExit("write_text_file replace failure should clean its sibling temp file")
        assert_file_refusal_handoff(
            failed_replace,
            label="write_text_file replace failure",
            source="write_text_file",
            mutation="file_write",
            reason="write_failed",
            reads_private_data=False,
        )

        invalid_unicode_target = root / "invalid-unicode.md"
        invalid_unicode_target.write_text("keep valid utf-8", encoding="utf-8")
        invalid_unicode = write_text_file(
            {"path": str(invalid_unicode_target), "content": "invalid surrogate: \ud800", "overwrite": True}
        )
        if invalid_unicode.ok or invalid_unicode_target.read_text(encoding="utf-8") != "keep valid utf-8":
            raise SystemExit("write_text_file encoding failure should preserve the existing complete file")
        if invalid_unicode.metadata.get("refusal_reason") != "write_failed":
            raise SystemExit(f"write_text_file encoding failure missed bounded refusal metadata: {invalid_unicode.metadata}")
        if list(root.glob(f".{invalid_unicode_target.name}.*.tmp")):
            raise SystemExit("write_text_file encoding failure should clean its sibling temp file")

        concurrent_target = root / "concurrent.md"
        concurrent_contents = [f"complete contender {index}:" + (str(index) * 1000) for index in range(8)]
        barrier = threading.Barrier(len(concurrent_contents))
        concurrent_results = []
        result_lock = threading.Lock()

        def concurrent_write(content: str) -> None:
            barrier.wait()
            result = write_text_file(
                {"path": str(concurrent_target), "content": content, "overwrite": False}
            )
            with result_lock:
                concurrent_results.append(result)

        threads = [threading.Thread(target=concurrent_write, args=(content,)) for content in concurrent_contents]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        winners = [result for result in concurrent_results if result.ok]
        final_content = concurrent_target.read_text(encoding="utf-8")
        if len(winners) != 1 or final_content not in concurrent_contents:
            raise SystemExit(
                "concurrent overwrite=false writers should produce exactly one complete winner: "
                f"winners={len(winners)} results={[(result.ok, result.metadata.get('refusal_reason')) for result in concurrent_results]}"
            )
        if list(root.glob(f".{concurrent_target.name}.*.tmp")):
            raise SystemExit("concurrent overwrite=false writers should clean all sibling temp files")

        found = find_files({"root": str(root), "pattern": "note", "limit": 5})
        if not found.ok or "note.md" not in found.output or found.metadata.get("count") != 1:
            raise SystemExit(f"find_files should find a normal temp file: {found.output} {found.metadata}")
        assert_file_metadata_safe(found.metadata, "find_files")
        assert_find_files_handoff(found, label="find_files")

        not_found = find_files({"root": str(root), "pattern": "missing", "limit": 5})
        if not not_found.ok or not_found.metadata.get("count") != 0:
            raise SystemExit(f"find_files should emit empty success metadata: {not_found.output} {not_found.metadata}")
        assert_find_files_handoff(not_found, label="find_files empty")

        missing_read_path = read_text_file({"path": ""})
        if missing_read_path.ok or "No file path provided" not in missing_read_path.output:
            raise SystemExit(f"read_text_file should refuse missing paths: {missing_read_path.output} {missing_read_path.metadata}")
        assert_file_refusal_handoff(
            missing_read_path,
            label="read_text_file missing path",
            source="read_text_file",
            mutation="file_read",
            reason="missing_path",
            reads_private_data=True,
        )

        missing_write_path = write_text_file({"path": "", "content": "no destination"})
        if missing_write_path.ok or "No file path provided" not in missing_write_path.output:
            raise SystemExit(f"write_text_file should refuse missing paths: {missing_write_path.output} {missing_write_path.metadata}")
        assert_file_refusal_handoff(
            missing_write_path,
            label="write_text_file missing path",
            source="write_text_file",
            mutation="file_write",
            reason="missing_path",
            reads_private_data=False,
        )

        missing_file = read_text_file({"path": str(root / "missing.md")})
        if missing_file.ok or "File not found: missing.md" not in missing_file.output:
            raise SystemExit(f"read_text_file should refuse missing files with sanitized output: {missing_file.output} {missing_file.metadata}")
        assert_resource_not_found_recovery(
            missing_file,
            "read_text_file missing file",
        )
        assert_file_refusal_handoff(
            missing_file,
            label="read_text_file missing file",
            source="read_text_file",
            mutation="file_read",
            reason="file_not_found",
            reads_private_data=True,
        )
        expected_file_recovery = ["list files in .", "find files missing.md", "read missing.md"]
        for command in expected_file_recovery:
            if command not in missing_file.output:
                raise SystemExit(f"Missing file refusal missed recovery command {command!r}: {missing_file.output}")
        if missing_file.metadata.get("next_command") != expected_file_recovery[0]:
            raise SystemExit(f"Missing file refusal missed first recovery command: {missing_file.metadata}")
        if missing_file.metadata.get("recovery_commands") != expected_file_recovery:
            raise SystemExit(f"Missing file refusal missed ordered recovery metadata: {missing_file.metadata}")
        if missing_file.metadata.get("retry_requires_path_refresh") is not True:
            raise SystemExit(f"Missing file refusal should require a path refresh: {missing_file.metadata}")
        if missing_file.metadata.get("retry_requires_fresh_approval") is not True:
            raise SystemExit(f"Missing file refusal should require normal fresh approval: {missing_file.metadata}")
        if missing_file.metadata.get("authorizes_retry") is not False:
            raise SystemExit(f"Missing file refusal should not authorize retry: {missing_file.metadata}")

        missing_root = list_files({"directory": str(root / "missing-dir")})
        if missing_root.ok or "Directory not found: missing-dir" not in missing_root.output:
            raise SystemExit(f"list_files should refuse missing directories with sanitized output: {missing_root.output} {missing_root.metadata}")
        assert_file_refusal_handoff(
            missing_root,
            label="list_files missing directory",
            source="list_files",
            mutation="directory_list",
            reason="directory_not_found",
            reads_private_data=True,
        )
        if "list files in ." not in missing_root.output:
            raise SystemExit(f"Missing directory refusal missed recovery command: {missing_root.output}")
        if missing_root.metadata.get("recovery_commands") != ["list files in ."]:
            raise SystemExit(f"Missing directory refusal missed ordered recovery metadata: {missing_root.metadata}")
        if missing_root.metadata.get("retry_requires_path_refresh") is not True:
            raise SystemExit(f"Missing directory refusal should require a path refresh: {missing_root.metadata}")
        if missing_root.metadata.get("retry_requires_fresh_approval") is not True:
            raise SystemExit(f"Missing directory refusal should require normal fresh approval: {missing_root.metadata}")
        if missing_root.metadata.get("authorizes_retry") is not False:
            raise SystemExit(f"Missing directory refusal should not authorize retry: {missing_root.metadata}")

        empty_pattern = find_files({"root": str(root), "pattern": ""})
        if empty_pattern.ok or "Search pattern is empty" not in empty_pattern.output:
            raise SystemExit(f"find_files should refuse empty patterns: {empty_pattern.output} {empty_pattern.metadata}")
        assert_file_refusal_handoff(
            empty_pattern,
            label="find_files empty pattern",
            source="find_files",
            mutation="file_find",
            reason="missing_pattern",
            reads_private_data=True,
        )

        existing_write = write_text_file({"path": str(note), "content": "replace later", "overwrite": False})
        if existing_write.ok or "File already exists: note.md" not in existing_write.output:
            raise SystemExit(f"write_text_file should refuse existing files without overwrite: {existing_write.output} {existing_write.metadata}")
        assert_file_refusal_handoff(
            existing_write,
            label="write_text_file existing file",
            source="write_text_file",
            mutation="file_write",
            reason="already_exists",
            reads_private_data=False,
        )

        oversized_write = write_text_file({"path": str(root / "large.md"), "content": "x" * 100001})
        if oversized_write.ok or oversized_write.metadata.get("chars") != 100001:
            raise SystemExit(f"write_text_file should refuse oversized content: {oversized_write.output} {oversized_write.metadata}")
        assert_file_refusal_handoff(
            oversized_write,
            label="write_text_file oversized content",
            source="write_text_file",
            mutation="file_write",
            reason="content_too_large",
            reads_private_data=False,
        )

        large_read_target = root / "large-read.md"
        large_read_target.write_text("x" * 1000001, encoding="utf-8")
        oversized_read = read_text_file({"path": str(large_read_target)})
        if oversized_read.ok or oversized_read.metadata.get("max_read_bytes") != 1000000:
            raise SystemExit(f"read_text_file should preserve max_read_bytes on oversized refusal: {oversized_read.output} {oversized_read.metadata}")
        assert_file_refusal_handoff(
            oversized_read,
            label="read_text_file oversized file",
            source="read_text_file",
            mutation="file_read",
            reason="file_too_large",
            reads_private_data=True,
        )

        for raw_limit in (
            "/\x55sers/example/private/file-limit",
            "/private/tmp/jarvis-file-limit",
            "/var/folders/zc/jarvis-file-limit",
            "/tmp/jarvis-file-limit",
        ):
            listed_bad_limit = list_files({"directory": str(root), "limit": raw_limit})
            if not listed_bad_limit.ok or listed_bad_limit.metadata.get("limit") != 80:
                raise SystemExit(f"list_files should default path-shaped bad limits: {listed_bad_limit.metadata}")
            assert_path_raw_number_redaction(listed_bad_limit, "raw_limit", "list_files path limit")
            assert_list_files_handoff(listed_bad_limit, label="list_files path limit")

            found_bad_limit = find_files({"root": str(root), "pattern": "note", "limit": raw_limit})
            if not found_bad_limit.ok or found_bad_limit.metadata.get("limit") != 50:
                raise SystemExit(f"find_files should default path-shaped bad limits: {found_bad_limit.metadata}")
            assert_path_raw_number_redaction(found_bad_limit, "raw_limit", "find_files path limit")
            assert_find_files_handoff(found_bad_limit, label="find_files path limit")

        for raw_max_chars in (
            "/\x55sers/example/private/max-chars",
            "/private/tmp/jarvis-max-chars",
            "/var/folders/zc/jarvis-max-chars",
            "/tmp/jarvis-max-chars",
        ):
            read_bad_max = read_text_file({"path": str(note), "max_chars": raw_max_chars})
            if not read_bad_max.ok or read_bad_max.metadata.get("max_chars") != 5000:
                raise SystemExit(f"read_text_file should default path-shaped bad max chars: {read_bad_max.metadata}")
            assert_path_raw_number_redaction(read_bad_max, "raw_max_chars", "read_text_file path max_chars")
            assert_read_text_file_handoff(read_bad_max, label="read_text_file path max_chars")

        print("file tool smoke passed")


if __name__ == "__main__":
    main()
