from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.types import RiskLevel, ToolResult
from jarvis_v2.memory.obsidian import _profile_note_block
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.files import read_text_file
from jarvis_v2.tools.notes import (
    make_write_jarvis_note_auto_mutation_preflight,
    make_write_jarvis_note_auto_mutation_preflight_result,
)
from jarvis_v2.tools.tasks import (
    make_import_tasks_from_note_auto_mutation_preflight,
    make_import_tasks_from_note_auto_mutation_preflight_result,
)


SENTINEL_TASK = "PROFILE_PROTECTED_SENTINEL_CHECKBOX_TASK"
PROFILE_SOURCE_KEY = "profile-note:v1:" + ("a" * 64)


def _fullwidth_ascii(value: str) -> str:
    return "".join(
        chr(ord(character) + 0xFEE0) if "!" <= character <= "~" else character
        for character in value
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _serialized(result: ToolResult) -> str:
    return result.output + "\n" + json.dumps(result.metadata, sort_keys=True, default=str)


def _assert_no_sentinel(result: ToolResult, *, label: str) -> None:
    exposed = _serialized(result)
    if SENTINEL_TASK in exposed:
        raise SystemExit(f"{label} exposed the protected checkbox payload: {exposed}")


def _assert_protected_refusal(result: ToolResult, *, label: str) -> None:
    _assert_no_sentinel(result, label=label)
    exposed = _serialized(result)
    if any(
        value in exposed
        for value in ("Unowned Profile Block", "Unowned Profile Memory", "jarvis-profile-note")
    ):
        raise SystemExit(f"{label} leaked protected source details: {exposed}")
    if result.ok or "read profile" not in result.output.lower():
        raise SystemExit(f"{label} should refuse non-leakily and direct to read profile: {result}")
    if result.metadata.get("state_changed") or any(
        result.metadata.get(key)
        for key in ("writes_database", "writes_files", "writes_memory", "writes_notes")
    ):
        raise SystemExit(f"{label} should report no mutation: {result.metadata}")


def _profile_memory_projection(search_token: str) -> str:
    return (
        "---\n"
        "jarvis_projection: memory\n"
        f"store_identity: {'b' * 32}\n"
        "memory_id: 7\n"
        "memory_revision: 1\n"
        f"source_digest: {'c' * 64}\n"
        "category: identity\n"
        f"title: {search_token}\n"
        "source: profile\n"
        "confidence: 1.0\n"
        "created_at: 2026-07-13T00:00:00Z\n"
        "---\n\n"
        f"# {search_token}\n\n"
        f"- [ ] {SENTINEL_TASK}\n"
    )


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-protected-paths-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        notes_dir = runtime.vault.root_path / "Projects"
        notes_dir.mkdir(parents=True, exist_ok=True)

        fixtures = (
            (
                "Projects/Unowned Profile Block.md",
                "PROFILE_BLOCK_SEARCH_TOKEN",
                _profile_note_block(
                    PROFILE_SOURCE_KEY,
                    "PROFILE_BLOCK_SEARCH_TOKEN",
                    f"- [ ] {SENTINEL_TASK}",
                ),
            ),
            (
                "Projects/Unowned Profile Memory.md",
                "PROFILE_MEMORY_SEARCH_TOKEN",
                _profile_memory_projection("PROFILE_MEMORY_SEARCH_TOKEN"),
            ),
            (
                "Projects/Malformed Profile Marker.md",
                "MALFORMED_PROFILE_MARKER_SEARCH_TOKEN",
                "<!-- jarvis-profile-note-start:v1:aaaa -->\n"
                "# MALFORMED_PROFILE_MARKER_SEARCH_TOKEN\n\n"
                f"- [ ] {SENTINEL_TASK}\n"
                "<!-- jarvis-profile-note:v1:aaaa -->\n",
            ),
            (
                "Projects/Bare Profile Namespace.md",
                "BARE_PROFILE_NAMESPACE_SEARCH_TOKEN",
                "reserved namespace jarvis-profile-note without a comment opener\n"
                "# BARE_PROFILE_NAMESPACE_SEARCH_TOKEN\n\n"
                f"- [ ] {SENTINEL_TASK}\n",
            ),
            (
                "Projects/Compatibility Profile Namespace.md",
                "COMPAT_PROFILE_NAMESPACE_SEARCH_TOKEN",
                "reserved namespace "
                + _fullwidth_ascii("jarvis-profile-note")
                + " without a comment opener\n"
                "# COMPAT_PROFILE_NAMESPACE_SEARCH_TOKEN\n\n"
                f"- [ ] {SENTINEL_TASK}\n",
            ),
        )
        for relative_path, _search_token, content in fixtures:
            (runtime.vault.root_path / relative_path).write_text(content, encoding="utf-8")

        search_note = runtime.registry.get("search_jarvis_notes").handler
        read_note = runtime.registry.get("read_jarvis_note").handler
        outline_note = runtime.registry.get("outline_jarvis_note").handler
        preview_tasks = runtime.registry.get("preview_tasks_from_note").handler
        import_tasks = runtime.registry.get("import_tasks_from_note").handler
        import_preflight = make_import_tasks_from_note_auto_mutation_preflight(runtime.vault)
        import_preflight_result = make_import_tasks_from_note_auto_mutation_preflight_result(
            runtime.vault
        )
        write_preflight = make_write_jarvis_note_auto_mutation_preflight(runtime.vault)
        write_preflight_result = make_write_jarvis_note_auto_mutation_preflight_result(
            runtime.vault
        )
        write_note = runtime.registry.get("write_jarvis_note").handler
        if runtime.registry.get("read_text_file").risk != RiskLevel.PERSONAL_DATA:
            raise SystemExit("generic file read lost its personal-data approval boundary")

        for relative_path, search_token, _content in fixtures:
            search = search_note({"query": search_token})
            _assert_no_sentinel(search, label=f"search {relative_path}")
            if not search.ok or search.metadata.get("count") != 0:
                raise SystemExit(f"search should omit protected profile content: {search}")

            for label, result in (
                ("note read", read_note({"path": relative_path})),
                ("note outline", outline_note({"path": relative_path})),
                (
                    "approved generic file read",
                    read_text_file({"path": str(runtime.vault.root_path / relative_path)}),
                ),
                ("task preview", preview_tasks({"path": relative_path})),
            ):
                _assert_protected_refusal(result, label=f"{label} {relative_path}")

            before = _snapshot(root)
            reason = import_preflight({"path": relative_path, "priority": "high"})
            if reason != "protected_profile":
                raise SystemExit(f"task import preflight missed protected profile content: {reason}")
            preflight_refusal = import_preflight_result(
                {"path": relative_path, "priority": "high"},
                reason,
            )
            _assert_protected_refusal(
                preflight_refusal,
                label=f"task import preflight {relative_path}",
            )
            imported = import_tasks({"path": relative_path, "priority": "high"})
            _assert_protected_refusal(imported, label=f"task import {relative_path}")
            after = _snapshot(root)
            if after != before:
                raise SystemExit(f"refused task import changed DB or files: {relative_path}")

        canonical_path = runtime.vault.root_path / "Profile.md"
        canonical_cases = (
            ("canonical note read", read_note({"path": "Profile.md"})),
            ("canonical note outline", outline_note({"path": "Profile.md"})),
            ("canonical generic file read", read_text_file({"path": str(canonical_path)})),
            ("canonical task preview", preview_tasks({"path": "Profile.md"})),
        )
        for label, result in canonical_cases:
            _assert_protected_refusal(result, label=label)

        symlink_alias = runtime.vault.root_path / "Profile Link.md"
        symlink_alias.symlink_to("Profile.md")
        profile_read_aliases = (
            ("canonical", "Profile.md"),
            ("canonical without suffix", "Profile"),
            ("casefold", "PROFILE.MD"),
            ("NFKC", _fullwidth_ascii("Profile.md")),
            ("symlink", symlink_alias.name),
        )
        for alias_label, profile_target in profile_read_aliases:
            for operation, result in (
                ("read", read_note({"path": profile_target})),
                ("outline", outline_note({"path": profile_target})),
            ):
                _assert_protected_refusal(
                    result,
                    label=f"{alias_label} profile {operation}",
                )
                if result.metadata.get("reason") != "protected_profile":
                    raise SystemExit(
                        f"{alias_label} profile {operation} lost protected-profile metadata: "
                        f"{result.metadata}"
                    )

        descriptor_note = runtime.vault.root_path / "Projects" / "Descriptor Boundary.md"
        descriptor_note.write_text(
            "# Descriptor Boundary\n\nDESCRIPTOR_BOUNDARY_PAYLOAD\n",
            encoding="utf-8",
        )
        with patch.object(
            runtime.vault,
            "read_note_bounded",
            wraps=runtime.vault.read_note_bounded,
        ) as bounded_read:
            with (
                patch.object(Path, "exists", side_effect=AssertionError("Path.exists called")),
                patch.object(Path, "is_file", side_effect=AssertionError("Path.is_file called")),
                patch.object(Path, "read_text", side_effect=AssertionError("Path.read_text called")),
            ):
                descriptor_read = read_note({"path": "Projects/Descriptor Boundary.md"})
                descriptor_outline = outline_note(
                    {"path": "Projects/Descriptor Boundary.md"}
                )
        if not descriptor_read.ok or "DESCRIPTOR_BOUNDARY_PAYLOAD" not in descriptor_read.output:
            raise SystemExit(
                f"descriptor-backed note read failed: {descriptor_read}"
            )
        if not descriptor_outline.ok or "Descriptor Boundary" not in descriptor_outline.output:
            raise SystemExit(
                f"descriptor-backed note outline failed: {descriptor_outline}"
            )
        if bounded_read.call_count != 2:
            raise SystemExit(
                "read and outline did not each use one descriptor-backed vault snapshot"
            )

        before = _snapshot(root)
        canonical_import = import_tasks({"path": "Profile.md", "priority": "normal"})
        _assert_protected_refusal(canonical_import, label="canonical task import")
        if _snapshot(root) != before:
            raise SystemExit("refused canonical profile task import changed DB or files")

        for profile_target in (
            "Profile.md",
            "PROFILE.MD",
            _fullwidth_ascii("Profile.md"),
        ):
            for mode in ("append", "create", "overwrite"):
                args = {"path": profile_target, "body": "forbidden profile write", "mode": mode}
                before = _snapshot(root)
                reason = write_preflight(args)
                if reason != "protected_profile":
                    raise SystemExit(
                        f"note write preflight missed protected profile target {profile_target!r}: {reason}"
                    )
                _assert_protected_refusal(
                    write_preflight_result(args, reason),
                    label=f"note write preflight {profile_target!r} {mode}",
                )
                _assert_protected_refusal(
                    write_note(args),
                    label=f"note write handler {profile_target!r} {mode}",
                )
                if _snapshot(root) != before:
                    raise SystemExit(
                        f"refused profile note write changed DB or files: {profile_target!r} {mode}"
                    )

        if runtime.store.list_tasks(status=None, limit=10):
            raise SystemExit("protected profile payload imported a task")

    print("profile protected paths smoke passed")


if __name__ == "__main__":
    main()
