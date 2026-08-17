"""Focused offline proof for canonical skill-refusal recovery guidance."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from jarvis_v2.tools.skills import (
    MAX_SKILL_BODY_CHARS,
    SKILL_REFUSAL_RECOVERY_ACTION,
    make_skill_tools,
)


class _Store:
    def __init__(self) -> None:
        self.identity_error = False
        self.save_error = False
        self.resolution = SimpleNamespace(status="not_found", candidates=[])
        self.delete_status = "not_found"

    def get_skill_by_identity(self, _name: str):
        if self.identity_error:
            raise ValueError("ambiguous")
        return None

    def save_skill_with_projection_target(self, _record):
        if self.save_error:
            raise ValueError("changed")
        raise AssertionError("focused refusal proof must stop before a skill write")

    def resolve_skill_delete_target(self, _name: str):
        return self.resolution

    def delete_skill_exact(self, _skill_id: int, _revision: int, _name: str):
        return SimpleNamespace(status=self.delete_status, deleted_skill=None)


class _Vault:
    root_path = Path(".")


def _assert_guided(result, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded")
    expected = {
        "version": 1,
        "action": SKILL_REFUSAL_RECOVERY_ACTION,
        "commands": ["list skills"],
    }
    if result.metadata.get("recovery_guidance") != expected:
        raise SystemExit(f"{label} missed canonical recovery guidance: {result.metadata}")
    if result.metadata.get("next_command") != "list skills":
        raise SystemExit(f"{label} missed the primary recovery command: {result.metadata}")
    if result.metadata.get("recovery_commands") != ["list skills"]:
        raise SystemExit(f"{label} missed bounded recovery commands: {result.metadata}")
    if SKILL_REFUSAL_RECOVERY_ACTION not in result.output or "list skills" not in result.output:
        raise SystemExit(f"{label} hid recovery guidance from visible output: {result.output}")
    if result.metadata.get("authorizes_execution") is not False:
        raise SystemExit(f"{label} recovery declaration authorized execution: {result.metadata}")
    if result.metadata.get("approval_granted") is not False:
        raise SystemExit(f"{label} recovery declaration granted approval: {result.metadata}")
    combined = result.output + repr(result.metadata)
    for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/"):
        if fragment in combined:
            raise SystemExit(f"{label} leaked a private local path: {combined}")


def _assert_all_refusal_sites_are_centralized() -> None:
    source_path = Path(__file__).parents[1] / "tools" / "skills.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_skill_refusal_result"
    ]
    if len(calls) != 20:
        raise SystemExit(f"expected 20 centralized skill-refusal sites, found {len(calls)}")
    for call in calls:
        if len(call.args) != 3:
            raise SystemExit(f"skill-refusal site at line {call.lineno} changed its bounded shape")
        metadata = call.args[2]
        if not (
            isinstance(metadata, ast.Call)
            and isinstance(metadata.func, ast.Name)
            and metadata.func.id == "_skill_refusal_metadata"
        ):
            raise SystemExit(f"skill-refusal site at line {call.lineno} bypassed refusal metadata")

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ToolResult"
            and len(node.args) >= 4
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value is False
        ):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_skill_refusal_metadata"
            for child in ast.walk(node)
        ):
            raise SystemExit(
                f"raw ToolResult refusal at line {node.lineno} bypassed canonical recovery guidance"
            )


def main() -> None:
    _assert_all_refusal_sites_are_centralized()
    store = _Store()
    (
        save_skill,
        _list_skills,
        search_skills,
        get_skill,
        delete_skill,
        _extract_linked_skills,
        _install_linked_skills,
        _skill_match_preview,
        resolve_delete_skill_approval,
    ) = make_skill_tools(store, _Vault())

    cases = [
        ("save missing name", save_skill({"name": "", "body": "procedure"})),
        (
            "save path-shaped name",
            save_skill({"name": "/\x55sers/example/private-skill", "body": "procedure"}),
        ),
        ("save missing procedure", save_skill({"name": "Example", "body": ""})),
        (
            "save oversized procedure",
            save_skill({"name": "Example", "body": "x" * (MAX_SKILL_BODY_CHARS + 1)}),
        ),
        ("search missing query", search_skills({"query": ""})),
        (
            "search path-shaped query",
            search_skills({"query": "/private/example/search"}),
        ),
        ("get missing name", get_skill({"name": ""})),
        ("get path-shaped name", get_skill({"name": "/tmp/example-skill"})),
        ("delete missing name", delete_skill({"name": ""})),
        ("delete path-shaped name", delete_skill({"name": "/var/folders/example"})),
        ("delete unbound target", delete_skill({"name": "Example"})),
        ("approval missing name", resolve_delete_skill_approval({"name": ""})),
        (
            "approval path-shaped name",
            resolve_delete_skill_approval({"name": "/\x55sers/example/private-skill"}),
        ),
        ("approval target absent", resolve_delete_skill_approval({"name": "Example"})),
    ]

    store.identity_error = True
    cases.append(("save ambiguous identity", save_skill({"name": "Example", "body": "procedure"})))
    store.identity_error = False
    store.save_error = True
    cases.append(("save changed state", save_skill({"name": "Example", "body": "procedure"})))
    store.save_error = False

    store.resolution = SimpleNamespace(
        status="ambiguous",
        candidates=[SimpleNamespace(name="Example A"), SimpleNamespace(name="Example B")],
    )
    cases.append(("approval ambiguous target", resolve_delete_skill_approval({"name": "Example"})))
    store.resolution = SimpleNamespace(
        status="resolved",
        candidates=[],
        skill_id=None,
        revision=None,
        name="",
    )
    cases.append(("approval unreadable target", resolve_delete_skill_approval({"name": "Example"})))

    store.delete_status = "stale"
    bound_delete = {
        "name": "Example",
        "target_skill_id": 1,
        "target_skill_revision": 1,
        "target_skill_name": "Example",
    }
    cases.append(("delete stale target", delete_skill(bound_delete)))
    store.delete_status = "other"
    cases.append(("delete absent fallback", delete_skill(bound_delete)))

    if len(cases) != 20:
        raise SystemExit(f"focused runtime refusal matrix drifted: {len(cases)}")
    for label, result in cases:
        _assert_guided(result, label)

    print("Skill failure-guidance smoke test passed.")


if __name__ == "__main__":
    main()
