from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, BrokenBarrierError
from typing import Any, Callable, Iterator, Mapping
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.agent.types import Plan, PlannedAction, ToolResult
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import GoalRecord, MemoryStore
from jarvis_v2.tools.goals import make_goal_tools
from jarvis_v2.tools.organize import make_organize_tools


GOAL_TOOL_NAMES = (
    "create_goal",
    "list_goals",
    "goal_status",
    "add_goal_step",
    "complete_goal_step",
    "set_goal_status",
    "export_goal",
    "next_actions",
)
ABSOLUTE_LOCAL_PATH_RE = re.compile(
    r"(?:^|[\s\"'])/(?:Users|private|tmp|var/folders)(?:/[^\s\"']*)?"
)


@dataclass(frozen=True)
class GoalFixture:
    store: MemoryStore
    vault: ObsidianVault
    handlers: Mapping[str, Callable[[dict[str, Any]], ToolResult]]


class StaticPlanner:
    def __init__(self, goal_id: int) -> None:
        self.goal_id = goal_id

    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Publish one goal projection.",
            [
                PlannedAction(
                    "export_goal",
                    {"goal_id": self.goal_id},
                    "goal projection ownership smoke",
                )
            ],
            needs_model=False,
        )


def _setup(root: Path, shared_vault: Path) -> GoalFixture:
    root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(root / "store.sqlite")
    store.init()
    vault = ObsidianVault(shared_vault, "Jarvis")
    vault.init()
    tools = make_goal_tools(store, vault)
    handlers = {name: handler for name, handler in zip(GOAL_TOOL_NAMES, tools)}
    if set(handlers) != set(GOAL_TOOL_NAMES):
        raise SystemExit("goal ownership fixture could not bind every goal handler")
    return GoalFixture(store, vault, handlers)


def _runtime(root: Path) -> JarvisRuntime:
    config = JarvisConfig(
        data_dir=root,
        db_path=root / "jarvis.sqlite",
        obsidian_vault=root / "Vault",
        obsidian_root="Jarvis",
        use_model_planner=False,
        watched_dirs=(root / "Watched",),
    )
    return JarvisRuntime(config)


def _invoke(
    fixture: GoalFixture,
    tool_name: str,
    args: dict[str, Any],
    label: str,
) -> ToolResult:
    result = fixture.handlers[tool_name](args)
    if not result.ok:
        raise SystemExit(
            f"{label} failed: failure_kind={result.metadata.get('failure_kind')!r}, "
            f"error_type={result.metadata.get('error_type')!r}"
        )
    return result


def _frontmatter_values(path: Path) -> dict[str, list[str]]:
    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeError):
        return {}
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---\n"):
        return {}
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return {}
    values: dict[str, list[str]] = {}
    for line in normalized[4:end].splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        values.setdefault(key.strip(), []).append(value.strip())
    return values


def _owned_goal_paths(
    vault: ObsidianVault,
    *,
    store_identity: str,
    goal_id: int,
) -> list[Path]:
    projects = vault.root_path / "Projects"
    matches: list[Path] = []
    for path in sorted(projects.rglob("*.md")):
        values = _frontmatter_values(path)
        if (
            values.get("jarvis_projection") == ["goal"]
            and values.get("store_identity") == [store_identity]
            and values.get("goal_id") == [str(goal_id)]
        ):
            matches.append(path)
    return matches


def _owned_goal_path(
    fixture: GoalFixture,
    goal_id: int,
    label: str,
) -> Path:
    identity = fixture.store.get_store_identity()
    matches = _owned_goal_paths(
        fixture.vault,
        store_identity=identity,
        goal_id=goal_id,
    )
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise SystemExit(
            f"{label} expected exactly one projection owned by "
            f"jarvis_projection=goal/store_identity/goal_id; found {len(matches)}: {names}"
        )
    return matches[0]


def _assert_marker_partition(
    first_path: Path,
    second_path: Path,
    *,
    first_marker: str,
    second_marker: str,
    label: str,
) -> None:
    first = first_path.read_bytes()
    second = second_path.read_bytes()
    if first_marker.encode() not in first or second_marker.encode() in first:
        raise SystemExit(f"{label} first owned projection lost or crossed private content")
    if second_marker.encode() not in second or first_marker.encode() in second:
        raise SystemExit(f"{label} second owned projection lost or crossed private content")


def test_two_stores_same_id_and_title_remain_separate() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-shared-") as temp:
        root = Path(temp)
        shared_vault = root / "Vault"
        first = _setup(root / "store-a", shared_vault)
        second = _setup(root / "store-b", shared_vault)
        first_identity = first.store.get_store_identity()
        second_identity = second.store.get_store_identity()
        if first_identity == second_identity:
            raise SystemExit("independent goal stores unexpectedly shared a store identity")

        first_marker = "PRIVATE-FIRST-SHARED-TITLE-GOAL"
        second_marker = "PRIVATE-SECOND-SHARED-TITLE-GOAL"
        first_id = first.store.create_goal(GoalRecord("Shared Goal", first_marker))
        second_id = second.store.create_goal(GoalRecord("Shared Goal", second_marker))
        if (first_id, second_id) != (1, 1):
            raise SystemExit(
                f"shared-vault collision fixture did not reproduce numeric goal #1: "
                f"{first_id}, {second_id}"
            )

        _invoke(first, "export_goal", {"goal_id": first_id}, "first shared-vault export")
        _invoke(second, "export_goal", {"goal_id": second_id}, "second shared-vault export")
        first_path = _owned_goal_path(first, first_id, "first shared-vault export")
        second_path = _owned_goal_path(second, second_id, "second shared-vault export")
        if first_path == second_path:
            raise SystemExit("two stores mapped goal #1 with the same title to one path")
        _assert_marker_partition(
            first_path,
            second_path,
            first_marker=first_marker,
            second_marker=second_marker,
            label="shared-vault same-title collision",
        )

        _invoke(second, "export_goal", {"goal_id": second_id}, "second stable re-export")
        _invoke(first, "export_goal", {"goal_id": first_id}, "first stable re-export")
        if _owned_goal_path(first, first_id, "first stable re-export") != first_path:
            raise SystemExit("first store changed its stable owned goal path on re-export")
        if _owned_goal_path(second, second_id, "second stable re-export") != second_path:
            raise SystemExit("second store changed its stable owned goal path on re-export")
        _assert_marker_partition(
            first_path,
            second_path,
            first_marker=first_marker,
            second_marker=second_marker,
            label="shared-vault stable re-export",
        )


def test_same_store_mutations_converge_to_one_owned_path() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-converge-") as temp:
        root = Path(temp)
        fixture = _setup(root / "store", root / "Vault")
        created = _invoke(
            fixture,
            "create_goal",
            {
                "title": "Stable / Owned Goal",
                "purpose": "PRIVATE-STABLE-GOAL-PURPOSE",
                "horizon": "this week",
            },
            "owned goal create",
        )
        goal_id = created.metadata.get("goal_id")
        if type(goal_id) is not int or goal_id < 1:
            raise SystemExit(f"owned goal create missed a numeric goal id: {goal_id!r}")
        stable_path = _owned_goal_path(fixture, goal_id, "owned goal create")

        operations: list[tuple[str, dict[str, Any], str]] = [
            ("export_goal", {"goal_id": goal_id}, "first explicit export"),
            ("export_goal", {"goal_id": goal_id}, "repeated explicit export"),
            (
                "add_goal_step",
                {"goal_id": goal_id, "body": "PRIVATE-STABLE-GOAL-STEP"},
                "step publication",
            ),
            (
                "set_goal_status",
                {"goal_id": goal_id, "status": "paused"},
                "status update publication",
            ),
        ]
        step_id: int | None = None
        for tool_name, args, label in operations:
            result = _invoke(fixture, tool_name, args, label)
            if tool_name == "add_goal_step":
                candidate = result.metadata.get("step_id")
                if type(candidate) is not int or candidate < 1:
                    raise SystemExit(f"step publication missed a numeric step id: {candidate!r}")
                step_id = candidate
            if _owned_goal_path(fixture, goal_id, label) != stable_path:
                raise SystemExit(f"{label} changed the store-owned goal projection path")

        if step_id is None:
            raise SystemExit("stable goal mutation fixture did not create its step")
        tail_operations = [
            ("complete_goal_step", {"step_id": step_id}, "step completion publication"),
            ("set_goal_status", {"goal_id": goal_id, "status": "done"}, "done publication"),
            ("set_goal_status", {"goal_id": goal_id, "status": "done"}, "repeated status publication"),
            ("export_goal", {"goal_id": goal_id}, "final repeated export"),
        ]
        for tool_name, args, label in tail_operations:
            _invoke(fixture, tool_name, args, label)
            if _owned_goal_path(fixture, goal_id, label) != stable_path:
                raise SystemExit(f"{label} changed the store-owned goal projection path")

        final = stable_path.read_text(encoding="utf-8")
        if (
            "PRIVATE-STABLE-GOAL-PURPOSE" not in final
            or "PRIVATE-STABLE-GOAL-STEP" not in final
            or 'status: "done"' not in final
        ):
            raise SystemExit("converged owned goal projection missed the latest source state")


def test_legacy_id_only_and_title_only_notes_are_preserved() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-legacy-") as temp:
        root = Path(temp)
        fixture = _setup(root / "store", root / "Vault")
        projects = fixture.vault.root_path / "Projects"

        id_only_id = fixture.store.create_goal(
            GoalRecord("Legacy ID Only", "PRIVATE-NEW-ID-ONLY-PROJECTION")
        )
        id_only_path = projects / "Legacy ID Only.md"
        id_only_bytes = b"---\nid: 1\n---\n\n# Hand-written ID-only goal\n\nKEEP-ID-ONLY\n"
        id_only_path.write_bytes(id_only_bytes)
        _invoke(fixture, "export_goal", {"goal_id": id_only_id}, "ID-only legacy export")
        id_owned = _owned_goal_path(fixture, id_only_id, "ID-only legacy export")
        if id_only_path.read_bytes() != id_only_bytes:
            raise SystemExit("ID-only legacy goal note was not preserved byte-for-byte")
        if id_owned == id_only_path:
            raise SystemExit("ID-only legacy goal note was adopted as an owned projection")

        title_only_id = fixture.store.create_goal(
            GoalRecord("Legacy Title Only", "PRIVATE-NEW-TITLE-ONLY-PROJECTION")
        )
        title_only_path = projects / "Legacy Title Only.md"
        title_only_bytes = b"# Legacy Title Only\r\n\r\nKEEP-TITLE-ONLY\r\n"
        title_only_path.write_bytes(title_only_bytes)
        _invoke(
            fixture,
            "export_goal",
            {"goal_id": title_only_id},
            "title-only legacy export",
        )
        title_owned = _owned_goal_path(fixture, title_only_id, "title-only legacy export")
        if title_only_path.read_bytes() != title_only_bytes:
            raise SystemExit("title-only legacy goal note was not preserved byte-for-byte")
        if title_owned == title_only_path:
            raise SystemExit("title-only legacy goal note was adopted as an owned projection")

        _invoke(fixture, "export_goal", {"goal_id": id_only_id}, "ID-only legacy re-export")
        _invoke(
            fixture,
            "export_goal",
            {"goal_id": title_only_id},
            "title-only legacy re-export",
        )
        if id_only_path.read_bytes() != id_only_bytes or title_only_path.read_bytes() != title_only_bytes:
            raise SystemExit("legacy goal notes changed during repeated owned publication")


def _expect_export_failure(
    fixture: GoalFixture,
    goal_id: int,
    label: str,
) -> None:
    try:
        fixture.handlers["export_goal"]({"goal_id": goal_id})
    except (FileExistsError, RuntimeError):
        return
    raise SystemExit(f"{label} did not fail closed at the occupied canonical destination")


def test_foreign_malformed_and_duplicate_ownership_is_preserved() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-adversarial-") as temp:
        root = Path(temp)
        fixture = _setup(root / "store", root / "Vault")
        projects = fixture.vault.root_path / "Projects"
        identity = fixture.store.get_store_identity()
        foreign_identity = "f" * 32 if identity != "f" * 32 else "e" * 32

        cases: list[tuple[int, Path, bytes, str]] = []
        foreign_id = fixture.store.create_goal(
            GoalRecord("Foreign Ownership", "PRIVATE-NEW-FOREIGN-PROJECTION")
        )
        cases.append(
            (
                foreign_id,
                projects / f"Foreign Ownership [{foreign_id}-{identity}].md",
                (
                    "---\n"
                    "jarvis_projection: goal\n"
                    f"store_identity: {foreign_identity}\n"
                    f"goal_id: {foreign_id}\n"
                    f"id: {foreign_id}\n"
                    "status: active\n"
                    "horizon: someday\n"
                    "updated: 2026-07-14\n"
                    "---\n\n# Foreign store projection\n\nKEEP-FOREIGN\n"
                ).encode(),
                "foreign ownership",
            )
        )

        malformed_id = fixture.store.create_goal(
            GoalRecord("Malformed Ownership", "PRIVATE-NEW-MALFORMED-PROJECTION")
        )
        cases.append(
            (
                malformed_id,
                projects / f"Malformed Ownership [{malformed_id}-{identity}].md",
                (
                    "---\n"
                    "jarvis_projection: goal\n"
                    f"store_identity: {identity}\n"
                    f"goal_id: {malformed_id}\n"
                    f"id: {malformed_id}\n"
                    "status: active\n"
                    "horizon: someday\n"
                    "updated: 2026-07-14\n"
                    "# missing closing frontmatter delimiter\n"
                    "KEEP-MALFORMED\n"
                ).encode(),
                "malformed ownership",
            )
        )

        duplicate_id = fixture.store.create_goal(
            GoalRecord("Duplicate Ownership", "PRIVATE-NEW-DUPLICATE-PROJECTION")
        )
        cases.append(
            (
                duplicate_id,
                projects / f"Duplicate Ownership [{duplicate_id}-{identity}].md",
                (
                    "---\r\n"
                    "jarvis_projection: goal\r\n"
                    f"store_identity: {identity}\r\n"
                    f"store_identity: {foreign_identity}\r\n"
                    f"goal_id: {duplicate_id}\r\n"
                    f"id: {duplicate_id}\r\n"
                    "status: active\r\n"
                    "horizon: someday\r\n"
                    "updated: 2026-07-14\r\n"
                    "---\r\n\r\n# Ambiguous duplicate owner\r\n\r\nKEEP-DUPLICATE\r\n"
                ).encode(),
                "duplicate ownership",
            )
        )

        quoted_foreign_id = fixture.store.create_goal(
            GoalRecord("Quoted Foreign Ownership", "PRIVATE-NEW-QUOTED-FOREIGN")
        )
        cases.append(
            (
                quoted_foreign_id,
                projects / f"Quoted Foreign Ownership [{quoted_foreign_id}-{identity}].md",
                (
                    "---\n"
                    "jarvis_projection: goal\n"
                    f"store_identity: {identity}\n"
                    f"\"store_identity\": {foreign_identity}\n"
                    f"goal_id: {quoted_foreign_id}\n"
                    f"id: {quoted_foreign_id}\n"
                    "status: active\n"
                    "horizon: someday\n"
                    "updated: 2026-07-14\n"
                    "---\n\n# Semantically foreign owner\n\nKEEP-QUOTED-FOREIGN\n"
                ).encode(),
                "quoted foreign ownership",
            )
        )

        invalid_yaml_id = fixture.store.create_goal(
            GoalRecord("Invalid YAML Ownership", "PRIVATE-NEW-INVALID-YAML")
        )
        cases.append(
            (
                invalid_yaml_id,
                projects / f"Invalid YAML Ownership [{invalid_yaml_id}-{identity}].md",
                (
                    "---\n"
                    "jarvis_projection: goal\n"
                    f"store_identity: {identity}\n"
                    f"goal_id: {invalid_yaml_id}\n"
                    f"id: {invalid_yaml_id}\n"
                    "status: active\n"
                    "horizon: someday\n"
                    "updated: 2026-07-14\n"
                    "broken: [\n"
                    "---\n\n# Invalid YAML owner\n\nKEEP-INVALID-YAML\n"
                ).encode(),
                "invalid YAML ownership",
            )
        )

        invalid_plain_id = fixture.store.create_goal(
            GoalRecord("Invalid Plain Scalar", "PRIVATE-NEW-INVALID-PLAIN")
        )
        cases.append(
            (
                invalid_plain_id,
                projects / f"Invalid Plain Scalar [{invalid_plain_id}-{identity}].md",
                (
                    "---\n"
                    "jarvis_projection: goal\n"
                    f"store_identity: {identity}\n"
                    f"goal_id: {invalid_plain_id}\n"
                    f"id: {invalid_plain_id}\n"
                    "status: active\n"
                    "horizon: - item\n"
                    "updated: 2026-07-14\n"
                    "---\n\n# Invalid plain scalar owner\n\nKEEP-INVALID-PLAIN\n"
                ).encode(),
                "invalid plain scalar ownership",
            )
        )

        for goal_id, path, original, label in cases:
            path.write_bytes(original)
            _expect_export_failure(fixture, goal_id, f"{label} export")
            if path.read_bytes() != original:
                raise SystemExit(f"{label} note was not preserved byte-for-byte")

        for goal_id, path, original, label in reversed(cases):
            _expect_export_failure(fixture, goal_id, f"{label} re-export")
            if path.read_bytes() != original:
                raise SystemExit(f"{label} note changed during repeated publication")


def test_generated_frontmatter_and_filename_remain_owned() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-generated-") as temp:
        root = Path(temp)
        fixture = _setup(root / "store", root / "Vault")
        identity = fixture.store.get_store_identity()
        goal = {
            "id": 1,
            "revision": 1,
            "title": "Owned\x00 Goal " + ("한" * 200),
            "purpose": "PRIVATE-GENERATED-GOAL",
            "horizon": "- item\u0085null\u2028true\nstore_identity: " + ("f" * 32),
            "status": "true\ngoal_id: 999",
            "created_at": "2026-07-14T00:00:00Z",
            "updated_at": "2026-07-14T00:00:00Z",
        }
        path, _digest, _revision = fixture.vault.write_goal_with_evidence(
            goal,
            [],
            store_identity=identity,
        )
        values = _frontmatter_values(path)
        if (
            values.get("jarvis_projection") != ["goal"]
            or values.get("store_identity") != [identity]
            or values.get("goal_id") != ["1"]
            or values.get("id") != ["1"]
        ):
            raise SystemExit(f"generated goal frontmatter lost exact ownership: {values!r}")
        if (
            json.loads(values.get("status", [""])[0]) != goal["status"]
            or json.loads(values.get("horizon", [""])[0]) != goal["horizon"]
            or "\u0085" in path.read_text(encoding="utf-8")
            or "\u2028" in path.read_text(encoding="utf-8")
        ):
            raise SystemExit("generated goal frontmatter did not preserve unsafe scalars as escaped strings")
        if "\x00" in path.name or len(path.name.encode("utf-8")) > 255:
            raise SystemExit("generated goal filename retained a control byte or exceeded filesystem bounds")
        fixture.vault.write_goal_with_evidence(goal, [], store_identity=identity)

        organized_path, _changed, _content_digest = fixture.vault.write_organized_goal(
            goal,
            [],
            store_identity=identity,
        )
        if organized_path != path or _frontmatter_values(organized_path).get("store_identity") != [identity]:
            raise SystemExit("organized goal writer diverged from exact store-owned frontmatter")


def test_identity_and_control_validation_precede_source_mutation() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-preflight-") as temp:
        root = Path(temp)
        fixture = _setup(root / "store", root / "Vault")
        before = len(fixture.store.list_goals(limit=200))

        control_result = fixture.handlers["create_goal"](
            {"title": "invalid\x00goal", "purpose": "must not commit", "horizon": "soon"}
        )
        if control_result.ok or len(fixture.store.list_goals(limit=200)) != before:
            raise SystemExit("control-bearing goal title reached the source database")

        organize = make_organize_tools(fixture.store, fixture.vault)
        organize_control = organize({"text": "goal: invalid\x00organizer goal"})
        if organize_control.ok or len(fixture.store.list_goals(limit=200)) != before:
            raise SystemExit("control-bearing organizer goal reached the source database")

        with patch.object(fixture.store, "get_store_identity", return_value="invalid"):
            try:
                fixture.handlers["create_goal"](
                    {"title": "Identity Must Preflight", "purpose": "must not commit"}
                )
            except RuntimeError:
                pass
            else:
                raise SystemExit("invalid goal store identity did not fail closed")
        if len(fixture.store.list_goals(limit=200)) != before:
            raise SystemExit("invalid identity was detected after goal source mutation")

        with patch.object(fixture.store, "get_store_identity", return_value="invalid"):
            try:
                organize({"text": "goal: Organizer Identity Must Preflight"})
            except RuntimeError:
                pass
            else:
                raise SystemExit("invalid organizer store identity did not fail closed")
        if len(fixture.store.list_goals(limit=200)) != before:
            raise SystemExit("organizer reserved goal state before identity validation")


def test_concurrent_cross_store_publication_cannot_replace_either_projection() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-race-") as temp:
        root = Path(temp)
        shared_vault = root / "Vault"
        first = _setup(root / "store-a", shared_vault)
        second = _setup(root / "store-b", shared_vault)
        first_marker = "PRIVATE-FIRST-CONCURRENT-GOAL"
        second_marker = "PRIVATE-SECOND-CONCURRENT-GOAL"
        first_id = first.store.create_goal(GoalRecord("Concurrent Shared Goal", first_marker))
        second_id = second.store.create_goal(GoalRecord("Concurrent Shared Goal", second_marker))
        if (first_id, second_id) != (1, 1):
            raise SystemExit("concurrent cross-store fixture did not create goal #1 in both stores")

        barrier = Barrier(2)
        first_write = first.vault.write_goal_with_evidence
        second_write = second.vault.write_goal_with_evidence

        def synchronized(real_write: Callable[..., tuple[Path, str, str]]):
            def publish(*args: Any, **kwargs: Any) -> tuple[Path, str, str]:
                try:
                    barrier.wait(timeout=5)
                except BrokenBarrierError as exc:
                    raise RuntimeError("cross-store goal publication barrier broke") from exc
                return real_write(*args, **kwargs)

            return publish

        def attempt(fixture: GoalFixture, goal_id: int) -> ToolResult | BaseException:
            try:
                return fixture.handlers["export_goal"]({"goal_id": goal_id})
            except BaseException as exc:
                return exc

        with (
            patch.object(
                first.vault,
                "write_goal_with_evidence",
                side_effect=synchronized(first_write),
            ),
            patch.object(
                second.vault,
                "write_goal_with_evidence",
                side_effect=synchronized(second_write),
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first_future = pool.submit(attempt, first, first_id)
            second_future = pool.submit(attempt, second, second_id)
            first_result = first_future.result(timeout=8)
            second_result = second_future.result(timeout=8)

        for label, result in (("first", first_result), ("second", second_result)):
            if isinstance(result, BaseException):
                raise SystemExit(
                    f"{label} concurrent cross-store publication raised "
                    f"{type(result).__name__}"
                )
            if not result.ok:
                raise SystemExit(
                    f"{label} concurrent cross-store publication failed: "
                    f"failure_kind={result.metadata.get('failure_kind')!r}"
                )

        first_path = _owned_goal_path(first, first_id, "first concurrent publication")
        second_path = _owned_goal_path(second, second_id, "second concurrent publication")
        if first_path == second_path:
            raise SystemExit("concurrent stores published goal #1 to the same owned path")
        _assert_marker_partition(
            first_path,
            second_path,
            first_marker=first_marker,
            second_marker=second_marker,
            label="concurrent cross-store publication",
        )

        second_before = second_path.read_bytes()
        _invoke(first, "export_goal", {"goal_id": first_id}, "first post-race export")
        if second_path.read_bytes() != second_before:
            raise SystemExit("first post-race export replaced the second store projection")
        first_before = first_path.read_bytes()
        _invoke(second, "export_goal", {"goal_id": second_id}, "second post-race export")
        if first_path.read_bytes() != first_before:
            raise SystemExit("second post-race export replaced the first store projection")


def _iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_strings(item)


def test_failure_metadata_hides_paths_and_private_markers() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-privacy-") as temp:
        root = Path(temp)
        runtime = _runtime(root)
        title_marker = "PRIVATE-FAILURE-GOAL-TITLE"
        purpose_marker = "PRIVATE-FAILURE-GOAL-PURPOSE"
        exception_marker = "PRIVATE-GOAL-WRITE-EXCEPTION"
        goal_id = runtime.store.create_goal(GoalRecord(title_marker, purpose_marker))
        runtime.planner = StaticPlanner(goal_id)
        private_failure_path = runtime.vault.root_path / "Projects" / "private-failure.md"

        def fail_write(*_args: Any, **_kwargs: Any) -> tuple[Path, str, str]:
            raise OSError(f"{exception_marker}: {private_failure_path}")

        with patch.object(runtime.vault, "write_goal_with_evidence", side_effect=fail_write):
            result = runtime.handle(
                "publish the goal ownership failure fixture",
                request_token="goal-projection-ownership-failure",
            )
        if len(result.tool_results) != 1 or result.tool_results[0].ok:
            raise SystemExit("mocked goal publication failure did not return one failed tool result")

        audit_metadata: list[object] = []
        with runtime.store.connect() as conn:
            rows = conn.execute(
                "SELECT metadata FROM tool_runs WHERE tool_name = 'export_goal' AND ok = 0 ORDER BY id"
            ).fetchall()
        for row in rows:
            raw = row["metadata"]
            try:
                audit_metadata.append(json.loads(raw or "{}"))
            except (TypeError, ValueError):
                audit_metadata.append(str(raw or ""))

        surfaces = {
            "tool_result": result.tool_results[0].metadata,
            "runtime": result.metadata,
            "audit": audit_metadata,
        }
        strings = list(_iter_strings(surfaces))
        forbidden = (
            title_marker,
            purpose_marker,
            exception_marker,
            runtime.store.get_store_identity(),
            str(root),
            str(runtime.vault.root_path),
            str(private_failure_path),
        )
        if any(secret and any(secret in value for value in strings) for secret in forbidden):
            raise SystemExit("goal publication failure metadata leaked a private marker or known path")
        if any(ABSOLUTE_LOCAL_PATH_RE.search(value) for value in strings):
            raise SystemExit("goal publication failure metadata exposed an absolute local path")


def test_success_receipts_hide_store_identity_and_absolute_path() -> None:
    with TemporaryDirectory(prefix="jarvis-goal-owner-success-privacy-") as temp:
        root = Path(temp)
        fixture = _setup(root / "store", root / "Vault")
        result = _invoke(
            fixture,
            "create_goal",
            {"title": "Safe Goal Receipt", "purpose": "success metadata privacy"},
            "success metadata privacy",
        )
        exported = _invoke(
            fixture,
            "export_goal",
            {"goal_id": 1},
            "success export metadata privacy",
        )
        identity = fixture.store.get_store_identity()
        strings = list(
            _iter_strings(
                {
                    "create_output": result.output,
                    "create_metadata": result.metadata,
                    "export_output": exported.output,
                    "export_metadata": exported.metadata,
                }
            )
        )
        if result.metadata.get("path") is not None or exported.metadata.get("path") is not None:
            raise SystemExit("goal success metadata retained an absolute projection path")
        if any(identity in value or str(root) in value for value in strings):
            raise SystemExit("goal success receipt exposed store identity or local root")
        if any(ABSOLUTE_LOCAL_PATH_RE.search(value) for value in strings):
            raise SystemExit("goal success receipt exposed an absolute local path")
        path_display = str(result.metadata.get("path_display") or "")
        if (
            path_display != "Projects/Goal 1.md"
            or exported.metadata.get("path_display") != path_display
            or path_display not in exported.output
        ):
            raise SystemExit(f"goal success receipt missed its opaque relative display: {result.metadata!r}")


def main() -> None:
    test_two_stores_same_id_and_title_remain_separate()
    test_same_store_mutations_converge_to_one_owned_path()
    test_legacy_id_only_and_title_only_notes_are_preserved()
    test_foreign_malformed_and_duplicate_ownership_is_preserved()
    test_generated_frontmatter_and_filename_remain_owned()
    test_identity_and_control_validation_precede_source_mutation()
    test_concurrent_cross_store_publication_cannot_replace_either_projection()
    test_failure_metadata_hides_paths_and_private_markers()
    test_success_receipts_hide_store_identity_and_absolute_path()
    print("Goal projection ownership smoke passed")


if __name__ == "__main__":
    main()
