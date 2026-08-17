from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import sqlite3
import threading
from unittest.mock import patch

from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryRecord, MemoryStore, PreferenceRecord, SkillRecord, TaskRecord, normalized_task_identity
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.memory_curator import (
    MAX_LEARNING_REVIEW_BODY_CHARS,
    MEMORY_BOUNDARY,
    _learning_review_display,
    _memory_handoff_metadata,
    _memory_metadata_bool,
    make_memory_curator_tools,
)
from jarvis_v2.tools.knowledge_promotion import make_knowledge_promotion_tools


def test_planner_routes_learning_review_aliases() -> None:
    # Real gap found live 2026-07-09: "show learning review" / "show my
    # learning review" fell through to chat while bare "learning review" and
    # "what have you learned" both worked.
    p = RuleBasedPlanner()
    for q in (
        "learning review",
        "show learning review",
        "show my learning review",
        "show my learning review?",
        "please show my learning review",
        "show my learning review please",
    ):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["learning_review"]:
            raise SystemExit(f"learning_review route missed: {q!r} -> {[a.tool_name for a in actions]}")
    for q in (
        "queue learning tasks",
        "create learning tasks",
        "add learning tasks",
        "task learning review",
        "turn learning review into tasks",
        "queue learning tasks?",
        "please queue learning tasks",
        "queue learning tasks please",
    ):
        actions = p.plan(q).actions
        if [a.tool_name for a in actions] != ["queue_learning_tasks"]:
            raise SystemExit(
                f"queue_learning_tasks route missed: {q!r} -> {[a.tool_name for a in actions]}"
            )


class HostileMetadataValue:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def __bool__(self) -> bool:
        raise RuntimeError(self.marker)

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-learning-metadata {self.marker}>"


def assert_contains(text: str, expected: list[str], label: str) -> None:
    missing = [item for item in expected if item not in text]
    if missing:
        raise SystemExit(f"{label} missing expected text: {missing}")


def assert_memory_metadata_bool_is_exact() -> None:
    if _memory_metadata_bool(True) is not True:
        raise SystemExit("memory exact metadata bool rejected True")
    if _memory_metadata_bool(False) is not False:
        raise SystemExit("memory exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if _memory_metadata_bool(value) is not False:
            raise SystemExit(f"memory exact metadata bool accepted malformed truthy value: {value!r}")
    if _memory_metadata_bool("false", default=True) is not True:
        raise SystemExit("memory exact metadata bool did not preserve explicit default")


def assert_memory_malformed_handoff_flags_are_exact() -> None:
    handoff = {
        "source": "learning_review",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["learning review"],
        "boundaries": {"read_only": True},
    }
    metadata = _memory_handoff_metadata("learning_review_handoff", handoff)
    for key in ["state_changed", "content_in_handoff"]:
        if metadata.get(key) is not False:
            raise SystemExit(f"memory handoff accepted malformed {key}: {metadata}")


def test_fractional_memory_ids_are_rejected_before_lookup() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-review-fractional-id-") as temp:
        runtime = make_temp_runtime(Path(temp))
        memory_id = runtime.store.add_memory(
            MemoryRecord("facts", "Integral identifier", "Readable by its exact ID.", "manual")
        )
        get_memory = make_memory_curator_tools(runtime.store, runtime.vault)[2]
        integral = get_memory({"memory_id": float(memory_id)})
        if not integral.ok or integral.metadata.get("memory_id") != memory_id:
            raise SystemExit(f"integral float memory id was not preserved: {integral}")

        with patch.object(
            runtime.store,
            "memory_read_custody",
            side_effect=AssertionError("fractional ID reached memory lookup"),
        ) as custody_lookup:
            for fractional_id in (1.25, 2.5, 42.0000000001):
                result = get_memory({"memory_id": fractional_id})
                if result.ok or "must be a number" not in result.output:
                    raise SystemExit(
                        f"fractional memory id was truncated instead of rejected: {fractional_id!r} -> {result}"
                    )
            if custody_lookup.call_count:
                raise SystemExit("fractional memory ids reached custody lookup")


def _mock_promotion_row(
    *, memory_id: int, target_kind: str, target_total: int, candidate_total: int
) -> dict[str, object]:
    target_counts = {
        "decision": 0,
        "goal": 0,
        "person": 0,
        "preference": 0,
        "profile": 0,
    }
    if candidate_total == target_total * len(target_counts):
        target_counts = {kind: target_total for kind in target_counts}
    else:
        target_counts[target_kind] = target_total
    return {
        "id": memory_id,
        "revision": 1,
        "target_kind": target_kind,
        "target_total": target_total,
        "candidate_total": candidate_total,
        "category": "identity" if target_kind == "profile" else f"{target_kind}s",
        "title": f"Mock {target_kind} candidate {memory_id}",
        **{f"{kind}_total": count for kind, count in target_counts.items()},
    }


def test_learning_review_rotations_do_not_couple_target_and_row_offsets() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-review-independent-rotation-") as temp:
        runtime = make_temp_runtime(Path(temp))
        learning_review = make_memory_curator_tools(runtime.store, runtime.vault)[8]
        target_kinds = ("decision", "goal", "person", "preference", "profile")
        rows_per_target = 5
        candidate_total = len(target_kinds) * rows_per_target
        observed: dict[str, set[int]] = {kind: set() for kind in target_kinds}
        calls: list[tuple[int, int]] = []

        def coupled_store_rows(requested_limit: int, *, rotation_seed: int = 0):
            calls.append((requested_limit, rotation_seed))
            target_start = rotation_seed % len(target_kinds)
            target_order = target_kinds[target_start:] + target_kinds[:target_start]
            rows = []
            for rank in range(rows_per_target):
                row_offset = (rotation_seed + rank) % rows_per_target
                for target_kind in target_order:
                    target_index = target_kinds.index(target_kind)
                    rows.append(
                        _mock_promotion_row(
                            memory_id=(target_index * 100) + row_offset + 1,
                            target_kind=target_kind,
                            target_total=rows_per_target,
                            candidate_total=candidate_total,
                        )
                    )
            return rows[:requested_limit]

        with (
            patch.object(
                runtime.store,
                "list_knowledge_promotion_candidates",
                side_effect=coupled_store_rows,
            ),
            patch("jarvis_v2.tools.memory_curator.datetime") as clock,
        ):
            for ordinal in range(1000, 1000 + candidate_total):
                clock.now.return_value.date.return_value.toordinal.return_value = ordinal
                result = learning_review({"limit": 1})
                bindings = result.metadata.get("knowledge_promotion_candidate_bindings") or []
                if len(bindings) != 1:
                    raise SystemExit(f"limit-one rotation returned the wrong binding count: {result.metadata}")
                binding = bindings[0]
                target_kind = binding.get("target_kind")
                memory_id = binding.get("memory_id")
                if target_kind not in observed or type(memory_id) is not int:
                    raise SystemExit(f"rotation returned an invalid mocked binding: {binding}")
                observed[target_kind].add(memory_id)
                if (
                    result.metadata.get("knowledge_promotion_rotation_seed") != ordinal
                    or calls[-1] != (200, ordinal // len(target_kinds))
                ):
                    raise SystemExit(
                        "target and within-target rotation seeds were not independently derived: "
                        f"ordinal={ordinal}, call={calls[-1]}, metadata={result.metadata}"
                    )

        starved = {
            target_kind: sorted(memory_ids)
            for target_kind, memory_ids in observed.items()
            if len(memory_ids) != rows_per_target
        }
        if starved:
            raise SystemExit(f"coupled deterministic rotations starved candidate rows: {starved}")


def test_learning_review_partial_candidate_block_is_not_counted_as_rendered() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-review-partial-candidate-") as temp:
        runtime = make_temp_runtime(Path(temp))
        learning_review = make_memory_curator_tools(runtime.store, runtime.vault)[8]
        row = _mock_promotion_row(
            memory_id=71,
            target_kind="profile",
            target_total=1,
            candidate_total=1,
        )
        with patch.object(
            runtime.store,
            "list_knowledge_promotion_candidates",
            return_value=[row],
        ):
            complete = learning_review({"limit": 1})
            candidate_line = next(
                line
                for line in complete.output.splitlines()
                if line.startswith("  - memory #71 revision ")
            )
            review_line = next(
                line
                for line in complete.output.splitlines()
                if line.startswith("    Review first: `knowledge promotion packet 71`")
            )
            if (
                "Decision, new-preference, and profile transfer are available only after explicit fields and approval"
                not in review_line
                or "person and goal remain review-only" not in review_line
            ):
                raise SystemExit(
                    "profile promotion availability drifted from the learning-review candidate guidance"
                )
            review_start = complete.output.index(review_line)
            suffix = "\n\n[Learning review content truncated.]"
            clipped_body_limit = review_start + max(1, len(review_line) // 2)
            clipped_output_limit = clipped_body_limit + len(MEMORY_BOUNDARY) + len(suffix)
            with patch(
                "jarvis_v2.tools.memory_curator.MAX_LEARNING_REVIEW_BODY_CHARS",
                clipped_output_limit,
            ):
                clipped = learning_review({"limit": 1})

        if (
            candidate_line not in clipped.output.splitlines()
            or review_line in clipped.output
            or "[Learning review content truncated.]" not in clipped.output
            or clipped.metadata.get("knowledge_promotion_candidates") != 1
            or clipped.metadata.get("knowledge_promotion_rendered_candidates") != 0
            or clipped.metadata.get("knowledge_promotion_output_hidden_count") != 1
        ):
            raise SystemExit(
                f"partially clipped candidate was counted as fully rendered: {clipped.metadata}"
            )
        handoff = clipped.metadata.get("learning_review_handoff") or {}
        if (
            handoff.get("knowledge_promotion_rendered_candidates") != 0
            or handoff.get("knowledge_promotion_output_hidden_count") != 1
        ):
            raise SystemExit(f"partial candidate counts diverged in handoff metadata: {handoff}")


def test_learning_review_sanitizes_stored_display_content() -> None:
    with TemporaryDirectory(prefix="jarvis-learning-review-display-") as temp:
        runtime = make_temp_runtime(Path(temp))
        raw_path = "/\x55sers/example/private/learning-secret.txt"
        runtime.store.add_memory(
            MemoryRecord(
                category="feedback",
                title=f"Feedback title\n## injected-heading `tick` <script> {raw_path}",
                body="Feedback body\n- injected-list `tick` <b>unsafe</b>",
                source="display-smoke",
            )
        )
        runtime.store.add_memory(
            MemoryRecord(
                category="weak\n## category-injection",
                title=f"Likely weak title\n# title-injection {raw_path}",
                body="likely weak body",
                source="display-smoke",
                confidence=0.2,
            )
        )
        runtime.store.set_preference(
            PreferenceRecord(
                category="display\n## preference-category",
                key="key] `tick`\n# preference-key",
                value=f"value\n- preference-list <img> {raw_path}",
            )
        )
        runtime.store.save_skill(
            SkillRecord(
                name="skill\n## skill-heading `tick`",
                trigger="display-smoke",
                body=f"summary\n- skill-list <script> {raw_path} " + ("z" * 500),
            )
        )
        hostile_tool = f"tool`name\n## tool-heading <script> {raw_path}"
        runtime.store.log_tool_run(
            "display-smoke",
            hostile_tool,
            "READ_ONLY",
            False,
            False,
            "failed",
        )
        tools = make_memory_curator_tools(runtime.store, runtime.vault)
        learning_review = tools[8]
        save_learning_review = tools[9]
        review = learning_review({"limit": 12})
        saved = save_learning_review({"limit": 12})
        note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Learning Review.md"
        note_text = note.read_text(encoding="utf-8")
        for rendered, label in (
            (review.output, "read-only review"),
            (saved.output, "saved review output"),
            (note_text, "saved review note"),
        ):
            for forbidden in (
                raw_path,
                "\n## injected-heading",
                "\n- injected-list",
                "\n## category-injection",
                "\n# title-injection",
                "\n## preference-category",
                "\n# preference-key",
                "\n- preference-list",
                "\n## skill-heading",
                "\n- skill-list",
                "\n## tool-heading",
                "<script>",
                "<img>",
                "<b>",
            ):
                if forbidden in rendered:
                    raise SystemExit(f"{label} leaked unsafe stored Markdown content {forbidden!r}: {rendered[:1600]}")
            if "&lt;local-path&gt;" not in rendered or "&lt;script&gt;" not in rendered:
                raise SystemExit(f"{label} did not visibly sanitize hostile stored content: {rendered[:1600]}")
            if len(rendered) > MAX_LEARNING_REVIEW_BODY_CHARS + 256:
                raise SystemExit(f"{label} exceeded the bounded Learning Review output contract: {len(rendered)}")
        headings = [
            line
            for line in note_text.splitlines()
            if line.startswith("# ") or line.startswith("## ") or line.startswith("### ")
        ]
        if headings != ["# Learning Review"]:
            raise SystemExit(f"hostile stored content created Markdown headings: {headings}")
        if "\r" in note_text:
            raise SystemExit("saved review retained a hostile carriage return")
        if "`tick`" in note_text or "`name" in note_text:
            raise SystemExit(f"saved review note preserved hostile inline-code delimiters: {note_text[:1600]}")
        if "z" * 121 in note_text:
            raise SystemExit("saved review note did not bound a hostile skill summary")
        target_display = review.metadata.get("execution_learning_target_tool")
        if not isinstance(target_display, str) or target_display == hostile_tool or raw_path in target_display:
            raise SystemExit(f"learning-review target evidence was not display-safe: {review.metadata}")
        if "execution_learning_target_tool" not in review.metadata:
            raise SystemExit("display sanitization changed the existing learning-review metadata shape")
        assert_review_only_metadata(
            review.metadata, "learning_review_hostile_display", reads_private_data=True
        )
        assert_operator_metadata(saved.metadata, "save_learning_review_hostile_display")
        assert_learning_review_handoff(review.metadata, "learning_review_hostile_display", writes=False)
        assert_learning_review_handoff(saved.metadata, "save_learning_review_hostile_display", writes=True)


def test_learning_task_queue_identity_is_complete_and_atomic() -> None:
    candidate = "Review Jarvis feedback report and decide whether to add a preference, skill, or smoke test."
    with TemporaryDirectory(prefix="jarvis-learning-queue-old-identity-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for index in range(301):
            runtime.store.add_task(TaskRecord(body=f"newer open learning filler {index}", source="seed"))
        old_task_id = runtime.store.add_task(
            TaskRecord(body=f"  {candidate.upper()}  ", source="old-learning", status="open")
        )
        former_window_ids = {
            int(row["id"])
            for row in runtime.store.list_tasks(status="open", limit=300)
        }
        if old_task_id in former_window_ids:
            raise SystemExit("learning queue fixture did not place the target beyond the former 300-row window")
        runtime.store.add_memory(
            MemoryRecord(category="feedback", title="Queue identity feedback", body="Queue this once.")
        )
        runtime.store.set_preference(PreferenceRecord(category="work", key="queue style", value="reviewed"))
        runtime.store.save_skill(SkillRecord(name="Queue identity skill", trigger="queue", body="reviewed"))
        queue_learning_tasks = make_memory_curator_tools(runtime.store, runtime.vault)[10]
        result = queue_learning_tasks({})
        if (
            not result.ok
            or result.metadata.get("added") != 0
            or result.metadata.get("skipped") != 1
        ):
            raise SystemExit(f"learning queue duplicated an open task older than 300 rows: {result}")
        matching = [
            row
            for row in runtime.store.list_tasks(status="open", limit=None)
            if normalized_task_identity(row["body"]) == normalized_task_identity(candidate)
        ]
        if len(matching) != 1:
            raise SystemExit(f"older learning queue identity count diverged: {len(matching)}")
        assert_queue_learning_tasks_handoff(result.metadata, "queue_learning_tasks_old_identity")

        if runtime.store.set_task_status(old_task_id, "done") is None:
            raise SystemExit("learning queue fixture could not complete the prior reminder")
        reopened = queue_learning_tasks({})
        if reopened.metadata.get("added") != 1 or reopened.metadata.get("skipped") != 0:
            raise SystemExit(f"done-only learning reminder did not allow one new open row: {reopened}")
        repeat = queue_learning_tasks({})
        if repeat.metadata.get("added") != 0 or repeat.metadata.get("skipped") != 1:
            raise SystemExit(f"new open learning reminder did not block the next repeat: {repeat}")
        all_matches = [
            row
            for row in runtime.store.list_tasks(status=None, limit=None)
            if normalized_task_identity(row["body"]) == normalized_task_identity(candidate)
        ]
        open_matches = [row for row in all_matches if str(row["status"]).lower() == "open"]
        if len(all_matches) != 2 or len(open_matches) != 1:
            raise SystemExit(
                f"learning reminder lifecycle identity diverged: all={len(all_matches)} open={len(open_matches)}"
            )
        assert_queue_learning_tasks_handoff(reopened.metadata, "queue_learning_tasks_done_reopen")
        assert_queue_learning_tasks_handoff(repeat.metadata, "queue_learning_tasks_open_repeat")

    with TemporaryDirectory(prefix="jarvis-learning-queue-contention-") as temp:
        root = Path(temp)
        db_path = root / "memory.db"
        first_store = MemoryStore(db_path)
        first_store.init()
        second_store = MemoryStore(db_path)
        first_vault = ObsidianVault(root / "Vault")
        second_vault = ObsidianVault(root / "Vault")
        first_vault.init()
        first_store.add_memory(
            MemoryRecord(category="feedback", title="Concurrent queue feedback", body="Queue this once.")
        )
        first_store.set_preference(PreferenceRecord(category="work", key="queue style", value="reviewed"))
        first_store.save_skill(SkillRecord(name="Concurrent queue skill", trigger="queue", body="reviewed"))
        first_queue = make_memory_curator_tools(first_store, first_vault)[10]
        second_queue = make_memory_curator_tools(second_store, second_vault)[10]
        barrier = threading.Barrier(2)

        def contend(queue_tool: object) -> object:
            barrier.wait(timeout=5)
            return queue_tool({})

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [
                future.result(timeout=15)
                for future in (
                    pool.submit(contend, first_queue),
                    pool.submit(contend, second_queue),
                )
            ]
        if sorted(result.metadata.get("added") for result in results) != [0, 1]:
            raise SystemExit(f"concurrent learning queues did not choose one insertion: {results}")
        if sorted(result.metadata.get("skipped") for result in results) != [0, 1]:
            raise SystemExit(f"concurrent learning queues did not report one duplicate: {results}")
        matching = [
            row
            for row in first_store.list_tasks(status="open", limit=None)
            if normalized_task_identity(row["body"]) == normalized_task_identity(candidate)
        ]
        if len(matching) != 1:
            raise SystemExit(f"concurrent learning queue inserted {len(matching)} matching tasks")
        mirror = root / "Vault" / "Jarvis" / "Tasks" / "Open Tasks.md"
        mirror_text = mirror.read_text(encoding="utf-8")
        if mirror_text.count(candidate) != 1:
            raise SystemExit(f"concurrent learning queue mirror was not one complete canonical projection: {mirror_text}")
        for index, result in enumerate(results):
            assert_queue_learning_tasks_handoff(result.metadata, f"queue_learning_tasks_contention_{index}")

        first_store.add_task(TaskRecord(body=candidate, source="intentional-repeat"))
        direct_matches = [
            row
            for row in first_store.list_tasks(status="open", limit=None)
            if normalized_task_identity(row["body"]) == normalized_task_identity(candidate)
        ]
        if len(direct_matches) != 2:
            raise SystemExit("atomic learning queue scope changed intentional ordinary add_task repeats")


def test_learning_task_queue_binds_after_action_and_sanitizes_tool_names() -> None:
    raw_path = "/\x55sers/example/private/failed-tool.txt"
    hostile_tool = (
        "broken`tool\n## injected <script> "
        r"\[click\](javascript:alert(1))"
        f" \x00\u001b\u202e {raw_path}"
    )
    with TemporaryDirectory(prefix="jarvis-learning-queue-target-proof-") as temp:
        runtime = make_temp_runtime(Path(temp))
        failed_run_id = runtime.store.log_tool_run(
            "queue-target",
            hostile_tool,
            "LOCAL_SAFE",
            False,
            False,
            "failed",
        )
        unrelated_run_id = runtime.store.log_tool_run(
            "queue-target",
            "calculate",
            "READ_ONLY",
            True,
            False,
            "4",
        )
        runtime.store.log_tool_run(
            "queue-target",
            "after_action_learning_packet",
            "READ_ONLY",
            True,
            False,
            "reviewed another run",
            metadata={"run_id": unrelated_run_id},
        )
        runtime.store.log_tool_run(
            "queue-target",
            "after_action_learning_packet",
            "READ_ONLY",
            True,
            False,
            "malformed run identity",
            metadata={"run_id": True},
        )
        runtime.store.log_tool_run(
            "queue-target",
            "after_action_learning_packet",
            "READ_ONLY",
            False,
            False,
            "failed review packet",
            metadata={"run_id": failed_run_id},
        )
        queue_learning_tasks = make_memory_curator_tools(runtime.store, runtime.vault)[10]
        result = queue_learning_tasks({})
        expected_prefix = f"Review after-action learning for failed run #{failed_run_id}"
        if not result.ok or result.metadata.get("added") != 1 or expected_prefix not in result.output:
            raise SystemExit(f"unrelated after-action proof suppressed the failed-run reminder: {result}")
        task = runtime.store.list_tasks(status="open", limit=10)[0]
        mirror = runtime.vault.root_path / "Tasks" / "Open Tasks.md"
        rendered = "\n".join((result.output, str(task["body"]), mirror.read_text(encoding="utf-8")))
        for forbidden in (raw_path, "\n## injected", "<script>", "`tool"):
            if forbidden in rendered:
                raise SystemExit(f"learning failure reminder leaked hostile tool content {forbidden!r}: {rendered}")
        if "&lt;local-path&gt;" not in rendered or "&lt;script&gt;" not in rendered:
            raise SystemExit(f"learning failure reminder omitted safe display markers: {rendered}")
        for forbidden in ("\x00", "\u001b", "\u202e"):
            if forbidden in rendered:
                raise SystemExit(f"learning failure reminder retained active/control content {forbidden!r}: {rendered}")
        if re.search(r"(?<!\\)\[[^\]]+\]\(", rendered):
            raise SystemExit(f"learning failure reminder retained active Markdown link syntax: {rendered}")
        if str(task["priority"]).lower() != "high":
            raise SystemExit(f"failed-run learning reminder lost high priority: {dict(task)}")
        assert_queue_learning_tasks_handoff(result.metadata, "queue_learning_tasks_unrelated_after_action")

    with TemporaryDirectory(prefix="jarvis-learning-queue-matching-proof-") as temp:
        runtime = make_temp_runtime(Path(temp))
        failed_run_id = runtime.store.log_tool_run(
            "queue-target",
            "failing_tool",
            "LOCAL_SAFE",
            False,
            False,
            "failed",
        )
        runtime.store.log_tool_run(
            "queue-target",
            "after_action_learning_packet",
            "READ_ONLY",
            True,
            False,
            "reviewed target run",
            metadata={"run_id": failed_run_id},
        )
        queue_learning_tasks = make_memory_curator_tools(runtime.store, runtime.vault)[10]
        result = queue_learning_tasks({})
        task_bodies = [str(row["body"]) for row in runtime.store.list_tasks(status="open", limit=None)]
        if any(body.startswith("Review after-action learning for failed run") for body in task_bodies):
            raise SystemExit(f"matching after-action proof did not suppress the target reminder: {task_bodies}")
        if f"failed run #{failed_run_id}" in result.output:
            raise SystemExit(f"matching after-action proof left failed-run output: {result.output}")
        assert_queue_learning_tasks_handoff(result.metadata, "queue_learning_tasks_matching_after_action")

    with TemporaryDirectory(prefix="jarvis-learning-queue-pretarget-proof-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "queue-target",
            "after_action_learning_packet",
            "READ_ONLY",
            True,
            False,
            "claims a future target",
            metadata={"run_id": 2},
        )
        failed_run_id = runtime.store.log_tool_run(
            "queue-target",
            "future_target_tool",
            "LOCAL_SAFE",
            False,
            False,
            "failed",
        )
        if failed_run_id != 2:
            raise SystemExit(f"pre-target packet fixture expected run #2, got #{failed_run_id}")
        queue_learning_tasks = make_memory_curator_tools(runtime.store, runtime.vault)[10]
        result = queue_learning_tasks({})
        if f"failed run #{failed_run_id}" not in result.output:
            raise SystemExit(f"chronologically impossible after-action packet suppressed target: {result}")
        assert_queue_learning_tasks_handoff(result.metadata, "queue_learning_tasks_pretarget_after_action")

    with TemporaryDirectory(prefix="jarvis-learning-queue-held-display-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.store.log_tool_run(
            "queue-held",
            hostile_tool,
            "HIGH_RISK",
            False,
            False,
            "approval required",
            metadata={"requires_confirmation": True},
        )
        queue_learning_tasks = make_memory_curator_tools(runtime.store, runtime.vault)[10]
        result = queue_learning_tasks({})
        task = runtime.store.list_tasks(status="open", limit=10)[0]
        rendered = f"{result.output}\n{task['body']}"
        for forbidden in (raw_path, "\n## injected", "<script>", "`tool"):
            if forbidden in rendered:
                raise SystemExit(f"approval-held reminder leaked hostile tool content {forbidden!r}: {rendered}")
        if "Review approval-held run" not in rendered or "&lt;local-path&gt;" not in rendered:
            raise SystemExit(f"approval-held reminder lost safe actionable context: {rendered}")
        assert_queue_learning_tasks_handoff(result.metadata, "queue_learning_tasks_held_display")


def assert_review_only_metadata(
    metadata: dict, label: str, *, reads_private_data: bool = False
) -> None:
    forbidden_true = [
        "calls_model",
        "calls_external_service",
        "executes_tools",
        "executes_side_effect",
        "external_side_effect",
        "writes_files",
        "writes_database",
        "writes_memory",
        "writes_notes",
        "queues_approval",
        "requires_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "controls_computer",
        "speaks",
        "completes_tasks",
    ]
    violations = [key for key in forbidden_true if metadata.get(key)]
    if violations:
        raise SystemExit(f"{label} should be review-only but marked {violations}: {metadata}")
    if reads_private_data:
        if metadata.get("reads_personal_data") is not True:
            raise SystemExit(f"{label} has wrong personal-read truth: {metadata}")
        if metadata.get("reads_private_data") is not True:
            raise SystemExit(f"{label} has wrong private-read truth: {metadata}")
    elif metadata.get("reads_personal_data") or metadata.get("reads_private_data"):
        raise SystemExit(f"{label} unexpectedly marked private reads: {metadata}")


def assert_operator_metadata(metadata: dict, label: str) -> None:
    if metadata.get("operator_timeboxes_override_priority") is not True:
        raise SystemExit(f"{label} missed operator timebox metadata: {metadata}")
    if metadata.get("stop_times_override_priority") is not True:
        raise SystemExit(f"{label} missed stop-time metadata: {metadata}")
    assert_no_future_authority(metadata, label)


def assert_no_future_authority(metadata: dict, label: str) -> None:
    for key in ("authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should keep {key}=False: {metadata}")


def assert_operator_output(output: str, label: str) -> None:
    if "explicit stop times" not in output or "pause commands" not in output:
        raise SystemExit(f"{label} missed operator-limit output.")


def assert_vault_relative_receipt(result, root: Path, prefix: str, label: str) -> None:
    metadata = result.tool_results[0].metadata if hasattr(result, "tool_results") else result.metadata
    response = result.response if hasattr(result, "response") else result.output
    path_text = str(metadata.get("path") or "")
    path_display = metadata.get("path_display")
    receipt_line = response.split("\n", 1)[0]
    if not path_text or not Path(path_text).exists():
        raise SystemExit(f"{label} should preserve exact saved path metadata: {metadata}")
    if path_text in receipt_line:
        raise SystemExit(f"{label} should not print the raw local note path.")
    if (
        str(root) in receipt_line
        or "/private/" in receipt_line
        or "/\x55sers/" in receipt_line
        or "/var/folders/" in receipt_line
        or "/tmp/" in receipt_line
    ):
        raise SystemExit(f"{label} receipt should not expose local temp or user paths.")
    if not isinstance(path_display, str) or not path_display.startswith(prefix):
        raise SystemExit(f"{label} missed vault-relative display metadata: {metadata}")
    if path_display not in receipt_line:
        raise SystemExit(f"{label} should print the vault-relative saved-note label.")


def assert_no_local_paths(value, label: str) -> None:
    text = repr(value)
    if "/private/" in text or "/\x55sers/" in text or "/var/folders/" in text or "/tmp/" in text:
        raise SystemExit(f"{label} handoff should not expose raw local paths: {text[:1000]}")


def assert_memory_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    for key, expected in (
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if metadata.get(key) != expected or handoff.get(key) != expected:
            raise SystemExit(f"{label} missed memory contract {key}={expected}: metadata={metadata} handoff={handoff}")


def assert_learning_review_handoff(metadata: dict, label: str, *, writes: bool) -> None:
    handoff = metadata.get("learning_review_handoff")
    if not metadata.get("learning_review_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready learning_review_handoff: {metadata}")
    expected_source = "save_learning_review" if writes else "learning_review"
    if handoff.get("source") != expected_source or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong learning handoff source/readiness: {handoff}")
    assert_memory_contract(
        metadata,
        handoff,
        label,
        state_changed=writes,
        changed=["learning_review_export"] if writes else [],
        content_in_handoff=False,
    )
    for key in (
        "limit",
        "feedback",
        "weak_memories",
        "duplicate_groups",
        "preferences",
        "skills",
        "knowledge_promotion_candidates",
        "knowledge_promotion_selected_rows",
        "knowledge_promotion_candidate_total",
        "knowledge_promotion_rendered_candidates",
        "knowledge_promotion_hidden_count",
        "knowledge_promotion_output_hidden_count",
        "knowledge_promotion_state",
        "knowledge_promotion_unreadable_rows",
        "knowledge_promotion_classification_basis",
        "knowledge_promotion_semantic_proof",
        "knowledge_promotion_authorizes_promotion",
        "knowledge_promotion_rotation_seed",
        "recent_tool_runs",
        "recent_action_runs",
        "failed_or_blocked_action_runs",
        "approval_held_action_runs",
        "execution_learning_state",
        "execution_learning_blocks_completion_claim",
        "execution_learning_target_run_id",
        "execution_learning_missing_count",
    ):
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {handoff} vs {metadata}")
    if handoff.get("execution_learning_missing") != (metadata.get("execution_learning_missing") or []):
        raise SystemExit(f"{label} handoff missing list diverged: {handoff} vs {metadata}")
    if handoff.get("knowledge_promotion_candidate_ids") != (
        metadata.get("knowledge_promotion_candidate_ids") or []
    ):
        raise SystemExit(f"{label} handoff promotion ids diverged: {handoff} vs {metadata}")
    if handoff.get("knowledge_promotion_candidate_bindings") != (
        metadata.get("knowledge_promotion_candidate_bindings") or []
    ):
        raise SystemExit(f"{label} handoff promotion bindings diverged: {handoff} vs {metadata}")
    if handoff.get("knowledge_promotion_target_counts") != (
        metadata.get("knowledge_promotion_target_counts") or {}
    ):
        raise SystemExit(f"{label} handoff promotion counts diverged: {handoff} vs {metadata}")
    if handoff.get("knowledge_promotion_review_commands") != (
        metadata.get("knowledge_promotion_review_commands") or []
    ):
        raise SystemExit(f"{label} handoff promotion commands diverged: {handoff} vs {metadata}")
    if handoff.get("execution_learning_approval_review_commands") != (metadata.get("execution_learning_approval_review_commands") or []):
        raise SystemExit(f"{label} handoff approval review commands diverged: {handoff} vs {metadata}")
    proof_queue = metadata.get("execution_learning_proof_queue") or []
    if handoff.get("execution_learning_proof_queue") != proof_queue:
        raise SystemExit(f"{label} handoff proof queue diverged: {handoff} vs {metadata}")
    if handoff.get("execution_learning_proof_queue_count") != len(proof_queue):
        raise SystemExit(f"{label} handoff proof queue count diverged: {handoff} vs {metadata}")
    if handoff.get("execution_learning_next_proof_command") != (proof_queue[0] if proof_queue else ""):
        raise SystemExit(f"{label} handoff next proof command diverged: {handoff} vs {metadata}")
    if handoff.get("execution_learning_next_required_command") != metadata.get("execution_learning_next_required_command"):
        raise SystemExit(f"{label} handoff next required command diverged: {handoff} vs {metadata}")
    if handoff.get("execution_learning_next_required_command") != handoff.get("execution_learning_next_proof_command"):
        raise SystemExit(f"{label} handoff required/proof command aliases diverged: {handoff}")
    if handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} handoff should not include report content: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    if boundaries.get("reads_personal_data") is not True or boundaries.get("reads_private_data") is not True:
        raise SystemExit(f"{label} handoff hid private learning-review reads: {handoff}")
    if writes and handoff.get("path_display") != metadata.get("path_display"):
        raise SystemExit(f"{label} saved handoff path display diverged: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": not writes,
        "writes_files": writes,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": writes,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} handoff boundary {key} should be {value}: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_queue_learning_tasks_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("queue_learning_tasks_handoff")
    if not metadata.get("queue_learning_tasks_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready queue_learning_tasks_handoff: {metadata}")
    if handoff.get("source") != "queue_learning_tasks" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong queue handoff source/readiness: {handoff}")
    assert_memory_contract(
        metadata,
        handoff,
        label,
        state_changed=True,
        changed=["learning_tasks"] if metadata.get("added") else ["task_export"],
        content_in_handoff=bool(metadata.get("added") or metadata.get("skipped")),
    )
    for key in ("added", "skipped", "limit", "path_display"):
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {handoff} vs {metadata}")
    added_rows = handoff.get("added_tasks") or []
    if len(added_rows) != metadata.get("added"):
        raise SystemExit(f"{label} added task row count diverged: {handoff} vs {metadata}")
    if handoff.get("added_task_ids") != (metadata.get("added_task_ids") or []):
        raise SystemExit(f"{label} added task ids diverged: {handoff} vs {metadata}")
    if len(handoff.get("skipped_task_previews") or []) != min(metadata.get("skipped", 0), 20):
        raise SystemExit(f"{label} skipped preview count diverged: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": False,
        "writes_files": True,
        "writes_database": bool(metadata.get("added")),
        "writes_memory": bool(metadata.get("added")),
        "writes_notes": True,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} handoff boundary {key} should be {value}: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_memory_stats_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("memory_stats_handoff")
    if not metadata.get("memory_stats_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready memory_stats_handoff: {metadata}")
    if handoff.get("source") != "memory_stats" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong stats handoff source/readiness: {handoff}")
    assert_memory_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=bool(handoff.get("stats")))
    if handoff.get("count") != metadata.get("count"):
        raise SystemExit(f"{label} stats handoff count diverged: {handoff} vs {metadata}")
    if len(handoff.get("stats") or []) != metadata.get("count"):
        raise SystemExit(f"{label} stats rows should match metadata count: {handoff} vs {metadata}")
    if handoff.get("total_memories", 0) < metadata.get("count", 0):
        raise SystemExit(f"{label} total memory count looks too small: {handoff}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": True,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} handoff boundary {key} should be {value}: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_memory_tree_summary_handoff(metadata: dict, label: str, *, writes: bool) -> None:
    handoff = metadata.get("memory_tree_summary_handoff")
    if not metadata.get("memory_tree_summary_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready memory_tree_summary_handoff: {metadata}")
    if handoff.get("source") != "memory_tree_summary" or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong tree handoff source/readiness: {handoff}")
    assert_memory_contract(
        metadata,
        handoff,
        label,
        state_changed=writes,
        changed=["memory_tree_summary_export"] if writes else [],
        content_in_handoff=False,
    )
    for key in ("count", "limit", "draft_skill_count", "human_review_required_for_skill_promotion", "self_modifying_code"):
        if handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} tree handoff {key} diverged: {handoff} vs {metadata}")
    if writes and handoff.get("path_display") != metadata.get("path_display"):
        raise SystemExit(f"{label} tree path display diverged: {handoff} vs {metadata}")
    if handoff.get("content_in_handoff"):
        raise SystemExit(f"{label} tree handoff should not include snapshot content: {handoff}")
    if handoff.get("category_count") != len(handoff.get("categories") or []):
        raise SystemExit(f"{label} tree category count diverged: {handoff}")
    if sum(row.get("count", 0) for row in handoff.get("categories") or []) != metadata.get("count"):
        raise SystemExit(f"{label} tree category rows should sum to memory count: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": not writes,
        "writes_files": writes,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": writes,
        "queues_approval": False,
        "requires_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} handoff boundary {key} should be {value}: {boundaries}")
    assert_no_local_paths(handoff, label)


def assert_memory_refusal_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    mutation: str,
    read_only: bool = False,
    requires_approval: bool = True,
) -> None:
    handoff = metadata.get("memory_refusal_handoff")
    if not metadata.get("memory_refusal_handoff_ready") or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready memory_refusal_handoff: {metadata}")
    if handoff.get("source") != source or not handoff.get("ready_for_operator"):
        raise SystemExit(f"{label} has wrong memory refusal source/readiness: {handoff}")
    if handoff.get("mutation") != mutation or handoff.get("refused") is not True:
        raise SystemExit(f"{label} has wrong memory refusal mutation/refused state: {handoff}")
    assert_memory_contract(
        metadata,
        handoff,
        label,
        state_changed=False,
        changed=[],
        content_in_handoff=bool(
            handoff.get("raw_memory_id")
            or handoff.get("raw_keep_id")
            or handoff.get("raw_delete_id")
            or handoff.get("raw_confidence")
        ),
    )
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} refusal handoff should not report changed rows: {handoff}")
    if not isinstance(handoff.get("next_commands"), dict) or not handoff["next_commands"].get("retry"):
        raise SystemExit(f"{label} refusal handoff should include retry commands: {handoff}")
    for key in ("memory_id", "keep_id", "delete_id", "raw_memory_id", "raw_keep_id", "raw_delete_id", "raw_confidence"):
        if key in handoff and key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} refusal handoff {key} diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": read_only,
        "writes_files": False,
        "writes_database": False,
        "writes_memory": False,
        "writes_notes": False,
        "queues_approval": False,
        "requires_approval": requires_approval,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "edits_memory": False,
        "deletes_memory": False,
        "merges_memory": False,
        "operator_timeboxes_override_priority": True,
        "stop_times_override_priority": True,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} refusal boundary {key} should be {value}: {boundaries}")
    assert_no_future_authority(metadata, label)
    assert_no_local_paths(handoff, label)


def test_atomic_memory_merge() -> None:
    with TemporaryDirectory(prefix="jarvis-atomic-memory-merge-") as temp:
        runtime = make_temp_runtime(Path(temp))
        store = runtime.store
        merge_memory_handler = make_memory_curator_tools(store, runtime.vault)[5]

        def merge_memories(args):
            keep_id = args.get("keep_id")
            delete_id = args.get("delete_id")
            if type(keep_id) is not int or type(delete_id) is not int or keep_id == delete_id:
                return merge_memory_handler(args)
            targets = store.resolve_memory_merge_approval_targets(keep_id, delete_id)
            if targets is None:
                return merge_memory_handler(args)
            keep, delete = targets
            return merge_memory_handler(
                {
                    "keep_id": keep_id,
                    "keep_revision": keep.revision,
                    "keep_binding": keep.binding,
                    "delete_id": delete_id,
                    "delete_revision": delete.revision,
                    "delete_binding": delete.binding,
                }
            )

        keep_id = store.add_memory(
            MemoryRecord("learning", "Keep title", "Keep body", confidence=0.35)
        )
        delete_id = store.add_memory(
            MemoryRecord("other", "Delete title", "Delete body", confidence=0.85)
        )
        merged = merge_memories({"keep_id": keep_id, "delete_id": delete_id})
        kept = store.get_memory(keep_id)
        if not merged.ok or kept is None or store.get_memory(delete_id) is not None:
            raise SystemExit(f"atomic memory merge did not commit both changes: {merged}")
        expected_body = f"Keep body\n\nMerged from #{delete_id}: Delete body"
        if (
            kept["category"] != "learning"
            or kept["title"] != "Keep title"
            or kept["body"] != expected_body
            or float(kept["confidence"]) != 0.85
        ):
            raise SystemExit(f"atomic memory merge changed legacy merge behavior: {dict(kept)}")

        contained_keep_id = store.add_memory(
            MemoryRecord("learning", "Contained", "Already includes duplicate body", confidence=0.9)
        )
        contained_delete_id = store.add_memory(
            MemoryRecord("other", "Duplicate", "duplicate body", confidence=0.2)
        )
        if not store.merge_memories(contained_keep_id, contained_delete_id):
            raise SystemExit("atomic memory merge rejected two existing contained-body rows")
        contained = store.get_memory(contained_keep_id)
        if contained is None or contained["body"] != "Already includes duplicate body" or float(contained["confidence"]) != 0.9:
            raise SystemExit(f"atomic memory merge lost contained-body/confidence parity: {contained}")

        missing_keep_id = store.add_memory(
            MemoryRecord("learning", "Missing peer", "Must stay unchanged", confidence=0.4)
        )
        before_missing = dict(store.get_memory(missing_keep_id) or {})
        missing = merge_memories({"keep_id": missing_keep_id, "delete_id": 999999})
        after_missing = dict(store.get_memory(missing_keep_id) or {})
        if missing.ok or missing.metadata.get("reason") != "missing_memory" or after_missing != before_missing:
            raise SystemExit(f"missing atomic merge target was not a definite no-op: {missing}")

        rollback_keep_id = store.add_memory(
            MemoryRecord("learning", "Rollback keep", "Original keep", confidence=0.3)
        )
        rollback_delete_id = store.add_memory(
            MemoryRecord("other", "Rollback delete", "Must remain", confidence=0.7)
        )
        with store.connect() as conn:
            conn.execute(
                f"""
                CREATE TRIGGER fail_atomic_memory_merge_delete
                BEFORE DELETE ON memories
                WHEN OLD.id = {rollback_delete_id}
                BEGIN
                    SELECT RAISE(ABORT, 'private injected delete failure');
                END
                """
            )
        failed = merge_memories({"keep_id": rollback_keep_id, "delete_id": rollback_delete_id})
        rollback_keep = store.get_memory(rollback_keep_id)
        rollback_delete = store.get_memory(rollback_delete_id)
        if (
            failed.ok
            or failed.metadata.get("reason") != "merge_failed"
            or rollback_keep is None
            or rollback_keep["body"] != "Original keep"
            or float(rollback_keep["confidence"]) != 0.3
            or rollback_delete is None
        ):
            raise SystemExit(f"delete failure did not roll back the keep update: {failed}")
        if "private injected" in failed.output or len(failed.output) > 1200:
            raise SystemExit(f"merge failure leaked or failed to bound database details: {failed.output}")
        for key in ("writes_database", "writes_memory", "authorizes_retry", "authorizes_memory_mutation"):
            if failed.metadata.get(key):
                raise SystemExit(f"failed atomic merge claimed {key}: {failed.metadata}")
        assert_operator_metadata(failed.metadata, "failed_atomic_memory_merge")
        assert_memory_refusal_handoff(
            failed.metadata,
            "failed_atomic_memory_merge",
            source="merge_memories",
            mutation="memory_merge",
        )
        assert_operator_output(failed.output, "failed_atomic_memory_merge")
        with store.connect() as conn:
            conn.execute("DROP TRIGGER fail_atomic_memory_merge_delete")

        def run_contended_merge(*, delete_target: bool) -> tuple[object, int, int]:
            concurrent_keep_id = store.add_memory(
                MemoryRecord("learning", "Concurrent keep", "Concurrent original", confidence=0.25)
            )
            concurrent_delete_id = store.add_memory(
                MemoryRecord("other", "Concurrent delete", "Before mutation", confidence=0.6)
            )
            writer_ready = threading.Event()
            release_writer = threading.Event()
            merge_started = threading.Event()
            results: list[object] = []
            errors: list[BaseException] = []

            def writer() -> None:
                try:
                    with store.connect() as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        if delete_target:
                            conn.execute("DELETE FROM memories WHERE id = ?", (concurrent_delete_id,))
                        else:
                            conn.execute(
                                "UPDATE memories SET body = ?, confidence = ? WHERE id = ?",
                                ("Concurrent committed mutation", 0.95, concurrent_delete_id),
                            )
                        writer_ready.set()
                        if not release_writer.wait(5):
                            raise RuntimeError("timed out releasing concurrent memory writer")
                except BaseException as exc:
                    errors.append(exc)
                    writer_ready.set()

            def merger() -> None:
                try:
                    merge_started.set()
                    results.append(
                        merge_memories(
                            {"keep_id": concurrent_keep_id, "delete_id": concurrent_delete_id}
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            writer_thread = threading.Thread(target=writer)
            merge_thread = threading.Thread(target=merger)
            writer_thread.start()
            if not writer_ready.wait(5):
                raise SystemExit("concurrent memory writer did not acquire its transaction")
            merge_thread.start()
            if not merge_started.wait(5):
                raise SystemExit("concurrent memory merge did not start")
            release_writer.set()
            writer_thread.join(10)
            merge_thread.join(10)
            if writer_thread.is_alive() or merge_thread.is_alive() or errors or len(results) != 1:
                raise SystemExit(f"concurrent atomic memory merge did not settle: {errors!r}")
            return results[0], concurrent_keep_id, concurrent_delete_id

        mutation_result, mutation_keep_id, mutation_delete_id = run_contended_merge(delete_target=False)
        mutation_keep = store.get_memory(mutation_keep_id)
        if (
            not getattr(mutation_result, "ok", False)
            or mutation_keep is None
            or "Concurrent committed mutation" not in mutation_keep["body"]
            or float(mutation_keep["confidence"]) != 0.95
            or store.get_memory(mutation_delete_id) is not None
        ):
            raise SystemExit(f"atomic merge did not serialize after target mutation: {mutation_result}")

        deletion_result, deletion_keep_id, deletion_delete_id = run_contended_merge(delete_target=True)
        deletion_keep = store.get_memory(deletion_keep_id)
        if (
            getattr(deletion_result, "ok", True)
            or getattr(deletion_result, "metadata", {}).get("reason") != "missing_memory"
            or deletion_keep is None
            or deletion_keep["body"] != "Concurrent original"
            or store.get_memory(deletion_delete_id) is not None
        ):
            raise SystemExit(f"atomic merge partially succeeded after target deletion: {deletion_result}")


def test_learning_review_surfaces_only_unowned_structured_candidates() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-promotion-review-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        store = runtime.store
        inbox_goal_id = store.add_memory(
            MemoryRecord(
                "goals",
                "Inbox-owned goal",
                "This source-only Inbox memory is already under ingestion custody.",
                "obsidian-inbox",
            )
        )
        candidate_ids = {
            store.add_memory(MemoryRecord("identity", "Personal operating principle", "Decide from evidence.", "manual")),
            store.add_memory(MemoryRecord("preferences", "Writing style", "Prefer concise openings.", "manual")),
            store.add_memory(MemoryRecord("decisions", "Keep approval gates", "Risky actions require review.", "manual")),
            store.add_memory(MemoryRecord("relationships", "Fixture", "Synthetic relationship note.", "manual")),
            store.add_memory(
                MemoryRecord(
                    "goals",
                    "/\x55sers/private-goal <script>finish Jarvis</script>",
                    "Finish Jarvis safely.",
                    "manual",
                )
            ),
        }
        mislabeled_id = store.add_memory(
            MemoryRecord(
                "preferences",
                "Medical appointment happened Tuesday",
                "This label is intentionally not semantic proof of a preference.",
                "manual",
            )
        )
        candidate_ids.add(mislabeled_id)
        facts_id = store.add_memory(
            MemoryRecord("facts", "Ordinary fact", "Keep this as a generic fact.", "manual")
        )
        source_excluded_ids = {
            inbox_goal_id,
            store.add_memory(MemoryRecord("profile", "Owned profile", "Already structured.", "profile")),
            store.add_memory(MemoryRecord("preferences", "Owned preference", "Already structured.", "preferences")),
            store.add_memory(MemoryRecord("people", "Owned person", "Already structured.", "people-log")),
            store.add_memory(MemoryRecord("decisions", "Owned decision", "Already structured.", "decision-log")),
            store.add_memory(MemoryRecord("relationships", "Owned interaction", "Already structured.", "interaction-log")),
        }
        linked_memory_id = store.add_memory(
            MemoryRecord("decisions", "Linked decision", "Already linked.", "manual")
        )
        preference_linked_memory_id = store.add_memory(
            MemoryRecord("preferences", "Linked preference", "Already linked.", "manual")
        )
        person_linked_memory_id = store.add_memory(
            MemoryRecord("relationships", "Linked person", "Already linked.", "manual")
        )
        interaction_linked_memory_id = store.add_memory(
            MemoryRecord("relationships", "Linked interaction", "Already linked.", "manual")
        )
        organized_memory_id = store.add_memory(
            MemoryRecord(
                "decisions", "Organizer-owned decision", "Already organized.", "brain-dump"
            )
        )
        organizer_plain_memory_id = store.add_memory(
            MemoryRecord("profile", "Organizer-owned memory", "Already organized.", "brain-dump")
        )
        malformed_revision_id = store.add_memory(
            MemoryRecord("identity", "Malformed revision", "Must not be rounded.", "manual")
        )
        created_at = "2026-07-13T00:00:00+00:00"
        with store.connect() as conn:
            decision = conn.execute(
                "INSERT INTO decisions(title, rationale, impact, status, revision, created_at, updated_at) "
                "VALUES ('Linked decision', '', '', 'active', 1, ?, ?)",
                (created_at, created_at),
            )
            conn.execute(
                "INSERT INTO decision_memory_links(decision_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (int(decision.lastrowid), linked_memory_id, created_at, created_at),
            )
            preference = conn.execute(
                "INSERT INTO preferences(category, key, value, status, revision, created_at, updated_at) "
                "VALUES ('communication', 'linked', 'yes', 'active', 1, ?, ?)",
                (created_at, created_at),
            )
            conn.execute(
                "INSERT INTO preference_memory_links(preference_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (int(preference.lastrowid), preference_linked_memory_id, created_at, created_at),
            )
            person = conn.execute(
                "INSERT INTO people(name, relation, notes, revision, created_at, updated_at) "
                "VALUES ('Linked Person', 'test', '', 1, ?, ?)",
                (created_at, created_at),
            )
            person_id = int(person.lastrowid)
            conn.execute(
                "INSERT INTO person_memory_links(memory_id, person_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (person_linked_memory_id, person_id, created_at, created_at),
            )
            interaction = conn.execute(
                "INSERT INTO person_interactions(person_id, summary, happened_at, created_at) "
                "VALUES (?, 'linked', ?, ?)",
                (person_id, created_at, created_at),
            )
            conn.execute(
                "INSERT INTO interaction_memory_links(interaction_id, memory_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    int(interaction.lastrowid),
                    interaction_linked_memory_id,
                    created_at,
                    created_at,
                ),
            )
            organized_decision = conn.execute(
                "INSERT INTO decisions(title, rationale, impact, status, revision, created_at, updated_at) "
                "VALUES ('Organizer-owned decision', '', '', 'active', 1, ?, ?)",
                (created_at, created_at),
            )
            source_key = "organize-note:v1:" + ("a" * 64)
            conn.execute(
                "INSERT INTO organized_note_batches(source_key, state, created_at, updated_at) "
                "VALUES (?, 'pending', ?, ?)",
                (source_key, created_at, created_at),
            )
            conn.execute(
                "INSERT INTO organized_note_entries("
                "source_key, entry_index, kind, decision_id, decision_memory_id"
                ") VALUES (?, 0, 'decision', ?, ?)",
                (source_key, int(organized_decision.lastrowid), organized_memory_id),
            )
            conn.execute(
                "INSERT INTO organized_note_entries(source_key, entry_index, kind, memory_id) "
                "VALUES (?, 1, 'memory', ?)",
                (source_key, organizer_plain_memory_id),
            )
            conn.execute("PRAGMA ignore_check_constraints = ON")
            conn.execute(
                "UPDATE memories SET revision = 1.75 WHERE id = ?",
                (malformed_revision_id,),
            )

        count_tables = (
            "memories",
            "decisions",
            "decision_memory_links",
            "preferences",
            "preference_memory_links",
            "people",
            "person_memory_links",
            "interaction_memory_links",
            "organized_note_batches",
            "organized_note_entries",
            "tasks",
            "tool_runs",
        )

        def row_counts() -> dict[str, int]:
            with store.connect() as conn:
                return {
                    table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in count_tables
                }

        def database_dump() -> tuple[str, ...]:
            with store.connect() as conn:
                return tuple(conn.iterdump())

        def application_table_snapshot(*, exclude: set[str] | None = None) -> tuple:
            excluded = exclude or set()
            with store.connect() as conn:
                tables = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                    if str(row[0]) not in excluded
                ]
                return tuple(
                    (
                        table,
                        tuple(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"')),
                    )
                    for table in tables
                )

        def vault_bytes() -> dict[str, bytes]:
            return {
                str(path.relative_to(runtime.vault.root_path)): path.read_bytes()
                for path in runtime.vault.root_path.rglob("*")
                if path.is_file()
            }

        before_counts = row_counts()
        before_database = database_dump()
        before_vault = vault_bytes()
        curator_tools = make_memory_curator_tools(store, runtime.vault)
        learning_review = curator_tools[8]
        save_learning_review = curator_tools[9]
        queue_learning_tasks = curator_tools[10]
        unexpected_effect = RuntimeError("learning review attempted a forbidden effect")
        with (
            patch.object(ChatBrain, "respond", side_effect=unexpected_effect),
            patch("subprocess.run", side_effect=unexpected_effect),
            patch("subprocess.Popen", side_effect=unexpected_effect),
            patch("urllib.request.urlopen", side_effect=unexpected_effect),
        ):
            result = learning_review({"limit": 50})
        after_counts = row_counts()
        after_database = database_dump()
        after_vault = vault_bytes()
        metadata = result.metadata
        reported_ids = set(metadata.get("knowledge_promotion_candidate_ids") or [])
        if reported_ids != candidate_ids:
            raise SystemExit(
                f"learning review selected the wrong structured candidates: {reported_ids} vs {candidate_ids}"
            )
        if (
            facts_id in reported_ids
            or source_excluded_ids & reported_ids
            or linked_memory_id in reported_ids
            or preference_linked_memory_id in reported_ids
            or person_linked_memory_id in reported_ids
            or interaction_linked_memory_id in reported_ids
            or organized_memory_id in reported_ids
            or organizer_plain_memory_id in reported_ids
            or malformed_revision_id in reported_ids
        ):
            raise SystemExit(f"learning review included owned or unstructured memory: {metadata}")
        if metadata.get("knowledge_promotion_target_counts") != {
            "decision": 1,
            "goal": 1,
            "person": 1,
            "preference": 2,
            "profile": 1,
        }:
            raise SystemExit(f"learning review target counts are wrong: {metadata}")
        expected_commands = [
            f"knowledge promotion packet {memory_id}"
            for memory_id in metadata["knowledge_promotion_candidate_ids"]
        ]
        if metadata.get("knowledge_promotion_review_commands") != expected_commands:
            raise SystemExit(f"learning review promotion commands are not ID-only: {metadata}")
        bindings = metadata.get("knowledge_promotion_candidate_bindings") or []
        if (
            metadata.get("knowledge_promotion_classification_basis") != "category_label_only"
            or metadata.get("knowledge_promotion_semantic_proof") is not False
            or metadata.get("knowledge_promotion_authorizes_promotion") is not False
            or any(
                binding.get("classification_basis") != "category_label_only"
                or binding.get("semantic_proof") is not False
                or binding.get("authorizes_promotion") is not False
                for binding in bindings
            )
        ):
            raise SystemExit(f"label-only candidates looked promotion-authoritative: {metadata}")
        category_targets = {
            "identity": "profile",
            "preferences": "preference",
            "decisions": "decision",
            "relationships": "person",
            "goals": "goal",
        }
        if {
            (row.get("memory_id"), row.get("memory_revision"), row.get("target_kind"))
            for row in bindings
        } != {
            (
                memory_id,
                1,
                category_targets[str(store.get_memory(memory_id)["category"])],
            )
            for memory_id in candidate_ids
        }:
            raise SystemExit(f"learning review candidate bindings are incomplete: {metadata}")
        for memory_id in candidate_ids:
            if (
                f"memory #{memory_id}" not in result.output
                or f"knowledge promotion packet {memory_id}" not in result.output
            ):
                raise SystemExit(f"learning review omitted candidate #{memory_id}: {result.output}")
            command = f"knowledge promotion packet {memory_id}"
            plan = RuleBasedPlanner().plan(command)
            if (
                len(plan.actions) != 1
                or plan.actions[0].tool_name != "knowledge_promotion_packet"
                or plan.actions[0].args != {"memory_id": memory_id}
            ):
                raise SystemExit(f"learning review emitted an unroutable packet command: {command}")
            packet = make_knowledge_promotion_tools(store, runtime.vault)[0]
            packet_result = packet({"memory_id": memory_id})
            if not packet_result.ok:
                raise SystemExit(
                    f"learning review emitted an unusable packet command: {command}: {packet_result}"
                )
        if "/\x55sers/private-goal" in result.output or "<script>" in result.output:
            raise SystemExit(f"learning review leaked hostile candidate display content: {result.output}")
        if "&lt;local-path&gt;" not in result.output:
            raise SystemExit(f"learning review lost sanitized candidate context: {result.output}")
        metadata_text = repr(metadata)
        for private_marker in (
            "Personal operating principle",
            "Writing style",
            "Medical appointment",
            "Fixture",
            "private-goal",
        ):
            if private_marker in metadata_text:
                raise SystemExit(f"learning review metadata leaked candidate content: {metadata}")
        assert_review_only_metadata(
            metadata, "knowledge_promotion_learning_review", reads_private_data=True
        )
        assert_learning_review_handoff(
            metadata, "knowledge_promotion_learning_review", writes=False
        )
        if (
            before_counts != after_counts
            or before_database != after_database
            or before_vault != after_vault
        ):
            raise SystemExit(
                f"knowledge promotion review mutated durable state: {before_counts} -> {after_counts}"
            )

        for raw_path in (
            "/home/example/private.txt",
            "/srv/example/private.db",
            "path=/\x55sers/example/private.txt",
            "file:///\x55sers/example/private.txt",
            "~/private.txt",
            "~example/private.txt",
            r"C:\\Users\\example\\private.txt",
            r"\\server\\share\\private.txt",
            "//server/share/private.txt",
            "/\x55sers/example/Secret (Acme)/plan.md",
            "/\x55sers/example/private;payroll.xlsx",
        ):
            rendered = _learning_review_display(raw_path, limit=160)
            if (
                raw_path in rendered
                or "&lt;local-path&gt;" not in rendered
                or "Acme" in rendered
                or "payroll" in rendered
            ):
                raise SystemExit(f"learning review failed to redact path form {raw_path!r}: {rendered!r}")

        before_save_database = database_dump()
        before_save_vault = vault_bytes()
        saved_review = save_learning_review({"limit": 50})
        after_save_database = database_dump()
        after_save_vault = vault_bytes()
        changed_save_paths = {
            path
            for path in set(before_save_vault) | set(after_save_vault)
            if before_save_vault.get(path) != after_save_vault.get(path)
        }
        if (
            before_save_database != after_save_database
            or changed_save_paths
            != {
                saved_review.metadata.get("path_display"),
                "Automations/.Learning Review.md.lock",
            }
            or saved_review.metadata.get("path_display") != "Automations/Learning Review.md"
        ):
            raise SystemExit(
                f"save learning review exceeded its one-report effect: {changed_save_paths}"
            )

        non_task_state_before_queue = application_table_snapshot(exclude={"tasks"})
        before_queue_vault = vault_bytes()
        queue_result = queue_learning_tasks({"limit": 50})
        after_queue_vault = vault_bytes()
        changed_queue_paths = {
            path
            for path in set(before_queue_vault) | set(after_queue_vault)
            if before_queue_vault.get(path) != after_queue_vault.get(path)
        }
        non_task_state_after_queue = application_table_snapshot(exclude={"tasks"})
        with store.connect() as conn:
            queued_rows = list(
                conn.execute("SELECT body, source FROM tasks ORDER BY id")
            )
        if (
            non_task_state_before_queue != non_task_state_after_queue
            or changed_queue_paths
            != {"Tasks/Open Tasks.md", "Tasks/.Open Tasks.md.lock"}
            or [(row["body"], row["source"]) for row in queued_rows]
            != [
                (
                    "Capture one concrete Jarvis feedback item after the next meaningful assistant interaction.",
                    "learning-review",
                )
            ]
        ):
            raise SystemExit(
                f"label-only candidates entered the task or structured-record path: {queue_result}"
            )

        for index in range(20):
            store.add_memory(
                MemoryRecord(
                    "preferences",
                    f"Newer preference candidate {index}",
                    "Advisory candidate only.",
                    "manual",
                )
            )
        fair_rows = store.list_knowledge_promotion_candidates(5, rotation_seed=0)
        if {row["target_kind"] for row in fair_rows} != {
            "profile", "preference", "decision", "person", "goal"
        }:
            raise SystemExit(f"one candidate category starved the bounded review: {fair_rows}")
        preference_row = next(row for row in fair_rows if row["target_kind"] == "preference")
        if int(preference_row["target_total"]) != 22:
            raise SystemExit(f"bounded review hid the full preference candidate count: {fair_rows}")
        rotated_target_kinds = {
            store.list_knowledge_promotion_candidates(1, rotation_seed=seed)[0]["target_kind"]
            for seed in range(5)
        }
        if rotated_target_kinds != {"profile", "preference", "decision", "person", "goal"}:
            raise SystemExit(
                f"daily rotation did not give every target eventual visibility: {rotated_target_kinds}"
            )
        preference_ids_by_rotation = {
            next(
                row["id"]
                for row in store.list_knowledge_promotion_candidates(5, rotation_seed=seed)
                if row["target_kind"] == "preference"
            )
            for seed in (0, 1)
        }
        if len(preference_ids_by_rotation) != 2:
            raise SystemExit(
                f"daily rotation starved older candidates within one target: {preference_ids_by_rotation}"
            )
        one_row = store.list_knowledge_promotion_candidates(1, rotation_seed=0)
        if len(one_row) != 1 or int(one_row[0]["candidate_total"]) != 26:
            raise SystemExit(f"limit-one review lost the global candidate total: {one_row}")
        limited_result = learning_review({"limit": 1})
        if (
            limited_result.metadata.get("knowledge_promotion_candidate_total") != 26
            or limited_result.metadata.get("knowledge_promotion_selected_rows") != 1
            or limited_result.metadata.get("knowledge_promotion_hidden_count") != 25
            or limited_result.metadata.get("knowledge_promotion_target_counts")
            != {
                "decision": 1,
                "goal": 1,
                "person": 1,
                "preference": 22,
                "profile": 1,
            }
        ):
            raise SystemExit(
                f"limit-one learning review reported false shown/hidden totals: {limited_result.metadata}"
            )

        original_candidates = store.list_knowledge_promotion_candidates
        store.list_knowledge_promotion_candidates = lambda _limit, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            sqlite3.OperationalError("no such table: private_schema_name")
        )
        try:
            unavailable = learning_review({"limit": 5})
        finally:
            store.list_knowledge_promotion_candidates = original_candidates  # type: ignore[method-assign]
        if (
            not unavailable.ok
            or unavailable.metadata.get("knowledge_promotion_state") != "unavailable"
            or unavailable.metadata.get("knowledge_promotion_candidates") != 0
            or unavailable.metadata.get("knowledge_promotion_totals_reliable") is not False
            or unavailable.metadata.get("knowledge_promotion_exception_type")
            != "OperationalError"
            or "private_schema_name" in repr(unavailable.metadata)
            or "private_schema_name" in unavailable.output
        ):
            raise SystemExit(f"learning review did not degrade safely on old schema: {unavailable}")

        malformed_totals = {
            "decision_total": 0,
            "goal_total": 0,
            "person_total": 0,
            "preference_total": 0,
            "profile_total": 1,
        }
        store.list_knowledge_promotion_candidates = lambda _limit, **_kwargs: [  # type: ignore[method-assign]
            {
                "id": 1,
                "revision": 1,
                "target_kind": "profile",
                "target_total": 1,
                "candidate_total": 1,
                **malformed_totals,
            },
            {
                "id": 2,
                "revision": 1.75,
                "target_kind": "profile",
                "target_total": 1,
                "candidate_total": 1,
                "category": "identity",
                "title": "Fractional revision",
                **malformed_totals,
            },
            {
                "id": True,
                "revision": 1,
                "target_kind": "profile",
                "target_total": 1,
                "candidate_total": 1,
                "category": "identity",
                "title": "Boolean id",
                **malformed_totals,
            },
            {
                "id": 3,
                "revision": "1",
                "target_kind": "profile",
                "target_total": 1,
                "candidate_total": 1,
                "category": "identity",
                "title": "String revision",
                **malformed_totals,
            }
        ]
        try:
            malformed_row_review = learning_review({"limit": 5})
        finally:
            store.list_knowledge_promotion_candidates = original_candidates  # type: ignore[method-assign]
        if (
            not malformed_row_review.ok
            or malformed_row_review.metadata.get("knowledge_promotion_unreadable_rows") != 4
            or malformed_row_review.metadata.get("knowledge_promotion_candidates") != 0
            or malformed_row_review.metadata.get("knowledge_promotion_selected_rows") != 4
            or malformed_row_review.metadata.get("knowledge_promotion_candidate_total") != 4
            or malformed_row_review.metadata.get("knowledge_promotion_totals_reliable") is not False
            or malformed_row_review.metadata.get("knowledge_promotion_hidden_count") != 0
            or "candidate rows scanned for the review packet: 4; global totals unavailable"
            not in malformed_row_review.output
        ):
            raise SystemExit(
                f"malformed candidate row did not degrade safely: {malformed_row_review}"
            )

        store.list_knowledge_promotion_candidates = lambda _limit, **_kwargs: [  # type: ignore[method-assign]
            {
                "id": 10,
                "revision": 1,
                "target_kind": "profile",
                "target_total": 5,
                "candidate_total": 5,
                "category": "identity",
                "title": "First valid total",
                "decision_total": 0,
                "goal_total": 0,
                "person_total": 0,
                "preference_total": 0,
                "profile_total": 5,
            },
            {
                "id": 11,
                "revision": 1,
                "target_kind": "profile",
                "target_total": 6,
                "candidate_total": 6,
                "category": "identity",
                "title": "Conflicting total",
                "decision_total": 0,
                "goal_total": 0,
                "person_total": 0,
                "preference_total": 0,
                "profile_total": 6,
            },
        ]
        try:
            conflicting_totals_review = learning_review({"limit": 5})
        finally:
            store.list_knowledge_promotion_candidates = original_candidates  # type: ignore[method-assign]
        if (
            not conflicting_totals_review.ok
            or conflicting_totals_review.metadata.get("knowledge_promotion_selected_rows") != 2
            or conflicting_totals_review.metadata.get("knowledge_promotion_candidate_total") != 2
            or conflicting_totals_review.metadata.get("knowledge_promotion_candidates") != 1
            or conflicting_totals_review.metadata.get("knowledge_promotion_unreadable_rows") != 1
            or conflicting_totals_review.metadata.get("knowledge_promotion_totals_reliable") is not False
            or conflicting_totals_review.metadata.get("knowledge_promotion_totals_conflict") is not True
            or conflicting_totals_review.metadata.get("knowledge_promotion_target_counts") != {}
        ):
            raise SystemExit(
                "conflicting candidate totals were presented as authoritative: "
                f"{conflicting_totals_review.metadata}"
            )

        legacy_path = root / "legacy-read-only.sqlite"
        with sqlite3.connect(legacy_path) as legacy_conn:
            legacy_conn.execute(
                "CREATE TABLE memories("
                "id INTEGER PRIMARY KEY, category TEXT, title TEXT, body TEXT, source TEXT, "
                "confidence REAL, revision INTEGER, created_at TEXT, updated_at TEXT)"
            )
            legacy_conn.execute(
                "INSERT INTO memories VALUES(1, 'identity', 'Legacy', 'Private', 'manual', "
                "1.0, 1, '2026-07-13T00:00:00Z', '2026-07-13T00:00:00Z')"
            )
        legacy_before = legacy_path.read_bytes()
        legacy_path.chmod(0o444)
        try:
            try:
                MemoryStore(legacy_path).list_knowledge_promotion_candidates(5)
            except sqlite3.OperationalError:
                pass
            else:
                raise SystemExit("legacy schema without custody tables should fail closed")
            if legacy_path.read_bytes() != legacy_before:
                raise SystemExit("legacy read-only candidate inspection mutated the database")
        finally:
            legacy_path.chmod(0o644)

        unsupported_runtime = make_temp_runtime(root / "unsupported-categories")
        for category in ("research", "prompts", "work-history"):
            unsupported_runtime.store.add_memory(
                MemoryRecord(category, f"{category} memory", "Still personal knowledge.", "manual")
            )
        unsupported_review = make_memory_curator_tools(
            unsupported_runtime.store, unsupported_runtime.vault
        )[8]({"limit": 5})
        if (
            unsupported_review.metadata.get("knowledge_promotion_candidate_total") != 0
            or "other memories may still contain structured personal knowledge"
            not in unsupported_review.output
            or "either unstructured facts or already under structured custody"
            in unsupported_review.output
        ):
            raise SystemExit(
                f"empty label scan overclaimed supported knowledge coverage: {unsupported_review}"
            )

        truncation_runtime = make_temp_runtime(root / "truncated-candidates")
        for index in range(200):
            truncation_runtime.store.add_memory(
                MemoryRecord(
                    "identity",
                    f"Long classification candidate {index} " + ("x" * 220),
                    "Advisory only.",
                    "manual",
                )
            )
        truncation_review = make_memory_curator_tools(
            truncation_runtime.store, truncation_runtime.vault
        )[8]({"limit": 200})
        truncation_lines = truncation_review.output.splitlines()
        candidate_headers = sum(
            line.startswith("  - memory #") and " revision " in line
            for line in truncation_lines
        )
        rendered_candidates = sum(
            line.startswith("  - memory #")
            and " revision " in line
            and index + 1 < len(truncation_lines)
            and truncation_lines[index + 1].startswith(
                "    Review first: `knowledge promotion packet "
            )
            for index, line in enumerate(truncation_lines)
        )
        if (
            truncation_review.metadata.get("knowledge_promotion_candidates") != 200
            or truncation_review.metadata.get("knowledge_promotion_selected_rows") != 200
            or truncation_review.metadata.get("knowledge_promotion_rendered_candidates")
            != rendered_candidates
            or truncation_review.metadata.get("knowledge_promotion_output_hidden_count")
            != 200 - rendered_candidates
            or candidate_headers <= rendered_candidates
            or rendered_candidates >= 200
        ):
            raise SystemExit(
                f"truncated review reported false rendered counts: {truncation_review.metadata}"
            )
        assert_learning_review_handoff(
            truncation_review.metadata,
            "truncated_knowledge_classification_review",
            writes=False,
        )


def main() -> None:
    test_planner_routes_learning_review_aliases()
    test_fractional_memory_ids_are_rejected_before_lookup()
    test_learning_review_rotations_do_not_couple_target_and_row_offsets()
    test_learning_review_partial_candidate_block_is_not_counted_as_rendered()
    test_learning_review_surfaces_only_unowned_structured_candidates()
    test_atomic_memory_merge()
    test_learning_review_sanitizes_stored_display_content()
    test_learning_task_queue_identity_is_complete_and_atomic()
    test_learning_task_queue_binds_after_action_and_sanitizes_tool_names()
    assert_memory_metadata_bool_is_exact()
    assert_memory_malformed_handoff_flags_are_exact()
    with TemporaryDirectory(prefix="jarvis-learning-review-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        blocked_run = runtime.handle("run command python3 --version")
        print(f"[{'ok' if blocked_run.verified else 'blocked'}] run command python3 --version")
        print(blocked_run.response[:1800])
        print()
        if blocked_run.verified:
            raise SystemExit("Expected seeded shell command to be approval-blocked.")
        blocked_run_rows = runtime.store.recent_tool_runs(limit=5)
        blocked_run_id = int(blocked_run_rows[0]["id"])
        closure_result = runtime.handle(f"execution learning closure {blocked_run_id}")
        print(f"[{'ok' if closure_result.verified else 'blocked'}] execution learning closure {blocked_run_id}")
        print(closure_result.response[:1800])
        print()
        if not closure_result.verified:
            raise SystemExit("Expected execution learning closure packet to run read-only.")
        closure_metadata = closure_result.tool_results[0].metadata
        actionable_queue = closure_metadata.get("actionable_proof_queue") or []
        expected_after_action = f"after-action learning packet {blocked_run_id}"
        expected_closure = f"execution learning closure {blocked_run_id}"
        if expected_after_action not in actionable_queue or expected_closure not in actionable_queue:
            raise SystemExit(f"execution learning closure missed actionable learning commands: {closure_metadata}")
        if actionable_queue.index(expected_after_action) > actionable_queue.index(expected_closure):
            raise SystemExit(f"execution learning closure should put after-action evidence before closure recheck: {closure_metadata}")
        if closure_metadata.get("next_evidence_command") != actionable_queue[0]:
            raise SystemExit(f"execution learning closure missed next evidence command alias: {closure_metadata}")
        assert_review_only_metadata(closure_metadata, "execution learning closure")
        cases = [
            "feedback: Jarvis should explain approval risk before computer control",
            "set preference response style to concise and warm category communication",
            "remember that likely rough memory should be reviewed",
            "remember that duplicate learning memory should merge later",
            "remember that duplicate learning memory should merge later",
            "failure to test: Jarvis overlapped diagnostics text in the dashboard",
            "feedback: Jarvis dashboard diagnostics overlapped the composer",
            "feedback: Jarvis dashboard text overlapped the system diagnosis",
            "failure clusters",
            "failure promotion packet",
            "session learning preview",
            "learning review",
            "save learning review",
            "queue learning tasks",
            "tasks",
            "help memory",
            "list tools learning",
        ]
        review_note = Path(temp) / "Vault" / "Jarvis" / "Automations" / "Learning Review.md"
        (
            list_weak_memories,
            delete_memory,
            get_memory,
            edit_memory,
            list_duplicate_memories,
            merge_memories,
            memory_tree_summary,
            memory_stats,
            learning_review,
            save_learning_review,
            queue_learning_tasks,
            _personal_context_status,
        ) = make_memory_curator_tools(runtime.store, runtime.vault)
        for case in cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            if case == "learning review":
                assert_contains(
                    result.response,
                    [
                        "Jarvis learning review",
                        "review-only loop",
                        "Feedback signals",
                        "safety",
                        "Memory health",
                        "weak-looking memories",
                        "duplicate-looking memory groups",
                        "Personalization base",
                        "Execution learning debt",
                        "failed or blocked action runs",
                        "approval-held action runs",
                        "recent after-action learning packets",
                        "learning state: LEARNING_REVIEW_REQUIRED",
                        "next learning required commands",
                        "after-action learning packet",
                        "approval review commands",
                        "Safe learning actions",
                        "Still manual or approval-gated",
                        "explicit stop times",
                    ],
                    case,
                )
                if "next learning proof commands" in result.response:
                    raise SystemExit(f"{case} should render command-first learning handoff prose: {result.response}")
                assert_review_only_metadata(
                    result.tool_results[0].metadata, case, reads_private_data=True
                )
                metadata = result.tool_results[0].metadata
                assert_learning_review_handoff(metadata, case, writes=False)
                if metadata.get("failed_or_blocked_action_runs") != 0:
                    raise SystemExit(f"{case} should not count approval-held rows as failed learning debt: {metadata}")
                if metadata.get("approval_held_action_runs", 0) < 1:
                    raise SystemExit(f"{case} missed approval-held learning context metadata: {metadata}")
                if metadata.get("execution_learning_state") != "LEARNING_REVIEW_REQUIRED":
                    raise SystemExit(f"{case} missed learning review state: {metadata}")
                if metadata.get("execution_learning_missing_count", 0) < 1:
                    raise SystemExit(f"{case} missed missing learning proof metadata: {metadata}")
                next_commands = metadata.get("execution_learning_next_commands") or []
                if not any(str(command).startswith("after-action learning packet") for command in next_commands):
                    raise SystemExit(f"{case} missed after-action learning command: {metadata}")
                approval_commands = metadata.get("execution_learning_approval_review_commands") or []
                if not any(str(command).startswith("approval readiness") for command in approval_commands):
                    raise SystemExit(f"{case} missed approval-held review commands: {metadata}")
                assert_operator_output(result.response, case)
            if case == "session learning preview":
                assert_contains(
                    result.response,
                    [
                        "Jarvis session learning preview",
                        "read-only",
                        "Candidate memories",
                        "likely rough memory should be reviewed",
                        "Candidate preferences",
                        "concise and warm",
                        "Candidate tasks",
                        "Candidate skill/workflow signals",
                        "Safe follow-up commands",
                        "Boundary",
                        "approval-gated",
                    ],
                    case,
                )
            if case.startswith("failure to test:"):
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure-to-test preview",
                        "read-only",
                        "Likely regression surface",
                        "frontend layout and visual regression",
                        "Proposed smoke test",
                        "Proposed assertions",
                        "Safe next commands",
                        "Boundary",
                    ],
                    case,
                )
                assert_review_only_metadata(result.metadata, case)
            if case == "failure clusters":
                assert_contains(
                    result.response,
                    [
                        "Jarvis repeated-failure cluster report",
                        "read-only",
                        "Clusters",
                        "frontend layout and visual regression",
                        "strong repeated signal",
                        "Ranked next safe move",
                        "Boundary",
                    ],
                    case,
                )
                assert_review_only_metadata(result.metadata, case)
            if case == "failure promotion packet":
                assert_contains(
                    result.response,
                    [
                        "Jarvis failure promotion packet",
                        "read-only promotion plan",
                        "Selected cluster",
                        "Draft smoke-test assertions",
                        "Promotion checklist",
                        "Boundary",
                    ],
                    case,
                )
                assert_review_only_metadata(result.metadata, case)
            if case == "save learning review":
                assert_contains(
                    result.response,
                    [
                        "Learning review saved",
                        "Jarvis learning review",
                        "Feedback signals",
                        "Memory health",
                        "Safe learning actions",
                    ],
                    case,
                )
                if not review_note.exists():
                    raise SystemExit(f"{case} did not write expected note: {review_note}")
                assert_contains(
                    review_note.read_text(encoding="utf-8"),
                    [
                        "# Learning Review",
                        "Jarvis learning review",
                        "weak-looking memories",
                        "duplicate-looking memory groups",
                        "approval-gated",
                        "explicit stop times",
                    ],
                    f"{case} note",
                )
                assert_operator_metadata(result.tool_results[0].metadata, case)
                assert_operator_output(result.response, case)
                assert_vault_relative_receipt(result, root, "Automations/", case)
                assert_learning_review_handoff(result.tool_results[0].metadata, case, writes=True)
            if case == "queue learning tasks":
                assert_contains(
                    result.response,
                    [
                        "Queued learning tasks",
                        "local task reminders only",
                        "Review Jarvis feedback report",
                        "Review weak-looking memories",
                        "Inspect duplicate-looking memories",
                        "Review approval-held run",
                        "`approval readiness 1`",
                        "before treating it as execution evidence",
                        "Still approval-gated",
                        "explicit stop times",
                    ],
                    case,
                )
                assert_operator_metadata(result.tool_results[0].metadata, case)
                assert_operator_output(result.response, case)
                assert_queue_learning_tasks_handoff(result.tool_results[0].metadata, case)
            if case == "tasks":
                assert_contains(
                    result.response,
                    [
                        "Tasks:",
                        "Review Jarvis feedback report",
                        "Review weak-looking memories",
                        "Inspect duplicate-looking memories",
                        "Review approval-held run",
                        "`approval readiness 1`",
                        "before treating it as execution evidence",
                    ],
                    case,
                )
            if case == "help memory":
                assert_contains(result.response, ["learning review", "session learning preview", "failure to test:", "failure clusters", "failure promotion packet", "failure apply contract", "save learning review", "queue learning tasks"], case)
            if case == "list tools learning":
                assert_contains(result.response, ["learning_review", "session_learning_preview", "failure_to_test_preview", "repeated_failure_clusters", "failure_promotion_packet", "failure_apply_contract", "save_learning_review", "queue_learning_tasks"], case)

        weak_result = list_weak_memories({"limit": "not-a-number"})
        if weak_result.metadata.get("limit") != 25 or weak_result.metadata.get("writes_files"):
            raise SystemExit("list_weak_memories did not sanitize bad limits as read-only.")
        assert_review_only_metadata(weak_result.metadata, "list_weak_memories")
        assert_operator_output(weak_result.output, "list_weak_memories")
        weak_bool = list_weak_memories({"limit": False})
        if weak_bool.metadata.get("limit") != 25 or weak_bool.metadata.get("writes_files"):
            raise SystemExit(f"list_weak_memories should treat boolean limits as malformed defaults: {weak_bool.metadata}")
        assert_review_only_metadata(weak_bool.metadata, "list_weak_memories_bool_limit")

        duplicate_result = list_duplicate_memories({"limit": 999999})
        if duplicate_result.metadata.get("limit") != 200 or duplicate_result.metadata.get("controls_computer"):
            raise SystemExit("list_duplicate_memories did not clamp huge limits as local read-only.")
        assert_review_only_metadata(duplicate_result.metadata, "list_duplicate_memories")
        assert_operator_output(duplicate_result.output, "list_duplicate_memories")
        duplicate_bool = list_duplicate_memories({"limit": True})
        if duplicate_bool.metadata.get("limit") != 200 or duplicate_bool.metadata.get("controls_computer"):
            raise SystemExit(f"list_duplicate_memories should treat boolean limits as malformed defaults: {duplicate_bool.metadata}")
        assert_review_only_metadata(duplicate_bool.metadata, "list_duplicate_memories_bool_limit")

        review_result = learning_review({"limit": "bad"})
        if review_result.metadata.get("limit") != 12 or review_result.metadata.get("writes_files"):
            raise SystemExit("learning_review should clamp limits and remain read-only.")
        if review_result.metadata.get("failed_or_blocked_action_runs") != 0:
            raise SystemExit(f"learning_review should not count approval-held rows as failed direct execution metadata: {review_result.metadata}")
        if review_result.metadata.get("approval_held_action_runs", 0) < 1:
            raise SystemExit(f"learning_review missed direct approval-held execution metadata: {review_result.metadata}")
        if review_result.metadata.get("execution_learning_state") != "LEARNING_REVIEW_REQUIRED":
            raise SystemExit(f"learning_review missed direct execution learning metadata: {review_result.metadata}")
        if review_result.metadata.get("execution_learning_blocks_completion_claim") is not True:
            raise SystemExit(f"learning_review missed completion blocker flag for execution learning debt: {review_result.metadata}")
        learning_commands = review_result.metadata.get("execution_learning_required_commands") or []
        if learning_commands != review_result.metadata.get("execution_learning_next_commands"):
            raise SystemExit(f"learning_review canonical learning commands diverged from legacy next commands: {review_result.metadata}")
        if review_result.metadata.get("execution_learning_proof_queue") != learning_commands:
            raise SystemExit(f"learning_review missed execution learning proof queue alias: {review_result.metadata}")
        if review_result.metadata.get("execution_learning_proof_queue_count") != len(learning_commands):
            raise SystemExit(f"learning_review missed execution learning proof queue count: {review_result.metadata}")
        if learning_commands and review_result.metadata.get("execution_learning_next_proof_command") != learning_commands[0]:
            raise SystemExit(f"learning_review missed next proof command alias: {review_result.metadata}")
        if review_result.metadata.get("execution_learning_next_required_command") != review_result.metadata.get("execution_learning_next_command"):
            raise SystemExit(f"learning_review required and legacy next command aliases diverged: {review_result.metadata}")
        if "next learning proof commands" in review_result.output:
            raise SystemExit(f"learning_review should render next learning required commands, not proof-first prose: {review_result.output}")
        assert_review_only_metadata(
            review_result.metadata, "learning_review", reads_private_data=True
        )
        assert_operator_output(review_result.output, "learning_review")
        assert_learning_review_handoff(review_result.metadata, "learning_review", writes=False)
        review_bool = learning_review({"limit": True})
        if review_bool.metadata.get("limit") != 12 or review_bool.metadata.get("writes_files"):
            raise SystemExit(f"learning_review should treat boolean limits as malformed defaults: {review_bool.metadata}")
        assert_review_only_metadata(
            review_bool.metadata, "learning_review_bool_limit", reads_private_data=True
        )
        assert_learning_review_handoff(review_bool.metadata, "learning_review_bool_limit", writes=False)

        with TemporaryDirectory(prefix="jarvis-learning-meta-target-") as meta_temp:
            meta_runtime = make_temp_runtime(Path(meta_temp))
            _meta_tools = make_memory_curator_tools(meta_runtime.store, meta_runtime.vault)
            meta_learning_review = _meta_tools[8]
            action_run_id = meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="calculate",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="4",
                metadata={"route": "tools"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="completion_next_proof_packet",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis completion next proof packet",
                metadata={"next_proof_command": f"after-action learning packet {action_run_id}"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="harness_readiness_digest",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis harness readiness digest",
                metadata={"source_tool": "completion_next_proof_packet"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="specialist_handoff_quality_gate",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis specialist handoff quality gate",
                metadata={"quality_gate_state": "HANDOFF_QUALITY_READY_FOR_PROPOSAL_GATE"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="specialist_proposal_gate",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis specialist proposal gate",
                metadata={"proposal_gate_state": "PROPOSAL_FALLBACK_NO_MODEL"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="continuation_packet",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis continuation packet with checkpoint recovery contract",
                metadata={"checkpoint_freshness": "missing"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="checkpoint_recovery_preview",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis checkpoint recovery preview",
                metadata={"checkpoint_freshness": "missing"},
            )
            meta_runtime.store.log_tool_run(
                session_id="meta-target",
                tool_name="learning_review",
                risk="READ_ONLY",
                ok=True,
                approved=False,
                output="Jarvis learning review",
                metadata={"execution_learning_target_run_id": action_run_id},
            )
            meta_review = meta_learning_review({})
            if meta_review.metadata.get("execution_learning_target_run_id") != action_run_id:
                raise SystemExit(f"learning_review should ignore completion/readiness/learning proof packets as learning targets: {meta_review.metadata}")
            if meta_review.metadata.get("execution_learning_target_tool") != "calculate":
                raise SystemExit(f"learning_review selected the wrong non-meta learning target: {meta_review.metadata}")
            if meta_review.metadata.get("recent_action_runs") != 1:
                raise SystemExit(f"learning_review should classify completion/readiness/learning packets as meta runs: {meta_review.metadata}")
            assert_review_only_metadata(
                meta_review.metadata, "learning_review_meta_target", reads_private_data=True
            )
            assert_operator_output(meta_review.output, "learning_review_meta_target")
            assert_learning_review_handoff(meta_review.metadata, "learning_review_meta_target", writes=False)

        with TemporaryDirectory(prefix="jarvis-learning-approval-held-") as held_temp:
            held_runtime = make_temp_runtime(Path(held_temp))
            held_tools = make_memory_curator_tools(held_runtime.store, held_runtime.vault)
            held_learning_review = held_tools[8]
            held_queue_learning_tasks = held_tools[10]
            held_run_id = held_runtime.store.log_tool_run(
                session_id="held-target",
                tool_name="send_telegram",
                risk="HIGH_RISK",
                ok=False,
                approved=False,
                output="explicit approval required before execution",
                metadata={
                    "failure_kind": "approval_required",
                    "requires_confirmation": True,
                    "toolset": "messages",
                },
            )
            held_approval_id = held_runtime.store.add_pending_approval(
                "held-target",
                "send telegram to Fixture saying hi",
                "send_telegram",
                "explicit approval required before execution",
                {"to": "Fixture", "body": "hi"},
            )
            held_runtime.store.link_tool_run_approval(held_run_id, held_approval_id)

            held_review = held_learning_review({})
            assert_review_only_metadata(
                held_review.metadata,
                "learning_review_approval_held",
                reads_private_data=True,
            )
            assert_operator_output(held_review.output, "learning_review_approval_held")
            assert_learning_review_handoff(held_review.metadata, "learning_review_approval_held", writes=False)
            expected_approval_commands = [
                f"approval readiness {held_approval_id}",
                f"approval packet {held_approval_id}",
                f"approval chain proof {held_approval_id}",
            ]
            if held_review.metadata.get("failed_or_blocked_action_runs") != 0:
                raise SystemExit(f"approval-held learning review should not count held rows as failures: {held_review.metadata}")
            if held_review.metadata.get("approval_held_action_runs") != 1:
                raise SystemExit(f"approval-held learning review missed held-row count: {held_review.metadata}")
            if held_review.metadata.get("execution_learning_state") != "APPROVAL_REVIEW_REQUIRED":
                raise SystemExit(f"approval-held learning review selected wrong state: {held_review.metadata}")
            if held_review.metadata.get("execution_learning_blocks_completion_claim"):
                raise SystemExit(f"approval-held-only rows should not block completion as learning debt: {held_review.metadata}")
            if held_review.metadata.get("execution_learning_target_run_id") is not None:
                raise SystemExit(f"approval-held-only rows should not become execution learning targets: {held_review.metadata}")
            if held_review.metadata.get("execution_learning_approval_review_commands") != expected_approval_commands:
                raise SystemExit(f"approval-held learning review missed approval proof commands: {held_review.metadata}")
            assert_contains(
                held_review.output,
                [
                    "failed or blocked action runs: 0",
                    "approval-held action runs: 1",
                    "learning state: APPROVAL_REVIEW_REQUIRED",
                    "blocks completion claim: no",
                    "target run: none",
                    "approval review commands",
                    f"approval readiness {held_approval_id}",
                    f"approval packet {held_approval_id}",
                    f"approval chain proof {held_approval_id}",
                ],
                "learning_review_approval_held",
            )
            if "failed or blocked action runs: 1" in held_review.output:
                raise SystemExit(f"approval-held learning review should not render held rows as failed: {held_review.output}")

            held_queue = held_queue_learning_tasks({})
            assert_operator_metadata(held_queue.metadata, "queue_learning_tasks_approval_held")
            assert_operator_output(held_queue.output, "queue_learning_tasks_approval_held")
            assert_queue_learning_tasks_handoff(held_queue.metadata, "queue_learning_tasks_approval_held")
            assert_contains(
                held_queue.output,
                [
                    f"Review approval-held run #{held_run_id} send_telegram",
                    f"`approval readiness {held_approval_id}`",
                    "before treating it as execution evidence",
                ],
                "queue_learning_tasks_approval_held",
            )
            if f"failed run #{held_run_id}" in held_queue.output:
                raise SystemExit(f"approval-held queue should not add a false failed-run task: {held_queue.output}")

        with TemporaryDirectory(prefix="jarvis-learning-structured-held-") as structured_temp:
            structured_runtime = make_temp_runtime(Path(structured_temp))
            structured_tools = make_memory_curator_tools(structured_runtime.store, structured_runtime.vault)
            structured_learning_review = structured_tools[8]
            structured_queue_learning_tasks = structured_tools[10]
            leak_marker = "LEARNING_REVIEW_METADATA_SHOULD_NOT_LEAK /\x55sers/example/private/learning.sqlite"
            structured_rows = [
                {
                    "id": 901,
                    "session_id": "structured-held",
                    "tool_name": "send_telegram",
                    "risk": "HIGH_RISK",
                    "ok": False,
                    "approved": False,
                    "approval_id": 44,
                    "output": "explicit approval required before execution",
                    "metadata": {
                        "requires_confirmation": HostileMetadataValue(leak_marker),
                        "failure_kind": HostileMetadataValue(leak_marker),
                        "failure_stage": "explicit approval required",
                        "approval_id": 44,
                    },
                    "created_at": "2026-07-06T11:00:00Z",
                }
            ]
            structured_runtime.store.recent_tool_runs = lambda limit=80: structured_rows[:limit]  # type: ignore[method-assign]

            structured_review = structured_learning_review({})
            assert_review_only_metadata(
                structured_review.metadata,
                "learning_review_structured_approval_held",
                reads_private_data=True,
            )
            assert_operator_output(structured_review.output, "learning_review_structured_approval_held")
            assert_learning_review_handoff(structured_review.metadata, "learning_review_structured_approval_held", writes=False)
            if structured_review.metadata.get("failed_or_blocked_action_runs") != 0:
                raise SystemExit(f"structured approval-held row should not count as failed: {structured_review.metadata}")
            if structured_review.metadata.get("approval_held_action_runs") != 1:
                raise SystemExit(f"structured approval-held row was not counted: {structured_review.metadata}")
            if structured_review.metadata.get("execution_learning_state") != "APPROVAL_REVIEW_REQUIRED":
                raise SystemExit(f"structured approval-held row selected wrong learning state: {structured_review.metadata}")
            expected_structured_commands = [
                "approval readiness 44",
                "approval packet 44",
                "approval chain proof 44",
            ]
            if structured_review.metadata.get("execution_learning_approval_review_commands") != expected_structured_commands:
                raise SystemExit(f"structured approval-held row missed approval commands: {structured_review.metadata}")
            for haystack, label in (
                (structured_review.output, "structured approval-held output"),
                (repr(structured_review.metadata), "structured approval-held metadata"),
                (repr(structured_review.metadata.get("learning_review_handoff")), "structured approval-held handoff"),
            ):
                if leak_marker in haystack or "learning.sqlite" in haystack or "/\x55sers/example/private" in haystack:
                    raise SystemExit(f"{label} leaked hostile metadata marker: {haystack[:1000]}")

            structured_queue = structured_queue_learning_tasks({})
            assert_operator_metadata(structured_queue.metadata, "queue_learning_tasks_structured_approval_held")
            assert_operator_output(structured_queue.output, "queue_learning_tasks_structured_approval_held")
            assert_queue_learning_tasks_handoff(structured_queue.metadata, "queue_learning_tasks_structured_approval_held")
            assert_contains(
                structured_queue.output,
                [
                    "Review approval-held run #901 send_telegram",
                    "`approval readiness 44`",
                    "before treating it as execution evidence",
                ],
                "queue_learning_tasks_structured_approval_held",
            )
            for haystack, label in (
                (structured_queue.output, "structured queue output"),
                (repr(structured_queue.metadata), "structured queue metadata"),
                (repr(structured_queue.metadata.get("queue_learning_tasks_handoff")), "structured queue handoff"),
            ):
                if leak_marker in haystack or "learning.sqlite" in haystack or "/\x55sers/example/private" in haystack:
                    raise SystemExit(f"{label} leaked hostile metadata marker: {haystack[:1000]}")

        save_result = save_learning_review({"limit": -20})
        if save_result.metadata.get("limit") != 1 or not save_result.metadata.get("writes_files"):
            raise SystemExit("save_learning_review should clamp low limits and mark file writes.")
        if not save_result.metadata.get("writes_notes") or save_result.metadata.get("writes_database"):
            raise SystemExit("save_learning_review should mark note writes without database writes.")
        assert_operator_metadata(save_result.metadata, "save_learning_review")
        assert_operator_output(save_result.output, "save_learning_review")
        if save_result.output.count("Memory curation boundary:") != 1:
            raise SystemExit("save_learning_review should render its memory boundary exactly once.")
        if (
            "`save learning review` publishes this report to Obsidian and maintains a local lock file"
            not in save_result.output
            or "No promotion, model call, file write" in save_result.output
            or "No safe ownership-transfer path exists yet" in save_result.output
            or "Decision, new-preference, and profile candidates have exact, approval-gated ownership-transfer paths"
            not in save_result.output
        ):
            raise SystemExit("save_learning_review should disclose its report-only Obsidian write.")
        assert_learning_review_handoff(save_result.metadata, "save_learning_review", writes=True)
        save_bool = save_learning_review({"limit": False})
        if save_bool.metadata.get("limit") != 12 or not save_bool.metadata.get("writes_files"):
            raise SystemExit(f"save_learning_review should treat boolean limits as malformed defaults: {save_bool.metadata}")
        assert_operator_metadata(save_bool.metadata, "save_learning_review_bool_limit")
        assert_learning_review_handoff(save_bool.metadata, "save_learning_review_bool_limit", writes=True)

        queue_result = queue_learning_tasks({"limit": 50000})
        if queue_result.metadata.get("limit") != 200 or not queue_result.metadata.get("writes_files"):
            raise SystemExit("queue_learning_tasks should clamp huge limits and mark file writes.")
        if not queue_result.metadata.get("writes_notes"):
            raise SystemExit("queue_learning_tasks should mark local note writes.")
        if queue_result.metadata.get("writes_memory") != queue_result.metadata.get("writes_database"):
            raise SystemExit("queue_learning_tasks should mark memory writes only when it adds task rows.")
        assert_operator_metadata(queue_result.metadata, "queue_learning_tasks")
        assert_operator_output(queue_result.output, "queue_learning_tasks")
        assert_queue_learning_tasks_handoff(queue_result.metadata, "queue_learning_tasks")
        queue_bool = queue_learning_tasks({"limit": True})
        if queue_bool.metadata.get("limit") != 12 or not queue_bool.metadata.get("writes_files"):
            raise SystemExit(f"queue_learning_tasks should treat boolean limits as malformed defaults: {queue_bool.metadata}")
        assert_operator_metadata(queue_bool.metadata, "queue_learning_tasks_bool_limit")
        assert_queue_learning_tasks_handoff(queue_bool.metadata, "queue_learning_tasks_bool_limit")

        bad_get = get_memory({"memory_id": 0})
        if bad_get.ok or "required" not in bad_get.output:
            raise SystemExit("get_memory should reject non-positive memory ids.")
        if bad_get.metadata.get("raw_memory_id") != "0":
            raise SystemExit(f"get_memory should preserve bounded raw memory id metadata: {bad_get.metadata}")
        assert_review_only_metadata(bad_get.metadata, "bad_get_memory")
        assert_memory_refusal_handoff(
            bad_get.metadata,
            "bad_get_memory",
            source="get_memory",
            mutation="memory_read",
            read_only=True,
            requires_approval=False,
        )
        assert_operator_output(bad_get.output, "bad_get_memory")
        bool_get = get_memory({"memory_id": True})
        if bool_get.ok or "must be a number" not in bool_get.output:
            raise SystemExit("get_memory should reject boolean memory ids.")
        if bool_get.metadata.get("raw_memory_id") != "True":
            raise SystemExit(f"get_memory should preserve boolean raw memory id metadata: {bool_get.metadata}")
        assert_review_only_metadata(bool_get.metadata, "bool_get_memory")

        malformed_get = get_memory({"memory_id": "memory-abc"})
        if malformed_get.ok or malformed_get.metadata.get("raw_memory_id") != "memory-abc":
            raise SystemExit(f"get_memory should preserve malformed raw memory id metadata: {malformed_get.metadata}")
        assert_review_only_metadata(malformed_get.metadata, "malformed_get_memory")
        path_bad_get = get_memory({"memory_id": "/\x55sers/example/private/memory-id"})
        if path_bad_get.ok or path_bad_get.metadata.get("raw_memory_id") != "<local-path>":
            raise SystemExit(f"get_memory should redact path-shaped bad memory ids: {path_bad_get.metadata}")
        assert_review_only_metadata(path_bad_get.metadata, "path_bad_get_memory")
        for raw_memory_id in ("/var/folders/zc/jarvis-memory-id", "/tmp/jarvis-memory-id"):
            temp_path_bad_get = get_memory({"memory_id": raw_memory_id})
            if temp_path_bad_get.ok or temp_path_bad_get.metadata.get("raw_memory_id") != "<local-path>":
                raise SystemExit(f"get_memory should redact temp-root path-shaped bad memory ids: {temp_path_bad_get.metadata}")
            assert_review_only_metadata(temp_path_bad_get.metadata, "temp_path_bad_get_memory")

        missing_get = get_memory({"memory_id": 999999})
        if missing_get.ok or missing_get.metadata.get("memory_id") != 999999:
            raise SystemExit(f"get_memory should fail closed for a missing positive id: {missing_get}")
        for token in (
            "show recent memories",
            "search memory for <title or detail>",
            "show memory <correct memory id>",
            "normal read-only policy",
        ):
            if token not in missing_get.output:
                raise SystemExit(f"get_memory missing-id output missed {token!r}: {missing_get.output}")
        if missing_get.metadata.get("recovery_commands") != [
            "show recent memories",
            "search memory for <title or detail>",
            "show memory <correct memory id>",
        ]:
            raise SystemExit(f"get_memory missing-id recovery order drifted: {missing_get.metadata}")
        for key in ("retry_requires_memory_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
            if missing_get.metadata.get(key) is not True:
                raise SystemExit(f"get_memory missing-id recovery missed {key}: {missing_get.metadata}")
        if missing_get.metadata.get("retry_requires_fresh_approval") is not False:
            raise SystemExit(f"get_memory read-only recovery should not require approval: {missing_get.metadata}")
        for key in ("authorizes_retry", "authorizes_memory_mutation", "writes_memory", "writes_database"):
            if missing_get.metadata.get(key):
                raise SystemExit(f"get_memory missing-id recovery unexpectedly set {key}: {missing_get.metadata}")
        assert_review_only_metadata(missing_get.metadata, "missing_get_memory")
        assert_memory_refusal_handoff(
            missing_get.metadata,
            "missing_get_memory",
            source="get_memory",
            mutation="memory_read",
            read_only=True,
            requires_approval=False,
        )
        assert_operator_output(missing_get.output, "missing_get_memory")

        oversized_edit = edit_memory({"memory_id": 1, "body": "x" * 50001})
        if oversized_edit.ok or "too large" not in oversized_edit.output:
            raise SystemExit("edit_memory should refuse oversized memory bodies.")
        assert_operator_metadata(oversized_edit.metadata, "oversized_edit_memory")
        assert_memory_refusal_handoff(
            oversized_edit.metadata,
            "oversized_edit_memory",
            source="edit_memory",
            mutation="memory_edit",
        )
        assert_operator_output(oversized_edit.output, "oversized_edit_memory")

        malformed_edit = edit_memory({"memory_id": "m" * 200, "body": "updated"})
        if malformed_edit.ok or malformed_edit.metadata.get("raw_memory_id") != ("m" * 79 + "…"):
            raise SystemExit(f"edit_memory should bound malformed raw memory id metadata: {malformed_edit.metadata}")
        assert_operator_metadata(malformed_edit.metadata, "malformed_edit_memory")
        assert_memory_refusal_handoff(
            malformed_edit.metadata,
            "malformed_edit_memory",
            source="edit_memory",
            mutation="memory_edit",
        )
        path_bad_edit = edit_memory({"memory_id": "/private/tmp/jarvis-edit-memory-id", "body": "updated"})
        if path_bad_edit.ok or path_bad_edit.metadata.get("raw_memory_id") != "<local-path>":
            raise SystemExit(f"edit_memory should redact path-shaped bad memory ids: {path_bad_edit.metadata}")
        assert_operator_metadata(path_bad_edit.metadata, "path_bad_edit_memory")
        for raw_memory_id in ("/var/folders/zc/jarvis-edit-memory-id", "/tmp/jarvis-edit-memory-id"):
            temp_path_bad_edit = edit_memory({"memory_id": raw_memory_id, "body": "updated"})
            if temp_path_bad_edit.ok or temp_path_bad_edit.metadata.get("raw_memory_id") != "<local-path>":
                raise SystemExit(f"edit_memory should redact temp-root path-shaped bad memory ids: {temp_path_bad_edit.metadata}")
            assert_operator_metadata(temp_path_bad_edit.metadata, "temp_path_bad_edit_memory")

        private_edit_body = "PRIVATE_MEMORY_EDIT_SHOULD_NOT_APPEAR"
        missing_edit = edit_memory({"memory_id": 999999, "body": private_edit_body})
        if missing_edit.ok or missing_edit.metadata.get("memory_id") != 999999:
            raise SystemExit(f"edit_memory should fail closed for a missing positive id: {missing_edit}")
        for token in (
            "show recent memories",
            "show memory <correct memory id>",
            "retry the memory edit with the corrected ID",
            "normal approval flow",
            "does not reuse or grant approval",
        ):
            if token not in missing_edit.output:
                raise SystemExit(f"edit_memory missing-id output missed {token!r}: {missing_edit.output}")
        if missing_edit.metadata.get("recovery_commands") != [
            "show recent memories",
            "search memory for <title or detail>",
            "show memory <correct memory id>",
        ]:
            raise SystemExit(f"edit_memory missing-id recovery order drifted: {missing_edit.metadata}")
        for key in (
            "retry_requires_memory_refresh",
            "retry_requires_corrected_id",
            "retry_requires_fresh_approval",
            "recovery_commands_require_normal_policy",
        ):
            if missing_edit.metadata.get(key) is not True:
                raise SystemExit(f"edit_memory missing-id recovery missed {key}: {missing_edit.metadata}")
        for key in ("authorizes_retry", "authorizes_memory_mutation", "writes_memory", "writes_database"):
            if missing_edit.metadata.get(key):
                raise SystemExit(f"edit_memory missing-id recovery unexpectedly set {key}: {missing_edit.metadata}")
        if private_edit_body in f"{missing_edit.output}\n{missing_edit.metadata}":
            raise SystemExit("edit_memory missing-id recovery leaked supplied body text")
        assert_operator_metadata(missing_edit.metadata, "missing_edit_memory")
        assert_memory_refusal_handoff(
            missing_edit.metadata,
            "missing_edit_memory",
            source="edit_memory",
            mutation="memory_edit",
        )
        assert_operator_output(missing_edit.output, "missing_edit_memory")

        bad_confidence = edit_memory({"memory_id": 1, "confidence": "certain"})
        if bad_confidence.ok or bad_confidence.metadata.get("raw_confidence") != "certain":
            raise SystemExit(f"edit_memory should preserve bounded raw confidence metadata: {bad_confidence.metadata}")
        assert_operator_metadata(bad_confidence.metadata, "bad_confidence_edit_memory")
        assert_memory_refusal_handoff(
            bad_confidence.metadata,
            "bad_confidence_edit_memory",
            source="edit_memory",
            mutation="memory_edit",
        )
        path_bad_confidence = edit_memory({"memory_id": 1, "confidence": "/\x55sers/example/private/confidence"})
        if path_bad_confidence.ok or path_bad_confidence.metadata.get("raw_confidence") != "<local-path>":
            raise SystemExit(f"edit_memory should redact path-shaped bad confidence metadata: {path_bad_confidence.metadata}")
        assert_operator_metadata(path_bad_confidence.metadata, "path_bad_confidence_edit_memory")
        for raw_confidence in ("/var/folders/zc/jarvis-confidence", "/tmp/jarvis-confidence"):
            temp_path_bad_confidence = edit_memory({"memory_id": 1, "confidence": raw_confidence})
            if temp_path_bad_confidence.ok or temp_path_bad_confidence.metadata.get("raw_confidence") != "<local-path>":
                raise SystemExit(f"edit_memory should redact temp-root path-shaped bad confidence metadata: {temp_path_bad_confidence.metadata}")
            assert_operator_metadata(temp_path_bad_confidence.metadata, "temp_path_bad_confidence_edit_memory")

        nonfinite_confidence = edit_memory({"memory_id": 1, "confidence": "nan"})
        if nonfinite_confidence.ok or "finite" not in nonfinite_confidence.output:
            raise SystemExit(f"edit_memory should reject non-finite confidence values: {nonfinite_confidence.output}")
        if nonfinite_confidence.metadata.get("raw_confidence") != "nan":
            raise SystemExit(f"edit_memory should preserve non-finite raw confidence metadata: {nonfinite_confidence.metadata}")
        assert_operator_metadata(nonfinite_confidence.metadata, "nonfinite_confidence_edit_memory")

        long_bad_confidence = edit_memory({"memory_id": 1, "confidence": "c" * 200})
        if long_bad_confidence.ok or long_bad_confidence.metadata.get("raw_confidence") != ("c" * 79 + "…"):
            raise SystemExit(f"edit_memory should bound raw confidence metadata: {long_bad_confidence.metadata}")
        assert_operator_metadata(long_bad_confidence.metadata, "long_bad_confidence_edit_memory")

        same_merge = merge_memories({"keep_id": 1, "delete_id": 1})
        if same_merge.ok or "different" not in same_merge.output:
            raise SystemExit("merge_memories should reject merging the same memory.")
        assert_operator_metadata(same_merge.metadata, "same_merge_memories")
        assert_memory_refusal_handoff(
            same_merge.metadata,
            "same_merge_memories",
            source="merge_memories",
            mutation="memory_merge",
        )
        assert_operator_output(same_merge.output, "same_merge_memories")

        malformed_merge = merge_memories({"keep_id": "keep-abc", "delete_id": "delete-abc"})
        if malformed_merge.ok or malformed_merge.metadata.get("raw_keep_id") != "keep-abc" or malformed_merge.metadata.get("raw_delete_id") != "delete-abc":
            raise SystemExit(f"merge_memories should preserve malformed raw id metadata: {malformed_merge.metadata}")
        assert_operator_metadata(malformed_merge.metadata, "malformed_merge_memories")
        assert_memory_refusal_handoff(
            malformed_merge.metadata,
            "malformed_merge_memories",
            source="merge_memories",
            mutation="memory_merge",
        )
        path_bad_merge = merge_memories({"keep_id": "/private/tmp/jarvis-keep-id", "delete_id": "/\x55sers/example/private/delete-id"})
        if path_bad_merge.ok or path_bad_merge.metadata.get("raw_keep_id") != "<local-path>" or path_bad_merge.metadata.get("raw_delete_id") != "<local-path>":
            raise SystemExit(f"merge_memories should redact path-shaped bad ids: {path_bad_merge.metadata}")
        assert_operator_metadata(path_bad_merge.metadata, "path_bad_merge_memories")
        temp_path_bad_merge = merge_memories({"keep_id": "/var/folders/zc/jarvis-keep-id", "delete_id": "/tmp/jarvis-delete-id"})
        if (
            temp_path_bad_merge.ok
            or temp_path_bad_merge.metadata.get("raw_keep_id") != "<local-path>"
            or temp_path_bad_merge.metadata.get("raw_delete_id") != "<local-path>"
        ):
            raise SystemExit(f"merge_memories should redact temp-root path-shaped bad ids: {temp_path_bad_merge.metadata}")
        assert_operator_metadata(temp_path_bad_merge.metadata, "temp_path_bad_merge_memories")
        bool_merge = merge_memories({"keep_id": True, "delete_id": False})
        if bool_merge.ok or "must be a number" not in bool_merge.output:
            raise SystemExit("merge_memories should reject boolean memory ids.")
        if bool_merge.metadata.get("raw_keep_id") != "True" or bool_merge.metadata.get("raw_delete_id") != "False":
            raise SystemExit(f"merge_memories should preserve boolean raw id metadata: {bool_merge.metadata}")
        assert_operator_metadata(bool_merge.metadata, "bool_merge_memories")

        missing_merge = merge_memories({"keep_id": 999998, "delete_id": 999999})
        if missing_merge.ok or missing_merge.metadata.get("reason") != "missing_memory":
            raise SystemExit(f"merge_memories should fail closed for missing positive ids: {missing_merge}")
        expected_merge_commands = [
            "show recent memories",
            "search memory for <title or detail>",
            "show memory <correct keep memory id>",
            "show memory <correct delete memory id>",
            "merge memory <correct delete memory id> into <correct keep memory id>",
        ]
        if missing_merge.metadata.get("recovery_commands") != expected_merge_commands:
            raise SystemExit(f"merge_memories missing-id recovery order drifted: {missing_merge.metadata}")
        for token in (
            "show recent memories",
            "search memory for <title or detail>",
            "merge memory <correct delete memory id> into <correct keep memory id>",
            "normal approval flow",
            "does not reuse or grant approval",
        ):
            if token not in missing_merge.output:
                raise SystemExit(f"merge_memories missing-id output missed {token!r}: {missing_merge.output}")
        for key in (
            "retry_requires_memory_refresh",
            "retry_requires_corrected_id",
            "retry_requires_fresh_approval",
            "recovery_commands_require_normal_policy",
        ):
            if missing_merge.metadata.get(key) is not True:
                raise SystemExit(f"merge_memories missing-id recovery missed {key}: {missing_merge.metadata}")
        for key in ("authorizes_retry", "authorizes_memory_mutation", "writes_memory", "writes_database"):
            if missing_merge.metadata.get(key):
                raise SystemExit(f"merge_memories missing-id recovery unexpectedly set {key}: {missing_merge.metadata}")
        assert_operator_metadata(missing_merge.metadata, "missing_merge_memories")
        assert_memory_refusal_handoff(
            missing_merge.metadata,
            "missing_merge_memories",
            source="merge_memories",
            mutation="memory_merge",
        )
        assert_operator_output(missing_merge.output, "missing_merge_memories")

        delete_result = delete_memory({"memory_id": 999999})
        if delete_result.metadata.get("writes_database"):
            raise SystemExit("delete_memory should not mark database writes when nothing was deleted.")
        if delete_result.metadata.get("writes_memory"):
            raise SystemExit("delete_memory should not mark memory writes when nothing was deleted.")
        expected_delete_commands = [
            "show recent memories",
            "search memory for <title or detail>",
            "show memory <correct memory id>",
            "delete memory <correct memory id>",
        ]
        if delete_result.metadata.get("recovery_commands") != expected_delete_commands:
            raise SystemExit(f"delete_memory missing-id recovery order drifted: {delete_result.metadata}")
        for token in (
            "show recent memories",
            "show memory <correct memory id>",
            "delete memory <correct memory id>",
            "normal approval flow",
            "does not reuse or grant approval",
        ):
            if token not in delete_result.output:
                raise SystemExit(f"delete_memory missing-id output missed {token!r}: {delete_result.output}")
        for key in (
            "retry_requires_memory_refresh",
            "retry_requires_corrected_id",
            "retry_requires_fresh_approval",
            "recovery_commands_require_normal_policy",
        ):
            if delete_result.metadata.get(key) is not True:
                raise SystemExit(f"delete_memory missing-id recovery missed {key}: {delete_result.metadata}")
        if delete_result.metadata.get("authorizes_retry") or delete_result.metadata.get("authorizes_memory_mutation"):
            raise SystemExit(f"delete_memory missing-id recovery granted authority: {delete_result.metadata}")
        assert_operator_metadata(delete_result.metadata, "delete_memory_missing")
        assert_memory_refusal_handoff(
            delete_result.metadata,
            "delete_memory_missing",
            source="delete_memory",
            mutation="memory_delete",
        )
        assert_operator_output(delete_result.output, "delete_memory_missing")

        malformed_delete = delete_memory({"memory_id": "memory-abc"})
        if malformed_delete.ok or malformed_delete.metadata.get("raw_memory_id") != "memory-abc":
            raise SystemExit(f"delete_memory should preserve malformed raw memory id metadata: {malformed_delete.metadata}")
        assert_operator_metadata(malformed_delete.metadata, "malformed_delete_memory")
        assert_memory_refusal_handoff(
            malformed_delete.metadata,
            "malformed_delete_memory",
            source="delete_memory",
            mutation="memory_delete",
        )
        path_bad_delete = delete_memory({"memory_id": "/private/tmp/jarvis-delete-memory-id"})
        if path_bad_delete.ok or path_bad_delete.metadata.get("raw_memory_id") != "<local-path>":
            raise SystemExit(f"delete_memory should redact path-shaped bad memory ids: {path_bad_delete.metadata}")
        assert_operator_metadata(path_bad_delete.metadata, "path_bad_delete_memory")
        for raw_memory_id in ("/var/folders/zc/jarvis-delete-memory-id", "/tmp/jarvis-delete-memory-id"):
            temp_path_bad_delete = delete_memory({"memory_id": raw_memory_id})
            if temp_path_bad_delete.ok or temp_path_bad_delete.metadata.get("raw_memory_id") != "<local-path>":
                raise SystemExit(f"delete_memory should redact temp-root path-shaped bad memory ids: {temp_path_bad_delete.metadata}")
            assert_operator_metadata(temp_path_bad_delete.metadata, "temp_path_bad_delete_memory")
        bool_delete = delete_memory({"memory_id": True})
        if bool_delete.ok or "must be a number" not in bool_delete.output:
            raise SystemExit("delete_memory should reject boolean memory ids.")
        if bool_delete.metadata.get("raw_memory_id") != "True":
            raise SystemExit(f"delete_memory should preserve boolean raw memory id metadata: {bool_delete.metadata}")
        assert_operator_metadata(bool_delete.metadata, "bool_delete_memory")

        stats_result = memory_stats({})
        if stats_result.metadata.get("writes_files") or stats_result.metadata.get("queues_approval"):
            raise SystemExit("memory_stats should remain a read-only local diagnostic.")
        assert_review_only_metadata(stats_result.metadata, "memory_stats")
        assert_operator_output(stats_result.output, "memory_stats")
        assert_memory_stats_handoff(stats_result.metadata, "memory_stats")

        with TemporaryDirectory(prefix="jarvis-empty-memory-summary-") as empty_temp:
            empty_runtime = make_temp_runtime(Path(empty_temp))
            empty_tools = make_memory_curator_tools(empty_runtime.store, empty_runtime.vault)
            empty_tree = empty_tools[6]({"limit": True})
            empty_stats = empty_tools[7]({})
            if empty_tree.metadata.get("count") != 0 or empty_tree.metadata.get("writes_files"):
                raise SystemExit(f"empty memory_tree_summary should stay read-only: {empty_tree.metadata}")
            if empty_stats.metadata.get("count") != 0 or empty_stats.metadata.get("writes_files"):
                raise SystemExit(f"empty memory_stats should stay read-only: {empty_stats.metadata}")
            assert_memory_tree_summary_handoff(empty_tree.metadata, "empty_memory_tree_summary", writes=False)
            assert_memory_stats_handoff(empty_stats.metadata, "empty_memory_stats")

        skill_draft = runtime.handle("draft skill from this session called Learning Review Skill Draft")
        if not skill_draft.verified:
            raise SystemExit("Expected draft skill creation to run before memory tree summary.")
        skill_metadata = skill_draft.tool_results[0].metadata
        if skill_metadata.get("human_review_required") is not True or skill_metadata.get("self_modifying_code") is not False:
            raise SystemExit(f"draft skill missed human-review metadata: {skill_metadata}")

        tree_result = memory_tree_summary({"limit": "bad"})
        if tree_result.metadata.get("limit") != 200 or not tree_result.metadata.get("writes_files"):
            raise SystemExit("memory_tree_summary should clamp limits and mark file writes.")
        if not tree_result.metadata.get("writes_notes") or tree_result.metadata.get("writes_database"):
            raise SystemExit("memory_tree_summary should mark note writes without database writes.")
        if tree_result.metadata.get("draft_skill_count", 0) < 1:
            raise SystemExit(f"memory_tree_summary should count draft skills: {tree_result.metadata}")
        if tree_result.metadata.get("human_review_required_for_skill_promotion") is not True or tree_result.metadata.get("self_modifying_code") is not False:
            raise SystemExit(f"memory_tree_summary missed review boundary metadata: {tree_result.metadata}")
        tree_path = Path(tree_result.metadata["path"])
        tree_text = tree_path.read_text(encoding="utf-8")
        assert_contains(
            tree_text,
            [
                "Reviewable Learning Surfaces",
                "Skill drafts are review artifacts",
                "Promote a skill only after the operator reviews",
                "Learning Review Skill Draft",
                "human-review-required",
            ],
            "memory_tree_summary note",
        )
        assert_operator_metadata(tree_result.metadata, "memory_tree_summary")
        assert_operator_output(tree_result.output, "memory_tree_summary")
        assert_vault_relative_receipt(tree_result, root, "Memory Tree/", "memory_tree_summary")
        assert_memory_tree_summary_handoff(tree_result.metadata, "memory_tree_summary", writes=True)


if __name__ == "__main__":
    main()
