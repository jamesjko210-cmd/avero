from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from typing import Any

from jarvis_v2.memory.store import (
    MAX_SELF_KNOWLEDGE_DECISIONS,
    MAX_SELF_KNOWLEDGE_GOALS,
    MAX_SELF_KNOWLEDGE_PEOPLE,
    MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL,
    MAX_SELF_KNOWLEDGE_TASKS,
    MemoryStore,
)
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.conversation import (
    MAX_CHAT_CONTEXT_OUTPUT_CHARS,
    MAX_CHAT_PROMPT_PREVIEW_OUTPUT_CHARS,
)


def _seed_structured_state(store: MemoryStore, *, extra: bool = False) -> None:
    now = "2026-07-13T00:00:00Z"
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        goal_count = MAX_SELF_KNOWLEDGE_GOALS + (2 if extra else 0)
        for index in range(goal_count):
            goal = conn.execute(
                "INSERT INTO goals(title, purpose, horizon, status, created_at, updated_at) VALUES (?, ?, ?, 'active', ?, ?)",
                (f"Goal {index}", f"Purpose {index}", "this quarter", now, now),
            )
            for step in range(MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL + (2 if extra else 0)):
                conn.execute(
                    "INSERT INTO goal_steps(goal_id, body, status, created_at, updated_at) VALUES (?, ?, 'open', ?, ?)",
                    (int(goal.lastrowid), f"Step {index}.{step}", now, now),
                )
        conn.execute(
            "INSERT INTO goals(title, purpose, horizon, status, created_at, updated_at) VALUES ('Finished secret goal', '', '', 'done', ?, ?)",
            (now, now),
        )
        for index in range(MAX_SELF_KNOWLEDGE_TASKS + (2 if extra else 0)):
            priority = "high" if index == 1 else "normal"
            conn.execute(
                "INSERT INTO tasks(body, source, due, priority, status, created_at, updated_at) VALUES (?, ?, ?, ?, 'open', ?, ?)",
                (f"Task {index}", f"private-source-{index}", "2026-07-20" if index == 1 else "", priority, now, now),
            )
        conn.execute(
            "INSERT INTO tasks(body, source, due, priority, status, created_at, updated_at) VALUES ('Completed secret task', 'private', '', 'high', 'done', ?, ?)",
            (now, now),
        )
        decision_count = MAX_SELF_KNOWLEDGE_DECISIONS + (2 if extra else 0)
        for index in range(decision_count):
            conn.execute(
                "INSERT INTO decisions(title, rationale, impact, status, revision, created_at, updated_at) VALUES (?, ?, ?, 'active', 1, ?, ?)",
                (f"Decision {index}", f"Reason {index}", f"Impact {index}", now, now),
            )
        conn.execute(
            "INSERT INTO decisions(title, rationale, impact, status, revision, created_at, updated_at) VALUES ('Superseded secret decision', '', '', 'superseded', 1, ?, ?)",
            (now, now),
        )
        people_count = MAX_SELF_KNOWLEDGE_PEOPLE + (2 if extra else 0)
        for index in range(people_count):
            conn.execute(
                "INSERT INTO people(name, relation, notes, last_contact_at, revision, created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
                (f"Person {index}", f"Relation {index}", f"PRIVATE PERSON NOTE {index}", now, now, now),
            )


def test_snapshot_is_bounded_and_excludes_sensitive_fields() -> None:
    with TemporaryDirectory(prefix="jarvis-self-knowledge-store-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        _seed_structured_state(store, extra=True)
        snapshot = store.read_self_knowledge_snapshot()
        if len(snapshot.goals) != MAX_SELF_KNOWLEDGE_GOALS:
            raise SystemExit("self-knowledge goals were not strictly bounded")
        if len(snapshot.goal_steps) != MAX_SELF_KNOWLEDGE_GOALS * MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL:
            raise SystemExit("self-knowledge goal steps were not bounded per selected goal")
        if len(snapshot.tasks) != MAX_SELF_KNOWLEDGE_TASKS:
            raise SystemExit("self-knowledge tasks were not strictly bounded")
        if len(snapshot.decisions) != MAX_SELF_KNOWLEDGE_DECISIONS:
            raise SystemExit("self-knowledge decisions were not strictly bounded")
        if len(snapshot.people) != MAX_SELF_KNOWLEDGE_PEOPLE:
            raise SystemExit("self-knowledge people were not strictly bounded")
        if set(snapshot.truncated_sources) != {"goals", "goal_steps", "tasks", "decisions", "people"}:
            raise SystemExit(f"self-knowledge snapshot hid cap truncation: {snapshot.truncated_sources}")
        if any(str(row["status"]).lower() != "active" for row in snapshot.goals):
            raise SystemExit("self-knowledge snapshot included a non-active goal")
        if any(str(row["status"]).lower() != "open" for row in snapshot.goal_steps + snapshot.tasks):
            raise SystemExit("self-knowledge snapshot included a closed step or task")
        if any(str(row["status"]).lower() != "active" for row in snapshot.decisions):
            raise SystemExit("self-knowledge snapshot included a non-active decision")
        if any("source" in row.keys() or "completed_at" in row.keys() for row in snapshot.tasks):
            raise SystemExit("self-knowledge task rows exposed provenance or completion history")
        if any("notes" in row.keys() for row in snapshot.people):
            raise SystemExit("self-knowledge people rows exposed private notes")


def test_snapshot_prioritizes_recently_updated_goals() -> None:
    with TemporaryDirectory(prefix="jarvis-self-knowledge-goal-freshness-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for index in range(MAX_SELF_KNOWLEDGE_GOALS + 1):
                created_at = f"2026-07-13T00:00:0{index}Z"
                conn.execute(
                    "INSERT INTO goals(title, purpose, horizon, status, created_at, updated_at) "
                    "VALUES (?, '', '', 'active', ?, ?)",
                    (f"Goal {index}", created_at, created_at),
                )
            oldest_goal_id = 1
            recent_at = "2026-07-14T00:00:00Z"
            conn.execute(
                "UPDATE goals SET updated_at = ? WHERE id = ?",
                (recent_at, oldest_goal_id),
            )
            conn.execute(
                "INSERT INTO goal_steps(goal_id, body, status, created_at, updated_at) "
                "VALUES (?, 'Recently resumed next step', 'open', ?, ?)",
                (oldest_goal_id, recent_at, recent_at),
            )

        snapshot = store.read_self_knowledge_snapshot()
        goal_titles = [str(row["title"]) for row in snapshot.goals]
        step_bodies = [str(row["body"]) for row in snapshot.goal_steps]
        if "Goal 0" not in goal_titles or "Goal 1" in goal_titles:
            raise SystemExit(
                "self-knowledge goals did not prioritize the most recently updated active goal"
            )
        if "Recently resumed next step" not in step_bodies:
            raise SystemExit(
                "self-knowledge omitted the open next step for the recently updated goal"
            )


class _InterceptConnection:
    def __init__(self, conn: Any, first_read: Event, writer_done: Event):
        self._conn = conn
        self._first_read = first_read
        self._writer_done = writer_done
        self._intercepted = False

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._conn.__exit__(exc_type, exc_value, traceback)

    def execute(self, sql: str, parameters: Any = ()):
        cursor = self._conn.execute(sql, parameters)
        normalized = " ".join(sql.lower().split())
        if (
            not self._intercepted
            and "from goals" in normalized
            and "where lower(status) = 'active'" in normalized
            and "order by updated_at desc, id desc" in normalized
        ):
            self._intercepted = True
            self._first_read.set()
            if not self._writer_done.wait(timeout=10):
                raise TimeoutError("concurrent self-knowledge writer did not finish")
        return cursor


def test_snapshot_is_one_transaction_during_concurrent_write() -> None:
    with TemporaryDirectory(prefix="jarvis-self-knowledge-atomic-") as temp:
        db_path = Path(temp) / "memory.sqlite"
        store = MemoryStore(db_path)
        store.init()
        now = "2026-07-13T00:00:00Z"
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO goals(title, purpose, horizon, status, created_at, updated_at) VALUES ('Before goal', '', '', 'active', ?, ?)",
                (now, now),
            )

        first_read = Event()
        writer_done = Event()
        original_connect = store.connect

        def intercepted_connect():
            return _InterceptConnection(original_connect(), first_read, writer_done)

        store.connect = intercepted_connect  # type: ignore[method-assign]
        writer_store = MemoryStore(db_path)

        def mutate_after_first_read() -> None:
            if not first_read.wait(timeout=10):
                raise TimeoutError("self-knowledge reader did not reach first query")
            with writer_store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("UPDATE goals SET status = 'done', updated_at = ? WHERE title = 'Before goal'", (now,))
                conn.execute(
                    "INSERT INTO tasks(body, source, due, priority, status, created_at, updated_at) VALUES ('After task', 'smoke', '', 'normal', 'open', ?, ?)",
                    (now, now),
                )
            writer_done.set()

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                writer = pool.submit(mutate_after_first_read)
                captured = store.read_self_knowledge_snapshot()
                writer.result(timeout=15)
        finally:
            store.connect = original_connect  # type: ignore[method-assign]
        if [row["title"] for row in captured.goals] != ["Before goal"] or captured.tasks:
            raise SystemExit("self-knowledge snapshot mixed rows from before and after one mutation")
        refreshed = store.read_self_knowledge_snapshot()
        if refreshed.goals or [row["body"] for row in refreshed.tasks] != ["After task"]:
            raise SystemExit("refreshed self-knowledge snapshot missed the committed mutation")


class _RaisingRows:
    def __iter__(self):
        yield {"id": 1, "title": "Safe title /\x55sers/example/private\n## injected"}
        raise RuntimeError("iterator failure /\x55sers/example/secret")


def test_explicit_chat_context_is_private_bounded_and_truthful() -> None:
    with TemporaryDirectory(prefix="jarvis-self-knowledge-chat-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_structured_state(runtime.store)
        with runtime.store.connect() as conn:
            conn.execute(
                "INSERT INTO person_interactions(person_id, summary, happened_at, created_at) VALUES (1, 'PRIVATE INTERACTION', '2026-07-12', '2026-07-12')"
            )

        before = len(runtime.store.list_pending_approvals(limit=100))
        result = runtime.handle("chat context: what do you know about me")
        after = len(runtime.store.list_pending_approvals(limit=100))
        if not result.verified or len(result.tool_results) != 1 or result.tool_results[0].tool_name != "chat_context":
            raise SystemExit(f"explicit self-knowledge context route failed: {result.response}")
        if before != after:
            raise SystemExit("read-only self-knowledge context unexpectedly queued approval")
        metadata = result.tool_results[0].metadata
        if metadata.get("reads_private_data") is not True or metadata.get("reads_personal_data") is not True:
            raise SystemExit(f"self-knowledge context hid its private reads: {metadata}")
        if any(metadata.get(key) is not False for key in ("calls_model", "executes_tools", "queues_approval", "writes_memory", "controls_computer")):
            raise SystemExit(f"self-knowledge context crossed a read-only boundary: {metadata}")
        if metadata.get("self_knowledge_state") != "ok":
            raise SystemExit(f"self-knowledge context did not report verified state: {metadata}")
        for expected in (
            "Current goals and next steps",
            "Goal 0",
            "Open commitments",
            "Task 0",
            "Active decisions",
            "Decision 0",
            "Relationships on record",
            "Person 0",
            "does not send it to a model",
        ):
            if expected not in result.response:
                raise SystemExit(f"self-knowledge context missed {expected!r}: {result.response}")
        for forbidden in ("PRIVATE PERSON NOTE", "PRIVATE INTERACTION", "private-source", "Completed secret", "Superseded secret"):
            if forbidden in result.response:
                raise SystemExit(f"self-knowledge context exposed excluded content {forbidden!r}")

        for unrelated_prompt in (
            "explain photosynthesis",
            "explain virtual memory",
            "why do people dream",
            "write a profile of a volcano",
            "what do you know about the operator Bond",
            "explain personal context windows in language models",
        ):
            generic = runtime.registry.get("chat_context").handler(
                {"prompt": unrelated_prompt, "limit": 5}
            )
            if generic.metadata.get("self_knowledge_state") != "not_requested" or "Local personal state" in generic.output:
                raise SystemExit(f"unrelated prompt read broader personal state: {unrelated_prompt!r}")
        korean = runtime.registry.get("chat_context").handler(
            {"prompt": "내 목표와 할 일을 보여줘", "limit": 5}
        )
        if korean.metadata.get("self_knowledge_state") != "ok" or "Current goals and next steps" not in korean.output:
            raise SystemExit("explicit Korean personal-state preview was not recognized")
        for personal_prompt in (
            "what are my current goals?",
            "what do you know about me right now?",
            "tell me about myself",
            "show my open commitments",
        ):
            personal = runtime.registry.get("chat_context").handler(
                {"prompt": personal_prompt, "limit": 5}
            )
            if personal.metadata.get("self_knowledge_state") != "ok":
                raise SystemExit(f"explicit personal-state request was missed: {personal_prompt!r}")

        for natural_request in (
            "what do you know about me?",
            "tell me about myself",
            "what are my current goals?",
            "show my open commitments",
            "나에 대해 뭘 알아?",
            "나에 대해 알려줘",
            "저에 대해 알려주세요",
            "자비스가 나에 대해 뭘 기억해?",
        ):
            natural = runtime.handle(natural_request)
            if (
                not natural.verified
                or len(natural.tool_results) != 1
                or natural.tool_results[0].tool_name != "chat_context"
            ):
                raise SystemExit(f"natural self-knowledge route failed: {natural_request!r} -> {natural}")
            natural_metadata = natural.tool_results[0].metadata
            if (
                natural_metadata.get("self_knowledge_state") != "ok"
                or natural_metadata.get("calls_model") is not False
                or natural_metadata.get("executes_tools") is not False
                or natural_metadata.get("queues_approval") is not False
            ):
                raise SystemExit(
                    f"natural self-knowledge request crossed a read-only boundary: {natural_request!r} -> {natural_metadata}"
                )
            for section in (
                "Current goals and next steps",
                "Open commitments",
                "Active decisions",
                "Relationships on record",
            ):
                if section not in natural.response:
                    raise SystemExit(
                        f"natural self-knowledge request missed {section!r}: {natural_request!r} -> {natural.response}"
                    )

        for mutating_or_compound_request in (
            "나에 대해 메모해줘",
            "나에 대해 기억해줘: 나는 커피를 좋아해",
            "나에 대해 알려줘 그리고 프로필을 수정해줘",
        ):
            plan = runtime.planner.plan(mutating_or_compound_request)
            if any(action.tool_name == "chat_context" for action in plan.actions):
                raise SystemExit(
                    "Korean self-knowledge read route captured a mutation or compound request: "
                    f"{mutating_or_compound_request!r} -> {plan!r}"
                )

        original_search_memories = runtime.store.search_memories
        original_preferences = runtime.store.list_preferences
        original_skills = runtime.store.search_active_skills
        original_messages = runtime.store.recent_messages
        original_profile = runtime.vault.read_profile
        long_text = "bounded saturation " * 80
        runtime.store.search_memories = lambda query, limit=5: [
            {"id": index + 1, "category": "facts", "title": long_text, "body": long_text}
            for index in range(20)
        ]  # type: ignore[method-assign]
        runtime.store.list_preferences = lambda status="active", limit=8: [
            {"category": "style", "key": long_text, "value": long_text}
            for _ in range(8)
        ]  # type: ignore[method-assign]
        runtime.store.search_active_skills = lambda query, limit=3: [
            {"id": index + 1, "name": long_text, "trigger": long_text}
            for index in range(3)
        ]  # type: ignore[method-assign]
        runtime.store.recent_messages = lambda limit=8, session_id=None: [
            {
                "id": index + 1,
                "session_id": "bounded",
                "role": "assistant",
                "content": long_text,
                "created_at": "2026-07-13T00:00:00Z",
                "metadata": "{}",
            }
            for index in range(8)
        ]  # type: ignore[method-assign]
        runtime.vault.read_profile = lambda max_chars=900: "# Profile\n" + "\n".join(
            long_text for _ in range(8)
        )  # type: ignore[method-assign]
        try:
            saturated = runtime.registry.get("chat_context").handler(
                {"prompt": "what do you know about me", "limit": 20}
            )
            saturated_prompt_preview = runtime.registry.get("chat_prompt_preview").handler(
                {"prompt": "what do you know about me", "limit": 20}
            )
        finally:
            runtime.store.search_memories = original_search_memories  # type: ignore[method-assign]
            runtime.store.list_preferences = original_preferences  # type: ignore[method-assign]
            runtime.store.search_active_skills = original_skills  # type: ignore[method-assign]
            runtime.store.recent_messages = original_messages  # type: ignore[method-assign]
            runtime.vault.read_profile = original_profile  # type: ignore[method-assign]
        if saturated.metadata.get("context_output_truncated") is not True:
            raise SystemExit("saturated chat context did not report its global output cap")
        if "Local personal state" not in saturated.output or "Goal 0" not in saturated.output:
            raise SystemExit("global output cap erased requested personal state")
        if "Safety reminder:" not in saturated.output[-500:]:
            raise SystemExit("global chat-context cap erased the mandatory safety footer")
        if "Execution boundary:" not in saturated_prompt_preview.output[-700:]:
            raise SystemExit("global prompt-preview cap erased the mandatory execution footer")

        original_snapshot = runtime.store.read_self_knowledge_snapshot
        maximum_snapshot = SimpleNamespace(
            captured_at="2026-07-13T00:00:00Z",
            goals=tuple(
                {
                    "id": index + 1,
                    "title": "G" * 512,
                    "purpose": "P" * 1024,
                    "horizon": "H" * 160,
                    "status": "active",
                    "updated_at": "2026-07-13T00:00:00Z",
                }
                for index in range(MAX_SELF_KNOWLEDGE_GOALS)
            ),
            goal_steps=tuple(
                {
                    "id": (goal * 10) + step + 1,
                    "goal_id": goal + 1,
                    "body": "S" * 1024,
                    "status": "open",
                    "updated_at": "2026-07-13T00:00:00Z",
                }
                for goal in range(MAX_SELF_KNOWLEDGE_GOALS)
                for step in range(MAX_SELF_KNOWLEDGE_STEPS_PER_GOAL)
            ),
            tasks=tuple(
                {
                    "id": index + 1,
                    "body": "T" * 1024,
                    "due": "2026-07-20",
                    "priority": "normal",
                    "status": "open",
                    "updated_at": "2026-07-13T00:00:00Z",
                }
                for index in range(MAX_SELF_KNOWLEDGE_TASKS)
            ),
            decisions=tuple(
                {
                    "id": index + 1,
                    "title": "D" * 512,
                    "rationale": "R" * 1024,
                    "impact": "I" * 1024,
                    "status": "active",
                    "revision": 1,
                    "updated_at": "2026-07-13T00:00:00Z",
                }
                for index in range(MAX_SELF_KNOWLEDGE_DECISIONS)
            ),
            people=tuple(
                {
                    "id": index + 1,
                    "name": "N" * 512,
                    "relation": "R" * 512,
                    "last_contact_at": "2026-07-12",
                    "revision": 1,
                    "updated_at": "2026-07-13T00:00:00Z",
                }
                for index in range(MAX_SELF_KNOWLEDGE_PEOPLE)
            ),
            truncated_sources=(),
            clipped_sources=(),
        )
        runtime.store.read_self_knowledge_snapshot = lambda: maximum_snapshot  # type: ignore[method-assign]
        try:
            section_saturated = runtime.registry.get("chat_context").handler(
                {"prompt": "what do you know about me", "limit": 5}
            )
        finally:
            runtime.store.read_self_knowledge_snapshot = original_snapshot  # type: ignore[method-assign]
        for required_section in (
            "Current goals and next steps:",
            "Open commitments:",
            "Active decisions:",
            "Relationships on record:",
        ):
            if required_section not in section_saturated.output:
                raise SystemExit(f"per-section saturation erased {required_section!r}")
        if not section_saturated.metadata.get("self_knowledge_display_truncated_sources"):
            raise SystemExit("per-section display truncation was not receipted")

        runtime.store.read_self_knowledge_snapshot = lambda: (_ for _ in ()).throw(
            RuntimeError("database failed /\x55sers/example/private SHOULD_NOT_LEAK")
        )  # type: ignore[method-assign]
        try:
            unavailable = runtime.registry.get("chat_context").handler(
                {"prompt": "what do you know about me", "limit": 5}
            )
        finally:
            runtime.store.read_self_knowledge_snapshot = original_snapshot  # type: ignore[method-assign]
        if unavailable.metadata.get("self_knowledge_state") != "unavailable":
            raise SystemExit(f"self-knowledge outage was not explicit: {unavailable.metadata}")
        if "not assumed empty" not in unavailable.output or "No active goals recorded" in unavailable.output:
            raise SystemExit(f"self-knowledge outage was misreported as empty: {unavailable.output}")
        if "/\x55sers/operator" in unavailable.output or "SHOULD_NOT_LEAK" in str(unavailable.metadata):
            raise SystemExit("self-knowledge outage leaked exception detail")

        hostile = SimpleNamespace(
            captured_at="2026-07-13T00:00:00Z",
            goals=_RaisingRows(),
            goal_steps="not-a-row-container",
            tasks=[{"id": 2, "body": "Task\n## injected /private/tmp/secret", "due": "", "priority": "normal"}] * 50,
            decisions=[{"id": False, "title": "bad"}],
            people={"id": 1, "name": "not-a-container"},
        )
        runtime.store.read_self_knowledge_snapshot = lambda: hostile  # type: ignore[method-assign]
        try:
            degraded = runtime.registry.get("chat_context").handler(
                {"prompt": "what do you know about me", "limit": 20}
            )
        finally:
            runtime.store.read_self_knowledge_snapshot = original_snapshot  # type: ignore[method-assign]
        if degraded.metadata.get("self_knowledge_state") != "degraded":
            raise SystemExit(f"hostile self-knowledge rows were not marked degraded: {degraded.metadata}")
        if len(degraded.output) > MAX_CHAT_CONTEXT_OUTPUT_CHARS:
            raise SystemExit("hostile self-knowledge output exceeded the hard cap")
        if "/\x55sers/operator" in degraded.output or "/private/tmp" in degraded.output or "iterator failure" in degraded.output:
            raise SystemExit(f"hostile self-knowledge details leaked: {degraded.output}")
        prompt_preview = runtime.registry.get("chat_prompt_preview").handler(
            {"prompt": "what do you know about me", "limit": 20}
        )
        if len(prompt_preview.output) > MAX_CHAT_PROMPT_PREVIEW_OUTPUT_CHARS:
            raise SystemExit("hostile chat prompt preview exceeded the hard cap")
        if prompt_preview.metadata.get("reads_private_data") is not True:
            raise SystemExit("chat prompt preview hid its personal-context read")

        original_profile = runtime.vault.read_profile
        runtime.vault.read_profile = lambda max_chars=900: (
            "# Profile\nStyle\u202eELIFORP \u2066 \x1b[31m "
            "／Ｕｓｅｒｓ／operator／secret ~/another-secret file:///\x55sers/example/third-secret"
        )  # type: ignore[method-assign]
        try:
            controls = runtime.registry.get("chat_context").handler(
                {"prompt": "show my profile \u202e ／Ｕｓｅｒｓ／operator／prompt-secret", "limit": 5}
            )
        finally:
            runtime.vault.read_profile = original_profile  # type: ignore[method-assign]
        for forbidden in ("\u202e", "\u2066", "\x1b", "／Ｕｓｅｒｓ", "/\x55sers/operator", "~/"):
            if forbidden in controls.output or forbidden in str(controls.metadata):
                raise SystemExit(f"generic context sanitizer leaked {forbidden!r}")

        original_search_memories = runtime.store.search_memories
        original_preferences = runtime.store.list_preferences
        original_skills = runtime.store.search_active_skills
        variants = (
            (item for item in []),
            (item for item in [{"id": 1, "category": "facts", "title": "ok", "body": "ok"}]),
            "not-a-row-container",
            {"id": 1},
            _RaisingRows(),
        )
        try:
            for variant in variants:
                runtime.store.search_memories = lambda query, limit=5, value=variant: value  # type: ignore[method-assign]
                generator_result = runtime.registry.get("chat_context").handler(
                    {"prompt": "ordinary context query", "limit": 5}
                )
                if not generator_result.ok:
                    raise SystemExit("chat context rejected a defensive memory container")
            runtime.store.list_preferences = lambda status="active", limit=8: (
                row for row in [{"category": "style", "key": "tone", "value": "direct"}]
            )  # type: ignore[method-assign]
            runtime.store.search_active_skills = lambda query, limit=3: (
                row for row in [{"id": 1, "name": "Skill", "trigger": "context"}]
            )  # type: ignore[method-assign]
            generator_result = runtime.registry.get("chat_prompt_preview").handler(
                {"prompt": "ordinary context query", "limit": 5}
            )
            if not generator_result.ok:
                raise SystemExit("chat prompt preview rejected defensive preference/skill generators")
        finally:
            runtime.store.search_memories = original_search_memories  # type: ignore[method-assign]
            runtime.store.list_preferences = original_preferences  # type: ignore[method-assign]
            runtime.store.search_active_skills = original_skills  # type: ignore[method-assign]


def test_snapshot_clips_oversized_fields_before_returning() -> None:
    with TemporaryDirectory(prefix="jarvis-self-knowledge-clipped-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        huge = "X" * 1_500_000
        now = "2026-07-13T00:00:00Z"
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            goal = conn.execute(
                "INSERT INTO goals(title, purpose, horizon, status, created_at, updated_at) VALUES (?, ?, ?, 'active', ?, ?)",
                (huge, huge, huge, now, now),
            )
            conn.execute(
                "INSERT INTO goal_steps(goal_id, body, status, created_at, updated_at) VALUES (?, ?, 'open', ?, ?)",
                (int(goal.lastrowid), huge, now, now),
            )
            conn.execute(
                "INSERT INTO tasks(body, source, due, priority, status, created_at, updated_at) VALUES (?, 'smoke', '', 'normal', 'open', ?, ?)",
                (huge, now, now),
            )
            conn.execute(
                "INSERT INTO decisions(title, rationale, impact, status, revision, created_at, updated_at) VALUES (?, ?, ?, 'active', 1, ?, ?)",
                (huge, huge, huge, now, now),
            )
            conn.execute(
                "INSERT INTO people(name, relation, notes, last_contact_at, revision, created_at, updated_at) VALUES (?, ?, '', ?, 1, ?, ?)",
                (huge, huge, huge, now, now),
            )
        snapshot = store.read_self_knowledge_snapshot()
        if set(snapshot.clipped_sources) != {"goals", "goal_steps", "tasks", "decisions", "people"}:
            raise SystemExit(f"oversized self-knowledge fields were not receipted: {snapshot.clipped_sources}")
        returned_text = [
            snapshot.goals[0]["title"],
            snapshot.goals[0]["purpose"],
            snapshot.goal_steps[0]["body"],
            snapshot.tasks[0]["body"],
            snapshot.decisions[0]["rationale"],
            snapshot.people[0]["name"],
        ]
        if max(map(len, returned_text)) > 1024:
            raise SystemExit("self-knowledge snapshot returned an unbounded text field")


def main() -> None:
    test_snapshot_is_bounded_and_excludes_sensitive_fields()
    test_snapshot_prioritizes_recently_updated_goals()
    test_snapshot_is_one_transaction_during_concurrent_write()
    test_snapshot_clips_oversized_fields_before_returning()
    test_explicit_chat_context_is_private_bounded_and_truthful()
    print("Self-knowledge context smoke passed")


if __name__ == "__main__":
    main()
