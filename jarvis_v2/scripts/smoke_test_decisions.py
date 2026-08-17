from __future__ import annotations

import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools.decisions import _decision_handoff_metadata, _metadata_bool


class HostileRow:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def keys(self):
        raise RuntimeError(self.marker)

    def __getitem__(self, key: str):
        raise RuntimeError(f"{self.marker}:{key}")

    def __str__(self) -> str:
        raise RuntimeError(self.marker)

    def __repr__(self) -> str:
        return f"<hostile-row {self.marker}>"


def assert_write_receipt(result, *, root: Path, label: str) -> None:
    metadata = result.metadata
    path_display = metadata.get("path_display")
    decision_id = metadata.get("decision_id")
    if "path" in metadata:
        raise SystemExit(f"{label} exposed a raw saved path: {metadata}")
    if (
        type(decision_id) is not int
        or decision_id < 1
        or path_display != f"Decisions/decision-{decision_id}.md"
    ):
        raise SystemExit(f"{label} missed safe Decisions path_display: {metadata}")
    if not any((root / "Decisions").glob(f"{decision_id:04d} *.md")):
        raise SystemExit(f"{label} did not publish a decision note inside the vault.")
    if f"Saved note: {path_display}" not in result.output:
        raise SystemExit(f"{label} missed safe saved-note receipt: {result.output}")
    if any(fragment in result.output for fragment in (str(root), "/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
        raise SystemExit(f"{label} leaked a local path in output: {result.output}")


def assert_no_local_paths(value, label: str) -> None:
    text = repr(value)
    if "/private/" in text or "/\x55sers/" in text or "/var/folders/" in text or "/tmp/" in text:
        raise SystemExit(f"{label} handoff should not expose raw local paths: {text[:1000]}")


def assert_decision_exact_metadata_bool() -> None:
    for value in ("true", "false", "yes", "0", 1, 0, [], ["value"], None, object()):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"decision metadata bool should reject malformed value {value!r}")
    if _metadata_bool(True) is not True:
        raise SystemExit("decision metadata bool should preserve exact True")
    if _metadata_bool(False) is not False:
        raise SystemExit("decision metadata bool should preserve exact False")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("decision metadata bool should honor explicit default for malformed values")


def assert_decision_malformed_handoff_flags() -> None:
    handoff = {
        "source": "get_decision",
        "state_changed": "false",
        "changed": [],
        "content_in_handoff": "true",
        "next_commands": ["decisions"],
        "boundaries": {"read_only": True},
    }
    metadata = _decision_handoff_metadata("decision_read_handoff", handoff)
    if metadata.get("state_changed") is not False or metadata.get("decision_read_state_changed") is not False:
        raise SystemExit(f"malformed decision state_changed should fail closed: {metadata}")
    if metadata.get("content_in_handoff") is not False or metadata.get("decision_read_content_in_handoff") is not False:
        raise SystemExit(f"malformed decision content_in_handoff should fail closed: {metadata}")


def assert_planner_routes_decision_phone_aliases() -> None:
    planner = RuleBasedPlanner()
    for text in (
        "decisions please",
        "show decisions",
        "show latest decisions",
        "decision list please",
        "show decision list",
        "decision log please",
        "what decisions did we make",
        # Real gap found live 2026-07-09: "what decisions have I made" fell
        # through to chat -- only the "did we make" phrasing was recognized.
        "what decisions have I made",
        "what decisions have we made",
        # Real gap found live 2026-07-10, same class as the round-39 "list my
        # open tasks" bug, surfaced via a WS4 mixed-conversation
        # remeasurement: "list decisions" and "my decisions" both worked
        # separately, but combining "my" with a "recent" qualifier
        # ("list my recent decisions") fell through to chat because the
        # original regex only had one qualifier slot shared between the verb
        # and the "my" position, not room for both as separate words.
        "list my recent decisions",
        "show my recent decisions",
        "my recent decisions",
        "show my latest decisions",
    ):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("list_decisions", {"status": "active"})]:
            raise SystemExit(f"planner missed decision list alias {text!r}: {plan.actions}")
    for text in ("all decisions please", "show all decisions please"):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [("list_decisions", {"status": "all"})]:
            raise SystemExit(f"planner missed all-decisions alias {text!r}: {plan.actions}")
    # Real gap found live 2026-07-09: "record a decision to use postgres" /
    # "add a decision to use postgres" / "log a decision to use postgres" fell
    # through to chat -- decision_match only recognized the bare verbs
    # "record"/"save"/"log" directly followed by "decision " with no article
    # and no "to" lead-in before the title, and "add" wasn't a recognized verb.
    for text in (
        "record a decision to use postgres",
        "add a decision to use postgres",
        "log a decision to use postgres",
        "save a decision to use postgres",
    ):
        plan = planner.plan(text)
        if [(action.tool_name, action.args) for action in plan.actions] != [
            ("record_decision", {"title": "use postgres", "rationale": "", "impact": ""})
        ]:
            raise SystemExit(f"planner missed natural decision-record phrasing {text!r}: {plan.actions}")
    # Real gap found live 2026-07-10, compound-sentence clause-bleed class
    # (same as this session's round-33 add_task/create_reminder/find_contact
    # fixes): "record decision use postgres and then show my goals" would
    # have written a decision literally titled "use postgres and then show
    # my goals" instead of "use postgres", silently dropping the second
    # intent.
    compound_decision_plan = planner.plan("record decision use postgres and then show my goals")
    if [(action.tool_name, action.args) for action in compound_decision_plan.actions] != [
        ("record_decision", {"title": "use postgres", "rationale": "", "impact": ""})
    ]:
        raise SystemExit(f"planner should stop decision title at a compound-sentence boundary: {compound_decision_plan.actions}")


def assert_runtime_routes_decision_count_aliases() -> None:
    with TemporaryDirectory(prefix="jarvis-decision-count-alias-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for text in (
            "decision count",
            "decisions count",
            "count decisions",
            "how many decisions do i have",
            "결정 몇 개",
            "의사결정 개수",
        ):
            result = runtime.handle(text)
            planned = (result.metadata or {}).get("runtime_trace", {}).get("planned_actions") or []
            if [action.get("tool_name") for action in planned] != ["list_decisions"]:
                raise SystemExit(f"runtime missed decision-count alias {text!r}: {planned} / {result.response!r}")
            if len(result.tool_results) != 1 or result.tool_results[0].tool_name != "list_decisions":
                raise SystemExit(f"decision-count alias should execute one list_decisions tool for {text!r}: {result.tool_results}")
            metadata = result.tool_results[0].metadata
            if "Count: 0" not in result.response or metadata.get("total_decisions") != 0:
                raise SystemExit(f"empty decision-count alias should answer with zero count for {text!r}: {result.response!r} / {metadata}")
            for key in (
                "writes_files",
                "writes_memory",
                "writes_notes",
                "queues_approval",
                "controls_computer",
                "reads_private_data",
                "authorizes_execution",
                "authorizes_completion_claim",
                "approval_granted",
            ):
                if metadata.get(key):
                    raise SystemExit(f"decision-count alias unexpectedly set {key} for {text!r}: {metadata}")
        if runtime.store.list_decisions(status=None, limit=100):
            raise SystemExit("decision count aliases must not create or mutate decisions.")
        if runtime.store.list_memories(limit=100):
            raise SystemExit("decision count aliases must not create memory rows.")


def assert_decision_projection_concurrency() -> None:
    def run_thread(
        target,
        errors: list[BaseException],
        results: list[object] | None = None,
    ) -> threading.Thread:
        def guarded() -> None:
            try:
                result = target()
                if results is not None:
                    results.append(result)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=guarded)
        thread.start()
        return thread

    def assert_finished(thread: threading.Thread, label: str) -> None:
        thread.join(timeout=5)
        if thread.is_alive():
            raise SystemExit(f"{label} did not finish.")

    with TemporaryDirectory(prefix="jarvis-decision-projection-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        record = runtime.registry.get("record_decision").handler
        set_status = runtime.registry.get("set_decision_status").handler
        original_write = runtime.vault.write_decision_with_evidence
        older_write_entered = threading.Event()
        release_older_write = threading.Event()
        errors: list[BaseException] = []
        record_results: list[object] = []
        first_status_results: list[object] = []

        def delayed_initial_write(decision, *, store_identity: str):
            if decision["status"] == "active":
                older_write_entered.set()
                if not release_older_write.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release initial decision projection")
            return original_write(decision, store_identity=store_identity)

        runtime.vault.write_decision_with_evidence = delayed_initial_write
        record_thread = run_thread(
            lambda: record({"title": "Projection race", "rationale": "older", "impact": "test"}),
            errors,
            record_results,
        )
        if not older_write_entered.wait(timeout=5):
            raise SystemExit("record_decision did not reach the delayed projection write.")
        status_thread = run_thread(
            lambda: set_status({"decision_id": 1, "status": "retired"}),
            errors,
            first_status_results,
        )
        status_thread.join(timeout=0.2)
        release_older_write.set()
        assert_finished(record_thread, "delayed record_decision")
        assert_finished(status_thread, "record-vs-status update")
        if errors:
            raise SystemExit(f"record-vs-status projection race raised: {errors!r}")
        row = runtime.store.get_decision(1)
        note = next((runtime.vault.root_path / "Decisions").glob("0001 *.md")).read_text()
        if row is None or row["status"] != "retired" or "status: retired" not in note:
            raise SystemExit("a delayed initial writer left the decision note behind SQLite.")
        if (
            len(record_results) != 1
            or len(first_status_results) != 1
            or record_results[0].metadata.get("status") != "retired"
            or first_status_results[0].metadata.get("status") != "retired"
            or record_results[0].metadata.get("path") is not None
            or first_status_results[0].metadata.get("path") is not None
            or record_results[0].metadata.get("path_display") != "Decisions/decision-1.md"
            or first_status_results[0].metadata.get("path_display") != "Decisions/decision-1.md"
        ):
            raise SystemExit(
                "record-vs-status receipts diverged from the current fenced projection."
            )

        runtime.vault.write_decision_with_evidence = original_write
        older_write_entered.clear()
        release_older_write.clear()

        def delayed_status_write(decision, *, store_identity: str):
            if decision["status"] == "superseded":
                older_write_entered.set()
                if not release_older_write.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release older status projection")
            return original_write(decision, store_identity=store_identity)

        runtime.vault.write_decision_with_evidence = delayed_status_write
        older_status_results: list[object] = []
        newer_status_results: list[object] = []
        older_status_thread = run_thread(
            lambda: set_status({"decision_id": 1, "status": "superseded"}),
            errors,
            older_status_results,
        )
        if not older_write_entered.wait(timeout=5):
            raise SystemExit("set_decision_status did not reach the delayed projection write.")
        newer_status_thread = run_thread(
            lambda: set_status({"decision_id": 1, "status": "retired"}),
            errors,
            newer_status_results,
        )
        newer_status_thread.join(timeout=0.2)
        release_older_write.set()
        assert_finished(older_status_thread, "older status update")
        assert_finished(newer_status_thread, "newer status update")
        if errors:
            raise SystemExit(f"status-vs-status projection race raised: {errors!r}")
        row = runtime.store.get_decision(1)
        note = next((runtime.vault.root_path / "Decisions").glob("0001 *.md")).read_text()
        if row is None or row["status"] != "retired" or "status: retired" not in note:
            raise SystemExit("a delayed older status writer left the decision note behind SQLite.")
        if (
            len(older_status_results) != 1
            or len(newer_status_results) != 1
            or older_status_results[0].ok
            or older_status_results[0].metadata.get("status") != "retired"
            or older_status_results[0].metadata.get("requested_status") != "superseded"
            or older_status_results[0].metadata.get("mutation_superseded") is not True
            or newer_status_results[0].metadata.get("status") != "retired"
            or "Current status: retired" not in older_status_results[0].output
            or "to retired" not in newer_status_results[0].output
            or older_status_results[0].metadata.get("path") is not None
            or newer_status_results[0].metadata.get("path") is not None
            or older_status_results[0].metadata.get("path_display") != "Decisions/decision-1.md"
            or newer_status_results[0].metadata.get("path_display") != "Decisions/decision-1.md"
        ):
            raise SystemExit(
                "concurrent decision receipts did not match their fenced publications."
            )

        runtime.vault.write_decision_with_evidence = original_write
        committed_before_snapshot = threading.Event()
        release_committed_writer = threading.Event()
        original_set_status = runtime.store.set_decision_status_with_projection

        def delayed_after_commit(decision_id: int, requested_status: str):
            updated = original_set_status(decision_id, requested_status)
            if requested_status == "superseded":
                committed_before_snapshot.set()
                if not release_committed_writer.wait(timeout=5):
                    raise RuntimeError("timed out releasing committed decision writer")
            return updated

        runtime.store.set_decision_status_with_projection = delayed_after_commit  # type: ignore[method-assign]
        delayed_results: list[object] = []
        winning_results: list[object] = []
        delayed_thread = run_thread(
            lambda: set_status({"decision_id": 1, "status": "superseded"}),
            errors,
            delayed_results,
        )
        if not committed_before_snapshot.wait(timeout=5):
            raise SystemExit("status update did not reach the post-commit snapshot gap")
        winning_thread = run_thread(
            lambda: set_status({"decision_id": 1, "status": "retired"}),
            errors,
            winning_results,
        )
        assert_finished(winning_thread, "newer post-commit status update")
        release_committed_writer.set()
        assert_finished(delayed_thread, "delayed post-commit status update")
        runtime.store.set_decision_status_with_projection = original_set_status  # type: ignore[method-assign]
        if errors:
            raise SystemExit(f"commit-to-snapshot decision race raised: {errors!r}")
        row = runtime.store.get_decision(1)
        note = next((runtime.vault.root_path / "Decisions").glob("0001 *.md")).read_text()
        if row is None or row["status"] != "retired" or "status: retired" not in note:
            raise SystemExit("post-commit delayed publication left the decision note stale")
        if (
            len(delayed_results) != 1
            or len(winning_results) != 1
            or delayed_results[0].ok
            or delayed_results[0].metadata.get("status") != "retired"
            or delayed_results[0].metadata.get("requested_status") != "superseded"
            or delayed_results[0].metadata.get("mutation_superseded") is not True
            or winning_results[0].metadata.get("status") != "retired"
            or "Current status: retired" not in delayed_results[0].output
            or delayed_results[0].metadata.get("path") is not None
            or winning_results[0].metadata.get("path") is not None
            or delayed_results[0].metadata.get("path_display") != "Decisions/decision-1.md"
            or winning_results[0].metadata.get("path_display") != "Decisions/decision-1.md"
        ):
            raise SystemExit(
                "post-commit receipts did not preserve their revisioned mutation states"
            )


def assert_decision_contract(
    metadata: dict,
    handoff: dict,
    label: str,
    *,
    state_changed: bool,
    changed: list[str],
    content_in_handoff: bool,
) -> None:
    handoff_key = next((key for key, value in metadata.items() if key.endswith("_handoff") and value is handoff), "")
    if not handoff_key:
        raise SystemExit(f"{label} could not discover decision handoff key: {metadata}")
    prefix = handoff_key.removesuffix("_handoff")
    raw_next = handoff.get("next_commands") or []
    if isinstance(raw_next, dict):
        expected_next = [str(value) for value in raw_next.values() if str(value or "").strip()]
    else:
        expected_next = [str(value) for value in raw_next if str(value or "").strip()]
    expected_first = expected_next[0] if expected_next else ""
    for key, expected in (
        ("handoff_ready", True),
        ("ready_for_operator", True),
        ("state_changed", state_changed),
        ("changed", changed),
        ("content_in_handoff", content_in_handoff),
        ("authorizes_execution", False),
        ("authorizes_completion_claim", False),
        ("approval_granted", False),
    ):
        if handoff.get(key) != expected:
            raise SystemExit(f"{label} missed decision handoff {key}={expected}: handoff={handoff}")
        if key != "handoff_ready" and metadata.get(key) != expected:
            raise SystemExit(f"{label} missed decision contract {key}={expected}: metadata={metadata} handoff={handoff}")
    for key, expected in (
        (f"{handoff_key}_ready", True),
        (f"{prefix}_handoff_ready", True),
        (f"{prefix}_ready_for_operator", True),
        (f"{prefix}_state_changed", state_changed),
        (f"{prefix}_changed", changed),
        (f"{prefix}_content_in_handoff", content_in_handoff),
        (f"{prefix}_authorizes_execution", False),
        (f"{prefix}_authorizes_completion_claim", False),
        (f"{prefix}_approval_granted", False),
    ):
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} missed decision prefixed alias {key}={expected}: metadata={metadata}")
    for container, container_label in ((handoff, "handoff"), (metadata, "metadata")):
        if container.get("next_safe_command") != expected_first:
            raise SystemExit(f"{label} {container_label} next_safe_command mismatch: {container}")
        if container.get("next_safe_commands") != expected_next:
            raise SystemExit(f"{label} {container_label} next_safe_commands mismatch: {container}")
        if container.get("next_safe_command_count") != len(expected_next):
            raise SystemExit(f"{label} {container_label} next_safe_command_count mismatch: {container}")
    for key, expected in (
        (f"{prefix}_next_safe_command", expected_first),
        (f"{prefix}_next_safe_commands", expected_next),
        (f"{prefix}_next_safe_command_count", len(expected_next)),
    ):
        if metadata.get(key) != expected:
            raise SystemExit(f"{label} missed decision prefixed safe-command alias {key}: metadata={metadata}")


def assert_decision_mutation_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    mutation: str,
    changed: list[str],
) -> None:
    handoff = metadata.get("decision_mutation_handoff")
    if metadata.get("decision_mutation_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed decision_mutation_handoff readiness: {metadata}")
    if handoff.get("source") != source or handoff.get("mutation") != mutation:
        raise SystemExit(f"{label} mutation handoff missed source/mutation parity: {handoff}")
    if handoff.get("decision_id") != metadata.get("decision_id") or handoff.get("status") != metadata.get("status"):
        raise SystemExit(f"{label} mutation handoff missed decision/status parity: {metadata}")
    assert_decision_contract(metadata, handoff, label, state_changed=True, changed=changed, content_in_handoff=False)
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not False:
        raise SystemExit(f"{label} mutation handoff missed write boundary: {handoff}")
    if metadata.get("writes_files") is not True or boundaries.get("writes_files") is not True:
        raise SystemExit(f"{label} mutation handoff should write files: metadata={metadata} handoff={handoff}")
    if metadata.get("writes_notes") is not True or boundaries.get("writes_notes") is not True:
        raise SystemExit(f"{label} mutation handoff should write notes: metadata={metadata} handoff={handoff}")
    for key in (
        "queues_approval",
        "authorizes_execution",
        "authorizes_completion_claim",
        "approval_granted",
        "requires_approval",
        "controls_computer",
        "external_side_effect",
    ):
        if metadata.get(key) is not False or boundaries.get(key) is not False:
            raise SystemExit(f"{label} mutation handoff should keep {key}=False: metadata={metadata} handoff={handoff}")
    assert_no_local_paths(handoff, label)


def assert_decision_list_handoff(metadata: dict, label: str, *, content_in_handoff: bool) -> None:
    handoff = metadata.get("decision_list_handoff")
    if metadata.get("decision_list_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed decision_list_handoff readiness: {metadata}")
    assert_decision_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=content_in_handoff)
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} list handoff missed read-only boundary: {handoff}")
    for key in ("queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False or boundaries.get(key) is not False:
            raise SystemExit(f"{label} list handoff should keep {key}=False: metadata={metadata} handoff={handoff}")


def assert_decision_read_handoff(metadata: dict, label: str) -> None:
    handoff = metadata.get("decision_read_handoff")
    if metadata.get("decision_read_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} missed decision_read_handoff readiness: {metadata}")
    if handoff.get("source") != "get_decision" or handoff.get("decision_id") != metadata.get("decision_id"):
        raise SystemExit(f"{label} read handoff missed source/decision parity: {handoff}")
    if handoff.get("status") != metadata.get("status"):
        raise SystemExit(f"{label} read handoff missed status parity: {metadata}")
    assert_decision_contract(metadata, handoff, label, state_changed=False, changed=[], content_in_handoff=True)
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} read handoff missed read-only boundary: {handoff}")
    if boundaries.get("writes_files") or boundaries.get("writes_memory") or boundaries.get("writes_notes"):
        raise SystemExit(f"{label} read handoff should stay non-mutating: {handoff}")
    for key in ("queues_approval", "authorizes_execution", "authorizes_completion_claim", "approval_granted"):
        if metadata.get(key) is not False or boundaries.get(key) is not False:
            raise SystemExit(f"{label} read handoff should keep {key}=False: metadata={metadata} handoff={handoff}")


def assert_decision_refusal_handoff(
    metadata: dict,
    label: str,
    *,
    source: str,
    mutation: str,
    reason: str,
    read_only: bool,
) -> None:
    handoff = metadata.get("decision_refusal_handoff")
    if metadata.get("decision_refusal_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should emit a ready decision_refusal_handoff: {metadata}")
    if handoff.get("source") != source or handoff.get("ready_for_operator") is not True:
        raise SystemExit(f"{label} has wrong decision refusal source/readiness: {handoff}")
    if handoff.get("mutation") != mutation or handoff.get("reason") != reason or handoff.get("refused") is not True:
        raise SystemExit(f"{label} has wrong decision refusal mutation/reason: {handoff}")
    if handoff.get("changed") != []:
        raise SystemExit(f"{label} should not report changed decision rows: {handoff}")
    assert_decision_contract(
        metadata,
        handoff,
        label,
        state_changed=False,
        changed=[],
        content_in_handoff=any(key in handoff for key in ("raw_title", "raw_status", "raw_decision_id")),
    )
    if not isinstance(handoff.get("next_commands"), dict) or "decisions" not in handoff["next_commands"].values():
        raise SystemExit(f"{label} should include decision recovery commands: {handoff}")
    for key in ("decision_id", "status", "raw_title", "raw_status", "raw_decision_id"):
        if key in handoff and key in metadata and handoff.get(key) != metadata.get(key):
            raise SystemExit(f"{label} handoff {key} diverged from metadata: {handoff} vs {metadata}")
    boundaries = handoff.get("boundaries") or {}
    expected = {
        "read_only": read_only,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "writes_database": False,
        "queues_approval": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "requires_approval": False,
        "controls_computer": False,
        "external_side_effect": False,
        "calls_model": False,
        "executes_tools": False,
        "completes_tasks": False,
        "creates_decision": False,
        "updates_decision": False,
        "reads_private_data": False,
    }
    for key, value in expected.items():
        if boundaries.get(key) is not value:
            raise SystemExit(f"{label} refusal boundary {key} should be {value}: {boundaries}")
    assert_no_local_paths(handoff, label)


def main() -> None:
    assert_decision_exact_metadata_bool()
    assert_decision_malformed_handoff_flags()
    assert_planner_routes_decision_phone_aliases()
    assert_runtime_routes_decision_count_aliases()
    assert_decision_projection_concurrency()

    with TemporaryDirectory(prefix="jarvis-decisions-") as temp:
        runtime = make_temp_runtime(Path(temp))
        root = runtime.vault.root_path
        cases = [
            "record decision Jarvis uses explicit approval boundaries because computer control is risky impact high-risk actions stay reviewable",
            "decisions",
            "show decision 1",
            "decision 1 superseded",
            "all decisions",
            "search memory for approval boundaries",
            "export state",
        ]
        for case in cases:
            result = runtime.handle(case)
            status = "ok" if result.verified else "blocked"
            print(f"[{status}] {case}")
            print(result.response[:1600])
            print()
            if case == "record decision Jarvis uses explicit approval boundaries because computer control is risky impact high-risk actions stay reviewable":
                metadata = result.tool_results[0].metadata
                if metadata.get("decision_id") != 1 or metadata.get("writes_files") is not True:
                    raise SystemExit("record_decision missed durable write metadata.")
                if metadata.get("writes_memory") is not True or metadata.get("writes_notes") is not True:
                    raise SystemExit("record_decision missed memory/note write metadata.")
                if metadata.get("reads_private_data") is not False or metadata.get("controls_computer") is not False:
                    raise SystemExit("record_decision missed safety metadata.")
                if metadata.get("requires_approval") or metadata.get("external_side_effect"):
                    raise SystemExit("record_decision should stay local-safe without external side effects.")
                assert_decision_mutation_handoff(metadata, "runtime record_decision", source="record_decision", mutation="decision_create", changed=["decision"])
                assert_write_receipt(result.tool_results[0], root=root, label="runtime record_decision")
            if case == "decisions":
                metadata = result.tool_results[0].metadata
                if metadata.get("limit") != 25 or metadata.get("writes_files") is not False:
                    raise SystemExit("list_decisions missed sanitized limit/read-only metadata.")
                if "Count:" not in result.response or metadata.get("total_decisions") != metadata.get("count"):
                    raise SystemExit(f"list_decisions missed visible/metadata count parity: {result.response!r} / {metadata}")
                handoff = metadata.get("decision_list_handoff")
                if not isinstance(handoff, dict) or handoff.get("source") != "list_decisions":
                    raise SystemExit(f"list_decisions missed structured handoff metadata: {metadata}")
                if handoff.get("ready_for_operator") is not True or handoff.get("count") != metadata.get("count"):
                    raise SystemExit(f"list_decisions handoff missed readiness/count parity: {handoff}")
                assert_decision_list_handoff(metadata, "runtime list_decisions", content_in_handoff=True)
                if handoff.get("status") != "active" or handoff.get("limit") != 25:
                    raise SystemExit(f"list_decisions handoff missed normalized status/limit: {handoff}")
                if handoff.get("decision_ids") != [1]:
                    raise SystemExit(f"list_decisions handoff missed decision id parity: {handoff}")
                rows = handoff.get("rows")
                if not isinstance(rows, list) or rows[0].get("show_command") != "show decision 1":
                    raise SystemExit(f"list_decisions handoff missed show command: {handoff}")
                if rows[0].get("supersede_command") != "decision 1 superseded" or rows[0].get("retire_command") != "decision 1 retired":
                    raise SystemExit(f"list_decisions handoff missed status commands: {handoff}")
                if handoff.get("first_show_command") != "show decision 1" or "all decisions" not in handoff.get("next_commands", []):
                    raise SystemExit(f"list_decisions handoff missed next commands: {handoff}")
                boundaries = handoff.get("boundaries")
                if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
                    raise SystemExit(f"list_decisions handoff missed read-only boundary: {handoff}")
                if boundaries.get("writes_files") or boundaries.get("writes_memory") or boundaries.get("writes_notes"):
                    raise SystemExit(f"list_decisions handoff should stay non-mutating: {handoff}")
            if case == "show decision 1":
                metadata = result.tool_results[0].metadata
                if metadata.get("decision_id") != 1 or metadata.get("reads_private_data") is not False:
                    raise SystemExit("get_decision missed read-only metadata.")
                assert_decision_read_handoff(metadata, "runtime get_decision")
            if case == "decision 1 superseded":
                metadata = result.tool_results[0].metadata
                if metadata.get("decision_id") != 1 or metadata.get("status") != "superseded":
                    raise SystemExit("set_decision_status missed status metadata.")
                assert_decision_mutation_handoff(metadata, "runtime set_decision_status", source="set_decision_status", mutation="decision_status_update", changed=["status"])
                assert_write_receipt(result.tool_results[0], root=root, label="runtime set_decision_status")

        direct_list = runtime.registry.get("list_decisions").handler({"limit": "bad"})
        if not direct_list.ok or direct_list.metadata.get("limit") != 25 or direct_list.metadata.get("raw_limit") != "bad":
            raise SystemExit("list_decisions did not sanitize a bad limit.")
        if direct_list.metadata.get("writes_files") is not False or direct_list.metadata.get("reads_private_data") is not False:
            raise SystemExit("list_decisions missed safety metadata.")
        if direct_list.metadata.get("decision_list_handoff", {}).get("limit") != 25:
            raise SystemExit(f"list_decisions handoff should use sanitized limit: {direct_list.metadata}")
        assert_decision_list_handoff(direct_list.metadata, "direct list bad limit", content_in_handoff=False)
        direct_list_bool = runtime.registry.get("list_decisions").handler({"limit": False})
        if not direct_list_bool.ok or direct_list_bool.metadata.get("limit") != 25 or direct_list_bool.metadata.get("raw_limit") != "False":
            raise SystemExit(f"list_decisions should treat boolean limits as malformed defaults: {direct_list_bool.metadata}")
        if direct_list_bool.metadata.get("writes_files") or direct_list_bool.metadata.get("writes_memory"):
            raise SystemExit(f"list_decisions boolean limit should stay read-only: {direct_list_bool.metadata}")
        direct_list_long_limit = runtime.registry.get("list_decisions").handler({"limit": "l" * 200})
        if direct_list_long_limit.metadata.get("raw_limit") != ("l" * 77 + "..."):
            raise SystemExit(f"list_decisions did not bound raw bad limit metadata: {direct_list_long_limit.metadata}")
        for path_limit in (
            "/\x55sers/example/private/decision-limit",
            "/var/folders/zc/jarvis/decision-limit",
            "/tmp/jarvis/decision-limit",
        ):
            path_bad_list = runtime.registry.get("list_decisions").handler({"limit": path_limit})
            if path_bad_list.metadata.get("raw_limit") != "<local-path>":
                raise SystemExit(f"list_decisions leaked local path in raw limit metadata: {path_bad_list.metadata}")

        direct_list_large = runtime.registry.get("list_decisions").handler({"limit": 999999, "status": "all"})
        if not direct_list_large.ok or direct_list_large.metadata.get("limit") != 200:
            raise SystemExit("list_decisions did not clamp a large limit.")
        if direct_list_large.metadata.get("decision_list_handoff", {}).get("status") != "all":
            raise SystemExit(f"list_decisions handoff missed all-status mode: {direct_list_large.metadata}")
        assert_decision_list_handoff(direct_list_large.metadata, "direct list all", content_in_handoff=True)

        malformed_runtime = make_temp_runtime(Path(temp) / "malformed")
        leak_markers = ["DECISION_LIST_ROW_SECRET", "DECISION_READ_ROW_SECRET"]
        decision_rows = [
            HostileRow(leak_markers[0]),
            {
                "id": 44,
                "title": "Use durable memory",
                "status": "active",
                "rationale": "because context matters",
                "impact": "better handoffs",
                "created_at": "2026-07-04T00:00:00Z",
                "updated_at": "2026-07-04T00:00:00Z",
            },
        ]
        malformed_runtime.store.list_decisions = lambda status=None, limit=25: decision_rows[:limit]
        malformed_list = malformed_runtime.registry.get("list_decisions").handler({"limit": 10, "status": "all"})
        malformed_list_text = malformed_list.output + repr(malformed_list.metadata)
        if not malformed_list.ok:
            raise SystemExit(f"list_decisions should survive hostile local rows: {malformed_list.output}")
        if any(marker in malformed_list_text for marker in leak_markers):
            raise SystemExit(f"list_decisions leaked hostile row text: {malformed_list_text}")
        if "Use durable memory" not in malformed_list.output or "could not be read safely" not in malformed_list.output:
            raise SystemExit(f"list_decisions should preserve readable rows and report unreadable rows: {malformed_list.output}")
        malformed_handoff = malformed_list.metadata.get("decision_list_handoff")
        if not isinstance(malformed_handoff, dict):
            raise SystemExit(f"list_decisions hostile row case missed handoff: {malformed_list.metadata}")
        if malformed_list.metadata.get("count") != 2 or malformed_handoff.get("count") != 2:
            raise SystemExit(f"list_decisions hostile row case should preserve total count: {malformed_list.metadata}")
        if malformed_list.metadata.get("readable_decision_rows") != 1 or malformed_list.metadata.get("unreadable_decision_rows") != 1:
            raise SystemExit(f"list_decisions hostile row case missed readable/unreadable metadata: {malformed_list.metadata}")
        if malformed_handoff.get("decision_ids") != [44] or malformed_handoff.get("first_show_command") != "show decision 44":
            raise SystemExit(f"list_decisions hostile row case missed readable decision commands: {malformed_handoff}")
        assert_decision_list_handoff(malformed_list.metadata, "list_decisions hostile row", content_in_handoff=True)

        malformed_runtime.store.get_decision = lambda decision_id: HostileRow(leak_markers[1])
        malformed_read = malformed_runtime.registry.get("get_decision").handler({"decision_id": 44})
        malformed_read_text = malformed_read.output + repr(malformed_read.metadata)
        if not malformed_read.ok:
            raise SystemExit(f"get_decision should fail closed without crashing on hostile rows: {malformed_read.output}")
        if any(marker in malformed_read_text for marker in leak_markers):
            raise SystemExit(f"get_decision leaked hostile row text: {malformed_read_text}")
        if "could not be read safely" not in malformed_read.output:
            raise SystemExit(f"get_decision hostile row case should explain safe unreadable state: {malformed_read.output}")
        read_handoff = malformed_read.metadata.get("decision_read_handoff")
        if not isinstance(read_handoff, dict) or read_handoff.get("status") != "unreadable":
            raise SystemExit(f"get_decision hostile row case missed unreadable handoff: {malformed_read.metadata}")
        if malformed_read.metadata.get("decision_id") != 44 or malformed_read.metadata.get("status") != "unreadable":
            raise SystemExit(f"get_decision hostile row case missed stable id/status metadata: {malformed_read.metadata}")
        if read_handoff.get("content_in_handoff") is not False or malformed_read.metadata.get("content_in_handoff") is not False:
            raise SystemExit(f"get_decision hostile row should not claim content handoff: {malformed_read.metadata}")
        if malformed_read.metadata.get("readable_decision_row") is not False or malformed_read.metadata.get("unreadable_decision_row") is not True:
            raise SystemExit(f"get_decision hostile row missed readable/unreadable flags: {malformed_read.metadata}")
        assert_decision_contract(
            malformed_read.metadata,
            read_handoff,
            "get_decision hostile row",
            state_changed=False,
            changed=[],
            content_in_handoff=False,
        )

        bad_list_status = runtime.registry.get("list_decisions").handler({"status": "archived"})
        if bad_list_status.ok or bad_list_status.metadata.get("reason") != "bad_status":
            raise SystemExit("list_decisions bad status did not include refusal metadata.")
        if bad_list_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"list_decisions bad status should preserve bounded raw status: {bad_list_status.metadata}")
        if bad_list_status.metadata.get("writes_files") or bad_list_status.metadata.get("queues_approval"):
            raise SystemExit("list_decisions bad status should stay read-only and approval-free.")
        assert_decision_refusal_handoff(
            bad_list_status.metadata,
            "list_decisions bad status",
            source="list_decisions",
            mutation="decision_list",
            reason="bad_status",
            read_only=True,
        )
        for path_status in (
            "/private/tmp/jarvis-decision-status",
            "/var/folders/zc/jarvis/decision-status",
            "/tmp/jarvis/decision-status",
        ):
            path_bad_list_status = runtime.registry.get("list_decisions").handler({"status": path_status})
            if path_bad_list_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"list_decisions leaked local path in raw status metadata: {path_bad_list_status.metadata}")
            assert_decision_refusal_handoff(
                path_bad_list_status.metadata,
                "list_decisions path bad status",
                source="list_decisions",
                mutation="decision_list",
                reason="bad_status",
                read_only=True,
            )

        bad_decision = runtime.registry.get("get_decision").handler({"decision_id": "bad"})
        if bad_decision.ok or "must be a number" not in bad_decision.output:
            raise SystemExit("get_decision did not handle a bad id cleanly.")
        if bad_decision.metadata.get("reason") != "bad_decision_id" or bad_decision.metadata.get("writes_files"):
            raise SystemExit("get_decision bad id missed safe refusal metadata.")
        if bad_decision.metadata.get("raw_decision_id") != "bad":
            raise SystemExit("get_decision bad id missed bounded raw id metadata.")
        assert_decision_refusal_handoff(
            bad_decision.metadata,
            "get_decision bad id",
            source="get_decision",
            mutation="decision_read",
            reason="bad_decision_id",
            read_only=True,
        )
        bool_decision = runtime.registry.get("get_decision").handler({"decision_id": True})
        if bool_decision.ok or "must be a number" not in bool_decision.output:
            raise SystemExit("get_decision should reject boolean ids instead of coercing them to decision ids.")
        if bool_decision.metadata.get("raw_decision_id") != "True":
            raise SystemExit(f"get_decision should preserve boolean raw id metadata: {bool_decision.metadata}")
        if bool_decision.metadata.get("writes_files") or bool_decision.metadata.get("writes_memory") or bool_decision.metadata.get("writes_notes"):
            raise SystemExit(f"get_decision boolean id should not write: {bool_decision.metadata}")

        long_bad_decision = runtime.registry.get("get_decision").handler({"decision_id": "x" * 200})
        if long_bad_decision.metadata.get("raw_decision_id") != ("x" * 77 + "..."):
            raise SystemExit("get_decision bad id did not bound raw id metadata.")
        for path_id in (
            "/\x55sers/example/private/decision-id",
            "/var/folders/zc/jarvis/decision-id",
            "/tmp/jarvis/decision-id",
        ):
            path_bad_decision = runtime.registry.get("get_decision").handler({"decision_id": path_id})
            if path_bad_decision.metadata.get("raw_decision_id") != "<local-path>":
                raise SystemExit(f"get_decision leaked local path in raw id metadata: {path_bad_decision.metadata}")
            assert_decision_refusal_handoff(
                path_bad_decision.metadata,
                "get_decision path bad id",
                source="get_decision",
                mutation="decision_read",
                reason="bad_decision_id",
                read_only=True,
            )

        for bad_numeric_id in (0, -1):
            bad_numeric_decision = runtime.registry.get("get_decision").handler({"decision_id": bad_numeric_id})
            if bad_numeric_decision.ok or "positive number" not in bad_numeric_decision.output:
                raise SystemExit("get_decision should reject non-positive ids before lookup.")
            if bad_numeric_decision.metadata.get("reason") != "bad_decision_id" or bad_numeric_decision.metadata.get("decision_id") is not None:
                raise SystemExit(f"get_decision non-positive id should include bad-id metadata: {bad_numeric_decision.metadata}")
            if bad_numeric_decision.metadata.get("raw_decision_id") != str(bad_numeric_id):
                raise SystemExit(f"get_decision should preserve non-positive raw id metadata: {bad_numeric_decision.metadata}")
            if bad_numeric_decision.metadata.get("writes_files") or bad_numeric_decision.metadata.get("writes_memory") or bad_numeric_decision.metadata.get("writes_notes"):
                raise SystemExit(f"get_decision non-positive id should not write: {bad_numeric_decision.metadata}")
            assert_decision_refusal_handoff(
                bad_numeric_decision.metadata,
                "get_decision non-positive id",
                source="get_decision",
                mutation="decision_read",
                reason="bad_decision_id",
                read_only=True,
            )

        exported_status = runtime.registry.get("set_decision_status").handler({"decision_id": 1, "status": "active"})
        if not exported_status.ok or exported_status.metadata.get("writes_files") is not True:
            raise SystemExit("set_decision_status missed write metadata.")
        if exported_status.metadata.get("writes_memory") or exported_status.metadata.get("writes_notes") is not True:
            raise SystemExit("set_decision_status should write decision note but not index new memory.")
        assert_decision_mutation_handoff(exported_status.metadata, "direct set_decision_status", source="set_decision_status", mutation="decision_status_update", changed=["status"])
        assert_write_receipt(exported_status, root=root, label="direct set_decision_status")

        missing_title = runtime.registry.get("record_decision").handler({"title": "", "rationale": "none"})
        if missing_title.ok or missing_title.metadata.get("reason") != "missing_title":
            raise SystemExit("record_decision missing title should include safe refusal metadata.")
        if missing_title.metadata.get("raw_title") != "":
            raise SystemExit(f"record_decision missing title should preserve bounded raw title: {missing_title.metadata}")
        assert_decision_refusal_handoff(
            missing_title.metadata,
            "record_decision missing title",
            source="record_decision",
            mutation="decision_create",
            reason="missing_title",
            read_only=False,
        )
        for path_title in (
            "/\x55sers/example/private/decision-title",
            "/var/folders/zc/jarvis/decision-title",
            "/tmp/jarvis/decision-title",
        ):
            path_bad_title = runtime.registry.get("record_decision").handler({"title": path_title, "rationale": "none"})
            if path_bad_title.ok or path_bad_title.metadata.get("reason") != "invalid_title":
                raise SystemExit("record_decision path-shaped title should include safe refusal metadata.")
            if path_bad_title.metadata.get("raw_title") != "<local-path>":
                raise SystemExit(f"record_decision leaked local path in raw title metadata: {path_bad_title.metadata}")
            if path_bad_title.metadata.get("writes_files") or path_bad_title.metadata.get("writes_memory") or path_bad_title.metadata.get("writes_notes"):
                raise SystemExit(f"record_decision path-shaped title should not write: {path_bad_title.metadata}")
            if any(fragment in path_bad_title.output for fragment in ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")):
                raise SystemExit(f"record_decision leaked local path in refusal output: {path_bad_title.output}")
            assert_decision_refusal_handoff(
                path_bad_title.metadata,
                "record_decision path bad title",
                source="record_decision",
                mutation="decision_create",
                reason="invalid_title",
                read_only=False,
            )

        bad_status = runtime.registry.get("set_decision_status").handler({"decision_id": 1, "status": "archived"})
        if bad_status.ok or bad_status.metadata.get("reason") != "bad_status":
            raise SystemExit("set_decision_status bad status missed safe refusal metadata.")
        if bad_status.metadata.get("raw_status") != "archived":
            raise SystemExit(f"set_decision_status bad status should preserve bounded raw status: {bad_status.metadata}")
        assert_decision_refusal_handoff(
            bad_status.metadata,
            "set_decision_status bad status",
            source="set_decision_status",
            mutation="decision_status_update",
            reason="bad_status",
            read_only=False,
        )
        for path_status in (
            "/private/tmp/jarvis-set-decision-status",
            "/var/folders/zc/jarvis/set-decision-status",
            "/tmp/jarvis/set-decision-status",
        ):
            path_bad_status = runtime.registry.get("set_decision_status").handler({"decision_id": 1, "status": path_status})
            if path_bad_status.metadata.get("raw_status") != "<local-path>":
                raise SystemExit(f"set_decision_status leaked local path in raw status metadata: {path_bad_status.metadata}")
            assert_decision_refusal_handoff(
                path_bad_status.metadata,
                "set_decision_status path bad status",
                source="set_decision_status",
                mutation="decision_status_update",
                reason="bad_status",
                read_only=False,
            )

        bad_status_id = runtime.registry.get("set_decision_status").handler({"decision_id": "bad", "status": "retired"})
        if bad_status_id.ok or "must be a number" not in bad_status_id.output:
            raise SystemExit("set_decision_status did not handle a bad id cleanly.")
        if bad_status_id.metadata.get("raw_decision_id") != "bad" or bad_status_id.metadata.get("writes_files"):
            raise SystemExit("set_decision_status bad id missed safe bounded raw id metadata.")
        assert_decision_refusal_handoff(
            bad_status_id.metadata,
            "set_decision_status bad id",
            source="set_decision_status",
            mutation="decision_status_update",
            reason="bad_decision_id",
            read_only=False,
        )
        bool_status_id = runtime.registry.get("set_decision_status").handler({"decision_id": True, "status": "retired"})
        if bool_status_id.ok or "must be a number" not in bool_status_id.output:
            raise SystemExit("set_decision_status should reject boolean ids before mutation.")
        if bool_status_id.metadata.get("raw_decision_id") != "True":
            raise SystemExit(f"set_decision_status should preserve boolean raw id metadata: {bool_status_id.metadata}")
        if bool_status_id.metadata.get("writes_files") or bool_status_id.metadata.get("writes_memory") or bool_status_id.metadata.get("writes_notes"):
            raise SystemExit(f"set_decision_status boolean id should not write: {bool_status_id.metadata}")
        for path_id in (
            "/\x55sers/example/private/status-id",
            "/var/folders/zc/jarvis/status-id",
            "/tmp/jarvis/status-id",
        ):
            path_bad_status_id = runtime.registry.get("set_decision_status").handler({"decision_id": path_id, "status": "retired"})
            if path_bad_status_id.metadata.get("raw_decision_id") != "<local-path>":
                raise SystemExit(f"set_decision_status leaked local path in raw id metadata: {path_bad_status_id.metadata}")
            assert_decision_refusal_handoff(
                path_bad_status_id.metadata,
                "set_decision_status path bad id",
                source="set_decision_status",
                mutation="decision_status_update",
                reason="bad_decision_id",
                read_only=False,
            )

        for bad_numeric_id in (0, -1):
            bad_numeric_status_id = runtime.registry.get("set_decision_status").handler({"decision_id": bad_numeric_id, "status": "retired"})
            if bad_numeric_status_id.ok or "positive number" not in bad_numeric_status_id.output:
                raise SystemExit("set_decision_status should reject non-positive ids before mutation.")
            if bad_numeric_status_id.metadata.get("reason") != "bad_decision_id" or bad_numeric_status_id.metadata.get("decision_id") is not None:
                raise SystemExit(f"set_decision_status non-positive id should include bad-id metadata: {bad_numeric_status_id.metadata}")
            if bad_numeric_status_id.metadata.get("raw_decision_id") != str(bad_numeric_id):
                raise SystemExit(f"set_decision_status should preserve non-positive raw id metadata: {bad_numeric_status_id.metadata}")
            if bad_numeric_status_id.metadata.get("writes_files") or bad_numeric_status_id.metadata.get("writes_memory") or bad_numeric_status_id.metadata.get("writes_notes"):
                raise SystemExit(f"set_decision_status non-positive id should not write: {bad_numeric_status_id.metadata}")
            assert_decision_refusal_handoff(
                bad_numeric_status_id.metadata,
                "set_decision_status non-positive id",
                source="set_decision_status",
                mutation="decision_status_update",
                reason="bad_decision_id",
                read_only=False,
            )

        missing_status_decision = runtime.registry.get("set_decision_status").handler({"decision_id": 999999, "status": "retired"})
        if missing_status_decision.ok or missing_status_decision.metadata.get("reason") != "not_found":
            raise SystemExit(f"set_decision_status missing target missed stable metadata: {missing_status_decision.metadata}")
        if missing_status_decision.metadata.get("writes_files") or missing_status_decision.metadata.get("writes_memory") or missing_status_decision.metadata.get("writes_notes"):
            raise SystemExit(f"set_decision_status missing target should not write: {missing_status_decision.metadata}")
        expected_status_recovery = [
            "decisions",
            "show decision <correct decision id>",
            "decision <correct decision id> <active|superseded|retired>",
        ]
        if missing_status_decision.metadata.get("recovery_commands") != expected_status_recovery:
            raise SystemExit(f"set_decision_status missing target recovery order drifted: {missing_status_decision.metadata}")
        for token in ("Run `decisions`", "refresh decision IDs", "show decision <correct decision id>", "normal local-safe policy"):
            if token not in missing_status_decision.output:
                raise SystemExit(f"set_decision_status missing target output missed {token!r}: {missing_status_decision.output}")
        for key in ("retry_requires_decision_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
            if missing_status_decision.metadata.get(key) is not True:
                raise SystemExit(f"set_decision_status missing target missed {key}: {missing_status_decision.metadata}")
        for key in ("retry_requires_fresh_approval", "authorizes_retry", "authorizes_decision_mutation"):
            if missing_status_decision.metadata.get(key):
                raise SystemExit(f"set_decision_status missing target unexpectedly set {key}: {missing_status_decision.metadata}")
        assert_decision_refusal_handoff(
            missing_status_decision.metadata,
            "set_decision_status missing target",
            source="set_decision_status",
            mutation="decision_status_update",
            reason="not_found",
            read_only=False,
        )

        missing_decision = runtime.registry.get("get_decision").handler({"decision_id": 999})
        if missing_decision.ok or missing_decision.metadata.get("reason") != "not_found":
            raise SystemExit("get_decision missing path missed stable metadata.")
        assert_decision_refusal_handoff(
            missing_decision.metadata,
            "get_decision missing decision",
            source="get_decision",
            mutation="decision_read",
            reason="not_found",
            read_only=True,
        )
        if missing_decision.metadata.get("recovery_commands") != ["decisions", "show decision <correct decision id>"]:
            raise SystemExit(f"get_decision missing recovery order drifted: {missing_decision.metadata}")
        for token in ("Run `decisions`", "refresh decision IDs", "show decision <correct decision id>", "normal read-only policy"):
            if token not in missing_decision.output:
                raise SystemExit(f"get_decision missing output missed {token!r}: {missing_decision.output}")
        for key in ("retry_requires_decision_refresh", "retry_requires_corrected_id", "recovery_commands_require_normal_policy"):
            if missing_decision.metadata.get(key) is not True:
                raise SystemExit(f"get_decision missing recovery missed {key}: {missing_decision.metadata}")
        for key in ("retry_requires_fresh_approval", "authorizes_retry", "authorizes_decision_mutation"):
            if missing_decision.metadata.get(key):
                raise SystemExit(f"get_decision missing recovery unexpectedly set {key}: {missing_decision.metadata}")

        empty_runtime = make_temp_runtime(Path(temp) / "empty")
        empty_list = empty_runtime.registry.get("list_decisions").handler({"status": "active"})
        empty_handoff = empty_list.metadata.get("decision_list_handoff")
        if not empty_list.ok or not isinstance(empty_handoff, dict):
            raise SystemExit(f"empty list_decisions missed handoff metadata: {empty_list.metadata}")
        if empty_handoff.get("count") != 0 or empty_handoff.get("decision_ids") != [] or empty_handoff.get("first_show_command") is not None:
            raise SystemExit(f"empty list_decisions handoff should preserve empty state: {empty_handoff}")
        if empty_handoff.get("boundaries", {}).get("read_only") is not True or empty_handoff.get("boundaries", {}).get("writes_files"):
            raise SystemExit(f"empty list_decisions handoff should stay read-only: {empty_handoff}")
        assert_decision_list_handoff(empty_list.metadata, "empty list_decisions", content_in_handoff=False)

        bounded = runtime.registry.get("record_decision").handler({"title": "t" * 500, "rationale": "r" * 5000, "impact": "i" * 5000})
        if not bounded.ok:
            raise SystemExit("record_decision should accept bounded long fields.")
        if bounded.metadata.get("title_chars") != 240 or bounded.metadata.get("rationale_chars") != 4000:
            raise SystemExit("record_decision did not bound long title/rationale fields.")
        assert_decision_mutation_handoff(bounded.metadata, "bounded record_decision", source="record_decision", mutation="decision_create", changed=["decision"])
        assert_write_receipt(bounded, root=root, label="bounded record_decision")


if __name__ == "__main__":
    main()
