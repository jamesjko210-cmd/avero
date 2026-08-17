from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.memory.store import MemoryRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime


READ_ONLY_FALSE_FLAGS = [
    "calls_model",
    "writes_memory",
    "writes_files",
    "writes_notes",
    "executes_tools",
    "executes_side_effect",
    "external_calls",
    "controls_computer",
    "queues_approval",
    "approves_request",
    "dismisses_request",
    "reads_private_data",
    "reads_personal_data",
    "authorizes_execution",
    "authorizes_completion_claim",
    "approval_granted",
    "speaks",
    "completes_tasks",
]


def _assert_read_only(metadata: dict, label: str) -> None:
    for key in READ_ONLY_FALSE_FLAGS:
        if metadata.get(key):
            raise SystemExit(f"{label} should stay read-only for {key}: {metadata}")


def _assert_no_local_path(value: str, label: str) -> None:
    for fragment in ["/\x55sers/", "/private/", "/var/folders/", "/tmp/"]:
        if fragment in value:
            raise SystemExit(f"{label} leaked a local path: {value}")


def _assert_brain_think_route(command: str, question: str) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != "brain_think":
        raise SystemExit(f"{command!r} should route to brain_think: {[(action.tool_name, action.args) for action in actions]}")
    expected_args = {"question": question, "limit": 8}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} brain_think args mismatch: {actions[0].args} != {expected_args}")


def _assert_brain_search_route(command: str, query: str) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != "brain_search":
        raise SystemExit(f"{command!r} should route to brain_search: {[(action.tool_name, action.args) for action in actions]}")
    expected_args = {"query": query, "limit": 8}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} brain_search args mismatch: {actions[0].args} != {expected_args}")


def _assert_brain_graph_route(command: str) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != "brain_graph":
        raise SystemExit(f"{command!r} should route to brain_graph: {[(action.tool_name, action.args) for action in actions]}")
    expected_args = {"limit": 12}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} brain_graph args mismatch: {actions[0].args} != {expected_args}")


def _assert_brain_neighbors_route(command: str, memory_id: int) -> None:
    plan = RuleBasedPlanner().plan(command)
    actions = plan.actions
    if len(actions) != 1 or actions[0].tool_name != "brain_neighbors":
        raise SystemExit(f"{command!r} should route to brain_neighbors: {[(action.tool_name, action.args) for action in actions]}")
    expected_args = {"memory_id": memory_id, "limit": 6}
    if actions[0].args != expected_args:
        raise SystemExit(f"{command!r} brain_neighbors args mismatch: {actions[0].args} != {expected_args}")


def _assert_no_brain_neighbor_route(command: str) -> None:
    plan = RuleBasedPlanner().plan(command)
    if plan.actions:
        raise SystemExit(f"{command!r} should stay unclaimed without a memory id: {[(action.tool_name, action.args) for action in plan.actions]}")


def _assert_brain_handoff(result, label: str, source: str, status: str, reason: str | None = None) -> None:
    metadata = result.metadata
    handoff = metadata.get("brain_handoff")
    if metadata.get("brain_handoff_ready") is not True or not isinstance(handoff, dict):
        raise SystemExit(f"{label} should expose a ready brain_handoff: {metadata}")
    if handoff.get("handoff_ready") is not True:
        raise SystemExit(f"{label} handoff should report handoff_ready: {handoff}")
    if metadata.get("brain_status") != status or handoff.get("status") != status or handoff.get("source") != source:
        raise SystemExit(f"{label} handoff source/status mismatch: {handoff}")
    if reason:
        if metadata.get("refusal_reason") != reason or handoff.get("reason") != reason or handoff.get("refused") is not True:
            raise SystemExit(f"{label} handoff reason mismatch: {handoff}")
    elif handoff.get("refused") is not False:
        raise SystemExit(f"{label} handoff should not be refused: {handoff}")
    if handoff.get("ready_for_operator") is not True or handoff.get("state_changed") is not False or handoff.get("changed") != [] or handoff.get("content_in_handoff") is not False:
        raise SystemExit(f"{label} handoff should be content-free and unchanged: {handoff}")
    if (
        metadata.get("brain_ready_for_operator") is not True
        or metadata.get("brain_state_changed") is not False
        or metadata.get("brain_changed") != []
        or metadata.get("brain_content_in_handoff") is not False
        or metadata.get("brain_next_commands") != handoff.get("next_commands")
        or metadata.get("brain_next_command_count") != len(handoff.get("next_commands", []))
        or metadata.get("brain_boundaries") != handoff.get("boundaries")
    ):
        raise SystemExit(f"{label} flat brain handoff aliases diverged: {metadata}")
    commands = handoff.get("next_commands")
    if not isinstance(commands, list) or not {"brain search <query>", "brain think <question>", "brain neighbors <memory_id>", "brain graph"}.issubset(set(commands)):
        raise SystemExit(f"{label} handoff should expose brain recovery commands: {handoff}")
    boundaries = handoff.get("boundaries")
    if not isinstance(boundaries, dict) or boundaries.get("read_only") is not True:
        raise SystemExit(f"{label} handoff should expose read-only boundaries: {handoff}")
    for key in READ_ONLY_FALSE_FLAGS:
        if boundaries.get(key):
            raise SystemExit(f"{label} handoff boundary should be false for {key}: {handoff}")
    _assert_no_local_path(str(handoff), f"{label} handoff")

    if source == "brain_search":
        if handoff.get("count") != metadata.get("count") or len(handoff.get("citations", [])) != len(metadata.get("citations", [])):
            raise SystemExit(f"{label} search handoff parity failed: {handoff} / {metadata}")
    if source == "brain_think":
        if (
            handoff.get("count") != metadata.get("count")
            or len(handoff.get("citations", [])) != len(metadata.get("citations", []))
            or len(handoff.get("gaps", [])) != len(metadata.get("gaps", []))
        ):
            raise SystemExit(f"{label} think handoff parity failed: {handoff} / {metadata}")
    if source == "brain_neighbors":
        if handoff.get("count") != metadata.get("count") or len(handoff.get("citations", [])) != len(metadata.get("citations", [])):
            raise SystemExit(f"{label} neighbors handoff parity failed: {handoff} / {metadata}")
        if isinstance(metadata.get("memory_id"), int) and handoff.get("memory_id") != metadata.get("memory_id"):
            raise SystemExit(f"{label} neighbors memory id parity failed: {handoff} / {metadata}")
    if source == "brain_graph":
        graph_counts = handoff.get("graph_counts")
        expected = {key: metadata.get(key) for key in ["memory_nodes", "people_nodes", "category_nodes", "edges"]}
        if graph_counts != expected:
            raise SystemExit(f"{label} graph count parity failed: {handoff} / {metadata}")


def main() -> None:
    default_question = "what should Jarvis do next"
    for command in [
        "brain think",
        "brain think please",
        "gbrain think please",
        "show brain think",
        "show latest brain think",
        "show me gbrain think",
        "think with brain please",
        "show think with brain",
    ]:
        _assert_brain_think_route(command, default_question)
    _assert_brain_think_route("brain think: summarize this work please", "summarize this work please")
    for command in [
        "brain search",
        "brain search please",
        "show brain search",
        "show latest brain search",
        "show me gbrain search",
    ]:
        _assert_brain_search_route(command, "")
    for command in [
        "brain search safety please",
        "gbrain search safety",
        "show brain search safety please",
        "show latest gbrain search safety",
    ]:
        _assert_brain_search_route(command, "safety")
    _assert_brain_search_route("brain search: safety please", "safety")
    # Real privacy-relevant gap found live 2026-07-09: "search my brain for X"
    # (reversed word order from "brain search X") silently misrouted to a
    # PUBLIC web_lookup instead of the local, private brain_search tool.
    for command in ["search brain safety", "search my brain safety", "search my brain for safety", "search the brain for safety"]:
        _assert_brain_search_route(command, "safety")
    for command in [
        "brain graph",
        "brain graph please",
        "gbrain graph please",
        "show brain graph",
        "show latest brain graph",
        "show me gbrain graph",
        "knowledge graph please",
        "brain graph preview please",
    ]:
        _assert_brain_graph_route(command)
    for command in [
        "brain neighbors 1",
        "brain neighbor 1",
        "brain neighbors #1",
        "brain neighbors 1 please",
        "show brain neighbors 1",
        "show latest brain neighbors 1",
        "show me gbrain neighbors 1",
        "gbrain neighbors 1 please",
        "brain related 1",
        "related brain memory 1",
        "show related brain memory 1",
    ]:
        _assert_brain_neighbors_route(command, 1)
    _assert_no_brain_neighbor_route("show brain neighbors")

    with TemporaryDirectory(prefix="jarvis-brain-") as temp:
        runtime = make_temp_runtime(Path(temp))
        runtime.handle("remember that Jarvis brain graph stays local and reviewable")
        runtime.handle("remember that Jarvis brain neighbors cite local memory only")
        runtime.store.add_memory(
            MemoryRecord(
                category="/tmp/brain-category",
                title="/var/folders/zc/brain-title",
                body="Legacy memory mentions /\x55sers/example/private/brain-note and /tmp/brain-body.",
                source="smoke",
            )
        )

        search = runtime.registry.get("brain_search").handler({"query": "brain graph", "limit": 99})
        if not search.ok or search.metadata.get("count", 0) < 1:
            raise SystemExit(f"brain_search should find local memory rows: {search.metadata}")
        _assert_read_only(search.metadata, "brain_search")
        _assert_no_local_path(search.output, "brain_search output")
        _assert_brain_handoff(search, "brain_search", "brain_search", "ok")

        missing_search = runtime.registry.get("brain_search").handler({"query": ""})
        if missing_search.ok:
            raise SystemExit("brain_search should reject missing queries.")
        _assert_brain_handoff(missing_search, "missing brain_search", "brain_search", "refused", "missing_query")

        path_search = runtime.registry.get("brain_search").handler({"query": "/var/folders/zc/brain-search-query"})
        if path_search.ok or path_search.metadata.get("raw_query") != "<local-path>" or path_search.metadata.get("query") != "<local-path>":
            raise SystemExit(f"brain_search should refuse and redact path-shaped queries: {path_search.metadata}")
        _assert_no_local_path(path_search.output, "brain_search path-query output")
        _assert_read_only(path_search.metadata, "brain_search path query")
        _assert_brain_handoff(path_search, "brain_search path query", "brain_search", "refused", "local_path_query")

        think = runtime.registry.get("brain_think").handler({"question": "what should brain neighbors cite?"})
        if not think.ok or "local memory" not in think.output.lower():
            raise SystemExit("brain_think should produce a local-memory-only answer.")
        _assert_read_only(think.metadata, "brain_think")
        _assert_no_local_path(think.output, "brain_think output")
        _assert_brain_handoff(think, "brain_think", "brain_think", "ok")

        missing_think = runtime.registry.get("brain_think").handler({"question": ""})
        if missing_think.ok:
            raise SystemExit("brain_think should reject missing questions.")
        _assert_brain_handoff(missing_think, "missing brain_think", "brain_think", "refused", "missing_question")

        path_think = runtime.registry.get("brain_think").handler({"question": "/tmp/brain-think-question"})
        if path_think.ok or path_think.metadata.get("raw_question") != "<local-path>" or path_think.metadata.get("question") != "<local-path>":
            raise SystemExit(f"brain_think should refuse and redact path-shaped questions: {path_think.metadata}")
        _assert_no_local_path(path_think.output, "brain_think path-question output")
        _assert_read_only(path_think.metadata, "brain_think path question")
        _assert_brain_handoff(path_think, "brain_think path question", "brain_think", "refused", "local_path_question")

        legacy_search = runtime.registry.get("brain_search").handler({"query": "legacy memory mentions brain body", "limit": 10})
        if not legacy_search.ok or "<local-path>" not in legacy_search.output:
            raise SystemExit(f"brain_search should scrub legacy path-bearing memory rows: {legacy_search.output}")
        _assert_no_local_path(legacy_search.output, "brain_search legacy-memory output")
        _assert_brain_handoff(legacy_search, "legacy brain_search", "brain_search", "ok")

        graph = runtime.registry.get("brain_graph").handler({"limit": "bad"})
        if not graph.ok or graph.metadata.get("edges") is None:
            raise SystemExit(f"brain_graph should expose graph preview metadata: {graph.metadata}")
        _assert_read_only(graph.metadata, "brain_graph")
        _assert_brain_handoff(graph, "brain_graph", "brain_graph", "ok")

        bool_graph = runtime.registry.get("brain_graph").handler({"limit": True})
        if not bool_graph.ok or bool_graph.metadata.get("memory_nodes", 0) < 1:
            raise SystemExit(f"brain_graph should treat boolean limits as malformed defaults: {bool_graph.metadata}")
        _assert_read_only(bool_graph.metadata, "brain_graph boolean limit")
        _assert_brain_handoff(bool_graph, "brain_graph boolean limit", "brain_graph", "ok")

        bool_search = runtime.registry.get("brain_search").handler({"query": "brain graph", "limit": False})
        if not bool_search.ok or bool_search.metadata.get("count", 0) < 1:
            raise SystemExit(f"brain_search should treat boolean limits as malformed defaults: {bool_search.metadata}")
        _assert_read_only(bool_search.metadata, "brain_search boolean limit")
        _assert_brain_handoff(bool_search, "brain_search boolean limit", "brain_search", "ok")

        neighbors = runtime.registry.get("brain_neighbors").handler({"memory_id": 1, "limit": 99})
        if not neighbors.ok or neighbors.metadata.get("memory_id") != 1 or neighbors.metadata.get("count", 0) < 1:
            raise SystemExit(f"brain_neighbors should expose bounded neighbor metadata: {neighbors.metadata}")
        _assert_read_only(neighbors.metadata, "brain_neighbors")
        _assert_no_local_path(neighbors.output, "brain_neighbors output")
        _assert_brain_handoff(neighbors, "brain_neighbors", "brain_neighbors", "ok")

        missing = runtime.registry.get("brain_neighbors").handler({"memory_id": 999})
        if not missing.ok or missing.metadata.get("count") != 0 or missing.metadata.get("memory_id") != 999:
            raise SystemExit(f"brain_neighbors missing memory path should be stable: {missing.metadata}")
        _assert_read_only(missing.metadata, "missing brain_neighbors")
        _assert_brain_handoff(missing, "missing brain_neighbors", "brain_neighbors", "empty")

        bad = runtime.registry.get("brain_neighbors").handler({"memory_id": "memory-abc"})
        if bad.ok or "memory_id must be a number" not in bad.output:
            raise SystemExit("brain_neighbors should reject bad memory ids.")
        if bad.metadata.get("raw_memory_id") != "memory-abc" or bad.metadata.get("memory_id") is not None:
            raise SystemExit(f"brain_neighbors should preserve bounded raw memory id metadata: {bad.metadata}")
        _assert_read_only(bad.metadata, "bad brain_neighbors")
        _assert_brain_handoff(bad, "bad brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")

        bool_id = runtime.registry.get("brain_neighbors").handler({"memory_id": True})
        if bool_id.ok or "memory_id must be a number" not in bool_id.output:
            raise SystemExit("brain_neighbors should reject boolean memory ids.")
        if bool_id.metadata.get("raw_memory_id") != "True" or bool_id.metadata.get("memory_id") is not None:
            raise SystemExit(f"brain_neighbors should preserve boolean raw memory id metadata: {bool_id.metadata}")
        _assert_read_only(bool_id.metadata, "boolean brain_neighbors")
        _assert_brain_handoff(bool_id, "boolean brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")

        zero_with_alias = runtime.registry.get("brain_neighbors").handler({"memory_id": 0, "id": 1})
        if zero_with_alias.ok or "positive number" not in zero_with_alias.output:
            raise SystemExit("brain_neighbors should reject explicit zero memory_id before using aliases.")
        if zero_with_alias.metadata.get("raw_memory_id") != "0" or zero_with_alias.metadata.get("memory_id") is not None:
            raise SystemExit(f"brain_neighbors should preserve explicit zero id metadata: {zero_with_alias.metadata}")
        _assert_read_only(zero_with_alias.metadata, "zero brain_neighbors")
        _assert_brain_handoff(zero_with_alias, "zero brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")

        negative = runtime.registry.get("brain_neighbors").handler({"memory_id": -1})
        if negative.ok or "positive number" not in negative.output:
            raise SystemExit("brain_neighbors should reject negative memory ids.")
        if negative.metadata.get("raw_memory_id") != "-1" or negative.metadata.get("memory_id") is not None:
            raise SystemExit(f"brain_neighbors should preserve negative id metadata: {negative.metadata}")
        _assert_read_only(negative.metadata, "negative brain_neighbors")
        _assert_brain_handoff(negative, "negative brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")

        long_bad = runtime.registry.get("brain_neighbors").handler({"memory_id": "m" * 200})
        if long_bad.ok or long_bad.metadata.get("raw_memory_id") != ("m" * 77 + "..."):
            raise SystemExit(f"brain_neighbors should bound raw memory id metadata: {long_bad.metadata}")
        _assert_brain_handoff(long_bad, "long bad brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")
        path_bad = runtime.registry.get("brain_neighbors").handler({"memory_id": "/\x55sers/example/private/brain-memory-id"})
        if path_bad.ok or path_bad.metadata.get("raw_memory_id") != "<local-path>" or path_bad.metadata.get("memory_id") is not None:
            raise SystemExit(f"brain_neighbors should redact path-shaped bad memory ids: {path_bad.metadata}")
        _assert_read_only(path_bad.metadata, "path bad brain_neighbors")
        _assert_no_local_path(path_bad.output, "path bad brain_neighbors output")
        _assert_brain_handoff(path_bad, "path bad brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")
        for value in ["/var/folders/zc/brain-memory-id", "/tmp/brain-memory-id"]:
            path_bad_temp = runtime.registry.get("brain_neighbors").handler({"memory_id": value})
            if path_bad_temp.ok or path_bad_temp.metadata.get("raw_memory_id") != "<local-path>" or path_bad_temp.metadata.get("memory_id") is not None:
                raise SystemExit(f"brain_neighbors should redact temp-root bad memory ids: {path_bad_temp.metadata}")
            _assert_read_only(path_bad_temp.metadata, "temp path bad brain_neighbors")
            _assert_no_local_path(path_bad_temp.output, "temp path bad brain_neighbors output")
            _assert_brain_handoff(path_bad_temp, "temp path bad brain_neighbors", "brain_neighbors", "refused", "bad_memory_id")

    print("Brain smoke passed")


if __name__ == "__main__":
    main()
