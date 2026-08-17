from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime, review_pending_runtime_approval


PRIVATE_SENTINEL = "PRIVATE-REMINDER-CONTENT-DO-NOT-RETAIN-7319"
PERSISTENCE_SUMMARY = (
    "jarvis_status: sensitive result verified; private content was displayed transiently and not retained."
)
PARENT_PERSISTENCE_SUMMARY = (
    "approve_pending_approval: sensitive result verified; private content was displayed transiently and not retained."
)


class StaticPlanner:
    def plan(self, _user_input: str) -> Plan:
        return Plan(
            "Exercise transient personal-data output retention.",
            [PlannedAction("jarvis_status", {}, "read mocked private data")],
        )


def _database_contains(db_path: Path, needle: str) -> bool:
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ):
            table = str(row[0])
            columns = [
                str(column[1])
                for column in conn.execute(f'PRAGMA table_info("{table}")')
                if str(column[2]).upper() in {"TEXT", "", "JSON"}
            ]
            if not columns:
                continue
            quoted = ", ".join(f'"{column}"' for column in columns)
            for values in conn.execute(f'SELECT {quoted} FROM "{table}"'):
                if any(needle in str(value) for value in values if value is not None):
                    return True
    return False


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-sensitive-retention-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        original = runtime.registry.get("jarvis_status")

        def private_result(_args: dict[str, object]) -> ToolResult:
            return ToolResult(
                "jarvis_status",
                True,
                f"Reminder: {PRIVATE_SENTINEL}",
                {
                    "suppress_output_persistence": True,
                    "persistence_summary": f"unsafe connector summary {PRIVATE_SENTINEL}",
                    "reads_personal_data": True,
                    "reads_private_data": True,
                },
            )

        runtime.registry._tools["jarvis_status"] = replace(
            original,
            handler=private_result,
            risk=RiskLevel.PERSONAL_DATA,
        )
        runtime.planner = StaticPlanner()

        held = runtime.handle("run the bounded transient retention proof")
        approval_ids = [
            item.metadata.get("approval_id")
            for item in held.tool_results
            if item.metadata.get("requires_confirmation") is True
        ]
        if len(approval_ids) != 1 or type(approval_ids[0]) is not int:
            raise AssertionError(f"sensitive read did not queue one approval: {held}")
        approval_id = approval_ids[0]
        review_pending_runtime_approval(runtime, approval_id)
        runtime.planner = RuleBasedPlanner()
        result = runtime.handle(f"approve approval {approval_id}")
        if PRIVATE_SENTINEL not in result.response:
            raise AssertionError("private result was not returned transiently to the operator")
        trace = result.metadata.get("runtime_trace") or {}
        if PRIVATE_SENTINEL in str(trace):
            raise AssertionError("runtime trace retained private result content")
        if PERSISTENCE_SUMMARY not in str(trace):
            raise AssertionError("runtime trace missed the content-free retention receipt")

        rows = runtime.store.recent_tool_runs(limit=5)
        approved_rows = [row for row in rows if str(row["output"]) == PERSISTENCE_SUMMARY]
        if len(approved_rows) != 1:
            raise AssertionError(f"tool audit did not retain only the safe summary: {rows}")
        audit_metadata = str(approved_rows[0]["metadata"])
        for expected in (
            '"suppress_output_persistence": true',
            '"private_content_retained": false',
            '"content_retention": "transient_only"',
        ):
            if expected not in audit_metadata:
                raise AssertionError(f"tool audit missed retention fact {expected}: {audit_metadata}")
        assistant_rows = [
            row for row in runtime.store.recent_messages(limit=5) if row["role"] == "assistant"
        ]
        if not assistant_rows or any(
            PRIVATE_SENTINEL in str(row["content"]) for row in assistant_rows
        ):
            raise AssertionError("assistant message retained private result content")
        if not any(PARENT_PERSISTENCE_SUMMARY in str(row["content"]) for row in assistant_rows):
            raise AssertionError("assistant message missed the content-free retention receipt")
        if any(PRIVATE_SENTINEL in str(entry) for entry in runtime.chat.history):
            raise AssertionError("model chat history retained private result content")
        if _database_contains(root / "jarvis.sqlite", PRIVATE_SENTINEL):
            raise AssertionError("private result content survived in SQLite")

    print("PASS: transient personal-data output is returned live but absent from durable/runtime history")


if __name__ == "__main__":
    main()
