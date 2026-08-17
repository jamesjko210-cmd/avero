from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from jarvis_v2.agent.types import Plan, PlannedAction, RiskLevel, ToolResult
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import email_connector as email_module
from jarvis_v2.tools.registry import TOOL_ARGUMENT_CONTRACT_VERSION


EMAIL_READ_TOOLS = ("read_emails", "search_emails", "read_email_body")
PRIVATE_SENTINEL = "gmail-contract-private-sentinel"

EXPECTED_SCHEMAS = {
    "read_emails": (
        RiskLevel.LOCAL_SAFE,
        (
            ("limit", ("integer",), False, 1, 20),
            ("unread", ("boolean",), False, None, None),
        ),
    ),
    "search_emails": (
        RiskLevel.PERSONAL_DATA,
        (
            ("from", ("string",), False, None, None),
            ("sender", ("string",), False, None, None),
            ("subject", ("string",), False, None, None),
            ("text", ("string",), False, None, None),
            ("query", ("string",), False, None, None),
            ("about", ("string",), False, None, None),
            ("limit", ("integer",), False, 1, 20),
        ),
    ),
    "read_email_body": (
        RiskLevel.PERSONAL_DATA,
        (
            ("from", ("string",), False, None, None),
            ("sender", ("string",), False, None, None),
            ("subject", ("string",), False, None, None),
            ("text", ("string",), False, None, None),
            ("query", ("string",), False, None, None),
            ("about", ("string",), False, None, None),
            ("index", ("integer",), False, 1, 20),
            ("limit", ("integer",), False, 1, 20),
        ),
    ),
}


class StaticPlanner:
    def __init__(self, action: PlannedAction):
        self.action = action

    def plan(self, _user_input: str) -> Plan:
        return Plan("Exercise Gmail read argument preflight.", [self.action])


def _new_probe() -> dict[str, Any]:
    return {
        "credentials": 0,
        "imap": 0,
        "handlers": {name: 0 for name in EMAIL_READ_TOOLS},
        "searches": [],
    }


def _probe_snapshot(probe: dict[str, Any]) -> tuple[Any, ...]:
    return (
        probe["credentials"],
        probe["imap"],
        tuple((name, probe["handlers"][name]) for name in EMAIL_READ_TOOLS),
        tuple(probe["searches"]),
    )


def _install_counting_handlers(runtime: Any, probe: dict[str, Any]) -> None:
    for name in EMAIL_READ_TOOLS:
        tool = runtime.registry.get(name)
        original_handler = tool.handler

        def handler(
            args: dict[str, Any],
            *,
            tool_name: str = name,
            delegate: Any = original_handler,
        ) -> ToolResult:
            probe["handlers"][tool_name] += 1
            return delegate(args)

        runtime.registry._tools[name] = replace(tool, handler=handler)


def _fake_credential_status(probe: dict[str, Any]) -> dict[str, Any]:
    probe["credentials"] += 1
    return {
        "valid": True,
        "reason": "ok",
        "gmail_address": "smoke@example.com",
        "app_password": "x" * 16,
    }


def _fake_imap(probe: dict[str, Any]):
    class EmptyIMAP:
        def __init__(self, host: str, port: int):
            if (host, port) != ("imap.gmail.com", 993):
                raise AssertionError(f"unexpected IMAP endpoint: {host}:{port}")
            probe["imap"] += 1

        def login(self, _address: str, _password: str):
            return "OK", []

        def select(self, mailbox: str, *, readonly: bool = False):
            if mailbox != "INBOX" or readonly is not True:
                raise AssertionError("Gmail read smoke must select INBOX read-only")
            return "OK", []

        def search(self, charset: Any, criterion: str):
            if charset is not None:
                raise AssertionError("Gmail read smoke expected a server-side charset default")
            probe["searches"].append(criterion)
            return "OK", [b""]

        def logout(self):
            return "BYE", []

    return EmptyIMAP


def _assert_contract_shapes(runtime: Any) -> None:
    for name, (expected_risk, expected_fields) in EXPECTED_SCHEMAS.items():
        tool = runtime.registry.get(name)
        contract = tool.argument_contract
        if tool.risk is not expected_risk or tool.toolset != "personal":
            raise SystemExit(
                f"{name} risk/toolset drifted: {tool.risk!r} / {tool.toolset!r}"
            )
        if (
            contract is None
            or contract.version != TOOL_ARGUMENT_CONTRACT_VERSION
            or contract.allow_unknown is not False
        ):
            raise SystemExit(f"{name} does not have the expected strict contract")
        actual_fields = tuple(
            (
                field.name,
                tuple(sorted(argument_type.value for argument_type in field.types)),
                field.required,
                field.minimum,
                field.maximum,
            )
            for field in contract.fields
        )
        if actual_fields != expected_fields:
            raise SystemExit(f"{name} argument schema drifted: {actual_fields}")
        if not isinstance(contract.fields, tuple) or any(
            not isinstance(field.types, frozenset) for field in contract.fields
        ):
            raise SystemExit(f"{name} argument contract is not structurally immutable")
        try:
            contract.allow_unknown = True  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise SystemExit(f"{name} argument contract can be mutated")


def _assert_argument_rejection(
    result: ToolResult,
    *,
    name: str,
    args: Any,
    expected_status: str,
    expected_keys: list[str],
) -> None:
    if (
        result.ok
        or result.metadata.get("failure_kind") != "tool_arguments_invalid"
        or result.metadata.get("argument_contract_version")
        != TOOL_ARGUMENT_CONTRACT_VERSION
        or result.metadata.get("argument_validation_status") != expected_status
        or result.metadata.get("requires_confirmation") is not False
        or result.metadata.get("handler_invoked") is not False
        or result.metadata.get("executed_handler") is not False
    ):
        raise SystemExit(f"{name} malformed arguments were not stopped: {args} / {result}")
    actual_keys = (
        result.metadata.get("unknown_arg_keys")
        if expected_status == "unknown_arguments"
        else result.metadata.get("type_mismatch_arg_keys")
        if expected_status == "type_mismatch"
        else []
    )
    if actual_keys != expected_keys:
        raise SystemExit(
            f"{name} returned the wrong rejected keys: {args} / {result.metadata}"
        )
    if PRIVATE_SENTINEL in result.output or PRIVATE_SENTINEL in str(result.metadata):
        raise SystemExit("unknown Gmail argument content leaked into rejection output")


def _assert_malformed_preflight(runtime: Any, probe: dict[str, Any]) -> None:
    malformed: tuple[tuple[str, Any, str, list[str]], ...] = (
        (
            "read_emails",
            ["not", "an", "object"],
            "arguments_not_object",
            [],
        ),
        (
            "read_emails",
            {"private_payload": PRIVATE_SENTINEL},
            "unknown_arguments",
            ["<unknown>"],
        ),
        ("read_emails", {"limit": "5"}, "type_mismatch", ["limit"]),
        ("read_emails", {"limit": 5.0}, "type_mismatch", ["limit"]),
        ("read_emails", {"limit": True}, "type_mismatch", ["limit"]),
        ("read_emails", {"limit": 0}, "type_mismatch", ["limit"]),
        ("read_emails", {"limit": 21}, "type_mismatch", ["limit"]),
        ("read_emails", {"unread": "false"}, "type_mismatch", ["unread"]),
        (
            "search_emails",
            {"query": "invoice", "private_payload": PRIVATE_SENTINEL},
            "unknown_arguments",
            ["<unknown>"],
        ),
        (
            "search_emails",
            {"query": "invoice", "limit": "4"},
            "type_mismatch",
            ["limit"],
        ),
        (
            "search_emails",
            {"query": "invoice", "limit": 4.0},
            "type_mismatch",
            ["limit"],
        ),
        (
            "search_emails",
            {"query": "invoice", "limit": False},
            "type_mismatch",
            ["limit"],
        ),
        (
            "search_emails",
            {"query": "invoice", "limit": 0},
            "type_mismatch",
            ["limit"],
        ),
        (
            "search_emails",
            {"query": "invoice", "limit": 21},
            "type_mismatch",
            ["limit"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "private_payload": PRIVATE_SENTINEL},
            "unknown_arguments",
            ["<unknown>"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "index": "2"},
            "type_mismatch",
            ["index"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "index": 2.0},
            "type_mismatch",
            ["index"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "index": True},
            "type_mismatch",
            ["index"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "index": 0},
            "type_mismatch",
            ["index"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "index": 21},
            "type_mismatch",
            ["index"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "limit": "4"},
            "type_mismatch",
            ["limit"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "limit": 4.0},
            "type_mismatch",
            ["limit"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "limit": False},
            "type_mismatch",
            ["limit"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "limit": 0},
            "type_mismatch",
            ["limit"],
        ),
        (
            "read_email_body",
            {"query": "invoice", "limit": 21},
            "type_mismatch",
            ["limit"],
        ),
    )
    for index, (name, args, expected_status, expected_keys) in enumerate(malformed):
        before = _probe_snapshot(probe)
        direct = runtime.executor.execute(
            PlannedAction(name, args, "Gmail read executor preflight smoke.")  # type: ignore[arg-type]
        )
        _assert_argument_rejection(
            direct,
            name=name,
            args=args,
            expected_status=expected_status,
            expected_keys=expected_keys,
        )
        if _probe_snapshot(probe) != before:
            raise SystemExit(f"{name} executor rejection reached Gmail access: {args}")

        runtime.planner = StaticPlanner(
            PlannedAction(name, args, "Gmail read runtime preflight smoke.")  # type: ignore[arg-type]
        )
        handled = runtime.handle(f"exercise malformed Gmail read arguments {index}")
        if len(handled.tool_results) != 1:
            raise SystemExit(f"{name} runtime preflight lost its rejection result")
        runtime_result = handled.tool_results[0]
        _assert_argument_rejection(
            runtime_result,
            name=name,
            args=args,
            expected_status=expected_status,
            expected_keys=expected_keys,
        )
        trace = handled.metadata.get("runtime_trace") or {}
        if trace.get("executed_handler_count") != 0 or _probe_snapshot(probe) != before:
            raise SystemExit(f"{name} runtime rejection reached Gmail access: {args}")


def _assert_valid_executor_inputs(runtime: Any, probe: dict[str, Any]) -> None:
    cases = (
        ("read_emails", {"limit": 3, "unread": True}, False, "UNSEEN"),
        ("search_emails", {"query": "invoice", "limit": 4}, True, 'TEXT "invoice"'),
        (
            "read_email_body",
            {"sender": "alex@example.com", "index": 2, "limit": 4},
            True,
            'FROM "alex@example.com"',
        ),
    )
    for name, args, approved, expected_criterion in cases:
        before_handlers = probe["handlers"][name]
        before_credentials = probe["credentials"]
        before_imap = probe["imap"]
        before_searches = len(probe["searches"])
        result = runtime.executor.execute(
            PlannedAction(name, args, "Gmail read typed compatibility smoke."),
            approved=approved,
        )
        if (
            not result.ok
            or result.metadata.get("handler_invoked") is not True
            or result.metadata.get("executed_handler") is not True
            or result.metadata.get("external_attempted") is not True
            or probe["handlers"][name] != before_handlers + 1
            or probe["credentials"] != before_credentials + 1
            or probe["imap"] != before_imap + 1
            or probe["searches"][before_searches:] != [expected_criterion]
        ):
            raise SystemExit(f"{name} rejected valid typed arguments: {args} / {result}")


def _assert_hostile_search_and_index_fail_closed(
    runtime: Any,
    probe: dict[str, Any],
) -> None:
    hostile = f'{PRIVATE_SENTINEL}"\r\nA1 SELECT INBOX'
    for name in ("search_emails", "read_email_body"):
        before_handler = probe["handlers"][name]
        before_credentials = probe["credentials"]
        before_imap = probe["imap"]
        before_searches = tuple(probe["searches"])
        result = runtime.executor.execute(
            PlannedAction(
                name,
                {"query": hostile},
                "Gmail hostile IMAP criterion smoke.",
            ),
            approved=True,
        )
        if (
            result.ok
            or result.metadata.get("reason") != "invalid_search_term"
            or result.metadata.get("criterion") != "<invalid-search-term>"
            or result.metadata.get("handler_invoked") is not True
            or probe["handlers"][name] != before_handler + 1
            or probe["credentials"] != before_credentials
            or probe["imap"] != before_imap
            or tuple(probe["searches"]) != before_searches
            or PRIVATE_SENTINEL in result.output
            or PRIVATE_SENTINEL in str(result.metadata)
        ):
            raise SystemExit(
                f"{name} hostile criterion crossed the local refusal boundary: {result}"
            )

    for args in (
        {"query": "invoice", "index": 20},
        {"query": "invoice", "index": 20, "limit": 4},
    ):
        before_handler = probe["handlers"]["read_email_body"]
        before_credentials = probe["credentials"]
        before_imap = probe["imap"]
        before_searches = tuple(probe["searches"])
        result = runtime.executor.execute(
            PlannedAction(
                "read_email_body",
                args,
                "Gmail index binding smoke.",
            ),
            approved=True,
        )
        if (
            result.ok
            or result.metadata.get("reason") != "index_exceeds_limit"
            or result.metadata.get("selected_index") != 20
            or result.metadata.get("handler_invoked") is not True
            or probe["handlers"]["read_email_body"] != before_handler + 1
            or probe["credentials"] != before_credentials
            or probe["imap"] != before_imap
            or tuple(probe["searches"]) != before_searches
        ):
            raise SystemExit(
                f"read_email_body transformed an approval-bound message index: {args} / {result}"
            )

    safe_query = 'invoice "Q4" \\ archive'
    before_searches = len(probe["searches"])
    result = runtime.executor.execute(
        PlannedAction(
            "search_emails",
            {"query": safe_query, "limit": 4},
            "Gmail quoted criterion escaping smoke.",
        ),
        approved=True,
    )
    expected_criterion = 'TEXT "invoice \\"Q4\\" \\\\ archive"'
    if (
        not result.ok
        or probe["searches"][before_searches:] != [expected_criterion]
    ):
        raise SystemExit(
            f"safe Gmail quote/backslash escaping drifted: {probe['searches']} / {result}"
        )


def _assert_missing_match_index_never_falls_back(
    runtime: Any,
    probe: dict[str, Any],
) -> None:
    fetches: list[bytes] = []

    class TwoMatchIMAP:
        def __init__(self, host: str, port: int):
            if (host, port) != ("imap.gmail.com", 993):
                raise AssertionError(f"unexpected IMAP endpoint: {host}:{port}")
            probe["imap"] += 1

        def login(self, _address: str, _password: str):
            return "OK", []

        def select(self, mailbox: str, *, readonly: bool = False):
            if mailbox != "INBOX" or readonly is not True:
                raise AssertionError("Gmail body smoke must select INBOX read-only")
            return "OK", []

        def search(self, charset: Any, criterion: str):
            if charset is not None:
                raise AssertionError("Gmail body smoke expected the default charset")
            probe["searches"].append(criterion)
            return "OK", [b"1 2"]

        def fetch(self, message_id: bytes, _spec: str):
            fetches.append(message_id)
            raise AssertionError("an unavailable approved index must not fetch any body")

        def logout(self):
            return "BYE", []

    before_handler = probe["handlers"]["read_email_body"]
    before_credentials = probe["credentials"]
    before_imap = probe["imap"]
    with patch.object(email_module.imaplib, "IMAP4_SSL", TwoMatchIMAP):
        result = runtime.executor.execute(
            PlannedAction(
                "read_email_body",
                {"query": "invoice", "index": 5, "limit": 10},
                "Gmail missing match index smoke.",
            ),
            approved=True,
        )
    handoff = result.metadata.get("email_body_handoff") or {}
    if (
        result.ok
        or result.metadata.get("reason") != "index_not_found"
        or result.metadata.get("selected_index") != 5
        or result.metadata.get("count") != 2
        or result.metadata.get("handler_invoked") is not True
        or handoff.get("status") != "refused"
        or handoff.get("reason") != "index_not_found"
        or handoff.get("selected_index") != 5
        or handoff.get("count") != 2
        or fetches
        or probe["handlers"]["read_email_body"] != before_handler + 1
        or probe["credentials"] != before_credentials + 1
        or probe["imap"] != before_imap + 1
    ):
        raise SystemExit(
            f"read_email_body fell back from an unavailable approved index: {result}"
        )


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-email-read-contract-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _assert_contract_shapes(runtime)
        probe = _new_probe()
        _install_counting_handlers(runtime, probe)
        with (
            patch.object(
                email_module,
                "_credential_status",
                side_effect=lambda: _fake_credential_status(probe),
            ),
            patch.object(email_module.imaplib, "IMAP4_SSL", _fake_imap(probe)),
        ):
            _assert_malformed_preflight(runtime, probe)
            _assert_valid_executor_inputs(runtime, probe)
            _assert_hostile_search_and_index_fail_closed(runtime, probe)
            _assert_missing_match_index_never_falls_back(runtime, probe)
    print("Email read argument contract smoke passed")


if __name__ == "__main__":
    main()
