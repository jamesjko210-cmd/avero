"""Offline smoke coverage for the staged exhaustive error-guidance inventory."""

from __future__ import annotations

import ast
from pathlib import Path

from jarvis_v2.agent.failure_guidance import (
    EXTERNAL_INFORMATION_RECOVERY_ACTION,
    KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    MAX_FAILURE_GUIDANCE_ACTION_CHARS,
    MAX_FAILURE_GUIDANCE_COMMAND_CHARS,
    MAX_FAILURE_GUIDANCE_COMMANDS,
    PERSONAL_READ_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    declare_failure_guidance,
    declare_known_not_sent_failure,
    declare_retryable_local_read_failure,
    declare_resource_not_found_failure,
    declare_retryable_external_information_failure,
    declare_retryable_personal_read_failure,
)
from jarvis_v2.config import load_config
from jarvis_v2.scripts.error_guidance_inventory import (
    EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES,
    EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES,
    EXPECTED_EXTERNAL_INFORMATION_SITES,
    EXPECTED_EXTERNAL_INFORMATION_TOOLS,
    EXPECTED_LOCAL_PRODUCTIVITY_READ_SITES,
    EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS,
    EXPECTED_LOCAL_READ_INPUT_SITES,
    EXPECTED_LOCAL_READ_INPUT_TOOLS,
    EXPECTED_OFFLINE_UTILITY_INPUT_SITES,
    EXPECTED_OFFLINE_UTILITY_INPUT_TOOLS,
    EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS,
    EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS,
    EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS,
    EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS,
    EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS,
    EXPECTED_PERSONAL_READ_FAILURE_SITES,
    EXPECTED_PERSONAL_READ_FAILURE_TOOLS,
    EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS,
    EXPECTED_RESOURCE_NOT_FOUND_PATHS,
    EXPECTED_RESOURCE_NOT_FOUND_SITES,
    _TOOL_RESULT_CONTEXTS,
    _direct_guidance_kind,
    _normalized_output_surface_fingerprint,
    _parent_map,
    _tool_result_context,
    _tool_result_ok_kind,
    build_failure_inventory,
    format_failure_inventory,
)
from jarvis_v2.tools import air_connector as air
from jarvis_v2.tools import dictionary_connector as dictionary
from jarvis_v2.tools import history_connector as history
from jarvis_v2.tools import sun_connector as sun


PRIVATE_MARKERS = (
    "/\x55sers/example/private/guidance.txt",
    "/private/guidance.txt",
    "/var/folders/guidance.txt",
    "/tmp/guidance.txt",
    "s" + "k_live_SUPERSECRET123",
    "g" + "hp_SUPERSECRET123",
    "x" + "oxb-SUPERSECRET123",
    "123456789" + ":SUPERSECRET123456789012345",
)


def _expect_value_error(label: str, operation) -> None:
    try:
        operation()
    except ValueError:
        return
    raise SystemExit(f"{label} did not fail closed")


def test_guidance_declaration_schema() -> None:
    output = (
        "The tool is unavailable. Run `setup check`, correct the reported setup issue, "
        "then retry the request."
    )
    original = {"failure_kind": "unavailable"}
    declared = declare_failure_guidance(
        original,
        output=output,
        action="Run `setup check`, correct the reported setup issue, then retry the request.",
        commands=("setup check",),
    )
    if original != {"failure_kind": "unavailable"}:
        raise SystemExit("guidance declaration mutated caller metadata")
    guidance = declared.get("recovery_guidance")
    if not isinstance(guidance, dict):
        raise SystemExit(f"guidance declaration is not machine-readable: {declared}")
    if (
        guidance.get("version") != 1
        or guidance.get("commands") != ["setup check"]
        or declared.get("next_command") != "setup check"
        or declared.get("recovery_commands") != ["setup check"]
    ):
        raise SystemExit(f"guidance declaration aliases drifted: {declared}")

    _expect_value_error(
        "empty action",
        lambda: declare_failure_guidance({}, output="Failed.", action=""),
    )
    _expect_value_error(
        "hidden action",
        lambda: declare_failure_guidance(
            {},
            output="Failed.",
            action="Run setup check.",
        ),
    )
    _expect_value_error(
        "oversized action",
        lambda: declare_failure_guidance(
            {},
            output="a" * (MAX_FAILURE_GUIDANCE_ACTION_CHARS + 1),
            action="a" * (MAX_FAILURE_GUIDANCE_ACTION_CHARS + 1),
        ),
    )
    _expect_value_error(
        "too many commands",
        lambda: declare_failure_guidance(
            {},
            output=" ".join(f"command-{index}" for index in range(MAX_FAILURE_GUIDANCE_COMMANDS + 1)),
            action="command-0",
            commands=tuple(
                f"command-{index}"
                for index in range(MAX_FAILURE_GUIDANCE_COMMANDS + 1)
            ),
        ),
    )
    long_command = "c" * (MAX_FAILURE_GUIDANCE_COMMAND_CHARS + 1)
    _expect_value_error(
        "oversized command",
        lambda: declare_failure_guidance(
            {},
            output=f"Run {long_command}.",
            action=f"Run {long_command}.",
            commands=(long_command,),
        ),
    )
    _expect_value_error(
        "conflicting aliases",
        lambda: declare_failure_guidance(
            {"next_command": "different command"},
            output="Run setup check.",
            action="Run setup check.",
            commands=("setup check",),
        ),
    )
    for marker in PRIVATE_MARKERS:
        _expect_value_error(
            f"private action {marker[:16]}",
            lambda marker=marker: declare_failure_guidance(
                {},
                output=f"Inspect {marker}.",
                action=f"Inspect {marker}.",
            ),
        )
        _expect_value_error(
            f"private command {marker[:16]}",
            lambda marker=marker: declare_failure_guidance(
                {},
                output=f"Run recovery. {marker}",
                action="Run recovery.",
                commands=(marker,),
            ),
        )

    known_output = f"Transport rejected the send. {KNOWN_NOT_SENT_SEND_RECOVERY_ACTION}"
    known = declare_known_not_sent_failure(
        {"retry_safe": False},
        output=known_output,
        action=KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        commands=("setup check",),
    )
    expected_known = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
        "retry_safe": False,
    }
    for key, value in expected_known.items():
        if known.get(key) != value:
            raise SystemExit(
                f"known-not-sent recovery field {key} drifted: {known}"
            )
    if known.get("recovery_guidance") != {
        "version": 1,
        "action": KNOWN_NOT_SENT_SEND_RECOVERY_ACTION,
        "commands": ["setup check"],
    }:
        raise SystemExit(f"known-not-sent recovery declaration drifted: {known}")

    read_output = f"Personal integration unavailable. {PERSONAL_READ_RECOVERY_ACTION}"
    read_failure = declare_retryable_personal_read_failure(
        {"reads_personal_data": True},
        output=read_output,
        action=PERSONAL_READ_RECOVERY_ACTION,
        commands=("setup check",),
    )
    expected_read = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected_read.items():
        if read_failure.get(key) != value:
            raise SystemExit(
                f"personal-read recovery field {key} drifted: {read_failure}"
            )
    if read_failure.get("recovery_guidance") != {
        "version": 1,
        "action": PERSONAL_READ_RECOVERY_ACTION,
        "commands": ["setup check"],
    }:
        raise SystemExit(
            f"personal-read recovery declaration drifted: {read_failure}"
        )

    missing_output = f"Resource missing. {RESOURCE_NOT_FOUND_RECOVERY_ACTION}"
    missing = declare_resource_not_found_failure(
        {"next_command": "list resources", "recovery_commands": ["list resources"]},
        output=missing_output,
        action=RESOURCE_NOT_FOUND_RECOVERY_ACTION,
    )
    expected_missing = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected_missing.items():
        if missing.get(key) != value:
            raise SystemExit(
                f"resource-not-found recovery field {key} drifted: {missing}"
            )
    if missing.get("recovery_guidance") != {
        "version": 1,
        "action": RESOURCE_NOT_FOUND_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(
            f"resource-not-found recovery declaration drifted: {missing}"
        )

    lookup_output = (
        f"Public information unavailable. "
        f"{EXTERNAL_INFORMATION_RECOVERY_ACTION}"
    )
    lookup_failure = declare_retryable_external_information_failure(
        {"calls_external_service": True},
        output=lookup_output,
        action=EXTERNAL_INFORMATION_RECOVERY_ACTION,
        commands=("setup check",),
    )
    expected_lookup = {
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
    for key, value in expected_lookup.items():
        if lookup_failure.get(key) != value:
            raise SystemExit(
                f"external-information recovery field {key} drifted: "
                f"{lookup_failure}"
            )
    if lookup_failure.get("recovery_guidance") != {
        "version": 1,
        "action": EXTERNAL_INFORMATION_RECOVERY_ACTION,
        "commands": ["setup check"],
    }:
        raise SystemExit(
            "external-information recovery declaration drifted: "
            f"{lookup_failure}"
        )

    local_output = f"Local read rejected. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
    local_failure = declare_retryable_local_read_failure(
        {"read_only": True},
        output=local_output,
        action=LOCAL_READ_INPUT_RECOVERY_ACTION,
    )
    expected_local = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": False,
        "retry_safe": True,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected_local.items():
        if local_failure.get(key) != value:
            raise SystemExit(
                f"local-read recovery field {key} drifted: {local_failure}"
            )
    if local_failure.get("recovery_guidance") != {
        "version": 1,
        "action": LOCAL_READ_INPUT_RECOVERY_ACTION,
        "commands": [],
    }:
        raise SystemExit(
            f"local-read recovery declaration drifted: {local_failure}"
        )


def test_candidate_ast_classification() -> None:
    source = '''
def direct(item):
    return ToolResult(
        "direct",
        False,
        f"Failure {item}. Run `setup check`.",
        declare_failure_guidance({}, output="Failure. Run `setup check`.", action="Run `setup check`.", commands=("setup check",)),
    )

def direct_equivalent(other):
    return ToolResult("equivalent", False, f"Failure {other}. Run `setup check`.", {})

def assigned():
    result = ToolResult("assigned", False, "Assigned failure.", {})
    return result

def wrapped():
    return normalize(ToolResult("wrapped", False, "Wrapped failure.", {}))

def dynamic(ok):
    return ToolResult("dynamic", ok, "Dynamic outcome.", {})

def success():
    return ToolResult("success", True, "Complete.", {})
'''
    tree = ast.parse(source)
    parents = _parent_map(tree)
    calls: dict[str, ast.Call] = {}
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "ToolResult":
                calls[stack[-1]] = node
            self.generic_visit(node)

    Visitor().visit(tree)
    expected = {
        "direct": ("literal_failure", "direct_return"),
        "direct_equivalent": ("literal_failure", "direct_return"),
        "assigned": ("literal_failure", "assigned"),
        "wrapped": ("literal_failure", "wrapped_return"),
        "dynamic": ("dynamic", "direct_return"),
        "success": ("literal_success", "direct_return"),
    }
    if set(calls) != set(expected):
        raise SystemExit(f"synthetic ToolResult discovery drifted: {sorted(calls)}")
    for name, (ok_kind, context) in expected.items():
        node = calls[name]
        if _tool_result_ok_kind(node) != ok_kind:
            raise SystemExit(
                f"synthetic {name} ok classification drifted: "
                f"{_tool_result_ok_kind(node)}"
            )
        if _tool_result_context(node, parents) != context:
            raise SystemExit(
                f"synthetic {name} context drifted: "
                f"{_tool_result_context(node, parents)}"
            )
    if _direct_guidance_kind(calls["direct"]) != "declare_failure_guidance":
        raise SystemExit("direct canonical guidance declaration was not classified")
    for name in ("direct_equivalent", "assigned", "wrapped", "dynamic", "success"):
        if _direct_guidance_kind(calls[name]):
            raise SystemExit(f"synthetic {name} invented a guidance classification")
    if (
        _normalized_output_surface_fingerprint(calls["direct"])
        != _normalized_output_surface_fingerprint(calls["direct_equivalent"])
    ):
        raise SystemExit("equivalent f-string outputs did not share a normalized surface")
    if (
        _normalized_output_surface_fingerprint(calls["assigned"])
        == _normalized_output_surface_fingerprint(calls["wrapped"])
    ):
        raise SystemExit("different public outputs collapsed to one normalized surface")


def test_production_failure_inventory() -> None:
    project_root = Path(__file__).resolve().parents[2]
    inventory = build_failure_inventory(project_root)
    problems = (
        *inventory.central_problems,
        *inventory.contact_resolution_problems,
        *inventory.post_attempt_call_problems,
        *inventory.post_attempt_send_problems,
        *inventory.personal_read_problems,
        *inventory.local_productivity_read_problems,
        *inventory.local_read_input_problems,
        *inventory.offline_utility_input_problems,
        *inventory.external_information_problems,
        *inventory.resource_not_found_problems,
        *inventory.post_attempt_writer_problems,
        *inventory.post_attempt_calendar_mutation_problems,
        *inventory.post_attempt_reminder_mutation_problems,
        *inventory.reconciliation_guidance_problems,
    )
    if problems:
        raise SystemExit("; ".join(problems))
    central_count = len(inventory.literal_failures) - len(inventory.legacy_failures)
    if central_count != EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES:
        raise SystemExit(
            f"central executor enforced count drifted: "
            f"{central_count}/{EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES}"
        )
    if (
        set(inventory.contact_resolution_guidance_statuses)
        != EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES
        or len(inventory.contact_resolution_guidance_statuses)
        != len(EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES)
    ):
        raise SystemExit(
            "contact-resolution enforced scope drifted: "
            f"{inventory.contact_resolution_guidance_statuses}"
        )
    if (
        set(inventory.post_attempt_call_failure_tools)
        != EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS
        or len(inventory.post_attempt_call_failure_tools)
        != len(EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS)
    ):
        raise SystemExit(
            "post-attempt call enforced scope drifted: "
            f"{inventory.post_attempt_call_failure_tools}"
        )
    if (
        set(inventory.post_attempt_send_failure_tools)
        != EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS
        or len(inventory.post_attempt_send_failure_tools)
        != len(EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS)
    ):
        raise SystemExit(
            "post-attempt send enforced scope drifted: "
            f"{inventory.post_attempt_send_failure_tools}"
        )
    if (
        set(inventory.post_attempt_writer_failure_tools)
        != EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS
        or len(inventory.post_attempt_writer_failure_tools)
        != len(EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS)
    ):
        raise SystemExit(
            "post-attempt writer enforced scope drifted: "
            f"{inventory.post_attempt_writer_failure_tools}"
        )
    if (
        set(inventory.personal_read_failure_tools)
        != EXPECTED_PERSONAL_READ_FAILURE_TOOLS
        or len(inventory.personal_read_failure_tools)
        != len(EXPECTED_PERSONAL_READ_FAILURE_TOOLS)
        or inventory.personal_read_failure_site_count
        != EXPECTED_PERSONAL_READ_FAILURE_SITES
    ):
        raise SystemExit(
            "personal-read enforced scope drifted: "
            f"{inventory.personal_read_failure_site_count} sites across "
            f"{inventory.personal_read_failure_tools}"
        )
    if (
        set(inventory.local_productivity_read_tools)
        != EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS
        or len(inventory.local_productivity_read_tools)
        != len(EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS)
        or inventory.local_productivity_read_site_count
        != EXPECTED_LOCAL_PRODUCTIVITY_READ_SITES
    ):
        raise SystemExit(
            "local-productivity read enforced scope drifted: "
            f"{inventory.local_productivity_read_site_count} sites across "
            f"{inventory.local_productivity_read_tools}"
        )
    if (
        set(inventory.local_read_input_tools)
        != EXPECTED_LOCAL_READ_INPUT_TOOLS
        or len(inventory.local_read_input_tools)
        != len(EXPECTED_LOCAL_READ_INPUT_TOOLS)
        or inventory.local_read_input_site_count
        != EXPECTED_LOCAL_READ_INPUT_SITES
    ):
        raise SystemExit(
            "local-read input enforced scope drifted: "
            f"{inventory.local_read_input_site_count} sites across "
            f"{inventory.local_read_input_tools}"
        )
    if (
        set(inventory.offline_utility_input_tools)
        != EXPECTED_OFFLINE_UTILITY_INPUT_TOOLS
        or len(inventory.offline_utility_input_tools)
        != len(EXPECTED_OFFLINE_UTILITY_INPUT_TOOLS)
        or inventory.offline_utility_input_site_count
        != EXPECTED_OFFLINE_UTILITY_INPUT_SITES
    ):
        raise SystemExit(
            "offline-utility input enforced scope drifted: "
            f"{inventory.offline_utility_input_site_count} sites across "
            f"{inventory.offline_utility_input_tools}"
        )
    if (
        set(inventory.resource_not_found_paths)
        != EXPECTED_RESOURCE_NOT_FOUND_PATHS
        or len(inventory.resource_not_found_paths)
        != len(EXPECTED_RESOURCE_NOT_FOUND_PATHS)
        or inventory.resource_not_found_site_count
        != EXPECTED_RESOURCE_NOT_FOUND_SITES
    ):
        raise SystemExit(
            "resource-not-found enforced scope drifted: "
            f"{inventory.resource_not_found_site_count} sites across "
            f"{inventory.resource_not_found_paths}"
        )
    if (
        set(inventory.external_information_tools)
        != EXPECTED_EXTERNAL_INFORMATION_TOOLS
        or len(inventory.external_information_tools)
        != len(EXPECTED_EXTERNAL_INFORMATION_TOOLS)
        or inventory.external_information_site_count
        != EXPECTED_EXTERNAL_INFORMATION_SITES
    ):
        raise SystemExit(
            "external-information enforced scope drifted: "
            f"{inventory.external_information_site_count} sites across "
            f"{inventory.external_information_tools}"
        )
    if (
        set(inventory.post_attempt_calendar_mutation_failure_tools)
        != EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS
        or len(inventory.post_attempt_calendar_mutation_failure_tools)
        != len(EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS)
    ):
        raise SystemExit(
            "post-attempt calendar mutation enforced scope drifted: "
            f"{inventory.post_attempt_calendar_mutation_failure_tools}"
        )
    if (
        set(inventory.post_attempt_reminder_mutation_failure_tools)
        != EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS
        or len(inventory.post_attempt_reminder_mutation_failure_tools)
        != len(EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS)
    ):
        raise SystemExit(
            "post-attempt reminder mutation enforced scope drifted: "
            f"{inventory.post_attempt_reminder_mutation_failure_tools}"
        )
    if (
        set(inventory.reconciliation_guidance_failure_kinds)
        != EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS
        or len(inventory.reconciliation_guidance_failure_kinds)
        != len(EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS)
    ):
        raise SystemExit(
            "auto-mutation reconciliation enforced scope drifted: "
            f"{inventory.reconciliation_guidance_failure_kinds}"
        )
    if len(inventory.helper_declarations) < 1:
        raise SystemExit("production recovery guidance has no explicit helper declaration")
    if not inventory.legacy_failures:
        raise SystemExit("legacy inventory unexpectedly empty; review scope before claiming closure")
    allowed_contexts = set(_TOOL_RESULT_CONTEXTS)
    legacy_contexts = dict(inventory.legacy_context_counts())
    dynamic_contexts = dict(inventory.dynamic_context_counts())
    if (
        not legacy_contexts
        or not set(legacy_contexts).issubset(allowed_contexts)
        or sum(legacy_contexts.values()) != len(inventory.legacy_failures)
    ):
        raise SystemExit(
            f"legacy candidate context partition drifted: {legacy_contexts}"
        )
    if (
        not dynamic_contexts
        or not set(dynamic_contexts).issubset(allowed_contexts)
        or sum(dynamic_contexts.values()) != len(inventory.dynamic_results)
    ):
        raise SystemExit(
            f"dynamic candidate context partition drifted: {dynamic_contexts}"
        )
    if any(not site.surface_fingerprint for site in inventory.legacy_failures):
        raise SystemExit("legacy candidate is missing its normalized output surface")
    if any(not site.surface_fingerprint for site in inventory.dynamic_results):
        raise SystemExit("dynamic candidate is missing its normalized output surface")
    if not (0 < inventory.legacy_surface_count <= len(inventory.legacy_failures)):
        raise SystemExit(
            f"legacy normalized surface count drifted: {inventory.legacy_surface_count}"
        )
    if not (
        0
        < len(inventory.directly_declared_legacy_failures)
        <= len(inventory.legacy_failures)
    ):
        raise SystemExit(
            "directly declared reviewed lower bound is missing or exceeds raw candidates"
        )
    if inventory.literal_success_count < 1:
        raise SystemExit("literal success constructors disappeared from the source partition")
    report = format_failure_inventory(inventory)
    personal_read_report_line = (
        "enforced retryable personal-read failures: "
        f"{EXPECTED_PERSONAL_READ_FAILURE_SITES} across "
        f"{len(EXPECTED_PERSONAL_READ_FAILURE_TOOLS)} tools"
    )
    local_productivity_report_line = (
        "enforced local-productivity read failures: "
        f"{EXPECTED_LOCAL_PRODUCTIVITY_READ_SITES} across "
        f"{len(EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS)} paths"
    )
    local_read_input_report_line = (
        "enforced local read-input failures: "
        f"{EXPECTED_LOCAL_READ_INPUT_SITES} across "
        f"{len(EXPECTED_LOCAL_READ_INPUT_TOOLS)} tools"
    )
    if (
        "raw legacy literal failure candidates (includes enforced/reviewed sites; not residual unguided debt)" not in report
        or "reviewed lower bound with direct canonical guidance declarations" not in report
        or "normalized output surfaces among raw legacy literal candidates" not in report
        or "raw legacy literal candidate contexts" not in report
        or "raw dynamic-result candidates (mixed outcomes; not residual unguided debt)" not in report
        or "raw dynamic candidate contexts" not in report
        or "literal success constructors (excluded from failure candidates)" not in report
        or "acceptance scope: still open" not in report
        or "legacy literal failures awaiting explicit classification" in report
        or "dynamic ToolResult sites awaiting classification" in report
        or "enforced pre-approval contact-resolution failures: 5" not in report
        or "enforced post-attempt call failures: 4" not in report
        or "enforced post-attempt send failures: 5" not in report
        or personal_read_report_line not in report
        or local_productivity_report_line not in report
        or local_read_input_report_line not in report
        or "enforced offline-utility input failures: 11 across 4 tools" not in report
        or "enforced external-information fetch failures: 14 across 10 tools" not in report
        or "enforced resource-not-found failures: 5 across 5 paths" not in report
        or "enforced post-attempt writer failures: 2" not in report
        or "enforced post-attempt calendar mutation failures: 3" not in report
        or "enforced post-attempt reminder mutation failures: 1" not in report
        or "enforced auto-mutation reconciliation failures: 4" not in report
        or "/\x55sers/" in report
        or "/private/" in report
        or "/var/folders/" in report
        or "/tmp/" in report
    ):
        raise SystemExit(f"inventory report lost bounded debt/privacy labels: {report}")


def _assert_external_information_failure(result, label: str) -> None:
    if result.ok:
        raise SystemExit(f"{label} unexpectedly succeeded: {result}")
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
        if result.metadata.get(key) is not value:
            raise SystemExit(
                f"{label} external-information field {key} drifted: "
                f"{result.metadata}"
            )
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": EXTERNAL_INFORMATION_RECOVERY_ACTION,
        "commands": ["setup check"],
    }:
        raise SystemExit(
            f"{label} external-information guidance drifted: {result.metadata}"
        )
    if (
        EXTERNAL_INFORMATION_RECOVERY_ACTION not in result.output
        or result.metadata.get("next_command") != "setup check"
        or result.metadata.get("recovery_commands") != ["setup check"]
    ):
        raise SystemExit(
            f"{label} external-information aliases/output drifted: {result}"
        )
    combined = f"{result.output}\n{result.metadata}".lower()
    for marker in (*PRIVATE_MARKERS, "traceback"):
        if marker.lower() in combined:
            raise SystemExit(f"{label} leaked private failure detail: {combined}")


def test_mocked_external_information_failures() -> None:
    config = load_config()
    originals = {
        "air": air._fetch,
        "sun": sun._fetch,
        "history": history._fetch,
        "dictionary": dictionary._fetch,
    }

    def fail(*_args, **_kwargs):
        raise TimeoutError("request timed out near /private/guidance.txt")

    try:
        air._fetch = lambda _lat, _lng: {"current": {}}  # type: ignore[assignment]
        air_tool = air.make_air_tools(config)[0].handler
        _assert_external_information_failure(
            air_tool({}),
            "air missing data",
        )
        air._fetch = fail  # type: ignore[assignment]
        _assert_external_information_failure(
            air_tool({}),
            "air fetch exception",
        )

        sun._fetch = lambda _lat, _lng, _day: {"results": {}}  # type: ignore[assignment]
        sun_tool = sun.make_sun_tools(config)[0].handler
        _assert_external_information_failure(
            sun_tool({}),
            "sun missing data",
        )
        sun._fetch = fail  # type: ignore[assignment]
        _assert_external_information_failure(
            sun_tool({}),
            "sun fetch exception",
        )

        history._fetch = fail  # type: ignore[assignment]
        _assert_external_information_failure(
            history.make_history_tools(config)[0].handler({}),
            "history fetch exception",
        )

        dictionary._fetch = fail  # type: ignore[assignment]
        _assert_external_information_failure(
            dictionary.make_dictionary_tools(config)[0].handler({"word": "privacy"}),
            "dictionary fetch exception",
        )
    finally:
        air._fetch = originals["air"]  # type: ignore[assignment]
        sun._fetch = originals["sun"]  # type: ignore[assignment]
        history._fetch = originals["history"]  # type: ignore[assignment]
        dictionary._fetch = originals["dictionary"]  # type: ignore[assignment]


def main() -> None:
    test_guidance_declaration_schema()
    test_candidate_ast_classification()
    test_production_failure_inventory()
    test_mocked_external_information_failures()
    inventory = build_failure_inventory(Path(__file__).resolve().parents[2])
    print(
        "Error guidance inventory smoke passed: "
        f"{EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES} central executor failures enforced; "
        f"{len(inventory.contact_resolution_guidance_statuses)} contact-resolution failures enforced; "
        f"{len(inventory.legacy_failures)} raw legacy literal candidates; "
        f"{len(inventory.dynamic_results)} raw dynamic candidates; "
        f"{len(inventory.directly_declared_legacy_failures)} direct canonical declarations; "
        "global acceptance remains open."
    )


if __name__ == "__main__":
    main()
