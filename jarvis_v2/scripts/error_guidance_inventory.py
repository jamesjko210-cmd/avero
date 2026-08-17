"""Deterministic source inventory for user-visible ToolResult failures."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from jarvis_v2.agent.failure_guidance import (
    MAX_FAILURE_GUIDANCE_COMMAND_CHARS,
    MAX_FAILURE_GUIDANCE_COMMANDS,
)


PRODUCTION_ROOT_NAMES = ("agent", "automations", "tools")
CENTRAL_EXECUTOR_PATH = "jarvis_v2/agent/executor.py"
CENTRAL_EXECUTOR_QUALNAME = "Executor.execute"
EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES = 6
CONTACT_RESOLUTION_QUALNAME = "_resolve_contact_action_before_approval"
EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES = frozenset(
    {
        "unavailable",
        "not_found",
        "missing_handle",
        "invalid_phone_handle",
        "missing_phone_handle",
    }
)
CALL_CONNECTOR_PATH = "jarvis_v2/tools/call_connector.py"
EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS = frozenset(
    {"call_contact", "call_kakao", "call_instagram", "call_telegram"}
)
POST_ATTEMPT_SEND_TARGETS = {
    (
        "jarvis_v2/tools/imessage_connector.py",
        "make_imessage_tools.send_imessage",
    ): "send_imessage",
    (
        "jarvis_v2/tools/kakao_connector.py",
        "make_kakao_tools.send_kakao",
    ): "send_kakao",
    (
        "jarvis_v2/tools/instagram_connector.py",
        "make_instagram_tools.send_instagram_dm",
    ): "send_instagram_dm",
    (
        "jarvis_v2/tools/call_connector.py",
        "make_call_tools.send_telegram",
    ): "send_telegram",
    (
        "jarvis_v2/tools/email_connector.py",
        "make_email_tools.send_email",
    ): "send_email",
}
EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS = frozenset(
    POST_ATTEMPT_SEND_TARGETS.values()
)
PERSONAL_READ_TARGETS = {
    (
        "jarvis_v2/tools/imessage_connector.py",
        "make_imessage_tools.read_recent_imessages",
    ): ("read_recent_imessages", 1),
    (
        "jarvis_v2/tools/email_connector.py",
        "make_email_tools.read_emails",
    ): ("read_emails", 2),
    (
        "jarvis_v2/tools/email_connector.py",
        "make_email_tools.search_emails",
    ): ("search_emails", 2),
    (
        "jarvis_v2/tools/email_connector.py",
        "make_email_tools.read_email_body",
    ): ("read_email_body", 2),
    (
        "jarvis_v2/tools/calendar_connector.py",
        "make_calendar_tools.list_calendars",
    ): ("list_calendars", 1),
    (
        "jarvis_v2/tools/calendar_connector.py",
        "make_calendar_tools.list_events",
    ): ("list_events", 1),
    (
        "jarvis_v2/tools/calendar_connector.py",
        "make_calendar_tools.check_availability",
    ): ("check_availability", 1),
    (
        "jarvis_v2/tools/contacts_connector.py",
        "make_contacts_tools.find_contact",
    ): ("find_contact", 1),
}
EXPECTED_PERSONAL_READ_FAILURE_TOOLS = frozenset(
    tool_name for tool_name, _count in PERSONAL_READ_TARGETS.values()
)
EXPECTED_PERSONAL_READ_FAILURE_SITES = sum(
    count for _tool_name, count in PERSONAL_READ_TARGETS.values()
)
LOCAL_PRODUCTIVITY_READ_TARGETS = {
    (
        "jarvis_v2/tools/notes.py",
        "_note_io_failure",
    ): ("note_read_io", 1),
    (
        "jarvis_v2/tools/files.py",
        "read_text_file",
    ): ("read_text_file", 2),
    (
        "jarvis_v2/tools/reminder_tools.py",
        "make_reminder_tools.list_reminders",
    ): ("list_reminders", 2),
    (
        "jarvis_v2/tools/voice.py",
        "list_voices",
    ): ("list_voices", 2),
    (
        "jarvis_v2/tools/ocr.py",
        "make_ocr_tools.ocr_image",
    ): ("ocr_image_setup", 1),
}
EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS = frozenset(
    tool_name
    for tool_name, _count in LOCAL_PRODUCTIVITY_READ_TARGETS.values()
)
EXPECTED_LOCAL_PRODUCTIVITY_READ_SITES = sum(
    count for _tool_name, count in LOCAL_PRODUCTIVITY_READ_TARGETS.values()
)
LOCAL_READ_INPUT_TARGETS = {
    (
        "jarvis_v2/tools/tasks.py",
        "make_task_tools.list_tasks",
    ): ("list_tasks", 1),
    (
        "jarvis_v2/tools/tasks.py",
        "make_task_tools.inspect_task",
    ): ("inspect_task", 1),
    (
        "jarvis_v2/tools/tasks.py",
        "make_task_tools.search_tasks",
    ): ("search_tasks", 2),
    (
        "jarvis_v2/tools/goals.py",
        "make_goal_tools.list_goals",
    ): ("list_goals", 1),
    (
        "jarvis_v2/tools/goals.py",
        "make_goal_tools.goal_status",
    ): ("goal_status", 1),
    (
        "jarvis_v2/tools/people.py",
        "make_people_tools.get_person",
    ): ("get_person", 2),
    (
        "jarvis_v2/tools/preferences.py",
        "make_preference_tools.list_preferences",
    ): ("list_preferences", 1),
    (
        "jarvis_v2/tools/preferences.py",
        "make_preference_tools.get_preference",
    ): ("get_preference", 2),
    (
        "jarvis_v2/tools/voice.py",
        "speak_text",
    ): ("speak", 3),
    (
        "jarvis_v2/tools/ocr.py",
        "make_ocr_tools.ocr_image",
    ): ("ocr_image_input", 3),
}
EXPECTED_LOCAL_READ_INPUT_TOOLS = frozenset(
    tool_name for tool_name, _count in LOCAL_READ_INPUT_TARGETS.values()
)
EXPECTED_LOCAL_READ_INPUT_SITES = sum(
    count for _tool_name, count in LOCAL_READ_INPUT_TARGETS.values()
)
OFFLINE_UTILITY_INPUT_TARGETS = {
    (
        "jarvis_v2/tools/utilities.py",
        "calculate",
    ): ("calculate", 5),
    (
        "jarvis_v2/tools/utilities.py",
        "spell_word",
    ): ("spell_word", 3),
    (
        "jarvis_v2/tools/utilities.py",
        "count_text",
    ): ("count_text", 2),
    (
        "jarvis_v2/tools/countdown_connector.py",
        "make_countdown_tools.days_until",
    ): ("days_until", 1),
}
EXPECTED_OFFLINE_UTILITY_INPUT_TOOLS = frozenset(
    tool_name for tool_name, _count in OFFLINE_UTILITY_INPUT_TARGETS.values()
)
EXPECTED_OFFLINE_UTILITY_INPUT_SITES = sum(
    count for _tool_name, count in OFFLINE_UTILITY_INPUT_TARGETS.values()
)
EXTERNAL_INFORMATION_TARGETS = {
    (
        "jarvis_v2/tools/weather_connector.py",
        "make_weather_tools.get_weather",
    ): ("get_weather", 2),
    (
        "jarvis_v2/tools/wikipedia_connector.py",
        "make_wikipedia_tools.wiki_summary",
    ): ("wiki_summary", 1),
    (
        "jarvis_v2/tools/news_connector.py",
        "make_news_tools.get_news",
    ): ("get_news", 1),
    (
        "jarvis_v2/tools/markets_connector.py",
        "make_markets_tools.get_crypto_price",
    ): ("get_crypto_price", 1),
    (
        "jarvis_v2/tools/markets_connector.py",
        "make_markets_tools.get_stock_price",
    ): ("get_stock_price", 1),
    (
        "jarvis_v2/tools/currency_connector.py",
        "make_currency_tools.convert_currency",
    ): ("convert_currency", 2),
    (
        "jarvis_v2/tools/air_connector.py",
        "make_air_tools.get_air_quality",
    ): ("get_air_quality", 2),
    (
        "jarvis_v2/tools/sun_connector.py",
        "make_sun_tools.get_sun_times",
    ): ("get_sun_times", 2),
    (
        "jarvis_v2/tools/history_connector.py",
        "make_history_tools.on_this_day",
    ): ("on_this_day", 1),
    (
        "jarvis_v2/tools/dictionary_connector.py",
        "make_dictionary_tools.define",
    ): ("define", 1),
}
EXPECTED_EXTERNAL_INFORMATION_TOOLS = frozenset(
    tool_name for tool_name, _count in EXTERNAL_INFORMATION_TARGETS.values()
)
EXPECTED_EXTERNAL_INFORMATION_SITES = sum(
    count for _tool_name, count in EXTERNAL_INFORMATION_TARGETS.values()
)
RESOURCE_NOT_FOUND_TARGETS = {
    (
        "jarvis_v2/tools/tasks.py",
        "_missing_task_result",
    ): "missing_task",
    (
        "jarvis_v2/tools/tasks.py",
        "_note_task_not_found_result",
    ): "missing_task_note",
    (
        "jarvis_v2/tools/notes.py",
        "_missing_note_result",
    ): "missing_note",
    (
        "jarvis_v2/tools/notes.py",
        "make_note_tools.outline_jarvis_note",
    ): "missing_note_outline",
    (
        "jarvis_v2/tools/files.py",
        "_missing_text_file_result",
    ): "missing_text_file",
}
EXPECTED_RESOURCE_NOT_FOUND_PATHS = frozenset(
    RESOURCE_NOT_FOUND_TARGETS.values()
)
EXPECTED_RESOURCE_NOT_FOUND_SITES = len(RESOURCE_NOT_FOUND_TARGETS)
POST_ATTEMPT_WRITER_TARGETS = {
    (
        "jarvis_v2/tools/writer_connector.py",
        "make_writer_tools.paste_text",
    ): "paste_text",
    (
        "jarvis_v2/tools/writer_connector.py",
        "make_writer_tools.human_write",
    ): "human_write",
}
EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS = frozenset(
    POST_ATTEMPT_WRITER_TARGETS.values()
)
POST_ATTEMPT_CALENDAR_MUTATION_TARGETS = {
    (
        "jarvis_v2/tools/calendar_connector.py",
        "make_calendar_tools.create_event",
    ): "create_event",
    (
        "jarvis_v2/tools/calendar_connector.py",
        "make_calendar_tools.update_event",
    ): "update_event",
    (
        "jarvis_v2/tools/calendar_connector.py",
        "make_calendar_tools.delete_event",
    ): "delete_event",
}
EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS = frozenset(
    POST_ATTEMPT_CALENDAR_MUTATION_TARGETS.values()
)
POST_ATTEMPT_REMINDER_MUTATION_TARGETS = {
    (
        "jarvis_v2/tools/personal.py",
        "make_personal_tools.create_reminder.post_attempt_failure",
    ): "create_reminder",
}
EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS = frozenset(
    POST_ATTEMPT_REMINDER_MUTATION_TARGETS.values()
)
AUTO_MUTATION_RECONCILIATION_PATH = (
    "jarvis_v2/tools/auto_mutation_reconciliation.py"
)
EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS = frozenset(
    {
        "malformed_uncertainty_proof",
        "reconciliation_review_required",
        "reconciliation_conflict",
        "reconciliation_review_mismatch",
    }
)
_PRIVATE_FRAGMENTS = ("/" + "Users" + "/", "/private/", "/var/folders/", "/tmp/")
_EXPLICIT_GUIDANCE_DECLARATIONS = frozenset(
    {
        "declare_failure_guidance",
        "declare_known_not_sent_failure",
        "declare_outcome_unknown_failure",
        "declare_resource_not_found_failure",
        "declare_retryable_external_information_failure",
        "declare_retryable_local_read_failure",
        "declare_retryable_personal_read_failure",
    }
)
_TOOL_RESULT_CONTEXTS = frozenset(
    {"direct_return", "assigned", "wrapped_return", "expression"}
)


@dataclass(frozen=True)
class FailureSite:
    path: str
    qualname: str
    lineno: int
    ok_kind: str
    fingerprint: str
    context: str = "expression"
    surface_fingerprint: str = ""
    guidance_kind: str = ""


@dataclass(frozen=True)
class FailureInventory:
    literal_failures: tuple[FailureSite, ...]
    dynamic_results: tuple[FailureSite, ...]
    helper_declarations: tuple[FailureSite, ...]
    central_problems: tuple[str, ...]
    contact_resolution_guidance_statuses: tuple[str, ...]
    contact_resolution_problems: tuple[str, ...]
    post_attempt_call_failure_tools: tuple[str, ...]
    post_attempt_call_problems: tuple[str, ...]
    post_attempt_send_failure_tools: tuple[str, ...]
    post_attempt_send_problems: tuple[str, ...]
    personal_read_failure_tools: tuple[str, ...]
    personal_read_failure_site_count: int
    personal_read_problems: tuple[str, ...]
    local_productivity_read_tools: tuple[str, ...]
    local_productivity_read_site_count: int
    local_productivity_read_problems: tuple[str, ...]
    local_read_input_tools: tuple[str, ...]
    local_read_input_site_count: int
    local_read_input_problems: tuple[str, ...]
    offline_utility_input_tools: tuple[str, ...]
    offline_utility_input_site_count: int
    offline_utility_input_problems: tuple[str, ...]
    external_information_tools: tuple[str, ...]
    external_information_site_count: int
    external_information_problems: tuple[str, ...]
    resource_not_found_paths: tuple[str, ...]
    resource_not_found_site_count: int
    resource_not_found_problems: tuple[str, ...]
    post_attempt_writer_failure_tools: tuple[str, ...]
    post_attempt_writer_problems: tuple[str, ...]
    post_attempt_calendar_mutation_failure_tools: tuple[str, ...]
    post_attempt_calendar_mutation_problems: tuple[str, ...]
    post_attempt_reminder_mutation_failure_tools: tuple[str, ...]
    post_attempt_reminder_mutation_problems: tuple[str, ...]
    reconciliation_guidance_failure_kinds: tuple[str, ...]
    reconciliation_guidance_problems: tuple[str, ...]
    literal_success_count: int

    @property
    def legacy_failures(self) -> tuple[FailureSite, ...]:
        return tuple(
            site
            for site in self.literal_failures
            if not (
                site.path == CENTRAL_EXECUTOR_PATH
                and site.qualname == CENTRAL_EXECUTOR_QUALNAME
            )
        )

    def legacy_counts_by_file(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(site.path for site in self.legacy_failures)
        return tuple(sorted(counts.items()))

    def legacy_context_counts(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(site.context for site in self.legacy_failures)
        return tuple(sorted(counts.items()))

    def dynamic_context_counts(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(site.context for site in self.dynamic_results)
        return tuple(sorted(counts.items()))

    @property
    def legacy_surface_count(self) -> int:
        return len(
            {
                site.surface_fingerprint
                for site in self.legacy_failures
                if site.surface_fingerprint
            }
        )

    @property
    def directly_declared_legacy_failures(self) -> tuple[FailureSite, ...]:
        """Reviewed lower bound, not a residual-debt subtraction.

        These sites bind one canonical declaration in the ToolResult metadata
        expression itself. Other enforced families use reviewed helper/dataflow
        contracts and intentionally remain in the raw candidate inventory.
        """

        return tuple(site for site in self.legacy_failures if site.guidance_kind)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _tool_result_ok_node(node: ast.Call) -> ast.expr | None:
    if len(node.args) >= 2:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg == "ok":
            return keyword.value
    return None


def _tool_result_ok_kind(node: ast.Call) -> str:
    ok_node = _tool_result_ok_node(node)
    if isinstance(ok_node, ast.Constant) and ok_node.value is False:
        return "literal_failure"
    if isinstance(ok_node, ast.Constant) and ok_node.value is True:
        return "literal_success"
    return "dynamic"


def _tool_result_output_node(node: ast.Call) -> ast.expr | None:
    if len(node.args) >= 3:
        return node.args[2]
    for keyword in node.keywords:
        if keyword.arg == "output":
            return keyword.value
    return None


def _tool_result_metadata_node(node: ast.Call) -> ast.expr | None:
    if len(node.args) >= 4:
        return node.args[3]
    for keyword in node.keywords:
        if keyword.arg == "metadata":
            return keyword.value
    return None


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _tool_result_context(
    node: ast.Call,
    parents: dict[ast.AST, ast.AST],
) -> str:
    """Classify syntax only; this does not claim runtime reachability."""

    parent = parents.get(node)
    if isinstance(parent, ast.Return):
        return "direct_return"
    current = parent
    while current is not None and not isinstance(
        current,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
    ):
        if isinstance(current, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            return "assigned"
        if isinstance(current, ast.Return):
            return "wrapped_return"
        current = parents.get(current)
    return "expression"


def _normalized_output_surface_fingerprint(node: ast.Call) -> str:
    output_node = _tool_result_output_node(node)
    static_output = _static_text(output_node)
    if static_output is not None:
        material = "text:" + " ".join(static_output.split())
    elif output_node is None:
        material = "missing"
    else:
        material = "ast:" + ast.dump(
            output_node,
            annotate_fields=True,
            include_attributes=False,
        )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _direct_guidance_kind(node: ast.Call) -> str:
    metadata_node = _tool_result_metadata_node(node)
    if metadata_node is None:
        return ""
    declarations = sorted(
        {
            _call_name(child)
            for child in ast.walk(metadata_node)
            if isinstance(child, ast.Call)
            and _call_name(child) in _EXPLICIT_GUIDANCE_DECLARATIONS
        }
    )
    return "+".join(declarations)


def _static_text(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and type(value.value) is str:
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("<dynamic>")
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_text(node.left)
        right = _static_text(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _literal_string_list(node: ast.AST | None) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or type(item.value) is not str:
            return None
        values.append(item.value)
    return values


def _dict_value(node: ast.AST | None, key: str) -> ast.AST | None:
    if not isinstance(node, ast.Dict):
        return None
    for item_key, item_value in zip(node.keys, node.values):
        if isinstance(item_key, ast.Constant) and item_key.value == key:
            return item_value
    return None


def _keyword_value(node: ast.Call, key: str) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == key:
            return keyword.value
    return None


def _qualnames(tree: ast.AST) -> dict[ast.AST, str]:
    names: dict[ast.AST, str] = {}

    def visit(node: ast.AST, stack: tuple[str, ...]) -> None:
        next_stack = stack
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            next_stack = (*stack, node.name)
        names[node] = ".".join(next_stack) or "<module>"
        for child in ast.iter_child_nodes(node):
            visit(child, next_stack)

    visit(tree, ())
    return names


def _fingerprint(node: ast.Call) -> str:
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _central_executor_problem(node: ast.Call, site: FailureSite) -> str | None:
    output = _static_text(_tool_result_output_node(node))
    metadata = _tool_result_metadata_node(node)
    if output is None:
        return f"{site.path}:{site.lineno} central failure output is not statically bounded"
    commands = _literal_string_list(_dict_value(metadata, "recovery_commands"))
    next_command = _static_text(_dict_value(metadata, "next_command"))
    if commands is None or not commands:
        return f"{site.path}:{site.lineno} central failure lacks literal recovery_commands"
    if len(commands) > MAX_FAILURE_GUIDANCE_COMMANDS:
        return f"{site.path}:{site.lineno} central failure has too many recovery commands"
    if any(
        not command
        or command != command.strip()
        or len(command) > MAX_FAILURE_GUIDANCE_COMMAND_CHARS
        for command in commands
    ):
        return f"{site.path}:{site.lineno} central failure has an invalid recovery command"
    if len(set(commands)) != len(commands):
        return f"{site.path}:{site.lineno} central failure repeats a recovery command"
    if next_command != commands[0]:
        return f"{site.path}:{site.lineno} next_command does not mirror recovery_commands"
    if commands[0] not in output:
        return f"{site.path}:{site.lineno} primary recovery command is absent from visible output"
    combined = "\n".join((output, next_command or "", *commands))
    if any(fragment.lower() in combined.lower() for fragment in _PRIVATE_FRAGMENTS):
        return f"{site.path}:{site.lineno} central guidance contains a local path"
    return None


def _contact_resolution_guidance_problem(
    node: ast.Call,
    *,
    path: str,
) -> tuple[str | None, str | None]:
    status = _static_text(_keyword_value(node, "status"))
    if status not in EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES:
        return status, None
    prefix = f"{path}:{node.lineno} contact-resolution {status}"
    output = _static_text(_keyword_value(node, "output"))
    action = _static_text(_keyword_value(node, "recovery_action"))
    commands = _literal_string_list(_keyword_value(node, "recovery_commands"))
    if output is None:
        return status, f"{prefix} output is not statically bounded"
    if action is None or not action.strip():
        return status, f"{prefix} lacks a literal recovery action"
    if action not in output:
        return status, f"{prefix} recovery action is absent from visible output"
    if commands is None or not commands:
        return status, f"{prefix} lacks literal recovery commands"
    if len(commands) > MAX_FAILURE_GUIDANCE_COMMANDS:
        return status, f"{prefix} has too many recovery commands"
    if any(
        not command
        or command != command.strip()
        or len(command) > MAX_FAILURE_GUIDANCE_COMMAND_CHARS
        for command in commands
    ):
        return status, f"{prefix} has an invalid recovery command"
    if len(set(commands)) != len(commands):
        return status, f"{prefix} repeats a recovery command"
    if any(command not in output for command in commands):
        return status, f"{prefix} hides a recovery command from visible output"
    combined = "\n".join((output, action, *commands))
    if any(fragment.lower() in combined.lower() for fragment in _PRIVATE_FRAGMENTS):
        return status, f"{prefix} contains a local path"
    return status, None


def _post_attempt_call_problem(
    node: ast.Call,
    *,
    path: str,
    label: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {label}"
    output = _tool_result_output_node(node)
    metadata = _tool_result_metadata_node(node)
    output_is_fixed = (
        isinstance(output, ast.Name)
        and output.id == "failure_output"
    ) or (
        isinstance(output, ast.Call)
        and _call_name(output) == "_call_attempt_recovery_guidance"
    )
    if not output_is_fixed:
        return f"{prefix} does not use fixed post-attempt recovery guidance"
    if not isinstance(metadata, ast.Call) or _call_name(metadata) != "_call_attempt_failure_metadata":
        return f"{prefix} does not declare outcome-unknown recovery metadata"
    return None


def _post_attempt_send_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} post-attempt send declaration"
    output = _keyword_value(node, "output")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _known_not_sent_send_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} known-not-sent declaration"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if (
        not isinstance(action, ast.Name)
        or action.id != "KNOWN_NOT_SENT_SEND_RECOVERY_ACTION"
    ):
        return f"{prefix} does not bind the canonical send recovery action"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _personal_read_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} personal-read declaration"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if (
        not isinstance(action, ast.Name)
        or action.id != "PERSONAL_READ_RECOVERY_ACTION"
    ):
        return f"{prefix} does not bind the canonical read recovery action"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _local_productivity_read_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} local-productivity declaration"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id not in {
        "failure_output",
        "output",
    }:
        return f"{prefix} does not bind a reviewed failure output"
    if (
        not isinstance(action, ast.Name)
        or action.id != "LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION"
    ):
        return f"{prefix} does not bind the canonical local recovery action"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _local_read_input_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} local-read input declaration"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    commands = _keyword_value(node, "commands")
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if (
        not isinstance(action, ast.Name)
        or action.id != "LOCAL_READ_INPUT_RECOVERY_ACTION"
    ):
        return f"{prefix} does not bind the canonical local-read recovery action"
    if commands is not None:
        return f"{prefix} unexpectedly declares a retry command"
    return None


def _external_information_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} external-information declaration"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if (
        not isinstance(action, ast.Name)
        or action.id != "EXTERNAL_INFORMATION_RECOVERY_ACTION"
    ):
        return f"{prefix} does not bind the canonical lookup recovery action"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _resource_not_found_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    label: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {label} resource-not-found declaration"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    if not isinstance(output, ast.Name) or output.id != "output":
        return f"{prefix} does not bind the fixed public output"
    if (
        not isinstance(action, ast.Name)
        or action.id != "RESOURCE_NOT_FOUND_RECOVERY_ACTION"
    ):
        return f"{prefix} does not bind the canonical recovery action"
    return None


def _post_attempt_writer_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} post-attempt writer declaration"
    output = _keyword_value(node, "output")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _post_attempt_calendar_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} post-attempt calendar declaration"
    output = _keyword_value(node, "output")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if commands != ["what is on my calendar <date>", "recent tool runs"]:
        return f"{prefix} does not pin the bounded calendar recovery commands"
    return None


def _post_attempt_reminder_declaration_problem(
    node: ast.Call,
    *,
    path: str,
    tool_name: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} {tool_name} post-attempt reminder declaration"
    output = _keyword_value(node, "output")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id != "failure_output":
        return f"{prefix} does not bind the fixed failure output"
    if commands != ["setup check"]:
        return f"{prefix} does not pin the bounded setup recovery command"
    return None


def _reconciliation_guidance_problem(
    node: ast.Call,
    *,
    path: str,
    failure_kind: str,
) -> str | None:
    prefix = f"{path}:{node.lineno} reconciliation {failure_kind}"
    output = _keyword_value(node, "output")
    action = _keyword_value(node, "action")
    commands = _literal_string_list(_keyword_value(node, "commands"))
    if not isinstance(output, ast.Name) or output.id not in {
        "output",
        "recovery_action",
    }:
        return f"{prefix} does not bind a reviewed public output"
    if not isinstance(action, ast.Name) or action.id != "recovery_action":
        return f"{prefix} does not bind the fixed recovery action"
    if not commands:
        return f"{prefix} lacks literal recovery commands"
    if len(commands) > MAX_FAILURE_GUIDANCE_COMMANDS:
        return f"{prefix} has too many recovery commands"
    if any(
        not command
        or command != command.strip()
        or len(command) > MAX_FAILURE_GUIDANCE_COMMAND_CHARS
        for command in commands
    ):
        return f"{prefix} has an invalid recovery command"
    return None


def build_failure_inventory(project_root: Path) -> FailureInventory:
    literal_failures: list[FailureSite] = []
    dynamic_results: list[FailureSite] = []
    helper_declarations: list[FailureSite] = []
    literal_success_count = 0
    central_problems: list[str] = []
    contact_resolution_guidance_statuses: list[str] = []
    contact_resolution_problems: list[str] = []
    post_attempt_call_failure_tools: list[str] = []
    post_attempt_call_problems: list[str] = []
    post_attempt_send_failure_tools: list[str] = []
    post_attempt_send_problems: list[str] = []
    post_attempt_send_declarations: Counter[str] = Counter()
    known_not_sent_send_declarations: Counter[str] = Counter()
    post_attempt_send_guidance_calls: Counter[str] = Counter()
    personal_read_failure_tools: list[str] = []
    personal_read_problems: list[str] = []
    personal_read_declarations: Counter[str] = Counter()
    local_productivity_read_tools: list[str] = []
    local_productivity_read_problems: list[str] = []
    local_productivity_read_declarations: Counter[str] = Counter()
    local_read_input_tools: list[str] = []
    local_read_input_problems: list[str] = []
    local_read_input_declarations: Counter[str] = Counter()
    offline_utility_input_tools: list[str] = []
    offline_utility_input_problems: list[str] = []
    offline_utility_input_declarations: Counter[str] = Counter()
    external_information_tools: list[str] = []
    external_information_problems: list[str] = []
    external_information_declarations: Counter[str] = Counter()
    resource_not_found_paths: list[str] = []
    resource_not_found_problems: list[str] = []
    resource_not_found_declarations: Counter[str] = Counter()
    post_attempt_writer_failure_tools: list[str] = []
    post_attempt_writer_problems: list[str] = []
    post_attempt_writer_declarations: Counter[str] = Counter()
    post_attempt_writer_guidance_calls: Counter[str] = Counter()
    post_attempt_calendar_mutation_failure_tools: list[str] = []
    post_attempt_calendar_mutation_problems: list[str] = []
    post_attempt_calendar_declarations: Counter[str] = Counter()
    post_attempt_calendar_guidance_calls: Counter[str] = Counter()
    post_attempt_reminder_mutation_failure_tools: list[str] = []
    post_attempt_reminder_mutation_problems: list[str] = []
    post_attempt_reminder_declarations: Counter[str] = Counter()
    reconciliation_guidance_failure_kinds: list[str] = []
    reconciliation_guidance_problems: list[str] = []
    native_call_failure_sites = 0
    gui_call_failure_sites = 0
    gui_call_factory_tools: set[str] = set()
    telegram_call_declarations = 0
    telegram_call_guidance_calls = 0
    source_paths: list[Path] = []
    for root_name in PRODUCTION_ROOT_NAMES:
        source_paths.extend((project_root / "jarvis_v2" / root_name).rglob("*.py"))
    for path in sorted(set(source_paths)):
        relative = path.relative_to(project_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        qualnames = _qualnames(tree)
        parents = _parent_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node)
            send_tool_name = POST_ATTEMPT_SEND_TARGETS.get(
                (relative, qualnames[node])
            )
            writer_tool_name = POST_ATTEMPT_WRITER_TARGETS.get(
                (relative, qualnames[node])
            )
            calendar_tool_name = POST_ATTEMPT_CALENDAR_MUTATION_TARGETS.get(
                (relative, qualnames[node])
            )
            reminder_tool_name = POST_ATTEMPT_REMINDER_MUTATION_TARGETS.get(
                (relative, qualnames[node])
            )
            personal_read_target = PERSONAL_READ_TARGETS.get(
                (relative, qualnames[node])
            )
            local_productivity_read_target = LOCAL_PRODUCTIVITY_READ_TARGETS.get(
                (relative, qualnames[node])
            )
            local_read_input_target = LOCAL_READ_INPUT_TARGETS.get(
                (relative, qualnames[node])
            )
            offline_utility_input_target = OFFLINE_UTILITY_INPUT_TARGETS.get(
                (relative, qualnames[node])
            )
            external_information_target = EXTERNAL_INFORMATION_TARGETS.get(
                (relative, qualnames[node])
            )
            resource_not_found_target = RESOURCE_NOT_FOUND_TARGETS.get(
                (relative, qualnames[node])
            )
            if send_tool_name and call_name == "declare_outcome_unknown_failure":
                post_attempt_send_declarations[send_tool_name] += 1
                problem = _post_attempt_send_declaration_problem(
                    node,
                    path=relative,
                    tool_name=send_tool_name,
                )
                if problem:
                    post_attempt_send_problems.append(problem)
            if send_tool_name and call_name == "declare_known_not_sent_failure":
                known_not_sent_send_declarations[send_tool_name] += 1
                problem = _known_not_sent_send_declaration_problem(
                    node,
                    path=relative,
                    tool_name=send_tool_name,
                )
                if problem:
                    post_attempt_send_problems.append(problem)
            if send_tool_name and call_name in {
                "_post_attempt_send_recovery_guidance",
                "_post_attempt_telegram_send_recovery_guidance",
                "_gmail_uncertain_send_error",
            }:
                post_attempt_send_guidance_calls[send_tool_name] += 1
            if (
                personal_read_target
                and call_name == "declare_retryable_personal_read_failure"
            ):
                personal_read_tool_name = personal_read_target[0]
                personal_read_declarations[personal_read_tool_name] += 1
                problem = _personal_read_declaration_problem(
                    node,
                    path=relative,
                    tool_name=personal_read_tool_name,
                )
                if problem:
                    personal_read_problems.append(problem)
            if (
                local_productivity_read_target
                and call_name == "declare_retryable_personal_read_failure"
            ):
                local_tool_name = local_productivity_read_target[0]
                local_productivity_read_declarations[local_tool_name] += 1
                problem = _local_productivity_read_declaration_problem(
                    node,
                    path=relative,
                    tool_name=local_tool_name,
                )
                if problem:
                    local_productivity_read_problems.append(problem)
            if (
                local_read_input_target
                and call_name == "declare_retryable_local_read_failure"
            ):
                local_read_tool_name = local_read_input_target[0]
                local_read_input_declarations[local_read_tool_name] += 1
                problem = _local_read_input_declaration_problem(
                    node,
                    path=relative,
                    tool_name=local_read_tool_name,
                )
                if problem:
                    local_read_input_problems.append(problem)
            if (
                offline_utility_input_target
                and call_name == "declare_retryable_local_read_failure"
            ):
                utility_tool_name = offline_utility_input_target[0]
                offline_utility_input_declarations[utility_tool_name] += 1
                problem = _local_read_input_declaration_problem(
                    node,
                    path=relative,
                    tool_name=utility_tool_name,
                )
                if problem:
                    offline_utility_input_problems.append(problem)
            if (
                external_information_target
                and call_name
                == "declare_retryable_external_information_failure"
            ):
                external_tool_name = external_information_target[0]
                external_information_declarations[external_tool_name] += 1
                problem = _external_information_declaration_problem(
                    node,
                    path=relative,
                    tool_name=external_tool_name,
                )
                if problem:
                    external_information_problems.append(problem)
            if (
                resource_not_found_target
                and call_name == "declare_resource_not_found_failure"
            ):
                resource_not_found_declarations[resource_not_found_target] += 1
                problem = _resource_not_found_declaration_problem(
                    node,
                    path=relative,
                    label=resource_not_found_target,
                )
                if problem:
                    resource_not_found_problems.append(problem)
            if writer_tool_name and call_name == "declare_outcome_unknown_failure":
                post_attempt_writer_declarations[writer_tool_name] += 1
                problem = _post_attempt_writer_declaration_problem(
                    node,
                    path=relative,
                    tool_name=writer_tool_name,
                )
                if problem:
                    post_attempt_writer_problems.append(problem)
            if writer_tool_name and call_name == "_post_attempt_writer_recovery_guidance":
                post_attempt_writer_guidance_calls[writer_tool_name] += 1
            if calendar_tool_name and call_name == "declare_outcome_unknown_failure":
                post_attempt_calendar_declarations[calendar_tool_name] += 1
                problem = _post_attempt_calendar_declaration_problem(
                    node,
                    path=relative,
                    tool_name=calendar_tool_name,
                )
                if problem:
                    post_attempt_calendar_mutation_problems.append(problem)
            if calendar_tool_name and call_name == "_calendar_mutation_outcome_unknown_guidance":
                post_attempt_calendar_guidance_calls[calendar_tool_name] += 1
            if reminder_tool_name and call_name == "declare_outcome_unknown_failure":
                post_attempt_reminder_declarations[reminder_tool_name] += 1
                problem = _post_attempt_reminder_declaration_problem(
                    node,
                    path=relative,
                    tool_name=reminder_tool_name,
                )
                if problem:
                    post_attempt_reminder_mutation_problems.append(problem)
            if (
                relative == AUTO_MUTATION_RECONCILIATION_PATH
                and call_name == "_reconciliation_failure_metadata"
            ):
                failure_kind = _static_text(_keyword_value(node, "failure_kind"))
                if failure_kind in EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS:
                    reconciliation_guidance_failure_kinds.append(failure_kind)
                    problem = _reconciliation_guidance_problem(
                        node,
                        path=relative,
                        failure_kind=failure_kind,
                    )
                    if problem:
                        reconciliation_guidance_problems.append(problem)
            if call_name == "declare_failure_guidance":
                helper_declarations.append(
                    FailureSite(
                        relative,
                        qualnames[node],
                        node.lineno,
                        "helper",
                        _fingerprint(node),
                    )
                )
            if (
                call_name == "_contact_resolution_failure"
                and relative == CENTRAL_EXECUTOR_PATH
                and qualnames[node] == CONTACT_RESOLUTION_QUALNAME
            ):
                status, problem = _contact_resolution_guidance_problem(
                    node,
                    path=relative,
                )
                if status in EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES:
                    contact_resolution_guidance_statuses.append(status)
                if problem:
                    contact_resolution_problems.append(problem)
            if (
                relative == CALL_CONNECTOR_PATH
                and qualnames[node] == "make_call_tools"
                and call_name == "_gui_call_tool"
                and node.args
            ):
                tool_name = _static_text(node.args[0])
                if tool_name in {"call_kakao", "call_instagram"}:
                    gui_call_factory_tools.add(tool_name)
            if (
                relative == CALL_CONNECTOR_PATH
                and qualnames[node] == "make_call_tools.call_telegram"
            ):
                if call_name == "_call_attempt_failure_metadata":
                    telegram_call_declarations += 1
                if call_name == "_call_attempt_recovery_guidance":
                    telegram_call_guidance_calls += 1
            if call_name != "ToolResult":
                continue
            ok_kind = _tool_result_ok_kind(node)
            if ok_kind == "literal_failure":
                site = FailureSite(
                    relative,
                    qualnames[node],
                    node.lineno,
                    "false",
                    _fingerprint(node),
                    _tool_result_context(node, parents),
                    _normalized_output_surface_fingerprint(node),
                    _direct_guidance_kind(node),
                )
                literal_failures.append(site)
                if (
                    relative == CENTRAL_EXECUTOR_PATH
                    and site.qualname == CENTRAL_EXECUTOR_QUALNAME
                ):
                    problem = _central_executor_problem(node, site)
                    if problem:
                        central_problems.append(problem)
                if (
                    relative == CALL_CONNECTOR_PATH
                    and site.qualname == "make_call_tools.call_contact"
                    and isinstance(_tool_result_metadata_node(node), ast.Call)
                    and _call_name(_tool_result_metadata_node(node)) == "_call_attempt_failure_metadata"
                ):
                    native_call_failure_sites += 1
                    problem = _post_attempt_call_problem(
                        node,
                        path=relative,
                        label="native call post-attempt failure",
                    )
                    if problem:
                        post_attempt_call_problems.append(problem)
                if (
                    relative == CALL_CONNECTOR_PATH
                    and site.qualname == "make_call_tools._gui_call_tool.handler"
                    and isinstance(_tool_result_metadata_node(node), ast.Call)
                    and _call_name(_tool_result_metadata_node(node)) == "_call_attempt_failure_metadata"
                ):
                    gui_call_failure_sites += 1
                    problem = _post_attempt_call_problem(
                        node,
                        path=relative,
                        label="GUI call post-attempt failure",
                    )
                    if problem:
                        post_attempt_call_problems.append(problem)
            elif ok_kind == "dynamic":
                dynamic_results.append(
                    FailureSite(
                        relative,
                        qualnames[node],
                        node.lineno,
                        "dynamic",
                        _fingerprint(node),
                        _tool_result_context(node, parents),
                        _normalized_output_surface_fingerprint(node),
                        _direct_guidance_kind(node),
                    )
                )
            else:
                literal_success_count += 1
    central_count = sum(
        site.path == CENTRAL_EXECUTOR_PATH
        and site.qualname == CENTRAL_EXECUTOR_QUALNAME
        for site in literal_failures
    )
    if central_count != EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES:
        central_problems.append(
            "central executor failure-site count changed "
            f"({central_count}/{EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES}); review the enforced scope"
        )
    actual_contact_statuses = set(contact_resolution_guidance_statuses)
    if (
        actual_contact_statuses != EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES
        or len(contact_resolution_guidance_statuses)
        != len(EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES)
    ):
        contact_resolution_problems.append(
            "contact-resolution guidance scope changed "
            f"({sorted(contact_resolution_guidance_statuses)!r}/"
            f"{sorted(EXPECTED_CONTACT_RESOLUTION_GUIDANCE_STATUSES)!r}); "
            "review the enforced scope"
        )
    if native_call_failure_sites != 1:
        post_attempt_call_problems.append(
            "native call post-attempt failure scope changed "
            f"({native_call_failure_sites}/1); review the enforced scope"
        )
    else:
        post_attempt_call_failure_tools.append("call_contact")
    if gui_call_failure_sites != 1:
        post_attempt_call_problems.append(
            "GUI call post-attempt failure scope changed "
            f"({gui_call_failure_sites}/1); review the enforced scope"
        )
    else:
        post_attempt_call_failure_tools.extend(sorted(gui_call_factory_tools))
    if telegram_call_declarations != 1 or telegram_call_guidance_calls != 1:
        post_attempt_call_problems.append(
            "Telegram call post-attempt failure scope changed "
            f"(declarations={telegram_call_declarations}, "
            f"guidance={telegram_call_guidance_calls}); review the enforced scope"
        )
    else:
        post_attempt_call_failure_tools.append("call_telegram")
    actual_post_attempt_tools = set(post_attempt_call_failure_tools)
    if (
        actual_post_attempt_tools != EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS
        or len(post_attempt_call_failure_tools)
        != len(EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS)
    ):
        post_attempt_call_problems.append(
            "post-attempt call guidance scope changed "
            f"({sorted(post_attempt_call_failure_tools)!r}/"
            f"{sorted(EXPECTED_POST_ATTEMPT_CALL_FAILURE_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name in sorted(EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS):
        unknown_declaration_count = post_attempt_send_declarations[tool_name]
        known_not_sent_declaration_count = known_not_sent_send_declarations[
            tool_name
        ]
        guidance_count = post_attempt_send_guidance_calls[tool_name]
        if (
            unknown_declaration_count != 1
            or known_not_sent_declaration_count != 1
            or guidance_count != 1
        ):
            post_attempt_send_problems.append(
                f"{tool_name} post-attempt send scope changed "
                f"(outcome_unknown={unknown_declaration_count}, "
                f"known_not_sent={known_not_sent_declaration_count}, "
                f"guidance={guidance_count}); "
                "review the enforced scope"
            )
        else:
            post_attempt_send_failure_tools.append(tool_name)
    if set(post_attempt_send_failure_tools) != EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS:
        post_attempt_send_problems.append(
            "post-attempt send guidance scope changed "
            f"({sorted(post_attempt_send_failure_tools)!r}/"
            f"{sorted(EXPECTED_POST_ATTEMPT_SEND_FAILURE_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name, expected_count in sorted(PERSONAL_READ_TARGETS.values()):
        declaration_count = personal_read_declarations[tool_name]
        if declaration_count != expected_count:
            personal_read_problems.append(
                f"{tool_name} personal-read guidance scope changed "
                f"({declaration_count}/{expected_count}); review the enforced scope"
            )
        else:
            personal_read_failure_tools.append(tool_name)
    if set(personal_read_failure_tools) != EXPECTED_PERSONAL_READ_FAILURE_TOOLS:
        personal_read_problems.append(
            "personal-read guidance tool scope changed "
            f"({sorted(personal_read_failure_tools)!r}/"
            f"{sorted(EXPECTED_PERSONAL_READ_FAILURE_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name, expected_count in sorted(
        LOCAL_PRODUCTIVITY_READ_TARGETS.values()
    ):
        declaration_count = local_productivity_read_declarations[tool_name]
        if declaration_count != expected_count:
            local_productivity_read_problems.append(
                f"{tool_name} local-productivity read guidance scope changed "
                f"({declaration_count}/{expected_count}); review the enforced scope"
            )
        else:
            local_productivity_read_tools.append(tool_name)
    if (
        set(local_productivity_read_tools)
        != EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS
    ):
        local_productivity_read_problems.append(
            "local-productivity read guidance tool scope changed "
            f"({sorted(local_productivity_read_tools)!r}/"
            f"{sorted(EXPECTED_LOCAL_PRODUCTIVITY_READ_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name, expected_count in sorted(LOCAL_READ_INPUT_TARGETS.values()):
        declaration_count = local_read_input_declarations[tool_name]
        if declaration_count != expected_count:
            local_read_input_problems.append(
                f"{tool_name} local-read input guidance scope changed "
                f"({declaration_count}/{expected_count}); review the enforced scope"
            )
        else:
            local_read_input_tools.append(tool_name)
    if set(local_read_input_tools) != EXPECTED_LOCAL_READ_INPUT_TOOLS:
        local_read_input_problems.append(
            "local-read input guidance tool scope changed "
            f"({sorted(local_read_input_tools)!r}/"
            f"{sorted(EXPECTED_LOCAL_READ_INPUT_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name, expected_count in sorted(OFFLINE_UTILITY_INPUT_TARGETS.values()):
        declaration_count = offline_utility_input_declarations[tool_name]
        if declaration_count != expected_count:
            offline_utility_input_problems.append(
                f"{tool_name} offline-utility input guidance scope changed "
                f"({declaration_count}/{expected_count}); review the enforced scope"
            )
        else:
            offline_utility_input_tools.append(tool_name)
    if (
        set(offline_utility_input_tools)
        != EXPECTED_OFFLINE_UTILITY_INPUT_TOOLS
    ):
        offline_utility_input_problems.append(
            "offline-utility input guidance tool scope changed "
            f"({sorted(offline_utility_input_tools)!r}/"
            f"{sorted(EXPECTED_OFFLINE_UTILITY_INPUT_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name, expected_count in sorted(
        EXTERNAL_INFORMATION_TARGETS.values()
    ):
        declaration_count = external_information_declarations[tool_name]
        if declaration_count != expected_count:
            external_information_problems.append(
                f"{tool_name} external-information guidance scope changed "
                f"({declaration_count}/{expected_count}); review the enforced scope"
            )
        else:
            external_information_tools.append(tool_name)
    if set(external_information_tools) != EXPECTED_EXTERNAL_INFORMATION_TOOLS:
        external_information_problems.append(
            "external-information guidance tool scope changed "
            f"({sorted(external_information_tools)!r}/"
            f"{sorted(EXPECTED_EXTERNAL_INFORMATION_TOOLS)!r}); "
            "review the enforced scope"
        )
    for resource_path in sorted(EXPECTED_RESOURCE_NOT_FOUND_PATHS):
        declaration_count = resource_not_found_declarations[resource_path]
        if declaration_count != 1:
            resource_not_found_problems.append(
                f"{resource_path} resource-not-found guidance scope changed "
                f"({declaration_count}/1); review the enforced scope"
            )
        else:
            resource_not_found_paths.append(resource_path)
    if set(resource_not_found_paths) != EXPECTED_RESOURCE_NOT_FOUND_PATHS:
        resource_not_found_problems.append(
            "resource-not-found guidance path scope changed "
            f"({sorted(resource_not_found_paths)!r}/"
            f"{sorted(EXPECTED_RESOURCE_NOT_FOUND_PATHS)!r}); "
            "review the enforced scope"
        )
    for tool_name in sorted(EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS):
        declaration_count = post_attempt_writer_declarations[tool_name]
        guidance_count = post_attempt_writer_guidance_calls[tool_name]
        if declaration_count != 1 or guidance_count != 1:
            post_attempt_writer_problems.append(
                f"{tool_name} post-attempt writer scope changed "
                f"(declarations={declaration_count}, guidance={guidance_count}); "
                "review the enforced scope"
            )
        else:
            post_attempt_writer_failure_tools.append(tool_name)
    if set(post_attempt_writer_failure_tools) != EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS:
        post_attempt_writer_problems.append(
            "post-attempt writer guidance scope changed "
            f"({sorted(post_attempt_writer_failure_tools)!r}/"
            f"{sorted(EXPECTED_POST_ATTEMPT_WRITER_FAILURE_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name in sorted(EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS):
        declaration_count = post_attempt_calendar_declarations[tool_name]
        guidance_count = post_attempt_calendar_guidance_calls[tool_name]
        if declaration_count != 1 or guidance_count != 1:
            post_attempt_calendar_mutation_problems.append(
                f"{tool_name} post-attempt calendar mutation scope changed "
                f"(declarations={declaration_count}, guidance={guidance_count}); "
                "review the enforced scope"
            )
        else:
            post_attempt_calendar_mutation_failure_tools.append(tool_name)
    if (
        set(post_attempt_calendar_mutation_failure_tools)
        != EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS
    ):
        post_attempt_calendar_mutation_problems.append(
            "post-attempt calendar mutation guidance scope changed "
            f"({sorted(post_attempt_calendar_mutation_failure_tools)!r}/"
            f"{sorted(EXPECTED_POST_ATTEMPT_CALENDAR_MUTATION_FAILURE_TOOLS)!r}); "
            "review the enforced scope"
        )
    for tool_name in sorted(EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS):
        declaration_count = post_attempt_reminder_declarations[tool_name]
        if declaration_count != 1:
            post_attempt_reminder_mutation_problems.append(
                f"{tool_name} post-attempt reminder mutation scope changed "
                f"(declarations={declaration_count}); review the enforced scope"
            )
        else:
            post_attempt_reminder_mutation_failure_tools.append(tool_name)
    if (
        set(post_attempt_reminder_mutation_failure_tools)
        != EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS
    ):
        post_attempt_reminder_mutation_problems.append(
            "post-attempt reminder mutation guidance scope changed "
            f"({sorted(post_attempt_reminder_mutation_failure_tools)!r}/"
            f"{sorted(EXPECTED_POST_ATTEMPT_REMINDER_MUTATION_FAILURE_TOOLS)!r}); "
            "review the enforced scope"
        )
    if (
        set(reconciliation_guidance_failure_kinds)
        != EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS
        or len(reconciliation_guidance_failure_kinds)
        != len(EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS)
    ):
        reconciliation_guidance_problems.append(
            "auto-mutation reconciliation guidance scope changed "
            f"({sorted(reconciliation_guidance_failure_kinds)!r}/"
            f"{sorted(EXPECTED_RECONCILIATION_GUIDANCE_FAILURE_KINDS)!r}); "
            "review the enforced scope"
        )
    return FailureInventory(
        tuple(literal_failures),
        tuple(dynamic_results),
        tuple(helper_declarations),
        tuple(central_problems),
        tuple(sorted(contact_resolution_guidance_statuses)),
        tuple(contact_resolution_problems),
        tuple(sorted(post_attempt_call_failure_tools)),
        tuple(post_attempt_call_problems),
        tuple(sorted(post_attempt_send_failure_tools)),
        tuple(post_attempt_send_problems),
        tuple(sorted(personal_read_failure_tools)),
        sum(personal_read_declarations.values()),
        tuple(personal_read_problems),
        tuple(sorted(local_productivity_read_tools)),
        sum(local_productivity_read_declarations.values()),
        tuple(local_productivity_read_problems),
        tuple(sorted(local_read_input_tools)),
        sum(local_read_input_declarations.values()),
        tuple(local_read_input_problems),
        tuple(sorted(offline_utility_input_tools)),
        sum(offline_utility_input_declarations.values()),
        tuple(offline_utility_input_problems),
        tuple(sorted(external_information_tools)),
        sum(external_information_declarations.values()),
        tuple(external_information_problems),
        tuple(sorted(resource_not_found_paths)),
        sum(resource_not_found_declarations.values()),
        tuple(resource_not_found_problems),
        tuple(sorted(post_attempt_writer_failure_tools)),
        tuple(post_attempt_writer_problems),
        tuple(sorted(post_attempt_calendar_mutation_failure_tools)),
        tuple(post_attempt_calendar_mutation_problems),
        tuple(sorted(post_attempt_reminder_mutation_failure_tools)),
        tuple(post_attempt_reminder_mutation_problems),
        tuple(sorted(reconciliation_guidance_failure_kinds)),
        tuple(reconciliation_guidance_problems),
        literal_success_count,
    )


def _format_context_counts(counts: tuple[tuple[str, int], ...]) -> str:
    return ", ".join(f"{context}={count}" for context, count in counts) or "none"


def format_failure_inventory(inventory: FailureInventory) -> str:
    lines = [
        "Error-guidance source inventory (read-only)",
        (
            f"- enforced central executor failures: "
            f"{EXPECTED_CENTRAL_EXECUTOR_FAILURE_SITES}"
        ),
        (
            "- enforced pre-approval contact-resolution failures: "
            f"{len(inventory.contact_resolution_guidance_statuses)}"
        ),
        (
            "- enforced post-attempt call failures: "
            f"{len(inventory.post_attempt_call_failure_tools)}"
        ),
        (
            "- enforced post-attempt send failures: "
            f"{len(inventory.post_attempt_send_failure_tools)}"
        ),
        (
            "- enforced retryable personal-read failures: "
            f"{inventory.personal_read_failure_site_count} across "
            f"{len(inventory.personal_read_failure_tools)} tools"
        ),
        (
            "- enforced local-productivity read failures: "
            f"{inventory.local_productivity_read_site_count} across "
            f"{len(inventory.local_productivity_read_tools)} paths"
        ),
        (
            "- enforced local read-input failures: "
            f"{inventory.local_read_input_site_count} across "
            f"{len(inventory.local_read_input_tools)} tools"
        ),
        (
            "- enforced offline-utility input failures: "
            f"{inventory.offline_utility_input_site_count} across "
            f"{len(inventory.offline_utility_input_tools)} tools"
        ),
        (
            "- enforced external-information fetch failures: "
            f"{inventory.external_information_site_count} across "
            f"{len(inventory.external_information_tools)} tools"
        ),
        (
            "- enforced resource-not-found failures: "
            f"{inventory.resource_not_found_site_count} across "
            f"{len(inventory.resource_not_found_paths)} paths"
        ),
        (
            "- enforced post-attempt writer failures: "
            f"{len(inventory.post_attempt_writer_failure_tools)}"
        ),
        (
            "- enforced post-attempt calendar mutation failures: "
            f"{len(inventory.post_attempt_calendar_mutation_failure_tools)}"
        ),
        (
            "- enforced post-attempt reminder mutation failures: "
            f"{len(inventory.post_attempt_reminder_mutation_failure_tools)}"
        ),
        (
            "- enforced auto-mutation reconciliation failures: "
            f"{len(inventory.reconciliation_guidance_failure_kinds)}"
        ),
        f"- helper declarations: {len(inventory.helper_declarations)}",
        (
            "- raw legacy literal failure candidates (includes enforced/reviewed sites; "
            "not residual unguided debt): "
            f"{len(inventory.legacy_failures)}"
        ),
        (
            "- reviewed lower bound with direct canonical guidance declarations: "
            f"{len(inventory.directly_declared_legacy_failures)}"
        ),
        (
            "- normalized output surfaces among raw legacy literal candidates: "
            f"{inventory.legacy_surface_count}"
        ),
        (
            "- raw legacy literal candidate contexts: "
            f"{_format_context_counts(inventory.legacy_context_counts())}"
        ),
        (
            "- raw dynamic-result candidates (mixed outcomes; not residual unguided debt): "
            f"{len(inventory.dynamic_results)}"
        ),
        (
            "- raw dynamic candidate contexts: "
            f"{_format_context_counts(inventory.dynamic_context_counts())}"
        ),
        f"- literal success constructors (excluded from failure candidates): {inventory.literal_success_count}",
        (
            "- acceptance scope: still open; this source inventory does not prove every "
            "user-visible error surface"
        ),
        "- raw legacy literal candidates by file:",
    ]
    lines.extend(
        f"  - {path}: {count}"
        for path, count in inventory.legacy_counts_by_file()
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    inventory = build_failure_inventory(project_root)
    print(format_failure_inventory(inventory))
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
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()
