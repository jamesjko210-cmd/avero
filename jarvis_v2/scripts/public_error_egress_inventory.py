"""Conservative inventory of non-ToolResult public error-egress candidates.

This is intentionally a first-phase inventory, not an exhaustive proof that every
user-visible error has recovery guidance.  It covers five explicit static/string shapes:

* status-dashboard ``_send_json({... "error": ...})`` calls and its canonical
  JSON-body error-payload helper shape;
* terminal calls whose static text looks failure-related, plus startup failures;
* Telegram bridge send/callback calls whose dynamic reply may carry an error; and
* HUD ``HudReply(..., error=...)`` construction; and
* reviewed embedded-dashboard client error sinks plus known raw-reflection shapes.

The existing :mod:`error_guidance_inventory` owns ``ToolResult`` failures.  Keeping
this inventory separate prevents the two counts from being mistaken for one broad
coverage claim.
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


STATUS_SERVER_PATH = "jarvis_v2/ui/status_server.py"
TELEGRAM_ADAPTER_PATH = "jarvis_v2/automations/telegram_control.py"
HUD_PATH = "jarvis_v2/scripts/hud.py"
CLI_PATHS = (
    "jarvis_v2/scripts/ask.py",
    "jarvis_v2/scripts/chat.py",
    "jarvis_v2/scripts/run_status_server.py",
    "jarvis_v2/scripts/run_telegram_control.py",
    "jarvis_v2/scripts/telegram_whoami.py",
)

FAILURE_TEXT = re.compile(
    r"\b(?:error|fail(?:ed|ure)?|could not|did not|refus(?:e|ed)|unavailable|"
    r"disabled|required|not set|not configured|not found|rejected|retry|diagnostic)\b",
    re.IGNORECASE,
)

DASHBOARD_CLIENT_ERROR_SINK_MARKERS = (
    (
        "appendSystemLog('[VOICE] native recorder unavailable. Run `voice setup check`",
        "voice-native-start",
    ),
    (
        "preflightOutput.textContent = 'Risk preview could not be loaded.",
        "risk-preflight",
    ),
    (
        "actionPacketOutput.textContent = 'Action packet could not be loaded.",
        "action-packet",
    ),
    (
        "brainMemoryDetailOutput.textContent = 'Memory detail could not be loaded.",
        "memory-detail",
    ),
    (
        "brainMemoryDetailOutput.textContent = 'Related memories could not be loaded.",
        "memory-neighbors",
    ),
    (
        "brainResultOutput.textContent = 'Brain result could not be loaded.",
        "brain-workbench",
    ),
    (
        "setSpeechState('confirm', 'Voice confirmation was not confirmed.",
        "voice-confirmation",
    ),
)
DASHBOARD_CLIENT_RAW_ERROR_MARKERS = (
    ("data.output || data.error || JSON.stringify(data, null, 2)", "raw-response-fallback"),
    ("String(error.message || error)", "raw-error-message"),
    ("String(error)", "raw-error-string"),
    ("throw new Error(data.error ||", "raw-server-error-propagation"),
)


@dataclass(frozen=True, order=True)
class PublicErrorEgressCandidate:
    path: str
    line: int
    qualname: str
    category: str
    shape: str


@dataclass(frozen=True)
class PublicErrorEgressInventory:
    candidates: tuple[PublicErrorEgressCandidate, ...]
    counts_by_category: tuple[tuple[str, int], ...]
    counts_by_path: tuple[tuple[str, int], ...]
    fingerprint: str


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.candidates: list[PublicErrorEgressCandidate] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    @property
    def qualname(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _add(self, node: ast.AST, category: str, shape: str) -> None:
        self.candidates.append(
            PublicErrorEgressCandidate(
                path=self.path,
                line=int(getattr(node, "lineno", 0)),
                qualname=self.qualname,
                category=category,
                shape=shape,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _call_name(node.func)

        if self.path == STATUS_SERVER_PATH and call_name.endswith("._send_json"):
            if node.args and _dict_has_string_key(node.args[0], "error"):
                self._add(node, "dashboard_json_error", "_send_json:error-key")
            elif (
                node.args
                and isinstance(node.args[0], ast.Call)
                and _call_name(node.args[0].func) == "_dashboard_json_body_error_payload"
            ):
                self._add(
                    node,
                    "dashboard_json_error",
                    "_send_json:dashboard-json-body-error",
                )

        if self.path in CLI_PATHS:
            static_text = " ".join(_static_string_fragments(node))
            terminal_error = call_name in {"print", "parser.error"} and bool(
                FAILURE_TEXT.search(static_text)
            )
            if terminal_error or call_name == "print_startup_failure":
                shape = "startup-failure" if call_name == "print_startup_failure" else call_name
                self._add(node, "terminal_error_output", shape)

        if self.path == TELEGRAM_ADAPTER_PATH and call_name in {
            "self.send_func",
            "self.answer_callback_func",
        }:
            self._add(node, "telegram_dynamic_reply", call_name)

        if self.path == HUD_PATH and call_name == "HudReply":
            error_keyword = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "error"),
                None,
            )
            if error_keyword is not None and not _is_empty_string(error_keyword):
                self._add(node, "hud_error_reply", "HudReply:error-keyword")

        self.generic_visit(node)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _dict_has_string_key(node: ast.AST, wanted: str) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    return any(
        isinstance(key, ast.Constant) and key.value == wanted
        for key in node.keys
        if key is not None
    )


def _static_string_fragments(node: ast.AST) -> list[str]:
    fragments: list[str] = []
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            fragments.append(item.value)
    return fragments


def _is_empty_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == ""


def _dashboard_client_error_candidates(
    source: str,
) -> list[PublicErrorEgressCandidate]:
    candidates: list[PublicErrorEgressCandidate] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for marker, shape in DASHBOARD_CLIENT_ERROR_SINK_MARKERS:
            if marker in line:
                candidates.append(
                    PublicErrorEgressCandidate(
                        path=STATUS_SERVER_PATH,
                        line=line_number,
                        qualname="DASHBOARD_HTML.script",
                        category="dashboard_client_error",
                        shape=shape,
                    )
                )
        for marker, shape in DASHBOARD_CLIENT_RAW_ERROR_MARKERS:
            if marker in line:
                candidates.append(
                    PublicErrorEgressCandidate(
                        path=STATUS_SERVER_PATH,
                        line=line_number,
                        qualname="DASHBOARD_HTML.script",
                        category="dashboard_client_raw_error",
                        shape=shape,
                    )
                )
    return candidates


def build_public_error_egress_inventory(repo_root: Path) -> PublicErrorEgressInventory:
    paths = (STATUS_SERVER_PATH, TELEGRAM_ADAPTER_PATH, HUD_PATH, *CLI_PATHS)
    candidates: list[PublicErrorEgressCandidate] = []
    for relative_path in paths:
        source_path = repo_root / relative_path
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        visitor = _Visitor(relative_path)
        visitor.visit(tree)
        candidates.extend(visitor.candidates)
        if relative_path == STATUS_SERVER_PATH:
            candidates.extend(_dashboard_client_error_candidates(source))

    ordered = tuple(sorted(candidates))
    category_counts = Counter(item.category for item in ordered)
    path_counts = Counter(item.path for item in ordered)
    fingerprint_input = "\n".join(
        f"{item.path}:{item.line}:{item.qualname}:{item.category}:{item.shape}"
        for item in ordered
    )
    return PublicErrorEgressInventory(
        candidates=ordered,
        counts_by_category=tuple(sorted(category_counts.items())),
        counts_by_path=tuple(sorted(path_counts.items())),
        fingerprint=hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest(),
    )


def format_public_error_egress_inventory(inventory: PublicErrorEgressInventory) -> str:
    category_text = ", ".join(
        f"{category}={count}" for category, count in inventory.counts_by_category
    )
    return (
        f"non-ToolResult public error-egress candidates: {len(inventory.candidates)} "
        f"({category_text}); fingerprint={inventory.fingerprint}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(format_public_error_egress_inventory(build_public_error_egress_inventory(root)))
