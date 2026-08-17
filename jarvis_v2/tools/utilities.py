from __future__ import annotations

import ast
import math
import re
import secrets
import string
import uuid
from typing import Any

from jarvis_v2.agent.failure_guidance import (
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    declare_retryable_local_read_failure,
)
from jarvis_v2.agent.types import ToolResult


MAX_EXPRESSION_CHARS = 500
MAX_EXPRESSION_NODES = 100
MAX_EXPRESSION_SEQUENCE_ITEMS = 20
MAX_EXPRESSION_POWER = 1000
MAX_EXPRESSION_INTEGER_BITS = 12000
MAX_PASSWORD_LENGTH = 128
MAX_UNIT_CHARS = 40
MAX_SPELL_TEXT_CHARS = 80
MAX_COUNT_TEXT_CHARS = 500
MAX_TRANSFORM_TEXT_CHARS = 500
MAX_REPEAT_TEXT_COUNT = 20
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*")
SECRET_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk_(?:live|test)_[A-Za-z0-9_-]+"
    r"|\bgh[pousr]_[A-Za-z0-9_-]+"
    r"|\bxox[baprs]-[A-Za-z0-9-]+"
    r"|\bAIza[A-Za-z0-9_-]{16,}"
    r"|\b\d{6,}:[A-Za-z0-9_-]{20,}"
    r")",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*", re.UNICODE)


def _short(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_raw(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value).strip()
    text = " ".join(text.split())
    text = LOCAL_PATH_RE.sub("<local-path>", text)
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _short_failure_raw(value: Any, *, limit: int) -> str:
    """Bound and redact private-looking values retained in failure metadata."""
    return SECRET_VALUE_RE.sub("<private-value>", _short_raw(value, limit=limit))


def _has_local_path(value: Any) -> bool:
    return bool(LOCAL_PATH_RE.search(str(value or "")))


def _words(value: str) -> list[str]:
    """Return bounded utility words without restricting text to ASCII."""
    return WORD_RE.findall(value)


def _normalize_calculation_expression(expression: str) -> str:
    normalized = str(expression or "").strip()
    normalized = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)\b",
        r"(\1/100) * \2",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*(?:percent|per cent)\s+of\s+(\d+(?:\.\d+)?)\b",
        r"(\1/100) * \2",
        normalized,
        flags=re.IGNORECASE,
    )
    replacements = [
        (r"\bdivided\s+by\b", "/"),
        (r"\bover\b", "/"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\btimes\b", "*"),
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\bof\b", "*"),
        (r"\bpercent\b", "/100"),
        (r"\bper\s+cent\b", "/100"),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _format_calculation_result(result: Any) -> str:
    if isinstance(result, float) and math.isfinite(result):
        rounded = round(result, 12)
        if rounded.is_integer():
            return str(int(rounded))
        return f"{rounded:.12g}"
    return str(result)


def _bounded_numeric_result(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("boolean results are not supported")
    if isinstance(value, int):
        if value.bit_length() > MAX_EXPRESSION_INTEGER_BITS:
            raise OverflowError("integer result is too large")
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_EXPRESSION_SEQUENCE_ITEMS:
            raise ValueError("numeric sequence is too large")
        bounded = [_bounded_numeric_result(item) for item in value]
        return bounded if isinstance(value, list) else tuple(bounded)
    raise ValueError("only numeric results are supported")


def _bounded_math_call(name: str, func: Any, args: list[Any]) -> Any:
    if name in {"factorial", "comb", "perm"}:
        integer_args = [arg for arg in args if isinstance(arg, int) and not isinstance(arg, bool)]
        if len(integer_args) != len(args) or any(arg < 0 or arg > MAX_EXPRESSION_POWER for arg in integer_args):
            raise ValueError(f"{name} arguments must be integers from 0 to {MAX_EXPRESSION_POWER}")
    return _bounded_numeric_result(func(*args))


def _evaluate_numeric_expression(expression: str, allowed: dict[str, Any]) -> Any:
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise ValueError("expression is too complex")

    binary_operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
    }

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numeric literals are supported")
            return _bounded_numeric_result(node.value)
        if isinstance(node, ast.Name):
            value = allowed.get(node.id)
            if value is None or callable(value):
                raise NameError(node.id)
            return _bounded_numeric_result(value)
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > MAX_EXPRESSION_SEQUENCE_ITEMS:
                raise ValueError("numeric sequence is too large")
            values = [evaluate(item) for item in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            if isinstance(value, (list, tuple)):
                raise ValueError("sequence arithmetic is not supported")
            result = value if isinstance(node.op, ast.UAdd) else -value
            return _bounded_numeric_result(result)
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
                raise ValueError("sequence arithmetic is not supported")
            if isinstance(node.op, ast.Pow):
                if not isinstance(right, (int, float)) or abs(right) > MAX_EXPRESSION_POWER:
                    raise OverflowError("power is too large")
                return _bounded_numeric_result(left ** right)
            operator = binary_operators.get(type(node.op))
            if operator is None:
                raise ValueError("operator is not supported")
            return _bounded_numeric_result(operator(left, right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("only named numeric functions are supported")
            func = allowed.get(node.func.id)
            if not callable(func):
                raise NameError(node.func.id)
            if len(node.args) > MAX_EXPRESSION_SEQUENCE_ITEMS:
                raise ValueError("too many function arguments")
            return _bounded_math_call(node.func.id, func, [evaluate(arg) for arg in node.args])
        raise ValueError("expression form is not supported")

    return evaluate(tree)


def _safe_metadata(**extra: Any) -> dict[str, Any]:
    metadata = {
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }
    metadata.update(extra)
    return metadata


def _utility_boundaries() -> dict[str, bool]:
    return {
        "read_only": True,
        "calls_model": False,
        "executes_tools": False,
        "queues_approval": False,
        "approves_request": False,
        "dismisses_request": False,
        "reads_private_data": False,
        "reads_personal_data": False,
        "executes_side_effect": False,
        "writes_files": False,
        "writes_memory": False,
        "writes_notes": False,
        "controls_computer": False,
        "external_side_effect": False,
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
    }


def _utility_handoff(*, source: str, status: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "source": source,
        "status": status,
        "ready_for_operator": True,
        "changed": [],
        "authorizes_execution": False,
        "authorizes_completion_claim": False,
        "approval_granted": False,
        "next_commands": [
            "calculate <expression>",
            "convert <value> <unit> to <unit>",
            "generate password <length>",
        ],
        "boundaries": _utility_boundaries(),
    }
    if reason:
        handoff["reason"] = reason
        handoff["refused"] = True
    else:
        handoff["refused"] = False
    handoff.update(fields)
    return handoff


def _utility_metadata(*, source: str, status: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    handoff_fields: dict[str, Any] = {}
    for key in (
        "expression_chars",
        "max_chars",
        "local_path_expression",
        "rejected_private_name",
        "evaluation_error",
        "exception_type",
        "length",
        "include_symbols",
        "uuid_version",
        "raw_length",
        "value",
        "raw_value",
        "from_unit",
        "to_unit",
        "conversion_result_available",
        "spell_text",
        "spell_text_chars",
        "letter_count",
        "count_text",
        "count_text_chars",
        "word_count",
        "char_count",
        "transform_mode",
        "transform_text",
        "transform_text_chars",
        "transform_word_count",
        "repeat_count",
        "weight_kg",
        "height_m",
        "bmi",
        "bmi_category",
    ):
        if key in extra:
            handoff_fields[key] = extra[key]
    handoff = _utility_handoff(source=source, status=status, reason=reason, **handoff_fields)
    return _safe_metadata(
        utility_handoff=handoff,
        utility_handoff_ready=True,
        utility_status=status,
        refusal_reason=reason,
        **extra,
    )


def calculate(args: dict[str, Any]) -> ToolResult:
    raw_expression = str(args.get("expression") or "").strip()
    expression = _normalize_calculation_expression(raw_expression)
    if not expression:
        failure_output = (
            f"Expression is empty. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(
                    source="calculate",
                    status="refused",
                    reason="missing_expression",
                    expression_chars=0,
                ),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if len(raw_expression) > MAX_EXPRESSION_CHARS or len(expression) > MAX_EXPRESSION_CHARS:
        failure_output = (
            f"Expression is too large ({len(raw_expression)} chars). "
            f"Limit is {MAX_EXPRESSION_CHARS}. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate", status="refused", reason="expression_too_large", expression_chars=len(raw_expression), max_chars=MAX_EXPRESSION_CHARS),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if _has_local_path(raw_expression) or _has_local_path(expression):
        failure_output = (
            "Expression cannot contain local file paths. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate", status="refused", reason="local_path_expression", expression_chars=len(raw_expression), local_path_expression=True, raw_expression=_short_raw(raw_expression, limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if "__" in expression:
        failure_output = (
            "Expression cannot contain dunder/private names. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate", status="refused", reason="rejected_private_name", expression_chars=len(expression), rejected_private_name=True),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    allowed = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
    try:
        result = _evaluate_numeric_expression(expression, allowed)
        formatted_result = _format_calculation_result(result)
    except Exception as exc:
        failure_output = (
            "Could not calculate expression. Check syntax and supported math functions. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate", status="error", reason="evaluation_error", expression_chars=len(expression), evaluation_error=True, exception_type=type(exc).__name__),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    return ToolResult(
        "calculate",
        True,
        f"{expression} = {formatted_result}",
        _utility_metadata(source="calculate", status="ok", expression_chars=len(expression)),
    )


def generate_password(args: dict[str, Any]) -> ToolResult:
    if isinstance(args.get("length"), bool):
        failure_output = (
            f"Password length must be a number. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "generate_password",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="generate_password", status="refused", reason="bad_length", length=None, raw_length=str(args.get("length"))),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    try:
        length = max(8, min(MAX_PASSWORD_LENGTH, int(args.get("length", 20))))
    except (TypeError, ValueError):
        failure_output = (
            f"Password length must be a number. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "generate_password",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="generate_password", status="refused", reason="bad_length", length=None, raw_length=_short_failure_raw(args.get("length"), limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    include_symbols = bool(args.get("include_symbols", True))
    alphabet = string.ascii_letters + string.digits
    if include_symbols:
        alphabet += "!@#$%^&*()-_=+[]{};:,.?"
    password = "".join(secrets.choice(alphabet) for _ in range(length))
    return ToolResult(
        "generate_password",
        True,
        password,
        _utility_metadata(source="generate_password", status="ok", length=length, include_symbols=include_symbols, password_in_handoff=False),
    )


def generate_uuid(args: dict[str, Any]) -> ToolResult:
    value = str(uuid.uuid4())
    return ToolResult(
        "generate_uuid",
        True,
        value,
        _utility_metadata(source="generate_uuid", status="ok", uuid_version=4, uuid_in_handoff=False),
    )


def spell_word(args: dict[str, Any]) -> ToolResult:
    raw_text = str(args.get("word") or args.get("text") or "").strip()
    text = _short_raw(raw_text.strip("?.!\"'"), limit=MAX_SPELL_TEXT_CHARS)
    if not text:
        failure_output = (
            f"Which word should I spell? {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "spell_word",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="spell_word", status="refused", reason="missing_text", spell_text_chars=0),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if _has_local_path(raw_text):
        failure_output = (
            "Spelling text cannot contain local file paths. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "spell_word",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="spell_word", status="refused", reason="local_path_text", spell_text="<local-path>", spell_text_chars=len(text)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if not any(ch.isalnum() for ch in text):
        failure_output = (
            "Please give me a word or short phrase to spell. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "spell_word",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="spell_word", status="refused", reason="invalid_text", spell_text_chars=len(text)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    letters = " ".join(ch for ch in text)
    return ToolResult(
        "spell_word",
        True,
        f"{text}: {letters}",
        _utility_metadata(source="spell_word", status="ok", spell_text=text, spell_text_chars=len(text), letter_count=len(text)),
    )


def count_text(args: dict[str, Any]) -> ToolResult:
    raw_text = str(args.get("text") or args.get("phrase") or "").strip()
    text = _short_raw(raw_text, limit=MAX_COUNT_TEXT_CHARS)
    if not text:
        failure_output = (
            f"What text should I count? {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "count_text",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="count_text", status="refused", reason="missing_text", count_text_chars=0),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if _has_local_path(raw_text):
        failure_output = (
            "Text counting cannot contain local file paths. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "count_text",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="count_text", status="refused", reason="local_path_text", count_text="<local-path>", count_text_chars=len(text)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    words = _words(text)
    word_count = len(words)
    char_count = len(text)
    word_label = "word" if word_count == 1 else "words"
    char_label = "character" if char_count == 1 else "characters"
    return ToolResult(
        "count_text",
        True,
        f"{word_count} {word_label}, {char_count} {char_label}",
        _utility_metadata(source="count_text", status="ok", count_text=text, count_text_chars=char_count, word_count=word_count, char_count=char_count),
    )


def transform_text(args: dict[str, Any]) -> ToolResult:
    mode = str(args.get("mode") or "").strip().lower().replace("_", " ")
    raw_text = str(args.get("text") or args.get("phrase") or "").strip()
    text = _short_raw(raw_text, limit=MAX_TRANSFORM_TEXT_CHARS)
    repeat_count = args.get("count")
    count: int | None = None
    if not text:
        failure_output = (
            f"What text should I transform? {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "transform_text",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="transform_text", status="refused", reason="missing_text", transform_text_chars=0, transform_mode=_short_failure_raw(mode or "unknown", limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if _has_local_path(raw_text):
        failure_output = (
            "Text transforms cannot contain local file paths. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "transform_text",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="transform_text", status="refused", reason="local_path_text", transform_text="<local-path>", transform_text_chars=len(text), transform_mode=_short_failure_raw(mode or "unknown", limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    normalized_modes = {
        "upper": "uppercase",
        "uppercase": "uppercase",
        "lower": "lowercase",
        "lowercase": "lowercase",
        "title": "title case",
        "title case": "title case",
        "capitalize": "capitalize",
        "sentence": "sentence case",
        "sentence case": "sentence case",
        "swap case": "swap case",
        "swapcase": "swap case",
        "camel": "camel case",
        "camel case": "camel case",
        "pascal": "pascal case",
        "pascal case": "pascal case",
        "snake": "snake case",
        "snake case": "snake case",
        "kebab": "kebab case",
        "kebab case": "kebab case",
        "initials": "initials",
        "initial": "initials",
        "acronym": "initials",
        "first letters": "initials",
        "slug": "slugify",
        "slugify": "slugify",
        "slugify text": "slugify",
        "normalize spaces": "normalize spaces",
        "normalise spaces": "normalize spaces",
        "collapse spaces": "normalize spaces",
        "clean spaces": "normalize spaces",
        "remove extra spaces": "normalize spaces",
        "remove spaces": "remove spaces",
        "no spaces": "remove spaces",
        "trim": "trim spaces",
        "trim spaces": "trim spaces",
        "trim whitespace": "trim spaces",
        "sort words": "sort words",
        "alphabetize": "sort words",
        "alphabetize words": "sort words",
        "reverse": "reverse text",
        "reverse text": "reverse text",
        "reverse words": "reverse words",
        "unique words": "unique words",
        "dedupe words": "unique words",
        "remove punctuation": "remove punctuation",
        "strip punctuation": "remove punctuation",
        "repeat": "repeat",
    }
    normalized = normalized_modes.get(mode)
    if not normalized:
        failure_output = (
            "Supported text transforms are uppercase, lowercase, title case, sentence case, swap case, camel case, pascal case, snake case, kebab case, initials, slugify, normalize spaces, remove spaces, trim spaces, sort words, reverse, unique words, remove punctuation, and repeat. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "transform_text",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="transform_text", status="refused", reason="unsupported_mode", transform_text=_short_failure_raw(text, limit=MAX_TRANSFORM_TEXT_CHARS), transform_text_chars=len(text), transform_mode=_short_failure_raw(mode or "unknown", limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if normalized == "repeat":
        try:
            count = int(repeat_count)
        except (TypeError, ValueError):
            failure_output = (
                f"Repeat needs a whole-number count. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "transform_text",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _utility_metadata(source="transform_text", status="refused", reason="invalid_count", transform_text=_short_failure_raw(text, limit=MAX_TRANSFORM_TEXT_CHARS), transform_text_chars=len(text), transform_mode=normalized),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
        if count < 1 or count > MAX_REPEAT_TEXT_COUNT:
            failure_output = (
                f"Repeat count must be between 1 and {MAX_REPEAT_TEXT_COUNT}. "
                f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
            )
            return ToolResult(
                "transform_text",
                False,
                failure_output,
                declare_retryable_local_read_failure(
                    _utility_metadata(source="transform_text", status="refused", reason="count_out_of_range", transform_text=_short_failure_raw(text, limit=MAX_TRANSFORM_TEXT_CHARS), transform_text_chars=len(text), transform_mode=normalized, repeat_count=count),
                    output=failure_output,
                    action=LOCAL_READ_INPUT_RECOVERY_ACTION,
                ),
            )
    if normalized == "uppercase":
        output = text.upper()
    elif normalized == "lowercase":
        output = text.lower()
    elif normalized == "title case":
        output = text.title()
    elif normalized == "capitalize":
        output = text[:1].upper() + text[1:] if text else text
    elif normalized == "sentence case":
        output = text[:1].upper() + text[1:].lower() if text else text
    elif normalized == "swap case":
        output = text.swapcase()
    elif normalized in {"camel case", "pascal case", "snake case", "kebab case", "initials"}:
        words = _words(text)
        if normalized == "camel case":
            if words:
                output = words[0].lower() + "".join(word[:1].upper() + word[1:].lower() for word in words[1:])
            else:
                output = ""
        elif normalized == "pascal case":
            output = "".join(word[:1].upper() + word[1:].lower() for word in words)
        elif normalized == "snake case":
            output = "_".join(word.lower() for word in words)
        elif normalized == "kebab case":
            output = "-".join(word.lower() for word in words)
        else:
            output = "".join(word[:1].upper() for word in words)
    elif normalized == "slugify":
        output = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")
    elif normalized == "normalize spaces":
        output = re.sub(r"\s+", " ", text).strip()
    elif normalized == "remove spaces":
        output = re.sub(r"\s+", "", text)
    elif normalized == "trim spaces":
        output = text.strip()
    elif normalized == "sort words":
        output = " ".join(sorted(_words(text), key=str.casefold))
    elif normalized == "reverse text":
        output = text[::-1]
    elif normalized == "reverse words":
        output = " ".join(reversed(_words(text)))
    elif normalized == "unique words":
        words = _words(text)
        seen: set[str] = set()
        unique = []
        for word in words:
            key = word.casefold()
            if key not in seen:
                seen.add(key)
                unique.append(word)
        output = " ".join(unique)
    elif normalized == "remove punctuation":
        output = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()
    else:
        output = " ".join(text for _ in range(count or 1))
    return ToolResult(
        "transform_text",
        True,
        output,
        _utility_metadata(
            source="transform_text",
            status="ok",
            transform_text=text,
            transform_text_chars=len(text),
            transform_mode=normalized,
            transform_word_count=len(_words(text)),
            repeat_count=count,
        ),
    )


def calculate_bmi(args: dict[str, Any]) -> ToolResult:
    def _number(name: str) -> float | None:
        try:
            value = float(args.get(name))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _weight_to_kg(value: float, unit: str) -> float | None:
        normalized = unit.strip().lower()
        if normalized in {"kg", "kilogram", "kilograms"}:
            return value
        if normalized in {"lb", "lbs", "pound", "pounds"}:
            return value * 0.45359237
        return None

    def _height_to_m(value: float, unit: str) -> float | None:
        normalized = unit.strip().lower()
        if normalized in {"m", "meter", "meters"}:
            return value
        if normalized in {"cm", "centimeter", "centimeters"}:
            return value / 100.0
        if normalized in {"in", "inch", "inches"}:
            return value * 0.0254
        if normalized in {"ft", "foot", "feet"}:
            return value * 0.3048
        return None

    weight = _number("weight")
    height = _number("height")
    weight_unit = _short_failure_raw(args.get("weight_unit"), limit=MAX_UNIT_CHARS).lower()
    height_unit = _short_failure_raw(args.get("height_unit"), limit=MAX_UNIT_CHARS).lower()
    if weight is None or height is None:
        failure_output = (
            f"BMI needs numeric weight and height. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate_bmi",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate_bmi", status="refused", reason="bad_value"),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    weight_kg = _weight_to_kg(weight, weight_unit)
    height_m = _height_to_m(height, height_unit)
    if weight_kg is None or height_m is None:
        failure_output = (
            "BMI supports weight in kg or pounds and height in cm, meters, inches, or feet. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate_bmi",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate_bmi", status="refused", reason="unsupported_units", raw_value=f"{weight_unit}/{height_unit}"),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if weight_kg <= 0 or height_m <= 0:
        failure_output = (
            f"BMI needs positive weight and height. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate_bmi",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate_bmi", status="refused", reason="non_positive_value"),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if weight_kg > 1000 or height_m < 0.3 or height_m > 3.0:
        failure_output = (
            "BMI values look outside the supported human-height/weight range. "
            f"{LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "calculate_bmi",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="calculate_bmi", status="refused", reason="out_of_range", weight_kg=round(weight_kg, 4), height_m=round(height_m, 4)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    bmi = weight_kg / (height_m * height_m)
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"
    return ToolResult(
        "calculate_bmi",
        True,
        f"BMI: {_format_calculation_result(bmi)} ({category}). This is a read-only calculation, not a medical diagnosis.",
        _utility_metadata(source="calculate_bmi", status="ok", weight_kg=round(weight_kg, 4), height_m=round(height_m, 4), bmi=round(bmi, 4), bmi_category=category),
    )


def convert_units(args: dict[str, Any]) -> ToolResult:
    try:
        value = float(args.get("value"))
    except (TypeError, ValueError):
        failure_output = (
            f"Value must be a number. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "convert_units",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="convert_units", status="refused", reason="bad_value", raw_value=_short_failure_raw(args.get("value"), limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    if not math.isfinite(value):
        failure_output = (
            f"Value must be finite. {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        )
        return ToolResult(
            "convert_units",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                _utility_metadata(source="convert_units", status="refused", reason="non_finite_value", raw_value=_short_failure_raw(args.get("value"), limit=80)),
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    from_unit = _short_failure_raw(args.get("from_unit"), limit=MAX_UNIT_CHARS).lower()
    to_unit = _short_failure_raw(args.get("to_unit"), limit=MAX_UNIT_CHARS).lower()
    result = _convert(value, from_unit, to_unit)
    ok = not result.startswith("Cannot") and not result.startswith("Unknown")
    metadata = _utility_metadata(
        source="convert_units",
        status="ok" if ok else "refused",
        reason=None if ok else "unsupported_conversion",
        value=value,
        from_unit=from_unit,
        to_unit=to_unit,
        conversion_result_available=ok,
    )
    if not ok:
        failure_output = f"{result} {LOCAL_READ_INPUT_RECOVERY_ACTION}"
        return ToolResult(
            "convert_units",
            False,
            failure_output,
            declare_retryable_local_read_failure(
                metadata,
                output=failure_output,
                action=LOCAL_READ_INPUT_RECOVERY_ACTION,
            ),
        )
    return ToolResult(
        "convert_units",
        True,
        result,
        metadata,
    )


def _convert(value: float, from_unit: str, to_unit: str) -> str:
    temp = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
    if from_unit in temp or to_unit in temp:
        if from_unit not in temp or to_unit not in temp:
            return "Cannot mix temperature with other unit types."
        celsius = (
            (value - 32) * 5 / 9
            if from_unit in {"f", "fahrenheit"}
            else value - 273.15
            if from_unit in {"k", "kelvin"}
            else value
        )
        out = (
            celsius * 9 / 5 + 32
            if to_unit in {"f", "fahrenheit"}
            else celsius + 273.15
            if to_unit in {"k", "kelvin"}
            else celsius
        )
        return f"{value:g} {from_unit} = {out:.6g} {to_unit}"

    units = {
        "m": ("length", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
        "km": ("length", 1000.0), "kilometer": ("length", 1000.0), "kilometers": ("length", 1000.0),
        "cm": ("length", 0.01), "centimeter": ("length", 0.01), "centimeters": ("length", 0.01),
        "mm": ("length", 0.001), "ft": ("length", 0.3048), "feet": ("length", 0.3048),
        "foot": ("length", 0.3048), "in": ("length", 0.0254), "inch": ("length", 0.0254),
        "inches": ("length", 0.0254), "mi": ("length", 1609.344), "mile": ("length", 1609.344),
        "miles": ("length", 1609.344),
        "kg": ("weight", 1.0), "kilogram": ("weight", 1.0), "kilograms": ("weight", 1.0),
        "g": ("weight", 0.001), "gram": ("weight", 0.001), "grams": ("weight", 0.001),
        "lb": ("weight", 0.45359237), "lbs": ("weight", 0.45359237), "pound": ("weight", 0.45359237),
        "pounds": ("weight", 0.45359237), "oz": ("weight", 0.0283495), "ounce": ("weight", 0.0283495),
        "ounces": ("weight", 0.0283495),
        "l": ("volume", 1.0), "liter": ("volume", 1.0), "liters": ("volume", 1.0),
        "ml": ("volume", 0.001), "milliliter": ("volume", 0.001), "milliliters": ("volume", 0.001),
        "gal": ("volume", 3.78541), "gallon": ("volume", 3.78541), "gallons": ("volume", 3.78541),
        "cup": ("volume", 0.236588), "cups": ("volume", 0.236588),
        "s": ("time", 1.0), "sec": ("time", 1.0), "secs": ("time", 1.0),
        "second": ("time", 1.0), "seconds": ("time", 1.0),
        "min": ("time", 60.0), "mins": ("time", 60.0), "minute": ("time", 60.0), "minutes": ("time", 60.0),
        "hr": ("time", 3600.0), "hrs": ("time", 3600.0), "hour": ("time", 3600.0), "hours": ("time", 3600.0),
        "day": ("time", 86400.0), "days": ("time", 86400.0),
        "week": ("time", 604800.0), "weeks": ("time", 604800.0),
    }
    if from_unit not in units:
        return f"Unknown unit: {from_unit}"
    if to_unit not in units:
        return f"Unknown unit: {to_unit}"
    from_group, from_factor = units[from_unit]
    to_group, to_factor = units[to_unit]
    if from_group != to_group:
        return f"Cannot convert {from_group} to {to_group}."
    return f"{value:g} {from_unit} = {value * from_factor / to_factor:.6g} {to_unit}"
