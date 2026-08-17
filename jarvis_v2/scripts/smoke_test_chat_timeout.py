from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import jarvis_v2.agent.model_provider as model_provider_module
from jarvis_v2.agent.model_planner import ModelBackedPlanner
from jarvis_v2.agent.receipts import runtime_result_receipt
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import (
    MAX_CHAT_MAX_HISTORY_MESSAGES,
    MAX_CHAT_MAX_REPLY_TOKENS,
    MAX_OLLAMA_CHAT_TIMEOUT_SECONDS,
    MAX_OLLAMA_MODEL_TIMEOUT_SECONDS,
    MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
    MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
    JarvisConfig,
    load_config,
)
from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile


MODEL_PLANNER_PROBE = "please frobnicate workspace"
REPO_ROOT = Path(__file__).resolve().parents[2]
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(REPO_ROOT) is not None
OLLAMA_LOOPBACK_HOST = "http://127.0.0.1:11434"
OLLAMA_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _with_env(updates: dict[str, str | None]):
    class EnvGuard:
        def __enter__(self):
            self.previous = {key: os.environ.get(key) for key in updates}
            for key, value in updates.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            return self

        def __exit__(self, exc_type, exc, tb):
            for key, value in self.previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    return EnvGuard()


def _local_ollama_env(*, stored_history: bool):
    updates = {key: None for key in OLLAMA_PROXY_ENV_KEYS}
    updates["OLLAMA_HOST"] = OLLAMA_LOOPBACK_HOST
    updates["OLLAMA_NO_CLOUD"] = "1" if stored_history else None
    updates["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = (
        "1" if stored_history else None
    )
    return _with_env(updates)


def test_load_config_splits_planner_and_chat_timeouts() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-timeout-config-") as temp:
        root = Path(temp)
        with _with_env(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_MODEL_TIMEOUT_SECONDS": "3.5",
                "JARVIS_CHAT_TIMEOUT_SECONDS": "21.5",
                "JARVIS_CHAT_MAX_REPLY_TOKENS": "444",
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES": "7",
            }
        ):
            config = load_config()
    if config.model_timeout_seconds != 3.5:
        raise SystemExit(f"planner timeout not loaded: {config.model_timeout_seconds}")
    if config.chat_timeout_seconds != 21.5:
        raise SystemExit(f"chat timeout not loaded: {config.chat_timeout_seconds}")
    if config.chat_max_reply_tokens != 444:
        raise SystemExit(f"chat reply token cap not loaded: {config.chat_max_reply_tokens}")
    if config.chat_max_history_messages != 7:
        raise SystemExit(f"chat history window not loaded: {config.chat_max_history_messages}")


def test_load_config_rejects_non_finite_timeouts() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-timeout-finite-") as temp:
        root = Path(temp)
        with _with_env(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_MODEL_TIMEOUT_SECONDS": "nan",
                "JARVIS_CHAT_TIMEOUT_SECONDS": "inf",
                "JARVIS_CHAT_MAX_REPLY_TOKENS": "bad",
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES": "bad",
            }
        ):
            config = load_config()
    if config.model_timeout_seconds != 2.5:
        raise SystemExit(f"planner timeout should fall back on nan: {config.model_timeout_seconds}")
    if config.chat_timeout_seconds != 20.0:
        raise SystemExit(f"chat timeout should fall back on inf: {config.chat_timeout_seconds}")
    if config.chat_max_reply_tokens != 300:
        raise SystemExit(f"chat reply token cap should fall back on bad int: {config.chat_max_reply_tokens}")
    if config.chat_max_history_messages != 16:
        raise SystemExit(f"chat history window should fall back on bad int: {config.chat_max_history_messages}")


def test_load_config_clamps_tiny_timeouts() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-timeout-clamp-") as temp:
        root = Path(temp)
        with _with_env(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_MODEL_TIMEOUT_SECONDS": "0.1",
                "JARVIS_CHAT_TIMEOUT_SECONDS": "0.1",
                "JARVIS_CHAT_MAX_REPLY_TOKENS": "12",
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES": "-10",
            }
        ):
            config = load_config()
    if config.model_timeout_seconds != 0.5:
        raise SystemExit(f"planner timeout should clamp to lower bound: {config.model_timeout_seconds}")
    if config.chat_timeout_seconds != 1.0:
        raise SystemExit(f"chat timeout should clamp to lower bound: {config.chat_timeout_seconds}")
    if config.chat_max_reply_tokens != 40:
        raise SystemExit(f"chat reply token cap should clamp to lower bound: {config.chat_max_reply_tokens}")
    if config.chat_max_history_messages != 0:
        raise SystemExit(f"chat history window should clamp to lower bound: {config.chat_max_history_messages}")


def test_load_config_caps_provider_specific_timeouts() -> None:
    cases = [
        (
            "ollama",
            MAX_OLLAMA_MODEL_TIMEOUT_SECONDS,
            MAX_OLLAMA_CHAT_TIMEOUT_SECONDS,
        ),
        (
            "openai",
            MAX_OPENAI_MODEL_TIMEOUT_SECONDS,
            MAX_OPENAI_CHAT_TIMEOUT_SECONDS,
        ),
    ]
    for provider, planner_maximum, chat_maximum in cases:
        for planner_raw, chat_raw in (
            (str(planner_maximum), str(chat_maximum)),
            (str(planner_maximum + 1), str(chat_maximum + 1)),
            ("1000000000", "1000000000"),
        ):
            with TemporaryDirectory(prefix=f"jarvis-{provider}-timeout-cap-") as temp:
                root = Path(temp)
                with _with_env(
                    {
                        "JARVIS_DATA_DIR": str(root),
                        "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                        "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                        "JARVIS_MODEL_PROVIDER": provider,
                        "JARVIS_MODEL_TIMEOUT_SECONDS": planner_raw,
                        "JARVIS_CHAT_TIMEOUT_SECONDS": chat_raw,
                    }
                ):
                    config = load_config()
            if config.model_timeout_seconds != planner_maximum:
                raise SystemExit(
                    f"{provider} planner timeout escaped its maximum: {config.model_timeout_seconds}"
                )
            if config.chat_timeout_seconds != chat_maximum:
                raise SystemExit(
                    f"{provider} chat timeout escaped its maximum: {config.chat_timeout_seconds}"
                )


def test_load_config_caps_chat_resources_and_runtime_applies_caps() -> None:
    cases = [
        (str(MAX_CHAT_MAX_REPLY_TOKENS), str(MAX_CHAT_MAX_HISTORY_MESSAGES)),
        (str(MAX_CHAT_MAX_REPLY_TOKENS + 1), str(MAX_CHAT_MAX_HISTORY_MESSAGES + 1)),
        ("9" * 120, "8" * 120),
    ]
    for reply_tokens, history_messages in cases:
        with TemporaryDirectory(prefix="jarvis-chat-resource-caps-") as temp:
            root = Path(temp)
            with _with_env(
                {
                    "JARVIS_DATA_DIR": str(root),
                    "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                    "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                    "JARVIS_CHAT_MAX_REPLY_TOKENS": reply_tokens,
                    "JARVIS_CHAT_MAX_HISTORY_MESSAGES": history_messages,
                }
            ):
                config = load_config()
        if config.chat_max_reply_tokens != MAX_CHAT_MAX_REPLY_TOKENS:
            raise SystemExit(f"chat reply token cap exceeded maximum: {config.chat_max_reply_tokens}")
        if config.chat_max_history_messages != MAX_CHAT_MAX_HISTORY_MESSAGES:
            raise SystemExit(f"chat history window exceeded maximum: {config.chat_max_history_messages}")

    with TemporaryDirectory(prefix="jarvis-chat-resource-runtime-") as temp:
        root = Path(temp)
        with _with_env(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_CHAT_MAX_REPLY_TOKENS": str(MAX_CHAT_MAX_REPLY_TOKENS + 1),
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES": str(MAX_CHAT_MAX_HISTORY_MESSAGES + 1),
                "JARVIS_USE_MODEL_PLANNER": "false",
            }
        ):
            runtime = JarvisRuntime(load_config())
        if runtime.chat.max_reply_tokens != MAX_CHAT_MAX_REPLY_TOKENS:
            raise SystemExit(f"runtime missed bounded chat reply cap: {runtime.chat.max_reply_tokens}")
        if runtime.chat.max_history_messages != MAX_CHAT_MAX_HISTORY_MESSAGES:
            raise SystemExit(f"runtime missed bounded chat history cap: {runtime.chat.max_history_messages}")


def test_load_config_trims_model_planner_flag() -> None:
    with TemporaryDirectory(prefix="jarvis-model-planner-flag-") as temp:
        root = Path(temp)
        with _with_env(
            {
                "JARVIS_DATA_DIR": str(root),
                "JARVIS_DB_PATH": str(root / "jarvis.sqlite"),
                "JARVIS_OBSIDIAN_VAULT": str(root / "Vault"),
                "JARVIS_USE_MODEL_PLANNER": " false ",
            }
        ):
            config = load_config()
    if config.use_model_planner:
        raise SystemExit("load_config did not trim JARVIS_USE_MODEL_PLANNER before parsing")


def test_runtime_uses_chat_timeout_without_changing_planner_timeout() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-timeout-runtime-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-missing-smoke-model",
            planner_model="jarvis-v2-missing-smoke-planner",
            model_timeout_seconds=2.0,
            chat_timeout_seconds=19.0,
            chat_max_reply_tokens=77,
            chat_max_history_messages=5,
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)
        if not isinstance(runtime.planner, ModelBackedPlanner):
            raise SystemExit("runtime did not build a model-backed planner")
        if runtime.planner.timeout_seconds != 2.0:
            raise SystemExit(f"planner timeout drifted: {runtime.planner.timeout_seconds}")
    if runtime.chat.model_timeout_seconds != 19.0:
        raise SystemExit(f"chat timeout was not applied: {runtime.chat.model_timeout_seconds}")
    if runtime.chat.max_reply_tokens != 77:
        raise SystemExit(f"chat reply token cap was not applied: {runtime.chat.max_reply_tokens}")
    if runtime.chat.max_history_messages != 5:
        raise SystemExit(f"chat history window was not applied: {runtime.chat.max_history_messages}")


def test_zero_history_window_sends_and_retains_no_history() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-zero-history-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="mock-chat-model",
            chat_max_history_messages=0,
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        runtime.chat.history = [
            {"role": "user", "content": "private prior question"},
            {"role": "assistant", "content": "private prior answer"},
        ]
        retained_history = runtime.chat.history
        captured: dict[str, object] = {}

        def fake_generate_model_text(**kwargs):
            captured["messages"] = kwargs.get("messages")
            return "mock reply"

        prompt = "why is the sky blue?"
        with _local_ollama_env(stored_history=True):
            preview = runtime.chat.preview_loop(prompt)
            expected_preview_history = {
                "recent_history_messages": 0,
                "history_retained_messages": 0,
                "history_eligible_messages": 0,
                "history_withheld_messages": 0,
                "history_withheld_reason_counts": {},
            }
            wrong_preview_history = {
                key: preview.get(key)
                for key, expected in expected_preview_history.items()
                if preview.get(key) != expected
            }
            if wrong_preview_history:
                raise SystemExit(f"zero history preview retained prior messages: {preview}")
            with patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text):
                if runtime.chat.respond(prompt) != "mock reply":
                    raise SystemExit("zero history model smoke did not use the mocked response")
        messages = captured.get("messages")
        if not isinstance(messages, list):
            raise SystemExit(f"zero history model smoke missed captured messages: {captured}")
        contents = [str(message.get("content") or "") for message in messages if isinstance(message, dict)]
        if "private prior question" in contents or "private prior answer" in contents:
            raise SystemExit(f"zero history window sent prior conversation to the model: {contents}")
        if runtime.chat.history is not retained_history:
            raise SystemExit("zero history retention replaced the history list instead of clearing it in place")
        if runtime.chat.history:
            raise SystemExit(f"zero history window retained conversation after response: {runtime.chat.history}")
        metadata = runtime.chat.last_turn_metadata
        for key, expected in expected_preview_history.items():
            if key == "recent_history_messages":
                continue
            if metadata.get(key) != expected:
                raise SystemExit(f"zero history response metadata drifted for {key}: {metadata}")


def test_finite_history_window_evicts_oldest_messages_in_place() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-finite-history-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="mock-chat-model",
            chat_max_history_messages=4,
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        legacy_history_marker = "LEGACY-PRIVATE-HISTORY-MARKER"
        eligible_history_marker = "ELIGIBLE-PRIVATE-HISTORY-MARKER"
        eligible_history_prompt = "recent question"
        with (
            _local_ollama_env(stored_history=True),
            patch(
                "jarvis_v2.agent.chat.generate_model_text",
                return_value=eligible_history_marker,
            ) as seed_call,
        ):
            if not runtime._refresh_history_policy_binding():
                raise SystemExit("finite history could not activate the consented runtime policy")
            seed_answer = runtime.chat.respond(eligible_history_prompt)
        if seed_answer != eligible_history_marker or seed_call.call_count != 1:
            raise SystemExit("finite history seed did not complete one mocked ChatBrain turn")
        retained_history = runtime.chat.history
        retained_history[:0] = [
            {"role": "user", "content": legacy_history_marker},
            {"role": "assistant", "content": "legacy private answer"},
        ]
        captured: dict[str, object] = {}

        def fake_generate_model_text(**kwargs):
            captured["messages"] = kwargs.get("messages")
            return "new mock reply"

        prompt = "explain refraction simply"
        with (
            _local_ollama_env(stored_history=True),
            patch("jarvis_v2.agent.chat.generate_model_text", side_effect=fake_generate_model_text),
        ):
            if runtime.chat.respond(prompt) != "new mock reply":
                raise SystemExit("finite history model smoke did not use the mocked response")

        messages = captured.get("messages")
        if not isinstance(messages, list):
            raise SystemExit(f"finite history model smoke missed captured messages: {captured}")
        model_contents = [
            str(message.get("content") or "") for message in messages if isinstance(message, dict)
        ]
        model_text = "\n".join(model_contents)
        if model_text.count(eligible_history_marker) != 1:
            raise SystemExit(
                "finite history did not send eligible history exactly once before applying retention: "
                f"messages={model_contents}, metadata={runtime.chat.last_turn_metadata}"
            )
        legacy_markers = (legacy_history_marker, "legacy private answer")
        disclosed_legacy = [marker for marker in legacy_markers if model_text.count(marker) != 0]
        if disclosed_legacy:
            raise SystemExit(f"finite history disclosed unmarked legacy rows: {disclosed_legacy}")
        metadata = runtime.chat.last_turn_metadata
        expected_history_metadata = {
            "history_retained_messages": 4,
            "history_eligible_messages": 2,
            "history_withheld_messages": 2,
            "history_withheld_reason_counts": {"legacy_unmarked": 2},
            "history_disclosure_prepare_status": "prepared",
            "history_disclosure_finalize_status": "confirmed",
            "history_disclosure_receipt_prepared": True,
        }
        wrong_history_metadata = {
            key: metadata.get(key)
            for key, expected in expected_history_metadata.items()
            if metadata.get(key) != expected
        }
        if wrong_history_metadata:
            raise SystemExit(
                f"finite history eligibility metadata drifted: {wrong_history_metadata} / {metadata}"
            )
        if runtime.chat.history is not retained_history:
            raise SystemExit("finite history retention replaced the history list instead of trimming it in place")
        if len(runtime.chat.history) != 4:
            raise SystemExit(f"finite history window retained the wrong message count: {runtime.chat.history}")
        retained_contents = [message["content"] for message in runtime.chat.history]
        if legacy_history_marker in retained_contents or "legacy private answer" in retained_contents:
            raise SystemExit(f"finite history retained the oldest private turn: {retained_contents}")
        if retained_contents != [
            eligible_history_prompt,
            eligible_history_marker,
            prompt,
            "new mock reply",
        ]:
            raise SystemExit(f"finite history evicted the wrong messages: {retained_contents}")


def _with_fake_ollama(chat_impl):
    class ModuleGuard:
        def __enter__(self):
            self.environment = _local_ollama_env(stored_history=False)
            self.environment.__enter__()

            class FakeResponse:
                def __init__(self, payload):
                    self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self, size=-1):
                    return self.payload if size < 0 else self.payload[:size]

            class FakeOpener:
                def open(self, request, *, timeout):
                    payload = json.loads(request.data.decode("utf-8"))
                    result = chat_impl(
                        model=payload.get("model"),
                        messages=payload.get("messages"),
                        options=payload.get("options"),
                        keep_alive=payload.get("keep_alive"),
                    )
                    if not isinstance(result, dict):
                        raise TypeError("mocked Ollama result must be a dict")
                    wrapped = dict(result)
                    wrapped.setdefault("done", True)
                    wrapped.setdefault("done_reason", "stop")
                    return FakeResponse(wrapped)

            self.opener_patch = patch.object(
                model_provider_module,
                "build_opener",
                return_value=FakeOpener(),
            )
            self.opener_patch.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.opener_patch.__exit__(exc_type, exc, tb)
            return self.environment.__exit__(exc_type, exc, tb)

    return ModuleGuard()


def test_model_planner_records_used_and_fallback_metadata() -> None:
    with TemporaryDirectory(prefix="jarvis-model-planner-diagnostics-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-smoke-chat",
            planner_model="jarvis-v2-smoke-planner",
            model_timeout_seconds=1.25,
            chat_timeout_seconds=3.0,
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)

        def valid_tool_response(*args, **kwargs):
            return {"message": {"content": '{"mode":"tool","goal":"show help","actions":[{"tool_name":"jarvis_help","args":{"topic":"voice"},"reason":"safe help lookup"}]}' }}

        with _with_fake_ollama(valid_tool_response):
            plan = runtime.planner.plan(MODEL_PLANNER_PROBE)
        if plan.notes != "model_planner" or not plan.actions or plan.actions[0].tool_name != "jarvis_help":
            raise SystemExit(f"model planner did not use the valid mocked tool plan: {plan}")
        metadata = plan.metadata
        if metadata.get("model_planner_state") != "used" or metadata.get("model_planner_used") is not True:
            raise SystemExit(f"model planner used plan missed used metadata: {metadata}")
        if metadata.get("model_planner_timeout_seconds") != 1.25 or metadata.get("model_planner_model") != "jarvis-v2-smoke-planner":
            raise SystemExit(f"model planner used plan missed model/timeout metadata: {metadata}")
        for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
            if metadata.get(key) is not False:
                raise SystemExit(f"model planner used metadata should not grant {key}: {metadata}")

        def unknown_tool_response(*args, **kwargs):
            return {"message": {"content": '{"mode":"tool","goal":"unsafe mystery","actions":[{"tool_name":"missing_secret_tool","args":{"path":"/\x55sers/example/private"},"reason":"bad route"}]}' }}

        with _with_fake_ollama(unknown_tool_response):
            fallback_plan = runtime.planner.plan(MODEL_PLANNER_PROBE)
        fallback_metadata = fallback_plan.metadata
        if fallback_plan.actions or fallback_plan.needs_model is not True:
            raise SystemExit(f"unknown model tool should fall back to base chat plan: {fallback_plan}")
        if fallback_metadata.get("model_planner_state") != "fallback" or fallback_metadata.get("model_planner_fallback_reason") != "no_valid_model_actions":
            raise SystemExit(f"unknown model tool missed fallback metadata: {fallback_metadata}")
        if fallback_metadata.get("model_planner_ignored_unknown_tools") != ["<unrecognized-model-tool>"]:
            raise SystemExit(f"unknown model tool should retain content-free count evidence: {fallback_metadata}")
        if "/\x55sers/" in str(fallback_metadata) or "/private" in str(fallback_metadata):
            raise SystemExit(f"model planner fallback metadata leaked a local path: {fallback_metadata}")


def test_runtime_trace_carries_model_planner_fallback_metadata() -> None:
    with TemporaryDirectory(prefix="jarvis-model-planner-runtime-trace-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-smoke-chat",
            planner_model="jarvis-v2-smoke-planner",
            model_timeout_seconds=1.5,
            chat_timeout_seconds=3.0,
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)

        def chat_mode_response(*args, **kwargs):
            return {"message": {"content": '{"mode":"chat","goal":"chat instead","actions":[]}' }}

        with _with_fake_ollama(chat_mode_response):
            result = runtime.handle(MODEL_PLANNER_PROBE)
        trace = result.metadata.get("runtime_trace") or {}
        planner_metadata = trace.get("planner_metadata") or {}
        if result.metadata.get("runtime_route") != "chat" or trace.get("route") != "chat":
            raise SystemExit(f"fallback planner trace should continue through chat route: {result.metadata}")
        if planner_metadata.get("model_planner_state") != "fallback" or planner_metadata.get("model_planner_fallback_reason") != "model_chose_chat":
            raise SystemExit(f"runtime trace missed model planner fallback metadata: {trace}")
        planning_stage = next((stage for stage in trace.get("stages", []) if stage.get("stage") == "planning"), {})
        if (planning_stage.get("planner_metadata") or {}).get("model_planner_fallback_reason") != "model_chose_chat":
            raise SystemExit(f"planning stage missed planner metadata: {trace}")
        for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
            if planner_metadata.get(key) is not False:
                raise SystemExit(f"runtime planner metadata should not grant {key}: {planner_metadata}")

        receipt = runtime_result_receipt(result, pending_approvals=0)
        if "model planner" not in str(receipt.get("planner_notes") or "").lower():
            raise SystemExit(f"runtime receipt missed planner notes: {receipt}")
        receipt_planner = receipt.get("planner_metadata") or {}
        if receipt_planner.get("model_planner_state") != "fallback":
            raise SystemExit(f"runtime receipt missed planner metadata: {receipt}")
        if receipt.get("planner_model_planner_fallback_reason") != "model_chose_chat":
            raise SystemExit(f"runtime receipt missed flat fallback reason: {receipt}")
        for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
            if receipt_planner.get(key) is not False:
                raise SystemExit(f"runtime receipt planner metadata should not grant {key}: {receipt_planner}")


def test_model_planner_exception_names_recovery_without_raw_error() -> None:
    with TemporaryDirectory(prefix="jarvis-model-planner-exception-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-smoke-chat",
            planner_model="jarvis-v2-smoke-planner",
            model_timeout_seconds=1.5,
            chat_timeout_seconds=3.0,
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)

        def unavailable_model(*args, **kwargs):
            raise OSError("ollama offline near /\x55sers/example/private/model-cache and /tmp/jarvis-plan")

        with _with_fake_ollama(unavailable_model):
            result = runtime.handle(MODEL_PLANNER_PROBE)
        trace = result.metadata.get("runtime_trace") or {}
        planner_metadata = trace.get("planner_metadata") or {}
        if result.metadata.get("runtime_route") != "chat":
            raise SystemExit(f"model planner exception should fall back through chat: {result.metadata}")
        expected_hint_parts = [
            "model routing status",
            "response stream failed",
            "Diagnostic: planner_model_unavailable",
        ]
        if planner_metadata.get("model_planner_fallback_reason") != "model_exception":
            raise SystemExit(f"model planner exception missed fallback reason: {planner_metadata}")
        if planner_metadata.get("model_planner_exception_type") != "ModelProviderError":
            raise SystemExit(f"model planner exception missed bounded exception type: {planner_metadata}")
        if planner_metadata.get("model_planner_error") != "ollama_stream_failed":
            raise SystemExit(f"model planner exception missed adapter diagnostic: {planner_metadata}")
        hint = str(planner_metadata.get("model_planner_recovery_hint") or "")
        for expected in expected_hint_parts:
            if expected not in hint:
                raise SystemExit(f"model planner recovery hint missed {expected!r}: {planner_metadata}")
        if any(fragment in str(planner_metadata) for fragment in ["/\x55sers/", "/tmp/", "private/model-cache", "ollama offline"]):
            raise SystemExit(f"model planner exception leaked raw local error: {planner_metadata}")

        receipt = runtime_result_receipt(result, pending_approvals=0)
        receipt_planner = receipt.get("planner_metadata") or {}
        if receipt.get("planner_model_planner_recovery_hint") != receipt_planner.get("model_planner_recovery_hint"):
            raise SystemExit(f"runtime receipt missed flat recovery hint: {receipt}")
        for expected in expected_hint_parts:
            if expected not in str(receipt):
                raise SystemExit(f"runtime receipt missed recovery hint text {expected!r}: {receipt}")
        if any(fragment in str(receipt) for fragment in ["/\x55sers/", "/tmp/", "private/model-cache", "ollama offline"]):
            raise SystemExit(f"runtime receipt leaked raw model planner exception: {receipt}")
        for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
            if receipt_planner.get(key) is not False:
                raise SystemExit(f"runtime receipt planner metadata should not grant {key}: {receipt_planner}")


def test_runtime_result_receipt_bounds_model_planner_diagnostics() -> None:
    with TemporaryDirectory(prefix="jarvis-model-planner-receipt-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-smoke-chat",
            planner_model="jarvis-v2-smoke-planner",
            model_timeout_seconds=1.5,
            chat_timeout_seconds=3.0,
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)

        def unknown_tool_response(*args, **kwargs):
            return {
                "message": {
                    "content": (
                        '{"mode":"tool","goal":"unsafe mystery","actions":['
                        '{"tool_name":"missing_secret_tool","args":{"path":"/\x55sers/example/private"},"reason":"bad route"},'
                        '{"tool_name":"/private/tmp/jarvis-secret-tool","args":{},"reason":"bad path"}]}'
                    )
                }
            }

        with _with_fake_ollama(unknown_tool_response):
            result = runtime.handle(MODEL_PLANNER_PROBE)
        receipt = runtime_result_receipt(result, pending_approvals=0)
        receipt_text = str(receipt)
        for forbidden in ["/\x55sers/", "/private/", "/tmp/"]:
            if forbidden in receipt_text:
                raise SystemExit(f"runtime receipt leaked local planner diagnostic path: {receipt}")
        if receipt.get("planner_model_planner_attempted") is not True:
            raise SystemExit(f"runtime receipt missed model planner attempted flag: {receipt}")
        if receipt.get("planner_model_planner_state") != "fallback":
            raise SystemExit(f"runtime receipt missed model planner fallback state: {receipt}")
        if receipt.get("planner_model_planner_fallback_reason") != "no_valid_model_actions":
            raise SystemExit(f"runtime receipt missed model planner fallback reason: {receipt}")
        if receipt.get("planner_model_planner_ignored_unknown_tools") != ["<unrecognized-model-tool>", "<unrecognized-model-tool>"]:
            raise SystemExit(f"runtime receipt missed bounded ignored tool diagnostics: {receipt}")
        receipt_planner = receipt.get("planner_metadata") or {}
        if receipt_planner.get("model_planner_ignored_unknown_tools") != ["<unrecognized-model-tool>", "<unrecognized-model-tool>"]:
            raise SystemExit(f"runtime receipt nested planner metadata diverged: {receipt}")
        for key in ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]:
            if receipt_planner.get(key) is not False:
                raise SystemExit(f"runtime receipt planner metadata should not grant {key}: {receipt_planner}")


def test_chat_metadata_reports_chat_timeout() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-timeout-chat-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-missing-smoke-model",
            model_timeout_seconds=1.5,
            chat_timeout_seconds=18.0,
            chat_max_reply_tokens=88,
            chat_max_history_messages=4,
            use_model_planner=False,
        )
        runtime = JarvisRuntime(config)
        with _local_ollama_env(stored_history=False):
            result = runtime.handle("what's up?")
        chat_response = result.metadata.get("chat_response", {})
        if result.metadata.get("runtime_route") != "chat":
            raise SystemExit("free-form chat did not use chat route")
        if chat_response.get("model_timeout_seconds") != 18.0:
            raise SystemExit(f"chat metadata missed chat timeout: {chat_response}")
        if chat_response.get("max_reply_tokens") != 88:
            raise SystemExit(f"chat metadata missed reply token cap: {chat_response}")
        if chat_response.get("max_history_messages") != 4:
            raise SystemExit(f"chat metadata missed history window: {chat_response}")
        if chat_response.get("source") not in {"model", "fallback"}:
            raise SystemExit(f"unexpected chat response source: {chat_response}")


def test_model_status_reports_both_timeouts() -> None:
    with TemporaryDirectory(prefix="jarvis-chat-timeout-status-") as temp:
        root = Path(temp)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-missing-smoke-model",
            planner_model="jarvis-v2-missing-smoke-planner",
            model_timeout_seconds=2.25,
            chat_timeout_seconds=20.25,
            chat_max_reply_tokens=222,
            chat_max_history_messages=6,
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)
        with _local_ollama_env(stored_history=False):
            result = runtime.handle("model routing status")
        if not result.verified:
            raise SystemExit("model routing status did not run")
        output = result.response
        metadata = result.tool_results[0].metadata
        for expected in [
            "model timeout split",
            "planner timeout: 2.25s",
            "chat timeout: 20.25s",
            "chat reply token cap: 222",
            "chat history window: 6 message(s)",
        ]:
            if expected not in output:
                raise SystemExit(f"model status missed split timeout text: {expected}")
        if metadata.get("model_timeout_seconds") != 2.25:
            raise SystemExit("model status metadata missed planner timeout")
        if metadata.get("chat_timeout_seconds") != 20.25:
            raise SystemExit("model status metadata missed chat timeout")
        if metadata.get("chat_max_reply_tokens") != 222:
            raise SystemExit("model status metadata missed chat reply token cap")
        if metadata.get("chat_max_history_messages") != 6:
            raise SystemExit("model status metadata missed chat history window")


def test_readme_documents_split_timeouts() -> None:
    readme = Path("README.md").read_text()
    expected_parts = [
        "JARVIS_MODEL_TIMEOUT_SECONDS",
        "JARVIS_CHAT_TIMEOUT_SECONDS",
        "JARVIS_CHAT_MAX_REPLY_TOKENS",
        "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        "Planner model calls are bounded",
        "Free-form chat uses",
        "history window",
    ]
    for expected in expected_parts:
        if expected not in readme:
            raise SystemExit(f"README missed split timeout documentation: {expected}")


def test_finish_plan_documents_history_window_knob_and_live_acceptance() -> None:
    plan_path = REPO_ROOT / "FINISH_PLAN_AUGUST.md"
    if IS_PUBLIC_CANDIDATE:
        if plan_path.exists():
            raise SystemExit(
                "public candidate retained private operational reference: FINISH_PLAN_AUGUST.md"
            )
        return
    plan = plan_path.read_text()
    normalized = " ".join(plan.split())
    expected_parts = [
        "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
        "chat_max_history_messages",
        "default remains 16",
        "opt-in tuning knob",
        "this acceptance item remains open",
        "fresh mixed-conversation run passes",
    ]
    for expected in expected_parts:
        if expected not in normalized:
            raise SystemExit(f"FINISH_PLAN missed chat history window handoff: {expected}")
    if "trimming that window is the next knob, not yet touched" in plan:
        raise SystemExit("FINISH_PLAN still says the chat history knob has not been added")
    if "- [x] A full mixed conversation" not in plan:
        raise SystemExit("FINISH_PLAN should record the live mixed-conversation acceptance as closed")
    for expected in (
        "LIVE ACCEPTANCE CLOSED 2026-07-28",
        "maximum 4734.1ms",
        "no fallback or safety-preflight",
    ):
        if expected not in plan:
            raise SystemExit(f"FINISH_PLAN missed live mixed-conversation acceptance evidence: {expected}")


def test_codex_tasks_documents_split_timeouts() -> None:
    tasks_path = REPO_ROOT / "CODEX_TASKS.md"
    if IS_PUBLIC_CANDIDATE:
        if tasks_path.exists():
            raise SystemExit("public candidate retained private operational reference: CODEX_TASKS.md")
        return
    tasks = tasks_path.read_text()
    expected_parts = [
        "split planner/chat timeouts",
        "JARVIS_CHAT_TIMEOUT_SECONDS",
        "kept\nplanner/model timeout separate",
        "T1 through T8 are complete",
        "readiness-maintenance work only",
        "T1 through T8 are DONE and verified",
        "do not restart the completed T3 → T8 queue",
        "Live checks are user-approved only",
        "do not restart\n   daemons, message the bot, send real messages, or trigger other external side effects",
        "Live Telegram/phone verification is optional and user-approved only",
        "do not perform external side\n  effects, daemon restarts, or real message sends during autonomous readiness passes",
    ]
    for expected in expected_parts:
        if expected not in tasks:
            raise SystemExit(f"CODEX_TASKS missed split timeout documentation: {expected}")
    if "8s timeout" in tasks:
        raise SystemExit("CODEX_TASKS still documents a single stale 8s model timeout")
    if "prioritized remaining work" in tasks:
        raise SystemExit("CODEX_TASKS still frames the completed queue as remaining work")
    if "Begin with T3" in tasks:
        raise SystemExit("CODEX_TASKS still tells future agents to restart the completed T3 queue")
    if "Live-verified by restarting the Telegram daemon" in tasks:
        raise SystemExit("CODEX_TASKS still requires live Telegram verification during autonomous readiness")
    if "launchctl kickstart" in tasks:
        raise SystemExit("CODEX_TASKS still instructs autonomous agents to restart the Telegram daemon")


def test_codex_tasks_marks_c1_c4_complete_not_pending() -> None:
    tasks_path = REPO_ROOT / "CODEX_TASKS.md"
    if IS_PUBLIC_CANDIDATE:
        if tasks_path.exists():
            raise SystemExit("public candidate retained private operational reference: CODEX_TASKS.md")
        return
    tasks = tasks_path.read_text()
    current = tasks.split("## Autonomous Maintenance Log", 1)[0]
    expected_parts = [
        "C1-C4 are DONE and verified",
        "do NOT redo C1-C4",
        "Completed C1-C4 queue",
        "Current Codex lane while the live-proof freeze remains",
    ]
    for expected in expected_parts:
        if expected not in current:
            raise SystemExit(f"CURRENT INSTRUCTIONS missed C1-C4 completion handoff: {expected}")
    if re.search(r"Wait for [^\n]+ live matrix results", current) is None:
        raise SystemExit("CURRENT INSTRUCTIONS missed the operator-present live-matrix boundary")
    stale_parts = [
        "Work the queue below, in order",
        "### C1 — Morning brief at 09:00 KST",
        "### C2 — Channel health report",
        "### C3 — Dashboard voice parity",
        "### C4 — Whisper warm start",
        "only after C1–C3 are green",
    ]
    for stale in stale_parts:
        if stale in current:
            raise SystemExit(f"CURRENT INSTRUCTIONS still frames completed C1-C4 work as pending: {stale}")


def test_codex_tasks_labels_historical_live_proof_summary() -> None:
    tasks_path = REPO_ROOT / "CODEX_TASKS.md"
    if IS_PUBLIC_CANDIDATE:
        if tasks_path.exists():
            raise SystemExit("public candidate retained private operational reference: CODEX_TASKS.md")
        return
    tasks = tasks_path.read_text()
    current = tasks.split("## Autonomous Maintenance Log", 1)[0]
    if "## Verification Summary (Current Session)" in current:
        raise SystemExit("CODEX_TASKS still labels historical live-proof notes as the current session")
    expected_parts = [
        "## Historical Verification Summary (Prior Live-Proof Session; Not Current Permission)",
        "preserved history from a prior live-proof session",
        "does not override the 2026-07-04",
        "current instructions, freeze list, or no-daemon/no-live-action standing rules",
    ]
    for expected in expected_parts:
        if expected not in current:
            raise SystemExit(f"CODEX_TASKS missed historical live-proof caveat: {expected}")


def test_codex_tasks_current_log_contains_latest_handoff_notes() -> None:
    tasks_path = REPO_ROOT / "CODEX_TASKS.md"
    if IS_PUBLIC_CANDIDATE:
        if tasks_path.exists():
            raise SystemExit("public candidate retained private operational reference: CODEX_TASKS.md")
        return
    tasks = tasks_path.read_text()
    if "## Autonomous Maintenance Log" not in tasks:
        raise SystemExit("CODEX_TASKS is missing the Autonomous Maintenance Log heading")
    current_log = tasks.split("## Autonomous Maintenance Log", 1)[1].split("\n## ", 1)[0]
    entries: list[str] = []
    for line in current_log.splitlines():
        if line.startswith("- "):
            entries.append(line)
        elif entries and line.startswith("  "):
            entries[-1] += "\n" + line

    # This check validates CODEX'S OWN logging template specifically
    # (aggregate-proof command + numeric module count, exact freeze-boundary
    # phrases). It used to
    # assume Codex's entries always occupy the literal head of the log, but
    # Claude now also prepends entries to this same shared file (its own,
    # differently-formatted handoffs) -- so filter to the most recent entries
    # that actually MATCH Codex's anchored header shape (reusing the exact
    # same regex used to validate them below), not a loose "(Codex)" substring
    # search -- a Claude entry that merely *mentions* "(Codex)" in its prose
    # (e.g. while describing this very check) must not be misidentified as
    # one of Codex's own entries just because the substring appears somewhere
    # in its (single-line, unwrapped) body text. The module count is numeric
    # rather than fixed because the suite legitimately grows when new smoke
    # modules are registered.
    codex_header_re = re.compile(r"^- (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) KST \(Codex\): (DONE|PROGRESS)\b")
    codex_entries = [entry for entry in entries if codex_header_re.match(entry)]
    top_entries = codex_entries[:3]
    if len(top_entries) < 3:
        raise SystemExit("Autonomous Maintenance Log should keep at least three current Codex handoff entries")

    parsed_times: list[datetime] = []
    boundary_parts = [
        "Freeze honored",
        "did not edit planner",
        "call/Kakao/Instagram connectors",
        "HUD",
        "no daemon restart",
        "live send/call",
        "approval",
        "secret access",
        "personal-data action",
        "external side effect",
        "destructive action",
        "computer/browser-control action",
        "microphone/audio/transcription/live channel action",
    ]
    for entry in top_entries:
        match = codex_header_re.match(entry)
        if match is None:
            raise SystemExit(f"Autonomous Maintenance Log entry has an unexpected shape: {entry[:120]}")
        parsed_times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M"))
        if "python3 -m jarvis_v2.scripts.smoke_test_all" not in entry:
            raise SystemExit(
                "Autonomous Maintenance Log entry missed aggregate proof: "
                "python3 -m jarvis_v2.scripts.smoke_test_all"
            )
        if not re.search(r"green \([1-9][0-9]{2,} modules", entry):
            raise SystemExit("Autonomous Maintenance Log entry missed aggregate module count")
        for expected in boundary_parts:
            if expected not in entry:
                raise SystemExit(f"Autonomous Maintenance Log entry missed freeze boundary proof: {expected}")

    if parsed_times != sorted(parsed_times, reverse=True):
        raise SystemExit("Autonomous Maintenance Log top entries should be newest-first")


def main() -> None:
    test_load_config_splits_planner_and_chat_timeouts()
    test_load_config_rejects_non_finite_timeouts()
    test_load_config_clamps_tiny_timeouts()
    test_load_config_caps_provider_specific_timeouts()
    test_load_config_caps_chat_resources_and_runtime_applies_caps()
    test_load_config_trims_model_planner_flag()
    test_runtime_uses_chat_timeout_without_changing_planner_timeout()
    test_zero_history_window_sends_and_retains_no_history()
    test_finite_history_window_evicts_oldest_messages_in_place()
    test_model_planner_records_used_and_fallback_metadata()
    test_runtime_trace_carries_model_planner_fallback_metadata()
    test_model_planner_exception_names_recovery_without_raw_error()
    test_runtime_result_receipt_bounds_model_planner_diagnostics()
    test_chat_metadata_reports_chat_timeout()
    test_model_status_reports_both_timeouts()
    test_readme_documents_split_timeouts()
    test_finish_plan_documents_history_window_knob_and_live_acceptance()
    test_codex_tasks_documents_split_timeouts()
    test_codex_tasks_marks_c1_c4_complete_not_pending()
    test_codex_tasks_labels_historical_live_proof_summary()
    test_codex_tasks_current_log_contains_latest_handoff_notes()
    print("Chat timeout smoke passed")


if __name__ == "__main__":
    main()
