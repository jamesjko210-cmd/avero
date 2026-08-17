"""Smoke tests for conversation compaction (the Memory Trees job).

All model calls are injected fakes — no Ollama, no network, no live DB.
"""

from __future__ import annotations

import json
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.automations import compaction as compaction_module
from jarvis_v2.automations.compaction import (
    CompactionSummarizer,
    KEEP_RECENT_HOURS,
    MIN_BATCH_MESSAGES,
    NOTHING_TO_COMPACT,
    build_conversation_compaction as _build_conversation_compaction,
    fetch_compactable_messages,
)
from jarvis_v2.automations.scheduler import Scheduler, iso
from jarvis_v2.scripts.test_runtime import make_temp_runtime


LOCAL_SUMMARIZER_ROUTE = CompactionSummarizer(
    callback=lambda _text: "NOTHING_DURABLE",
    provider="test-local",
    model_identifier="deterministic-v1",
    destination_class="in_process_test",
    explicit_consent_satisfied=True,
    remote_history_egress=True,
)


def _declared(callback, *, route: CompactionSummarizer = LOCAL_SUMMARIZER_ROUTE) -> CompactionSummarizer:
    return replace(route, callback=callback)


def _remote_route(
    provider: str,
    model: str,
    destination: str,
    *,
    consent: bool,
) -> CompactionSummarizer:
    return CompactionSummarizer(
        callback=lambda _text: "NOTHING_DURABLE",
        provider=provider,
        model_identifier=model,
        destination_class=destination,
        explicit_consent_satisfied=consent,
        remote_history_egress=True,
    )


def _activate_route(store, route: CompactionSummarizer) -> str:
    material = "|".join(
        (
            route.provider,
            route.model_identifier,
            route.destination_class,
            "1" if route.explicit_consent_satisfied else "0",
        )
    )
    return store.ensure_active_history_policy_epoch(
        policy_fingerprint=hashlib.sha256(material.encode("ascii")).hexdigest(),
        provider=route.provider,
        model_identifier=route.model_identifier,
        destination_class=route.destination_class,
        session_generation=1,
        explicit_consent_satisfied=route.explicit_consent_satisfied,
    )


def _local_route_decision():
    return compaction_module._route_from_declared_summarizer(LOCAL_SUMMARIZER_ROUTE)


def _seed_messages(
    store,
    count: int,
    *,
    hours_old: float,
    start_role: str = "user",
    route: CompactionSummarizer = LOCAL_SUMMARIZER_ROUTE,
) -> None:
    """Insert messages with explicit (old) UTC timestamps."""
    stamp = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours_old)).strftime("%Y-%m-%dT%H:%M:%SZ")
    epoch_id = _activate_route(store, route)
    message_ids: list[int] = []
    for index in range(count):
        role = start_role if index % 2 == 0 else "assistant"
        message_ids.append(
            store.log_message(
                "smoke",
                role,
                f"message {index}: project jarvis milestone {index}",
                provenance={
                    "role": role,
                    "policy_epoch_id": epoch_id,
                    "lineage_state": "direct_current",
                    "destination_class": route.destination_class,
                    "remote_eligible": bool(
                        route.remote_history_egress
                        and route.explicit_consent_satisfied
                    ),
                },
            )
        )
    with store.connect() as conn:
        conn.executemany(
            "UPDATE messages SET created_at = ? WHERE id = ?",
            [(stamp, message_id) for message_id in message_ids],
        )


def build_conversation_compaction(store, vault, config, **kwargs):
    summarize = kwargs.get("summarize")
    if summarize is not None and not isinstance(summarize, CompactionSummarizer):
        kwargs["summarize"] = _declared(summarize)
    return _build_conversation_compaction(store, vault, config, **kwargs)


def _compaction_rows(store) -> list[dict]:
    with store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM conversation_compaction_batches ORDER BY id")]


def _digest_rows(store) -> list[dict]:
    with store.connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM memories WHERE category = 'conversation-digest' ORDER BY id"
            )
        ]


def _daily_text(runtime) -> str:
    path = runtime.vault.root_path / "Daily" / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_small_backlog_is_quiet_and_keeps_watermark() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-quiet-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, MIN_BATCH_MESSAGES - 1, hours_old=KEEP_RECENT_HOURS + 24)
        output, watermark = build_conversation_compaction(
            runtime.store, runtime.vault, runtime.config,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("model must not be called")),
        )
        if NOTHING_TO_COMPACT not in output or watermark is not None:
            raise SystemExit(f"small backlog should be quiet without touching the model: {output!r} {watermark}")
        if runtime.store.recent_memories(limit=5):
            raise SystemExit("quiet run must not write memories")


def test_recent_messages_are_never_compacted() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-recent-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 60, hours_old=1)  # live context, too fresh
        rows = fetch_compactable_messages(runtime.store, 0)
        if rows:
            raise SystemExit(f"messages inside the {KEEP_RECENT_HOURS}h window must be excluded: {len(rows)}")


def test_batch_compacts_into_memory_and_advances_watermark() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-batch-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 50, hours_old=KEEP_RECENT_HOURS + 24)
        calls: list[str] = []
        output, watermark = build_conversation_compaction(
            runtime.store, runtime.vault, runtime.config,
            summarize=lambda text: calls.append(text) or "- the operator is building Jarvis V2\n- decision: tiny model for voice",
        )
        if not calls:
            raise SystemExit("summarizer should be called for a full batch")
        if watermark is None or watermark <= 0:
            raise SystemExit(f"watermark should advance after compaction: {watermark}")
        memories = runtime.store.recent_memories(limit=5)
        if not memories or memories[0]["category"] != "conversation-digest":
            raise SystemExit(f"compaction should write a conversation-digest memory: {[dict(m) for m in memories]}")
        if "Jarvis V2" not in memories[0]["body"]:
            raise SystemExit(f"digest body should carry the summary: {memories[0]['body']!r}")
        if "Compacted 50 messages" not in output:
            raise SystemExit(f"report should state what was compacted: {output!r}")

        # Second run: everything already compacted -> quiet, no new memory.
        output2, watermark2 = build_conversation_compaction(
            runtime.store, runtime.vault, runtime.config,
            last_compacted_id=watermark,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("nothing left to summarize")),
        )
        if NOTHING_TO_COMPACT not in output2 or watermark2 is not None:
            raise SystemExit(f"second run should be quiet: {output2!r}")


def test_model_failure_keeps_watermark_and_writes_nothing() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-fail-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        output, watermark = build_conversation_compaction(
            runtime.store, runtime.vault, runtime.config,
            summarize=lambda text: (_ for _ in ()).throw(ConnectionError("ollama down")),
        )
        if "model unavailable" not in output or watermark is not None:
            raise SystemExit(f"model failure must skip compaction and keep the watermark: {output!r} {watermark}")
        if runtime.store.recent_memories(limit=5):
            raise SystemExit("failed run must not write partial memories")


def test_openai_compaction_is_blocked_without_separate_opt_in() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-openai-blocked-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(
            runtime.config,
            model_provider="openai",
            chat_model="gpt-5.6-terra",
            allow_remote_conversation_compaction=False,
        )
        _seed_messages(
            runtime.store,
            40,
            hours_old=KEEP_RECENT_HOURS + 24,
            route=_remote_route(
                "openai", "gpt-5.6-terra", "external_provider", consent=False
            ),
        )
        with mock.patch(
            "jarvis_v2.agent.model_provider.generate_model_text",
            side_effect=AssertionError("remote model must not be called without compaction opt-in"),
        ) as model_call:
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
            )
        for expected in (
            "not approved for this model route",
            "No conversation text was sent",
            "watermark was not advanced",
            "JARVIS_ALLOW_REMOTE_COMPACTION=1",
        ):
            if expected not in output:
                raise SystemExit(f"blocked remote compaction missed {expected!r}: {output!r}")
        if model_call.called or watermark is not None:
            raise SystemExit(f"blocked remote compaction crossed its boundary: called={model_call.called}, watermark={watermark}")
        if runtime.store.recent_memories(limit=5):
            raise SystemExit("blocked remote compaction must not write a digest")


def test_openai_compaction_runs_only_after_explicit_opt_in() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-openai-opt-in-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(
            runtime.config,
            model_provider="openai",
            chat_model="gpt-5.6-terra",
            chat_reasoning_effort="medium",
            allow_remote_conversation_compaction=True,
        )
        _seed_messages(
            runtime.store,
            40,
            hours_old=KEEP_RECENT_HOURS + 24,
            route=_remote_route(
                "openai", "gpt-5.6-terra", "external_provider", consent=True
            ),
        )
        captured: list[dict] = []

        def fake_generate(**kwargs):
            captured.append(kwargs)
            return "- the operator is building Jarvis with explicit remote compaction consent"

        with mock.patch("jarvis_v2.agent.model_provider.generate_model_text", side_effect=fake_generate):
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
            )
        if watermark is None or not captured or "Compacted 40 messages" not in output:
            raise SystemExit(f"opted-in OpenAI compaction did not run: {output!r} {watermark} {captured}")
        if any(call.get("provider") != "openai" or call.get("model") != "gpt-5.6-terra" for call in captured):
            raise SystemExit(f"opted-in compaction missed configured OpenAI route: {captured}")
        transcripts = [str((call.get("messages") or [])[-1].get("content") or "") for call in captured]
        if not all("project jarvis milestone" in transcript for transcript in transcripts):
            raise SystemExit(f"opted-in compaction missed the mocked transcript: {transcripts}")
        memories = runtime.store.recent_memories(limit=5)
        if not memories or "explicit remote compaction consent" not in memories[0]["body"]:
            raise SystemExit(f"opted-in compaction did not persist its digest: {[dict(row) for row in memories]}")


def test_ollama_compaction_runs_with_no_cloud_and_unverified_processing_consent() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-ollama-local-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(
            runtime.config,
            model_provider="ollama",
            chat_model="qwen3:8b",
            allow_remote_conversation_compaction=False,
        )
        _seed_messages(
            runtime.store,
            40,
            hours_old=KEEP_RECENT_HOURS + 24,
            route=_remote_route(
                "ollama", "qwen3:8b", "loopback_daemon_unverified", consent=True
            ),
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "OLLAMA_NO_CLOUD": "1",
                    "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
                },
                clear=True,
            ),
            mock.patch(
                "jarvis_v2.agent.model_provider.resolve_ollama_destination",
                return_value=mock.Mock(allowed=True),
            ) as destination,
            mock.patch(
                "jarvis_v2.agent.model_provider.generate_model_text",
                return_value="- explicitly consented loopback Ollama compaction was allowed",
            ) as model_call,
        ):
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
            )
        if watermark != 40 or "Compacted 40 messages" not in output or not model_call.called:
            raise SystemExit(f"consented loopback Ollama compaction did not run: {output!r} {watermark}")
        destination.assert_called_once_with()


def test_ollama_compaction_uses_stdlib_without_third_party_runtime() -> None:
    from unittest import mock

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size: int = -1) -> bytes:
            payload = json.dumps(
                {
                    "done": True,
                    "done_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "- stdlib-only Ollama compaction was proven",
                    },
                    "prompt_eval_count": 200,
                    "eval_count": 20,
                }
            ).encode("utf-8")
            return payload if size < 0 else payload[:size]

    class FakeOpener:
        def __init__(self) -> None:
            self.calls: list[tuple[object, float]] = []

        def open(self, request, *, timeout: float):
            self.calls.append((request, timeout))
            return FakeResponse()

    with TemporaryDirectory(prefix="jarvis-compaction-ollama-stdlib-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(
            runtime.config,
            model_provider="ollama",
            chat_model="qwen3:8b",
            allow_remote_conversation_compaction=False,
        )
        route = _remote_route(
            "ollama", "qwen3:8b", "loopback_daemon_unverified", consent=True
        )
        _seed_messages(
            runtime.store,
            40,
            hours_old=KEEP_RECENT_HOURS + 24,
            route=route,
        )
        opener = FakeOpener()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "OLLAMA_HOST": "127.0.0.1:11434",
                    "OLLAMA_NO_CLOUD": "1",
                    "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
                },
                clear=True,
            ),
            mock.patch.dict(
                sys.modules,
                {"ollama": None, "httpx": None, "pydantic": None},
            ),
            mock.patch(
                "jarvis_v2.agent.model_provider.build_opener",
                return_value=opener,
            ),
        ):
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
            )
        if watermark != 40 or "Compacted 40 messages" not in output:
            raise SystemExit(f"stdlib Ollama compaction did not commit: {output!r} {watermark}")
        if len(opener.calls) != 1:
            raise SystemExit(f"stdlib Ollama compaction HTTP call count drifted: {opener.calls!r}")
        request, timeout = opener.calls[0]
        payload = json.loads(request.data.decode("utf-8"))
        transcript = "\n".join(
            str(message.get("content") or "")
            for message in payload.get("messages", [])
            if isinstance(message, dict)
        )
        if (
            request.full_url != "http://127.0.0.1:11434/api/chat"
            or timeout != compaction_module.MODEL_TIMEOUT_SECONDS
            or payload.get("stream") is not False
            or "project jarvis milestone" not in transcript
        ):
            raise SystemExit(
                f"stdlib Ollama compaction request drifted: {request.full_url!r} "
                f"{timeout!r} {payload!r}"
            )
        memories = runtime.store.recent_memories(limit=5)
        if not memories or "stdlib-only Ollama compaction was proven" not in memories[0]["body"]:
            raise SystemExit("stdlib Ollama compaction did not persist its bounded digest")


def test_ollama_no_cloud_intent_alone_never_releases_old_history() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-ollama-no-cloud-only-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(
            runtime.config,
            model_provider="ollama",
            chat_model="qwen3:8b",
            allow_remote_conversation_compaction=False,
        )
        _seed_messages(
            runtime.store,
            40,
            hours_old=KEEP_RECENT_HOURS + 24,
            route=_remote_route(
                "ollama", "qwen3:8b", "loopback_daemon_unverified", consent=False
            ),
        )
        with (
            mock.patch.dict(os.environ, {"OLLAMA_NO_CLOUD": "1"}, clear=True),
            mock.patch(
                "jarvis_v2.agent.model_provider.resolve_ollama_destination",
                return_value=mock.Mock(allowed=True),
            ),
            mock.patch(
                "jarvis_v2.agent.model_provider.generate_model_text",
                side_effect=AssertionError("no-cloud intent alone must not receive old conversation markers"),
            ) as model_call,
        ):
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
            )
        for expected in (
            "not approved for this model route",
            "OLLAMA_NO_CLOUD=1",
            "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1",
        ):
            if expected not in output:
                raise SystemExit(f"no-cloud-only skip missed {expected!r}: {output!r}")
        if model_call.called or watermark is not None:
            raise SystemExit(f"no-cloud intent alone crossed the compaction boundary: {output!r} {watermark}")
        if runtime.store.conversation_compaction_watermark() != 0 or _digest_rows(runtime.store):
            raise SystemExit("no-cloud intent alone changed durable compaction state")


def test_remote_compaction_opt_in_cannot_override_ollama_policy_blocks() -> None:
    from unittest import mock

    cases = (
        (
            "qwen3:8b",
            False,
            {
                "OLLAMA_NO_CLOUD": "1",
                "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
            },
            "disallowed destination",
        ),
        (
            "qwen3:8b",
            True,
            {"OLLAMA_NO_CLOUD": "1"},
            "missing unverified processing consent",
        ),
        (
            "kimi-k2-thinking",
            True,
            {
                "OLLAMA_NO_CLOUD": "1",
                "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
            },
            "cloud model alias",
        ),
    )
    for model, destination_allowed, environment, label in cases:
        with TemporaryDirectory(prefix="jarvis-compaction-ollama-policy-block-") as temp:
            runtime = make_temp_runtime(Path(temp))
            config = replace(
                runtime.config,
                model_provider="ollama",
                chat_model=model,
                allow_remote_conversation_compaction=True,
            )
            consent = bool(
                environment.get("JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT")
                == "1"
            )
            destination_class = (
                "loopback_daemon_unverified"
                if destination_allowed
                else "blocked_destination"
            )
            _seed_messages(
                runtime.store,
                40,
                hours_old=KEEP_RECENT_HOURS + 24,
                route=_remote_route(
                    "ollama",
                    model,
                    destination_class,
                    consent=consent,
                ),
            )
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch(
                    "jarvis_v2.agent.model_provider.resolve_ollama_destination",
                    return_value=mock.Mock(allowed=destination_allowed),
                ),
                mock.patch(
                    "jarvis_v2.agent.model_provider.generate_model_text",
                    side_effect=AssertionError("remote compaction opt-in must not bypass central Ollama blocks"),
                ) as model_call,
            ):
                output, watermark = build_conversation_compaction(
                    runtime.store,
                    runtime.vault,
                    config,
                )
            if model_call.called or watermark is not None:
                raise SystemExit(f"{label} was overridden by remote compaction opt-in: {output!r} {watermark}")
            if runtime.store.conversation_compaction_watermark() != 0 or _digest_rows(runtime.store):
                raise SystemExit(f"{label} changed durable compaction state")


def test_invalid_provider_makes_zero_model_calls_and_preserves_watermark() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-invalid-provider-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(
            runtime.config,
            model_provider="not-a-provider",
            allow_remote_conversation_compaction=True,
        )
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        _output, initial_watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            config,
            summarize=lambda text: "- established watermark before invalid-provider retry",
        )
        if initial_watermark != 40:
            raise SystemExit(f"invalid-provider setup did not establish a watermark: {initial_watermark}")
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        digests_before = _digest_rows(runtime.store)
        with mock.patch(
            "jarvis_v2.agent.model_provider.generate_model_text",
            side_effect=AssertionError("invalid provider must be rejected before model dispatch"),
        ) as model_call:
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
                last_compacted_id=initial_watermark,
            )
        if model_call.call_count != 0 or watermark is not None:
            raise SystemExit(f"invalid provider crossed the model boundary: {output!r} {watermark}")
        if "invalid provider" not in output or runtime.store.conversation_compaction_watermark() != 40:
            raise SystemExit(f"invalid provider did not preserve the watermark: {output!r}")
        if _digest_rows(runtime.store) != digests_before:
            raise SystemExit("invalid provider changed the existing digest set")


def test_injected_summarizer_keeps_deferred_route_policy_contract() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-injected-policy-") as temp:
        runtime = make_temp_runtime(Path(temp))
        config = replace(runtime.config, model_provider="not-a-provider")
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        calls: list[str] = []
        with mock.patch(
            "jarvis_v2.automations.compaction._default_compaction_allowed",
            side_effect=AssertionError("injected summarizer must not enter the default route gate"),
        ):
            output, watermark = build_conversation_compaction(
                runtime.store,
                runtime.vault,
                config,
                summarize=lambda text: calls.append(text) or "- injected summary remains deferred",
            )
        if watermark != 40 or len(calls) != 1 or "Compacted 40 messages" not in output:
            raise SystemExit(f"injected summarizer contract changed: {output!r} {watermark} {len(calls)}")


def test_chitchat_batch_advances_watermark_without_memory() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-chitchat-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 30, hours_old=KEEP_RECENT_HOURS + 24)
        output, watermark = build_conversation_compaction(
            runtime.store, runtime.vault, runtime.config,
            summarize=lambda text: "NOTHING_DURABLE",
        )
        if watermark is None:
            raise SystemExit("chit-chat batch must still advance the watermark (never re-summarize)")
        if runtime.store.recent_memories(limit=5):
            raise SystemExit("chit-chat batch must not write a memory row")
        if "nothing durable" not in output:
            raise SystemExit(f"report should say nothing durable was found: {output!r}")


def test_credential_values_are_redacted_from_digest() -> None:
    """A live run proved the model copies generated passwords into digests.
    Any credential-keyword line must have its value tokens masked before the
    digest is persisted."""
    with TemporaryDirectory(prefix="jarvis-compaction-redact-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 30, hours_old=KEEP_RECENT_HOURS + 24)
        leaked = (
            "- User's password generation requests resulted in: `_x,G+(y$yP1D`\n"
            "- api key for glif: sk-9f8e7d6c5b4a\n"
            "- User prefers the password manager workflow for logins\n"
            "- decision: use faster-whisper for voice"
        )
        output, watermark = build_conversation_compaction(
            runtime.store, runtime.vault, runtime.config,
            summarize=lambda text: leaked,
        )
        if watermark is None:
            raise SystemExit(f"redaction test batch should compact: {output!r}")
        body = runtime.store.recent_memories(limit=1)[0]["body"]
        for secret in ("_x,G+(y$yP1D", "sk-9f8e7d6c5b4a"):
            if secret in body:
                raise SystemExit(f"credential value survived into the stored digest: {body!r}")
        if "[redacted]" not in body:
            raise SystemExit(f"masked lines should show the redaction marker: {body!r}")
        if "faster-whisper" not in body:
            raise SystemExit(f"non-credential content must survive redaction: {body!r}")
        if "password manager workflow" not in body:
            raise SystemExit(f"plain words on credential-keyword lines must survive: {body!r}")


def test_scheduler_persists_watermark_and_does_not_deliver() -> None:
    from jarvis_v2.automations import scheduler as scheduler_module
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-sched-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        past = iso(datetime.now() - timedelta(minutes=5))
        runtime.store.upsert_job("Conversation Compaction", 1440, "conversation_compaction", past)

        sent: list[str] = []
        with mock.patch(
            "jarvis_v2.automations.compaction._default_route",
            return_value=_local_route_decision(),
        ), mock.patch(
            "jarvis_v2.automations.compaction._default_summarize",
            return_value=_declared(lambda text: "- durable fact for scheduler test"),
        ), mock.patch.object(
            scheduler_module, "_send_owner_telegram",
            side_effect=lambda text: sent.append(text) or {"ok": True},
        ):
            output = scheduler.run_due_jobs()
        if "Compacted 40 messages" not in output:
            raise SystemExit(f"scheduler should run the compaction job: {output!r}")
        if sent:
            raise SystemExit(f"compaction is a local job and must NOT deliver to Telegram: {sent}")
        import json

        row = [r for r in runtime.store.list_jobs() if r["name"] == "Conversation Compaction"][0]
        metadata = json.loads(row["metadata"] or "{}")
        if int(metadata.get("last_compacted_id") or 0) <= 0:
            raise SystemExit(f"scheduler must persist the watermark in job metadata: {metadata}")


def test_database_failure_rolls_back_memory_and_range() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-rollback-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        with runtime.store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER fail_compaction_batch BEFORE INSERT ON conversation_compaction_batches "
                "BEGIN SELECT RAISE(ABORT, 'synthetic compaction commit failure'); END"
            )
        try:
            build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=lambda text: "- rollback-safe durable fact",
            )
        except Exception:
            pass
        else:
            raise SystemExit("synthetic compaction transaction failure should propagate")
        if _digest_rows(runtime.store) or _compaction_rows(runtime.store):
            raise SystemExit("failed compaction transaction left partial memory/range state")
        if runtime.store.conversation_compaction_watermark() != 0:
            raise SystemExit("failed compaction transaction advanced the global watermark")
        with runtime.store.connect() as conn:
            conn.execute("DROP TRIGGER fail_compaction_batch")
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: "- rollback-safe durable fact",
        )
        if watermark != 40 or "Compacted 40 messages" not in output:
            raise SystemExit(f"compaction retry after rollback did not succeed: {output!r} {watermark}")
        if len(_digest_rows(runtime.store)) != 1 or len(_compaction_rows(runtime.store)) != 1:
            raise SystemExit("compaction retry did not create exactly one digest/range")


def test_scheduler_cache_failure_recovers_without_resummarizing() -> None:
    from jarvis_v2.automations import scheduler as scheduler_module
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-cache-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        runtime.store.upsert_job(
            "Conversation Compaction",
            1440,
            "conversation_compaction",
            iso(datetime.now() - timedelta(minutes=5)),
        )
        original_merge = scheduler._merge_job_metadata
        scheduler._merge_job_metadata = lambda job_id, updates: False  # type: ignore[assignment]
        with mock.patch(
            "jarvis_v2.automations.compaction._default_route",
            return_value=_local_route_decision(),
        ), mock.patch(
            "jarvis_v2.automations.compaction._default_summarize",
            return_value=_declared(lambda text: "- cache-recovery durable fact"),
        ), mock.patch.object(
            scheduler_module,
            "_send_owner_telegram",
            side_effect=AssertionError("compaction must remain local"),
        ):
            first = scheduler.run_due_jobs()
        if "Retry scheduled" not in first:
            raise SystemExit(f"failed watermark cache update should schedule recovery: {first!r}")
        if runtime.store.conversation_compaction_watermark() != 40:
            raise SystemExit("authoritative compaction ledger did not survive cache failure")
        if len(_digest_rows(runtime.store)) != 1 or len(_compaction_rows(runtime.store)) != 1:
            raise SystemExit("cache failure created the wrong durable compaction state")

        scheduler._merge_job_metadata = original_merge  # type: ignore[assignment]
        with mock.patch(
            "jarvis_v2.automations.compaction._default_route",
            return_value=_local_route_decision(),
        ), mock.patch(
            "jarvis_v2.automations.compaction._default_summarize",
            return_value=_declared(
                lambda text: (_ for _ in ()).throw(
                    AssertionError("recovery must not resummarize")
                )
            ),
        ):
            second = scheduler.run_job_now("Conversation Compaction")
        if "Nothing to compact" not in second:
            raise SystemExit(f"cache recovery did not use authoritative ledger state: {second!r}")
        if len(_digest_rows(runtime.store)) != 1 or len(_compaction_rows(runtime.store)) != 1:
            raise SystemExit("cache recovery duplicated the digest or range")
        job = [row for row in runtime.store.list_jobs() if row["name"] == "Conversation Compaction"][0]
        metadata = json.loads(job["metadata"] or "{}")
        if int(metadata.get("last_compacted_id") or 0) != 40:
            raise SystemExit(f"cache recovery did not restore the compatibility watermark: {metadata}")


def test_scheduler_model_failure_uses_retry_path() -> None:
    from unittest import mock

    with TemporaryDirectory(prefix="jarvis-compaction-model-retry-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        runtime.store.upsert_job(
            "Conversation Compaction",
            1440,
            "conversation_compaction",
            iso(datetime.now() - timedelta(minutes=5)),
        )
        before = datetime.now()
        with mock.patch(
            "jarvis_v2.automations.compaction._default_route",
            return_value=_local_route_decision(),
        ), mock.patch(
            "jarvis_v2.automations.compaction._default_summarize",
            return_value=_declared(
                lambda text: (_ for _ in ()).throw(
                    ConnectionError("synthetic model outage")
                )
            ),
        ):
            output = scheduler.run_due_jobs()
        if "Retry scheduled" not in output or "RuntimeError" not in output:
            raise SystemExit(f"compaction model outage did not use scheduler retry path: {output!r}")
        row = [row for row in runtime.store.list_jobs() if row["name"] == "Conversation Compaction"][0]
        retry_at = datetime.fromisoformat(str(row["next_run_at"]))
        if not before < retry_at < before + timedelta(minutes=10):
            raise SystemExit(f"compaction model retry was not scheduled promptly: {row['next_run_at']}")
        if runtime.store.conversation_compaction_watermark() != 0 or _compaction_rows(runtime.store):
            raise SystemExit("model outage advanced durable compaction state")


def test_lost_lease_cannot_commit_compaction() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-lost-lease-") as temp:
        runtime = make_temp_runtime(Path(temp))
        scheduler = Scheduler(runtime.store, runtime.vault, runtime.config)
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        runtime.store.upsert_job(
            "Conversation Compaction",
            1440,
            "conversation_compaction",
            iso(datetime.now() - timedelta(minutes=5)),
        )
        row = [row for row in runtime.store.list_jobs() if row["name"] == "Conversation Compaction"][0]
        claimed = scheduler._claim_job(row)
        if claimed is None:
            raise SystemExit("could not create synthetic compaction lease")
        claimed_row, token = claimed
        scheduler._release_claim(claimed_row, token)
        try:
            build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=lambda text: "- stale owner must not commit",
                job_id=int(claimed_row["id"]),
                lease_token=token,
            )
        except RuntimeError:
            pass
        else:
            raise SystemExit("lost compaction lease should fail before durable commit")
        if _digest_rows(runtime.store) or _compaction_rows(runtime.store):
            raise SystemExit("lost compaction lease wrote durable state")
        if runtime.store.conversation_compaction_watermark() != 0:
            raise SystemExit("lost compaction lease advanced the watermark")


def test_concurrent_workers_commit_one_range_memory_and_mirror() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        barrier = threading.Barrier(2)
        callback_lock = threading.Lock()
        callback_count = 0

        def summarize(_text: str) -> str:
            nonlocal callback_count
            with callback_lock:
                callback_count += 1
            time.sleep(0.1)
            return "- one durable fact from concurrent workers"

        def run(_index: int):
            barrier.wait()
            return build_conversation_compaction(
                runtime.store,
                runtime.vault,
                runtime.config,
                summarize=summarize,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, range(2)))
        if any(watermark != 40 for _output, watermark in results):
            raise SystemExit(f"concurrent compaction workers diverged: {results!r}")
        if callback_count != 1:
            raise SystemExit(f"concurrent compaction disclosed the range {callback_count} times")
        if len(_digest_rows(runtime.store)) != 1 or len(_compaction_rows(runtime.store)) != 1:
            raise SystemExit("concurrent compaction created duplicate range or memory")
        marker = "jarvis-conversation-compaction:v1:1:40"
        if _daily_text(runtime).count(marker) != 1:
            raise SystemExit("concurrent compaction did not produce one idempotent mirror")


def test_chitchat_receipt_prevents_resummarization() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-chitchat-recovery-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 30, hours_old=KEEP_RECENT_HOURS + 24)
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: "NOTHING_DURABLE",
        )
        if watermark != 30 or "nothing durable" not in output:
            raise SystemExit(f"chitchat range did not commit: {output!r} {watermark}")
        rows = _compaction_rows(runtime.store)
        if len(rows) != 1 or rows[0]["outcome"] != "nothing_durable" or rows[0]["memory_id"] is not None:
            raise SystemExit(f"chitchat range receipt was wrong: {rows!r}")
        output2, watermark2 = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            last_compacted_id=0,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("chitchat must not resummarize")),
        )
        if NOTHING_TO_COMPACT not in output2 or watermark2 != 30:
            raise SystemExit(f"chitchat receipt did not recover stale cache: {output2!r} {watermark2}")


def test_pending_mirror_repairs_without_duplicate_after_ambiguous_write() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-mirror-repair-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        original_append = runtime.vault.append_daily_once

        def append_then_fail(*args, **kwargs):
            original_append(*args, **kwargs)
            raise OSError("synthetic post-write ambiguity")

        runtime.vault.append_daily_once = append_then_fail  # type: ignore[assignment]
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: "- mirror repair durable fact",
        )
        if watermark != 40 or "Compacted 40 messages" not in output:
            raise SystemExit(f"ambiguous mirror run did not commit primary state: {output!r} {watermark}")
        rows = _compaction_rows(runtime.store)
        if rows[0]["mirror_state"] != "pending":
            raise SystemExit(f"ambiguous mirror should remain pending: {rows!r}")
        marker = "jarvis-conversation-compaction:v1:1:40"
        if _daily_text(runtime).count(marker) != 1:
            raise SystemExit("ambiguous mirror write did not leave exactly one marker")

        runtime.vault.append_daily_once = original_append  # type: ignore[assignment]
        output2, watermark2 = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            last_compacted_id=0,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("mirror repair must not resummarize")),
        )
        if NOTHING_TO_COMPACT not in output2 or watermark2 != 40:
            raise SystemExit(f"mirror repair did not recover the stale cache: {output2!r} {watermark2}")
        rows = _compaction_rows(runtime.store)
        if rows[0]["mirror_state"] != "completed" or _daily_text(runtime).count(marker) != 1:
            raise SystemExit(f"mirror repair was not idempotent: {rows!r} {_daily_text(runtime)!r}")


def test_legacy_exact_digest_is_adopted_without_new_memory() -> None:
    from jarvis_v2.memory.store import MemoryRecord

    with TemporaryDirectory(prefix="jarvis-compaction-legacy-adopt-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        rows = fetch_compactable_messages(runtime.store, 0)
        span = f"{str(rows[0]['created_at'])[:10]} → {str(rows[-1]['created_at'])[:10]}"
        title = f"Conversation digest {span} (messages #1–#40)"
        for _ in range(2):
            runtime.store.add_memory(
                MemoryRecord(
                    "conversation-digest",
                    title,
                    "- identical legacy digest",
                    "conversation_compaction",
                    0.8,
                )
            )
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("legacy adoption must not summarize")),
        )
        if watermark != 40 or "Recovered legacy" not in output:
            raise SystemExit(f"legacy digest was not adopted: {output!r} {watermark}")
        if len(_digest_rows(runtime.store)) != 2 or len(_compaction_rows(runtime.store)) != 1:
            raise SystemExit("legacy adoption inserted another digest or missed its range receipt")


def test_conflicting_legacy_digests_block_without_advancing() -> None:
    from jarvis_v2.memory.store import MemoryRecord

    with TemporaryDirectory(prefix="jarvis-compaction-legacy-conflict-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        rows = fetch_compactable_messages(runtime.store, 0)
        span = f"{str(rows[0]['created_at'])[:10]} → {str(rows[-1]['created_at'])[:10]}"
        title = f"Conversation digest {span} (messages #1–#40)"
        runtime.store.add_memory(MemoryRecord("conversation-digest", title, "- legacy A", "conversation_compaction", 0.8))
        runtime.store.add_memory(MemoryRecord("conversation-digest", title, "- legacy B", "conversation_compaction", 0.8))
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("legacy conflict must not summarize")),
        )
        if "conflicting legacy digests" not in output or watermark is not None:
            raise SystemExit(f"legacy conflict should block safely: {output!r} {watermark}")
        if runtime.store.conversation_compaction_watermark() != 0 or _compaction_rows(runtime.store):
            raise SystemExit("legacy conflict advanced compaction state")


def test_legacy_adoption_rechecks_reviewed_content_atomically() -> None:
    from jarvis_v2.memory.store import MemoryRecord

    with TemporaryDirectory(prefix="jarvis-compaction-legacy-race-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        rows = fetch_compactable_messages(runtime.store, 0)
        span = f"{str(rows[0]['created_at'])[:10]} → {str(rows[-1]['created_at'])[:10]}"
        title = f"Conversation digest {span} (messages #1–#40)"
        memory_id = runtime.store.add_memory(
            MemoryRecord("conversation-digest", title, "- reviewed legacy body", "conversation_compaction", 0.8)
        )
        original_commit = runtime.store.commit_conversation_compaction

        def edit_then_commit(**kwargs):
            runtime.store.update_memory(
                memory_id,
                "conversation-digest",
                title,
                "- concurrently changed private body SHOULD NOT APPEAR",
                0.8,
            )
            return original_commit(**kwargs)

        runtime.store.commit_conversation_compaction = edit_then_commit
        try:
            try:
                build_conversation_compaction(
                    runtime.store,
                    runtime.vault,
                    runtime.config,
                    summarize=lambda text: (_ for _ in ()).throw(AssertionError("changed legacy row must not summarize")),
                )
            except RuntimeError as exc:
                message = str(exc)
            else:
                raise SystemExit("concurrently edited legacy memory should not be adopted")
        finally:
            runtime.store.commit_conversation_compaction = original_commit
        if message != "conversation compaction legacy memory changed before commit":
            raise SystemExit(f"legacy race returned the wrong bounded diagnostic: {message!r}")
        if "SHOULD NOT APPEAR" in message or "private body" in message:
            raise SystemExit(f"legacy race leaked changed body content: {message!r}")
        if runtime.store.conversation_compaction_watermark() != 0 or _compaction_rows(runtime.store):
            raise SystemExit("concurrently edited legacy memory advanced compaction state")


def test_nonmonotonic_timestamp_stops_at_first_ineligible_id() -> None:
    with TemporaryDirectory(prefix="jarvis-compaction-frontier-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _activate_route(runtime.store, LOCAL_SUMMARIZER_ROUTE)
        recent = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%SZ")
        old = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=KEEP_RECENT_HOURS + 24)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with runtime.store.connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, metadata, created_at) VALUES ('smoke','user','recent frontier','{}',?)",
                (recent,),
            )
            for index in range(25):
                conn.execute(
                    "INSERT INTO messages(session_id, role, content, metadata, created_at) VALUES ('smoke','user',?,'{}',?)",
                    (f"older later id {index}", old),
                )
        rows = fetch_compactable_messages(runtime.store, 0)
        if rows:
            raise SystemExit(f"compaction skipped over an ineligible frontier row: {[row['id'] for row in rows]}")
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: (_ for _ in ()).throw(AssertionError("blocked frontier must not summarize")),
        )
        if NOTHING_TO_COMPACT not in output or watermark is not None:
            raise SystemExit(f"blocked frontier should stay quiet without advancing: {output!r} {watermark}")


def test_legacy_review_is_bounded_content_free_and_read_only() -> None:
    from jarvis_v2.memory.store import MemoryRecord

    with TemporaryDirectory(prefix="jarvis-compaction-legacy-review-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_messages(runtime.store, 40, hours_old=KEEP_RECENT_HOURS + 24)
        output, watermark = build_conversation_compaction(
            runtime.store,
            runtime.vault,
            runtime.config,
            summarize=lambda text: "- linked private digest SHOULD NOT APPEAR",
        )
        if watermark != 40 or "Compacted" not in output:
            raise SystemExit(f"legacy review fixture did not create linked coverage: {output!r} {watermark}")
        linked = _digest_rows(runtime.store)[0]
        duplicate_id = runtime.store.add_memory(
            MemoryRecord(
                "conversation-digest",
                linked["title"],
                linked["body"],
                "conversation_compaction",
                0.8,
            )
        )
        conflict_title = "Conversation digest 2026-01-01 → 2026-01-02 (messages #41–#80)"
        conflict_ids = [
            runtime.store.add_memory(
                MemoryRecord(
                    "conversation-digest",
                    conflict_title,
                    f"- conflict private body {index} SHOULD NOT APPEAR",
                    "conversation_compaction",
                    0.8,
                )
            )
            for index in range(2)
        ]
        runtime.store.add_memory(
            MemoryRecord(
                "conversation-digest",
                "Conversation digest 2026-01-03 → 2026-01-04 (messages #81–#120)",
                "- canonical pending private body SHOULD NOT APPEAR",
                "conversation_compaction",
                0.8,
            )
        )
        malformed_id = runtime.store.add_memory(
            MemoryRecord(
                "conversation-digest",
                "Malformed private digest /\x55sers/operator SHOULD NOT APPEAR",
                "- malformed private body SHOULD NOT APPEAR",
                "conversation_compaction",
                0.8,
            )
        )
        before = (_digest_rows(runtime.store), _compaction_rows(runtime.store), runtime.store.conversation_compaction_watermark())
        review = runtime.store.conversation_compaction_legacy_review(limit=20)
        after = (_digest_rows(runtime.store), _compaction_rows(runtime.store), runtime.store.conversation_compaction_watermark())
        if before != after:
            raise SystemExit("legacy compaction review mutated memory, batch, or watermark state")
        expected = {
            "total_digest_rows": 6,
            "linked_digest_rows": 1,
            "unlinked_legacy_rows": 5,
            "canonical_unlinked_groups": 3,
            "preserved_duplicate_rows": 1,
            "conflicting_groups": 1,
            "unparseable_rows": 1,
            "review_required": True,
            "review_memory_ids": sorted([*conflict_ids, malformed_id]),
            "review_memory_ids_truncated": False,
            "watermark": 40,
        }
        if review != expected:
            raise SystemExit(f"legacy compaction review classification drifted: {review!r}")
        if duplicate_id in review["review_memory_ids"]:
            raise SystemExit(f"preserved identical duplicate was incorrectly flagged for repair: {review!r}")
        serialized = json.dumps(review, sort_keys=True)
        for forbidden in ("SHOULD NOT APPEAR", "/\x55sers/", "private body", conflict_title, linked["title"]):
            if forbidden in serialized:
                raise SystemExit(f"legacy compaction review leaked content {forbidden!r}: {serialized}")
        truncated = runtime.store.conversation_compaction_legacy_review(limit=1)
        if truncated["review_memory_ids"] != expected["review_memory_ids"][:1] or truncated["review_memory_ids_truncated"] is not True:
            raise SystemExit(f"legacy compaction review limit was not truthful: {truncated!r}")
        boolean_limit = runtime.store.conversation_compaction_legacy_review(limit=True)
        if boolean_limit["review_memory_ids"] != expected["review_memory_ids"]:
            raise SystemExit(f"boolean legacy review limit should use the bounded default: {boolean_limit!r}")


def main() -> None:
    test_small_backlog_is_quiet_and_keeps_watermark()
    test_recent_messages_are_never_compacted()
    test_batch_compacts_into_memory_and_advances_watermark()
    test_model_failure_keeps_watermark_and_writes_nothing()
    test_openai_compaction_is_blocked_without_separate_opt_in()
    test_openai_compaction_runs_only_after_explicit_opt_in()
    test_ollama_compaction_runs_with_no_cloud_and_unverified_processing_consent()
    test_ollama_compaction_uses_stdlib_without_third_party_runtime()
    test_ollama_no_cloud_intent_alone_never_releases_old_history()
    test_remote_compaction_opt_in_cannot_override_ollama_policy_blocks()
    test_invalid_provider_makes_zero_model_calls_and_preserves_watermark()
    test_injected_summarizer_keeps_deferred_route_policy_contract()
    test_chitchat_batch_advances_watermark_without_memory()
    test_credential_values_are_redacted_from_digest()
    test_scheduler_persists_watermark_and_does_not_deliver()
    test_database_failure_rolls_back_memory_and_range()
    test_scheduler_cache_failure_recovers_without_resummarizing()
    test_scheduler_model_failure_uses_retry_path()
    test_lost_lease_cannot_commit_compaction()
    test_concurrent_workers_commit_one_range_memory_and_mirror()
    test_chitchat_receipt_prevents_resummarization()
    test_pending_mirror_repairs_without_duplicate_after_ambiguous_write()
    test_legacy_exact_digest_is_adopted_without_new_memory()
    test_conflicting_legacy_digests_block_without_advancing()
    test_legacy_adoption_rechecks_reviewed_content_atomically()
    test_nonmonotonic_timestamp_stops_at_first_ineligible_id()
    test_legacy_review_is_bounded_content_free_and_read_only()
    print("Conversation compaction smoke passed")


if __name__ == "__main__":
    main()
