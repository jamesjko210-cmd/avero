from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import jarvis_v2.memory.store as store_module
from jarvis_v2.agent.chat import ChatBrain
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.memory.store import MemoryRecord, MemoryStore, PreferenceRecord, SkillRecord


PRIVATE_MARKER = "RUNTIME_PRIVATE_HISTORY_91c64b"
PROFILE_MARKER = "RUNTIME_STORED_PROFILE_331ad1"
PREFERENCE_MARKER = "RUNTIME_STORED_PREFERENCE_7bc9e2"
MEMORY_MARKER = "RUNTIME_STORED_MEMORY_f59a04"
SKILL_MARKER = "RUNTIME_STORED_SKILL_b78d20"
STORED_MARKERS = (PROFILE_MARKER, PREFERENCE_MARKER, MEMORY_MARKER, SKILL_MARKER)


def _rotate_epoch_process(db_path, start, attempted, completed, results) -> None:
    store = MemoryStore(Path(db_path))
    if not start.wait(5):
        results.put(("start_timeout", 0.0, ""))
        return
    attempted.set()
    started = time.monotonic()
    try:
        epoch_id = store.start_history_policy_epoch(
            policy_fingerprint="d" * 64,
            provider="runtime",
            model_identifier="rotated",
            destination_class="local-only",
            session_generation=999,
            explicit_consent_satisfied=False,
        )
    except Exception as exc:
        results.put((type(exc).__name__, time.monotonic() - started, ""))
    else:
        results.put(("rotated", time.monotonic() - started, epoch_id))
    finally:
        completed.set()


def _crash_fence_owner_process(db_path, acquired, crash) -> None:
    store = MemoryStore(Path(db_path))
    with store.history_authority_fence():
        acquired.set()
        if crash.wait(5):
            os._exit(23)


def _fork_fence_contender_process(db_path, attempted, acquired) -> None:
    store = MemoryStore(Path(db_path))
    attempted.set()
    with store.history_authority_fence():
        acquired.set()


def _config(root: Path, *, db_path: Path | None = None) -> JarvisConfig:
    return JarvisConfig(
        data_dir=root,
        db_path=db_path or root / "jarvis.sqlite",
        obsidian_vault=root / "Vault",
        obsidian_root="Jarvis",
        chat_model="runtime-history-mock-model",
        model_provider="openai",
        allow_remote_personal_context=True,
        use_model_planner=False,
    )


def _messages(runtime: JarvisRuntime) -> list[dict[str, object]]:
    with runtime.store.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM messages ORDER BY id")]


def _provider_text(call: object) -> str:
    kwargs = getattr(call, "kwargs", {})
    return "\n".join(str(item.get("content", "")) for item in kwargs.get("messages", []))


def _chat(runtime: JarvisRuntime, prompt: str, answer: str):
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text", return_value=answer
    ):
        return runtime.handle(prompt)


def _history_callbacks(brain: ChatBrain):
    callbacks = brain._history_disclosure_callbacks
    if callbacks is None:
        raise AssertionError("ChatBrain did not publish a history disclosure callback bundle")
    return callbacks


@contextmanager
def _storage_fallback_env(fallback_root: Path):
    with patch.dict(os.environ, {"JARVIS_STORAGE_FALLBACK_DIR": str(fallback_root)}):
        for key in ("JARVIS_DB_PATH", "JARVIS_DATA_DIR", "JARVIS_DISABLE_STORAGE_FALLBACK"):
            os.environ.pop(key, None)
        yield


def _seed_stored_context(runtime: JarvisRuntime) -> None:
    runtime.vault.append_profile("Runtime provenance", PROFILE_MARKER)
    runtime.store.set_preference(
        PreferenceRecord(
            key="runtime_provenance",
            value=PREFERENCE_MARKER,
            category="architecture",
        )
    )
    runtime.store.add_memory(
        MemoryRecord(
            category="architecture",
            title="Runtime provenance",
            body=MEMORY_MARKER,
        )
    )
    skill_id = runtime.store.save_skill(
        SkillRecord(
            name="Runtime provenance procedure",
            trigger="discuss architecture",
            body=f"Use the runtime provenance procedure. {SKILL_MARKER}",
            tags="reviewed,architecture",
        )
    )
    row = runtime.store.get_skill_by_id(skill_id)
    if row is None or row["origin"] != "user_authored":
        raise AssertionError(f"stored skill did not receive durable user custody: {row}")


def test_normal_tools_suggestions_and_receipts(root: Path) -> None:
    runtime = JarvisRuntime(_config(root))
    if type(runtime.history_session_generation) is not int or runtime.history_session_generation <= 0:
        raise AssertionError("runtime history generation is not an opaque positive integer")
    first_epoch = runtime.store.get_active_history_policy_epoch()
    if first_epoch["epoch_id"] != runtime._history_policy_epoch_id:
        raise AssertionError("runtime did not retain its fresh durable policy epoch")

    first = _chat(runtime, "Discuss runtime provenance", PRIVATE_MARKER)
    second = _chat(runtime, "Continue the runtime provenance discussion", "confirmed reply")
    if first.response != PRIVATE_MARKER or second.response != "confirmed reply":
        raise AssertionError("mocked normal chat response drifted")
    messages = _messages(runtime)
    if len(messages) != 4:
        raise AssertionError(f"normal chat message count drifted: {messages}")
    for row in messages:
        provenance = runtime.store.read_message_provenance(int(row["id"]))
        if provenance is None or provenance["provenance_state"] != "recorded":
            raise AssertionError(f"normal chat lost provenance: {provenance}")
        if not provenance["remote_eligible"]:
            raise AssertionError(f"consented current-epoch message lost eligibility: {provenance}")
    assistant = runtime.store.read_message_provenance(int(messages[-1]["id"]))
    if assistant["lineage_state"] != "personal_derived" or assistant["source_count"] != 1:
        raise AssertionError(f"history-derived assistant lineage drifted: {assistant}")

    with runtime.store.connect() as conn:
        confirmed = [dict(row) for row in conn.execute("SELECT * FROM history_disclosure_receipts")]
    if (
        len(confirmed) != 2
        or any(row["state"] != "confirmed" for row in confirmed)
        or not any(row["source_count"] == 1 for row in confirmed)
        or not any(row["source_count"] > 1 for row in confirmed)
        or len({row["source_digest"] for row in confirmed}) != 2
    ):
        raise AssertionError(f"prepared receipt did not confirm: {confirmed}")
    decision = runtime.chat._history_policy_decision(commit=False)
    callbacks = _history_callbacks(runtime.chat)
    blocked_handle = callbacks.prepare(
        {
            "session_generation": runtime.chat._session_generation,
            "policy_epoch_id": decision.epoch_id,
            "policy_fingerprint": decision.fingerprint,
            "provider": decision.provider,
            "destination_class": decision.destination_class,
            "source_count": 1,
            "source_digest": "d" * 64,
        }
    )
    callbacks.finalize(blocked_handle, "blocked")
    with runtime.store.connect() as conn:
        blocked_state = conn.execute(
            "SELECT state FROM history_disclosure_receipts "
            "WHERE source_digest = ?",
            ("d" * 64,),
        ).fetchone()[0]
    if blocked_state != "blocked":
        raise AssertionError(f"blocked receipt finalization drifted: {blocked_state}")

    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None):
        tool = runtime.handle("capability count")
    if tool.metadata.get("runtime_route") != "tools":
        raise AssertionError(f"tool route drifted: {tool.metadata}")
    tool_rows = _messages(runtime)[-2:]
    if any(
        runtime.store.read_message_provenance(int(row["id"]))["provenance_state"]
        != "legacy_unknown"
        for row in tool_rows
    ):
        raise AssertionError("tool route acquired history provenance")
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None):
        calculation = runtime.handle("calculate 7 + 8")
    if calculation.metadata.get("runtime_route") != "tools" or calculation.response != "7 + 8 = 15":
        raise AssertionError(f"read-only continuity fixture drifted: {calculation}")
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text",
        return_value="tool follow-up grounded",
    ) as follow_up_model:
        follow_up = runtime.handle("What did that tool result report?")
    if follow_up.response != "tool follow-up grounded":
        raise AssertionError(f"tool follow-up response drifted: {follow_up.response!r}")
    follow_up_payload = _provider_text(follow_up_model.call_args)
    if calculation.response not in follow_up_payload:
        raise AssertionError("verified read-only tool output was absent from the next chat request")

    with patch(
        "jarvis_v2.agent.runtime.suggest_command", return_value="Try `safe command`."
    ), patch(
        "jarvis_v2.agent.chat.generate_model_text",
        side_effect=AssertionError("command suggestion called a provider"),
    ):
        suggestion = runtime.handle("A deliberately ambiguous request")
    if suggestion.metadata.get("chat_response", {}).get("runtime_route") != "command_suggestion":
        raise AssertionError(f"command suggestion route drifted: {suggestion.metadata}")
    suggestion_rows = _messages(runtime)[-2:]
    if any(
        runtime.store.read_message_provenance(int(row["id"]))["provenance_state"]
        != "legacy_unknown"
        for row in suggestion_rows
    ):
        raise AssertionError("command suggestion acquired history provenance")

    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text", side_effect=RuntimeError("mock provider failed")
    ):
        runtime.handle("Continue after the mocked provider failure")
    with runtime.store.connect() as conn:
        states = [row[0] for row in conn.execute(
            "SELECT state FROM history_disclosure_receipts ORDER BY prepared_at"
        )]
    if states[-1] != "uncertain" or "confirmed" not in states:
        raise AssertionError(f"receipt final-state mapping drifted: {states}")


def test_prepare_failure_strips_history(root: Path) -> None:
    runtime = JarvisRuntime(_config(root))
    _chat(runtime, "Seed private runtime history", PRIVATE_MARKER)
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch.object(
        runtime.store,
        "prepare_history_disclosure_receipt",
        side_effect=sqlite3.OperationalError("mock receipt database failure"),
    ), patch("jarvis_v2.agent.chat.generate_model_text", return_value="stripped") as model:
        result = runtime.handle("Continue after receipt failure")
    payload = _provider_text(model.call_args)
    if PRIVATE_MARKER in payload:
        raise AssertionError(f"failed receipt prepare disclosed private history: {payload}")
    chat = result.metadata.get("chat_response", {})
    if chat.get("history_disclosure_prepare_status") != "failed_stripped":
        raise AssertionError(f"prepare failure was not reported as stripped: {chat}")
    if chat.get("history_in_model_request") or chat.get("stored_personal_context_in_model_request"):
        raise AssertionError(f"prepare failure retained private context metadata: {chat}")


def test_malformed_and_attachment_failure_stay_local(root: Path) -> None:
    malformed = JarvisRuntime(_config(root / "malformed"))

    def malformed_response(_prompt: str) -> str:
        malformed.chat.last_turn_metadata = {
            "source": "model",
            "history_message_provenance": {"user": {}, "assistant": {}},
        }
        return "malformed provenance reply"

    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch.object(
        malformed.chat, "respond", side_effect=malformed_response
    ):
        malformed.handle("Malformed provenance producer")
    if any(
        malformed.store.read_message_provenance(int(row["id"]))["provenance_state"]
        != "legacy_unknown"
        for row in _messages(malformed)
    ):
        raise AssertionError("malformed producer metadata acquired provenance")

    failed_attach = JarvisRuntime(_config(root / "attachment"))
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text", return_value="successful model reply"
    ), patch.object(
        failed_attach.store,
        "attach_message_provenance",
        side_effect=RuntimeError("PRIVATE /tmp/path must not escape"),
    ):
        result = failed_attach.handle("Exercise attachment failure")
    rows = _messages(failed_attach)
    user = failed_attach.store.read_message_provenance(int(rows[0]["id"]))
    assistant = failed_attach.store.read_message_provenance(int(rows[1]["id"]))
    if user["provenance_state"] != "legacy_unknown" or assistant["remote_eligible"]:
        raise AssertionError(f"attachment failure did not stay local-only: {user} / {assistant}")
    warning = result.metadata.get("history_provenance_warning", {})
    if result.response != "successful model reply" or warning.get("phase") != "user_attachment":
        raise AssertionError(f"attachment failure changed the successful result contract: {result}")
    if "PRIVATE" in repr(warning) or "/tmp" in repr(warning):
        raise AssertionError(f"provenance warning exposed exception detail: {warning}")


def test_cross_runtime_staleness_and_rotation(root: Path) -> None:
    shared_db = root / "shared.sqlite"
    first = JarvisRuntime(_config(root / "first", db_path=shared_db))
    _chat(first, "Seed first runtime private history", PRIVATE_MARKER)
    first_epoch = first._history_policy_epoch_id
    second = JarvisRuntime(_config(root / "second", db_path=shared_db))
    second_epoch = second._history_policy_epoch_id
    if first_epoch == second_epoch or second.store.active_history_policy_epoch_id() != second_epoch:
        raise AssertionError("new runtime did not supersede the first runtime")

    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text", return_value="stale stripped"
    ) as model:
        stale_result = first.handle("Continue from first runtime history")
    if PRIVATE_MARKER in _provider_text(model.call_args):
        raise AssertionError("stale runtime made a private-history provider call")
    stale_rows = [
        row for row in _messages(first) if row["session_id"] == first.session_id
    ][-2:]
    if any(
        first.store.read_message_provenance(int(row["id"]))["remote_eligible"]
        for row in stale_rows
    ):
        raise AssertionError("stale runtime stamped remote-eligible provenance")
    if stale_result.metadata.get("chat_response", {}).get(
        "history_disclosure_prepare_status"
    ) != "failed_stripped":
        raise AssertionError("stale runtime did not fail its prepare callback closed")

    first.chat.model = "stale-runtime-must-not-rotate"
    _chat(first, "Attempt stale policy rotation", "still stale")
    if first.store.active_history_policy_epoch_id() != second_epoch:
        raise AssertionError("stale runtime rotated itself back into authority")

    old_second_epoch = second._history_policy_epoch_id
    second.chat.model = "runtime-history-rotated-model"
    _chat(second, "Rotate the active runtime policy", "rotated")
    if second._history_policy_epoch_id == old_second_epoch:
        raise AssertionError("active runtime policy change did not rotate its epoch")
    active = second.store.get_active_history_policy_epoch()
    if active["model_identifier"] != "runtime-history-rotated-model":
        raise AssertionError(f"rotated epoch has the wrong policy: {active}")


def test_stale_runtime_with_empty_history_strips_all_stored_context(root: Path) -> None:
    shared_db = root / "shared.sqlite"
    stale = JarvisRuntime(_config(root / "stale", db_path=shared_db))
    _seed_stored_context(stale)
    if stale.chat.history:
        raise AssertionError("stored-context stale fixture unexpectedly has chat history")
    active = JarvisRuntime(_config(root / "active", db_path=shared_db))
    if active._history_policy_epoch_id == stale._history_policy_epoch_id:
        raise AssertionError("cross-runtime fixture did not rotate the durable epoch")

    prompt = "Discuss architecture conceptually"
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text", return_value="current-only"
    ) as model:
        result = stale.handle(prompt)
    if model.call_count != 1:
        raise AssertionError("stale runtime did not preserve current-message-only chat")
    messages = model.call_args.kwargs.get("messages")
    if not isinstance(messages, list) or any(set(message) != {"role", "content"} for message in messages):
        raise AssertionError(f"provider payload shape drifted: {messages}")
    payload = _provider_text(model.call_args)
    if prompt not in payload or any(marker in payload for marker in STORED_MARKERS):
        raise AssertionError(f"stale empty-history runtime disclosed stored context: {payload}")
    chat = result.metadata.get("chat_response", {})
    if (
        chat.get("history_disclosure_prepare_status") != "failed_stripped"
        or chat.get("stored_personal_context_in_model_request") is not False
    ):
        raise AssertionError(f"stale empty-history stripping metadata drifted: {chat}")


def test_epoch_rotation_between_prepare_and_provider_blocks_egress(root: Path) -> None:
    shared_db = root / "shared.sqlite"
    runtime = JarvisRuntime(_config(root / "prepared", db_path=shared_db))
    _seed_stored_context(runtime)
    callbacks = _history_callbacks(runtime.chat)
    original_validate = callbacks.validate
    rotated: list[JarvisRuntime] = []

    def rotate_then_validate(handle: object) -> bool:
        if not rotated:
            rotated.append(JarvisRuntime(_config(root / "rotator", db_path=shared_db)))
        return original_validate(handle)

    runtime.chat.install_history_disclosure_callbacks(
        prepare=callbacks.prepare,
        validate=rotate_then_validate,
        finalize=callbacks.finalize,
        fence=callbacks.fence,
    )
    with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
        "jarvis_v2.agent.chat.generate_model_text",
        side_effect=AssertionError("stale prepared context reached provider invocation"),
    ) as model:
        result = runtime.handle("Discuss architecture before provider rotation")
    if model.call_count:
        raise AssertionError("epoch rotation after prepare reached provider invocation")
    with runtime.store.connect() as conn:
        receipts = [dict(row) for row in conn.execute(
            "SELECT * FROM history_disclosure_receipts WHERE session_id = ?",
            (runtime.session_id,),
        )]
    if len(receipts) != 1 or receipts[0]["state"] != "blocked":
        raise AssertionError(f"pre-invocation rotation receipt drifted: {receipts}")
    if any(marker in json.dumps(receipts, sort_keys=True) for marker in STORED_MARKERS):
        raise AssertionError("pre-invocation receipt retained stored context content")
    chat = result.metadata.get("chat_response", {})
    if chat.get("history_disclosure_finalize_status") != "blocked":
        raise AssertionError(f"pre-invocation rotation was not blocked: {chat}")


def _run_chat_rotation_at_final_validation(
    root: Path,
    *,
    fence_enabled: bool,
) -> tuple[bool, dict[str, object], list[dict[str, object]], str]:
    shared_db = root / "shared.sqlite"
    runtime = JarvisRuntime(_config(root / "sender", db_path=shared_db))
    _seed_stored_context(runtime)
    callbacks = _history_callbacks(runtime.chat)
    original_validate = callbacks.validate
    original_fence = callbacks.fence
    original_finalize = callbacks.finalize
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    attempted = ctx.Event()
    completed = ctx.Event()
    results = ctx.Queue()
    process = ctx.Process(
        target=_rotate_epoch_process,
        args=(str(shared_db), start, attempted, completed, results),
    )
    process.start()
    validate_calls = 0
    final_validation_succeeded = threading.Event()
    durable_finalization_completed = threading.Event()
    rotation_completed_before_invocation = False

    def synchronized_validate(handle: object) -> bool:
        nonlocal validate_calls
        active = original_validate(handle)
        validate_calls += 1
        if active and validate_calls == 3:
            final_validation_succeeded.set()
            start.set()
            if not attempted.wait(5):
                raise AssertionError("rotation process did not attempt at final Chat validation")
            if fence_enabled:
                if completed.wait(0.2):
                    raise AssertionError("rotation completed after validation while egress held the fence")
            elif not completed.wait(5):
                raise AssertionError("fence-disabled rotation did not complete before invocation")
        return active

    def synchronized_finalize(handle: object, outcome: str) -> str | None:
        if fence_enabled and completed.is_set():
            raise AssertionError("rotation completed before durable Chat finalization")
        finalized = original_finalize(handle, outcome)
        durable_finalization_completed.set()
        if fence_enabled and completed.is_set():
            raise AssertionError("rotation completed during durable Chat finalization")
        return finalized

    def invoke_at_boundary(**_kwargs) -> str:
        nonlocal rotation_completed_before_invocation
        if not final_validation_succeeded.is_set() or not attempted.is_set():
            raise AssertionError("provider ran before the exact post-validation race boundary")
        rotation_completed_before_invocation = completed.is_set()
        if fence_enabled and rotation_completed_before_invocation:
            raise AssertionError("rotation completed before fenced provider invocation")
        if not fence_enabled and not rotation_completed_before_invocation:
            raise AssertionError("fence-disabled negative control did not expose the race")
        return "provider returned at exact validation boundary"

    runtime.chat.install_history_disclosure_callbacks(
        prepare=callbacks.prepare,
        validate=synchronized_validate,
        finalize=synchronized_finalize,
        fence=original_fence if fence_enabled else lambda _handle: nullcontext(),
    )
    try:
        with patch("jarvis_v2.agent.runtime.suggest_command", return_value=None), patch(
            "jarvis_v2.agent.chat.generate_model_text",
            side_effect=invoke_at_boundary,
        ):
            result = runtime.handle("Discuss architecture during provider rotation")
    finally:
        runtime.chat.install_history_disclosure_callbacks(
            prepare=callbacks.prepare,
            validate=original_validate,
            finalize=original_finalize,
            fence=original_fence,
        )
    process.join(5)
    if process.is_alive():
        process.terminate()
        process.join(2)
        raise AssertionError("chat epoch rotation deadlocked after egress released")
    status, elapsed, rotated_epoch = results.get(timeout=2)
    if process.exitcode != 0 or status != "rotated":
        raise AssertionError(
            f"chat rotation did not wait for egress: exit={process.exitcode} result="
            f"{(status, elapsed, rotated_epoch)!r}"
        )
    with runtime.store.connect() as conn:
        receipts = [dict(row) for row in conn.execute(
            "SELECT * FROM history_disclosure_receipts WHERE session_id = ?",
            (runtime.session_id,),
        )]
    if validate_calls != 3 or not final_validation_succeeded.is_set():
        raise AssertionError(f"Chat final validation boundary count drifted: {validate_calls}")
    if fence_enabled and not durable_finalization_completed.is_set():
        raise AssertionError("fenced Chat disclosure did not durably finalize before release")
    return rotation_completed_before_invocation, result.metadata, receipts, rotated_epoch


def test_cross_process_rotation_waits_for_chat_invocation_and_finalization(root: Path) -> None:
    raced, metadata, receipts, rotated_epoch = _run_chat_rotation_at_final_validation(
        root / "fenced",
        fence_enabled=True,
    )
    if raced:
        raise AssertionError("fenced rotation completed before provider invocation")
    if len(receipts) != 1 or receipts[0]["state"] != "confirmed":
        raise AssertionError(f"fenced chat disclosure receipt was not confirmed: {receipts}")
    if any(marker in json.dumps(receipts, sort_keys=True) for marker in STORED_MARKERS):
        raise AssertionError("fenced chat receipt retained stored context content")
    chat = metadata.get("chat_response", {})
    if (
        chat.get("history_disclosure_finalize_status") != "confirmed"
        or chat.get("model_error")
    ):
        raise AssertionError(f"fenced chat receipt metadata drifted: {chat}")
    shared_db = root / "fenced" / "shared.sqlite"
    store = MemoryStore(shared_db)
    if store.active_history_policy_epoch_id() != rotated_epoch:
        raise AssertionError("chat rotation did not become active after invocation and finalization")
    lock_path = Path(f"{shared_db.resolve()}.history-authority.lock")
    if lock_path.stat().st_mode & 0o777 != 0o600:
        raise AssertionError("history authority sidecar permissions are not owner-only")

    raced_without_fence, _metadata, _receipts, _epoch = (
        _run_chat_rotation_at_final_validation(
            root / "negative-control",
            fence_enabled=False,
        )
    )
    if not raced_without_fence:
        raise AssertionError("removing the Chat fence did not make the race assertion fail")


def test_crashed_fence_owner_releases_os_lock(root: Path) -> None:
    db_path = root / "shared.sqlite"
    store = MemoryStore(db_path)
    store.init()
    ctx = mp.get_context("spawn")
    acquired = ctx.Event()
    crash = ctx.Event()
    holder = ctx.Process(
        target=_crash_fence_owner_process,
        args=(str(db_path), acquired, crash),
    )
    holder.start()
    if not acquired.wait(5):
        holder.terminate()
        holder.join(2)
        raise AssertionError("crash fixture did not acquire the authority fence")

    start = ctx.Event()
    attempted = ctx.Event()
    completed = ctx.Event()
    results = ctx.Queue()
    rotator = ctx.Process(
        target=_rotate_epoch_process,
        args=(str(db_path), start, attempted, completed, results),
    )
    rotator.start()
    start.set()
    if not attempted.wait(5):
        raise AssertionError("crash fixture rotator did not reach the authority fence")
    time.sleep(0.2)
    if completed.is_set():
        raise AssertionError("rotation bypassed the live crash-owner fence")
    crash.set()
    holder.join(5)
    rotator.join(5)
    if holder.is_alive() or rotator.is_alive():
        holder.terminate()
        rotator.terminate()
        holder.join(2)
        rotator.join(2)
        raise AssertionError("OS lock ownership did not release after the holder crashed")
    status, elapsed, _epoch_id = results.get(timeout=2)
    if holder.exitcode != 23 or rotator.exitcode != 0 or status != "rotated" or elapsed < 0.15:
        raise AssertionError(
            f"crash release result drifted: holder={holder.exitcode} rotator={rotator.exitcode} "
            f"result={(status, elapsed)!r}"
        )


def test_history_authority_thread_reentrancy_and_outer_release(root: Path) -> None:
    store = MemoryStore(root / "shared.sqlite")
    store.init()
    attempted = threading.Event()
    acquired = threading.Event()
    errors: list[BaseException] = []

    def contend() -> None:
        attempted.set()
        try:
            with MemoryStore(store.db_path).history_authority_fence():
                acquired.set()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=contend, name="history-fence-contender")
    with store.history_authority_fence():
        with store.history_authority_fence():
            pass
        thread.start()
        if not attempted.wait(2):
            raise AssertionError("thread contender did not attempt the authority fence")
        if acquired.wait(0.2):
            raise AssertionError("nested fence exit released the outer thread authority fence")
    thread.join(2)
    if thread.is_alive() or errors or not acquired.is_set():
        raise AssertionError(f"thread reentrant fence did not release cleanly: {errors!r}")


def test_history_authority_fork_cleanup(root: Path) -> None:
    if "fork" not in mp.get_all_start_methods():
        return
    store = MemoryStore(root / "shared.sqlite")
    store.init()
    ctx = mp.get_context("fork")
    attempted = ctx.Event()
    acquired = ctx.Event()
    process = ctx.Process(
        target=_fork_fence_contender_process,
        args=(str(store.db_path), attempted, acquired),
    )
    with store.history_authority_fence():
        process.start()
        if not attempted.wait(2):
            raise AssertionError("fork child did not attempt the authority fence")
        if acquired.wait(0.2):
            raise AssertionError("fork child inherited reentrant authority ownership")
    process.join(3)
    if process.is_alive():
        process.terminate()
        process.join(2)
        raise AssertionError("fork child did not acquire after the parent released authority")
    if process.exitcode != 0 or not acquired.is_set():
        raise AssertionError(f"fork cleanup failed: exit={process.exitcode}")


def test_history_authority_fence_timeout(root: Path) -> None:
    store = MemoryStore(root / "shared.sqlite")
    store.init()
    holder_acquired = threading.Event()
    release_holder = threading.Event()

    def hold() -> None:
        with store.history_authority_fence():
            holder_acquired.set()
            if not release_holder.wait(3):
                raise AssertionError("timeout fixture holder was not released")

    holder = threading.Thread(target=hold, name="history-fence-timeout-holder")
    holder.start()
    if not holder_acquired.wait(2):
        raise AssertionError("timeout fixture did not acquire the authority fence")
    original_timeout = store_module.HISTORY_AUTHORITY_FENCE_TIMEOUT_SECONDS
    store_module.HISTORY_AUTHORITY_FENCE_TIMEOUT_SECONDS = 0.15
    started = time.monotonic()
    try:
        try:
            with MemoryStore(store.db_path).history_authority_fence():
                raise AssertionError("contended authority fence unexpectedly acquired")
        except TimeoutError:
            pass
    finally:
        elapsed = time.monotonic() - started
        store_module.HISTORY_AUTHORITY_FENCE_TIMEOUT_SECONDS = original_timeout
        release_holder.set()
        holder.join(2)
    if holder.is_alive() or elapsed < 0.1:
        raise AssertionError(f"authority fence timeout did not wait predictably: {elapsed:.3f}s")


def test_storage_fallback_during_atomic_bundle_capture_strips_egress(root: Path) -> None:
    runtime = JarvisRuntime(_config(root / "primary"))
    _seed_stored_context(runtime)
    old_chat = runtime.chat
    old_store = runtime.store
    _history_callbacks(old_chat)
    fallback_done = threading.Event()
    fallback_results: list[object] = []
    capture_interleaved = False

    def activate_fallback() -> None:
        try:
            fallback_results.append(
                runtime._activate_default_storage_fallback(
                    sqlite3.OperationalError("attempt to write a readonly database")
                )
            )
        except BaseException as exc:
            fallback_results.append(exc)
        finally:
            fallback_done.set()

    fallback_thread = threading.Thread(target=activate_fallback, name="atomic-bundle-fallback")
    original_getattribute = ChatBrain.__getattribute__

    def interleave_at_bundle_read(instance: ChatBrain, name: str):
        nonlocal capture_interleaved
        value = original_getattribute(instance, name)
        if (
            instance is old_chat
            and name == "_history_disclosure_callbacks"
            and not capture_interleaved
        ):
            capture_interleaved = True
            fallback_thread.start()
            if not fallback_done.wait(5):
                raise AssertionError("storage fallback did not complete at atomic bundle capture")
        return value

    fallback_root = root / "fallback"
    prompt = "Discuss architecture during atomic callback publication"
    with _storage_fallback_env(fallback_root), patch.object(
        ChatBrain,
        "__getattribute__",
        new=interleave_at_bundle_read,
    ), patch(
        "jarvis_v2.agent.chat.generate_model_text",
        return_value="current-message-only after atomic capture",
    ) as model:
        answer = old_chat.respond(prompt)
    fallback_thread.join(5)
    if fallback_thread.is_alive() or fallback_results != [True] or not capture_interleaved:
        raise AssertionError(f"atomic bundle fallback fixture drifted: {fallback_results!r}")
    if answer != "current-message-only after atomic capture" or model.call_count != 1:
        raise AssertionError("atomic bundle capture did not preserve current-message-only chat")
    payload = _provider_text(model.call_args)
    if prompt not in payload or any(marker in payload for marker in STORED_MARKERS):
        raise AssertionError(f"torn-capture regression disclosed stored context: {payload}")
    with old_store.connect() as conn:
        old_receipts = [dict(row) for row in conn.execute("SELECT * FROM history_disclosure_receipts")]
    with runtime.store.connect() as conn:
        new_receipts = [dict(row) for row in conn.execute("SELECT * FROM history_disclosure_receipts")]
    if old_receipts or new_receipts:
        raise AssertionError(
            f"atomic capture race prepared a receipt without egress: {old_receipts!r} / {new_receipts!r}"
        )
    if (
        old_chat.last_turn_metadata.get("history_disclosure_prepare_status") != "failed_stripped"
        or old_chat.last_turn_metadata.get("history_disclosure_finalize_status") == "confirmed"
    ):
        raise AssertionError(
            f"atomic capture race claimed confirmation: {old_chat.last_turn_metadata}"
        )


def test_storage_fallback_callback_swap_uses_pinned_finalizer(root: Path) -> None:
    runtime = JarvisRuntime(_config(root / "primary"))
    _seed_stored_context(runtime)
    old_chat = runtime.chat
    old_store = runtime.store
    _history_callbacks(old_chat)
    original_mutation_fence = old_store.history_epoch_mutation_fence
    callbacks_replaced = threading.Event()
    fallback_done = threading.Event()
    fallback_results: list[object] = []

    @contextmanager
    def observe_callback_replacement():
        callbacks_replaced.set()
        with original_mutation_fence():
            yield

    old_store.history_epoch_mutation_fence = observe_callback_replacement  # type: ignore[assignment]

    def activate_fallback() -> None:
        try:
            fallback_results.append(
                runtime._activate_default_storage_fallback(
                    sqlite3.OperationalError("attempt to write a readonly database")
                )
            )
        except BaseException as exc:
            fallback_results.append(exc)
        finally:
            fallback_done.set()

    fallback_thread = threading.Thread(target=activate_fallback, name="storage-fallback-swap")

    def invoke_while_callbacks_are_replaced(**_kwargs) -> str:
        fallback_thread.start()
        if not callbacks_replaced.wait(5):
            raise AssertionError("storage fallback did not reach callback replacement")
        if old_chat._history_disclosure_callbacks is not None:
            raise AssertionError("fallback fixture did not atomically clear the live callback bundle")
        if fallback_done.is_set():
            raise AssertionError("storage fallback completed while disclosure held authority")
        with old_store.connect() as conn:
            states = [
                str(row["state"])
                for row in conn.execute(
                    "SELECT state FROM history_disclosure_receipts WHERE session_id = ?",
                    (runtime.session_id,),
                )
            ]
        if states != ["prepared"]:
            raise AssertionError(f"callback swap did not occur during prepared egress: {states}")
        return "provider completed during callback swap"

    fallback_root = root / "fallback"
    try:
        with _storage_fallback_env(fallback_root), patch(
            "jarvis_v2.agent.chat.generate_model_text",
            side_effect=invoke_while_callbacks_are_replaced,
        ):
            answer = old_chat.respond("Discuss architecture during storage fallback")
    finally:
        fallback_thread.join(5)
        old_store.history_epoch_mutation_fence = original_mutation_fence  # type: ignore[assignment]
    if fallback_thread.is_alive():
        raise AssertionError("storage fallback deadlocked after pinned finalization released authority")
    if fallback_results != [True] or answer != "provider completed during callback swap":
        raise AssertionError(f"storage fallback regression fixture drifted: {fallback_results!r}")
    with old_store.connect() as conn:
        receipts = [dict(row) for row in conn.execute("SELECT * FROM history_disclosure_receipts")]
    final_state = receipts[0]["state"] if len(receipts) == 1 else None
    metadata_state = old_chat.last_turn_metadata.get("history_disclosure_finalize_status")
    if final_state not in {"confirmed", "uncertain"}:
        raise AssertionError(f"callback swap left the old receipt nonterminal: {receipts!r}")
    if metadata_state == "confirmed" and final_state != "confirmed":
        raise AssertionError(
            f"callback swap metadata claimed confirmed over durable {final_state!r}: {receipts!r}"
        )


def test_storage_fallback_rebinds_and_tables_are_content_free(root: Path) -> None:
    runtime = JarvisRuntime(_config(root / "primary"))
    old_chat = runtime.chat
    old_store = runtime.store
    old_epoch = runtime._history_policy_epoch_id
    fallback_root = root / "fallback"
    with _storage_fallback_env(fallback_root):
        activated = runtime._activate_default_storage_fallback(
            sqlite3.OperationalError("attempt to write a readonly database")
        )
    if not activated or runtime.store is old_store or runtime.chat is old_chat:
        raise AssertionError("storage fallback did not replace the store and ChatBrain")
    if runtime._history_policy_epoch_id in {None, old_epoch}:
        raise AssertionError("storage fallback did not mint a fresh policy epoch")
    if runtime.store.active_history_policy_epoch_id() != runtime._history_policy_epoch_id:
        raise AssertionError("fallback callback binding does not own its active epoch")
    if old_chat._history_disclosure_callbacks is not None:
        raise AssertionError("old-store callback authority survived fallback")
    _chat(runtime, "Chat on rebound fallback storage", PRIVATE_MARKER)
    if runtime.store.read_message_provenance(int(_messages(runtime)[-1]["id"]))[
        "provenance_state"
    ] != "recorded":
        raise AssertionError("fallback store did not persist provenance")

    with runtime.store.connect() as conn:
        forbidden = {"content", "metadata", "request", "args", "output", "body"}
        for table in (
            "history_policy_epochs",
            "active_history_policy_epoch",
            "message_provenance",
            "history_disclosure_receipts",
        ):
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if columns & forbidden:
                raise AssertionError(f"{table} has content-bearing columns: {columns}")
            rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
            if PRIVATE_MARKER in json.dumps(rows, sort_keys=True):
                raise AssertionError(f"{table} retained message content")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-runtime-history-provenance-") as temp:
        root = Path(temp)
        test_normal_tools_suggestions_and_receipts(root / "normal")
        test_prepare_failure_strips_history(root / "prepare-failure")
        test_malformed_and_attachment_failure_stay_local(root / "local-only")
        test_cross_runtime_staleness_and_rotation(root / "authority")
        test_stale_runtime_with_empty_history_strips_all_stored_context(root / "empty-history")
        test_epoch_rotation_between_prepare_and_provider_blocks_egress(root / "pre-egress-race")
        test_cross_process_rotation_waits_for_chat_invocation_and_finalization(
            root / "during-egress-race"
        )
        test_crashed_fence_owner_releases_os_lock(root / "crash-release")
        test_history_authority_thread_reentrancy_and_outer_release(root / "thread-reentrancy")
        test_history_authority_fork_cleanup(root / "fork-cleanup")
        test_history_authority_fence_timeout(root / "fence-timeout")
        test_storage_fallback_during_atomic_bundle_capture_strips_egress(
            root / "fallback-atomic-capture"
        )
        test_storage_fallback_callback_swap_uses_pinned_finalizer(root / "fallback-swap")
        test_storage_fallback_rebinds_and_tables_are_content_free(root / "fallback-case")
    print("runtime history provenance smoke passed")


if __name__ == "__main__":
    main()
