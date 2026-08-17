from __future__ import annotations

import multiprocessing as mp
import threading
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory

from jarvis_v2.memory.store import (
    MAX_PROFILE_OWNERSHIP_EGRESS_MEMORY_IDS,
    MemoryRecord,
    MemoryStore,
    profile_note_source_key,
)


def _profile_record(label: str) -> tuple[MemoryRecord, str]:
    record = MemoryRecord(
        "profile-egress-fence",
        f"Profile ownership {label}",
        f"Durable profile ownership created by the {label} fixture.",
        "profile",
        1.0,
    )
    return record, profile_note_source_key(record.title, record.body, record.category)


def _spawn_generic_profile_creator(
    db_path: str,
    attempted: object,
    completed: object,
    results: object,
) -> None:
    attempted.set()
    try:
        store = MemoryStore(Path(db_path))
        record, source_key = _profile_record("spawned process")
        inserted, target = store.add_memory_if_source_new_with_projection(
            record,
            source_key,
            "profile_note",
        )
        results.put(("ok", inserted, target.memory_id))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))
    finally:
        completed.set()


def _assert_invalid_ids_are_rejected(store: MemoryStore) -> None:
    invalid_cases: tuple[object, ...] = (
        (0,),
        (-1,),
        (True,),
        (9223372036854775808,),
        tuple(range(1, MAX_PROFILE_OWNERSHIP_EGRESS_MEMORY_IDS + 2)),
    )
    for memory_ids in invalid_cases:
        try:
            with store.profile_memory_ownership_egress_fence(memory_ids):
                pass
        except ValueError:
            continue
        raise AssertionError(f"invalid profile ownership egress ids were accepted: {memory_ids!r}")


def test_thread_fence_is_reentrant_and_does_not_hold_sqlite_write_lock() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-ownership-thread-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        ordinary_id = store.add_memory(
            MemoryRecord(
                "ordinary",
                "Thread ordinary control",
                "This memory must remain usable while profile egress is in flight.",
                "manual",
                0.8,
            )
        )
        _assert_invalid_ids_are_rejected(store)

        profile_attempted = threading.Event()
        profile_completed = threading.Event()
        ordinary_attempted = threading.Event()
        ordinary_completed = threading.Event()
        profile_results: list[object] = []
        ordinary_results: list[object] = []

        def create_profile() -> None:
            profile_attempted.set()
            try:
                record, source_key = _profile_record("thread")
                profile_results.append(
                    store.ensure_profile_note_memory_with_projection(record, source_key)
                )
            except BaseException as exc:
                profile_results.append(exc)
            finally:
                profile_completed.set()

        def create_ordinary() -> None:
            ordinary_attempted.set()
            try:
                ordinary_results.append(
                    store.add_memory(
                        MemoryRecord(
                            "ordinary",
                            "Concurrent ordinary control",
                            "Ordinary writes are not serialized behind remote profile egress.",
                            "manual",
                            0.7,
                        )
                    )
                )
            except BaseException as exc:
                ordinary_results.append(exc)
            finally:
                ordinary_completed.set()

        profile_thread = threading.Thread(target=create_profile, name="profile-owner-thread")
        ordinary_thread = threading.Thread(target=create_ordinary, name="ordinary-memory-thread")
        with store.profile_memory_ownership_egress_fence((ordinary_id,)) as owned:
            if owned:
                raise AssertionError(f"ordinary capture was already profile-owned: {owned}")
            with store.profile_memory_ownership_egress_fence((ordinary_id,)) as nested_owned:
                if nested_owned:
                    raise AssertionError(f"nested ownership check drifted: {nested_owned}")

            profile_thread.start()
            if not profile_attempted.wait(2):
                raise AssertionError("profile creator did not reach the thread fence")
            if profile_completed.wait(0.2):
                raise AssertionError("profile ownership creation bypassed the thread fence")

            ordinary_thread.start()
            if not ordinary_attempted.wait(2) or not ordinary_completed.wait(2):
                raise AssertionError("ordinary memory creation was globally blocked by egress")
            if not ordinary_results or isinstance(ordinary_results[0], BaseException):
                raise AssertionError(f"ordinary memory creation failed: {ordinary_results!r}")
            if profile_completed.is_set():
                raise AssertionError("unrelated memory work released the profile ownership fence")

        profile_thread.join(3)
        ordinary_thread.join(3)
        if profile_thread.is_alive() or ordinary_thread.is_alive():
            raise AssertionError("thread fixture did not finish after ownership fence release")
        if len(profile_results) != 1 or isinstance(profile_results[0], BaseException):
            raise AssertionError(f"profile creation failed after thread release: {profile_results!r}")
        _inserted, _repaired, target = profile_results[0]
        with store.profile_memory_ownership_egress_fence(
            (ordinary_id, target.memory_id)
        ) as owned_after:
            if owned_after != frozenset({target.memory_id}):
                raise AssertionError(f"durable thread ownership was not observed: {owned_after}")
        if store.get_memory(target.memory_id) is not None:
            raise AssertionError("generic direct memory read exposed thread-created profile ownership")


def test_spawned_process_profile_creation_waits_for_release() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-ownership-spawn-") as temp:
        store = MemoryStore(Path(temp) / "memory.sqlite")
        store.init()
        ordinary_id = store.add_memory(
            MemoryRecord(
                "ordinary",
                "Process ordinary control",
                "The spawned profile writer must wait while this ID is cleared for egress.",
                "manual",
                0.8,
            )
        )
        ctx = mp.get_context("spawn")
        attempted = ctx.Event()
        completed = ctx.Event()
        results = ctx.Queue()
        process = ctx.Process(
            target=_spawn_generic_profile_creator,
            args=(str(store.db_path), attempted, completed, results),
        )
        try:
            with store.profile_memory_ownership_egress_fence((ordinary_id,)) as owned:
                if owned:
                    raise AssertionError(f"spawn capture was already profile-owned: {owned}")
                process.start()
                if not attempted.wait(3):
                    raise AssertionError("spawned profile creator did not reach the process fence")
                if completed.wait(0.3):
                    raise AssertionError("spawned profile ownership bypassed the process fence")

            process.join(5)
            if process.is_alive():
                raise AssertionError("spawned profile creator did not finish after fence release")
            try:
                result = results.get(timeout=2)
            except Empty as exc:
                raise AssertionError("spawned profile creator returned no result") from exc
            if process.exitcode != 0 or result[0] != "ok" or result[1] is not True:
                raise AssertionError(
                    f"spawned profile creation failed after release: exit={process.exitcode}, "
                    f"result={result!r}"
                )
            profile_id = int(result[2])
            with store.profile_memory_ownership_egress_fence(
                (ordinary_id, profile_id)
            ) as owned_after:
                if owned_after != frozenset({profile_id}):
                    raise AssertionError(
                        f"durable spawned ownership was not observed: {owned_after}"
                    )
        finally:
            if process.is_alive():
                process.terminate()
                process.join(2)
            results.close()
            results.join_thread()


def main() -> None:
    test_thread_fence_is_reentrant_and_does_not_hold_sqlite_write_lock()
    test_spawned_process_profile_creation_waits_for_release()
    print("profile ownership egress fence smoke: PASS")


if __name__ == "__main__":
    main()
