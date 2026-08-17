from __future__ import annotations

import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.obsidian import (
    ObsidianVault,
    ProfileEvidenceRevalidationError,
    _profile_note_block,
)
from jarvis_v2.memory.store import MemoryRecord


SOURCE_KEY = "profile-note:v1:" + ("d" * 64)
MANUAL_HEAD = "MANUAL_DIRECT_PROFILE_HEAD"
MANUAL_TAIL = "MANUAL_DIRECT_PROFILE_TAIL"
UNOWNED_HEADING = "UNOWNED_CANONICAL_HEADING"
UNOWNED_PAYLOAD = "UNOWNED_CANONICAL_PROFILE_PAYLOAD"
RAW_MANUAL_PATH = "/\x55sers/example/private/manual-profile.txt"


def _fullwidth_ascii(value: str) -> str:
    return "".join(
        chr(ord(character) + 0xFEE0) if "!" <= character <= "~" else character
        for character in value
    )


def test_direct_read_never_exposes_unowned_canonical_block() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-direct-custody-") as temp:
        vault = ObsidianVault(Path(temp))
        vault.init()
        profile_path = vault.root_path / "Profile.md"
        canonical = _profile_note_block(
            SOURCE_KEY,
            UNOWNED_HEADING,
            UNOWNED_PAYLOAD,
        )
        profile_path.write_text(
            (
                f"# Profile\n\n{MANUAL_HEAD}\n"
                f"Manual path: {RAW_MANUAL_PATH}\n"
                f"{canonical}\n{MANUAL_TAIL}\n"
            ),
            encoding="utf-8",
        )

        visible = vault.read_profile(20_000)
        forbidden = (
            UNOWNED_HEADING,
            UNOWNED_PAYLOAD,
            "jarvis-profile-note",
            "<!--",
            RAW_MANUAL_PATH,
        )
        if any(value in visible for value in forbidden):
            raise AssertionError(f"direct profile read exposed unowned generated data: {visible!r}")
        if MANUAL_HEAD not in visible or MANUAL_TAIL not in visible:
            raise AssertionError(f"direct profile read discarded manual text: {visible!r}")
        if "<local-path>" not in visible:
            raise AssertionError(f"direct profile read did not redact the manual path: {visible!r}")

        bounded = vault.read_profile(32)
        if len(bounded) > 32:
            raise AssertionError(f"direct profile read exceeded its max bound: {len(bounded)}")
        if UNOWNED_PAYLOAD in bounded or "jarvis-profile-note" in bounded:
            raise AssertionError(f"bounded direct read exposed generated data: {bounded!r}")


def test_nfkc_markers_are_never_manual_profile_text() -> None:
    digest = "e" * 64
    start = f"<!-- jarvis-profile-note-start:v1:{digest} -->\n"
    end = f"<!-- jarvis-profile-note:v1:{digest} -->\n"
    future_start = "<!-- jarvis-profile-note-start:v2:future -->\n"
    future_end = "<!-- jarvis-profile-note:v2:future -->\n"
    payload = "COMPATIBILITY_MARKER_UNTRUSTED_PAYLOAD"
    cases = {
        "compatibility pair": (
            f"{MANUAL_HEAD}\n{_fullwidth_ascii(start)}{payload}\n"
            f"{_fullwidth_ascii(end)}{MANUAL_TAIL}\n"
        ),
        "future pair": (
            f"{MANUAL_HEAD}\n{_fullwidth_ascii(future_start)}{payload}\n"
            f"{_fullwidth_ascii(future_end)}{MANUAL_TAIL}\n"
        ),
        "nested pair": (
            f"{MANUAL_HEAD}\n{start}{payload}\n{_fullwidth_ascii(start)}"
            f"NESTED_{payload}\n{_fullwidth_ascii(end)}{end}{MANUAL_TAIL}\n"
        ),
        "orphaned end": (
            f"ORPHANED_{payload}\n{_fullwidth_ascii(end)}{MANUAL_TAIL}\n"
        ),
        "truncated start": (
            f"{MANUAL_HEAD}\n{_fullwidth_ascii(start.rstrip())}{payload}"
        ),
        "indented truncated prefix": (
            f"{MANUAL_HEAD}\n   <!-- jarvis-profi"
        ),
    }

    with TemporaryDirectory(prefix="jarvis-profile-nfkc-markers-") as temp:
        vault = ObsidianVault(Path(temp))
        vault.init()
        profile_path = vault.root_path / "Profile.md"
        for label, content in cases.items():
            profile_path.write_text(content, encoding="utf-8")
            visible = vault.read_profile(20_000)
            normalized_visible = unicodedata.normalize("NFKC", visible).lower()
            if (
                payload in visible
                or "jarvis-profile-note" in normalized_visible
                or "<!-- jarvis-profi" in normalized_visible
            ):
                raise AssertionError(f"{label} exposed normalized marker content: {visible!r}")
            if label != "orphaned end" and MANUAL_HEAD not in visible:
                raise AssertionError(f"{label} discarded trusted manual prefix: {visible!r}")
            if label not in {"truncated start", "indented truncated prefix"} and MANUAL_TAIL not in visible:
                raise AssertionError(f"{label} discarded trusted manual suffix: {visible!r}")
            grounding = vault.read_profile_grounding((), max_chars=20_000)
            normalized_grounding = unicodedata.normalize("NFKC", grounding.text).lower()
            if (
                not grounding.invalid
                or payload in grounding.text
                or "jarvis-profile-note" in normalized_grounding
            ):
                raise AssertionError(
                    f"{label} did not fail closed in profile grounding: {grounding!r}"
                )


def test_nfkc_marker_repair_preserves_exact_evidence() -> None:
    heading = "NFKC_REPAIR_HEADING"
    body = "NFKC_REPAIR_PAYLOAD"
    marker = f"jarvis-{SOURCE_KEY}"
    start_marker = marker.replace(
        "jarvis-profile-note:v1:",
        "jarvis-profile-note-start:v1:",
        1,
    )
    start_line = f"<!-- {start_marker} -->\n"
    end_line = f"<!-- {marker} -->\n"
    canonical = _profile_note_block(SOURCE_KEY, heading, body).lstrip("\n")
    compatibility = canonical.replace(
        start_line,
        _fullwidth_ascii(start_line),
        1,
    ).replace(
        end_line,
        _fullwidth_ascii(end_line),
        1,
    )

    with TemporaryDirectory(prefix="jarvis-profile-nfkc-repair-") as temp:
        vault = ObsidianVault(Path(temp))
        vault.init()
        profile_path = vault.root_path / "Profile.md"
        profile_path.write_text(
            f"# Profile\n\n{compatibility}",
            encoding="utf-8",
        )
        before = profile_path.read_bytes()
        try:
            vault.append_profile_once(SOURCE_KEY, heading, body)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "NFKC compatibility markers were accepted for ASCII repair"
            )
        after = profile_path.read_bytes()
        if after != before:
            raise AssertionError(
                "failed NFKC marker repair changed or duplicated the original evidence"
            )


def test_profile_evidence_locks_preserve_inflight_exceptions() -> None:
    class ProviderFailure(RuntimeError):
        pass

    class ToolFailure(RuntimeError):
        pass

    heading = "EVIDENCE_LOCK_HEADING"
    body = "EVIDENCE_LOCK_BODY"

    with TemporaryDirectory(prefix="jarvis-profile-evidence-lock-") as temp:
        vault = ObsidianVault(Path(temp))
        vault.init()
        profile_path = vault.root_path / "Profile.md"

        provider_error = ProviderFailure("provider failed in flight")
        try:
            with vault.profile_grounding_evidence_lock():
                profile_path.write_text("# Profile\n\nPROVIDER_DRIFT\n", encoding="utf-8")
                raise provider_error
        except ProviderFailure as exc:
            if exc is not provider_error:
                raise AssertionError("provider exception identity was not preserved") from exc
        except ProfileEvidenceRevalidationError as exc:
            raise AssertionError("profile teardown replaced the provider exception") from exc
        else:
            raise AssertionError("provider exception was swallowed by profile teardown")

        profile_path.write_text("# Profile\n\nGROUNDING_STABLE\n", encoding="utf-8")
        grounding_returned: str | None = None

        def disclose_grounding_after_drift() -> str:
            with vault.profile_grounding_evidence_lock():
                profile_path.write_text(
                    "# Profile\n\nGROUNDING_RETURN_DRIFT\n",
                    encoding="utf-8",
                )
                return "UNSAFE_GROUNDING_RETURNED_DATA"

        try:
            grounding_returned = disclose_grounding_after_drift()
        except ProfileEvidenceRevalidationError:
            pass
        else:
            raise AssertionError(
                "successful profile grounding body skipped exit revalidation"
            )
        if grounding_returned is not None:
            raise AssertionError("profile grounding drift allowed returned data to escape")

        profile_path.write_text("# Profile\n\n", encoding="utf-8")
        vault.append_profile_once(SOURCE_KEY, heading, body)
        canonical_bytes = profile_path.read_bytes()
        tool_error = ToolFailure("tool failed in flight")
        try:
            with vault.canonical_profile_note_evidence_lock(
                source_key=SOURCE_KEY,
                heading=heading,
                body=body,
            ) as verified:
                if not verified:
                    raise AssertionError("canonical profile evidence was not valid on entry")
                profile_path.write_text("# Profile\n\nTOOL_DRIFT\n", encoding="utf-8")
                raise tool_error
        except ToolFailure as exc:
            if exc is not tool_error:
                raise AssertionError("tool exception identity was not preserved") from exc
        except ProfileEvidenceRevalidationError as exc:
            raise AssertionError("profile teardown replaced the tool exception") from exc
        else:
            raise AssertionError("tool exception was swallowed by profile teardown")

        profile_path.write_bytes(canonical_bytes)
        returned: str | None = None

        def disclose_after_drift() -> str:
            with vault.canonical_profile_note_evidence_lock(
                source_key=SOURCE_KEY,
                heading=heading,
                body=body,
            ) as verified:
                if not verified:
                    raise AssertionError("canonical profile evidence was not valid on entry")
                profile_path.write_text("# Profile\n\nRETURN_DRIFT\n", encoding="utf-8")
                return "UNSAFE_RETURNED_DATA"

        try:
            returned = disclose_after_drift()
        except ProfileEvidenceRevalidationError:
            pass
        else:
            raise AssertionError("successful profile evidence body skipped exit revalidation")
        if returned is not None:
            raise AssertionError("profile evidence drift allowed returned data to escape")

        store_identity = "a" * 32
        projection_record = MemoryRecord(
            "profile",
            "Projection evidence",
            "Projection evidence must preserve an in-flight provider error.",
            "profile",
            1.0,
        )
        projection_path, projection_digest = vault.write_memory_projection_with_evidence(
            projection_record,
            memory_id=17,
            store_identity=store_identity,
            memory_revision=1,
            source_digest="b" * 64,
            created_at="2026-07-13T00:00:00Z",
        )
        projection_error = ProviderFailure("provider failed with projection evidence held")
        try:
            with vault.canonical_memory_projection_evidence_lock(
                memory_id=17,
                store_identity=store_identity,
                expected_relative_path=str(projection_path.relative_to(vault.root_path)),
                expected_content_digest=projection_digest,
            ) as verified:
                if not verified:
                    raise AssertionError("canonical memory projection was not valid on entry")
                projection_path.write_text("PROJECTION_DRIFT\n", encoding="utf-8")
                raise projection_error
        except ProviderFailure as exc:
            if exc is not projection_error:
                raise AssertionError("projection lock changed provider exception identity") from exc
        except ProfileEvidenceRevalidationError as exc:
            raise AssertionError("projection teardown replaced the provider exception") from exc
        else:
            raise AssertionError("projection lock swallowed the provider exception")


def main() -> None:
    test_direct_read_never_exposes_unowned_canonical_block()
    test_nfkc_markers_are_never_manual_profile_text()
    test_nfkc_marker_repair_preserves_exact_evidence()
    test_profile_evidence_locks_preserve_inflight_exceptions()
    print("profile direct read custody smoke passed")


if __name__ == "__main__":
    main()
