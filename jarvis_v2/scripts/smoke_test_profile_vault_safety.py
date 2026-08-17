from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.obsidian import (
    ObsidianVault,
    _profile_note_block,
    safe_text,
)


SOURCE_KEY = "profile-note:v1:" + ("a" * 64)
OTHER_SOURCE_KEY = "profile-note:v1:" + ("b" * 64)
HEADING = "Vault safety"
BODY = "Canonical generated profile body."
ATTACK = "UNTRUSTED_GENERATED_PAYLOAD"
MANUAL = "MANUAL_PROFILE_TEXT"


def _vault(root: Path) -> ObsidianVault:
    vault = ObsidianVault(root)
    vault.init()
    return vault


def _write_profile(vault: ObsidianVault, content: str) -> None:
    (vault.root_path / "Profile.md").write_text(content, encoding="utf-8")


def _assert_failed_closed(
    vault: ObsidianVault,
    *,
    label: str,
    require_manual: bool = True,
) -> None:
    visible = vault.read_profile(20_000)
    if ATTACK in visible or "jarvis-profile-note" in visible or "<!--" in visible:
        raise AssertionError(f"{label} exposed owned marker content: {visible!r}")
    if require_manual and MANUAL not in visible:
        raise AssertionError(f"{label} discarded unambiguous manual profile text: {visible!r}")


def test_read_profile_uses_owned_block_parser() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-vault-parser-") as tmp:
        vault = _vault(Path(tmp))
        canonical = _profile_note_block(SOURCE_KEY, HEADING, BODY)
        _write_profile(vault, f"# Profile\n\n{MANUAL}\n" + canonical + "\nMANUAL_TAIL\n")
        visible = vault.read_profile(20_000)
        if (
            MANUAL not in visible
            or "MANUAL_TAIL" not in visible
            or BODY in visible
            or "jarvis-profile-note" in visible
        ):
            raise AssertionError(
                f"unowned canonical profile block was not withheld safely: {visible!r}"
            )

        digest = SOURCE_KEY.rsplit(":", 1)[1]
        other_digest = OTHER_SOURCE_KEY.rsplit(":", 1)[1]
        malformed_cases = {
            "future version": (
                (
                    f"<!-- jarvis-profile-note-start:v2:{digest} -->\n"
                    f"{ATTACK}\n"
                    f"<!-- jarvis-profile-note:v2:{digest} -->\n"
                ),
                True,
            ),
            "mismatched pair": (
                (
                    f"<!-- jarvis-profile-note-start:v1:{digest} -->\n"
                    f"{ATTACK}\n"
                    f"<!-- jarvis-profile-note:v1:{other_digest} -->\n"
                ),
                True,
            ),
            "nested pair": (
                (
                    f"<!-- jarvis-profile-note-start:v1:{digest} -->\n"
                    f"{ATTACK}\n"
                    f"<!-- jarvis-profile-note-start:v1:{other_digest} -->\n"
                    f"{ATTACK}\n"
                    f"<!-- jarvis-profile-note:v1:{other_digest} -->\n"
                    f"<!-- jarvis-profile-note:v1:{digest} -->\n"
                ),
                True,
            ),
            "orphan end": (
                f"{ATTACK}\n<!-- jarvis-profile-note:v1:{digest} -->\n",
                False,
            ),
            "duplicate block": (
                canonical.lstrip("\n") + canonical.lstrip("\n"),
                True,
            ),
        }
        for label, (owned_text, require_manual) in malformed_cases.items():
            _write_profile(
                vault,
                f"# Profile\n\n{MANUAL}\n{owned_text}\nMANUAL_TAIL\n",
            )
            _assert_failed_closed(vault, label=label, require_manual=require_manual)


def test_read_profile_rejects_every_partial_marker_prefix() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-vault-truncated-") as tmp:
        vault = _vault(Path(tmp))
        digest = SOURCE_KEY.rsplit(":", 1)[1]
        start_marker = f"<!-- jarvis-profile-note-start:v1:{digest} -->"
        end_marker = f"<!-- jarvis-profile-note:v1:{digest} -->"
        manual_prefix = f"# Profile\n\n{MANUAL}\n"

        for cut in range(1, len(start_marker)):
            _write_profile(vault, manual_prefix + start_marker[:cut])
            _assert_failed_closed(vault, label=f"partial start marker byte {cut}")

        owned_prefix = manual_prefix + start_marker + "\n" + ATTACK + "\n"
        _write_profile(vault, owned_prefix)
        _assert_failed_closed(vault, label="missing end marker")
        for cut in range(1, len(end_marker)):
            _write_profile(vault, owned_prefix + end_marker[:cut])
            _assert_failed_closed(vault, label=f"partial end marker byte {cut}")


def test_safe_text_redacts_raw_paths_without_consuming_urls() -> None:
    urls = (
        "https://example.com/\x55sers/review",
        "https://example.com/System/Volumes/Data/\x55sers/review",
        "https://example.com/root/reference",
        "http://C:/\x55sers/reference/public.txt",
    )
    raw = (
        "windows C:/\x55sers/example-user/private.txt then " + urls[0] + "\n"
        "mac /System/Volumes/Data/\x55sers/example-user/private.txt\n"
        "linux /root/.ssh/example-key\n"
        "exact roots /System/Volumes/Data/Users and /root\n"
        + "\n".join(urls[1:])
    )
    redacted = safe_text(raw)
    forbidden = (
        "C:/\x55sers/example-user",
        "/System/Volumes/Data/\x55sers/example-user",
        "/root/.ssh",
        "exact roots /System/Volumes/Data/Users",
        "and /root",
    )
    if "<local-path>" not in redacted or any(item in redacted for item in forbidden):
        raise AssertionError(f"raw local paths were not fully redacted: {redacted!r}")
    if any(url not in redacted for url in urls):
        raise AssertionError(f"path redaction consumed an HTTP URL: {redacted!r}")


def test_exceptional_evidence_lock_exit_preserves_original_and_detects_drift() -> None:
    with TemporaryDirectory(prefix="jarvis-profile-vault-lock-") as tmp:
        vault = _vault(Path(tmp))
        vault.append_profile_once(SOURCE_KEY, HEADING, BODY)

        unchanged_error = RuntimeError("unchanged evidence failure")
        try:
            with vault.canonical_profile_note_evidence_lock(
                source_key=SOURCE_KEY,
                heading=HEADING,
                body=BODY,
            ) as evidence_current:
                if not evidence_current:
                    raise AssertionError("canonical profile evidence was not current at lock entry")
                raise unchanged_error
        except RuntimeError as exc:
            if exc is not unchanged_error:
                raise AssertionError("unchanged evidence did not preserve the original exception") from exc
        else:
            raise AssertionError("unchanged exceptional lock exit unexpectedly succeeded")

        drift_error = ValueError("original exceptional exit")
        profile_path = vault.root_path / "Profile.md"
        try:
            with vault.canonical_profile_note_evidence_lock(
                source_key=SOURCE_KEY,
                heading=HEADING,
                body=BODY,
            ) as evidence_current:
                if not evidence_current:
                    raise AssertionError("canonical profile evidence was not current before drift")
                profile_path.write_text(
                    profile_path.read_text(encoding="utf-8").replace(BODY, ATTACK),
                    encoding="utf-8",
                )
                raise drift_error
        except ValueError as exc:
            if exc is not drift_error:
                raise AssertionError("profile drift did not preserve the original exception") from exc
        else:
            raise AssertionError("exceptional evidence drift did not raise the typed error")

        with vault.canonical_profile_note_evidence_lock(
            source_key=SOURCE_KEY,
            heading=HEADING,
            body=BODY,
        ) as evidence_current:
            if evidence_current:
                raise AssertionError("profile drift was not rejected on the next clean lock entry")


def main() -> None:
    test_read_profile_uses_owned_block_parser()
    test_read_profile_rejects_every_partial_marker_prefix()
    test_safe_text_redacts_raw_paths_without_consuming_urls()
    test_exceptional_evidence_lock_exit_preserves_original_and_detects_drift()
    print("profile vault safety smoke passed")


if __name__ == "__main__":
    main()
