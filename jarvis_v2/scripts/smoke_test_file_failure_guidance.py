"""Focused offline proof for file-tool failure recovery declarations."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.failure_guidance import (
    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
    LOCAL_READ_INPUT_RECOVERY_ACTION,
    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
)
from jarvis_v2.tools import files


PRIVATE_FRAGMENTS = ("/\x55sers/", "/private/", "/var/folders/", "/tmp/")


def _assert_guided(
    result: object,
    *,
    action: str,
    label: str,
    retry_safe: bool = True,
    side_effect_possible: bool = False,
) -> None:
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} hid its public recovery action: {result}")
    guidance = result.metadata.get("recovery_guidance")
    if not isinstance(guidance, dict) or guidance.get("version") != 1:
        raise SystemExit(f"{label} omitted canonical recovery guidance: {result.metadata}")
    if guidance.get("action") != action:
        raise SystemExit(f"{label} recovery action drifted: {guidance}")
    expected = {
        "outcome_known": True,
        "outcome_unknown": False,
        "execution_outcome_unknown": False,
        "side_effect_possible": side_effect_possible,
        "retry_safe": retry_safe,
        "automatic_retry_allowed": False,
        "authorizes_retry": False,
    }
    for key, value in expected.items():
        if result.metadata.get(key) is not value:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    public = f"{result.output}\n{result.metadata}"
    for fragment in PRIVATE_FRAGMENTS:
        if fragment in public:
            raise SystemExit(f"{label} leaked private local detail: {public}")


class _FakeReadPath:
    suffix = ".md"
    name = "private.md"

    def __init__(self, failure: str) -> None:
        self.failure = failure

    def __fspath__(self) -> str:
        return "/\x55sers/example/private/private.md"

    def __str__(self) -> str:
        return self.__fspath__()

    def exists(self) -> bool:
        return True

    def is_file(self) -> bool:
        return True

    def stat(self):
        if self.failure == "inspect":
            raise OSError("private inspect failure near /\x55sers/example/private")
        return type("Stat", (), {"st_size": 12})()

    def read_text(self, **_kwargs: object) -> str:
        raise OSError("private read failure near /\x55sers/example/private")


def main() -> None:
    cases: list[tuple[object, str, str, bool, bool]] = []
    with TemporaryDirectory(prefix="jarvis-file-guidance-") as temp:
        root = Path(temp)
        text = root / "text.md"
        text.write_text("ordinary text", encoding="utf-8")
        folder = root / "folder"
        folder.mkdir()
        binary = root / "image.bin"
        binary.write_bytes(b"binary")
        large = root / "large.md"
        large.write_bytes(b"x" * (files.MAX_READ_BYTES + 1))

        cases.extend(
            (
                (
                    files.list_files({"directory": root / "missing"}),
                    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
                    "list missing directory",
                    True,
                    False,
                ),
                (
                    files.list_files({"directory": text}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "list non-directory",
                    True,
                    False,
                ),
                (
                    files.read_text_file({"path": ""}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "read missing path",
                    True,
                    False,
                ),
                (
                    files.read_text_file({"path": "Profile.md"}),
                    files.PROFILE_READ_RECOVERY_ACTION,
                    "read protected profile path",
                    True,
                    False,
                ),
                (
                    files.read_text_file({"path": root / "missing.md"}),
                    RESOURCE_NOT_FOUND_RECOVERY_ACTION,
                    "read missing file",
                    True,
                    False,
                ),
                (
                    files.read_text_file({"path": folder}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "read non-file",
                    True,
                    False,
                ),
                (
                    files.read_text_file({"path": binary}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "read non-text file",
                    True,
                    False,
                ),
                (
                    files.read_text_file({"path": large}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "read oversized file",
                    True,
                    False,
                ),
            )
        )

        for failure in ("inspect", "read"):
            with (
                patch.object(files, "expand_path", return_value=_FakeReadPath(failure)),
                patch.object(files, "is_protected_profile_path_or_content", return_value=False),
            ):
                result = files.read_text_file({"path": "private.md"})
            cases.append(
                (
                    result,
                    LOCAL_PRODUCTIVITY_READ_RECOVERY_ACTION,
                    f"read {failure} I/O failure",
                    True,
                    False,
                )
            )

        with patch.object(
            files,
            "is_protected_profile_path_or_content",
            side_effect=(False, True),
        ):
            protected_content = files.read_text_file({"path": text})
        cases.append(
            (
                protected_content,
                files.PROFILE_READ_RECOVERY_ACTION,
                "read protected content",
                True,
                False,
            )
        )

        existing = root / "existing.md"
        existing.write_text("keep", encoding="utf-8")
        cases.extend(
            (
                (
                    files.write_text_file({"path": "", "content": "body"}),
                    files.FILE_WRITE_INPUT_RECOVERY_ACTION,
                    "write missing path",
                    True,
                    False,
                ),
                (
                    files.write_text_file({"path": root / "bad.bin", "content": "body"}),
                    files.FILE_WRITE_INPUT_RECOVERY_ACTION,
                    "write non-text file",
                    True,
                    False,
                ),
                (
                    files.write_text_file(
                        {
                            "path": root / "oversized.md",
                            "content": "x" * (files.MAX_WRITE_CHARS + 1),
                        }
                    ),
                    files.FILE_WRITE_INPUT_RECOVERY_ACTION,
                    "write oversized content",
                    True,
                    False,
                ),
                (
                    files.write_text_file({"path": existing, "content": "new"}),
                    files.FILE_WRITE_INPUT_RECOVERY_ACTION,
                    "write existing file preflight",
                    True,
                    False,
                ),
            )
        )

        race_target = root / "race.md"
        with patch.object(
            files,
            "_durable_atomic_write_text",
            side_effect=files._TextFilePublicationError(
                FileExistsError("private race /\x55sers/example/private"),
                committed=False,
            ),
        ):
            race = files.write_text_file({"path": race_target, "content": "new"})
        cases.append(
            (
                race,
                files.FILE_WRITE_INPUT_RECOVERY_ACTION,
                "write existing file race",
                True,
                False,
            )
        )

        for committed, label, retry_safe in (
            (False, "write known unpublished", True),
            (True, "write published durability failure", False),
        ):
            target = root / f"publication-{committed}.md"
            with patch.object(
                files,
                "_durable_atomic_write_text",
                side_effect=files._TextFilePublicationError(
                    OSError("private storage failure /\x55sers/example/private"),
                    committed=committed,
                ),
            ):
                result = files.write_text_file({"path": target, "content": "new"})
            cases.append(
                (
                    result,
                    (
                        files.FILE_WRITE_VERIFY_RECOVERY_ACTION
                        if committed
                        else files.FILE_WRITE_RETRY_RECOVERY_ACTION
                    ),
                    label,
                    retry_safe,
                    True,
                )
            )

        mkdir_target = root / "new-parent" / "write.md"
        with patch.object(
            Path,
            "mkdir",
            side_effect=OSError("private mkdir failure /\x55sers/example/private"),
        ):
            mkdir_failure = files.write_text_file(
                {"path": mkdir_target, "content": "new"}
            )
        cases.append(
            (
                mkdir_failure,
                files.FILE_WRITE_RETRY_RECOVERY_ACTION,
                "write parent creation failure",
                True,
                True,
            )
        )

        cases.extend(
            (
                (
                    files.find_files({"root": text, "pattern": "text"}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "find invalid root",
                    True,
                    False,
                ),
                (
                    files.find_files({"root": root, "pattern": ""}),
                    LOCAL_READ_INPUT_RECOVERY_ACTION,
                    "find missing pattern",
                    True,
                    False,
                ),
            )
        )

    if len(cases) != 21:
        raise SystemExit(f"file failure-guidance branch scope drifted: {len(cases)}/21")
    for result, action, label, retry_safe, side_effect_possible in cases:
        _assert_guided(
            result,
            action=action,
            label=label,
            retry_safe=retry_safe,
            side_effect_possible=side_effect_possible,
        )

    print("File failure-guidance smoke passed: 21 branches across 4 tools.")


if __name__ == "__main__":
    main()
