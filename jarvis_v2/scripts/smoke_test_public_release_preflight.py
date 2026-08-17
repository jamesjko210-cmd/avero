"""Focused offline tests for the read-only public release preflight."""

from __future__ import annotations

import contextlib
import io
import json
import os
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from jarvis_v2.scripts import public_release_preflight as preflight_module
from jarvis_v2.scripts.public_release_preflight import (
    EXTRA_DENY_ENV,
    PreflightConfigError,
    _render_text,
    extra_deny_literals_from_env,
    scan_release_tree,
)


def _write(root: Path, relative: str, content: str = "safe\n") -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def _write_bytes(root: Path, relative: str, content: bytes) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def test_clean_synthetic_tree_passes_without_mutation() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-clean-") as temp:
        root = Path(temp)
        _write(root, ".env.example", "OPENAI_API_KEY=\n")
        _write(root, "README.md", "Contact maintainer@example.com for synthetic tests.\n")
        _write(
            root,
            "PUBLIC_CANDIDATE_MANIFEST.json",
            '{"history_included":false,"manifest_digest":"' + ("a" * 64) + '"}\n',
        )
        _write(root, "package/module.py", "VALUE = 'offline'\n")
        before = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        report = scan_release_tree(root)
        after = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
        if not report.ok or report.findings:
            raise SystemExit(f"clean release tree failed: {report}")
        if before != after:
            raise SystemExit("public release preflight mutated the candidate tree")
        if any((root / name).exists() for name in (".git", "public-export", "dist")):
            raise SystemExit("public release preflight created release or repository state")
        summary = report.summary_json_payload()
        if (
            summary.get("extra_private_deny_literals_supplied") is not False
            or summary.get("extra_private_deny_literal_count") != 0
        ):
            raise SystemExit("empty private deny-list receipt was not explicit")


def test_extended_token_formats_fail_closed() -> None:
    token_values = {
        "fine-grained.txt": "github_pat_" + ("A1_" * 16),
        "telegram.txt": "123456789012345:" + ("Ab_" * 16),
    }
    with TemporaryDirectory(prefix="jarvis-public-preflight-token-formats-") as temp:
        root = Path(temp)
        for name, value in token_values.items():
            _write(root, name, f"fixture={value}\n")
        report = scan_release_tree(root)
        credential_paths = {
            item.path for item in report.findings if item.code == "likely_credential"
        }
        if report.ok or credential_paths != set(token_values):
            raise SystemExit(f"extended credential format escaped preflight: {report}")
        rendered = _render_text(report)
        payload = json.dumps(report.json_payload(), sort_keys=True)
        if any(value in rendered or value in payload for value in token_values.values()):
            raise SystemExit("extended credential value leaked through preflight output")


def test_high_confidence_provider_tokens_detect_without_generic_secret_heuristics() -> None:
    token_values = {
        "gitlab.cfg": "glpat-" + ("A1_" * 7),
        "npm-prefixed.cfg": "npm_" + ("A1" * 18),
        "npm-hex.cfg": "NPM_TOKEN=" + ("a1" * 20),
        "pypi.cfg": "pypi-" + ("A" * 85),
        "hugging-face.cfg": "hf_" + ("A1" * 16),
        "anthropic.cfg": "sk-ant-api03-" + ("A1_" * 16),
        "aws-sts-id.cfg": "ASIA" + ("A1" * 8),
        "aws-secret.cfg": "aws_secret_access_key = " + ("aB3/" * 10),
        "aws-session.cfg": "AWS_SESSION_TOKEN=" + ("AQoD" * 25),
    }
    with TemporaryDirectory(prefix="jarvis-public-preflight-provider-tokens-") as temp:
        root = Path(temp)
        for name, value in token_values.items():
            _write(root, name, f"fixture={value}\n")
        report = scan_release_tree(root)
        credential_paths = {
            item.path for item in report.findings if item.code == "likely_credential"
        }
        if report.ok or credential_paths != set(token_values):
            raise SystemExit(f"provider credential family escaped preflight: {report}")
        rendered = _render_text(report)
        payload = json.dumps(report.json_payload(), sort_keys=True)
        summary = json.dumps(report.summary_json_payload(), sort_keys=True)
        if any(
            value in rendered or value in payload or value in summary
            for value in token_values.values()
        ):
            raise SystemExit("provider credential value leaked through preflight output")
        if report.policy_version != "3":
            raise SystemExit("credential-family expansion did not advance policy version")

    near_misses = "\n".join(
        (
            "GitLab tokens use the documented glpat- prefix.",
            "gitlab_short=glpat-short",
            "gitlab_embedded=Xglpat-" + ("A1_" * 7),
            "npm_placeholder=npm_...",
            "npm_short=npm_" + ("A" * 35),
            "npm_variable=NPM_TOKEN=${NPM_TOKEN}",
            "bare_hex=" + ("a1" * 20),
            "pypi_short=pypi-" + ("A" * 84),
            "hf_placeholder=hf_...",
            "hf_short=hf_" + ("A" * 29),
            "anthropic_placeholder=sk-ant-api03-short",
            "aws_sts_short=ASIA" + ("A1" * 7) + "A",
            "AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}",
            "AWS_SESSION_TOKEN=" + ("AQoD" * 19) + "AQo",
            "aws_session_token = ...",
        )
    )
    with TemporaryDirectory(prefix="jarvis-public-preflight-provider-near-miss-") as temp:
        root = Path(temp)
        _write(root, "provider-format-notes.md", near_misses + "\n")
        report = scan_release_tree(root)
        if not report.ok or report.findings:
            raise SystemExit(f"provider near-misses caused false positives: {report}")


def test_all_required_categories_fail_with_content_free_findings() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-findings-") as temp:
        root = Path(temp)
        _write(root, "CODEX_TASKS.md")
        _write(root, "com.jarvis-v3.worker.plist")
        _write(root, "source.py", "path = '" + "/" + "Users/example/private'\n")
        _write(root, "state/session.sqlite", "runtime")
        _write(root, "notes.txt", "owner=" + "private.person" + "@sample.org\n")
        _write(root, "fixture.txt", "key=" + "sk-" + ("Q" * 32) + "\n")
        _write(root, ".DS_Store", "finder metadata")
        _write(root, "artifacts/release.zip", "opaque archive fixture")
        _write(root, "artifacts/helper.dylib", "compiled fixture")
        _write(root, "artifacts/Synthetic.app/Contents/Info.plist", "application bundle fixture")
        _write_bytes(root, "artifacts/extensionless-binary", b"synthetic\x00binary")
        _write_bytes(root, "artifacts/invalid-utf8", b"synthetic\xffbinary")
        _write(root, ".netrc", "synthetic credential file")
        _write(root, ".envrc", "synthetic environment loader")
        _write(root, ".ssh/config", "synthetic SSH config")
        _write(root, ".git/config")

        first = scan_release_tree(root)
        second = scan_release_tree(root)
        if first != second or first.ok:
            raise SystemExit(f"release preflight was not deterministic: {first} / {second}")
        codes = {finding.code for finding in first.findings}
        required = {
            "absolute_users_path",
            "binary_content",
            "compiled_artifact",
            "credential_or_token_file",
            "filesystem_metadata",
            "likely_credential",
            "likely_email",
            "opaque_archive",
            "private_document",
            "runtime_artifact",
            "runtime_or_repository_directory",
            "service_plist",
        }
        if not required.issubset(codes):
            raise SystemExit(f"release preflight missed required categories: {required - codes}")
        rendered = _render_text(first)
        if "/" + "Users/example" in rendered or "private.person" in rendered or "Q" * 20 in rendered:
            raise SystemExit("release preflight printed private content instead of finding metadata")


def test_runtime_only_extra_deny_literals_are_hidden() -> None:
    private_literal = "synthetic-private-contact-marker"
    environ = {EXTRA_DENY_ENV: json.dumps([private_literal])}
    parsed = extra_deny_literals_from_env(environ)
    if parsed != (private_literal.encode("utf-8"),):
        raise SystemExit("runtime deny-list parsing drifted")

    with TemporaryDirectory(prefix="jarvis-public-preflight-deny-") as temp:
        root = Path(temp)
        _write(root, f"docs/{private_literal}.txt", "safe body\n")
        report = scan_release_tree(root, extra_deny_literals=parsed)
        rendered = _render_text(report)
        payload = json.dumps(report.json_payload(), sort_keys=True)
        summary_payload = json.dumps(report.summary_json_payload(), sort_keys=True)
        if report.ok or "extra_private_literal" not in {item.code for item in report.findings}:
            raise SystemExit(f"runtime private literal was not denied: {report}")
        if private_literal in rendered or private_literal in payload:
            raise SystemExit("runtime private deny literal leaked through the report")
        if private_literal in summary_payload or "docs/" in summary_payload:
            raise SystemExit("summary report leaked a private literal or candidate path")
        if report.summary_json_payload().get("paths_included") is not False:
            raise SystemExit("summary report did not explicitly declare that paths are omitted")
        if report.summary_json_payload().get("finding_counts_by_code") != {"extra_private_literal": 1}:
            raise SystemExit("summary report lost aggregate finding counts")
        if (
            report.summary_json_payload().get("extra_private_deny_literals_supplied") is not True
            or report.summary_json_payload().get("extra_private_deny_literal_count") != 1
        ):
            raise SystemExit("summary report lost content-free private deny-list evidence")
        if "<private>" not in rendered:
            raise SystemExit("private path segment was not visibly redacted")

    for invalid in ('{"not":"a-list"}', '[1]', '[""]', json.dumps(["\ud800"])):
        try:
            extra_deny_literals_from_env({EXTRA_DENY_ENV: invalid})
        except PreflightConfigError:
            pass
        else:
            raise SystemExit("invalid runtime deny-list configuration was accepted")


def test_private_deny_matching_covers_case_and_canonical_unicode_variants() -> None:
    latin_literal = "SyntheticPrivateAlias"
    hangul_nfc = "가나다"
    hangul_nfd = unicodedata.normalize("NFD", hangul_nfc)
    if hangul_nfc == hangul_nfd:
        raise SystemExit("canonical Hangul fixture did not decompose")

    cases = (
        (latin_literal, "prefix-sYnThEtIcPrIvAtEaLiAs-suffix", "latin-content.txt"),
        (hangul_nfc, f"prefix-{hangul_nfd}-suffix", "nfd-content.txt"),
        (hangul_nfd, f"prefix-{hangul_nfc}-suffix", "nfc-content.txt"),
    )
    for index, (literal, variant, filename) in enumerate(cases):
        parsed = extra_deny_literals_from_env(
            {EXTRA_DENY_ENV: json.dumps([literal], ensure_ascii=False)}
        )
        with TemporaryDirectory(
            prefix=f"jarvis-public-preflight-unicode-{index}-"
        ) as temp:
            root = Path(temp)
            _write(root, filename, variant + "\n")
            report = scan_release_tree(root, extra_deny_literals=parsed)
            if report.ok or "extra_private_literal" not in {
                item.code for item in report.findings
            }:
                raise SystemExit(f"private Unicode/case variant escaped: {report}")
            rendered = _render_text(report)
            payload = json.dumps(report.json_payload(), ensure_ascii=False, sort_keys=True)
            summary = json.dumps(
                report.summary_json_payload(), ensure_ascii=False, sort_keys=True
            )
            if any(value in rendered or value in payload or value in summary for value in (
                literal,
                variant,
                hangul_nfc,
                hangul_nfd,
            )):
                raise SystemExit("private Unicode/case variant leaked through a report")

    parsed = extra_deny_literals_from_env(
        {EXTRA_DENY_ENV: json.dumps([hangul_nfc], ensure_ascii=False)}
    )
    with TemporaryDirectory(prefix="jarvis-public-preflight-unicode-path-") as temp:
        root = Path(temp)
        _write(root, f"docs/{hangul_nfd}.txt", "safe body\n")
        report = scan_release_tree(root, extra_deny_literals=parsed)
        matching = [item for item in report.findings if item.code == "extra_private_literal"]
        if report.ok or not matching or any(item.path != "<private>" for item in matching):
            raise SystemExit(f"canonical private path was not fully redacted: {report}")
        output = _render_text(report) + json.dumps(
            report.json_payload(), ensure_ascii=False, sort_keys=True
        )
        if hangul_nfc in output or hangul_nfd in output:
            raise SystemExit("canonical private path leaked through output")


def test_private_variant_scan_is_bounded_and_binary_scan_stays_exact() -> None:
    literal = b"SyntheticPrivateMarker"
    malformed_content = b"\xff-prefix-" + literal + b"-suffix"
    with TemporaryDirectory(prefix="jarvis-public-preflight-binary-exact-") as temp:
        root = Path(temp)
        _write_bytes(root, "malformed.bin", malformed_content)
        report = scan_release_tree(root, extra_deny_literals=(literal,))
        codes = {item.code for item in report.findings}
        if not {"binary_content", "extra_private_literal"}.issubset(codes):
            raise SystemExit(f"malformed content lost exact-byte deny matching: {report}")

    with TemporaryDirectory(prefix="jarvis-public-preflight-variant-limit-") as temp:
        root = Path(temp)
        content = b"a" * (preflight_module.MAX_VARIANT_SCAN_INPUT_BYTES + 1)
        _write_bytes(root, "large-text.txt", content)
        normalized_input_lengths: list[int] = []
        real_normalize = preflight_module._nfkc_casefold

        def record_normalize(value: str) -> str:
            normalized_input_lengths.append(len(value))
            return real_normalize(value)

        with mock.patch.object(
            preflight_module,
            "_nfkc_casefold",
            side_effect=record_normalize,
        ):
            report = scan_release_tree(root, extra_deny_literals=(literal,))
        if report.ok or "private_variant_scan_limit" not in {
            item.code for item in report.findings
        }:
            raise SystemExit(f"over-limit private variant scan did not fail closed: {report}")
        if normalized_input_lengths and max(normalized_input_lengths) > len(literal.decode("utf-8")):
            raise SystemExit("over-limit candidate content reached Unicode normalization")

    with TemporaryDirectory(prefix="jarvis-public-preflight-no-deny-normalize-") as temp:
        root = Path(temp)
        _write(root, "safe.txt", "ordinary mixed Case 가나다 text\n")
        with mock.patch.object(
            preflight_module,
            "_nfkc_casefold",
            side_effect=AssertionError("preview scan normalized without a private deny-list"),
        ):
            report = scan_release_tree(root)
        if not report.ok:
            raise SystemExit("empty-deny preview behavior changed")


def test_scan_wide_file_and_variant_work_budgets_fail_closed() -> None:
    literal = b"SyntheticPrivateMarker"
    with TemporaryDirectory(prefix="jarvis-public-preflight-variant-total-") as temp:
        root = Path(temp)
        for index in range(5):
            _write(root, f"file-{index}.txt", f"ordinary variant-safe text {index}\n")
        real_content_findings = preflight_module._content_findings
        variant_permissions: list[bool] = []

        def record_variant_permission(content, literals, matcher=None, *, variant_scan_allowed=True):
            variant_permissions.append(variant_scan_allowed)
            return real_content_findings(
                content,
                literals,
                matcher,
                variant_scan_allowed=variant_scan_allowed,
            )

        with mock.patch.object(
            preflight_module,
            "MAX_VARIANT_SCAN_FILE_COUNT",
            2,
        ), mock.patch.object(
            preflight_module,
            "_content_findings",
            side_effect=record_variant_permission,
        ):
            report = scan_release_tree(root, extra_deny_literals=(literal,))
        if (
            report.ok
            or report.files_scanned != 5
            or variant_permissions != [True, True, False, False, False]
            or "private_variant_scan_limit" not in {
                item.code for item in report.findings
            }
        ):
            raise SystemExit(f"scan-wide variant budget drifted: {report} {variant_permissions}")

    with TemporaryDirectory(prefix="jarvis-public-preflight-file-total-") as temp:
        root = Path(temp)
        for index in range(5):
            _write(root, f"file-{index}.txt", f"ordinary safe text {index}\n")
        with mock.patch.object(preflight_module, "MAX_SCAN_FILES", 3):
            report = scan_release_tree(root)
        if (
            report.ok
            or report.files_scanned != 3
            or {item.code for item in report.findings} != {"scan_work_limit"}
            or {item.path for item in report.findings} != {"<bounded-scan>"}
        ):
            raise SystemExit(f"scan-wide file budget did not fail closed: {report}")

    with TemporaryDirectory(prefix="jarvis-public-preflight-directory-total-") as temp:
        root = Path(temp)
        for index in range(4):
            _write(root, f"entry-{index}.txt", "ordinary safe text\n")
        with mock.patch.object(preflight_module, "MAX_DIRECTORY_ENTRIES", 2):
            report = scan_release_tree(root)
        if (
            report.ok
            or report.files_scanned != 0
            or {item.code for item in report.findings} != {"scan_work_limit"}
            or {item.path for item in report.findings} != {"<bounded-scan>"}
        ):
            raise SystemExit(f"bounded directory enumeration drifted: {report}")


def test_oversized_and_unreadable_attempts_charge_scan_budget() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-attempt-oversized-") as temp:
        root = Path(temp)
        for index in range(5):
            _write_bytes(root, f"large-{index}.txt", b"12345678")
        read_calls = 0
        real_read = preflight_module._read_regular_at

        def count_reads(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return real_read(*args, **kwargs)

        with mock.patch.object(preflight_module, "MAX_SCAN_FILES", 3), mock.patch.object(
            preflight_module,
            "_read_regular_at",
            side_effect=count_reads,
        ):
            report = scan_release_tree(root, max_file_bytes=4)
        codes = {item.code for item in report.findings}
        if (
            report.ok
            or read_calls != 3
            or report.files_scanned != 0
            or not {"oversized_file", "scan_work_limit"}.issubset(codes)
        ):
            raise SystemExit(f"oversized attempts escaped scan budget: {report} {read_calls}")

    with TemporaryDirectory(prefix="jarvis-public-preflight-attempt-oversized-bytes-") as temp:
        root = Path(temp)
        for index in range(5):
            _write_bytes(root, f"large-{index}.txt", b"12345678")
        read_calls = 0
        real_read = preflight_module._read_regular_at

        def count_byte_budget_reads(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            return real_read(*args, **kwargs)

        with mock.patch.object(preflight_module, "MAX_SCAN_TOTAL_BYTES", 12), mock.patch.object(
            preflight_module,
            "_read_regular_at",
            side_effect=count_byte_budget_reads,
        ):
            report = scan_release_tree(root, max_file_bytes=4)
        codes = {item.code for item in report.findings}
        if (
            report.ok
            or read_calls != 2
            or report.files_scanned != 0
            or not {"oversized_file", "scan_work_limit"}.issubset(codes)
        ):
            raise SystemExit(
                f"oversized attempts escaped byte budget: {report} {read_calls}"
            )

    with TemporaryDirectory(prefix="jarvis-public-preflight-attempt-unreadable-") as temp:
        root = Path(temp)
        for index in range(5):
            _write(root, f"unreadable-{index}.txt", "ordinary safe text\n")
        read_calls = 0

        def unreadable(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            raise OSError("synthetic unreadable input")

        with mock.patch.object(preflight_module, "MAX_SCAN_FILES", 3), mock.patch.object(
            preflight_module,
            "_read_regular_at",
            side_effect=unreadable,
        ):
            report = scan_release_tree(root)
        codes = {item.code for item in report.findings}
        if (
            report.ok
            or read_calls != 3
            or report.files_scanned != 0
            or not {"unreadable_entry", "scan_work_limit"}.issubset(codes)
        ):
            raise SystemExit(f"unreadable attempts escaped scan budget: {report} {read_calls}")

    with TemporaryDirectory(prefix="jarvis-public-preflight-attempt-unreadable-bytes-") as temp:
        root = Path(temp)
        for index in range(5):
            _write_bytes(root, f"unreadable-{index}.txt", b"12345678")
        read_calls = 0

        def unreadable_with_byte_budget(*args, **kwargs):
            nonlocal read_calls
            read_calls += 1
            raise OSError("synthetic unreadable input")

        with mock.patch.object(preflight_module, "MAX_SCAN_TOTAL_BYTES", 17), mock.patch.object(
            preflight_module,
            "_read_regular_at",
            side_effect=unreadable_with_byte_budget,
        ):
            report = scan_release_tree(root)
        codes = {item.code for item in report.findings}
        if (
            report.ok
            or read_calls != 2
            or report.files_scanned != 0
            or not {"unreadable_entry", "scan_work_limit"}.issubset(codes)
        ):
            raise SystemExit(
                f"unreadable attempts escaped byte budget: {report} {read_calls}"
            )


def test_cli_output_never_echoes_invalid_hidden_configuration() -> None:
    from jarvis_v2.scripts import public_release_preflight

    hidden = "hidden-runtime-only-value"
    original = public_release_preflight.os.environ.get(EXTRA_DENY_ENV)
    public_release_preflight.os.environ[EXTRA_DENY_ENV] = json.dumps({"value": hidden})
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = public_release_preflight.main(["--root", "/definitely/not/a/tree"])
    finally:
        if original is None:
            public_release_preflight.os.environ.pop(EXTRA_DENY_ENV, None)
        else:
            public_release_preflight.os.environ[EXTRA_DENY_ENV] = original
    if code != 2 or hidden in output.getvalue() or "/definitely" in output.getvalue():
        raise SystemExit("invalid CLI configuration leaked hidden values or root path")


def test_summary_cli_omits_candidate_paths() -> None:
    from jarvis_v2.scripts import public_release_preflight

    with TemporaryDirectory(prefix="jarvis-public-preflight-summary-") as temp:
        root = Path(temp)
        private_path = "private-candidate-name.txt"
        _write(root, private_path, "/" + "Users/example/private\n")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = public_release_preflight.main(["--root", str(root), "--summary-json"])
        payload = json.loads(output.getvalue())
        if code != 1 or payload.get("ok") is not False:
            raise SystemExit("summary CLI did not preserve the failing preflight result")
        if private_path in output.getvalue() or str(root) in output.getvalue():
            raise SystemExit("summary CLI leaked a candidate path")
        if payload.get("finding_counts_by_code") != {"absolute_users_path": 1}:
            raise SystemExit("summary CLI lost aggregate finding counts")


def test_growing_file_read_is_bounded_and_fails_closed() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-growing-") as temp:
        root = Path(temp)
        target = root / "growing.txt"
        target.write_bytes(b"safe")
        real_read = os.read
        read_sizes: list[int] = []
        grew = False

        def grow_before_read(fd: int, size: int) -> bytes:
            nonlocal grew
            read_sizes.append(size)
            if not grew:
                grew = True
                with target.open("ab") as handle:
                    handle.write(b"private-growth")
            return real_read(fd, size)

        with mock.patch.object(preflight_module.os, "read", side_effect=grow_before_read):
            report = scan_release_tree(root, max_file_bytes=4)
        codes = {item.code for item in report.findings}
        if report.ok or "unreadable_entry" not in codes or "oversized_file" in codes:
            raise SystemExit(f"growing release file did not fail closed: {report}")
        if not read_sizes or max(read_sizes) > 5:
            raise SystemExit(f"release file read exceeded max+1 bound: {read_sizes}")


def test_stable_oversized_file_is_classified_without_full_read() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-oversized-") as temp:
        root = Path(temp)
        (root / "large.txt").write_bytes(b"12345678")
        real_read = os.read
        read_sizes: list[int] = []

        def record_read(fd: int, size: int) -> bytes:
            read_sizes.append(size)
            return real_read(fd, size)

        with mock.patch.object(preflight_module.os, "read", side_effect=record_read):
            report = scan_release_tree(root, max_file_bytes=4)
        codes = {item.code for item in report.findings}
        if report.ok or codes != {"oversized_file"}:
            raise SystemExit(f"stable oversized file was misclassified: {report}")
        if not read_sizes or max(read_sizes) > 5:
            raise SystemExit(f"oversized file read exceeded max+1 bound: {read_sizes}")


def test_leaf_symlink_swap_never_reads_outside_target() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-swap-") as temp:
        root = Path(temp)
        target = root / "safe.txt"
        outside = root.parent / f"{root.name}-outside.txt"
        target.write_bytes(b"safe")
        outside.write_bytes(b"outside-private-marker")
        real_open = os.open
        swapped = False

        def swap_before_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path == "safe.txt" and kwargs.get("dir_fd") is not None and not swapped:
                swapped = True
                target.unlink()
                target.symlink_to(outside)
            return real_open(path, flags, *args, **kwargs)

        try:
            with mock.patch.object(preflight_module.os, "open", side_effect=swap_before_open):
                report = scan_release_tree(root)
        finally:
            outside.unlink(missing_ok=True)
        if report.ok or "unreadable_entry" not in {item.code for item in report.findings}:
            raise SystemExit(f"symlink-swapped release file did not fail closed: {report}")


def test_directory_drift_produces_a_failing_finding() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-dir-drift-") as temp:
        root = Path(temp)
        _write(root, "package/module.py")
        real_names = preflight_module._bounded_directory_names
        changed = False

        def mutate_after_listing(path):
            nonlocal changed
            result = real_names(path)
            if not changed:
                changed = True
                _write(root, "late.txt")
            return result

        with mock.patch.object(
            preflight_module,
            "_bounded_directory_names",
            side_effect=mutate_after_listing,
        ):
            report = scan_release_tree(root)
        if report.ok or "unreadable_entry" not in {item.code for item in report.findings}:
            raise SystemExit(f"directory drift was not surfaced as a failure: {report}")


def test_earlier_file_mutation_during_later_scan_fails_end_revalidation() -> None:
    with TemporaryDirectory(prefix="jarvis-public-preflight-file-token-") as temp:
        root = Path(temp)
        first = root / "a.txt"
        first.write_bytes(b"safe-a")
        (root / "b.txt").write_bytes(b"safe-b")
        real_read = preflight_module._read_regular_at
        changed = False

        def mutate_earlier_after_later(parent_fd, name, **kwargs):
            nonlocal changed
            result = real_read(parent_fd, name, **kwargs)
            if name == "b.txt" and not changed:
                changed = True
                first.write_bytes(b"evil-a")
            return result

        with mock.patch.object(
            preflight_module,
            "_read_regular_at",
            side_effect=mutate_earlier_after_later,
        ):
            report = scan_release_tree(root)
        matching = {
            item.path for item in report.findings if item.code == "unreadable_entry"
        }
        if report.ok or "a.txt" not in matching:
            raise SystemExit(f"earlier-file mutation escaped end revalidation: {report}")


def main() -> None:
    test_clean_synthetic_tree_passes_without_mutation()
    test_extended_token_formats_fail_closed()
    test_high_confidence_provider_tokens_detect_without_generic_secret_heuristics()
    test_all_required_categories_fail_with_content_free_findings()
    test_runtime_only_extra_deny_literals_are_hidden()
    test_private_deny_matching_covers_case_and_canonical_unicode_variants()
    test_private_variant_scan_is_bounded_and_binary_scan_stays_exact()
    test_scan_wide_file_and_variant_work_budgets_fail_closed()
    test_oversized_and_unreadable_attempts_charge_scan_budget()
    test_cli_output_never_echoes_invalid_hidden_configuration()
    test_summary_cli_omits_candidate_paths()
    test_growing_file_read_is_bounded_and_fails_closed()
    test_stable_oversized_file_is_classified_without_full_read()
    test_leaf_symlink_swap_never_reads_outside_target()
    test_directory_drift_produces_a_failing_finding()
    test_earlier_file_mutation_during_later_scan_fails_end_revalidation()
    print("public release preflight smoke passed")


if __name__ == "__main__":
    main()
