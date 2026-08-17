from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.scripts.requirements_contract import (
    EXPECTED_DIRECT_DISTRIBUTIONS,
    RequirementsContractError,
    main as contract_main,
    parse_direct_requirements,
    production_third_party_imports,
    validate_requirements_contract,
)


ROOT = Path(__file__).resolve().parents[2]


def test_current_direct_dependency_and_python_contract_is_truthful_and_inert() -> None:
    tracked = (
        ROOT / "requirements.txt",
        ROOT / "QUICKSTART.md",
        ROOT / "PUBLIC_RELEASE_PRECHECK.md",
    )
    before = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked)
    report = validate_requirements_contract(ROOT)
    summary = report.summary_json_payload()
    if (
        not report.direct_distribution_count == len(EXPECTED_DIRECT_DISTRIBUTIONS)
        or summary.get("direct_declarations_valid") is not True
        or summary.get("direct_imports_declared") is not True
        or summary.get("deterministic_requirement_syntax") is not True
        or summary.get("supported_python_minimum") != "3.11"
        or summary.get("supported_python_contract_valid") is not True
        or summary.get("requirements_lock_present") is not False
        or summary.get("version_constraints_present") is not False
        or summary.get("reproducible_install") is not False
    ):
        raise SystemExit(f"dependency contract overstated or lost evidence: {summary}")
    for key in (
        "resolves_dependencies",
        "imports_installed_packages",
        "inspects_installed_versions",
        "network_access",
        "writes_files",
        "paths_included",
        "private_content_included",
    ):
        if summary.get(key) is not False:
            raise SystemExit(f"dependency contract boundary drifted: {summary}")
    after = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked)
    if after != before:
        raise SystemExit("dependency contract validation mutated public inputs")


def test_stdlib_ollama_transport_removes_sdk_closure_and_optional_adapters_stay_explicit() -> None:
    imports = set(production_third_party_imports(ROOT))
    required = {
        "certifi",
        "google",
        "google_auth_oauthlib",
        "googleapiclient",
        "pyautogui",
    }
    optional = {"AppKit", "Foundation", "faster_whisper", "objc", "whisper"}
    if not required.issubset(imports) or not optional.issubset(imports):
        raise SystemExit(f"production dependency import inventory drifted: {sorted(imports)}")
    if {"httpx", "ollama", "pydantic"}.intersection(imports):
        raise SystemExit(f"stdlib Ollama adapter retained an SDK dependency import: {sorted(imports)}")
    declarations = parse_direct_requirements((ROOT / "requirements.txt").read_bytes())
    if any(name in declarations for name in ("httpx", "ollama", "pydantic")):
        raise SystemExit("stdlib Ollama adapter retained an SDK dependency declaration")
    if any(name in declarations for name in ("faster-whisper", "openai-whisper", "pyobjc-core")):
        raise SystemExit("guarded optional adapters silently became core requirements")


def test_requirement_syntax_rejects_ambiguous_or_nondeterministic_forms() -> None:
    valid = (
        "# direct declarations\n"
        + "\n".join(EXPECTED_DIRECT_DISTRIBUTIONS)
        + "\n"
    ).encode("utf-8")
    if parse_direct_requirements(valid) != EXPECTED_DIRECT_DISTRIBUTIONS:
        raise SystemExit("canonical direct declaration syntax was rejected")

    invalid_variants = (
        valid.replace(b"pillow\n", b""),
        valid + b"pillow\n",
        valid.replace(b"pillow\n", b"pillow>=10\n"),
        valid.replace(b"pillow\n", b"pillow; python_version >= '3.11'\n"),
        valid.replace(b"pillow\n", b"pillow @ https://example.com/archive.whl\n"),
        b"-r requirements-extra.txt\n" + valid,
        valid.replace(b"pillow\npyautogui\n", b"pyautogui\npillow\n"),
        valid.replace(b"pillow\n", b" pillow\n"),
        valid.replace(b"pillow\n", b"Pillow\n"),
        valid.replace(b"\n", b"\r\n"),
    )
    for content in invalid_variants:
        try:
            parse_direct_requirements(content)
        except RequirementsContractError:
            pass
        else:
            raise SystemExit("ambiguous requirements syntax escaped the contract")


def test_unknown_production_import_fails_without_inspecting_installed_packages() -> None:
    with TemporaryDirectory(prefix="jarvis-requirements-contract-") as temp:
        root = Path(temp)
        package = root / "jarvis_v2"
        package.mkdir()
        (package / "module.py").write_text("import unknown_external_sdk\n", encoding="utf-8")
        (root / "requirements.txt").write_text(
            "\n".join(EXPECTED_DIRECT_DISTRIBUTIONS) + "\n",
            encoding="utf-8",
        )
        try:
            validate_requirements_contract(root)
        except RequirementsContractError as exc:
            if exc.code != "unclassified_third_party_import":
                raise SystemExit(f"unknown import failure drifted: {exc.code}")
        else:
            raise SystemExit("unknown production dependency import was accepted")


def test_symlinked_contract_inputs_are_refused_before_source_parsing() -> None:
    with TemporaryDirectory(prefix="jarvis-requirements-symlink-") as temp:
        root = Path(temp)
        package = root / "jarvis_v2"
        package.mkdir()
        outside = root / "outside-private.py"
        outside.write_text("PRIVATE_MARKER = 'must-not-be-imported'\n", encoding="utf-8")
        (package / "module.py").symlink_to(outside)
        requirements_target = root / "requirements-target.txt"
        requirements_target.write_text(
            "\n".join(EXPECTED_DIRECT_DISTRIBUTIONS) + "\n",
            encoding="utf-8",
        )
        (root / "requirements.txt").symlink_to(requirements_target)
        try:
            validate_requirements_contract(root)
        except RequirementsContractError as exc:
            if exc.code != "requirements_invalid":
                raise SystemExit(f"symlinked requirements failure drifted: {exc.code}")
        else:
            raise SystemExit("symlinked dependency contract inputs were accepted")


def test_cli_failure_is_path_free_and_docs_preserve_nonreproducibility_truth() -> None:
    hidden_root = "/private/tmp/private-dependency-contract-owner-marker"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = contract_main(["--root", hidden_root])
    rendered = output.getvalue()
    payload = json.loads(rendered)
    if code != 2 or payload.get("ok") is not False or hidden_root in rendered:
        raise SystemExit(f"dependency contract CLI failure leaked or overstated: {payload}")
    for key in ("network_access", "writes_files", "paths_included", "private_content_included"):
        if payload.get(key) is not False:
            raise SystemExit(f"dependency contract failure boundary drifted: {payload}")
    if Path(hidden_root).exists():
        raise SystemExit("dependency contract CLI created its supplied root")

    quickstart = " ".join((ROOT / "QUICKSTART.md").read_text(encoding="utf-8").split())
    precheck = " ".join(
        (ROOT / "PUBLIC_RELEASE_PRECHECK.md").read_text(encoding="utf-8").split()
    )
    for expected in (
        "jarvis_v2.scripts.requirements_contract",
        "direct declarations are valid",
        "does not make installation reproducible",
        "Do not guess package versions",
    ):
        if expected not in quickstart + " " + precheck:
            raise SystemExit(f"public dependency guidance missed truth boundary: {expected}")


def main() -> None:
    test_current_direct_dependency_and_python_contract_is_truthful_and_inert()
    test_stdlib_ollama_transport_removes_sdk_closure_and_optional_adapters_stay_explicit()
    test_requirement_syntax_rejects_ambiguous_or_nondeterministic_forms()
    test_unknown_production_import_fails_without_inspecting_installed_packages()
    test_symlinked_contract_inputs_are_refused_before_source_parsing()
    test_cli_failure_is_path_free_and_docs_preserve_nonreproducibility_truth()
    print("requirements contract smoke passed")


if __name__ == "__main__":
    main()
