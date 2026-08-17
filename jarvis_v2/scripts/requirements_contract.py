"""Read-only contract for Jarvis V3's direct dependency declarations.

The contract validates names, ordering, direct production imports, and the
documented Python floor.  It intentionally does not resolve packages, inspect
the active environment, contact an index, or invent version pins.  Until a
separately reviewed lock or constraints artifact exists, it reports that a
fresh install is not reproducible.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys

from jarvis_v2.scripts.v3_python_runtime import MINIMUM_V3_PYTHON


MAX_REQUIREMENTS_BYTES = 32 * 1024
MAX_PRODUCTION_PYTHON_FILES = 2_000
MAX_PRODUCTION_SOURCE_BYTES = 64 * 1024 * 1024
REQUIREMENT_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
EXPECTED_DIRECT_DISTRIBUTIONS = (
    "certifi",
    "google-api-python-client",
    "google-auth",
    "google-auth-httplib2",
    "google-auth-oauthlib",
    "pillow",
    "pyautogui",
)
REQUIRED_IMPORT_DISTRIBUTIONS = {
    "certifi": "certifi",
    "google": "google-auth",
    "google_auth_oauthlib": "google-auth-oauthlib",
    "googleapiclient": "google-api-python-client",
    "pyautogui": "pyautogui",
}
# These imports are guarded feature adapters with truthful fallback/setup
# behavior. They are not part of the fresh-install core dependency contract.
OPTIONAL_IMPORT_DISTRIBUTIONS = {
    "AppKit": "pyobjc-framework-Cocoa",
    "Foundation": "pyobjc-framework-Cocoa",
    "faster_whisper": "faster-whisper",
    "objc": "pyobjc-core",
    "whisper": "openai-whisper",
}
READINESS_ONLY_DISTRIBUTIONS = {
    "google-auth-httplib2",
    "pillow",
}


class RequirementsContractError(ValueError):
    """Content-free failure in the direct-dependency contract."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class RequirementsContractReport:
    direct_distribution_count: int
    required_import_module_count: int
    optional_import_module_count: int
    production_python_file_count: int
    requirements_lock_present: bool

    def summary_json_payload(self) -> dict[str, object]:
        return {
            "ok": True,
            "direct_declarations_valid": True,
            "direct_imports_declared": True,
            "deterministic_requirement_syntax": True,
            "supported_python_minimum": ".".join(str(part) for part in MINIMUM_V3_PYTHON),
            "supported_python_contract_valid": MINIMUM_V3_PYTHON == (3, 11),
            "direct_distribution_count": self.direct_distribution_count,
            "required_import_module_count": self.required_import_module_count,
            "optional_import_module_count": self.optional_import_module_count,
            "production_python_file_count": self.production_python_file_count,
            "requirements_lock_present": self.requirements_lock_present,
            "version_constraints_present": False,
            "reproducible_install": False,
            "resolves_dependencies": False,
            "imports_installed_packages": False,
            "inspects_installed_versions": False,
            "network_access": False,
            "writes_files": False,
            "paths_included": False,
            "private_content_included": False,
        }


def parse_direct_requirements(content: bytes) -> tuple[str, ...]:
    if not content or len(content) > MAX_REQUIREMENTS_BYTES or b"\x00" in content:
        raise RequirementsContractError("requirements_invalid")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequirementsContractError("requirements_invalid") from exc
    if "\r" in text or any(
        character not in "\n\t" and ord(character) < 32 for character in text
    ):
        raise RequirementsContractError("requirements_invalid")
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line != raw_line or not REQUIREMENT_NAME_RE.fullmatch(line):
            raise RequirementsContractError("requirements_syntax_invalid")
        names.append(line)
    if len(names) != len(set(names)):
        raise RequirementsContractError("requirements_duplicate")
    if tuple(names) != EXPECTED_DIRECT_DISTRIBUTIONS:
        raise RequirementsContractError("requirements_contract_mismatch")
    return tuple(names)


def _production_python_files(root: Path) -> tuple[Path, ...]:
    package = root / "jarvis_v2"
    if not package.is_dir() or package.is_symlink():
        raise RequirementsContractError("source_tree_invalid")
    files = [
        path
        for path in package.rglob("*.py")
        if not path.name.startswith(("smoke_test_", "live_test_"))
    ]
    files.extend(root.glob("launch_jarvis_v3*.py"))
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    if (
        not files
        or len(files) > MAX_PRODUCTION_PYTHON_FILES
        or any(not path.is_file() or path.is_symlink() for path in files)
    ):
        raise RequirementsContractError("source_tree_invalid")
    return tuple(files)


def production_third_party_imports(root: Path) -> tuple[str, ...]:
    imports: set[str] = set()
    total_bytes = 0
    for path in _production_python_files(root):
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RequirementsContractError("source_tree_invalid") from exc
        total_bytes += len(content)
        if total_bytes > MAX_PRODUCTION_SOURCE_BYTES:
            raise RequirementsContractError("source_tree_invalid")
        try:
            tree = ast.parse(content, filename="<public-source>")
        except (SyntaxError, ValueError) as exc:
            raise RequirementsContractError("production_source_invalid") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".", 1)[0])
    third_party = {
        name
        for name in imports
        if name not in sys.stdlib_module_names
        and name not in {"__future__", "jarvis_v2"}
    }
    return tuple(sorted(third_party))


def validate_requirements_contract(root: str | Path) -> RequirementsContractReport:
    source_root = Path(root)
    if not source_root.is_dir() or source_root.is_symlink():
        raise RequirementsContractError("source_tree_invalid")
    requirements_path = source_root / "requirements.txt"
    if not requirements_path.is_file() or requirements_path.is_symlink():
        raise RequirementsContractError("requirements_invalid")
    try:
        requirements_content = requirements_path.read_bytes()
    except OSError as exc:
        raise RequirementsContractError("requirements_invalid") from exc
    declared = parse_direct_requirements(requirements_content)
    imports = production_third_party_imports(source_root)
    known_imports = frozenset(REQUIRED_IMPORT_DISTRIBUTIONS) | frozenset(
        OPTIONAL_IMPORT_DISTRIBUTIONS
    )
    if not frozenset(imports).issubset(known_imports):
        raise RequirementsContractError("unclassified_third_party_import")
    required_distributions = {
        REQUIRED_IMPORT_DISTRIBUTIONS[module]
        for module in imports
        if module in REQUIRED_IMPORT_DISTRIBUTIONS
    } | READINESS_ONLY_DISTRIBUTIONS
    if not required_distributions.issubset(declared):
        raise RequirementsContractError("direct_import_requirement_missing")
    if frozenset(declared) != frozenset(EXPECTED_DIRECT_DISTRIBUTIONS):
        raise RequirementsContractError("requirements_contract_mismatch")
    if MINIMUM_V3_PYTHON != (3, 11):
        raise RequirementsContractError("supported_python_contract_drift")
    return RequirementsContractReport(
        direct_distribution_count=len(declared),
        required_import_module_count=sum(
            module in REQUIRED_IMPORT_DISTRIBUTIONS for module in imports
        ),
        optional_import_module_count=sum(
            module in OPTIONAL_IMPORT_DISTRIBUTIONS for module in imports
        ),
        production_python_file_count=len(_production_python_files(source_root)),
        requirements_lock_present=(source_root / "requirements.lock").is_file(),
    )


def _failure_payload(code: str) -> dict[str, object]:
    return {
        "ok": False,
        "error_code": code,
        "network_access": False,
        "writes_files": False,
        "paths_included": False,
        "private_content_included": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate direct requirement declarations without resolving or installing packages."
    )
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        report = validate_requirements_contract(args.root)
    except RequirementsContractError as exc:
        print(json.dumps(_failure_payload(exc.code), sort_keys=True, separators=(",", ":")))
        return 2
    except (OSError, UnicodeError, ValueError):
        print(
            json.dumps(
                _failure_payload("requirements_validation_failed"),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(report.summary_json_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
