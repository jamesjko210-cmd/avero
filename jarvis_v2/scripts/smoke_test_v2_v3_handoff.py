from __future__ import annotations

import re
from pathlib import Path

from jarvis_v2.scripts.public_release_candidate import structural_public_candidate_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_OPERATIONAL_REFERENCES = (
    "AUGUST_15_FUNCTIONAL_PREVIEW.md",
    "V3_BACKLOG_PRIVATE.md",
)
IS_PUBLIC_CANDIDATE = structural_public_candidate_profile(REPO_ROOT) is not None


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def _private_reference(name: str) -> str | None:
    path = REPO_ROOT / name
    if IS_PUBLIC_CANDIDATE:
        if path.exists():
            raise SystemExit(f"public candidate retained private operational reference: {name}")
        return None
    return path.read_text(encoding="utf-8")


def test_generation_boundary_is_explicit() -> None:
    readme = _read("README.md")
    version = _read("VERSION").strip()
    manifest = _read("MIGRATION_MANIFEST.md")
    normalized_manifest = " ".join(manifest.split())
    backlog = _private_reference("V3_BACKLOG_PRIVATE.md")

    if version != "4.0.0-rc.1":
        raise SystemExit(f"unexpected current release version: {version!r}")
    for expected in (
        "Jarvis V4",
        "V4 is the public release identity of the frozen and tested V3 implementation",
        "New feature development after this release belongs to V5",
        "V2 is frozen",
        "internal Python package remains `jarvis_v2` temporarily",
        "AUGUST_15_FUNCTIONAL_PREVIEW.md",
    ):
        if expected not in readme:
            raise SystemExit(f"README missed V4 release boundary: {expected}")
    for expected in (
        "V3 Git repository preserves that local lineage and has no remote configured",
        "Populated environment files and credentials",
        "Existing V2 dashboard and messaging services still point to the V2 workspace",
        "V3 service cutover requires a separate approval",
    ):
        if expected not in normalized_manifest:
            raise SystemExit(f"migration manifest missed safety boundary: {expected}")
    if backlog is not None:
        for expected in (
            "Status: **active V3 input**",
            "waiting until August 8 or keeping V3 inactive are superseded",
            "safety contracts",
            "HyperResearch",
        ):
            if expected not in backlog:
                raise SystemExit(f"V3 backlog missed retained handoff context: {expected}")


def test_august_preview_gate_is_complete() -> None:
    gate = _private_reference("AUGUST_15_FUNCTIONAL_PREVIEW.md")
    if gate is None:
        return
    for expected in (
        "functional V3 preview, not an automatic production cutover",
        "V3 uses only V3 environment",
        "complete isolated regression suite",
        "CLI chat and the local dashboard/HUD",
        "10-or-more-turn mixed session",
        "Approval binding, argument contracts",
        "Supervised proofs that do not run unattended",
        "Fresh seven-consecutive-day V3 scheduler proof",
    ):
        if expected not in gate:
            raise SystemExit(f"August 15 preview gate missed: {expected}")
    if re.search(r"/(?:Users|private|var/folders)/", gate):
        raise SystemExit("August 15 preview gate contains a local absolute path")
    if re.search(r"(?im)^\s*(?:JARVIS_[A-Z0-9_]*TOKEN|TELEGRAM_BOT_TOKEN)\s*=\s*\S+", gate):
        raise SystemExit("August 15 preview gate contains a credential assignment")


def test_private_v3_canonical_references_exist() -> None:
    for relative_path in (
        "MIGRATION_MANIFEST.md",
        "V3_DEVELOPMENT.md",
        "CAPABILITIES.md",
        "README.md",
        "VERSION",
    ):
        if not (REPO_ROOT / relative_path).is_file():
            raise SystemExit(f"V3 canonical reference is missing: {relative_path}")
    if IS_PUBLIC_CANDIDATE:
        for relative_path in PRIVATE_OPERATIONAL_REFERENCES:
            if (REPO_ROOT / relative_path).exists():
                raise SystemExit(
                    f"public candidate retained private operational reference: {relative_path}"
                )
    else:
        for relative_path in PRIVATE_OPERATIONAL_REFERENCES:
            if not (REPO_ROOT / relative_path).is_file():
                raise SystemExit(f"V3 canonical reference is missing: {relative_path}")


def main() -> None:
    test_generation_boundary_is_explicit()
    test_august_preview_gate_is_complete()
    test_private_v3_canonical_references_exist()
    print("V4 release-boundary handoff smoke passed")


if __name__ == "__main__":
    main()
