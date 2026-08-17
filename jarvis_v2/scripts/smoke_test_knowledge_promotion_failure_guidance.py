"""Offline proof for knowledge-promotion failure guidance and replay fencing."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.store import MemoryStore
from jarvis_v2.scripts.error_guidance_inventory import build_failure_inventory
from jarvis_v2.tools.knowledge_promotion import (
    COMMITTED_PROMOTION_RECOVERY_ACTION,
    PROMOTION_REVIEW_RECOVERY_ACTION,
    PROMOTION_UNKNOWN_RECOVERY_ACTION,
    _committed_promotion_failure,
    make_knowledge_promotion_tools,
    make_profile_promotion_tools,
)


KNOWN_REFUSAL_FIELDS = {
    "state_changed": False,
    "promotion_committed": False,
}
UNKNOWN_FIELDS = {
    "outcome_known": False,
    "outcome_unknown": True,
    "execution_outcome_unknown": True,
    "side_effect_possible": True,
    "retry_safe": False,
    "automatic_retry_allowed": False,
    "authorizes_retry": False,
}
COMMITTED_FIELDS = {
    "outcome_known": True,
    "outcome_unknown": False,
    "execution_outcome_unknown": False,
    "side_effect_possible": True,
    "retry_safe": False,
    "automatic_retry_allowed": False,
    "authorizes_retry": False,
    "state_changed": True,
    "promotion_committed": True,
}
PRIVATE_MARKERS = (
    "/\x55sers/",
    "/private/",
    "/tmp/",
    "sk_" + "live_",
    "traceback",
)


def _assert_guidance(result, *, action: str, fields: dict[str, bool], label: str) -> None:
    if result.ok or action not in result.output:
        raise SystemExit(f"{label} did not expose its recovery action: {result}")
    if result.metadata.get("recovery_guidance") != {
        "version": 1,
        "action": action,
        "commands": [],
    }:
        raise SystemExit(f"{label} recovery declaration drifted: {result.metadata}")
    for key, expected in fields.items():
        if result.metadata.get(key) is not expected:
            raise SystemExit(f"{label} field {key} drifted: {result.metadata}")
    public = f"{result.output}\n{result.metadata}".lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in public:
            raise SystemExit(f"{label} leaked private failure detail: {public}")


def _raise_storage_failure(*_args, **_kwargs):
    raise RuntimeError("private storage detail must not escape")


def main() -> None:
    with TemporaryDirectory(prefix="jarvis-knowledge-guidance-") as temp:
        root = Path(temp)
        store = MemoryStore(root / "store.sqlite")
        store.init()
        vault = ObsidianVault(root / "vault")
        vault.init()
        packet, decision, _decision_resolver, preference, _preference_resolver = (
            make_knowledge_promotion_tools(store, vault)
        )
        profile, _profile_resolver = make_profile_promotion_tools(store, vault)

        refusals = (
            (packet({}), "packet input refusal"),
            (decision({}), "decision input refusal"),
            (preference({}), "preference input refusal"),
            (profile({}), "profile input refusal"),
        )
        for result, label in refusals:
            _assert_guidance(
                result,
                action=PROMOTION_REVIEW_RECOVERY_ACTION,
                fields=KNOWN_REFUSAL_FIELDS,
                label=label,
            )

        store.promote_memory_to_decision_exact = _raise_storage_failure  # type: ignore[method-assign]
        store.promote_memory_to_preference_exact = _raise_storage_failure  # type: ignore[method-assign]
        store.promote_memory_to_profile_note_exact = _raise_storage_failure  # type: ignore[method-assign]
        binding = "a" * 64
        unknowns = (
            (
                decision(
                    {
                        "memory_id": 1,
                        "reviewed_revision": 1,
                        "review_binding": binding,
                        "title": "Reviewed title",
                        "rationale": "Reviewed rationale",
                        "impact": "Reviewed impact",
                        "target_revision": 1,
                        "target_binding": binding,
                    }
                ),
                "decision unknown outcome",
            ),
            (
                preference(
                    {
                        "memory_id": 1,
                        "reviewed_revision": 1,
                        "review_binding": binding,
                        "category": "interaction",
                        "key": "tone",
                        "value": "concise",
                        "target_revision": 1,
                        "target_binding": binding,
                    }
                ),
                "preference unknown outcome",
            ),
            (
                profile(
                    {
                        "memory_id": 1,
                        "reviewed_revision": 1,
                        "review_binding": binding,
                        "heading": "Reviewed identity",
                        "category": "identity",
                        "body": "Reviewed identity detail.",
                        "target_revision": 1,
                        "target_binding": binding,
                    }
                ),
                "profile unknown outcome",
            ),
        )
        for result, label in unknowns:
            _assert_guidance(
                result,
                action=PROMOTION_UNKNOWN_RECOVERY_ACTION,
                fields=UNKNOWN_FIELDS,
                label=label,
            )
            if result.metadata.get("promotion_outcome") != "unknown":
                raise SystemExit(f"{label} lost its unknown-outcome marker: {result.metadata}")

        for tool_name in (
            "promote_memory_to_decision",
            "promote_memory_to_preference",
            "promote_memory_to_profile",
        ):
            committed = _committed_promotion_failure(
                tool_name,
                "The promotion committed, but projection verification remains pending.",
                {
                    "promotion_committed": True,
                    "state_changed": True,
                    "projection_pending": True,
                    "memory_id": 1,
                },
            )
            _assert_guidance(
                committed,
                action=COMMITTED_PROMOTION_RECOVERY_ACTION,
                fields=COMMITTED_FIELDS,
                label=f"{tool_name} committed failure",
            )

    inventory = build_failure_inventory(Path(__file__).resolve().parents[2])
    promotion_sites = [
        site
        for site in inventory.legacy_failures
        if site.path == "jarvis_v2/tools/knowledge_promotion.py"
    ]
    if len(promotion_sites) != 5 or any(
        site.guidance_kind != "declare_failure_guidance"
        for site in promotion_sites
    ):
        raise SystemExit(
            "knowledge-promotion failure declarations drifted: "
            f"{promotion_sites!r}"
        )

    print(
        "Knowledge-promotion failure-guidance smoke passed: "
        "4 refusals, 3 unknown outcomes, and 3 committed replay fences."
    )


if __name__ == "__main__":
    main()
