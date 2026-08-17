from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

from jarvis_v2.automations.jobs import build_daily_brief
from jarvis_v2.memory.obsidian import ObsidianVault
from jarvis_v2.memory.skill_projection import reconcile_skill_projection
from jarvis_v2.memory.store import MemoryRecord, MemoryStore, SkillRecord
from jarvis_v2.scripts.test_runtime import make_temp_runtime
from jarvis_v2.tools import conversation as conversation_tools


DRAFT_BODY = """> Review status: DRAFT. Human review is required before this skill is treated as reliable.

This is a reviewable skill draft inferred from a Jarvis session. It may help repeat a workflow, but it must be edited by the operator before promotion.

## Human Review Gate

- Review this draft.

## Source Session Signals

- quasarworkflow
"""


def _legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                trigger TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO skills(name, trigger, body, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, '2026-07-11T00:00:00Z', '2026-07-11T00:00:00Z')
            """,
            (
                (
                    "Historical Session Draft",
                    "quasarworkflow",
                    DRAFT_BODY,
                    "draft,session,human-review-required",
                ),
                (
                    "Historical Active Skill",
                    "activeworkflow",
                    "reviewed procedure",
                    "reviewed",
                ),
                (
                    "Tag Only Is Active",
                    "tagonly",
                    "ordinary procedure",
                    "draft,session,human-review-required",
                ),
            ),
        )


def _assert_migration(root: Path) -> None:
    database = root / "legacy.sqlite"
    _legacy_database(database)
    store = MemoryStore(database)
    store.init()
    statuses = {
        row["name"]: row["review_status"]
        for row in store.list_skills(limit=10)
    }
    if statuses != {
        "Historical Session Draft": "draft",
        "Historical Active Skill": "active",
        "Tag Only Is Active": "active",
    }:
        raise SystemExit(f"legacy skill review migration was unsafe: {statuses}")
    store.init()
    if store.get_skill("Historical Session Draft")["review_status"] != "draft":
        raise SystemExit("skill review migration was not idempotent")

    with store.connect() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            "UPDATE skills SET review_status = 'untrusted' WHERE name = 'Historical Active Skill'"
        )
    try:
        store.init()
    except RuntimeError:
        pass
    else:
        raise SystemExit("invalid skill review status did not fail startup closed")


def _assert_concurrent_final_state(root: Path) -> None:
    database = root / "race.sqlite"
    first = MemoryStore(database)
    first.init()
    second = MemoryStore(database)
    second.init()
    vault = ObsidianVault(root / "race-vault")
    vault.init()
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def save_active() -> None:
        barrier.wait()
        outcomes.append(
            first.save_skill_with_projection_target(
                SkillRecord("Race Skill", "active", "reviewed winner", "reviewed")
            )
        )

    def save_draft() -> None:
        barrier.wait()
        try:
            outcomes.append(
                second.save_skill_draft_with_projection_target(
                    SkillRecord(
                        "Race Skill",
                        "draft",
                        DRAFT_BODY,
                        "draft,session,human-review-required",
                    )
                )
            )
        except ValueError:
            outcomes.append("active_conflict")

    threads = [threading.Thread(target=save_active), threading.Thread(target=save_draft)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    if any(thread.is_alive() for thread in threads):
        raise SystemExit("concurrent draft/active save did not finish")
    row = first.get_skill("Race Skill")
    if row is None or row["review_status"] != "active" or row["body"] != "reviewed winner":
        raise SystemExit(f"concurrent draft save demoted reviewed behavior: {dict(row or {})}")
    job = first.get_skill_projection_job(int(row["id"]))
    outcome = reconcile_skill_projection(
        first,
        vault,
        int(row["id"]),
        expected_operation=str(job["operation"]),
        expected_revision=int(job["skill_revision"]),
        expected_source_digest=str(job["source_digest"]),
    )
    if outcome.status != "completed":
        raise SystemExit(f"concurrent skill projection did not converge: {outcome}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-skill-draft-") as temp:
        root = Path(temp)
        _assert_migration(root)
        _assert_concurrent_final_state(root)

        runtime = make_temp_runtime(root / "runtime")
        runtime.store.log_message(
            runtime.session_id,
            "user",
            "summarize the quasarworkflow and keep it review-only",
        )
        draft_name = "Quasar Review Draft"
        draft = runtime.registry.get("draft_skill_from_session").handler(
            {"name": draft_name, "trigger": "quasarworkflow", "limit": 20}
        )
        row = runtime.store.get_skill(draft_name)
        if (
            not draft.ok
            or row is None
            or row["review_status"] != "draft"
            or draft.metadata.get("behaviorally_active") is not False
            or draft.metadata.get("authorizes_skill_activation") is not False
        ):
            raise SystemExit(f"session draft did not persist as review-only: {draft}")
        if runtime.store.search_active_skills("quasarworkflow", limit=5):
            raise SystemExit("draft leaked through the active skill search API")
        if not runtime.store.search_skills("quasarworkflow", limit=5):
            raise SystemExit("draft was not visible through the review search API")

        listed = runtime.registry.get("list_skills").handler({})
        shown = runtime.registry.get("get_skill").handler({"name": draft_name})
        context = runtime.registry.get("chat_context").handler({"prompt": "quasarworkflow"})
        match = runtime.registry.get("skill_match_preview").handler(
            {"request": "quasarworkflow"}
        )
        rehearsal = runtime.registry.get("assistant_turn_rehearsal").handler(
            {"message": "talk about quasarworkflow"}
        )
        brief = build_daily_brief(runtime.store, runtime.vault)
        if "[DRAFT]" not in listed.output or "Review status: DRAFT" not in shown.output:
            raise SystemExit("review commands did not expose the draft state")
        if shown.metadata.get("behaviorally_active") is not False:
            raise SystemExit("get skill did not report the draft as behaviorally inactive")
        for label, text in (
            ("chat context", context.output),
            ("match preview", match.output),
            ("assistant rehearsal", rehearsal.output),
            ("daily brief", brief),
        ):
            if draft_name in text:
                raise SystemExit(f"draft leaked into {label}: {text}")
        if runtime.chat.preview_loop("quasarworkflow")["skills"] != 0:
            raise SystemExit("draft leaked into ChatBrain grounding")

        before_redraft = runtime.store.get_skill_by_identity(draft_name)
        redraft = runtime.registry.get("draft_skill_from_session").handler(
            {"name": draft_name, "trigger": "replacement", "limit": 20}
        )
        after_redraft = runtime.store.get_skill_by_identity(draft_name)
        if (
            redraft.ok
            or redraft.metadata.get("reason") != "skill_name_conflict"
            or after_redraft["body"] != before_redraft["body"]
            or int(after_redraft["revision"]) != int(before_redraft["revision"])
        ):
            raise SystemExit("re-drafting silently replaced the prior review draft")

        runtime.store.add_memory(
            MemoryRecord(
                category="feedback",
                title="Draft-only learning queue proof",
                body="A review-only draft must not count as an active learned skill.",
            )
        )
        learning_queue = runtime.registry.get("queue_learning_tasks").handler({})
        if "drafting one reviewed skill" not in learning_queue.output.lower():
            raise SystemExit("a review-only draft suppressed active-skill learning work")

        linked_draft_name = "Karpathy Goal-Driven Execution"
        linked_draft = runtime.registry.get("draft_skill_from_session").handler(
            {"name": linked_draft_name, "trigger": "linkedreviewworkflow", "limit": 20}
        )
        linked_install = runtime.registry.get("install_linked_skills").handler(
            {"source": "andrej-karpathy-skills", "limit": 1}
        )
        linked_row = runtime.store.get_skill_by_identity(linked_draft_name)
        if (
            not linked_draft.ok
            or linked_install.ok
            or linked_install.metadata.get("reason") != "linked_skill_review_required"
            or linked_install.metadata.get("writes_skills") is not False
            or linked_install.metadata.get("blocked_skill_count") != 1
            or linked_row["review_status"] != "draft"
        ):
            raise SystemExit("linked skill installation bypassed explicit draft promotion")

        raced_linked_name = "Hermes Learning Loop"
        original_active_save = runtime.store.save_skill_with_projection_target
        inserted_race_draft = False

        def insert_draft_before_linked_save(record, **kwargs):
            nonlocal inserted_race_draft
            if record.name == raced_linked_name and not inserted_race_draft:
                inserted_race_draft = True
                runtime.store.save_skill_draft_with_projection_target(
                    SkillRecord(
                        raced_linked_name,
                        "linkedraceworkflow",
                        DRAFT_BODY,
                        "draft,session,human-review-required",
                    )
                )
            return original_active_save(record, **kwargs)

        runtime.store.save_skill_with_projection_target = insert_draft_before_linked_save  # type: ignore[method-assign]
        raced_linked_install = runtime.registry.get("install_linked_skills").handler(
            {"source": "hermes-agent", "limit": 1}
        )
        runtime.store.save_skill_with_projection_target = original_active_save  # type: ignore[method-assign]
        raced_linked_row = runtime.store.get_skill_by_identity(raced_linked_name)
        if (
            raced_linked_install.ok
            or raced_linked_install.metadata.get("reason") != "linked_skill_review_required"
            or raced_linked_install.metadata.get("writes_skills") is not False
            or raced_linked_row["review_status"] != "draft"
            or runtime.store.search_active_skills("linkedraceworkflow", limit=5)
        ):
            raise SystemExit("linked install check/save race activated a concurrent draft")
        raced_linked_projection = reconcile_skill_projection(
            runtime.store, runtime.vault, int(raced_linked_row["id"])
        )
        if raced_linked_projection.status != "completed":
            raise SystemExit("concurrent linked draft projection did not converge")

        original_write = runtime.vault.write_skill_with_evidence

        def fail_projection(*args, **kwargs):
            raise OSError("injected draft projection failure")

        runtime.vault.write_skill_with_evidence = fail_projection  # type: ignore[method-assign]
        pending_name = "Pending Review Draft"
        pending_draft = runtime.registry.get("draft_skill_from_session").handler(
            {"name": pending_name, "trigger": "pendingdraftworkflow", "limit": 20}
        )
        runtime.vault.write_skill_with_evidence = original_write  # type: ignore[method-assign]
        pending_row = runtime.store.get_skill(pending_name)
        pending_job = runtime.store.get_skill_projection_job(int(pending_row["id"]))
        if (
            not pending_draft.ok
            or pending_draft.metadata.get("skill_projection_pending") is not True
            or pending_row["review_status"] != "draft"
            or pending_job["state"] != "pending"
            or runtime.store.search_active_skills("pendingdraftworkflow", limit=5)
        ):
            raise SystemExit("failed draft projection did not remain durable and inactive")
        repaired = reconcile_skill_projection(runtime.store, runtime.vault, int(pending_row["id"]))
        if repaired.status != "completed":
            raise SystemExit(f"pending draft projection did not repair: {repaired}")

        protected = runtime.registry.get("save_skill").handler(
            {
                "name": "Protected Active Skill",
                "trigger": "protectedworkflow",
                "body": "keep reviewed behavior",
            }
        )
        protected_row = runtime.store.get_skill("Protected Active Skill")
        before_revision = int(protected_row["revision"])
        conflict = runtime.registry.get("draft_skill_from_session").handler(
            {"name": "Protected Active Skill", "trigger": "replacement", "limit": 20}
        )
        preserved = runtime.store.get_skill("Protected Active Skill")
        if (
            not protected.ok
            or conflict.ok
            or conflict.metadata.get("reason") != "skill_name_conflict"
            or preserved["body"] != "keep reviewed behavior"
            or preserved["review_status"] != "active"
            or int(preserved["revision"]) != before_revision
        ):
            raise SystemExit("draft creation overwrote an active skill")

        for variant in ("protected active skill", "Ｐrotected Active Skill"):
            variant_conflict = runtime.registry.get("draft_skill_from_session").handler(
                {"name": variant, "trigger": "replacement", "limit": 20}
            )
            if variant_conflict.ok or variant_conflict.metadata.get("reason") != "skill_name_conflict":
                raise SystemExit(f"logical skill-name variant bypassed draft protection: {variant!r}")
        protected_identity_rows = [
            row
            for row in runtime.store.list_skills(limit=100)
            if "protected active skill" in str(row["name"]).casefold()
        ]
        if len(protected_identity_rows) != 1:
            raise SystemExit("logical skill-name variants created duplicate rows")

        guarded_name = "Projection Guard Draft"
        guarded = runtime.registry.get("draft_skill_from_session").handler(
            {"name": guarded_name, "trigger": "projectionguardworkflow", "limit": 20}
        )
        if not guarded.ok:
            raise SystemExit(f"projection-guard draft setup failed: {guarded}")
        runtime.vault.write_skill_with_evidence = fail_projection  # type: ignore[method-assign]
        guarded_promotion = runtime.registry.get("save_skill").handler(
            {
                "name": guarded_name,
                "trigger": "projectionguardworkflow",
                "body": "Reviewed projection guard procedure.",
                "tags": "reviewed",
            }
        )
        runtime.vault.write_skill_with_evidence = original_write  # type: ignore[method-assign]
        guarded_row = runtime.store.get_skill_by_identity(guarded_name)
        if (
            guarded_promotion.ok
            or guarded_promotion.metadata.get("reason") != "skill_promotion_projection_pending"
            or guarded_promotion.metadata.get("behaviorally_active") is not False
            or guarded_row["review_status"] != "draft"
            or runtime.store.search_active_skills("projectionguardworkflow", limit=5)
        ):
            raise SystemExit("failed promotion projection activated a review draft")
        guarded_repair = reconcile_skill_projection(
            runtime.store, runtime.vault, int(guarded_row["id"])
        )
        if guarded_repair.status != "completed":
            raise SystemExit(f"guarded promotion projection did not repair: {guarded_repair}")
        if runtime.store.get_skill_by_identity(guarded_name)["review_status"] != "draft":
            raise SystemExit("background projection repair activated a review draft")
        guarded_retry = runtime.registry.get("save_skill").handler(
            {
                "name": guarded_name,
                "trigger": "projectionguardworkflow",
                "body": "Reviewed projection guard procedure.",
                "tags": "reviewed",
            }
        )
        if not guarded_retry.ok or runtime.store.get_skill_by_identity(guarded_name)["review_status"] != "active":
            raise SystemExit("reviewed promotion retry did not activate after projection success")

        concurrent_name = "Concurrent Receipt Draft"
        real_reconcile = conversation_tools.reconcile_skill_projection
        superseded = False

        def supersede_draft(store, vault, skill_id, **kwargs):
            nonlocal superseded
            if not superseded:
                superseded = True
                store.save_skill_with_projection_target(
                    SkillRecord(
                        concurrent_name,
                        "concurrentreceiptworkflow",
                        "Concurrent reviewed procedure.",
                        "reviewed",
                    )
                )
            return real_reconcile(store, vault, skill_id, **kwargs)

        with patch.object(
            conversation_tools,
            "reconcile_skill_projection",
            side_effect=supersede_draft,
        ):
            concurrent_receipt = runtime.registry.get("draft_skill_from_session").handler(
                {
                    "name": concurrent_name,
                    "trigger": "concurrentreceiptworkflow",
                    "limit": 20,
                }
            )
        concurrent_row = runtime.store.get_skill_by_identity(concurrent_name)
        if (
            concurrent_receipt.ok
            or concurrent_receipt.metadata.get("reason") != "skill_draft_superseded"
            or concurrent_receipt.metadata.get("review_status") != "active"
            or concurrent_receipt.metadata.get("behaviorally_active") is not True
            or concurrent_row["review_status"] != "active"
        ):
            raise SystemExit("superseded draft handler returned a false inactive receipt")
        concurrent_projection = reconcile_skill_projection(
            runtime.store, runtime.vault, int(concurrent_row["id"])
        )
        if concurrent_projection.status != "completed":
            raise SystemExit("superseding active skill projection did not converge")

        replacement_name = "Replacement Receipt Draft"
        replacement_done = False

        def replace_draft_identity(store, vault, skill_id, **kwargs):
            nonlocal replacement_done
            if not replacement_done:
                replacement_done = True
                original = store.get_skill_by_id(skill_id)
                deleted = store.delete_skill_exact(
                    int(original["id"]),
                    int(original["revision"]),
                    str(original["name"]),
                )
                if deleted.status != "deleted":
                    raise RuntimeError("replacement race could not delete the draft")
                store.save_skill_with_projection_target(
                    SkillRecord(
                        replacement_name,
                        "replacementreceiptworkflow",
                        "Replacement reviewed procedure.",
                        "reviewed",
                    )
                )
            return real_reconcile(store, vault, skill_id, **kwargs)

        with patch.object(
            conversation_tools,
            "reconcile_skill_projection",
            side_effect=replace_draft_identity,
        ):
            replacement_receipt = runtime.registry.get("draft_skill_from_session").handler(
                {
                    "name": replacement_name,
                    "trigger": "replacementreceiptworkflow",
                    "limit": 20,
                }
            )
        replacement_row = runtime.store.get_skill_by_identity(replacement_name)
        if (
            replacement_receipt.ok
            or replacement_receipt.metadata.get("reason") != "skill_draft_superseded"
            or replacement_receipt.metadata.get("review_status") != "active"
            or replacement_receipt.metadata.get("behaviorally_active") is not True
            or replacement_row["review_status"] != "active"
        ):
            raise SystemExit("replacement-ID supersession returned a false inactive receipt")
        replacement_projection = reconcile_skill_projection(
            runtime.store, runtime.vault, int(replacement_row["id"])
        )
        if replacement_projection.status != "completed":
            raise SystemExit("replacement active skill projection did not converge")

        promoted = runtime.registry.get("save_skill").handler(
            {
                "name": draft_name,
                "trigger": "quasarworkflow",
                "body": "Use the reviewed quasar workflow and verify the result.",
                "tags": "reviewed",
            }
        )
        promoted_row = runtime.store.get_skill(draft_name)
        if (
            not promoted.ok
            or promoted_row["review_status"] != "active"
            or promoted_row["origin"] != "session_history"
            or promoted.metadata.get("promoted_from_draft") is not True
            or promoted.metadata.get("review_status") != "active"
        ):
            raise SystemExit(f"explicit save did not promote the reviewed draft: {promoted}")
        if runtime.chat.preview_loop("quasarworkflow")["skills"] != 0:
            raise SystemExit("activated session-derived skill entered ChatBrain grounding")

        rewritten = runtime.registry.get("save_skill").handler(
            {
                "name": draft_name,
                "trigger": "quasarworkflow",
                "body": "Marker-free rewrite of the reviewed quasar procedure.",
                "tags": "reviewed",
            }
        )
        rewritten_row = runtime.store.get_skill(draft_name)
        if (
            not rewritten.ok
            or rewritten_row["review_status"] != "active"
            or rewritten_row["origin"] != "session_history"
            or "Source Session Signals" in rewritten_row["body"]
            or "session-derived" in rewritten_row["tags"]
            or runtime.chat.preview_loop("quasarworkflow")["skills"] != 0
        ):
            raise SystemExit("activation or marker removal laundered session-derived origin")
        promoted_match = runtime.registry.get("skill_match_preview").handler(
            {"request": "quasarworkflow"}
        )
        if draft_name not in promoted_match.output:
            raise SystemExit("promoted skill did not enter behavioral match preview")

        user_skill_name = "Durable User Grounding Skill"
        user_skill_marker = "DURABLE_USER_SKILL_GROUNDING_7f29c1"
        user_skill = runtime.registry.get("save_skill").handler(
            {
                "name": user_skill_name,
                "trigger": "durableoriginworkflow",
                "body": f"Use the user-authored procedure. {user_skill_marker}",
                "tags": "reviewed",
            }
        )
        user_skill_row = runtime.store.get_skill(user_skill_name)
        if (
            not user_skill.ok
            or user_skill_row is None
            or user_skill_row["origin"] != "user_authored"
        ):
            raise SystemExit("genuinely user-authored skill lost durable origin")

        runtime.chat.provider = "openai"
        runtime.chat.allow_remote_personal_context = True
        if not runtime._refresh_history_policy_binding():
            raise SystemExit("could not activate consented durable provider epoch")
        active_epoch = runtime.store.get_active_history_policy_epoch()
        current_decision = runtime.chat._history_policy_decision(commit=False)
        if (
            active_epoch["epoch_id"] != runtime._history_policy_epoch_id
            or active_epoch["policy_fingerprint"] != current_decision.fingerprint
            or active_epoch["provider"] != current_decision.provider
            or active_epoch["destination_class"] != current_decision.destination_class
            or active_epoch["explicit_consent_satisfied"] is not True
        ):
            raise SystemExit("consented provider test did not hold the exact durable epoch")

        provider_prompt = "Apply durableoriginworkflow while considering quasarworkflow."
        with patch(
            "jarvis_v2.agent.chat.generate_model_text",
            return_value="durable grounding answer",
        ) as consented_provider:
            runtime.chat.respond(provider_prompt)
        consented_payload = "\n".join(
            str(message.get("content", ""))
            for message in consented_provider.call_args.kwargs["messages"]
        )
        if (
            user_skill_marker not in consented_payload
            or draft_name in consented_payload
            or "Marker-free rewrite of the reviewed quasar procedure." in consented_payload
            or runtime.chat.last_turn_metadata.get("history_disclosure_prepare_status")
            != "prepared"
            or runtime.chat.last_turn_metadata.get("history_disclosure_finalize_status")
            != "confirmed"
        ):
            raise SystemExit(
                "fresh durable consent did not admit only the user-authored skill"
            )

        runtime.chat.allow_remote_personal_context = False
        if not runtime._refresh_history_policy_binding():
            raise SystemExit("could not rotate to a durable no-consent epoch")
        if runtime.store.get_active_history_policy_epoch()[
            "explicit_consent_satisfied"
        ] is not False:
            raise SystemExit("durable no-consent epoch incorrectly recorded consent")
        with patch(
            "jarvis_v2.agent.chat.generate_model_text",
            return_value="no-consent answer",
        ) as no_consent_provider:
            runtime.chat.respond(provider_prompt)
        no_consent_payload = "\n".join(
            str(message.get("content", ""))
            for message in no_consent_provider.call_args.kwargs["messages"]
        )
        if (
            user_skill_marker in no_consent_payload
            or runtime.chat.last_turn_metadata.get(
                "stored_personal_context_policy_satisfied"
            )
            is not False
        ):
            raise SystemExit("user-authored skill bypassed durable consent")

        runtime.chat.allow_remote_personal_context = True
        if not runtime._refresh_history_policy_binding():
            raise SystemExit("could not restore the consented durable provider epoch")
        current_decision = runtime.chat._history_policy_decision(commit=False)
        stale_epoch = runtime.store.start_history_policy_epoch(
            policy_fingerprint=current_decision.fingerprint,
            provider=current_decision.provider,
            model_identifier=current_decision.model,
            destination_class=current_decision.destination_class,
            session_generation=runtime.history_session_generation,
            explicit_consent_satisfied=True,
        )
        if stale_epoch == runtime._history_policy_epoch_id:
            raise SystemExit("durable epoch rotation did not stale the runtime callbacks")
        with patch(
            "jarvis_v2.agent.chat.generate_model_text",
            return_value="stale-epoch answer",
        ) as stale_provider:
            runtime.chat.respond(provider_prompt)
        stale_payload = "\n".join(
            str(message.get("content", ""))
            for message in stale_provider.call_args.kwargs["messages"]
        )
        if (
            user_skill_marker in stale_payload
            or runtime.chat.last_turn_metadata.get("history_disclosure_prepare_status")
            != "failed_stripped"
        ):
            raise SystemExit(
                "stale durable callbacks disclosed a user-authored skill: "
                f"marker_present={user_skill_marker in stale_payload}, "
                f"prepare={runtime.chat.last_turn_metadata.get('history_disclosure_prepare_status')!r}"
            )

    print("Skill draft activation smoke passed")


if __name__ == "__main__":
    main()
