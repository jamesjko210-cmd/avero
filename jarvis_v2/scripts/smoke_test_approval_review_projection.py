from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
from jarvis_v2.scripts.test_runtime import make_temp_runtime


def _other_runtime(root: Path) -> JarvisRuntime:
    return JarvisRuntime(
        JarvisConfig(
            data_dir=root / "other-data",
            db_path=root / "other.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            use_model_planner=False,
            watched_dirs=(root / "Other Watched",),
        )
    )


def _seed_approval(runtime: JarvisRuntime, *, session: str = "approval-review-smoke") -> int:
    return runtime.store.add_pending_approval(
        session,
        "run command python3 --version",
        "run_shell_command",
        "explicit approval required",
        planned_args={"command": "python3 --version"},
    )


def test_owned_projection_and_generic_note_guard() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-owned-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        _seed_approval(runtime)
        tool = runtime.registry.get("save_approval_review").handler
        result = tool({"limit": 20})
        if not result.ok:
            raise SystemExit(f"owned Approval Review export failed: {result}")
        path = Path(result.metadata["path"])
        text = path.read_text(encoding="utf-8")
        if "jarvis_projection: approval_review" not in text or "# Approval Review" not in text:
            raise SystemExit("Approval Review missed ownership or exact content evidence")
        for forbidden_key in ("content_sha256", "source_revision", "write_atomic", "snapshot_consistency"):
            if forbidden_key in result.metadata:
                raise SystemExit(f"Approval Review changed protected metadata with {forbidden_key}: {result.metadata}")

        note_tool = runtime.registry.get("write_jarvis_note").handler
        before = path.read_bytes()
        for mode in ("append", "create"):
            refused = note_tool(
                {"path": "Automations/Approval Review", "body": "must not alter approval review", "mode": mode}
            )
            if refused.ok or refused.metadata.get("reason") != "managed_projection":
                raise SystemExit(f"generic note writer accepted Approval Review in {mode}: {refused}")
        if path.read_bytes() != before:
            raise SystemExit("generic note refusal changed Approval Review")

        pending_path = runtime.vault.root_path / "Automations" / "Pending Approvals.md"
        pending_before = pending_path.read_bytes()
        foreign = _other_runtime(root)
        _seed_approval(foreign, session="foreign-approval-review")
        try:
            foreign.registry.get("save_approval_review").handler({})
        except FileExistsError:
            pass
        else:
            raise SystemExit("a second database overwrote the owned Approval Review")
        if pending_path.read_bytes() != pending_before:
            raise SystemExit("foreign Approval Review refusal changed Pending Approvals")


def test_legacy_migration_and_unowned_note_refusal() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-legacy-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        _seed_approval(runtime)
        path = runtime.vault.root_path / "Automations" / "Approval Review.md"
        path.write_text(
            "# Approval Review\n\nUpdated: 2026-07-11 12:00\n\nApproval review:\n"
            "- No pending approvals.\n"
            "- Keep using `autonomy plan: ...`, `privacy report`, and `safety status` before risky work.\n",
            encoding="utf-8",
        )
        migrated = runtime.registry.get("save_approval_review").handler({})
        if not migrated.ok or "jarvis_projection: approval_review" not in path.read_text(encoding="utf-8"):
            raise SystemExit("exact legacy Approval Review did not migrate to owned projection")

    with TemporaryDirectory(prefix="jarvis-approval-review-unowned-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_approval(runtime)
        path = runtime.vault.root_path / "Automations" / "Approval Review.md"
        sentinel = "# My Approval Notes\n\nDo not replace this personal note.\n"
        path.write_text(sentinel, encoding="utf-8")
        try:
            runtime.registry.get("save_approval_review").handler({})
        except FileExistsError:
            pass
        else:
            raise SystemExit("unowned Approval Review note was accepted")
        if path.read_text(encoding="utf-8") != sentinel:
            raise SystemExit("unowned Approval Review note was changed")

    with TemporaryDirectory(prefix="jarvis-approval-review-spoof-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_approval(runtime)
        path = runtime.vault.root_path / "Automations" / "Approval Review.md"
        spoof = "# Approval Review\n\nUpdated: 2026-07-11 12:00\n\nApproval review:\n- personal text\n"
        path.write_text(spoof, encoding="utf-8")
        try:
            runtime.registry.get("save_approval_review").handler({})
        except FileExistsError:
            pass
        else:
            raise SystemExit("near-match Approval Review spoof was accepted as legacy")
        if path.read_text(encoding="utf-8") != spoof:
            raise SystemExit("near-match Approval Review spoof was changed")


def test_symlink_and_parent_symlink_refusal() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-symlink-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        _seed_approval(runtime)
        outside = root / "outside-review.md"
        sentinel = "outside approval review sentinel\n"
        outside.write_text(sentinel, encoding="utf-8")
        review = runtime.vault.root_path / "Automations" / "Approval Review.md"
        review.symlink_to(outside)
        try:
            runtime.registry.get("save_approval_review").handler({})
        except (OSError, ValueError):
            pass
        else:
            raise SystemExit("Approval Review destination symlink was accepted")
        if outside.read_text(encoding="utf-8") != sentinel:
            raise SystemExit("Approval Review destination symlink changed its external target")

    with TemporaryDirectory(prefix="jarvis-approval-review-parent-link-") as temp:
        root = Path(temp)
        runtime = make_temp_runtime(root)
        _seed_approval(runtime)
        automations = runtime.vault.root_path / "Automations"
        for child in automations.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
        automations.rmdir()
        outside_dir = root / "outside-automations"
        outside_dir.mkdir()
        automations.symlink_to(outside_dir, target_is_directory=True)
        try:
            runtime.registry.get("save_approval_review").handler({})
        except (OSError, ValueError):
            pass
        else:
            raise SystemExit("Approval Review parent-directory symlink was accepted")
        if list(outside_dir.iterdir()):
            raise SystemExit("Approval Review parent-directory symlink wrote outside the vault")


def test_failure_preserves_prior_note_and_concurrency_is_complete() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-failure-") as temp:
        runtime = make_temp_runtime(Path(temp))
        _seed_approval(runtime)
        tool = runtime.registry.get("save_approval_review").handler
        first = tool({})
        path = Path(first.metadata["path"])
        sentinel = path.read_bytes()

        from jarvis_v2.memory import obsidian as obsidian_module

        original_replace = obsidian_module._replace_text

        def fail_review(root, target, content, **kwargs):
            if target.name == "Approval Review.md":
                raise OSError("simulated approval review fsync failure")
            return original_replace(root, target, content, **kwargs)

        with patch("jarvis_v2.memory.obsidian._replace_text", side_effect=fail_review):
            try:
                tool({})
            except OSError:
                pass
            else:
                raise SystemExit("Approval Review durable-write failure was hidden")
        if path.read_bytes() != sentinel:
            raise SystemExit("failed Approval Review publication changed the prior note")

        runtime.store.set_pending_approval_status(1, "dismissed")
        _seed_approval(runtime, session="second-approval-review")
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending_future = pool.submit(tool, {"status": "pending"})
            dismissed_future = pool.submit(tool, {"status": "dismissed"})
            results = [pending_future.result(timeout=10), dismissed_future.result(timeout=10)]
        final_text = path.read_text(encoding="utf-8")
        if any(not result.ok for result in results):
            raise SystemExit(f"concurrent Approval Review export failed: {results}")
        if "jarvis_projection: approval_review" not in final_text or "# Approval Review" not in final_text:
            raise SystemExit("concurrent Approval Review left an incomplete projection")
        if not (
            "Approval #1: run_shell_command" in final_text
            or "Approval #2: run_shell_command" in final_text
        ):
            raise SystemExit("concurrent Approval Review did not match either complete writer")


def test_bounded_review_does_not_truncate_pending_mirror() -> None:
    with TemporaryDirectory(prefix="jarvis-approval-review-complete-pending-") as temp:
        runtime = make_temp_runtime(Path(temp))
        for index in range(21):
            _seed_approval(runtime, session=f"approval-review-{index}")
        tool = runtime.registry.get("save_approval_review").handler
        bounded = tool({"limit": 1, "status": "pending"})
        review_text = Path(bounded.metadata["path"]).read_text(encoding="utf-8")
        pending_text = Path(bounded.metadata["pending_approvals_path"]).read_text(encoding="utf-8")
        if review_text.count("Approval #") != 1:
            raise SystemExit("bounded Approval Review did not honor its display limit")
        if pending_text.count("## Approval #") != 21:
            raise SystemExit("bounded Approval Review truncated the canonical Pending Approvals mirror")

        for approval_id in range(1, 4):
            runtime.store.set_pending_approval_status(approval_id, "approved")
        approved = tool({"limit": 2, "status": "approved"})
        approved_text = Path(approved.metadata["path"]).read_text(encoding="utf-8")
        pending_text = Path(approved.metadata["pending_approvals_path"]).read_text(encoding="utf-8")
        if approved_text.count("Approval #") != 2:
            raise SystemExit("approved-status Approval Review missed approved rows")
        if pending_text.count("## Approval #") != 18 or "## Approval #1 " in pending_text:
            raise SystemExit("approved-status review polluted or truncated the pending-only mirror")


def main() -> None:
    test_owned_projection_and_generic_note_guard()
    test_legacy_migration_and_unowned_note_refusal()
    test_symlink_and_parent_symlink_refusal()
    test_failure_preserves_prior_note_and_concurrency_is_complete()
    test_bounded_review_does_not_truncate_pending_mirror()
    print("Approval Review projection smoke passed")


if __name__ == "__main__":
    main()
