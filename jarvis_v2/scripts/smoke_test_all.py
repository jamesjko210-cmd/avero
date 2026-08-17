from __future__ import annotations

import atexit
import ast
import fcntl
import os
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from importlib.util import find_spec
from math import isfinite
from pathlib import Path

from jarvis_v2.scripts.v3_python_runtime import V3_INHERITED_CUSTODY_KEYS


TEST_MODULES = [
    "jarvis_v2.scripts.smoke_test_core",
    "jarvis_v2.scripts.smoke_test_planner_hang_guard",
    "jarvis_v2.scripts.smoke_test_help",
    "jarvis_v2.scripts.smoke_test_status_server_core",
    "jarvis_v2.scripts.smoke_test_architecture",
    "jarvis_v2.scripts.smoke_test_executor",
    "jarvis_v2.scripts.smoke_test_executor_failure_guidance",
    "jarvis_v2.scripts.smoke_test_error_guidance_inventory",
    "jarvis_v2.scripts.smoke_test_public_error_egress_inventory",
    "jarvis_v2.scripts.smoke_test_public_release_preflight",
    "jarvis_v2.scripts.smoke_test_public_release_candidate",
    "jarvis_v2.scripts.smoke_test_public_release_handoff",
    "jarvis_v2.scripts.smoke_test_independent_credential_scan",
    "jarvis_v2.scripts.smoke_test_requirements_contract",
    "jarvis_v2.scripts.smoke_test_dashboard_json_error_guidance",
    "jarvis_v2.scripts.smoke_test_dashboard_client_error_recovery",
    "jarvis_v2.scripts.smoke_test_terminal_error_guidance",
    "jarvis_v2.scripts.smoke_test_v3_command_guidance",
    "jarvis_v2.scripts.smoke_test_local_read_error_guidance",
    "jarvis_v2.scripts.smoke_test_offline_utility_error_guidance",
    "jarvis_v2.scripts.smoke_test_tool_argument_contracts",
    "jarvis_v2.scripts.smoke_test_scheduler_manual_run_approval",
    "jarvis_v2.scripts.smoke_test_registry_failure_guidance",
    "jarvis_v2.scripts.smoke_test_email_read_argument_contracts",
    "jarvis_v2.scripts.smoke_test_task_read_argument_contracts",
    "jarvis_v2.scripts.smoke_test_personal_knowledge_read_argument_contracts",
    "jarvis_v2.scripts.smoke_test_learning_review_argument_contract",
    "jarvis_v2.scripts.smoke_test_get_person_selector_privacy",
    "jarvis_v2.scripts.smoke_test_note_read_argument_contracts",
    "jarvis_v2.scripts.smoke_test_skill_read_argument_contracts",
    "jarvis_v2.scripts.smoke_test_skill_failure_guidance",
    "jarvis_v2.scripts.smoke_test_auto_mutation_receipts",
    "jarvis_v2.scripts.smoke_test_auto_mutation_runtime",
    "jarvis_v2.scripts.smoke_test_auto_mutation_task_goal_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_task_status_fencing",
    "jarvis_v2.scripts.smoke_test_auto_mutation_task_completion_evidence",
    "jarvis_v2.scripts.smoke_test_auto_mutation_create_goal_preflight",
    "jarvis_v2.scripts.smoke_test_auto_mutation_goal_target_preflight",
    "jarvis_v2.scripts.smoke_test_auto_mutation_task_details_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_task_import_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_learning_queue_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_export_goal",
    "jarvis_v2.scripts.smoke_test_auto_mutation_export_tasks",
    "jarvis_v2.scripts.smoke_test_auto_mutation_local_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_feedback_projection_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_profile_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_note_write_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_ingest_rollout",
    "jarvis_v2.scripts.smoke_test_auto_mutation_reconciliation",
    "jarvis_v2.scripts.smoke_test_approval_execution_integrity",
    "jarvis_v2.scripts.smoke_test_audit_approval_chain",
    "jarvis_v2.scripts.smoke_test_prototype_readiness",
    "jarvis_v2.scripts.smoke_test_harness",
    "jarvis_v2.scripts.smoke_test_harness_failure_guidance",
    "jarvis_v2.scripts.smoke_test_roadmap",
    "jarvis_v2.scripts.smoke_test_v2_v3_handoff",
    "jarvis_v2.scripts.smoke_test_model_status",
    "jarvis_v2.scripts.smoke_test_model_status_failure_guidance",
    "jarvis_v2.scripts.smoke_test_model_provider",
    "jarvis_v2.scripts.smoke_test_ollama_destination",
    "jarvis_v2.scripts.smoke_test_ollama_probe",
    "jarvis_v2.scripts.smoke_test_model_planner",
    "jarvis_v2.scripts.smoke_test_functions",
    "jarvis_v2.scripts.smoke_test_system_failure_guidance",
    "jarvis_v2.scripts.smoke_test_utilities_failure_guidance",
    "jarvis_v2.scripts.smoke_test_private_app_approval_boundary",
    "jarvis_v2.scripts.smoke_test_skills",
    "jarvis_v2.scripts.smoke_test_skill_projection_reconciliation",
    "jarvis_v2.scripts.smoke_test_skill_draft_activation",
    "jarvis_v2.scripts.smoke_test_skill_delete_approval_binding",
    "jarvis_v2.scripts.smoke_test_memory_approval_binding",
    "jarvis_v2.scripts.smoke_test_memory_curator_failure_guidance",
    "jarvis_v2.scripts.smoke_test_clear_inbox_approval_binding",
    "jarvis_v2.scripts.smoke_test_knowledge_promotion",
    "jarvis_v2.scripts.smoke_test_knowledge_promotion_preference",
    "jarvis_v2.scripts.smoke_test_knowledge_promotion_profile",
    "jarvis_v2.scripts.smoke_test_knowledge_promotion_adversarial",
    "jarvis_v2.scripts.smoke_test_knowledge_promotion_failure_guidance",
    "jarvis_v2.scripts.smoke_test_memory_projection_reconciliation",
    "jarvis_v2.scripts.smoke_test_ingested_source_projection_store",
    "jarvis_v2.scripts.smoke_test_decision_projection_reconciliation",
    "jarvis_v2.scripts.smoke_test_decision_outcomes",
    "jarvis_v2.scripts.smoke_test_decision_error_guidance",
    "jarvis_v2.scripts.smoke_test_preference_projection_reconciliation",
    "jarvis_v2.scripts.smoke_test_decision_preference_mutation_atomicity",
    "jarvis_v2.scripts.smoke_test_person_mutation_atomicity",
    "jarvis_v2.scripts.smoke_test_person_projection_reconciliation",
    "jarvis_v2.scripts.smoke_test_person_identity_ownership",
    "jarvis_v2.scripts.smoke_test_agi_slice",
    "jarvis_v2.scripts.smoke_test_subagent_fleet",
    "jarvis_v2.scripts.smoke_test_subagent_assignment_store",
    "jarvis_v2.scripts.smoke_test_subagent_runner",
    "jarvis_v2.scripts.smoke_test_capabilities",
    "jarvis_v2.scripts.smoke_test_goals",
    "jarvis_v2.scripts.smoke_test_goal_failure_guidance",
    "jarvis_v2.scripts.smoke_test_goal_projection_ownership",
    "jarvis_v2.scripts.smoke_test_goal_projection_reconciliation",
    "jarvis_v2.scripts.smoke_test_tasks",
    "jarvis_v2.scripts.smoke_test_tasks_failure_guidance",
    "jarvis_v2.scripts.smoke_test_task_note_preview_error_guidance",
    "jarvis_v2.scripts.smoke_test_task_remaining_error_guidance",
    "jarvis_v2.scripts.smoke_test_decisions",
    "jarvis_v2.scripts.smoke_test_daily_plan",
    "jarvis_v2.scripts.smoke_test_auto_mutation_daily_plan_rollout",
    "jarvis_v2.scripts.smoke_test_daily_plan_source_snapshot",
    "jarvis_v2.scripts.smoke_test_auto_mutation_goal_nudge_rollout",
    "jarvis_v2.scripts.smoke_test_daily_brief_source_snapshot",
    "jarvis_v2.scripts.smoke_test_auto_mutation_daily_brief_rollout",
    "jarvis_v2.scripts.smoke_test_conversation",
    "jarvis_v2.scripts.smoke_test_continuity",
    "jarvis_v2.scripts.smoke_test_build_progress",
    "jarvis_v2.scripts.smoke_test_focus",
    "jarvis_v2.scripts.smoke_test_next_step",
    "jarvis_v2.scripts.smoke_test_organize",
    "jarvis_v2.scripts.smoke_test_auto_mutation_organize_rollout",
    "jarvis_v2.scripts.smoke_test_safety",
    "jarvis_v2.scripts.smoke_test_approval_review",
    "jarvis_v2.scripts.smoke_test_approval_review_projection",
    "jarvis_v2.scripts.smoke_test_privacy",
    "jarvis_v2.scripts.smoke_test_command_privacy_routing",
    "jarvis_v2.scripts.smoke_test_autonomy_plan",
    "jarvis_v2.scripts.smoke_test_autonomy_error_guidance",
    "jarvis_v2.scripts.smoke_test_action_rehearsal",
    "jarvis_v2.scripts.smoke_test_assistant_turn_rehearsal",
    "jarvis_v2.scripts.smoke_test_personal",
    "jarvis_v2.scripts.smoke_test_personal_error_guidance",
    "jarvis_v2.scripts.smoke_test_computer_plan",
    "jarvis_v2.scripts.smoke_test_computer_oav",
    "jarvis_v2.scripts.smoke_test_computer_error_guidance",
    "jarvis_v2.scripts.smoke_test_files",
    "jarvis_v2.scripts.smoke_test_file_failure_guidance",
    "jarvis_v2.scripts.smoke_test_ingest_argument_contracts",
    "jarvis_v2.scripts.smoke_test_ingest",
    "jarvis_v2.scripts.smoke_test_browser_basics",
    "jarvis_v2.scripts.smoke_test_browser_error_guidance",
    "jarvis_v2.scripts.smoke_test_notes",
    "jarvis_v2.scripts.smoke_test_note_error_guidance",
    "jarvis_v2.scripts.smoke_test_obsidian_exports",
    "jarvis_v2.scripts.smoke_test_proactive",
    "jarvis_v2.scripts.smoke_test_scheduler_basics",
    "jarvis_v2.scripts.smoke_test_scheduler_failure_guidance",
    "jarvis_v2.scripts.smoke_test_goal_nudge_routing",
    "jarvis_v2.scripts.smoke_test_goal_nudge_source_snapshot",
    "jarvis_v2.scripts.smoke_test_scheduler_run_history_custody",
    "jarvis_v2.scripts.smoke_test_morning_brief_scheduler",
    "jarvis_v2.scripts.smoke_test_scheduler_delivery_outbox",
    "jarvis_v2.scripts.smoke_test_scheduler_note_publication_custody",
    "jarvis_v2.scripts.smoke_test_scheduler_pause_inflight",
    "jarvis_v2.scripts.smoke_test_state_snapshot_scheduler_receipts",
    "jarvis_v2.scripts.smoke_test_conversation_compaction",
    "jarvis_v2.scripts.smoke_test_compaction_history_provenance",
    "jarvis_v2.scripts.smoke_test_profile",
    "jarvis_v2.scripts.smoke_test_profile_custody",
    "jarvis_v2.scripts.smoke_test_profile_projection_recovery",
    "jarvis_v2.scripts.smoke_test_profile_vault_safety",
    "jarvis_v2.scripts.smoke_test_profile_direct_read_custody",
    "jarvis_v2.scripts.smoke_test_profile_mutation_guards",
    "jarvis_v2.scripts.smoke_test_profile_generic_memory_isolation",
    "jarvis_v2.scripts.smoke_test_profile_ownership_egress_fence",
    "jarvis_v2.scripts.smoke_test_profile_protected_paths",
    "jarvis_v2.scripts.smoke_test_preferences",
    "jarvis_v2.scripts.smoke_test_preference_error_guidance",
    "jarvis_v2.scripts.smoke_test_preference_integrity",
    "jarvis_v2.scripts.smoke_test_people",
    "jarvis_v2.scripts.smoke_test_people_error_guidance",
    "jarvis_v2.scripts.smoke_test_memory_stats",
    "jarvis_v2.scripts.smoke_test_store_connections",
    "jarvis_v2.scripts.smoke_test_history_provenance_store",
    "jarvis_v2.scripts.smoke_test_approval_governor_word_boundary",
    "jarvis_v2.scripts.smoke_test_capability_cockpit",
    "jarvis_v2.scripts.smoke_test_calendar_auth",
    "jarvis_v2.scripts.smoke_test_calendar_readonly_auth",
    "jarvis_v2.scripts.smoke_test_operator_evals",
    "jarvis_v2.scripts.smoke_test_brain",
    "jarvis_v2.scripts.smoke_test_brain_error_guidance",
    "jarvis_v2.scripts.smoke_test_feedback",
    "jarvis_v2.scripts.smoke_test_feedback_input_error_guidance",
    "jarvis_v2.scripts.smoke_test_learning_review",
    "jarvis_v2.scripts.smoke_test_generated_report_projections",
    "jarvis_v2.scripts.smoke_test_audit",
    "jarvis_v2.scripts.smoke_test_audit_failure_guidance",
    "jarvis_v2.scripts.smoke_test_approvals",
    "jarvis_v2.scripts.smoke_test_approval_failure_guidance",
    "jarvis_v2.scripts.smoke_test_shell",
    "jarvis_v2.scripts.smoke_test_shell_failure_guidance",
    "jarvis_v2.scripts.smoke_test_voice",
    "jarvis_v2.scripts.smoke_test_voice_input_error_guidance",
    "jarvis_v2.scripts.smoke_test_voice_operational_failure_guidance",
    "jarvis_v2.scripts.smoke_test_voice_failure_guidance_full",
    "jarvis_v2.scripts.smoke_test_state",
    "jarvis_v2.scripts.smoke_test_state_projection_integrity",
    "jarvis_v2.scripts.smoke_test_storage",
    "jarvis_v2.scripts.smoke_test_doctor",
    "jarvis_v2.scripts.smoke_test_status_config",
    "jarvis_v2.scripts.smoke_test_setup_status_env",
    "jarvis_v2.scripts.smoke_test_setup_calendar_auth_guidance",
    "jarvis_v2.scripts.smoke_test_setup_status_auth",
    "jarvis_v2.scripts.smoke_test_copy_status_auth",
    "jarvis_v2.scripts.smoke_test_smoke_runner",
    "jarvis_v2.scripts.smoke_test_bootstrap_startup",
    "jarvis_v2.scripts.smoke_test_startup_recovery_audit",
    "jarvis_v2.scripts.smoke_test_startup_recovery_launchers",
    "jarvis_v2.scripts.smoke_test_run_scheduler_startup",
    "jarvis_v2.scripts.smoke_test_v3_scheduler_bootstrap",
    "jarvis_v2.scripts.smoke_test_v3_daemon_gate",
    "jarvis_v2.scripts.smoke_test_v3_active_launchagent_contracts",
    "jarvis_v2.scripts.smoke_test_v3_coexistence_preflight",
    "jarvis_v2.scripts.smoke_test_v3_runtime_isolation",
    "jarvis_v2.scripts.smoke_test_v3_preview_closure",
    "jarvis_v2.scripts.smoke_test_v3_clean_restart_persistence",
    "jarvis_v2.scripts.smoke_test_v3_recovery_receipt",
    "jarvis_v2.scripts.smoke_test_v3_model_chat_proof",
    "jarvis_v2.scripts.smoke_test_v3_mixed_session_proof",
    "jarvis_v2.scripts.smoke_test_v3_python_runtime",
    "jarvis_v2.scripts.smoke_test_daemon_reconstruction",
    "jarvis_v2.scripts.smoke_test_launcher_discovery",
    "jarvis_v2.scripts.smoke_test_env_loader",
    "jarvis_v2.scripts.smoke_test_live_check",
    "jarvis_v2.scripts.smoke_test_chat",
    "jarvis_v2.scripts.smoke_test_chat_preview_boundary",
    "jarvis_v2.scripts.smoke_test_chat_context_availability",
    "jarvis_v2.scripts.smoke_test_chat_provider_privacy",
    "jarvis_v2.scripts.smoke_test_chat_history_provenance",
    "jarvis_v2.scripts.smoke_test_chat_egress_custody",
    "jarvis_v2.scripts.smoke_test_runtime_history_provenance",
    "jarvis_v2.scripts.smoke_test_self_knowledge_context",
    "jarvis_v2.scripts.smoke_test_chat_timeout",
    "jarvis_v2.scripts.smoke_test_chat_cli",
    "jarvis_v2.scripts.smoke_test_ask_cli",
    "jarvis_v2.scripts.smoke_test_chat_context",
    "jarvis_v2.scripts.smoke_test_next_layer",
    "jarvis_v2.scripts.smoke_test_http_helper",
    "jarvis_v2.scripts.smoke_test_connector_metadata",
    "jarvis_v2.scripts.smoke_test_calendar_connector",
    "jarvis_v2.scripts.smoke_test_calendar_argument_contracts",
    "jarvis_v2.scripts.smoke_test_calendar_delete_routing",
    "jarvis_v2.scripts.smoke_test_calendar_error_guidance",
    "jarvis_v2.scripts.smoke_test_email_connector",
    "jarvis_v2.scripts.smoke_test_email_failure_guidance",
    "jarvis_v2.scripts.smoke_test_imessage_connector",
    "jarvis_v2.scripts.smoke_test_imessage_control",
    "jarvis_v2.scripts.smoke_test_telegram_control",
    "jarvis_v2.scripts.smoke_test_telegram_offset_state",
    "jarvis_v2.scripts.smoke_test_telegram_hud_error_guidance",
    "jarvis_v2.scripts.smoke_test_social_connectors",
    "jarvis_v2.scripts.smoke_test_call_connector",
    "jarvis_v2.scripts.smoke_test_channel_health",
    "jarvis_v2.scripts.smoke_test_nl_datetime",
    "jarvis_v2.scripts.smoke_test_weather_connector",
    "jarvis_v2.scripts.smoke_test_reminders",
    "jarvis_v2.scripts.smoke_test_reminder_error_guidance",
    "jarvis_v2.scripts.smoke_test_reminder_approval_boundary",
    "jarvis_v2.scripts.smoke_test_reminder_delivery_recovery",
    "jarvis_v2.scripts.smoke_test_news_connector",
    "jarvis_v2.scripts.smoke_test_brief",
    "jarvis_v2.scripts.smoke_test_currency_connector",
    "jarvis_v2.scripts.smoke_test_currency_failure_guidance",
    "jarvis_v2.scripts.smoke_test_translate_connector",
    "jarvis_v2.scripts.smoke_test_translate_failure_guidance",
    "jarvis_v2.scripts.smoke_test_markets_connector",
    "jarvis_v2.scripts.smoke_test_markets_failure_guidance",
    "jarvis_v2.scripts.smoke_test_research_connector",
    "jarvis_v2.scripts.smoke_test_research_failure_guidance",
    "jarvis_v2.scripts.smoke_test_dictionary_connector",
    "jarvis_v2.scripts.smoke_test_wikipedia_connector",
    "jarvis_v2.scripts.smoke_test_sun_connector",
    "jarvis_v2.scripts.smoke_test_history_connector",
    "jarvis_v2.scripts.smoke_test_air_connector",
    "jarvis_v2.scripts.smoke_test_holidays_connector",
    "jarvis_v2.scripts.smoke_test_fun_connector",
    "jarvis_v2.scripts.smoke_test_countdown_connector",
    "jarvis_v2.scripts.smoke_test_writer_connector",
    "jarvis_v2.scripts.smoke_test_clipboard_safety",
    "jarvis_v2.scripts.smoke_test_compose_connector",
    "jarvis_v2.scripts.smoke_test_contacts_connector",
    "jarvis_v2.scripts.smoke_test_talk",
    "jarvis_v2.scripts.smoke_test_ocr",
    "jarvis_v2.scripts.smoke_test_command_suggest",
    "jarvis_v2.scripts.smoke_test_apple_reminders",
    "jarvis_v2.scripts.smoke_test_sensitive_output_retention",
    "jarvis_v2.scripts.smoke_test_contacts_fuzzy",
    "jarvis_v2.scripts.smoke_test_hud",
]


DEFAULT_MODULE_TIMEOUT_SECONDS = 180
STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS = 360
MIN_MODULE_TIMEOUT_SECONDS = 5
MAX_MODULE_TIMEOUT_SECONDS = 3600
MAX_PASSING_MODULE_OUTPUT_CHARS = 1200
MAX_FAILING_MODULE_OUTPUT_CHARS = 8000
MAX_MODULE_LABEL_CHARS = 180
# Several established integration smokes intentionally print large diagnostic
# matrices.  Keep the runner bounded without misclassifying those legitimate
# modules as noisy subprocess failures.
MAX_MODULE_CAPTURE_BYTES = 4 * 1024 * 1024
MODULE_OUTPUT_LIMIT_MARKER = "[aggregate smoke output exceeded the combined capture limit]"
RESIDUAL_PROCESS_GROUP_MARKER = "[smoke module left descendant processes after exit]"
# A completed module's PGID can remain briefly observable while macOS finishes
# reaping its process tree.  Keep this success-path settle window
# separate from the shorter TERM/KILL cleanup deadlines below: persistent
# descendants still fail closed and are terminated.
COMPLETED_PROCESS_GROUP_SETTLE_SECONDS = 5.0
PROCESS_GROUP_TERMINATE_GRACE_SECONDS = 2.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 2.0
_ACTIVE_MODULE_PROCESS: subprocess.Popen[bytes] | subprocess.Popen[str] | None = None
STATUS_SERVER_SMOKE_MODULE = "jarvis_v2.scripts.smoke_test_status_server_core"
LOCAL_BIND_PERMISSION_ERROR = "PermissionError: [Errno 1] Operation not permitted"
DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS = 900
MIN_SUITE_LOCK_TIMEOUT_SECONDS = 0
MAX_SUITE_LOCK_TIMEOUT_SECONDS = 7200
SUITE_LOCK_ENV_KEY = "JARVIS_SMOKE_SUITE_LOCK_PATH"
SUITE_LOCK_TIMEOUT_ENV_KEY = "JARVIS_SMOKE_SUITE_LOCK_TIMEOUT_SECONDS"
LOCAL_PATH_RE = re.compile(r"/(?:Users|private|var/folders|tmp)/[^\n\r;]*", re.IGNORECASE)
EXCLUDED_DISCOVERED_SMOKE_MODULES = {
    "jarvis_v2.scripts.smoke_test_all",
    "jarvis_v2.scripts.smoke_test_live",
    "jarvis_v2.scripts.smoke_test_status_server",
}
SENSITIVE_CHILD_ENV_KEYS = {
    "TELEGRAM_BOT_TOKEN",
    "GMAIL_ADDRESS",
    "GMAIL_APP_PASSWORD",
    "JARVIS_GOOGLE_CREDS",
    "JARVIS_GOOGLE_READONLY_TOKEN",
    "JARVIS_GOOGLE_TOKEN",
}
SENSITIVE_CHILD_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "APP_PASSWORD",
    "BOT_TOKEN",
    "CREDENTIAL",
    "CREDS",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_EMPTY_CHILD_ENV_PATH: Path | None = None


def _should_strip_child_env_key(key: str) -> bool:
    upper = key.upper()
    if upper.startswith("JARVIS_LIVE_"):
        return True
    if upper in SENSITIVE_CHILD_ENV_KEYS:
        return True
    return any(marker in upper for marker in SENSITIVE_CHILD_ENV_MARKERS)


def _empty_child_env_path() -> Path:
    global _EMPTY_CHILD_ENV_PATH
    if _EMPTY_CHILD_ENV_PATH is None:
        descriptor, raw_path = tempfile.mkstemp(prefix="jarvis-v3-smoke-empty-", suffix=".env")
        os.close(descriptor)
        _EMPTY_CHILD_ENV_PATH = Path(raw_path)
        atexit.register(_EMPTY_CHILD_ENV_PATH.unlink, missing_ok=True)
    return _EMPTY_CHILD_ENV_PATH


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if _should_strip_child_env_key(key):
            env.pop(key, None)
    env["JARVIS_V3_ENV"] = str(_empty_child_env_path())
    env["JARVIS_V3_ENABLE_DAEMONS"] = "0"
    env["JARVIS_V3_ENABLE_SCHEDULER"] = "0"
    return env


def _isolated_child_env(root: Path) -> dict[str, str]:
    """Fence aggregate smokes away from configured production storage paths."""
    data_dir = root / "data"
    vault_dir = root / "vault"
    fallback_dir = root / "fallback"
    state_dir = root / "state"
    cache_dir = root / "cache"
    home_dir = root / "home"
    temp_dir = root / "tmp"
    for directory in (data_dir, vault_dir, fallback_dir, state_dir, cache_dir, home_dir, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    env = _child_env()
    # The selected V3 environment is authoritative for runtime custody.  An
    # empty selected file plus inherited production custody is intentionally
    # refused by the launch boundary, so discard all such inherited values and
    # record only the synthetic isolation paths below in a new selected file.
    for key in V3_INHERITED_CUSTODY_KEYS:
        env.pop(key, None)
    env.update(
        {
            "HOME": str(home_dir),
            "TMPDIR": str(temp_dir),
            "XDG_CACHE_HOME": str(cache_dir),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "share"),
            "JARVIS_DATA_DIR": str(data_dir),
            "JARVIS_DB_PATH": str(data_dir / "jarvis.sqlite"),
            "JARVIS_OBSIDIAN_VAULT": str(vault_dir),
            "OBSIDIAN_VAULT_PATH": str(vault_dir),
            "JARVIS_STORAGE_FALLBACK_DIR": str(fallback_dir),
            "JARVIS_WATCHED_DIRS": "",
            "JARVIS_TELEGRAM_STATE": str(state_dir / "telegram-control.json"),
            "JARVIS_V3_IMESSAGE_STATE": str(state_dir / "imessage-control.json"),
            "JARVIS_REMINDERS_FILE": str(state_dir / "telegram-reminders.json"),
            "JARVIS_CACHE_DIR": str(cache_dir),
            "JARVIS_STATUS_PORT": "8766",
            # Keep a marked public candidate byte-for-byte stable. The parent
            # is invoked with -B; every isolated child must inherit the same
            # no-bytecode boundary even though -B is not present in argv.
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    selected_values = {
        key: env[key]
        for key in sorted(V3_INHERITED_CUSTODY_KEYS)
        if key in env
    }
    for value in selected_values.values():
        if (
            value != value.strip()
            or any(character in value for character in ("\x00", "\r", "\n", "'", '"'))
            or " #" in value
        ):
            raise RuntimeError("aggregate smoke isolation path cannot be encoded safely")
    selected_env = root / "runtime.env"
    selected_env.write_text(
        "".join(f"{key}={value}\n" for key, value in selected_values.items()),
        encoding="utf-8",
    )
    selected_env.chmod(0o600)
    env["JARVIS_V3_ENV"] = str(selected_env)
    return env


def _module_timeout_seconds() -> float:
    raw = os.getenv("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_MODULE_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_MODULE_TIMEOUT_SECONDS)
    if not isfinite(value):
        return float(DEFAULT_MODULE_TIMEOUT_SECONDS)
    return min(float(MAX_MODULE_TIMEOUT_SECONDS), max(float(MIN_MODULE_TIMEOUT_SECONDS), value))


def _module_timeout_seconds_for(module: str) -> float:
    if os.getenv("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", "").strip():
        return _module_timeout_seconds()
    if module == STATUS_SERVER_SMOKE_MODULE:
        return float(STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS)
    return float(DEFAULT_MODULE_TIMEOUT_SECONDS)


def _suite_lock_path() -> Path:
    raw = os.getenv(SUITE_LOCK_ENV_KEY, "").strip()
    if raw:
        return Path(raw)
    return Path(tempfile.gettempdir()) / "jarvis-v3-smoke-test-all.lock"


def _suite_lock_timeout_seconds() -> float:
    raw = os.getenv(SUITE_LOCK_TIMEOUT_ENV_KEY, "").strip()
    if not raw:
        return float(DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except ValueError:
        return float(DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS)
    if not isfinite(value):
        return float(DEFAULT_SUITE_LOCK_TIMEOUT_SECONDS)
    return min(float(MAX_SUITE_LOCK_TIMEOUT_SECONDS), max(float(MIN_SUITE_LOCK_TIMEOUT_SECONDS), value))


@contextmanager
def _aggregate_run_lock():
    path = _suite_lock_path()
    timeout_seconds = _suite_lock_timeout_seconds()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another aggregate smoke run is active; waiting up to {timeout_seconds:g}s for the suite lock.")
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        print(
                            "FAILED: another aggregate smoke run is still active; "
                            "wait or set JARVIS_SMOKE_SUITE_LOCK_TIMEOUT_SECONDS."
                        )
                        raise SystemExit(1)
                    time.sleep(min(1.0, max(0.01, deadline - time.monotonic())))
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} started={time.time():.3f}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _run_module(module: str, timeout_seconds: float | None = None) -> subprocess.CompletedProcess[str]:
    global _ACTIVE_MODULE_PROCESS
    timeout = _module_timeout_seconds_for(module) if timeout_seconds is None else timeout_seconds
    if os.name != "posix" or not hasattr(os, "killpg"):
        raise RuntimeError("aggregate smoke process-group isolation is unavailable on this platform")

    command = [sys.executable, "-m", module]
    # Keep the isolation root short: macOS limits AF_UNIX paths to 103 bytes,
    # and socket-using smokes create another temporary directory beneath TMPDIR.
    with tempfile.TemporaryDirectory(prefix="jv2-", dir="/tmp") as isolated_root:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_isolated_child_env(Path(isolated_root)),
            start_new_session=True,
        )
        _ACTIVE_MODULE_PROCESS = process
        has_real_pipe_fds = _has_real_pipe_fds(process)
        try:
            if not has_real_pipe_fds:
                # Preserve the long-standing Popen mock seam. Production Popen
                # instances always take the bounded streaming path below.
                stdout, stderr = process.communicate(timeout=timeout)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            return _stream_module_output_bounded(process, command, timeout)
        except subprocess.TimeoutExpired as exc:
            if has_real_pipe_fds:
                _cleanup_real_process_group_best_effort(process)
                stdout, stderr = exc.stdout, exc.stderr
            else:
                stdout, stderr = _cleanup_process_group_best_effort(process, exc)
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from None
        except BaseException:
            if has_real_pipe_fds:
                _cleanup_real_process_group_best_effort(process)
            else:
                _cleanup_process_group_best_effort(process, None)
            raise
        finally:
            if _ACTIVE_MODULE_PROCESS is process:
                _ACTIVE_MODULE_PROCESS = None


def _has_real_pipe_fds(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> bool:
    for stream in (process.stdout, process.stderr):
        try:
            descriptor = stream.fileno() if stream is not None else None
        except (AttributeError, OSError, ValueError):
            return False
        if not isinstance(descriptor, int):
            return False
    return True


def _decode_module_capture(value: bytearray) -> str:
    return bytes(value).decode("utf-8", errors="replace")


def _bounded_failure_result(
    command: list[str],
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    stdout: bytearray,
    stderr: bytearray,
    marker_text: str,
) -> subprocess.CompletedProcess[str]:
    marker = ("\n" + marker_text + "\n").encode("utf-8")
    payload_limit = max(0, MAX_MODULE_CAPTURE_BYTES - len(marker))
    kept_stdout = stdout[:payload_limit]
    kept_stderr = stderr[: max(0, payload_limit - len(kept_stdout))]
    kept_stderr.extend(marker)
    returncode = process.returncode
    if returncode in (None, 0):
        returncode = 1
    return subprocess.CompletedProcess(
        command,
        returncode,
        _decode_module_capture(kept_stdout),
        _decode_module_capture(kept_stderr),
    )


def _stream_module_output_bounded(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    command: list[str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    stdout = bytearray()
    stderr = bytearray()
    streams = ((process.stdout, stdout), (process.stderr, stderr))
    deadline = time.monotonic() + timeout
    output_limit_exceeded = False
    selector: selectors.BaseSelector | None = None
    try:
        selector = selectors.DefaultSelector()
        for stream, capture in streams:
            if stream is None:
                raise RuntimeError("aggregate smoke output pipe was not created")
            selector.register(stream, selectors.EVENT_READ, capture)

        while selector.get_map():
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise subprocess.TimeoutExpired(
                    command,
                    timeout,
                    output=_decode_module_capture(stdout),
                    stderr=_decode_module_capture(stderr),
                )
            events = selector.select(min(0.1, remaining_time))
            for key, _mask in events:
                remaining_capture = MAX_MODULE_CAPTURE_BYTES - len(stdout) - len(stderr)
                chunk = os.read(key.fd, min(65536, max(1, remaining_capture + 1)))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                capture: bytearray = key.data
                if len(chunk) > remaining_capture:
                    if remaining_capture > 0:
                        capture.extend(chunk[:remaining_capture])
                    output_limit_exceeded = True
                    break
                capture.extend(chunk)
            if output_limit_exceeded:
                break

        if output_limit_exceeded:
            _cleanup_real_process_group_best_effort(process)
            return _bounded_failure_result(
                command,
                process,
                stdout,
                stderr,
                MODULE_OUTPUT_LIMIT_MARKER,
            )

        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=_decode_module_capture(stdout),
                stderr=_decode_module_capture(stderr),
            )
        try:
            returncode = process.wait(timeout=remaining_time)
        except subprocess.TimeoutExpired:
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=_decode_module_capture(stdout),
                stderr=_decode_module_capture(stderr),
            ) from None
        if _process_group_exists(process.pid) and not _wait_for_process_group_exit(
            process,
            COMPLETED_PROCESS_GROUP_SETTLE_SECONDS,
        ):
            _cleanup_real_process_group_best_effort(process)
            return _bounded_failure_result(
                command,
                process,
                stdout,
                stderr,
                RESIDUAL_PROCESS_GROUP_MARKER,
            )
        return subprocess.CompletedProcess(
            command,
            returncode,
            _decode_module_capture(stdout),
            _decode_module_capture(stderr),
        )
    finally:
        if selector is not None:
            selector.close()
        for stream, _capture in streams:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(process.pid):
        process.poll()
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(0.01, max(0.001, deadline - time.monotonic())))
    return True


def _cleanup_real_process_group_best_effort(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
) -> None:
    try:
        _terminate_process_group(process)
    except BaseException:
        pass
    group_exited = _wait_for_process_group_exit(
        process,
        PROCESS_GROUP_TERMINATE_GRACE_SECONDS,
    )
    if not group_exited:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException:
            pass
        _wait_for_process_group_exit(process, PROCESS_GROUP_KILL_GRACE_SECONDS)
    try:
        process.wait(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
    except BaseException:
        try:
            process.kill()
        except BaseException:
            pass
        try:
            process.wait(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
        except BaseException:
            pass


def _terminate_process_group(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
) -> None:
    # This reaches descendants that remain in the group. A descendant that calls
    # setsid() can escape; process-group cleanup does not contain that malicious case.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _merge_cleanup_stream(
    latest: str | bytes | None,
    previous: str | bytes | None,
) -> str | bytes | None:
    return latest if latest is not None else previous


def _kill_and_reap_direct_child_best_effort(
    process: subprocess.Popen[str],
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> tuple[str | bytes | None, str | bytes | None]:
    try:
        process.kill()
    except BaseException:
        pass
    try:
        final_stdout, final_stderr = process.communicate(
            timeout=PROCESS_GROUP_KILL_GRACE_SECONDS
        )
    except BaseException:
        try:
            process.wait(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
        except BaseException:
            pass
        return stdout, stderr
    return (
        _merge_cleanup_stream(final_stdout, stdout),
        _merge_cleanup_stream(final_stderr, stderr),
    )


def _cleanup_process_group_best_effort(
    process: subprocess.Popen[str],
    timeout_error: subprocess.TimeoutExpired | None,
) -> tuple[str | bytes | None, str | bytes | None]:
    stdout = timeout_error.stdout if timeout_error is not None else None
    stderr = timeout_error.stderr if timeout_error is not None else None
    try:
        _terminate_process_group(process)
    except BaseException:
        return _kill_and_reap_direct_child_best_effort(process, stdout, stderr)
    try:
        return _reap_process_group(process, timeout_error)
    except BaseException:
        return _kill_and_reap_direct_child_best_effort(process, stdout, stderr)


def _handle_shutdown_signal(signum: int, _frame: object) -> None:
    process = _ACTIVE_MODULE_PROCESS
    if process is not None:
        if _has_real_pipe_fds(process):
            _cleanup_real_process_group_best_effort(process)
        else:
            _cleanup_process_group_best_effort(process, None)
    raise SystemExit(128 + signum)


def _reap_process_group(
    process: subprocess.Popen[str],
    timeout_error: subprocess.TimeoutExpired | None,
) -> tuple[str | bytes | None, str | bytes | None]:
    stdout = timeout_error.stdout if timeout_error is not None else None
    stderr = timeout_error.stderr if timeout_error is not None else None
    try:
        return process.communicate(timeout=PROCESS_GROUP_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if exc.stdout is not None else stdout
        stderr = exc.stderr if exc.stderr is not None else stderr

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return process.communicate(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            final_stdout, final_stderr = process.communicate(timeout=PROCESS_GROUP_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired as final_exc:
            raise RuntimeError("smoke process group could not be reaped") from final_exc
        return (
            final_stdout if final_stdout is not None else (exc.stdout if exc.stdout is not None else stdout),
            final_stderr if final_stderr is not None else (exc.stderr if exc.stderr is not None else stderr),
        )


def _retryable_module_failure(module: str, result: subprocess.CompletedProcess[str]) -> bool:
    if module != STATUS_SERVER_SMOKE_MODULE or result.returncode == 0:
        return False
    return LOCAL_BIND_PERMISSION_ERROR in _stream_text(result.stdout) or LOCAL_BIND_PERMISSION_ERROR in _stream_text(result.stderr)


def _duplicate_modules(modules: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for module in modules:
        if module in seen and module not in duplicates:
            duplicates.append(module)
        seen.add(module)
    return duplicates


def _unclean_modules(modules: list[str]) -> list[str]:
    return [module for module in modules if not module or module != module.strip()]


def _missing_modules(modules: list[str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        if find_spec(module) is None:
            missing.append(module)
    return missing


def _non_smoke_modules(modules: list[str]) -> list[str]:
    prefix = "jarvis_v2.scripts.smoke_test_"
    return [module for module in modules if not module.startswith(prefix)]


def _discover_smoke_modules() -> list[str]:
    scripts_dir = Path(__file__).resolve().parent
    modules: list[str] = []
    for path in sorted(scripts_dir.glob("smoke_test_*.py")):
        module = f"jarvis_v2.scripts.{path.stem}"
        if module not in EXCLUDED_DISCOVERED_SMOKE_MODULES:
            modules.append(module)
    return modules


def _unlisted_smoke_modules(configured: list[str], discovered: list[str] | None = None) -> list[str]:
    configured_set = set(configured)
    discovered_modules = _discover_smoke_modules() if discovered is None else discovered
    return [module for module in discovered_modules if module not in configured_set]


def _python_compile_failures(root: Path | None = None) -> list[tuple[str, str]]:
    repository_root = root or Path(__file__).resolve().parents[2]
    paths = sorted(repository_root.glob("*.py"))
    package_root = repository_root / "jarvis_v2"
    if package_root.is_dir():
        paths.extend(sorted(package_root.rglob("*.py")))

    failures: list[tuple[str, str]] = []
    for path in paths:
        relative_path = path.relative_to(repository_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, relative_path, "exec", dont_inherit=True)
        except SyntaxError as exc:
            location = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
            failures.append((relative_path, f"{location}: {exc.msg}"))
        except (OSError, UnicodeError) as exc:
            failures.append((relative_path, type(exc).__name__))
    return failures


def _uninvoked_test_functions_from_source(source: str, module: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=module)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    defined_sync_functions = {
        name
        for name, node in functions.items()
        if isinstance(node, ast.FunctionDef) and name.startswith("test_")
    }
    defined_async_functions = {
        name
        for name, node in functions.items()
        if isinstance(node, ast.AsyncFunctionDef) and name.startswith("test_")
    }
    defined_functions = defined_sync_functions | defined_async_functions
    defined_methods = _class_test_methods(classes)
    defined = defined_functions | defined_methods
    entry_calls = _module_entry_call_names(tree)
    entry_method_calls = _called_test_method_names_for_module(tree, set(classes))
    entry_async_calls = _executed_async_test_names_for_module(tree, defined_async_functions)
    called_tests = (
        defined_sync_functions.intersection(entry_calls)
        | defined_async_functions.intersection(entry_async_calls)
        | defined_methods.intersection(entry_method_calls)
    )
    reachable_functions: set[str] = set()
    pending = [name for name in entry_calls if name in functions]
    while pending:
        name = pending.pop()
        if name in reachable_functions:
            continue
        reachable_functions.add(name)
        calls = _called_function_names(functions[name].body)
        async_calls = _executed_async_test_names(functions[name].body, defined_async_functions)
        method_calls = _called_test_method_names(functions[name].body, set(classes))
        called_tests.update(defined_sync_functions.intersection(calls))
        called_tests.update(defined_async_functions.intersection(async_calls))
        called_tests.update(defined_methods.intersection(method_calls))
        pending.extend(
            callee
            for callee in calls
            if callee in functions and callee not in reachable_functions
        )
    return sorted(defined - called_tests)


class _FunctionCallNameVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.names.add(node.func.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _called_function_names(nodes: ast.AST | list[ast.AST] | tuple[ast.AST, ...]) -> set[str]:
    visitor = _FunctionCallNameVisitor()
    if isinstance(nodes, ast.AST):
        visitor.visit(nodes)
    else:
        for node in nodes:
            visitor.visit(node)
    return visitor.names


class _AsyncTestExecutionNameVisitor(ast.NodeVisitor):
    def __init__(self, async_test_names: set[str]) -> None:
        self.async_test_names = async_test_names
        self.names: set[str] = set()

    def visit_Await(self, node: ast.Await) -> None:
        name = _async_test_call_name(node.value, self.async_test_names)
        if name:
            self.names.add(name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_asyncio_run_call(node) and node.args:
            name = _async_test_call_name(node.args[0], self.async_test_names)
            if name:
                self.names.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _executed_async_test_names(
    nodes: ast.AST | list[ast.AST] | tuple[ast.AST, ...],
    async_test_names: set[str],
) -> set[str]:
    visitor = _AsyncTestExecutionNameVisitor(async_test_names)
    if isinstance(nodes, ast.AST):
        visitor.visit(nodes)
    else:
        for node in nodes:
            visitor.visit(node)
    return visitor.names


def _executed_async_test_names_for_module(tree: ast.Module, async_test_names: set[str]) -> set[str]:
    calls: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If):
            if _is_main_guard(node.test):
                calls.update(_executed_async_test_names(node.body, async_test_names))
            continue
        calls.update(_executed_async_test_names(node, async_test_names))
    return calls


def _async_test_call_name(node: ast.AST, async_test_names: set[str]) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in async_test_names:
        return node.func.id
    return None


def _is_asyncio_run_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "run"
        and isinstance(func.value, ast.Name)
        and func.value.id == "asyncio"
    )


class _TestMethodCallNameVisitor(ast.NodeVisitor):
    def __init__(self, class_names: set[str]) -> None:
        self.class_names = class_names
        self.instance_names: dict[str, str] = {}
        self.names: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        name = _test_method_call_name(node, self.class_names, self.instance_names)
        if name:
            self.names.add(name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        class_name = _class_constructor_name(node.value, self.class_names)
        for target in node.targets:
            self._record_assignment_target(target, class_name)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        class_name = _class_constructor_name(node.value, self.class_names) if node.value else None
        self._record_assignment_target(node.target, class_name)
        if node.value:
            self.visit(node.value)

    def _record_assignment_target(self, target: ast.AST, class_name: str | None) -> None:
        if isinstance(target, ast.Name):
            if class_name:
                self.instance_names[target.id] = class_name
            else:
                self.instance_names.pop(target.id, None)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None


def _called_test_method_names(nodes: ast.AST | list[ast.AST] | tuple[ast.AST, ...], class_names: set[str]) -> set[str]:
    visitor = _TestMethodCallNameVisitor(class_names)
    if isinstance(nodes, ast.AST):
        visitor.visit(nodes)
    else:
        for node in nodes:
            visitor.visit(node)
    return visitor.names


def _called_test_method_names_for_module(tree: ast.Module, class_names: set[str]) -> set[str]:
    calls: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If):
            if _is_main_guard(node.test):
                calls.update(_called_test_method_names(node.body, class_names))
            continue
        calls.update(_called_test_method_names(node, class_names))
    return calls


def _class_test_methods(classes: dict[str, ast.ClassDef]) -> set[str]:
    methods: set[str] = set()
    for class_name, class_node in classes.items():
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                methods.add(f"{class_name}.{node.name}")
    return methods


def _test_method_call_name(node: ast.Call, class_names: set[str], instance_names: dict[str, str]) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute) or not func.attr.startswith("test_"):
        return None
    owner = func.value
    if isinstance(owner, ast.Call) and isinstance(owner.func, ast.Name) and owner.func.id in class_names:
        return f"{owner.func.id}.{func.attr}"
    if isinstance(owner, ast.Name) and owner.id in class_names:
        return f"{owner.id}.{func.attr}"
    if isinstance(owner, ast.Name) and owner.id in instance_names:
        return f"{instance_names[owner.id]}.{func.attr}"
    return None


def _class_constructor_name(node: ast.AST | None, class_names: set[str]) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in class_names:
        return node.func.id
    return None


def _module_entry_call_names(tree: ast.Module) -> set[str]:
    calls: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.If):
            if _is_main_guard(node.test):
                calls.update(_called_function_names(node.body))
            continue
        calls.update(_called_function_names(node))
    return calls


def _is_main_guard(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return (
        _is_name_dunder_main(left) and _is_string_dunder_main(right)
    ) or (
        _is_string_dunder_main(left) and _is_name_dunder_main(right)
    )


def _is_name_dunder_main(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "__name__"


def _is_string_dunder_main(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "__main__"


def _uninvoked_test_functions(modules: list[str]) -> dict[str, list[str]]:
    uninvoked: dict[str, list[str]] = {}
    for module in modules:
        spec = find_spec(module)
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin:
            continue
        path = Path(origin)
        try:
            missing_calls = _uninvoked_test_functions_from_source(path.read_text(encoding="utf-8"), module)
        except Exception:
            continue
        if missing_calls:
            uninvoked[module] = missing_calls
    return uninvoked


def _has_modules(modules: list[str]) -> bool:
    return bool(modules)


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _sensitive_child_env_values(env: dict[str, str] | None = None) -> list[str]:
    source = os.environ if env is None else env
    values = {
        value
        for key, value in source.items()
        if value and len(value) >= 4 and _should_strip_child_env_key(key)
    }
    return sorted(values, key=len, reverse=True)


def _safe_stream_text(value: str | bytes | None, env: dict[str, str] | None = None) -> str:
    text = _stream_text(value)
    for secret in _sensitive_child_env_values(env):
        text = text.replace(secret, "<redacted-env-value>")
    return text


def _safe_module_label(module: str) -> str:
    text = " ".join(LOCAL_PATH_RE.sub("<local-path>", _safe_stream_text(module)).split())
    if len(text) <= MAX_MODULE_LABEL_CHARS:
        return text
    return text[: MAX_MODULE_LABEL_CHARS - 1].rstrip() + "…"


def _passing_module_output(value: str | bytes | None) -> str:
    text = _safe_stream_text(value)
    if len(text) <= MAX_PASSING_MODULE_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_PASSING_MODULE_OUTPUT_CHARS
    return (
        f"{text[:MAX_PASSING_MODULE_OUTPUT_CHARS]}\n"
        f"[smoke output truncated for passing module; omitted {omitted} chars]"
    )


def _failing_module_output(value: str | bytes | None, *, stream_name: str) -> str:
    text = _safe_stream_text(value)
    if len(text) <= MAX_FAILING_MODULE_OUTPUT_CHARS:
        return text
    omitted = len(text) - MAX_FAILING_MODULE_OUTPUT_CHARS
    return (
        f"{text[:MAX_FAILING_MODULE_OUTPUT_CHARS]}\n"
        f"[{stream_name} truncated for failing module; omitted {omitted} chars]"
    )


def _suite_summary(modules: list[str], timeout_seconds: float) -> str:
    plural = "" if len(modules) == 1 else "s"
    return f"Running {len(modules)} smoke module{plural} with {timeout_seconds:g}s per-module timeout."


def _suite_timeout_note(modules: list[str]) -> str:
    if os.getenv("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", "").strip():
        return ""
    if STATUS_SERVER_SMOKE_MODULE not in modules:
        return ""
    default_timeout = float(DEFAULT_MODULE_TIMEOUT_SECONDS)
    status_timeout = float(STATUS_SERVER_DEFAULT_MODULE_TIMEOUT_SECONDS)
    if status_timeout == default_timeout:
        return ""
    return (
        f"Status-server smoke timeout: {status_timeout:g}s "
        f"(default {default_timeout:g}s for other modules)."
    )


def _timeout_for_timeout_message(module: str, fallback_timeout_seconds: float) -> float:
    if os.getenv("JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS", "").strip():
        return fallback_timeout_seconds
    return _module_timeout_seconds_for(module)


def _success_summary(modules: list[str]) -> str:
    plural = "" if len(modules) == 1 else "s"
    return f"All {len(modules)} smoke module{plural} passed."


def _failure_summary(failures: list[str], modules: list[str]) -> str:
    module_plural = "" if len(modules) == 1 else "s"
    return f"FAILED: {len(failures)} of {len(modules)} smoke module{module_plural} failed."


def _format_duration(seconds: float) -> str:
    return f"{max(0.0, seconds):.1f}s"


def _success_summary_with_duration(modules: list[str], seconds: float) -> str:
    return f"{_success_summary(modules)} Duration: {_format_duration(seconds)}."


def _failure_summary_with_duration(failures: list[str], modules: list[str], seconds: float) -> str:
    return f"{_failure_summary(failures, modules)} Duration: {_format_duration(seconds)}."


def _module_heading(index: int, total: int, module: str) -> str:
    return f"== [{index}/{total}] {_safe_module_label(module)} =="


def _run_module_bounded(
    module: str,
    module_timeout_seconds: float,
    suite_timeout_seconds: float,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return _run_module(module, module_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        print(
            f"TIMEOUT: {_safe_module_label(module)} exceeded "
            f"{_timeout_for_timeout_message(module, suite_timeout_seconds):g}s"
        )
        stdout = _failing_module_output(exc.stdout, stream_name="stdout")
        stderr = _failing_module_output(exc.stderr, stream_name="stderr")
        if stdout:
            print(stdout)
        if stderr:
            print("stderr:")
            print(stderr)
        return None


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    with _aggregate_run_lock():
        if not _has_modules(TEST_MODULES):
            print("FAILED: no smoke modules configured")
            raise SystemExit(1)

        duplicates = _duplicate_modules(TEST_MODULES)
        if duplicates:
            print("FAILED: duplicate smoke modules configured")
            for module in duplicates:
                print(f"- {_safe_module_label(module)}")
            raise SystemExit(1)

        unclean = _unclean_modules(TEST_MODULES)
        if unclean:
            print("FAILED: unclean smoke module names configured")
            for module in unclean:
                print(f"- {_safe_module_label(module)!r}")
            raise SystemExit(1)

        missing = _missing_modules(TEST_MODULES)
        if missing:
            print("FAILED: missing smoke modules configured")
            for module in missing:
                print(f"- {_safe_module_label(module)}")
            raise SystemExit(1)

        non_smoke = _non_smoke_modules(TEST_MODULES)
        if non_smoke:
            print("FAILED: non-smoke modules configured")
            for module in non_smoke:
                print(f"- {_safe_module_label(module)}")
            raise SystemExit(1)

        unlisted = _unlisted_smoke_modules(TEST_MODULES)
        if unlisted:
            print("FAILED: discovered smoke modules missing from aggregate suite")
            for module in unlisted:
                print(f"- {_safe_module_label(module)}")
            raise SystemExit(1)

        compile_failures = _python_compile_failures()
        if compile_failures:
            print("FAILED: repository Python compile preflight")
            for path, detail in compile_failures:
                print(f"- {_safe_module_label(path)}: {detail}")
            raise SystemExit(1)

        uninvoked = _uninvoked_test_functions(TEST_MODULES)
        if uninvoked:
            print("FAILED: smoke modules define test functions that are not called from their runners")
            for module, functions in uninvoked.items():
                print(f"- {_safe_module_label(module)}: {', '.join(functions)}")
            raise SystemExit(1)

        timeout_seconds = _module_timeout_seconds()
        print(_suite_summary(TEST_MODULES, timeout_seconds))
        timeout_note = _suite_timeout_note(TEST_MODULES)
        if timeout_note:
            print(timeout_note)

        started = time.monotonic()
        failures: list[str] = []
        total = len(TEST_MODULES)
        for index, module in enumerate(TEST_MODULES, start=1):
            print(_module_heading(index, total, module))
            module_timeout_seconds = _module_timeout_seconds_for(module)
            result = _run_module_bounded(module, module_timeout_seconds, timeout_seconds)
            if result is None:
                failures.append(module)
                continue
            if result.returncode != 0:
                if _retryable_module_failure(module, result):
                    print("RETRY: status-server smoke hit local socket permission edge; retrying once.")
                    result = _run_module_bounded(module, module_timeout_seconds, timeout_seconds)
                    if result is None:
                        failures.append(module)
                        continue
                    if result.returncode == 0:
                        print(_passing_module_output(result.stdout))
                        continue
                print(_failing_module_output(result.stdout, stream_name="stdout"))
                if result.stderr:
                    print("stderr:")
                    print(_failing_module_output(result.stderr, stream_name="stderr"))
                failures.append(module)
            else:
                print(_passing_module_output(result.stdout))
        if failures:
            print(_failure_summary_with_duration(failures, TEST_MODULES, time.monotonic() - started))
            for module in failures:
                print(f"- {_safe_module_label(module)}")
            raise SystemExit(1)
        print(_success_summary_with_duration(TEST_MODULES, time.monotonic() - started))


if __name__ == "__main__":
    main()
