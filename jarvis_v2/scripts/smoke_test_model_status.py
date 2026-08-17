from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

import jarvis_v2.agent.model_provider as model_provider
from jarvis_v2.agent.planner import RuleBasedPlanner
from jarvis_v2.agent.runtime import JarvisRuntime
from jarvis_v2.config import JarvisConfig
import jarvis_v2.tools.model_status as model_status
from jarvis_v2.tools.model_status import (
    _metadata_bool,
    _specialist_action_proposal_review_token_sha256,
    _specialist_action_proposal_scorecard_shape_ready,
    _specialist_combined_action_proposal_scorecard_shape_ready,
    _specialist_cycle_ledger_ready,
    _specialist_cycle_ledger_token_sha256,
    _specialist_cycle_ledger_token_ready_from_metadata,
    _specialist_fresh_review_boundary_token_sha256,
    _specialist_post_run_closure_token_sha256,
    _specialist_route_to_runtime_contract_token_sha256,
    _specialist_runtime_review_contract_ready,
)


def assert_model_status_metadata_bool_is_exact() -> None:
    if _metadata_bool(True) is not True:
        raise SystemExit("model_status exact metadata bool rejected True")
    if _metadata_bool(False) is not False:
        raise SystemExit("model_status exact metadata bool rejected False")
    for value in ("true", "false", "yes", "no", 1, 0, [True], {"ready": True}, None):
        if _metadata_bool(value) is not False:
            raise SystemExit(f"model_status exact metadata bool accepted malformed truthy value: {value!r}")
    if _metadata_bool("false", default=True) is not True:
        raise SystemExit("model_status exact metadata bool did not preserve explicit default")


def assert_missing_config_recovery_is_actionable() -> None:
    cases = [
        (
            model_status.make_model_status_tool(None)({}),
            "model_routing_status",
            "model routing status",
        ),
        (
            model_status.make_model_planner_prompt_preview_tool(None, lambda: [])({"request": "find README"}),
            "model_planner_prompt_preview",
            "model planner prompt preview: <request>",
        ),
    ]
    for result, tool_name, retry_command in cases:
        if result.ok or result.tool_name != tool_name:
            raise SystemExit(f"{tool_name} should fail closed without config: {result}")
        for expected in ["setup check", "fix the reported configuration issue", retry_command]:
            if expected not in result.output:
                raise SystemExit(f"{tool_name} missing actionable config recovery {expected!r}: {result.output}")
        if result.metadata.get("next_command") != "setup check":
            raise SystemExit(f"{tool_name} should name setup check first: {result.metadata}")
        if result.metadata.get("recovery_commands") != ["setup check", retry_command]:
            raise SystemExit(f"{tool_name} should expose ordered config recovery: {result.metadata}")
        if result.metadata.get("retry_requires_setup_repair") is not True:
            raise SystemExit(f"{tool_name} should block retry until setup repair: {result.metadata}")
        if result.metadata.get("authorizes_retry") is not False:
            raise SystemExit(f"{tool_name} config recovery must not authorize retry: {result.metadata}")
        assert_safe_metadata(result.metadata, f"{tool_name} missing-config result")


class _ProbeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.payload


class _ProbeOpener:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls: list[tuple[str, float]] = []

    def open(self, request, timeout: float):
        self.calls.append((request.full_url, timeout))
        return _ProbeResponse(self.payload)


def assert_ollama_http_probe_contract(root: Path) -> None:
    config = JarvisConfig(
        data_dir=root / "ollama-probe-environment",
        db_path=root / "ollama-probe-environment" / "jarvis.sqlite",
        obsidian_vault=root / "ollama-probe-environment" / "Vault",
        obsidian_root="Jarvis",
        chat_model="jarvis-v2-probe-smoke-chat",
        planner_model="jarvis-v2-probe-smoke-planner",
        use_model_planner=True,
    )
    handler = model_status.make_model_status_tool(config)
    raw_rejected_values = {
        "OLLAMA_HOST": "http://remote-model.private.invalid:22441",
        "HTTP_PROXY": "http://upper-http-proxy.private.invalid:22442",
        "HTTPS_PROXY": "http://upper-https-proxy.private.invalid:22443",
        "ALL_PROXY": "socks5://upper-all-proxy.private.invalid:22444",
        "http_proxy": "http://lower-http-proxy.private.invalid:22445",
        "https_proxy": "http://lower-https-proxy.private.invalid:22446",
        "all_proxy": "socks5://lower-all-proxy.private.invalid:22447",
    }
    rejected_opener = _ProbeOpener(b'{"models": []}')

    def rejected_probe():
        return model_provider.probe_ollama_models(opener=rejected_opener)

    with (
        patch.dict(os.environ, raw_rejected_values, clear=True),
        patch.object(model_status, "probe_ollama_models", rejected_probe),
    ):
        rejected = handler({})
    if rejected_opener.calls:
        raise SystemExit(f"Rejected remote Ollama destination reached the HTTP opener: {rejected_opener.calls}")
    if rejected.metadata.get("ollama_request_blocked") is not True:
        raise SystemExit(f"Rejected remote Ollama destination was not reported blocked: {rejected.metadata}")
    rejected_receipt = rejected.output + " " + repr(rejected.metadata)
    leaked_rejected_values = [value for value in raw_rejected_values.values() if value in rejected_receipt]
    if leaked_rejected_values:
        raise SystemExit(f"Rejected Ollama host/proxy values leaked into body or metadata: {leaked_rejected_values}")

    raw_proxy_values = {
        "HTTP_PROXY": "http://upper-http-proxy.private.invalid:22542",
        "HTTPS_PROXY": "http://upper-https-proxy.private.invalid:22543",
        "ALL_PROXY": "socks5://upper-all-proxy.private.invalid:22544",
        "http_proxy": "http://lower-http-proxy.private.invalid:22545",
        "https_proxy": "http://lower-https-proxy.private.invalid:22546",
        "all_proxy": "socks5://lower-all-proxy.private.invalid:22547",
    }
    accepted_environment = {"OLLAMA_HOST": "localhost:22541", **raw_proxy_values}
    accepted_opener = _ProbeOpener(
        b'{"models": ['
        b'{"name": "jarvis-v2-probe-smoke-chat"},'
        b'{"model": "jarvis-v2-probe-smoke-planner"}'
        b"]}"
    )
    opener_contract: list[tuple[object, ...]] = []

    def fake_build_opener(*handlers):
        opener_contract.append(handlers)
        proxy_handlers = [item for item in handlers if isinstance(item, model_provider.ProxyHandler)]
        if len(proxy_handlers) != 1 or proxy_handlers[0].proxies:
            raise AssertionError(f"Ollama probe must install one empty proxy handler: {handlers}")
        if not any(type(item).__name__ == "_NoRedirectHandler" for item in handlers):
            raise AssertionError(f"Ollama probe must install its no-redirect handler: {handlers}")
        return accepted_opener

    with (
        patch.dict(os.environ, accepted_environment, clear=True),
        patch.object(model_provider, "build_opener", fake_build_opener),
    ):
        accepted = handler({})
    if len(opener_contract) != 1 or accepted_opener.calls != [
        ("http://127.0.0.1:22541/api/tags", 5.0)
    ]:
        raise SystemExit(
            f"Validated loopback Ollama HTTP probe contract drifted: {opener_contract} / {accepted_opener.calls}"
        )
    if accepted.metadata.get("ollama_destination_allowed") is not True:
        raise SystemExit(f"Validated loopback Ollama destination was not accepted: {accepted.metadata}")
    if accepted.metadata.get("ollama_redirects_allowed") is not False:
        raise SystemExit(f"Validated Ollama probe missed redirect-disabled receipt: {accepted.metadata}")
    if accepted.metadata.get("ollama_proxy_environment_allowed") is not False:
        raise SystemExit(f"Validated Ollama probe missed proxy-free receipt: {accepted.metadata}")
    if "Ollama HTTP proxy use: disabled" not in accepted.output:
        raise SystemExit(f"Validated Ollama probe missed proxy-free status text: {accepted.output}")

    class RedirectOpener:
        def open(self, request, timeout: float):
            raise HTTPError(request.full_url, 302, "private redirect target", {}, None)

    def redirect_probe():
        return model_provider.probe_ollama_models(opener=RedirectOpener())

    with (
        patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:22542"}, clear=True),
        patch.object(model_status, "probe_ollama_models", redirect_probe),
    ):
        redirected = handler({})
    if redirected.metadata.get("ollama_reachable") is not False:
        raise SystemExit(f"Redirecting Ollama probe was incorrectly reachable: {redirected.metadata}")
    if redirected.metadata.get("ollama_exception_type") != "HTTPError":
        raise SystemExit(f"Redirecting Ollama probe lost content-free exception type: {redirected.metadata}")
    if "ollama_probe_redirect_blocked" not in redirected.output:
        raise SystemExit(f"Redirecting Ollama probe lost stable diagnostic: {redirected.output}")
    if "private redirect target" in redirected.output + repr(redirected.metadata):
        raise SystemExit("Redirecting Ollama probe leaked redirect content")


def assert_ollama_no_cloud_policy_matrix(root: Path) -> None:
    config = JarvisConfig(
        data_dir=root / "ollama-no-cloud-policy",
        db_path=root / "ollama-no-cloud-policy" / "jarvis.sqlite",
        obsidian_vault=root / "ollama-no-cloud-policy" / "Vault",
        obsidian_root="Jarvis",
        chat_model="jarvis-v2-policy-smoke-chat",
        planner_model="jarvis-v2-policy-smoke-planner",
        use_model_planner=True,
    )
    handler = model_status.make_model_status_tool(config)

    policy_probe = (
        True,
        ["jarvis-v2-policy-smoke-chat", "jarvis-v2-policy-smoke-planner"],
        "ollama_probe_ok",
        "",
    )

    invalid_raw_value = "invalid-private-no-cloud-policy-value"
    invalid_consent_value = "invalid-private-unverified-consent-value"
    cases = [
        ("absent", None, None, False, True, False, False, True, False, False, "cloud_mode_not_disabled"),
        ("no-cloud-only", "1", None, True, True, True, False, True, False, False, "unverified_daemon_consent_required"),
        ("explicit-consent", "1", "1", True, True, True, True, True, True, True, "unverified_local_only_consent_granted"),
        ("invalid-consent", "1", invalid_consent_value, True, True, True, True, False, False, False, "invalid_unverified_context_consent"),
        ("invalid-no-cloud", invalid_raw_value, "1", True, False, False, True, True, True, False, "invalid_no_cloud_setting"),
    ]
    for (
        label,
        raw_value,
        consent_value,
        configured,
        valid,
        requested,
        consent_configured,
        consent_valid,
        consent_allowed,
        context_allowed,
        diagnostic,
    ) in cases:
        environment = {"OLLAMA_HOST": "127.0.0.1:22641"}
        if raw_value is not None:
            environment["OLLAMA_NO_CLOUD"] = raw_value
        if consent_value is not None:
            environment["JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT"] = consent_value
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(model_status, "probe_ollama_models", return_value=policy_probe),
        ):
            result = handler({})
        expected_metadata = {
            "ollama_no_cloud_configured": configured,
            "ollama_no_cloud_valid": valid,
            "ollama_no_cloud_requested": requested,
            "ollama_unverified_context_consent_configured": consent_configured,
            "ollama_unverified_context_consent_valid": consent_valid,
            "ollama_unverified_context_consent_allowed": consent_allowed,
            "ollama_personal_context_allowed": context_allowed,
            "ollama_execution_locality_verified": False,
            "ollama_on_device_model_execution_verified": False,
            "ollama_cloud_features_disabled_verified": False,
            "ollama_cloud_policy_diagnostic": diagnostic,
            "stored_personal_context_stays_local": not context_allowed,
            "stored_personal_context_eligible_for_ollama": context_allowed,
            "stored_personal_context_execution_locality_unknown": context_allowed,
            "remote_personal_context_policy": (
                "local_provider_unverified_context_consent" if context_allowed else "local_provider_stateless"
            ),
        }
        wrong_metadata = {
            key: (result.metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if result.metadata.get(key) != expected
        }
        if wrong_metadata:
            raise SystemExit(f"Ollama no-cloud {label} policy metadata drifted: {wrong_metadata} / {result.metadata}")
        expected_context_line = (
            "stored personal context to Ollama: eligible for the loopback daemon under explicit unverified consent"
            if context_allowed
            else "stored personal context to Ollama: withheld; current-message-only mode"
        )
        if expected_context_line not in result.output:
            raise SystemExit(f"Ollama no-cloud {label} policy output drifted: {result.output}")
        for private_value in [invalid_raw_value, invalid_consent_value]:
            if private_value in result.output + " " + repr(result.metadata):
                raise SystemExit(f"Ollama policy raw value leaked for {label}")
        for expected in [
            "configuration request only; not a daemon attestation",
            "Ollama daemon cloud-disabled state: unknown",
            "Ollama model execution locality: unknown",
        ]:
            if expected not in result.output:
                raise SystemExit(f"Ollama policy status missed {expected!r}: {result.output}")

    cloud_alias_config = JarvisConfig(
        data_dir=root / "ollama-cloud-alias",
        db_path=root / "ollama-cloud-alias" / "jarvis.sqlite",
        obsidian_vault=root / "ollama-cloud-alias" / "Vault",
        obsidian_root="Jarvis",
        chat_model="kimi-k2-thinking",
        planner_model="jarvis-v2-policy-smoke-planner",
        use_model_planner=False,
    )

    with (
        patch.dict(
            os.environ,
            {
                "OLLAMA_HOST": "127.0.0.1:22642",
                "OLLAMA_NO_CLOUD": "1",
                "JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT": "1",
            },
            clear=True,
        ),
        patch.object(
            model_status,
            "probe_ollama_models",
            return_value=(True, ["kimi-k2-thinking"], "ollama_probe_ok", ""),
        ),
    ):
        cloud_alias = model_status.make_model_status_tool(cloud_alias_config)({})
    if cloud_alias.metadata.get("available_model_count") != 1:
        raise SystemExit(f"Cloud alias smoke fixture was not observed by the HTTP probe: {cloud_alias.metadata}")
    for key in ["chat_ready", "model_routing_ready"]:
        if cloud_alias.metadata.get(key) is not False:
            raise SystemExit(f"Configured cloud alias was incorrectly reported ready: {cloud_alias.metadata}")
    if cloud_alias.metadata.get("ollama_cloud_model_alias") is not True:
        raise SystemExit(f"Configured cloud alias was not identified by policy: {cloud_alias.metadata}")
    for expected in [
        "chat model: kimi-k2-thinking (needs attention)",
        "cloud aliases are blocked before client construction",
    ]:
        if expected not in cloud_alias.output:
            raise SystemExit(f"Cloud alias status missed policy text {expected!r}: {cloud_alias.output}")


def assert_invalid_provider_does_not_probe(root: Path) -> None:
    config = JarvisConfig(
        data_dir=root / "invalid-provider",
        db_path=root / "invalid-provider" / "jarvis.sqlite",
        obsidian_vault=root / "invalid-provider" / "Vault",
        obsidian_root="Jarvis",
        model_provider="invalid",
        chat_model="jarvis-v2-invalid-provider-chat",
        planner_model="jarvis-v2-invalid-provider-planner",
        use_model_planner=True,
    )
    probe_calls: list[object] = []

    def forbidden_probe(*args, **kwargs):
        probe_calls.append((args, kwargs))
        raise AssertionError("invalid model provider must not probe Ollama")

    with patch.object(model_status, "probe_ollama_models", forbidden_probe):
        result = model_status.make_model_status_tool(config)({})
        readiness = model_status._specialist_model_readiness(config, config.chat_model)
    if probe_calls:
        raise SystemExit(f"Invalid model provider probed Ollama: {probe_calls}")
    if result.metadata.get("model_provider") != "invalid":
        raise SystemExit(f"Invalid model provider was not identified: {result.metadata}")
    if result.metadata.get("stored_personal_context_stays_local") is not True:
        raise SystemExit(f"Invalid provider must keep stored context local: {result.metadata}")
    for key in ["model_provider_valid", "model_probe_attempted", "chat_ready", "model_routing_ready"]:
        if result.metadata.get(key) is not False:
            raise SystemExit(f"Invalid provider should expose {key}=False: {result.metadata}")
    if readiness.get("model_ready") is not False or readiness.get("model_configuration_ready") is not False:
        raise SystemExit(f"Invalid specialist provider was incorrectly ready: {readiness}")
    if "model probe: not attempted" not in result.output or "model execution: unknown" not in result.output:
        raise SystemExit(f"Invalid provider status missed fail-closed text: {result.output}")


def assert_safe_metadata(metadata: dict, label: str) -> None:
    if (
        metadata.get("calls_model")
        or metadata.get("executes_tools")
        or metadata.get("queues_approval")
        or metadata.get("approves_request")
        or metadata.get("dismisses_request")
        or metadata.get("writes_memory")
        or metadata.get("writes_files")
        or metadata.get("writes_notes")
        or metadata.get("reads_private_data")
        or metadata.get("reads_personal_data")
        or metadata.get("executes_side_effect")
        or metadata.get("external_side_effect")
        or metadata.get("controls_computer")
        or metadata.get("requires_approval")
        or metadata.get("authorizes_execution") is not False
        or metadata.get("authorizes_completion_claim") is not False
        or metadata.get("approval_granted") is not False
        or metadata.get("speaks")
    ):
        raise SystemExit(f"{label} should be read-only and side-effect free: {metadata}")


def assert_specialist_review_only_authority(metadata: dict, label: str) -> None:
    expected_true = ["draft_only", "requires_manual_send", "loads_without_execution"]
    expected_false = ["authorizes_execution", "authorizes_completion_claim", "approval_granted"]
    for key in expected_true:
        if metadata.get(key) is not True:
            raise SystemExit(f"{label} should expose {key}=True for review-only handling: {metadata}")
    for key in expected_false:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} should expose {key}=False and never grant authority: {metadata}")


def assert_specialist_proof_lane(metadata: dict, label: str) -> None:
    for key in ["target_model", "fallback_lane", "proof_target", "stop_condition"]:
        if not metadata.get(key):
            raise SystemExit(f"{label} missed specialist proof-lane metadata {key!r}: {metadata}")
    if not str(metadata["proof_target"]).startswith("specialist route quality:"):
        raise SystemExit(f"{label} proof target should point to route quality: {metadata}")
    if "bounded handoff receipt" not in str(metadata["fallback_lane"]):
        raise SystemExit(f"{label} fallback lane should preserve bounded handoff behavior: {metadata}")
    if "not proven" not in str(metadata["stop_condition"]):
        raise SystemExit(f"{label} stop condition should require proof before reliance: {metadata}")


def assert_specialist_fresh_review_boundary_token(metadata: dict, label: str) -> None:
    if len(str(metadata.get("specialist_fresh_review_boundary_token_sha256") or "")) != 64:
        raise SystemExit(f"{label} missed fresh-review boundary token: {metadata}")
    if metadata.get("specialist_fresh_review_boundary_token_present") is not True:
        raise SystemExit(f"{label} missed fresh-review boundary token presence: {metadata}")
    for key in [
        "specialist_fresh_review_boundary_token_authorizes_action_now",
        "specialist_fresh_review_boundary_token_authorizes_model_call",
        "specialist_fresh_review_boundary_token_authorizes_tool_execution",
        "specialist_fresh_review_boundary_token_authorizes_approval",
        "specialist_fresh_review_boundary_token_authorizes_personal_data_read",
        "specialist_fresh_review_boundary_token_authorizes_external_side_effect",
        "specialist_fresh_review_boundary_token_authorizes_fresh_review",
        "specialist_fresh_review_boundary_token_reusable_for_next_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} fresh-review boundary token should report {key}=False: {metadata}")
    if metadata.get("next_specialist_review_requires_new_fresh_review_boundary_token") is not True:
        raise SystemExit(f"{label} missed next fresh-review boundary token requirement: {metadata}")
    expected = _specialist_fresh_review_boundary_token_sha256(
        request=str(metadata.get("request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        stage_rows=list(metadata.get("stage_rows") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if metadata.get("specialist_fresh_review_boundary_token_sha256") != expected:
        raise SystemExit(f"{label} fresh-review boundary token did not match recomputed contract: {metadata}")
    tampered_stage_rows = [dict(row) for row in list(metadata.get("stage_rows") or [])]
    if tampered_stage_rows:
        tampered_stage_rows[0]["authorizes_tool_execution"] = True
    tampered = _specialist_fresh_review_boundary_token_sha256(
        request=str(metadata.get("request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        stage_rows=tampered_stage_rows,
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_stage_rows and tampered == expected:
        raise SystemExit(f"{label} fresh-review boundary token did not bind stage authorization flags: {metadata}")
    tampered_stage_proofs = [dict(row) for row in list(metadata.get("stage_rows") or [])]
    if tampered_stage_proofs:
        tampered_stage_proofs[0]["proof"] = "specialist router contract: tampered proof command"
    tampered_proof = _specialist_fresh_review_boundary_token_sha256(
        request=str(metadata.get("request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        stage_rows=tampered_stage_proofs,
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_stage_proofs and tampered_proof == expected:
        raise SystemExit(f"{label} fresh-review boundary token did not bind stage proof commands: {metadata}")


def assert_specialist_cycle_ledger_token(metadata: dict, label: str) -> None:
    token = str(metadata.get("specialist_cycle_ledger_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed specialist cycle ledger token: {metadata}")
    if metadata.get("specialist_cycle_ledger_token_present") is not True:
        raise SystemExit(f"{label} missed cycle ledger token presence flag: {metadata}")
    for key in [
        "specialist_cycle_ledger_token_authorizes_model_call",
        "specialist_cycle_ledger_token_authorizes_tool_execution",
        "specialist_cycle_ledger_token_authorizes_approval",
        "specialist_cycle_ledger_token_authorizes_personal_data_read",
        "specialist_cycle_ledger_token_authorizes_external_side_effect",
        "specialist_cycle_ledger_token_authorizes_completion_claim",
        "specialist_cycle_ledger_token_reusable_for_next_specialist_review",
        "specialist_post_run_closure_token_authorizes_model_call",
        "specialist_post_run_closure_token_authorizes_tool_execution",
        "specialist_post_run_closure_token_authorizes_approval",
        "specialist_post_run_closure_token_authorizes_personal_data_read",
        "specialist_post_run_closure_token_authorizes_external_side_effect",
        "specialist_post_run_closure_token_authorizes_completion_claim",
        "specialist_post_run_closure_token_reusable_for_next_specialist_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} cycle ledger/post-run token should report {key}=False: {metadata}")
    if metadata.get("next_specialist_review_requires_new_cycle_ledger_token") is not True:
        raise SystemExit(f"{label} missed next cycle-ledger token requirement: {metadata}")
    if metadata.get("next_specialist_review_requires_new_post_run_closure_token") is not True:
        raise SystemExit(f"{label} missed next post-run closure token requirement: {metadata}")
    if len(str(metadata.get("specialist_post_run_closure_token_sha256") or "")) != 64:
        raise SystemExit(f"{label} missed post-run closure token: {metadata}")
    if metadata.get("specialist_post_run_closure_token_present") is not True:
        raise SystemExit(f"{label} missed post-run closure token presence: {metadata}")

    expected = _specialist_cycle_ledger_token_sha256(
        request=str(metadata.get("request") or ""),
        raw_request=str(metadata.get("raw_request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
        specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
        runtime_trace_sha256=str(metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(metadata.get("completion_claim_sha256") or ""),
        action_contract_scorecard_rows=list(metadata.get("action_contract_scorecard_rows") or []),
        dry_run_scorecard_rows=list(metadata.get("dry_run_scorecard_rows") or []),
        completion_scorecard_rows=list(metadata.get("completion_scorecard_rows") or []),
        post_run_proof_queue=list((metadata.get("post_run_closure_metadata") or {}).get("proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if token != expected:
        raise SystemExit(f"{label} cycle ledger token did not match recomputed contract: {metadata}")

    tampered_hash = _specialist_cycle_ledger_token_sha256(
        request=str(metadata.get("request") or ""),
        raw_request=str(metadata.get("raw_request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
        specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
        runtime_trace_sha256="0" * 64,
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(metadata.get("completion_claim_sha256") or ""),
        action_contract_scorecard_rows=list(metadata.get("action_contract_scorecard_rows") or []),
        dry_run_scorecard_rows=list(metadata.get("dry_run_scorecard_rows") or []),
        completion_scorecard_rows=list(metadata.get("completion_scorecard_rows") or []),
        post_run_proof_queue=list((metadata.get("post_run_closure_metadata") or {}).get("proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if metadata.get("runtime_trace_sha256") != "0" * 64 and tampered_hash == expected:
        raise SystemExit(f"{label} cycle ledger token did not bind post-run artifact hashes: {metadata}")

    tampered_rows = [dict(row) for row in list(metadata.get("stage_rows") or [])]
    if tampered_rows:
        tampered_rows[0]["authorizes_tool_execution"] = True
    tampered_row_token = _specialist_cycle_ledger_token_sha256(
        request=str(metadata.get("request") or ""),
        raw_request=str(metadata.get("raw_request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=tampered_rows,
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
        specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
        runtime_trace_sha256=str(metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(metadata.get("completion_claim_sha256") or ""),
        action_contract_scorecard_rows=list(metadata.get("action_contract_scorecard_rows") or []),
        dry_run_scorecard_rows=list(metadata.get("dry_run_scorecard_rows") or []),
        completion_scorecard_rows=list(metadata.get("completion_scorecard_rows") or []),
        post_run_proof_queue=list((metadata.get("post_run_closure_metadata") or {}).get("proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_rows and tampered_row_token == expected:
        raise SystemExit(f"{label} cycle ledger token did not bind stage authorization flags: {metadata}")

    tampered_queue = list((metadata.get("post_run_closure_metadata") or {}).get("proof_queue") or [])
    if tampered_queue:
        tampered_queue[0] = "runtime trace receipt: tampered run"
    tampered_queue_token = _specialist_cycle_ledger_token_sha256(
        request=str(metadata.get("request") or ""),
        raw_request=str(metadata.get("raw_request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
        specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
        runtime_trace_sha256=str(metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(metadata.get("completion_claim_sha256") or ""),
        action_contract_scorecard_rows=list(metadata.get("action_contract_scorecard_rows") or []),
        dry_run_scorecard_rows=list(metadata.get("dry_run_scorecard_rows") or []),
        completion_scorecard_rows=list(metadata.get("completion_scorecard_rows") or []),
        post_run_proof_queue=tampered_queue,
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_queue and tampered_queue_token == expected:
        raise SystemExit(f"{label} cycle ledger token did not bind post-run proof queue: {metadata}")

    tampered_closure_token = _specialist_cycle_ledger_token_sha256(
        request=str(metadata.get("request") or ""),
        raw_request=str(metadata.get("raw_request") or ""),
        cycle_state=str(metadata.get("cycle_state") or ""),
        stage_rows=list(metadata.get("stage_rows") or []),
        fresh_review_preflight_queue=list(metadata.get("fresh_review_preflight_queue") or []),
        fresh_review_contract_rows=list(metadata.get("fresh_review_contract_rows") or []),
        required_commands=list(metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
        specialist_post_run_closure_token_sha256="0" * 64,
        runtime_trace_sha256=str(metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(metadata.get("completion_claim_sha256") or ""),
        action_contract_scorecard_rows=list(metadata.get("action_contract_scorecard_rows") or []),
        dry_run_scorecard_rows=list(metadata.get("dry_run_scorecard_rows") or []),
        completion_scorecard_rows=list(metadata.get("completion_scorecard_rows") or []),
        post_run_proof_queue=list((metadata.get("post_run_closure_metadata") or {}).get("proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if metadata.get("specialist_post_run_closure_token_sha256") != "0" * 64 and tampered_closure_token == expected:
        raise SystemExit(f"{label} cycle ledger token did not bind post-run closure token: {metadata}")


def assert_runtime_review_boundary_token(metadata: dict, label: str) -> None:
    if len(str(metadata.get("runtime_review_boundary_token_sha256") or "")) != 64:
        raise SystemExit(f"{label} missed runtime-review boundary token: {metadata}")
    if metadata.get("runtime_review_boundary_token_present") is not True:
        raise SystemExit(f"{label} missed runtime-review boundary token presence: {metadata}")
    for key in [
        "runtime_review_boundary_token_authorizes_model_call",
        "runtime_review_boundary_token_authorizes_tool_execution",
        "runtime_review_boundary_token_authorizes_approval",
        "runtime_review_boundary_token_authorizes_personal_data_read",
        "runtime_review_boundary_token_authorizes_external_side_effect",
        "runtime_review_boundary_token_authorizes_completion_claim",
        "runtime_review_boundary_token_bypasses_post_run_proof",
        "runtime_review_boundary_token_reusable_for_next_specialist_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} runtime-review boundary token should report {key}=False: {metadata}")
    if metadata.get("next_specialist_review_requires_new_runtime_review_boundary_token") is not True:
        raise SystemExit(f"{label} missed next runtime-review boundary token requirement: {metadata}")


def assert_route_to_runtime_contract_token(metadata: dict, label: str) -> None:
    token = str(metadata.get("route_to_runtime_contract_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed route-to-runtime contract token: {metadata}")
    if metadata.get("route_to_runtime_contract_token_present") is not True:
        raise SystemExit(f"{label} missed route-to-runtime contract token presence: {metadata}")
    for key in [
        "route_to_runtime_contract_token_authorizes_model_call",
        "route_to_runtime_contract_token_authorizes_tool_execution",
        "route_to_runtime_contract_token_authorizes_approval",
        "route_to_runtime_contract_token_authorizes_personal_data_read",
        "route_to_runtime_contract_token_authorizes_external_side_effect",
        "route_to_runtime_contract_token_authorizes_completion_claim",
        "route_to_runtime_contract_token_reusable_for_next_specialist_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} route-to-runtime token should report {key}=False: {metadata}")
    if metadata.get("next_specialist_review_requires_new_route_to_runtime_contract_token") is not True:
        raise SystemExit(f"{label} missed next route-to-runtime token requirement: {metadata}")


def assert_specialist_post_run_closure_token(metadata: dict, label: str) -> None:
    token = str(metadata.get("specialist_post_run_closure_token_sha256") or "")
    source_metadata = metadata
    nested_source = metadata.get("post_run_closure_metadata")
    if isinstance(nested_source, dict) and nested_source.get("closure_state"):
        source_metadata = nested_source
        token = str(token or source_metadata.get("specialist_post_run_closure_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed post-run closure token: {metadata}")
    if metadata.get("specialist_post_run_closure_token_present") is not True:
        raise SystemExit(f"{label} missed post-run closure token presence: {metadata}")
    for key in [
        "specialist_post_run_closure_token_authorizes_model_call",
        "specialist_post_run_closure_token_authorizes_tool_execution",
        "specialist_post_run_closure_token_authorizes_approval",
        "specialist_post_run_closure_token_authorizes_personal_data_read",
        "specialist_post_run_closure_token_authorizes_external_side_effect",
        "specialist_post_run_closure_token_authorizes_completion_claim",
        "specialist_post_run_closure_token_reusable_for_next_specialist_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} post-run closure token should report {key}=False: {metadata}")
    if metadata.get("next_specialist_review_requires_new_post_run_closure_token") is not True:
        raise SystemExit(f"{label} missed next post-run closure token requirement: {metadata}")

    expected = _specialist_post_run_closure_token_sha256(
        request=str(source_metadata.get("request") or ""),
        raw_request=str(source_metadata.get("raw_request") or ""),
        closure_state=str(source_metadata.get("closure_state") or ""),
        missing=list(source_metadata.get("missing") or []),
        required_commands=list(source_metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(source_metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(source_metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(source_metadata.get("route_to_runtime_contract_token_sha256") or ""),
        runtime_trace_sha256=str(source_metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(source_metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(source_metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(source_metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(source_metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(source_metadata.get("completion_claim_sha256") or ""),
        post_run_artifact_hashes_present=bool(source_metadata.get("post_run_artifact_hashes_present")),
        blockers_cleared=bool(source_metadata.get("blockers_cleared")),
        next_command=str(source_metadata.get("next_command") or ""),
    )
    if token != expected:
        raise SystemExit(f"{label} post-run closure token did not match recomputed contract: {metadata}")

    tampered_hash = _specialist_post_run_closure_token_sha256(
        request=str(source_metadata.get("request") or ""),
        raw_request=str(source_metadata.get("raw_request") or ""),
        closure_state=str(source_metadata.get("closure_state") or ""),
        missing=list(source_metadata.get("missing") or []),
        required_commands=list(source_metadata.get("required_commands") or []),
        specialist_review_token_sha256=str(source_metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(source_metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(source_metadata.get("route_to_runtime_contract_token_sha256") or ""),
        runtime_trace_sha256="0" * 64,
        verification_receipt_sha256=str(source_metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(source_metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(source_metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(source_metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(source_metadata.get("completion_claim_sha256") or ""),
        post_run_artifact_hashes_present=bool(source_metadata.get("post_run_artifact_hashes_present")),
        blockers_cleared=bool(source_metadata.get("blockers_cleared")),
        next_command=str(source_metadata.get("next_command") or ""),
    )
    if source_metadata.get("runtime_trace_sha256") != "0" * 64 and tampered_hash == expected:
        raise SystemExit(f"{label} post-run closure token did not bind artifact hashes: {metadata}")

    tampered_commands = list(source_metadata.get("required_commands") or [])
    if tampered_commands:
        tampered_commands[-1] = "completion claim gate: tampered specialist execution"
    tampered_command_token = _specialist_post_run_closure_token_sha256(
        request=str(source_metadata.get("request") or ""),
        raw_request=str(source_metadata.get("raw_request") or ""),
        closure_state=str(source_metadata.get("closure_state") or ""),
        missing=list(source_metadata.get("missing") or []),
        required_commands=tampered_commands,
        specialist_review_token_sha256=str(source_metadata.get("specialist_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(source_metadata.get("runtime_review_boundary_token_sha256") or ""),
        route_to_runtime_contract_token_sha256=str(source_metadata.get("route_to_runtime_contract_token_sha256") or ""),
        runtime_trace_sha256=str(source_metadata.get("runtime_trace_sha256") or ""),
        verification_receipt_sha256=str(source_metadata.get("verification_receipt_sha256") or ""),
        execution_audit_sha256=str(source_metadata.get("execution_audit_sha256") or ""),
        execution_recovery_sha256=str(source_metadata.get("execution_recovery_sha256") or ""),
        after_action_learning_sha256=str(source_metadata.get("after_action_learning_sha256") or ""),
        completion_claim_sha256=str(source_metadata.get("completion_claim_sha256") or ""),
        post_run_artifact_hashes_present=bool(source_metadata.get("post_run_artifact_hashes_present")),
        blockers_cleared=bool(source_metadata.get("blockers_cleared")),
        next_command=str(source_metadata.get("next_command") or ""),
    )
    if tampered_commands and tampered_command_token == expected:
        raise SystemExit(f"{label} post-run closure token did not bind required commands: {metadata}")


def assert_route_to_runtime_contract_token_boundary(metadata: dict, label: str) -> None:
    expected = _specialist_route_to_runtime_contract_token_sha256(
        request=str(metadata.get("request") or ""),
        handoff_state=str(metadata.get("handoff_state") or ""),
        proposed_tool=str(metadata.get("proposed_tool") or ""),
        proposed_arguments=str(metadata.get("proposed_arguments") or ""),
        verification=str(metadata.get("verification") or ""),
        primary_specialist=str(metadata.get("primary_specialist") or ""),
        target_model=str(metadata.get("target_model") or ""),
        risk_level=str(metadata.get("risk_level") or ""),
        completion_state=str(metadata.get("completion_state") or ""),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        action_proposal_review_token_sha256=str(metadata.get("action_proposal_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        runtime_review_contract_rows=list(metadata.get("runtime_review_contract_rows") or []),
        pre_run_proof_queue=list(metadata.get("pre_run_proof_queue") or []),
        post_run_proof_queue=list(metadata.get("post_run_proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if metadata.get("route_to_runtime_contract_token_sha256") != expected:
        raise SystemExit(f"{label} route-to-runtime token did not match recomputed contract: {metadata}")
    tampered = _specialist_route_to_runtime_contract_token_sha256(
        request=str(metadata.get("request") or ""),
        handoff_state=str(metadata.get("handoff_state") or ""),
        proposed_tool=str(metadata.get("proposed_tool") or ""),
        proposed_arguments=str(metadata.get("proposed_arguments") or ""),
        verification=str(metadata.get("verification") or ""),
        primary_specialist=str(metadata.get("primary_specialist") or ""),
        target_model=str(metadata.get("target_model") or ""),
        risk_level=str(metadata.get("risk_level") or ""),
        completion_state=str(metadata.get("completion_state") or ""),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        action_proposal_review_token_sha256=str(metadata.get("action_proposal_review_token_sha256") or ""),
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        runtime_review_contract_rows=list(metadata.get("runtime_review_contract_rows") or []),
        pre_run_proof_queue=list(metadata.get("pre_run_proof_queue") or []),
        post_run_proof_queue=list(metadata.get("post_run_proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
        authorizes_tool_execution=True,
    )
    if tampered == expected:
        raise SystemExit(f"{label} route-to-runtime token did not bind authorization flags: {metadata}")
    tampered_action_review = _specialist_route_to_runtime_contract_token_sha256(
        request=str(metadata.get("request") or ""),
        handoff_state=str(metadata.get("handoff_state") or ""),
        proposed_tool=str(metadata.get("proposed_tool") or ""),
        proposed_arguments=str(metadata.get("proposed_arguments") or ""),
        verification=str(metadata.get("verification") or ""),
        primary_specialist=str(metadata.get("primary_specialist") or ""),
        target_model=str(metadata.get("target_model") or ""),
        risk_level=str(metadata.get("risk_level") or ""),
        completion_state=str(metadata.get("completion_state") or ""),
        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
        action_proposal_review_token_sha256="0" * 64,
        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
        runtime_review_contract_rows=list(metadata.get("runtime_review_contract_rows") or []),
        pre_run_proof_queue=list(metadata.get("pre_run_proof_queue") or []),
        post_run_proof_queue=list(metadata.get("post_run_proof_queue") or []),
        next_command=str(metadata.get("next_command") or ""),
    )
    if tampered_action_review == expected:
        raise SystemExit(f"{label} route-to-runtime token did not bind action proposal review token: {metadata}")
    if metadata.get("route_to_runtime_contract_binds_action_proposal_review_token") is not True:
        raise SystemExit(f"{label} missed action-proposal review binding flag: {metadata}")


def assert_handoff_scorecard(metadata: dict, label: str) -> None:
    scorecard = metadata.get("handoff_scorecard")
    if not isinstance(scorecard, dict):
        raise SystemExit(f"{label} missed handoff scorecard metadata: {metadata}")
    score = scorecard.get("score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise SystemExit(f"{label} handoff score should be a bounded integer: {metadata}")
    if metadata.get("handoff_score") != score:
        raise SystemExit(f"{label} handoff score alias diverged: {metadata}")
    if scorecard.get("grade") not in {"strong", "review", "weak"}:
        raise SystemExit(f"{label} handoff scorecard missed grade: {metadata}")
    if metadata.get("handoff_score_grade") != scorecard.get("grade"):
        raise SystemExit(f"{label} handoff score grade alias diverged: {metadata}")
    if scorecard.get("max_points") != 100 or metadata.get("handoff_scorecard_max_points") != 100:
        raise SystemExit(f"{label} handoff scorecard missed max points: {metadata}")
    for key in ["route_signal_points", "route_margin_points", "verifier_points", "safety_points"]:
        if not isinstance(scorecard.get(key), int):
            raise SystemExit(f"{label} handoff scorecard missed integer component {key}: {metadata}")


def assert_action_proposal_scorecard(metadata: dict, label: str) -> None:
    scorecard = metadata.get("action_proposal_scorecard")
    if not isinstance(scorecard, dict):
        raise SystemExit(f"{label} missed action proposal scorecard metadata: {metadata}")
    score = scorecard.get("score")
    if not isinstance(score, int) or not 0 <= score <= 100:
        raise SystemExit(f"{label} action proposal score should be a bounded integer: {metadata}")
    if metadata.get("action_proposal_score") != score:
        raise SystemExit(f"{label} action proposal score alias diverged: {metadata}")
    if scorecard.get("grade") not in {"ready_for_operator_review", "review_required", "held"}:
        raise SystemExit(f"{label} action proposal scorecard missed grade: {metadata}")
    if metadata.get("action_proposal_grade") != scorecard.get("grade"):
        raise SystemExit(f"{label} action proposal grade alias diverged: {metadata}")
    if scorecard.get("max_points") != 100 or metadata.get("action_proposal_scorecard_max_points") != 100:
        raise SystemExit(f"{label} action proposal scorecard missed max points: {metadata}")
    rows = metadata.get("action_proposal_scorecard_rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{label} missed action proposal scorecard rows: {metadata}")
    if metadata.get("action_proposal_scorecard_row_count") != len(rows):
        raise SystemExit(f"{label} action proposal scorecard row count diverged: {metadata}")
    if len(rows) not in {2, 6}:
        raise SystemExit(f"{label} action proposal scorecard rows should have expected shape: {metadata}")
    if len(rows) == 6:
        if metadata.get("action_proposal_scorecard_shape_ready") is not True:
            raise SystemExit(f"{label} action proposal scorecard missed production shape-ready flag: {metadata}")
        if not _specialist_action_proposal_scorecard_shape_ready(rows):
            raise SystemExit(f"{label} action proposal scorecard failed production shape validator: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["item"] = "tampered_tool_registry"
        if _specialist_action_proposal_scorecard_shape_ready(tampered_rows):
            raise SystemExit(f"{label} action proposal scorecard validator accepted tampered item: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["authorizes_tool_execution"] = True
        if _specialist_action_proposal_scorecard_shape_ready(tampered_rows):
            raise SystemExit(f"{label} action proposal scorecard validator accepted authorizing row: {metadata}")
        expected_items = {
            "tool_registry",
            "exact_arguments",
            "verification_expectation",
            "permission_boundary",
            "proof_lane",
            "execution_lock",
        }
        if {row.get("item") for row in rows} != expected_items:
            raise SystemExit(f"{label} action proposal scorecard row items diverged: {metadata}")
        if sum(int(row.get("max_points") or 0) for row in rows) != 100:
            raise SystemExit(f"{label} action proposal scorecard rows should sum to 100 max points: {metadata}")
        if sum(int(row.get("points") or 0) for row in rows) != score:
            raise SystemExit(f"{label} action proposal scorecard rows should sum to score: {metadata}")
        for row in rows:
            for key in [
                "authorizes_model_call",
                "authorizes_tool_execution",
                "authorizes_approval",
                "authorizes_personal_data_read",
                "authorizes_external_side_effect",
                "authorizes_executable_action",
                "reusable_for_next_review",
            ]:
                if row.get(key) is not False:
                    raise SystemExit(f"{label} action proposal scorecard row should report {key}=False: {row}")
    else:
        if metadata.get("action_proposal_scorecard_shape_ready") is not True:
            raise SystemExit(f"{label} combined action proposal scorecard missed production shape-ready flag: {metadata}")
        if not _specialist_combined_action_proposal_scorecard_shape_ready(rows):
            raise SystemExit(f"{label} combined action proposal scorecard failed production shape validator: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["row_count"] = 5
        if _specialist_combined_action_proposal_scorecard_shape_ready(tampered_rows):
            raise SystemExit(f"{label} combined action proposal scorecard validator accepted row-count tampering: {metadata}")
        tampered_rows = [dict(row) for row in rows]
        tampered_rows[0]["authorizes_tool_execution"] = True
        if _specialist_combined_action_proposal_scorecard_shape_ready(tampered_rows):
            raise SystemExit(f"{label} combined action proposal scorecard validator accepted authorizing row: {metadata}")
        expected_items = {"action_contract_scorecard", "tool_dry_run_scorecard"}
        if {row.get("item") for row in rows} != expected_items:
            raise SystemExit(f"{label} combined action proposal scorecard rows diverged: {metadata}")
        if min(int(row.get("score") or 0) for row in rows) != score:
            raise SystemExit(f"{label} combined action proposal scorecard rows should explain score: {metadata}")
    if metadata.get("action_proposal_scorecard_required_rows_ready") is not all(
        bool(row.get("passed", row.get("required_rows_ready", False))) for row in rows if row.get("required_before_execution")
    ):
        raise SystemExit(f"{label} action proposal scorecard readiness flag diverged: {metadata}")


def assert_action_proposal_review_token(metadata: dict, label: str) -> None:
    rows = list(metadata.get("action_proposal_scorecard_rows") or [])
    token = str(metadata.get("action_proposal_review_token_sha256") or "")
    if len(token) != 64:
        raise SystemExit(f"{label} missed action proposal review token: {metadata}")
    if metadata.get("action_proposal_review_token_present") is not True:
        raise SystemExit(f"{label} missed action proposal review token presence flag: {metadata}")
    for key in [
        "action_proposal_review_token_authorizes_model_call",
        "action_proposal_review_token_authorizes_tool_execution",
        "action_proposal_review_token_authorizes_approval",
        "action_proposal_review_token_authorizes_personal_data_read",
        "action_proposal_review_token_authorizes_external_side_effect",
        "action_proposal_review_token_authorizes_executable_action",
        "action_proposal_review_token_reusable_for_next_review",
    ]:
        if metadata.get(key) is not False:
            raise SystemExit(f"{label} action proposal review token should report {key}=False: {metadata}")
    if metadata.get("next_action_proposal_review_requires_new_token") is not True:
        raise SystemExit(f"{label} missed fresh action proposal review token requirement: {metadata}")

    review_state = str(metadata.get("completion_state") or metadata.get("contract_state") or metadata.get("dry_run_state") or "")
    expected = _specialist_action_proposal_review_token_sha256(
        request=str(metadata.get("request") or ""),
        proposed_tool=str(metadata.get("proposed_tool") or ""),
        proposed_arguments=str(metadata.get("proposed_arguments") or ""),
        verification=str(metadata.get("verification") or ""),
        primary_specialist=str(metadata.get("primary_specialist") or ""),
        target_model=str(metadata.get("target_model") or ""),
        risk_level=str(metadata.get("risk_level") or ""),
        review_state=review_state,
        next_command=str(metadata.get("next_command") or ""),
        scorecard_rows=rows,
    )
    if token != expected:
        raise SystemExit(f"{label} action proposal review token did not match recomputed contract: {metadata}")
    tampered = _specialist_action_proposal_review_token_sha256(
        request=str(metadata.get("request") or ""),
        proposed_tool=str(metadata.get("proposed_tool") or ""),
        proposed_arguments=str(metadata.get("proposed_arguments") or ""),
        verification=str(metadata.get("verification") or ""),
        primary_specialist=str(metadata.get("primary_specialist") or ""),
        target_model=str(metadata.get("target_model") or ""),
        risk_level=str(metadata.get("risk_level") or ""),
        review_state=review_state,
        next_command=str(metadata.get("next_command") or ""),
        scorecard_rows=rows,
        authorizes_executable_action=True,
    )
    if tampered == expected:
        raise SystemExit(f"{label} action proposal review token did not bind authorization flags: {metadata}")
    tampered_rows = [dict(row) for row in rows]
    if tampered_rows:
        tampered_rows[0]["authorizes_tool_execution"] = True
    tampered_row_token = _specialist_action_proposal_review_token_sha256(
        request=str(metadata.get("request") or ""),
        proposed_tool=str(metadata.get("proposed_tool") or ""),
        proposed_arguments=str(metadata.get("proposed_arguments") or ""),
        verification=str(metadata.get("verification") or ""),
        primary_specialist=str(metadata.get("primary_specialist") or ""),
        target_model=str(metadata.get("target_model") or ""),
        risk_level=str(metadata.get("risk_level") or ""),
        review_state=review_state,
        next_command=str(metadata.get("next_command") or ""),
        scorecard_rows=tampered_rows,
    )
    if tampered_row_token == expected:
        raise SystemExit(f"{label} action proposal review token did not bind row authorization flags: {metadata}")


def main() -> None:
    assert_model_status_metadata_bool_is_exact()
    assert_missing_config_recovery_is_actionable()
    with TemporaryDirectory(prefix="jarvis-model-status-") as temp:
        root = Path(temp)
        assert_ollama_http_probe_contract(root)
        assert_ollama_no_cloud_policy_matrix(root)
        assert_invalid_provider_does_not_probe(root)
        config = JarvisConfig(
            data_dir=root,
            db_path=root / "jarvis.sqlite",
            obsidian_vault=root / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-missing-smoke-model",
            planner_model="jarvis-v2-missing-smoke-planner",
            use_model_planner=True,
        )
        runtime = JarvisRuntime(config)
        runtime_trace_sha256 = "a" * 64
        verification_receipt_sha256 = "b" * 64
        execution_audit_sha256 = "c" * 64
        execution_recovery_sha256 = "d" * 64
        after_action_learning_sha256 = "e" * 64
        completion_claim_sha256 = "f" * 64
        for case in ["model routing status", "ollama status"]:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:1800])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in [
                "Jarvis model routing status",
                "chat model",
                "planner model",
                "model timeout",
                "planner timeout",
                "chat timeout",
                "Conversation latency proof",
                "Mixed-conversation acceptance target: chat p95 <= 8000ms",
                "JARVIS_CHAT_MAX_REPLY_TOKENS",
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
                "rerun the mixed-conversation proof and live_check",
                "ModelBackedPlanner",
                "fall back instead of hanging",
                "PermissionPolicy",
                "approval gates",
            ]:
                if expected not in result.response:
                    raise SystemExit(f"Model routing status missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("model_timeout_seconds") != config.model_timeout_seconds:
                raise SystemExit("Model routing status metadata missed model timeout.")
            if metadata.get("chat_timeout_seconds") != config.chat_timeout_seconds:
                raise SystemExit("Model routing status metadata missed chat timeout.")
            if metadata.get("chat_latency_tuning_available") is not True:
                raise SystemExit("Model routing status metadata missed chat latency tuning availability.")
            if metadata.get("chat_latency_target_p95_ms") != 8000:
                raise SystemExit("Model routing status metadata missed chat latency target.")
            if metadata.get("chat_latency_tuning_knobs") != [
                "JARVIS_CHAT_MAX_REPLY_TOKENS",
                "JARVIS_CHAT_MAX_HISTORY_MESSAGES",
            ]:
                raise SystemExit("Model routing status metadata missed bounded chat latency knobs.")
            if metadata.get("chat_latency_requires_fresh_mixed_conversation_proof") is not True:
                raise SystemExit("Model routing status metadata must require a fresh mixed-conversation proof.")
            if metadata.get("chat_latency_tuning_authorizes_completion_claim") is not False:
                raise SystemExit("Model routing status metadata must not authorize latency completion claims.")
            assert_safe_metadata(metadata, "Model routing status")

        disabled_planner_config = JarvisConfig(
            data_dir=root / "disabled-planner",
            db_path=root / "disabled-planner" / "jarvis.sqlite",
            obsidian_vault=root / "disabled-planner" / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-ready-smoke-chat",
            planner_model="jarvis-v2-missing-smoke-planner",
            use_model_planner=False,
        )
        disabled_planner_runtime = JarvisRuntime(disabled_planner_config)

        with patch.object(
            model_status,
            "probe_ollama_models",
            return_value=(True, ["jarvis-v2-ready-smoke-chat"], "ollama_probe_ok", ""),
        ):
            disabled_planner = disabled_planner_runtime.registry.get("model_routing_status").handler({})
        print("[ok] model routing status treats disabled planner model as standby")
        print(disabled_planner.output[:900])
        print()
        if not disabled_planner.ok:
            raise SystemExit("Model routing status should render when the model planner is disabled.")
        for expected in [
            "chat model: jarvis-v2-ready-smoke-chat (ready)",
            "planner model: jarvis-v2-missing-smoke-planner (standby (model planner disabled))",
            "Planner model is optional right now because model planner is disabled",
            "Model routing looks ready",
        ]:
            if expected not in disabled_planner.output:
                raise SystemExit(f"Disabled-planner model status missed expected text {expected!r}: {disabled_planner.output}")
        disabled_metadata = disabled_planner.metadata
        expected_disabled_flags = {
            "chat_ready": True,
            "planner_ready": False,
            "planner_model_required": False,
            "planner_model_available": False,
            "planner_routing_ready": True,
            "model_routing_ready": True,
        }
        for key, expected in expected_disabled_flags.items():
            if disabled_metadata.get(key) is not expected:
                raise SystemExit(f"Disabled-planner model status metadata {key} mismatch: {disabled_metadata}")
        assert_safe_metadata(disabled_metadata, "Disabled planner model routing status")

        # Real gap found live 2026-07-09: the Ollama model list reports a full
        # tag (e.g. "llama3.1:latest"), but config values are commonly written
        # without one (e.g. "llama3.1", Ollama's own implicit default tag).
        # The old bare `in` membership check treated a real, pulled, working
        # model as unavailable all session, reporting "needs attention" even
        # though the model was demonstrably working in live chat the whole
        # time. The other mocked cases above never exercise this because their
        # fake model name has no tag on either side of the comparison.
        implicit_tag_config = JarvisConfig(
            data_dir=root / "implicit-tag",
            db_path=root / "implicit-tag" / "jarvis.sqlite",
            obsidian_vault=root / "implicit-tag" / "Vault",
            obsidian_root="Jarvis",
            chat_model="jarvis-v2-tag-smoke-chat",
            planner_model="jarvis-v2-tag-smoke-planner",
            use_model_planner=True,
        )
        implicit_tag_runtime = JarvisRuntime(implicit_tag_config)

        with patch.object(
            model_status,
            "probe_ollama_models",
            return_value=(
                True,
                ["jarvis-v2-tag-smoke-chat:latest", "jarvis-v2-tag-smoke-planner:latest"],
                "ollama_probe_ok",
                "",
            ),
        ):
            implicit_tag = implicit_tag_runtime.registry.get("model_routing_status").handler({})
            implicit_specialist_readiness = model_status._specialist_model_readiness(
                implicit_tag_config,
                implicit_tag_config.chat_model,
            )
        print("[ok] model routing status matches configured models against implicit :latest tags")
        print(implicit_tag.output[:900])
        print()
        if not implicit_tag.ok:
            raise SystemExit("Model routing status should render for implicit-tag models.")
        for expected in [
            "chat model: jarvis-v2-tag-smoke-chat (ready)",
            "planner model: jarvis-v2-tag-smoke-planner (ready)",
            "Ollama requested context window on each generation call: 24,000 tokens",
            "This read-only status makes no Ollama generation call",
            "Ollama effective context support: unknown",
            "Ollama prompt truncation status: unverified",
            "installed model presence is tracked separately",
            "may increase local memory use and compute time",
        ]:
            if expected not in implicit_tag.output:
                raise SystemExit(f"Implicit-tag model status missed expected text {expected!r}: {implicit_tag.output}")
        implicit_tag_metadata = implicit_tag.metadata
        expected_implicit_tag_flags = {
            "chat_model_present": True,
            "chat_model_context_compatible": None,
            "chat_ready": True,
            "planner_model_present": True,
            "planner_model_context_compatible": None,
            "planner_ready": True,
            "planner_model_available": True,
            "planner_routing_ready": True,
            "model_routing_ready": True,
            "ollama_num_ctx_requested": 24_000,
            "ollama_num_ctx_request_configured": True,
            "ollama_num_ctx_status_probe_sent_generation_request": False,
            "ollama_num_ctx_effective": None,
            "ollama_num_ctx_effective_available": False,
            "ollama_prompt_truncated": None,
            "ollama_prompt_truncation_verified": False,
            "ollama_context_compatibility_verified": False,
            "ollama_larger_context_may_increase_local_resource_use": True,
        }
        for key, expected in expected_implicit_tag_flags.items():
            if key not in implicit_tag_metadata:
                raise SystemExit(
                    f"Implicit-tag model status metadata missed {key}: {implicit_tag_metadata}"
                )
            actual = implicit_tag_metadata.get(key)
            matches = actual is expected if expected is None or type(expected) is bool else actual == expected
            if not matches:
                raise SystemExit(f"Implicit-tag model status metadata {key} mismatch: {implicit_tag_metadata}")
        assert_safe_metadata(implicit_tag_metadata, "Implicit-tag model routing status")
        expected_specialist_context = {
            "model_ready": True,
            "model_configuration_ready": True,
            "model_present": True,
            "model_context_compatible": None,
            "ollama_num_ctx_requested": 24_000,
            "ollama_num_ctx_request_configured": True,
            "ollama_num_ctx_effective": None,
            "ollama_num_ctx_effective_available": False,
            "ollama_prompt_truncated": None,
            "ollama_prompt_truncation_verified": False,
            "ollama_context_compatibility_verified": False,
            "ollama_larger_context_may_increase_local_resource_use": True,
        }
        wrong_specialist_context = {
            key: implicit_specialist_readiness.get(key)
            for key, expected in expected_specialist_context.items()
            if implicit_specialist_readiness.get(key) != expected
        }
        if wrong_specialist_context:
            raise SystemExit(
                "Implicit-tag specialist readiness conflated model presence with context "
                f"compatibility: {wrong_specialist_context} / {implicit_specialist_readiness}"
            )

        with patch.object(
            model_status,
            "probe_ollama_models",
            return_value=(False, [], "ollama_probe_unavailable", "OSError"),
        ):
            failed_ollama = runtime.registry.get("model_routing_status").handler({})
        print("[ok] model routing status reports stable HTTP probe failures")
        print(failed_ollama.output[:900])
        print()
        if not failed_ollama.ok:
            raise SystemExit("Model routing status should still render when the Ollama HTTP probe fails.")
        for expected in ["ollama_probe_unavailable", "setup check", "Start Ollama", "model routing status"]:
            if expected not in failed_ollama.output:
                raise SystemExit(f"Model routing status missed stable HTTP probe guidance {expected!r}: {failed_ollama.output}")
        failed_metadata = failed_ollama.metadata
        if failed_metadata.get("ollama_exception_type") != "OSError" or failed_metadata.get("ollama_reachable") is not False:
            raise SystemExit(f"Model routing status missed Ollama HTTP probe diagnostics: {failed_metadata}")
        if failed_metadata.get("ollama_recovery_commands") != ["setup check", "model routing status"]:
            raise SystemExit(f"Model routing status missed ordered Ollama recovery commands: {failed_metadata}")
        if failed_metadata.get("ollama_retry_requires_setup_repair") is not True:
            raise SystemExit(f"Model routing status should require Ollama setup repair before retry: {failed_metadata}")
        if failed_metadata.get("authorizes_retry") is not False:
            raise SystemExit(f"Model routing status should not authorize its own retry: {failed_metadata}")
        assert_safe_metadata(failed_metadata, "Failed model routing status")

        private_invalid_payload = "invalid response near /\x55sers/example/private/model-cache"
        with patch.object(
            model_status,
            "probe_ollama_models",
            return_value=(False, [], "ollama_probe_invalid_response", "InvalidResponse"),
        ):
            invalid_response = runtime.registry.get("model_routing_status").handler({})
        print("[ok] model routing status keeps invalid HTTP response details content-free")
        print(invalid_response.output[:900])
        print()
        if not invalid_response.ok:
            raise SystemExit("Model routing status should still render for an invalid HTTP response.")
        if private_invalid_payload in invalid_response.output + repr(invalid_response.metadata):
            raise SystemExit(f"Model routing status leaked invalid HTTP response content: {invalid_response.output}")
        invalid_response_metadata = invalid_response.metadata
        if (
            invalid_response_metadata.get("ollama_reachable") is not False
            or invalid_response_metadata.get("ollama_exception_type") != "InvalidResponse"
        ):
            raise SystemExit(
                f"Model routing status invalid-response metadata should stay stable: {invalid_response_metadata}"
            )
        if invalid_response_metadata.get("ollama_recovery_commands") != ["setup check", "model routing status"]:
            raise SystemExit(f"Model routing status invalid response missed recovery commands: {invalid_response_metadata}")
        if invalid_response_metadata.get("authorizes_retry") is not False:
            raise SystemExit(f"Model routing status invalid response should not authorize retry: {invalid_response_metadata}")
        assert_safe_metadata(invalid_response_metadata, "Invalid-response model routing status")

        planner = RuleBasedPlanner()
        model_status_route_cases = [
            "model routing status",
            "model status please",
            "show model status",
            "show latest model status",
            "model routing please",
            "model routing status please",
            "show model routing status",
            "chat model please",
            "planner model please",
            "ollama status please",
            "local model status please",
            "model health please",
            "show model health",
            "ai model status please",
            "llm status please",
            "show latest llm status",
        ]
        for command in model_status_route_cases:
            plan = planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [("model_routing_status", {})] or plan.needs_model:
                raise SystemExit(f"Model routing status alias mismatch for {command!r}: {actual}, needs_model={plan.needs_model}")

        prompt_route_cases = {
            "model planner prompt preview please": (
                "model_planner_prompt_preview",
                {"request": "what should Jarvis do next"},
            ),
            "planner prompt preview please": (
                "model_planner_prompt_preview",
                {"request": "what should Jarvis do next"},
            ),
            "show model planner prompt preview": (
                "model_planner_prompt_preview",
                {"request": "what should Jarvis do next"},
            ),
            "show latest model planner prompt preview": (
                "model_planner_prompt_preview",
                {"request": "what should Jarvis do next"},
            ),
            "preview model planner prompt": (
                "model_planner_prompt_preview",
                {"request": "what should Jarvis do next"},
            ),
            "model planner prompt preview: text fixture saying hi please": (
                "model_planner_prompt_preview",
                {"request": "text fixture saying hi please"},
            ),
        }
        for command, (expected_tool, expected_args) in prompt_route_cases.items():
            plan = planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Model planner prompt preview route mismatch for {command!r}: {actual}")

        specialist_preflight_route_cases = {
            "specialist router contract please": (
                "specialist_router_contract",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist router contract": (
                "specialist_router_contract",
                {"request": "what should Jarvis do next"},
            ),
            "show latest specialist router contract": (
                "specialist_router_contract",
                {"request": "what should Jarvis do next"},
            ),
            "which specialist should handle send email to Sam saying hi": (
                "specialist_router_contract",
                {"request": "send email to Sam saying hi"},
            ),
            "route this to the right specialist: send email to Sam saying hi": (
                "specialist_router_contract",
                {"request": "send email to Sam saying hi"},
            ),
            "which brain handles send email to Sam saying hi": (
                "specialist_router_contract",
                {"request": "send email to Sam saying hi"},
            ),
            "specialist route quality please": (
                "specialist_route_quality",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist route quality": (
                "specialist_route_quality",
                {"request": "what should Jarvis do next"},
            ),
            "specialist execution readiness please": (
                "specialist_execution_readiness",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist execution readiness": (
                "specialist_execution_readiness",
                {"request": "what should Jarvis do next"},
            ),
            "is specialist ready for send email to Sam saying hi": (
                "specialist_execution_readiness",
                {"request": "send email to Sam saying hi"},
            ),
            "can a specialist draft send email to Sam saying hi": (
                "specialist_execution_readiness",
                {"request": "send email to Sam saying hi"},
            ),
            "brain readiness for send email to Sam saying hi": (
                "specialist_execution_readiness",
                {"request": "send email to Sam saying hi"},
            ),
            "specialist handoff receipt please": (
                "specialist_handoff_receipt",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist handoff receipt": (
                "specialist_handoff_receipt",
                {"request": "what should Jarvis do next"},
            ),
            "make a specialist handoff for send email to Sam saying hi": (
                "specialist_handoff_receipt",
                {"request": "send email to Sam saying hi"},
            ),
            "specialist proposal gate please": (
                "specialist_proposal_gate",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist proposal gate": (
                "specialist_proposal_gate",
                {"request": "what should Jarvis do next"},
            ),
            "specialist router contract: summarize this work please": (
                "specialist_router_contract",
                {"request": "summarize this work please"},
            ),
            "specialist route quality: summarize this work please": (
                "specialist_route_quality",
                {"request": "summarize this work please"},
            ),
        }
        for command, (expected_tool, expected_args) in specialist_preflight_route_cases.items():
            plan = planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Specialist preflight route mismatch for {command!r}: {actual}")

        specialist_handoff_quality_route_cases = {
            "specialist handoff quality gate please": (
                "specialist_handoff_quality_gate",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist handoff quality gate": (
                "specialist_handoff_quality_gate",
                {"request": "what should Jarvis do next"},
            ),
            "show latest specialist handoff quality gate": (
                "specialist_handoff_quality_gate",
                {"request": "what should Jarvis do next"},
            ),
            "specialist quality gate please": (
                "specialist_handoff_quality_gate",
                {"request": "what should Jarvis do next"},
            ),
            "show handoff quality gate": (
                "specialist_handoff_quality_gate",
                {"request": "what should Jarvis do next"},
            ),
            "show model handoff quality gate": (
                "specialist_handoff_quality_gate",
                {"request": "what should Jarvis do next"},
            ),
            "specialist handoff quality gate: summarize this work please": (
                "specialist_handoff_quality_gate",
                {"request": "summarize this work please"},
            ),
        }
        for command, (expected_tool, expected_args) in specialist_handoff_quality_route_cases.items():
            plan = planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Specialist handoff quality route mismatch for {command!r}: {actual}")

        route_quality_guard = planner.plan("specialist quality please")
        actual_route_quality_guard = [(action.tool_name, action.args) for action in route_quality_guard.actions]
        if actual_route_quality_guard != [("specialist_route_quality", {"request": "what should Jarvis do next"})]:
            raise SystemExit(f"Specialist quality without gate should stay on route quality: {actual_route_quality_guard}")

        handoff_receipt_guard = planner.plan("show specialist handoff receipt")
        actual_handoff_receipt_guard = [(action.tool_name, action.args) for action in handoff_receipt_guard.actions]
        if actual_handoff_receipt_guard != [("specialist_handoff_receipt", {"request": "what should Jarvis do next"})]:
            raise SystemExit(f"Specialist handoff receipt should stay on receipt route: {actual_handoff_receipt_guard}")

        def empty_specialist_proof_args(request: str) -> dict[str, str]:
            return {
                "request": request,
                "tool": "",
                "arguments": "",
                "verification": "",
                "runtime_trace": "",
                "verification_receipt": "",
                "execution_audit": "",
                "execution_recovery": "",
                "after_action_learning": "",
                "completion_claim": "",
                "runtime_trace_sha256": "",
                "verification_receipt_sha256": "",
                "execution_audit_sha256": "",
                "execution_recovery_sha256": "",
                "after_action_learning_sha256": "",
                "completion_claim_sha256": "",
                "blockers": "",
            }

        specialist_downstream_route_cases = {
            "specialist orchestration packet please": (
                "specialist_orchestration_packet",
                {"request": "what should Jarvis do next"},
            ),
            "show latest specialist orchestration packet": (
                "specialist_orchestration_packet",
                {"request": "what should Jarvis do next"},
            ),
            "multi-brain plan for send email to Sam saying hi": (
                "specialist_orchestration_packet",
                {"request": "send email to Sam saying hi"},
            ),
            "specialist action proposal contract please": (
                "specialist_action_proposal_contract",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist action proposal contract": (
                "specialist_action_proposal_contract",
                {"request": "what should Jarvis do next"},
            ),
            "specialist tool dry run packet please": (
                "specialist_tool_dry_run_packet",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist tool dry run packet": (
                "specialist_tool_dry_run_packet",
                {"request": "what should Jarvis do next"},
            ),
            "specialist tool dry run packet: summarize this work please": (
                "specialist_tool_dry_run_packet",
                {"request": "summarize this work please"},
            ),
            "dry run the specialist tool proposal for send email to Sam saying hi": (
                "specialist_tool_dry_run_packet",
                {"request": "send email to Sam saying hi"},
            ),
            "specialist proposal completion gate please": (
                "specialist_proposal_completion_gate",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist proposal completion gate": (
                "specialist_proposal_completion_gate",
                {"request": "what should Jarvis do next"},
            ),
            "specialist execution handoff packet please": (
                "specialist_execution_handoff_packet",
                {"request": "what should Jarvis do next"},
            ),
            "show specialist execution handoff packet": (
                "specialist_execution_handoff_packet",
                {"request": "what should Jarvis do next"},
            ),
            "specialist post run closure packet please": (
                "specialist_post_run_closure_packet",
                empty_specialist_proof_args("what should Jarvis do next"),
            ),
            "close the specialist run for send email to Sam saying hi": (
                "specialist_post_run_closure_packet",
                empty_specialist_proof_args("send email to Sam saying hi"),
            ),
            "show specialist post run closure packet": (
                "specialist_post_run_closure_packet",
                empty_specialist_proof_args("what should Jarvis do next"),
            ),
            "specialist post run closure packet: summarize this work please; tool calculate; verification focused smoke": (
                "specialist_post_run_closure_packet",
                {
                    **empty_specialist_proof_args("summarize this work please"),
                    "tool": "calculate",
                    "verification": "focused smoke",
                },
            ),
            "specialist cycle ledger please": (
                "specialist_cycle_ledger",
                empty_specialist_proof_args("what should Jarvis do next"),
            ),
            "specialist cycle for send email to Sam saying hi": (
                "specialist_cycle_ledger",
                empty_specialist_proof_args("send email to Sam saying hi"),
            ),
            "show specialist cycle ledger": (
                "specialist_cycle_ledger",
                empty_specialist_proof_args("what should Jarvis do next"),
            ),
            "specialist cycle ledger: summarize this work please; tool calculate; verification focused smoke": (
                "specialist_cycle_ledger",
                {
                    **empty_specialist_proof_args("summarize this work please"),
                    "tool": "calculate",
                    "verification": "focused smoke",
                },
            ),
        }
        for command, (expected_tool, expected_args) in specialist_downstream_route_cases.items():
            plan = planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Specialist downstream proof route mismatch for {command!r}: {actual}")

        model_draft_guard = planner.plan("show specialist model draft")
        if model_draft_guard.actions:
            raise SystemExit(f"Specialist downstream display route should not trigger model draft: {model_draft_guard.actions}")

        specialist_model_draft_route_cases = {
            "specialist model draft": (
                "specialist_model_draft",
                {"request": "what should Jarvis do next"},
            ),
            "specialist model draft please": (
                "specialist_model_draft",
                {"request": "what should Jarvis do next"},
            ),
            "specialist draft please": (
                "specialist_model_draft",
                {"request": "what should Jarvis do next"},
            ),
            "model draft please": (
                "specialist_model_draft",
                {"request": "what should Jarvis do next"},
            ),
            "brain model draft please": (
                "specialist_model_draft",
                {"request": "what should Jarvis do next"},
            ),
            "multi-brain draft please": (
                "specialist_model_draft",
                {"request": "what should Jarvis do next"},
            ),
            "specialist model draft: summarize this work please": (
                "specialist_model_draft",
                {"request": "summarize this work please"},
            ),
        }
        for command, (expected_tool, expected_args) in specialist_model_draft_route_cases.items():
            plan = planner.plan(command)
            actual = [(action.tool_name, action.args) for action in plan.actions]
            if actual != [(expected_tool, expected_args)]:
                raise SystemExit(f"Specialist model draft route mismatch for {command!r}: {actual}")

        for command in ["show specialist model draft", "show latest specialist model draft", "preview specialist model draft"]:
            plan = planner.plan(command)
            if plan.actions:
                raise SystemExit(f"Specialist model draft display form should remain unclaimed for {command!r}: {plan.actions}")

        prompt_case = "model planner prompt preview: find files README in ."
        result = runtime.handle(prompt_case)
        print(f"[{'ok' if result.verified else 'blocked'}] {prompt_case}")
        print(result.response[:2200])
        print()
        if not result.verified:
            raise SystemExit("Expected model planner prompt preview to run.")
        for expected in [
            "Jarvis model planner prompt preview",
            "model timeout",
            "System prompt preview",
            "Available tool summary",
            "Tool examples",
            "Execution boundary",
            "does not call the configured model",
            "ToolRegistry",
            "PermissionPolicy",
            "approval-gated",
        ]:
            if expected not in result.response:
                raise SystemExit(f"Model planner prompt preview missing expected text: {expected}")
        metadata = result.tool_results[0].metadata
        if metadata.get("model_timeout_seconds") != config.model_timeout_seconds:
            raise SystemExit("Model planner prompt preview metadata missed model timeout.")
        assert_safe_metadata(metadata, "Model planner prompt preview")

        bounded = runtime.registry.get("model_planner_prompt_preview").handler(
            {"request": "find " + ("README " * 200)}
        )
        print("[ok] direct bounded model planner prompt preview")
        print(bounded.output[:900])
        print()
        if not bounded.ok:
            raise SystemExit("Bounded model planner prompt preview should remain valid.")
        assert_safe_metadata(bounded.metadata, "Bounded model planner prompt preview")
        if len(bounded.metadata.get("request", "")) > 500:
            raise SystemExit("Model planner prompt preview did not bound request metadata.")

        router_cases = [
            (
                "specialist router contract: summarize this work history into a brief",
                "summarizer",
                ["Jarvis specialist router contract", "primary specialist: summarizer", "Execution boundary"],
            ),
            (
                "specialist router contract: inspect the screen and verify the button is visible",
                "vision",
                ["Jarvis specialist router contract", "primary specialist: vision", "screen_verification_contract", "approval-gated"],
            ),
            (
                "specialist router contract: fix this code bug and run focused tests",
                "code",
                ["Jarvis specialist router contract", "primary specialist: code", "ToolRegistry", "PermissionPolicy"],
            ),
        ]
        for case, expected_primary, expected_text in router_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist router contract missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist router primary mismatch: {metadata}")
            assert_safe_metadata(metadata, "Specialist router contract")
            assert_specialist_proof_lane(metadata, "Specialist router contract")
            assert_specialist_review_only_authority(metadata, "Specialist router contract")

        bounded_router = runtime.registry.get("specialist_router_contract").handler(
            {"request": "summarize " + ("Jarvis harness " * 200)}
        )
        print("[ok] direct bounded specialist router contract")
        print(bounded_router.output[:900])
        print()
        if not bounded_router.ok:
            raise SystemExit("Bounded specialist router contract should remain valid.")
        assert_safe_metadata(bounded_router.metadata, "Bounded specialist router contract")
        assert_specialist_proof_lane(bounded_router.metadata, "Bounded specialist router contract")
        assert_specialist_review_only_authority(bounded_router.metadata, "Bounded specialist router contract")
        if len(bounded_router.metadata.get("request", "")) > 500:
            raise SystemExit("Specialist router contract did not bound request metadata.")

        orchestration_cases = [
            (
                "specialist orchestration packet: summarize this work history into a brief",
                "summarizer",
                "ORCHESTRATION_READY_FOR_BOUNDED_DRAFTS",
                [
                    "Jarvis specialist orchestration packet",
                    "multi-brain harness flow",
                    "specialist drafts locked: yes",
                    "executable actions locked: yes",
                    "can bypass ToolRegistry or PermissionPolicy: no",
                    "Specialist lanes:",
                    "Proof queue:",
                ],
            ),
            (
                "multi-brain orchestration: fix this code bug and run focused tests",
                "code",
                "ORCHESTRATION_HELD_FOR_APPROVAL_BOUNDARY",
                [
                    "state: ORCHESTRATION_HELD_FOR_APPROVAL_BOUNDARY",
                    "approval_boundary_required",
                    "approval readiness latest",
                    "ToolRegistry",
                    "PermissionPolicy",
                ],
            ),
            (
                "brain orchestration: hello",
                "planner",
                "ORCHESTRATION_HELD_FOR_ROUTE_REVIEW",
                [
                    "state: ORCHESTRATION_HELD_FOR_ROUTE_REVIEW",
                    "ambiguous route: yes",
                    "specialist route quality:",
                ],
            ),
        ]
        for case, expected_primary, expected_state, expected_text in orchestration_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist orchestration packet missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist orchestration primary mismatch: {metadata}")
            if metadata.get("orchestration_state") != expected_state:
                raise SystemExit(f"Specialist orchestration state mismatch: {metadata}")
            if metadata.get("specialist_drafts_locked") is not True or metadata.get("executable_actions_locked") is not True:
                raise SystemExit(f"Specialist orchestration should keep drafts/actions locked: {metadata}")
            if metadata.get("can_bypass_tool_registry") is not False or metadata.get("can_bypass_permission_policy") is not False:
                raise SystemExit(f"Specialist orchestration must preserve ToolRegistry and PermissionPolicy: {metadata}")
            if metadata.get("selected_specialist_count") != len(metadata.get("selected_specialists") or []):
                raise SystemExit(f"Specialist orchestration missed selected specialist count: {metadata}")
            if metadata.get("lane_count") != len(metadata.get("lanes") or []):
                raise SystemExit(f"Specialist orchestration missed lane count: {metadata}")
            if metadata.get("proof_queue_count") != len(metadata.get("proof_queue") or []):
                raise SystemExit(f"Specialist orchestration missed proof queue count: {metadata}")
            for expected_command in [
                "specialist handoff quality gate:",
                "specialist proposal gate:",
                "specialist model draft:",
                "specialist execution handoff:",
                "specialist post-run closure:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("proof_queue", [])):
                    raise SystemExit(f"Specialist orchestration missed proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist orchestration packet")
            assert_specialist_proof_lane(metadata, "Specialist orchestration packet")
            assert_specialist_review_only_authority(metadata, "Specialist orchestration packet")

        bounded_orchestration = runtime.registry.get("specialist_orchestration_packet").handler(
            {"request": "summarize " + ("Jarvis harness " * 250)}
        )
        print("[ok] direct bounded specialist orchestration packet")
        print(bounded_orchestration.output[:900])
        print()
        if not bounded_orchestration.ok:
            raise SystemExit("Bounded specialist orchestration packet should remain valid.")
        assert_safe_metadata(bounded_orchestration.metadata, "Bounded specialist orchestration packet")
        assert_specialist_proof_lane(bounded_orchestration.metadata, "Bounded specialist orchestration packet")
        assert_specialist_review_only_authority(
            bounded_orchestration.metadata,
            "Bounded specialist orchestration packet",
        )
        if len(bounded_orchestration.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist orchestration packet did not bound request metadata.")

        quality_cases = [
            (
                "specialist route quality: summarize this work history into a brief",
                "summarizer",
                ["Jarvis specialist route quality packet", "primary specialist: summarizer", "Verifier coverage:", "Measured handoff scorecard:", "measured quality:"],
            ),
            (
                "specialist route quality: inspect the screen and verify the button is visible",
                "vision",
                ["primary specialist: vision", "approval triggers detected: computer control or visual observation", "screen_verification_contract"],
            ),
            (
                "specialist quality: fix this code bug and run focused tests",
                "code",
                ["primary specialist: code", "approval triggers detected: shell/code execution", "verification_packet"],
            ),
        ]
        for case, expected_primary, expected_text in quality_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist route quality missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist route quality primary mismatch: {metadata}")
            if not metadata.get("verifier_tools") or metadata.get("verifier_coverage_percent", 0) <= 0:
                raise SystemExit(f"Specialist route quality missed verifier coverage: {metadata}")
            if metadata.get("verdict") not in {"ROUTE_STRONG_WITH_LOCAL_PROOF", "ROUTE_USABLE_BUT_REVIEW", "ROUTE_UNMEASURED_OR_AMBIGUOUS"}:
                raise SystemExit(f"Specialist route quality missed verdict: {metadata}")
            if metadata.get("primary_signal_score", 0) < 1:
                raise SystemExit(f"Specialist route quality missed route signal score: {metadata}")
            assert_safe_metadata(metadata, "Specialist route quality")
            assert_specialist_review_only_authority(metadata, "Specialist route quality")
            assert_specialist_proof_lane(metadata, "Specialist route quality")
            assert_handoff_scorecard(metadata, "Specialist route quality")

        ambiguous_quality = runtime.handle("specialist route quality: hello")
        print("[ok] specialist route quality ambiguous")
        print(ambiguous_quality.response[:1200])
        print()
        ambiguous_metadata = ambiguous_quality.tool_results[0].metadata
        if ambiguous_metadata.get("verdict") != "ROUTE_UNMEASURED_OR_AMBIGUOUS" or ambiguous_metadata.get("confidence") != "low":
            raise SystemExit(f"Specialist route quality should expose ambiguous low-confidence route: {ambiguous_metadata}")
        assert_safe_metadata(ambiguous_metadata, "Ambiguous specialist route quality")
        assert_specialist_review_only_authority(ambiguous_metadata, "Ambiguous specialist route quality")
        assert_specialist_proof_lane(ambiguous_metadata, "Ambiguous specialist route quality")
        assert_handoff_scorecard(ambiguous_metadata, "Ambiguous specialist route quality")
        if not ambiguous_metadata.get("handoff_scorecard", {}).get("ambiguous_penalty_applied"):
            raise SystemExit(f"Ambiguous specialist route quality should apply scorecard penalty: {ambiguous_metadata}")

        bounded_quality = runtime.registry.get("specialist_route_quality").handler(
            {"request": "summarize " + ("Jarvis harness " * 200)}
        )
        print("[ok] direct bounded specialist route quality")
        print(bounded_quality.output[:900])
        print()
        if not bounded_quality.ok:
            raise SystemExit("Bounded specialist route quality should remain valid.")
        assert_safe_metadata(bounded_quality.metadata, "Bounded specialist route quality")
        assert_specialist_review_only_authority(bounded_quality.metadata, "Bounded specialist route quality")
        assert_specialist_proof_lane(bounded_quality.metadata, "Bounded specialist route quality")
        assert_handoff_scorecard(bounded_quality.metadata, "Bounded specialist route quality")
        if len(bounded_quality.metadata.get("request", "")) > 500:
            raise SystemExit("Specialist route quality did not bound request metadata.")

        readiness_cases = [
            (
                "specialist execution readiness: fix this code bug and run focused tests",
                "code",
                "HOLD_FOR_APPROVAL_GATE",
                ["Jarvis specialist execution readiness", "primary specialist: code", "approval triggers detected: shell/code execution", "next command:"],
            ),
            (
                "brain readiness: hello",
                "planner",
                "HOLD_FOR_ROUTE_REVIEW",
                ["Jarvis specialist execution readiness", "ambiguous route: yes", "specialist router contract"],
            ),
        ]
        for case, expected_primary, expected_verdict, expected_text in readiness_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist execution readiness missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist execution readiness primary mismatch: {metadata}")
            if metadata.get("verdict") != expected_verdict:
                raise SystemExit(f"Specialist execution readiness verdict mismatch: {metadata}")
            if metadata.get("ready_for_specialist_model_draft"):
                raise SystemExit(f"Specialist execution readiness should not be draft-ready in smoke cases: {metadata}")
            if not metadata.get("next_command"):
                raise SystemExit(f"Specialist execution readiness missed next command: {metadata}")
            assert_safe_metadata(metadata, "Specialist execution readiness")
            assert_specialist_review_only_authority(metadata, "Specialist execution readiness")
            assert_specialist_proof_lane(metadata, "Specialist execution readiness")
            assert_handoff_scorecard(metadata, "Specialist execution readiness")

        bounded_readiness = runtime.registry.get("specialist_execution_readiness").handler(
            {"request": "summarize " + ("Jarvis harness " * 200)}
        )
        print("[ok] direct bounded specialist execution readiness")
        print(bounded_readiness.output[:900])
        print()
        if not bounded_readiness.ok:
            raise SystemExit("Bounded specialist execution readiness should remain valid.")
        assert_safe_metadata(bounded_readiness.metadata, "Bounded specialist execution readiness")
        assert_specialist_review_only_authority(bounded_readiness.metadata, "Bounded specialist execution readiness")
        assert_specialist_proof_lane(bounded_readiness.metadata, "Bounded specialist execution readiness")
        assert_handoff_scorecard(bounded_readiness.metadata, "Bounded specialist execution readiness")
        if len(bounded_readiness.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist execution readiness did not bound request metadata.")

        handoff_cases = [
            (
                "specialist handoff receipt: fix this code bug and run focused tests",
                "code",
                [
                    "Jarvis specialist handoff receipt",
                    "steering-to-engine packet",
                    "Handoff receipt id:",
                    "primary specialist: code",
                    "Input contract:",
                    "Output contract:",
                    "Verification hooks:",
                    "verifier coverage:",
                    "Measured handoff quality:",
                    "quality state:",
                    "scorecard score:",
                    "proof target:",
                    "stop condition:",
                    "Safety gates:",
                ],
            ),
            (
                "specialist handoff receipt: inspect the screen and verify the button is visible",
                "vision",
                ["primary specialist: vision", "approved screenshot or observation evidence only", "computer control or visual observation", "screen_verification_contract"],
            ),
            (
                "specialist handoff receipt: summarize this work history into a brief",
                "summarizer",
                ["primary specialist: summarizer", "faithful brief", "evidence, inference, omissions"],
            ),
            (
                "specialist handoff receipt: learn from this repeated failure and improve the workflow",
                "reflection",
                ["primary specialist: reflection", "reviewable learning packet", "failure_promotion_packet"],
            ),
        ]
        for case, expected_primary, expected_text in handoff_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist handoff receipt missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist handoff primary mismatch: {metadata}")
            if metadata.get("input_contract_items", 0) < 3 or not metadata.get("output_contract"):
                raise SystemExit(f"Specialist handoff missed contract metadata: {metadata}")
            if not metadata.get("verifier_tools"):
                raise SystemExit(f"Specialist handoff missed verifier tools: {metadata}")
            if not str(metadata.get("handoff_receipt_id", "")).startswith("handoff-"):
                raise SystemExit(f"Specialist handoff missed stable receipt id: {metadata}")
            if metadata.get("handoff_quality_state") not in {
                "MEASURED_HANDOFF_READY",
                "HANDOFF_REVIEW_REQUIRED",
                "HANDOFF_UNMEASURED_OR_AMBIGUOUS",
            }:
                raise SystemExit(f"Specialist handoff missed quality state: {metadata}")
            if metadata.get("handoff_confidence") not in {"high", "medium", "low"}:
                raise SystemExit(f"Specialist handoff missed handoff confidence: {metadata}")
            if metadata.get("verifier_coverage_percent", -1) < 0:
                raise SystemExit(f"Specialist handoff missed verifier coverage: {metadata}")
            if not str(metadata.get("proof_target", "")).startswith("specialist route quality:"):
                raise SystemExit(f"Specialist handoff missed proof target: {metadata}")
            if not metadata.get("stop_condition"):
                raise SystemExit(f"Specialist handoff missed stop condition: {metadata}")
            assert_safe_metadata(metadata, "Specialist handoff receipt")
            assert_specialist_review_only_authority(metadata, "Specialist handoff receipt")
            assert_specialist_proof_lane(metadata, "Specialist handoff receipt")
            assert_handoff_scorecard(metadata, "Specialist handoff receipt")

        bounded_handoff = runtime.registry.get("specialist_handoff_receipt").handler(
            {"request": "summarize " + ("Jarvis harness " * 250)}
        )
        print("[ok] direct bounded specialist handoff receipt")
        print(bounded_handoff.output[:900])
        print()
        if not bounded_handoff.ok:
            raise SystemExit("Bounded specialist handoff receipt should remain valid.")
        assert_safe_metadata(bounded_handoff.metadata, "Bounded specialist handoff receipt")
        assert_specialist_review_only_authority(bounded_handoff.metadata, "Bounded specialist handoff receipt")
        assert_specialist_proof_lane(bounded_handoff.metadata, "Bounded specialist handoff receipt")
        assert_handoff_scorecard(bounded_handoff.metadata, "Bounded specialist handoff receipt")
        if len(bounded_handoff.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist handoff receipt did not bound request metadata.")

        quality_gate_cases = [
            (
                "specialist handoff quality gate: summarize this work history into a brief",
                "summarizer",
                "HANDOFF_QUALITY_READY_FOR_PROPOSAL_GATE",
                [
                    "Jarvis specialist handoff quality gate",
                    "quality binding ready: yes",
                    "action proposal locked: yes",
                    "can emit tool proposals: no",
                    "handoff receipt id:",
                    "route and handoff specialist match: yes",
                    "specialist proposal gate:",
                ],
            ),
            (
                "specialist handoff quality gate: fix this code bug and run focused tests",
                "code",
                "HANDOFF_QUALITY_READY_WITH_APPROVAL_BOUNDARY",
                [
                    "Jarvis specialist handoff quality gate",
                    "state: HANDOFF_QUALITY_READY_WITH_APPROVAL_BOUNDARY",
                    "approval triggers detected: shell/code execution",
                    "ToolRegistry",
                    "PermissionPolicy",
                ],
            ),
            (
                "specialist handoff quality gate: hello",
                "planner",
                "HANDOFF_QUALITY_HELD_FOR_REVIEW",
                [
                    "Jarvis specialist handoff quality gate",
                    "quality binding ready: no",
                    "route_quality_unmeasured_or_ambiguous",
                    "specialist route quality:",
                ],
            ),
        ]
        for case, expected_primary, expected_state, expected_text in quality_gate_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist handoff quality gate missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist handoff quality gate primary mismatch: {metadata}")
            if metadata.get("quality_gate_state") != expected_state:
                raise SystemExit(f"Specialist handoff quality gate state mismatch: {metadata}")
            if metadata.get("can_emit_tool_proposals") is not False or metadata.get("action_proposal_locked") is not True:
                raise SystemExit(f"Specialist handoff quality gate should keep proposals locked: {metadata}")
            if not str(metadata.get("handoff_receipt_id", "")).startswith("handoff-"):
                raise SystemExit(f"Specialist handoff quality gate missed receipt binding: {metadata}")
            if metadata.get("request_binding_ready") is not True or metadata.get("specialist_binding_ready") is not True:
                raise SystemExit(f"Specialist handoff quality gate missed request/specialist binding: {metadata}")
            if metadata.get("required_proof_command_count") != len(metadata.get("required_proof_commands") or []):
                raise SystemExit(f"Specialist handoff quality gate missed required proof count: {metadata}")
            if metadata.get("required_proof_command_count", 0) < 4:
                raise SystemExit(f"Specialist handoff quality gate should expose route/readiness/handoff/proposal proof: {metadata}")
            assert_safe_metadata(metadata, "Specialist handoff quality gate")
            assert_specialist_review_only_authority(metadata, "Specialist handoff quality gate")
            assert_specialist_proof_lane(metadata, "Specialist handoff quality gate")
            assert_handoff_scorecard(metadata, "Specialist handoff quality gate")

        bounded_quality_gate = runtime.registry.get("specialist_handoff_quality_gate").handler(
            {"request": "summarize " + ("Jarvis harness " * 250)}
        )
        print("[ok] direct bounded specialist handoff quality gate")
        print(bounded_quality_gate.output[:900])
        print()
        if not bounded_quality_gate.ok:
            raise SystemExit("Bounded specialist handoff quality gate should remain valid.")
        assert_safe_metadata(bounded_quality_gate.metadata, "Bounded specialist handoff quality gate")
        assert_specialist_review_only_authority(bounded_quality_gate.metadata, "Bounded specialist handoff quality gate")
        assert_specialist_proof_lane(bounded_quality_gate.metadata, "Bounded specialist handoff quality gate")
        assert_handoff_scorecard(bounded_quality_gate.metadata, "Bounded specialist handoff quality gate")
        if len(bounded_quality_gate.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist handoff quality gate did not bound request metadata.")

        proposal_gate_cases = [
            (
                "specialist proposal gate: fix this code bug and run focused tests",
                "code",
                "PROPOSAL_HELD_FOR_APPROVAL_BOUNDARY",
                [
                    "Jarvis specialist proposal gate",
                    "Proposal gate:",
                    "state: PROPOSAL_HELD_FOR_APPROVAL_BOUNDARY",
                    "action proposal locked: yes",
                    "approval triggers detected: shell/code execution",
                    "approval readiness latest",
                    "ToolRegistry",
                    "PermissionPolicy",
                ],
            ),
            (
                "specialist proposal gate: hello",
                "planner",
                "PROPOSAL_HELD_FOR_ROUTE_REVIEW",
                [
                    "Jarvis specialist proposal gate",
                    "state: PROPOSAL_HELD_FOR_ROUTE_REVIEW",
                    "ambiguous route: yes",
                    "specialist execution readiness:",
                ],
            ),
            (
                "specialist proposal gate: summarize this work history into a brief",
                "summarizer",
                "PROPOSAL_FALLBACK_NO_MODEL",
                [
                    "Jarvis specialist proposal gate",
                    "state: PROPOSAL_FALLBACK_NO_MODEL",
                    "model status: needs_attention",
                    "model routing status",
                    "Required proof commands:",
                ],
            ),
        ]
        for case, expected_primary, expected_state, expected_text in proposal_gate_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist proposal gate missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary:
                raise SystemExit(f"Specialist proposal gate primary mismatch: {metadata}")
            if metadata.get("proposal_gate_state") != expected_state:
                raise SystemExit(f"Specialist proposal gate state mismatch: {metadata}")
            if metadata.get("can_emit_tool_proposals") is not False or metadata.get("action_proposal_locked") is not True:
                raise SystemExit(f"Specialist proposal gate should keep tool proposals locked: {metadata}")
            if not metadata.get("next_command"):
                raise SystemExit(f"Specialist proposal gate missed next command: {metadata}")
            if metadata.get("required_proof_command_count") != len(metadata.get("required_proof_commands") or []):
                raise SystemExit(f"Specialist proposal gate missed required proof command count: {metadata}")
            if metadata.get("required_proof_command_count", 0) < 4:
                raise SystemExit(f"Specialist proposal gate should expose route/readiness/handoff/draft proof commands: {metadata}")
            assert_safe_metadata(metadata, "Specialist proposal gate")
            assert_specialist_review_only_authority(metadata, "Specialist proposal gate")
            assert_specialist_proof_lane(metadata, "Specialist proposal gate")
            assert_handoff_scorecard(metadata, "Specialist proposal gate")

        bounded_proposal_gate = runtime.registry.get("specialist_proposal_gate").handler(
            {"request": "summarize " + ("Jarvis harness " * 250)}
        )
        print("[ok] direct bounded specialist proposal gate")
        print(bounded_proposal_gate.output[:900])
        print()
        if not bounded_proposal_gate.ok:
            raise SystemExit("Bounded specialist proposal gate should remain valid.")
        assert_safe_metadata(bounded_proposal_gate.metadata, "Bounded specialist proposal gate")
        assert_specialist_review_only_authority(bounded_proposal_gate.metadata, "Bounded specialist proposal gate")
        assert_specialist_proof_lane(bounded_proposal_gate.metadata, "Bounded specialist proposal gate")
        assert_handoff_scorecard(bounded_proposal_gate.metadata, "Bounded specialist proposal gate")
        if len(bounded_proposal_gate.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist proposal gate did not bound request metadata.")

        action_contract_cases = [
            (
                "specialist action proposal contract: summarize this work history into a brief; tool: brain_search; args: query=Jarvis harness",
                "brain_search",
                "READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW",
                [
                    "Jarvis specialist action proposal contract",
                    "state: READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW",
                    "action execution locked: yes",
                    "can emit executable tool action: no",
                    "registered in ToolRegistry: yes",
                    "risk level: READ_ONLY",
                    "approval required before execution: no",
                    "Measured action proposal scorecard:",
                    "specialist proposal gate:",
                ],
            ),
            (
                "specialist action proposal contract: fix this code bug and run focused tests; tool: run_shell_command; args: command=python3 -m py_compile jarvis_v2/tools/model_status.py",
                "run_shell_command",
                "HELD_FOR_APPROVAL_BOUNDARY",
                [
                    "Jarvis specialist action proposal contract",
                    "state: HELD_FOR_APPROVAL_BOUNDARY",
                    "approval_boundary_required",
                    "approval readiness latest",
                    "ToolRegistry and PermissionPolicy",
                    "Measured action proposal scorecard:",
                ],
            ),
            (
                "specialist action proposal contract: summarize this work history into a brief; tool: missing_tool; args: query=Jarvis harness",
                "missing_tool",
                "HELD_FOR_TOOL_REGISTRY",
                [
                    "Jarvis specialist action proposal contract",
                    "state: HELD_FOR_TOOL_REGISTRY",
                    "tool_not_registered",
                    "registered in ToolRegistry: no",
                    "Measured action proposal scorecard:",
                ],
            ),
            (
                "specialist action proposal contract: summarize this work history into a brief; tool: brain_search",
                "brain_search",
                "HELD_FOR_EXACT_ARGUMENTS",
                [
                    "Jarvis specialist action proposal contract",
                    "state: HELD_FOR_EXACT_ARGUMENTS",
                    "exact_arguments_missing",
                    "argument contract:",
                    "Measured action proposal scorecard:",
                ],
            ),
        ]
        for case, expected_tool, expected_state, expected_text in action_contract_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist action proposal contract missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("proposed_tool") != expected_tool:
                raise SystemExit(f"Specialist action proposal contract proposed tool mismatch: {metadata}")
            if metadata.get("contract_state") != expected_state:
                raise SystemExit(f"Specialist action proposal contract state mismatch: {metadata}")
            if metadata.get("action_execution_locked") is not True or metadata.get("can_emit_executable_tool_action") is not False:
                raise SystemExit(f"Specialist action proposal contract should keep execution locked: {metadata}")
            if metadata.get("required_proof_command_count") != len(metadata.get("required_proof_commands") or []):
                raise SystemExit(f"Specialist action proposal contract missed required proof count: {metadata}")
            for expected_command in [
                "specialist handoff quality gate:",
                "specialist proposal gate:",
                "specialist model draft:",
                "argument contract:",
                "risk preflight:",
                "verification packet:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("required_proof_commands", [])):
                    raise SystemExit(f"Specialist action proposal contract missed proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist action proposal contract")
            assert_specialist_review_only_authority(metadata, "Specialist action proposal contract")
            assert_specialist_proof_lane(metadata, "Specialist action proposal contract")
            assert_action_proposal_scorecard(metadata, "Specialist action proposal contract")
            assert_action_proposal_review_token(metadata, "Specialist action proposal contract")

        bounded_action_contract = runtime.registry.get("specialist_action_proposal_contract").handler(
            {
                "request": "summarize " + ("Jarvis harness " * 250),
                "tool": "brain_search",
                "arguments": "query=Jarvis harness",
            }
        )
        print("[ok] direct bounded specialist action proposal contract")
        print(bounded_action_contract.output[:900])
        print()
        if not bounded_action_contract.ok:
            raise SystemExit("Bounded specialist action proposal contract should remain valid.")
        assert_safe_metadata(bounded_action_contract.metadata, "Bounded specialist action proposal contract")
        assert_specialist_review_only_authority(bounded_action_contract.metadata, "Bounded specialist action proposal contract")
        assert_specialist_proof_lane(bounded_action_contract.metadata, "Bounded specialist action proposal contract")
        assert_action_proposal_scorecard(bounded_action_contract.metadata, "Bounded specialist action proposal contract")
        assert_action_proposal_review_token(bounded_action_contract.metadata, "Bounded specialist action proposal contract")
        if len(bounded_action_contract.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist action proposal contract did not bound request metadata.")

        dry_run_cases = [
            (
                "specialist tool dry run: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
                "brain_search",
                "DRY_RUN_READY_FOR_OPERATOR_REVIEW",
                ["Jarvis specialist tool dry-run packet", "Tool proposal review:", "tool executed: no", "verification expectation supplied: yes", "Measured action proposal scorecard:"],
            ),
            (
                "specialist tool dry run: run command; tool run_shell_command; args command=python3 --version; verification output includes Python",
                "run_shell_command",
                "DRY_RUN_HELD_FOR_APPROVAL_BOUNDARY",
                ["Jarvis specialist tool dry-run packet", "approval required before execution: yes", "approval readiness latest", "Measured action proposal scorecard:"],
            ),
            (
                "specialist tool dry run: summarize memory; tool missing_tool; args query=Jarvis harness; verification cites local memory only",
                "missing_tool",
                "DRY_RUN_HELD_FOR_TOOL_REGISTRY",
                ["Jarvis specialist tool dry-run packet", "registered in ToolRegistry: no", "tool search:", "Measured action proposal scorecard:"],
            ),
        ]
        for case, expected_tool, expected_state, expected_text in dry_run_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2400])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist tool dry-run packet missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("proposed_tool") != expected_tool:
                raise SystemExit(f"Specialist tool dry-run proposed tool mismatch: {metadata}")
            if metadata.get("dry_run_state") != expected_state:
                raise SystemExit(f"Specialist tool dry-run state mismatch: {metadata}")
            if metadata.get("tool_executed") is not False or metadata.get("executable_action_emitted") is not False:
                raise SystemExit(f"Specialist tool dry-run should not execute or emit actions: {metadata}")
            if metadata.get("can_emit_executable_tool_action") is not False:
                raise SystemExit(f"Specialist tool dry-run should not emit executable tool actions: {metadata}")
            if metadata.get("proof_queue_count") != len(metadata.get("proof_queue") or []):
                raise SystemExit(f"Specialist tool dry-run missed proof queue count: {metadata}")
            for expected_command in [
                "specialist action proposal contract:",
                "tool detail:",
                "argument contract:",
                "risk preflight:",
                "verification packet:",
            ]:
                if expected_tool == "missing_tool" and expected_command == "tool detail:":
                    continue
                if not any(str(command).startswith(expected_command) for command in metadata.get("proof_queue", [])):
                    raise SystemExit(f"Specialist tool dry-run missed proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist tool dry-run packet")
            assert_specialist_review_only_authority(metadata, "Specialist tool dry-run packet")
            assert_specialist_proof_lane(metadata, "Specialist tool dry-run packet")
            assert_action_proposal_scorecard(metadata, "Specialist tool dry-run packet")
            assert_action_proposal_review_token(metadata, "Specialist tool dry-run packet")

        completion_gate_cases = [
            (
                "specialist proposal completion gate: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
                "brain_search",
                "PROPOSAL_COMPLETION_READY_FOR_OPERATOR_REVIEW",
                [
                    "Jarvis specialist proposal completion gate",
                    "completion ready for operator review: yes",
                    "action execution locked: yes",
                    "executable action emitted: no",
                    "contract state: READY_FOR_LOCAL_SAFE_DRY_RUN_REVIEW",
                    "dry-run state: DRY_RUN_READY_FOR_OPERATOR_REVIEW",
                    "Measured action proposal scorecard:",
                    "Audit, recovery, and learning handoff:",
                ],
            ),
            (
                "specialist proposal completion gate: run command; tool run_shell_command; args command=python3 --version; verification output includes Python",
                "run_shell_command",
                "PROPOSAL_COMPLETION_HELD_FOR_ACTION_CONTRACT",
                [
                    "Jarvis specialist proposal completion gate",
                    "completion ready for operator review: no",
                    "approval required before execution: yes",
                    "action_contract_not_ready",
                    "dry_run_not_ready_for_operator_review",
                ],
            ),
            (
                "specialist proposal completion gate: summarize memory; tool brain_search; args query=Jarvis harness",
                "brain_search",
                "PROPOSAL_COMPLETION_HELD_FOR_DRY_RUN_PROOF",
                [
                    "Jarvis specialist proposal completion gate",
                    "verification: missing",
                    "verification_expectation_missing",
                    "specialist tool dry run:",
                ],
            ),
        ]
        for case, expected_tool, expected_state, expected_text in completion_gate_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist proposal completion gate missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("proposed_tool") != expected_tool:
                raise SystemExit(f"Specialist proposal completion gate proposed tool mismatch: {metadata}")
            if metadata.get("completion_state") != expected_state:
                raise SystemExit(f"Specialist proposal completion gate state mismatch: {metadata}")
            if metadata.get("action_execution_locked") is not True or metadata.get("executable_action_emitted") is not False:
                raise SystemExit(f"Specialist proposal completion gate should keep execution locked: {metadata}")
            if metadata.get("tool_executed") is not False or metadata.get("can_emit_executable_tool_action") is not False:
                raise SystemExit(f"Specialist proposal completion gate should not execute or emit executable actions: {metadata}")
            if metadata.get("proof_queue_count") != len(metadata.get("proof_queue") or []):
                raise SystemExit(f"Specialist proposal completion gate missed proof queue count: {metadata}")
            if metadata.get("audit_recovery_queue_count") != len(metadata.get("audit_recovery_queue") or []):
                raise SystemExit(f"Specialist proposal completion gate missed audit/recovery queue count: {metadata}")
            for expected_command in [
                "specialist action proposal contract:",
                "specialist tool dry run:",
                "execution audit gate:",
                "verification packet:",
                "execution recovery packet:",
                "after action learning packet:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("proof_queue", [])):
                    raise SystemExit(f"Specialist proposal completion gate missed proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist proposal completion gate")
            assert_specialist_review_only_authority(metadata, "Specialist proposal completion gate")
            assert_specialist_proof_lane(metadata, "Specialist proposal completion gate")
            assert_action_proposal_scorecard(metadata, "Specialist proposal completion gate")
            assert_action_proposal_review_token(metadata, "Specialist proposal completion gate")

        bounded_completion_gate = runtime.registry.get("specialist_proposal_completion_gate").handler(
            {
                "request": "summarize " + ("Jarvis harness " * 250),
                "tool": "brain_search",
                "arguments": "query=Jarvis harness",
                "verification": "cites local memory only",
            }
        )
        print("[ok] direct bounded specialist proposal completion gate")
        print(bounded_completion_gate.output[:900])
        print()
        if not bounded_completion_gate.ok:
            raise SystemExit("Bounded specialist proposal completion gate should remain valid.")
        assert_safe_metadata(bounded_completion_gate.metadata, "Bounded specialist proposal completion gate")
        assert_specialist_review_only_authority(bounded_completion_gate.metadata, "Bounded specialist proposal completion gate")
        assert_specialist_proof_lane(bounded_completion_gate.metadata, "Bounded specialist proposal completion gate")
        assert_action_proposal_scorecard(bounded_completion_gate.metadata, "Bounded specialist proposal completion gate")
        assert_action_proposal_review_token(bounded_completion_gate.metadata, "Bounded specialist proposal completion gate")
        if len(bounded_completion_gate.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist proposal completion gate did not bound request metadata.")

        execution_handoff_cases = [
            (
                "specialist execution handoff: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
                "brain_search",
                "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW",
                [
                    "Jarvis specialist execution handoff packet",
                    "ready for runtime review: yes",
                    "tool executed: no",
                    "executable action emitted: no",
                    "approval queued: no",
                    "normal runtime review path",
                    "Runtime review contract:",
                    "authorizes model call: no",
                    "authorizes tool execution: no",
                    "authorizes approval: no",
                    "authorizes personal-data read: no",
                    "authorizes external side effect: no",
                    "authorizes completion claim: no",
                    "bypasses post-run proof: no",
                    "reusable for next specialist review: no",
                    "route-to-runtime contract token sha256:",
                    "Pre-run proof queue:",
                    "Post-run proof queue:",
                ],
            ),
            (
                "specialist execution handoff: run command; tool run_shell_command; args command=python3 --version; verification output includes Python",
                "run_shell_command",
                "SPECIALIST_EXECUTION_HANDOFF_HELD_FOR_COMPLETION_GATE",
                [
                    "Jarvis specialist execution handoff packet",
                    "ready for runtime review: no",
                    "approval_boundary_required",
                    "specialist proposal completion gate:",
                ],
            ),
        ]
        for case, expected_tool, expected_state, expected_text in execution_handoff_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist execution handoff packet missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("proposed_tool") != expected_tool:
                raise SystemExit(f"Specialist execution handoff proposed tool mismatch: {metadata}")
            if metadata.get("handoff_state") != expected_state:
                raise SystemExit(f"Specialist execution handoff state mismatch: {metadata}")
            if metadata.get("tool_executed") is not False or metadata.get("executable_action_emitted") is not False:
                raise SystemExit(f"Specialist execution handoff should not execute or emit actions: {metadata}")
            if metadata.get("queues_approval") is not False:
                raise SystemExit(f"Specialist execution handoff should not queue approvals: {metadata}")
            runtime_review_rows = metadata.get("runtime_review_contract_rows") or []
            expected_runtime_review_items = {
                "proposal_completion_gate",
                "registered_tool_registry_match",
                "exact_argument_contract",
                "permission_policy_risk_boundary",
                "verification_packet_required",
                "post_run_proof_required",
            }
            if (
                len(runtime_review_rows) != 6
                or metadata.get("runtime_review_contract_row_count") != len(runtime_review_rows)
                or metadata.get("runtime_review_contract_ready") is not True
                or expected_runtime_review_items - {row.get("item") for row in runtime_review_rows}
                or metadata.get("runtime_review_authorizes_model_call")
                or metadata.get("runtime_review_authorizes_tool_execution")
                or metadata.get("runtime_review_authorizes_approval")
                or metadata.get("runtime_review_authorizes_personal_data_read")
                or metadata.get("runtime_review_authorizes_external_side_effect")
                or metadata.get("runtime_review_authorizes_completion_claim")
                or metadata.get("runtime_review_bypasses_post_run_proof")
                or metadata.get("runtime_review_reusable_for_next_specialist_review")
                or any(
                    row.get("authorizes_model_call")
                    or row.get("authorizes_tool_execution")
                    or row.get("authorizes_approval")
                    or row.get("authorizes_personal_data_read")
                    or row.get("authorizes_external_side_effect")
                    or row.get("authorizes_completion_claim")
                    or row.get("bypasses_post_run_proof")
                    or row.get("reusable_for_next_specialist_review")
                    for row in runtime_review_rows
                )
            ):
                raise SystemExit(f"Specialist execution handoff runtime-review contract unsafe: {metadata}")
            if not _specialist_runtime_review_contract_ready(
                runtime_review_rows,
                handoff_state=str(metadata.get("handoff_state") or ""),
                specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                proposed_tool=str(metadata.get("proposed_tool") or ""),
                risk_level=str(metadata.get("risk_level") or ""),
            ):
                raise SystemExit(f"Specialist execution handoff failed production runtime-review contract validator: {metadata}")
            tampered_runtime_review_rows = [dict(row) for row in runtime_review_rows]
            tampered_runtime_review_rows[0]["status"] = "ready_for_runtime_review" if expected_state != "SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW" else "held"
            if _specialist_runtime_review_contract_ready(
                tampered_runtime_review_rows,
                handoff_state=str(metadata.get("handoff_state") or ""),
                specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                proposed_tool=str(metadata.get("proposed_tool") or ""),
                risk_level=str(metadata.get("risk_level") or ""),
            ):
                raise SystemExit(f"Specialist execution handoff runtime-review validator accepted status tampering: {metadata}")
            tampered_runtime_review_rows = [dict(row) for row in runtime_review_rows]
            tampered_runtime_review_rows[1]["source"] = "unreviewed source"
            if _specialist_runtime_review_contract_ready(
                tampered_runtime_review_rows,
                handoff_state=str(metadata.get("handoff_state") or ""),
                specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                proposed_tool=str(metadata.get("proposed_tool") or ""),
                risk_level=str(metadata.get("risk_level") or ""),
            ):
                raise SystemExit(f"Specialist execution handoff runtime-review validator accepted source tampering: {metadata}")
            tampered_runtime_review_rows = [dict(row) for row in runtime_review_rows]
            tampered_runtime_review_rows[2]["authorizes_tool_execution"] = True
            if _specialist_runtime_review_contract_ready(
                tampered_runtime_review_rows,
                handoff_state=str(metadata.get("handoff_state") or ""),
                specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                proposed_tool=str(metadata.get("proposed_tool") or ""),
                risk_level=str(metadata.get("risk_level") or ""),
            ):
                raise SystemExit(f"Specialist execution handoff runtime-review validator accepted authority tampering: {metadata}")
            for expected_summary in [
                "normal-runtime-review-only",
                "model-call-not-authorized",
                "tool-execution-not-authorized",
                "approval-not-authorized",
                "post-run-proof-still-required",
                "fresh-specialist-review-token-required",
            ]:
                if expected_summary not in metadata.get("runtime_review_contract_summary", []):
                    raise SystemExit(f"Specialist execution handoff missed runtime-review summary {expected_summary}: {metadata}")
            if metadata.get("pre_run_proof_queue_count") != len(metadata.get("pre_run_proof_queue") or []):
                raise SystemExit(f"Specialist execution handoff missed pre-run proof count: {metadata}")
            if metadata.get("post_run_proof_queue_count") != len(metadata.get("post_run_proof_queue") or []):
                raise SystemExit(f"Specialist execution handoff missed post-run proof count: {metadata}")
            assert_route_to_runtime_contract_token(metadata, "Specialist execution handoff")
            assert_route_to_runtime_contract_token_boundary(metadata, "Specialist execution handoff")
            if expected_state.endswith("_READY_FOR_RUNTIME_REVIEW"):
                if len(str(metadata.get("specialist_review_token_sha256") or "")) != 64:
                    raise SystemExit(f"Specialist execution handoff missed review token: {metadata}")
                if metadata.get("specialist_review_token_reusable_for_next_review") is not False:
                    raise SystemExit(f"Specialist execution handoff should mark review token non-reusable: {metadata}")
                if metadata.get("next_specialist_review_requires_new_token") is not True:
                    raise SystemExit(f"Specialist execution handoff missed next-review token boundary: {metadata}")
                assert_runtime_review_boundary_token(metadata, "Specialist execution handoff")
            for expected_command in [
                "specialist proposal completion gate:",
                "execution contract:",
                "dispatch decision packet:",
                "verification packet:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("pre_run_proof_queue", [])):
                    raise SystemExit(f"Specialist execution handoff missed pre-run proof command {expected_command}: {metadata}")
            for expected_command in [
                "runtime trace receipt:",
                "verification receipt",
                "execution audit gate:",
                "execution recovery packet:",
                "after action learning packet:",
                "completion claim gate:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("post_run_proof_queue", [])):
                    raise SystemExit(f"Specialist execution handoff missed post-run proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist execution handoff packet")
            assert_specialist_review_only_authority(metadata, "Specialist execution handoff packet")
            assert_specialist_proof_lane(metadata, "Specialist execution handoff packet")

        bounded_execution_handoff = runtime.registry.get("specialist_execution_handoff_packet").handler(
            {
                "request": "summarize " + ("Jarvis harness " * 250),
                "tool": "brain_search",
                "arguments": "query=Jarvis harness",
                "verification": "cites local memory only",
            }
        )
        print("[ok] direct bounded specialist execution handoff")
        print(bounded_execution_handoff.output[:900])
        print()
        if not bounded_execution_handoff.ok:
            raise SystemExit("Bounded specialist execution handoff should remain valid.")
        assert_safe_metadata(bounded_execution_handoff.metadata, "Bounded specialist execution handoff")
        assert_specialist_review_only_authority(bounded_execution_handoff.metadata, "Bounded specialist execution handoff")
        assert_specialist_proof_lane(bounded_execution_handoff.metadata, "Bounded specialist execution handoff")
        if len(str(bounded_execution_handoff.metadata.get("specialist_review_token_sha256") or "")) != 64:
            raise SystemExit(f"Bounded specialist execution handoff missed review token: {bounded_execution_handoff.metadata}")
        if len(bounded_execution_handoff.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist execution handoff did not bound request metadata.")
        assert_runtime_review_boundary_token(bounded_execution_handoff.metadata, "Bounded specialist execution handoff")
        assert_route_to_runtime_contract_token(bounded_execution_handoff.metadata, "Bounded specialist execution handoff")
        assert_route_to_runtime_contract_token_boundary(bounded_execution_handoff.metadata, "Bounded specialist execution handoff")

        post_run_closure_cases = [
            (
                f"specialist post-run closure: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only; runtime_trace reviewed runtime trace receipt; runtime_trace_sha256 {runtime_trace_sha256}; verification_receipt reviewed verification receipt; verification_receipt_sha256 {verification_receipt_sha256}; audit reviewed execution audit; audit_sha256 {execution_audit_sha256}; recovery reviewed recovery packet; recovery_sha256 {execution_recovery_sha256}; learning reviewed learning packet; learning_sha256 {after_action_learning_sha256}; completion_claim reviewed completion claim; completion_claim_sha256 {completion_claim_sha256}",
                "SPECIALIST_POST_RUN_CLOSURE_READY",
                [
                    "Jarvis specialist post-run closure packet",
                    "ready to count specialist execution closed: yes",
                    "handoff state: SPECIALIST_EXECUTION_HANDOFF_READY_FOR_RUNTIME_REVIEW",
                    "runtime review boundary token sha256:",
                    "runtime trace receipt: reviewed runtime trace receipt",
                    "post-run artifact hashes present: yes",
                    "completion claim: reviewed completion claim",
                    "Required closure commands:",
                ],
            ),
            (
                "specialist post-run closure: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
                "SPECIALIST_POST_RUN_CLOSURE_HELD",
                [
                    "Jarvis specialist post-run closure packet",
                    "ready to count specialist execution closed: no",
                    "runtime trace receipt",
                    "verification receipt",
                    "completion claim gate",
                ],
            ),
        ]
        for case, expected_state, expected_text in post_run_closure_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:3000])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist post-run closure missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("closure_state") != expected_state:
                raise SystemExit(f"Specialist post-run closure state mismatch: {metadata}")
            if metadata.get("action_allowed_now") is not False or metadata.get("executable_action_emitted") is not False:
                raise SystemExit(f"Specialist post-run closure should not allow or emit actions: {metadata}")
            if metadata.get("can_auto_execute_now") is not False or metadata.get("can_emit_executable_tool_action") is not False:
                raise SystemExit(f"Specialist post-run closure should not auto-execute: {metadata}")
            if metadata.get("required_command_count") != len(metadata.get("required_commands") or []):
                raise SystemExit(f"Specialist post-run closure missed required command count: {metadata}")
            if metadata.get("proof_queue_count") != len(metadata.get("proof_queue") or []):
                raise SystemExit(f"Specialist post-run closure missed proof queue count: {metadata}")
            if expected_state.endswith("_READY") and metadata.get("ready_to_count_specialist_execution_closed") is not True:
                raise SystemExit(f"Specialist post-run closure should be ready with full receipts: {metadata}")
            if expected_state.endswith("_READY"):
                expected_hashes = {
                    "runtime_trace_sha256": runtime_trace_sha256,
                    "verification_receipt_sha256": verification_receipt_sha256,
                    "execution_audit_sha256": execution_audit_sha256,
                    "execution_recovery_sha256": execution_recovery_sha256,
                    "after_action_learning_sha256": after_action_learning_sha256,
                    "completion_claim_sha256": completion_claim_sha256,
                }
                for key, expected_hash in expected_hashes.items():
                    if metadata.get(key) != expected_hash:
                        raise SystemExit(f"Specialist post-run closure missed hash {key}: {metadata}")
                if metadata.get("post_run_artifact_hashes_present") is not True:
                    raise SystemExit(f"Specialist post-run closure missed artifact hash readiness: {metadata}")
                if len(str(metadata.get("specialist_review_token_sha256") or "")) != 64:
                    raise SystemExit(f"Specialist post-run closure missed review token: {metadata}")
                if metadata.get("specialist_review_token_reusable_for_next_review") is not False:
                    raise SystemExit(f"Specialist post-run closure should mark review token non-reusable: {metadata}")
                if metadata.get("next_specialist_review_requires_new_token") is not True:
                    raise SystemExit(f"Specialist post-run closure missed next-review token boundary: {metadata}")
                assert_runtime_review_boundary_token(metadata, "Specialist post-run closure")
                assert_route_to_runtime_contract_token(metadata, "Specialist post-run closure")
                assert_specialist_post_run_closure_token(metadata, "Specialist post-run closure")
            if expected_state.endswith("_HELD") and metadata.get("ready_to_count_specialist_execution_closed") is not False:
                raise SystemExit(f"Specialist post-run closure should be held without full receipts: {metadata}")
            for expected_command in [
                "specialist execution handoff:",
                "runtime trace receipt:",
                "verification receipt",
                "execution audit gate:",
                "execution recovery packet:",
                "after action learning packet:",
                "completion claim gate:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("proof_queue", [])):
                    raise SystemExit(f"Specialist post-run closure missed proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist post-run closure")
            assert_specialist_review_only_authority(metadata, "Specialist post-run closure")
            assert_specialist_proof_lane(metadata, "Specialist post-run closure")

        bounded_post_run_closure = runtime.registry.get("specialist_post_run_closure_packet").handler(
            {
                "request": "summarize " + ("Jarvis harness " * 250),
                "tool": "brain_search",
                "arguments": "query=Jarvis harness",
                "verification": "cites local memory only",
                "runtime_trace": "runtime trace reviewed",
                "verification_receipt": "verification receipt reviewed",
                "execution_audit": "audit reviewed",
                "execution_recovery": "recovery reviewed",
                "after_action_learning": "learning reviewed",
                "completion_claim": "completion claim reviewed",
                "runtime_trace_sha256": runtime_trace_sha256,
                "verification_receipt_sha256": verification_receipt_sha256,
                "execution_audit_sha256": execution_audit_sha256,
                "execution_recovery_sha256": execution_recovery_sha256,
                "after_action_learning_sha256": after_action_learning_sha256,
                "completion_claim_sha256": completion_claim_sha256,
            }
        )
        print("[ok] direct bounded specialist post-run closure")
        print(bounded_post_run_closure.output[:900])
        print()
        if not bounded_post_run_closure.ok:
            raise SystemExit("Bounded specialist post-run closure should remain valid.")
        assert_safe_metadata(bounded_post_run_closure.metadata, "Bounded specialist post-run closure")
        assert_specialist_review_only_authority(bounded_post_run_closure.metadata, "Bounded specialist post-run closure")
        assert_specialist_proof_lane(bounded_post_run_closure.metadata, "Bounded specialist post-run closure")
        if len(bounded_post_run_closure.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist post-run closure did not bound request metadata.")
        if bounded_post_run_closure.metadata.get("closure_state") != "SPECIALIST_POST_RUN_CLOSURE_READY":
            raise SystemExit(f"Bounded specialist post-run closure should be ready: {bounded_post_run_closure.metadata}")
        if bounded_post_run_closure.metadata.get("post_run_artifact_hashes_present") is not True:
            raise SystemExit(f"Bounded specialist post-run closure missed artifact hashes: {bounded_post_run_closure.metadata}")
        if bounded_post_run_closure.metadata.get("specialist_review_token_sha256") != bounded_execution_handoff.metadata.get("specialist_review_token_sha256"):
            raise SystemExit(f"Bounded specialist post-run closure review token diverged: {bounded_post_run_closure.metadata}")
        if bounded_post_run_closure.metadata.get("runtime_review_boundary_token_sha256") != bounded_execution_handoff.metadata.get("runtime_review_boundary_token_sha256"):
            raise SystemExit(f"Bounded specialist post-run closure runtime boundary token diverged: {bounded_post_run_closure.metadata}")
        if bounded_post_run_closure.metadata.get("route_to_runtime_contract_token_sha256") != bounded_execution_handoff.metadata.get("route_to_runtime_contract_token_sha256"):
            raise SystemExit(f"Bounded specialist post-run closure route-to-runtime token diverged: {bounded_post_run_closure.metadata}")
        assert_runtime_review_boundary_token(bounded_post_run_closure.metadata, "Bounded specialist post-run closure")
        assert_route_to_runtime_contract_token(bounded_post_run_closure.metadata, "Bounded specialist post-run closure")
        assert_specialist_post_run_closure_token(bounded_post_run_closure.metadata, "Bounded specialist post-run closure")

        cycle_ledger_cases = [
            (
                f"specialist cycle ledger: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only; runtime_trace reviewed runtime trace receipt; runtime_trace_sha256 {runtime_trace_sha256}; verification_receipt reviewed verification receipt; verification_receipt_sha256 {verification_receipt_sha256}; audit reviewed execution audit; audit_sha256 {execution_audit_sha256}; recovery reviewed recovery packet; recovery_sha256 {execution_recovery_sha256}; learning reviewed learning packet; learning_sha256 {after_action_learning_sha256}; completion_claim reviewed completion claim; completion_claim_sha256 {completion_claim_sha256}",
                "SPECIALIST_CYCLE_LEDGER_READY_FOR_FRESH_REVIEW",
                [
                    "Jarvis specialist cycle ledger",
                    "ready for fresh specialist review: yes",
                    "previous specialist execution permission reusable for next review: no",
                    "Cycle stages:",
                    "post_run_closure: ready",
                    "post-run artifact hashes present: yes",
                    "Fresh-review boundary:",
                    "Required cycle proof chain:",
                ],
            ),
            (
                "specialist cycle ledger: summarize memory; tool brain_search; args query=Jarvis harness; verification cites local memory only",
                "SPECIALIST_CYCLE_LEDGER_HELD",
                [
                    "Jarvis specialist cycle ledger",
                    "ready for fresh specialist review: no",
                    "post_run_closure: held",
                    "runtime trace",
                    "completion claim",
                ],
            ),
            (
                "specialist cycle ledger: run command; tool run_shell_command; args command=python3 --version; verification output includes Python",
                "SPECIALIST_CYCLE_LEDGER_HELD",
                [
                    "Jarvis specialist cycle ledger",
                    "ready for fresh specialist review: no",
                    "approval",
                    "action_proposal_contract: held",
                ],
            ),
        ]
        for case, expected_state, expected_text in cycle_ledger_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:3600])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist cycle ledger missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("cycle_state") != expected_state:
                raise SystemExit(f"Specialist cycle ledger state mismatch: {metadata}")
            if metadata.get("previous_specialist_execution_permission_reusable_for_next_review") is not False:
                raise SystemExit(f"Specialist cycle ledger must forbid permission reuse: {metadata}")
            fresh_review_queue = metadata.get("fresh_review_preflight_queue") or []
            if metadata.get("fresh_review_preflight_queue_count") != len(fresh_review_queue):
                raise SystemExit(f"Specialist cycle ledger missed fresh-review preflight queue count: {metadata}")
            for expected_command in [
                "specialist router contract:",
                "specialist route quality:",
                "specialist execution readiness:",
                "specialist handoff receipt:",
                "specialist proposal gate:",
                "specialist action proposal contract:",
                "specialist tool dry run:",
                "specialist execution handoff:",
                "specialist post-run closure:",
            ]:
                if not any(str(command).startswith(expected_command) for command in fresh_review_queue):
                    raise SystemExit(f"Specialist cycle ledger missed fresh-review preflight command {expected_command}: {metadata}")
            if not str(metadata.get("fresh_review_next_preflight_command") or "").startswith("specialist router contract:"):
                raise SystemExit(f"Specialist cycle ledger missed first fresh-review preflight command: {metadata}")
            fresh_review_contract = metadata.get("fresh_review_contract_rows") or []
            if metadata.get("fresh_review_contract_count") != len(fresh_review_contract):
                raise SystemExit(f"Specialist cycle ledger missed fresh-review contract count: {metadata}")
            if metadata.get("fresh_review_contract_count") != 12:
                raise SystemExit(f"Specialist cycle ledger should expose twelve fresh-review contract rows: {metadata}")
            expected_contract_items = {
                "router_contract",
                "route_quality",
                "execution_readiness",
                "handoff_receipt",
                "proposal_gate",
                "action_proposal_contract",
                "tool_dry_run",
                "execution_handoff",
                "post_run_closure",
                "post_run_closure_token",
                "specialist_review_token",
                "route_to_runtime_contract_token",
            }
            if {row.get("item") for row in fresh_review_contract} != expected_contract_items:
                raise SystemExit(f"Specialist cycle ledger missed fresh-review contract items: {metadata}")
            if any(
                row.get("fresh_required") is not True
                or row.get("prior_artifact_reusable") is not False
                or row.get("authorizes_action_now") is not False
                or row.get("authorizes_model_call") is not False
                or row.get("authorizes_tool_execution") is not False
                or row.get("authorizes_approval") is not False
                or row.get("authorizes_personal_data_read") is not False
                or row.get("authorizes_external_side_effect") is not False
                or row.get("authorizes_fresh_review") is not False
                for row in fresh_review_contract
            ):
                raise SystemExit(f"Specialist cycle ledger fresh-review contract should keep prior proof non-authorizing: {metadata}")
            if metadata.get("all_prior_artifacts_non_authorizing") is not True:
                raise SystemExit(f"Specialist cycle ledger missed non-authorizing prior-artifact summary: {metadata}")
            if metadata.get("next_specialist_review_requires_full_preflight") is not True:
                raise SystemExit(f"Specialist cycle ledger missed full-preflight requirement: {metadata}")
            if metadata.get("action_allowed_now") is not False or metadata.get("executable_action_emitted") is not False:
                raise SystemExit(f"Specialist cycle ledger should not allow or emit actions: {metadata}")
            if metadata.get("can_auto_execute_now") is not False or metadata.get("can_emit_executable_tool_action") is not False:
                raise SystemExit(f"Specialist cycle ledger should not auto-execute: {metadata}")
            if metadata.get("tool_executed") is not False or metadata.get("queues_approval") is not False:
                raise SystemExit(f"Specialist cycle ledger should not execute or queue approvals: {metadata}")
            if metadata.get("required_command_count") != len(metadata.get("required_commands") or []):
                raise SystemExit(f"Specialist cycle ledger missed required command count: {metadata}")
            if metadata.get("stage_count") != len(metadata.get("stage_rows") or []):
                raise SystemExit(f"Specialist cycle ledger missed stage count: {metadata}")
            for row in metadata.get("stage_rows") or []:
                for key in [
                    "authorizes_action_now",
                    "authorizes_model_call",
                    "authorizes_tool_execution",
                    "authorizes_approval",
                    "authorizes_personal_data_read",
                    "authorizes_external_side_effect",
                    "authorizes_fresh_review",
                    "reusable_for_next_cycle",
                ]:
                    if row.get(key) is not False:
                        raise SystemExit(f"Specialist cycle ledger stage row should report {key}=False: {row}")
            if metadata.get("cycle_state").endswith("_READY_FOR_FRESH_REVIEW") and not all(row.get("ready") for row in metadata.get("stage_rows", [])):
                raise SystemExit(f"Specialist cycle ledger should have all stages ready: {metadata}")
            if metadata.get("cycle_state").endswith("_READY_FOR_FRESH_REVIEW"):
                if metadata.get("specialist_cycle_ledger_ready") is not True:
                    raise SystemExit(f"Specialist cycle ledger missed shape-aware ready flag: {metadata}")
                if metadata.get("ready_for_fresh_specialist_review") is not metadata.get("specialist_cycle_ledger_ready"):
                    raise SystemExit(f"Specialist cycle ledger fresh-review ready flag diverged from validator: {metadata}")
                if metadata.get("ready_to_count_specialist_cycle_closed") is not metadata.get("specialist_cycle_ledger_ready"):
                    raise SystemExit(f"Specialist cycle ledger closed-cycle flag diverged from validator: {metadata}")
                if metadata.get("post_run_artifact_hashes_present") is not True:
                    raise SystemExit(f"Specialist cycle ledger missed post-run artifact hashes: {metadata}")
                for key, count_key, expected_count in [
                    ("action_contract_scorecard_rows", "action_contract_scorecard_row_count", 6),
                    ("dry_run_scorecard_rows", "dry_run_scorecard_row_count", 6),
                    ("completion_scorecard_rows", "completion_scorecard_row_count", 2),
                ]:
                    rows = metadata.get(key) or []
                    if metadata.get(count_key) != len(rows) or len(rows) != expected_count:
                        raise SystemExit(f"Specialist cycle ledger missed {key}: {metadata}")
                if metadata.get("action_contract_scorecard_required_rows_ready") is not False:
                    raise SystemExit(f"Specialist cycle ledger action contract should show verification row pending: {metadata}")
                if metadata.get("dry_run_scorecard_required_rows_ready") is not True:
                    raise SystemExit(f"Specialist cycle ledger dry-run scorecard should be ready: {metadata}")
                if metadata.get("completion_scorecard_required_rows_ready") is not False:
                    raise SystemExit(f"Specialist cycle ledger completion rows should reflect contract verification gap: {metadata}")
                if metadata.get("proposal_scorecard_contract_shape_ready") is not True:
                    raise SystemExit(f"Specialist cycle ledger missed production scorecard shape-ready contract: {metadata}")
                if not _specialist_action_proposal_scorecard_shape_ready(metadata.get("action_contract_scorecard_rows") or []):
                    raise SystemExit(f"Specialist cycle ledger action contract scorecard failed production validator: {metadata}")
                if not _specialist_action_proposal_scorecard_shape_ready(metadata.get("dry_run_scorecard_rows") or []):
                    raise SystemExit(f"Specialist cycle ledger dry-run scorecard failed production validator: {metadata}")
                if not _specialist_combined_action_proposal_scorecard_shape_ready(metadata.get("completion_scorecard_rows") or []):
                    raise SystemExit(f"Specialist cycle ledger completion scorecard failed production validator: {metadata}")
                if not _specialist_cycle_ledger_ready(
                    stage_rows=metadata.get("stage_rows") or [],
                    fresh_review_contract_rows=metadata.get("fresh_review_contract_rows") or [],
                    action_contract_scorecard_rows=metadata.get("action_contract_scorecard_rows") or [],
                    dry_run_scorecard_rows=metadata.get("dry_run_scorecard_rows") or [],
                    completion_scorecard_rows=metadata.get("completion_scorecard_rows") or [],
                    post_run_artifact_hashes_present=bool(metadata.get("post_run_artifact_hashes_present")),
                    specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                    runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
                    route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
                    specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
                    specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
                    specialist_cycle_ledger_token_sha256=str(metadata.get("specialist_cycle_ledger_token_sha256") or ""),
                ):
                    raise SystemExit(f"Specialist cycle ledger production validator rejected ready metadata: {metadata}")
                tampered_stage_rows = [dict(row) for row in metadata.get("stage_rows") or []]
                tampered_stage_rows[0]["authorizes_tool_execution"] = True
                if _specialist_cycle_ledger_ready(
                    stage_rows=tampered_stage_rows,
                    fresh_review_contract_rows=metadata.get("fresh_review_contract_rows") or [],
                    action_contract_scorecard_rows=metadata.get("action_contract_scorecard_rows") or [],
                    dry_run_scorecard_rows=metadata.get("dry_run_scorecard_rows") or [],
                    completion_scorecard_rows=metadata.get("completion_scorecard_rows") or [],
                    post_run_artifact_hashes_present=bool(metadata.get("post_run_artifact_hashes_present")),
                    specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                    runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
                    route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
                    specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
                    specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
                    specialist_cycle_ledger_token_sha256=str(metadata.get("specialist_cycle_ledger_token_sha256") or ""),
                ):
                    raise SystemExit(f"Specialist cycle ledger validator accepted stage authority tampering: {metadata}")
                for malformed_ready in ("true", "yes", 1):
                    tampered_stage_rows = [dict(row) for row in metadata.get("stage_rows") or []]
                    tampered_stage_rows[0]["ready"] = malformed_ready
                    if _specialist_cycle_ledger_ready(
                        stage_rows=tampered_stage_rows,
                        fresh_review_contract_rows=metadata.get("fresh_review_contract_rows") or [],
                        action_contract_scorecard_rows=metadata.get("action_contract_scorecard_rows") or [],
                        dry_run_scorecard_rows=metadata.get("dry_run_scorecard_rows") or [],
                        completion_scorecard_rows=metadata.get("completion_scorecard_rows") or [],
                        post_run_artifact_hashes_present=bool(metadata.get("post_run_artifact_hashes_present")),
                        specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                        runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
                        route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
                        specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
                        specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
                        specialist_cycle_ledger_token_sha256=str(metadata.get("specialist_cycle_ledger_token_sha256") or ""),
                    ):
                        raise SystemExit(
                            f"Specialist cycle ledger validator accepted malformed stage ready={malformed_ready!r}: {metadata}"
                        )
                tampered_fresh_review_rows = [dict(row) for row in metadata.get("fresh_review_contract_rows") or []]
                tampered_fresh_review_rows[0]["prior_artifact_reusable"] = True
                if _specialist_cycle_ledger_ready(
                    stage_rows=metadata.get("stage_rows") or [],
                    fresh_review_contract_rows=tampered_fresh_review_rows,
                    action_contract_scorecard_rows=metadata.get("action_contract_scorecard_rows") or [],
                    dry_run_scorecard_rows=metadata.get("dry_run_scorecard_rows") or [],
                    completion_scorecard_rows=metadata.get("completion_scorecard_rows") or [],
                    post_run_artifact_hashes_present=bool(metadata.get("post_run_artifact_hashes_present")),
                    specialist_review_token_sha256=str(metadata.get("specialist_review_token_sha256") or ""),
                    runtime_review_boundary_token_sha256=str(metadata.get("runtime_review_boundary_token_sha256") or ""),
                    route_to_runtime_contract_token_sha256=str(metadata.get("route_to_runtime_contract_token_sha256") or ""),
                    specialist_post_run_closure_token_sha256=str(metadata.get("specialist_post_run_closure_token_sha256") or ""),
                    specialist_fresh_review_boundary_token_sha256=str(metadata.get("specialist_fresh_review_boundary_token_sha256") or ""),
                    specialist_cycle_ledger_token_sha256=str(metadata.get("specialist_cycle_ledger_token_sha256") or ""),
                ):
                    raise SystemExit(f"Specialist cycle ledger validator accepted fresh-review reuse tampering: {metadata}")
                tampered_scorecard_rows = [dict(row) for row in metadata.get("dry_run_scorecard_rows") or []]
                tampered_scorecard_rows[0]["required_before_execution"] = False
                if _specialist_action_proposal_scorecard_shape_ready(tampered_scorecard_rows):
                    raise SystemExit(f"Specialist cycle ledger scorecard validator accepted required-flag tampering: {metadata}")
                if metadata.get("proposal_scorecard_contract_ready") is not False:
                    raise SystemExit(f"Specialist cycle ledger should not mark combined scorecard rows ready while contract row is pending: {metadata}")
                expected_hashes = {
                    "runtime_trace_sha256": runtime_trace_sha256,
                    "verification_receipt_sha256": verification_receipt_sha256,
                    "execution_audit_sha256": execution_audit_sha256,
                    "execution_recovery_sha256": execution_recovery_sha256,
                    "after_action_learning_sha256": after_action_learning_sha256,
                    "completion_claim_sha256": completion_claim_sha256,
                }
                for key, expected_hash in expected_hashes.items():
                    if metadata.get(key) != expected_hash:
                        raise SystemExit(f"Specialist cycle ledger missed hash {key}: {metadata}")
                if len(str(metadata.get("specialist_review_token_sha256") or "")) != 64:
                    raise SystemExit(f"Specialist cycle ledger missed review token: {metadata}")
                if metadata.get("previous_specialist_review_token_reusable_for_next_review") is not False:
                    raise SystemExit(f"Specialist cycle ledger should not reuse prior review token: {metadata}")
                if metadata.get("next_specialist_review_requires_new_token") is not True:
                    raise SystemExit(f"Specialist cycle ledger missed next-review token boundary: {metadata}")
                assert_route_to_runtime_contract_token(metadata, "Specialist cycle ledger")
                assert_runtime_review_boundary_token(metadata, "Specialist cycle ledger")
                assert_specialist_fresh_review_boundary_token(metadata, "Specialist cycle ledger")
                assert_specialist_post_run_closure_token(metadata, "Specialist cycle ledger")
                assert_specialist_cycle_ledger_token(metadata, "Specialist cycle ledger")
                if metadata.get("specialist_cycle_ledger_token_ready") is not True:
                    raise SystemExit(f"Specialist cycle ledger missed production token-ready flag: {metadata}")
                if not _specialist_cycle_ledger_token_ready_from_metadata(metadata):
                    raise SystemExit(f"Specialist cycle ledger token-ready validator rejected metadata: {metadata}")
                tampered_cycle_token_metadata = dict(metadata)
                tampered_cycle_token_metadata["specialist_cycle_ledger_token_sha256"] = "0" * 64
                if _specialist_cycle_ledger_token_ready_from_metadata(tampered_cycle_token_metadata):
                    raise SystemExit(f"Specialist cycle ledger token-ready validator accepted forged token: {tampered_cycle_token_metadata}")
                tampered_cycle_token_metadata = dict(metadata)
                tampered_cycle_token_metadata["specialist_cycle_ledger_token_authorizes_tool_execution"] = True
                if _specialist_cycle_ledger_token_ready_from_metadata(tampered_cycle_token_metadata):
                    raise SystemExit(f"Specialist cycle ledger token-ready validator accepted authority tampering: {tampered_cycle_token_metadata}")
                tampered_cycle_token_metadata = dict(metadata)
                tampered_post_run_metadata = dict(metadata.get("post_run_closure_metadata") or {})
                tampered_post_run_queue = list(tampered_post_run_metadata.get("proof_queue") or [])
                if tampered_post_run_queue:
                    tampered_post_run_queue[-1] = "completion claim gate: stale nested proof queue"
                    tampered_post_run_metadata["proof_queue"] = tampered_post_run_queue
                    tampered_cycle_token_metadata["post_run_closure_metadata"] = tampered_post_run_metadata
                    if _specialist_cycle_ledger_token_ready_from_metadata(tampered_cycle_token_metadata):
                        raise SystemExit(f"Specialist cycle ledger token-ready validator accepted stale nested post-run proof queue: {tampered_cycle_token_metadata}")
                tampered_cycle_token_metadata = dict(metadata)
                tampered_cycle_token_metadata["next_specialist_review_requires_new_cycle_ledger_token"] = False
                if _specialist_cycle_ledger_token_ready_from_metadata(tampered_cycle_token_metadata):
                    raise SystemExit(f"Specialist cycle ledger token-ready validator accepted next-review token reuse: {tampered_cycle_token_metadata}")
            else:
                if metadata.get("specialist_cycle_ledger_ready") is not False:
                    raise SystemExit(f"Specialist cycle ledger held state should not be shape-ready: {metadata}")
            if not isinstance(metadata.get("post_run_closure_metadata"), dict):
                raise SystemExit(f"Specialist cycle ledger should bind post-run closure metadata: {metadata}")
            for expected_command in [
                "specialist route quality:",
                "specialist execution readiness:",
                "specialist handoff receipt:",
                "specialist handoff quality gate:",
                "specialist proposal gate:",
                "specialist action proposal contract:",
                "specialist tool dry run:",
                "specialist proposal completion gate:",
                "specialist execution handoff:",
                "specialist post-run closure:",
            ]:
                if not any(str(command).startswith(expected_command) for command in metadata.get("proof_queue", [])):
                    raise SystemExit(f"Specialist cycle ledger missed proof command {expected_command}: {metadata}")
            assert_safe_metadata(metadata, "Specialist cycle ledger")
            assert_specialist_review_only_authority(metadata, "Specialist cycle ledger")
            assert_specialist_proof_lane(metadata, "Specialist cycle ledger")

        bounded_cycle_ledger = runtime.registry.get("specialist_cycle_ledger").handler(
            {
                "request": "summarize " + ("Jarvis harness " * 250),
                "tool": "brain_search",
                "arguments": "query=Jarvis harness",
                "verification": "cites local memory only",
                "runtime_trace": "runtime trace reviewed",
                "verification_receipt": "verification receipt reviewed",
                "execution_audit": "audit reviewed",
                "execution_recovery": "recovery reviewed",
                "after_action_learning": "learning reviewed",
                "completion_claim": "completion claim reviewed",
                "runtime_trace_sha256": runtime_trace_sha256,
                "verification_receipt_sha256": verification_receipt_sha256,
                "execution_audit_sha256": execution_audit_sha256,
                "execution_recovery_sha256": execution_recovery_sha256,
                "after_action_learning_sha256": after_action_learning_sha256,
                "completion_claim_sha256": completion_claim_sha256,
            }
        )
        print("[ok] direct bounded specialist cycle ledger")
        print(bounded_cycle_ledger.output[:900])
        print()
        if not bounded_cycle_ledger.ok:
            raise SystemExit("Bounded specialist cycle ledger should remain valid.")
        assert_safe_metadata(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        assert_specialist_proof_lane(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        if len(bounded_cycle_ledger.metadata.get("request", "")) > 700:
            raise SystemExit("Specialist cycle ledger did not bound request metadata.")
        if bounded_cycle_ledger.metadata.get("post_run_artifact_hashes_present") is not True:
            raise SystemExit(f"Bounded specialist cycle ledger missed artifact hashes: {bounded_cycle_ledger.metadata}")
        if bounded_cycle_ledger.metadata.get("specialist_cycle_ledger_ready") is not bounded_cycle_ledger.metadata.get("ready_for_fresh_specialist_review"):
            raise SystemExit(f"Bounded specialist cycle ledger shape-aware ready flag diverged from ready state: {bounded_cycle_ledger.metadata}")
        assert_specialist_review_only_authority(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        assert_runtime_review_boundary_token(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        assert_route_to_runtime_contract_token(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        assert_specialist_fresh_review_boundary_token(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        assert_specialist_post_run_closure_token(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        assert_specialist_cycle_ledger_token(bounded_cycle_ledger.metadata, "Bounded specialist cycle ledger")
        if bounded_cycle_ledger.metadata.get("specialist_cycle_ledger_token_ready") is not _specialist_cycle_ledger_token_ready_from_metadata(bounded_cycle_ledger.metadata):
            raise SystemExit(f"Bounded specialist cycle ledger token-ready flag diverged: {bounded_cycle_ledger.metadata}")
        bounded_tampered_metadata = dict(bounded_cycle_ledger.metadata)
        bounded_tampered_post_run = dict(bounded_cycle_ledger.metadata.get("post_run_closure_metadata") or {})
        bounded_tampered_queue = list(bounded_tampered_post_run.get("proof_queue") or [])
        if bounded_tampered_queue:
            bounded_tampered_queue[0] = "runtime trace receipt: stale bounded proof queue"
            bounded_tampered_post_run["proof_queue"] = bounded_tampered_queue
            bounded_tampered_metadata["post_run_closure_metadata"] = bounded_tampered_post_run
            if _specialist_cycle_ledger_token_ready_from_metadata(bounded_tampered_metadata):
                raise SystemExit(f"Bounded specialist cycle ledger token-ready validator accepted stale nested post-run proof queue: {bounded_tampered_metadata}")

        path_cycle_ledger = runtime.registry.get("specialist_cycle_ledger").handler(
            {
                "request": "summarize /\x55sers/example/private/harness-notes and /var/folders/zc/harness-notes plus /tmp/harness-notes; tool brain_search; args query=Jarvis harness; verification cites local memory only",
                "tool": "brain_search",
                "arguments": "query=Jarvis harness",
                "verification": "cites local memory only",
                "runtime_trace": "runtime trace reviewed",
                "verification_receipt": "verification receipt reviewed",
                "execution_audit": "audit reviewed",
                "execution_recovery": "recovery reviewed",
                "after_action_learning": "learning reviewed",
                "completion_claim": "completion claim reviewed",
                "runtime_trace_sha256": runtime_trace_sha256,
                "verification_receipt_sha256": verification_receipt_sha256,
                "execution_audit_sha256": execution_audit_sha256,
                "execution_recovery_sha256": execution_recovery_sha256,
                "after_action_learning_sha256": after_action_learning_sha256,
                "completion_claim_sha256": completion_claim_sha256,
            }
        )
        if not path_cycle_ledger.ok:
            raise SystemExit("Path-shaped specialist cycle ledger should remain valid.")
        joined_path_cycle = path_cycle_ledger.output + " " + str(path_cycle_ledger.metadata)
        if any(fragment in joined_path_cycle for fragment in ["/\x55sers/", "/var/folders/", "/tmp/"]):
            raise SystemExit(f"Specialist cycle ledger leaked raw local path: {path_cycle_ledger.output} / {path_cycle_ledger.metadata}")
        if "<local-path>" not in str(path_cycle_ledger.metadata.get("raw_request") or ""):
            raise SystemExit(f"Specialist cycle ledger should redact path-shaped raw_request metadata: {path_cycle_ledger.metadata}")
        if not any("<local-path>" in str(command) for command in path_cycle_ledger.metadata.get("proof_queue", [])):
            raise SystemExit(f"Specialist cycle ledger proof queue should preserve redacted path marker: {path_cycle_ledger.metadata}")
        assert_safe_metadata(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")
        assert_specialist_review_only_authority(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")
        assert_specialist_proof_lane(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")
        assert_route_to_runtime_contract_token(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")
        assert_specialist_fresh_review_boundary_token(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")
        assert_specialist_post_run_closure_token(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")
        assert_specialist_cycle_ledger_token(path_cycle_ledger.metadata, "Path-shaped specialist cycle ledger")

        draft_cases = [
            (
                "specialist model draft: summarize this work history into a brief",
                "summarizer",
                "DRAFT_FALLBACK_PREVIEW",
                ["Jarvis specialist model draft", "Draft gate:", "model status: needs_attention", "No model draft was produced", "specialist handoff receipt:"],
            ),
            (
                "specialist model draft: fix this code bug and run focused tests",
                "code",
                "DRAFT_HELD_FOR_APPROVAL_GATE",
                ["Jarvis specialist model draft", "approval triggers detected: shell/code execution", "specialist execution readiness:", "specialist handoff receipt:"],
            ),
            (
                "specialist model draft: hello",
                "planner",
                "DRAFT_HELD_FOR_ROUTE_REVIEW",
                ["Jarvis specialist model draft", "ambiguous route: yes", "specialist route quality:", "Stop conditions:"],
            ),
        ]
        for case, expected_primary, expected_state, expected_text in draft_cases:
            result = runtime.handle(case)
            print(f"[{'ok' if result.verified else 'blocked'}] {case}")
            print(result.response[:2200])
            print()
            if not result.verified:
                raise SystemExit(f"Expected '{case}' to run.")
            for expected in expected_text:
                if expected not in result.response:
                    raise SystemExit(f"Specialist model draft missing expected text: {expected}")
            metadata = result.tool_results[0].metadata
            if metadata.get("primary_specialist") != expected_primary or metadata.get("draft_state") != expected_state:
                raise SystemExit(f"Specialist model draft state mismatch: {metadata}")
            if metadata.get("calls_model") or metadata.get("draft_produced"):
                raise SystemExit(f"Smoke specialist model draft should fall back without calling a model: {metadata}")
            if metadata.get("action_proposal_locked") is not True or metadata.get("can_emit_tool_proposals") is not False:
                raise SystemExit(f"Specialist model draft should keep tool proposals locked: {metadata}")
            if not metadata.get("handoff_command") or not metadata.get("quality_command") or not metadata.get("readiness_command"):
                raise SystemExit(f"Specialist model draft missed proof commands: {metadata}")
            assert_safe_metadata(metadata, "Specialist model draft")
            assert_specialist_review_only_authority(metadata, "Specialist model draft")
            assert_specialist_proof_lane(metadata, "Specialist model draft")
            assert_handoff_scorecard(metadata, "Specialist model draft")


if __name__ == "__main__":
    main()
