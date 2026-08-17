# Frozen V3 implementation published as V4

## Starting point

V4 is the public release identity of the frozen and tested V3 implementation. This promotion does
not copy V2 runtime state, rename the tested V3-compatible runtime interfaces, or claim that
pending supervised proofs have completed. The internal `jarvis_v2` package,
`launch_jarvis_v3*` launchers, `JARVIS_V3_*` settings, and `~/.jarvis_v3` storage remain unchanged.

## Current finish line

- The private development checkout records the automated-core evidence for the August 15, 2026
  functional preview in `AUGUST_15_FUNCTIONAL_PREVIEW.md`. That private operational evidence file
  is intentionally excluded from the sanitized public candidate; this is not an automatic
  production cutover.
- The English/Korean local push-to-talk, transcription, safe time routing, and audible-reply proof
  is complete. The generated bilingual on-device OCR proof, bounded Calendar and Apple Reminders
  reads, and recipient-confirmed Telegram, iMessage, Instagram, and KakaoTalk sends are also
  complete. KakaoTalk remains conservatively recorded as an outcome-unknown GUI attempt whose
  correct-chat and delivery evidence came from operator inspection and recipient confirmation.
  Run reboot, network-loss, and daemon proofs only with the operator present, following
  `V3_SUPERVISED_PROOF_RUNBOOK.md`. `V3_SUPERVISED_PROOF_RUNBOOK.md` is a sanitized public
  runbook included in the public candidate; it contains bounded procedures, not private proof
  payloads. In the public candidate, use `QUICKSTART.md` for supported launch flows and
  `CAPABILITIES.md` for the truthful capability and limitation boundary.
- Fix only release-blocking defects found by those supervised proofs, then freeze the candidate and rerun the
  affected focused tests plus the complete isolated regression suite.
- Preserve privacy-safe error recovery, one-shot approvals, typed arguments, and truthful success,
  failure, and outcome-unknown reporting in V4. New features belong to V5.
- Keep V2 production services and data unchanged until a deliberate, separately approved V3
  migration and cutover.

## Candidate-excluded private checks

The scheduler-continuity, inert-runtime, recovery, LaunchAgent, and serial-cutover designs under
`private_ops/` are intentionally absent from the public candidate. Run their bounded aggregate from
the private checkout with `python3 -B -m private_ops.run_smoke_test_all`; run
`python3 -B -m private_ops.test_run_smoke_test_all` to verify the aggregate's inventory, isolation,
output-limit, timeout, and process-cleanup controls. These synthetic checks do not activate services,
read V2 runtime data, change schedules, perform a cutover, or prove real reboot/network recovery.

## V4 publication boundary

- Publish only a newly rebuilt, sanitized, immutable V4 candidate after the full aggregate,
  independent credential scan, manual tree review, and history-free handoff pass.
- Exclude credentials, tokens, personal information, contacts, messages, raw live-proof payloads,
  local paths, runtime state, databases, logs, caches, vaults, and private operational documents.
- GitHub publication does not authorize compatibility V3 service installation, scheduler
  activation, runtime migration, or cutover.

## V5 direction

All feature development after V4 belongs to V5. Buzz remains a candidate product direction for
more natural voice interaction, coordinated services, research support, contextual assistance,
and proactive behavior bounded by explicit permissions. The private handoff is
`V5_BACKLOG_PRIVATE.md`, which is intentionally excluded from the public candidate. These are
directions, not finished-capability claims.
