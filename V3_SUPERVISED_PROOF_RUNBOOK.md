# Frozen Jarvis V3 / Buzz — Supervised Proof Evidence Inherited by V4

This runbook contains the smallest remaining operator-present proofs for the August 15 functional
preview. Historical V2 delivery receipts do not count as V3 proof. Keep V2 running as the rollback
generation, and do not install or enable V3 daemons during these checks.

V4 is a release-number promotion of these unchanged tested artifacts. The V3 proof names,
attestation markers, commands, and digests remain frozen so their evidence semantics are preserved.

<!-- JARVIS_V3_PREVIEW_PROOF_STATUS_V1
voice=passed
ocr=passed
google_calendar=passed
apple_reminders=passed
telegram=passed
imessage=passed
instagram=passed
kakaotalk=passed
calls=disabled
recovery_services=deferred
END_JARVIS_V3_PREVIEW_PROOF_STATUS_V1 -->

<!-- JARVIS_V3_PREVIEW_PROOF_ATTESTATION_V1
apple_reminders=passed|bounded_read_observed|2026-08-06|de3e51b216df552ec1851c24a6646896effa98d81ea72269a916ab9610593c19
telegram=passed|approval_bound_sender_success_recipient_confirmed|2026-08-05|1b4d9282b9ab5dfb1f46d150c54e0a94c23a2b42a824d8b77b8f3941ed3f0aa4
imessage=passed|approval_bound_sender_success_recipient_confirmed|2026-08-14|a1794153de918c38fea669aaac786c5280691cfa6e943e3b038a33e9986119a8
instagram=passed|approval_bound_sender_success_recipient_confirmed|2026-08-14|0b2548e460c700ae8f0c68459d8be5d1e51ba13666318dde6598a5d5a5683f7d
kakaotalk=passed|approval_bound_target_verified_recipient_confirmed|2026-08-07|42fd6c94669b18b33f124a8be4a01ef9af597f98ccd5045152e456f888ff3ad1
recovery_network=none
recovery_reboot=none
END_JARVIS_V3_PREVIEW_PROOF_ATTESTATION_V1 -->

The machine-readable attestation block is content-free: it stores only the proof lane, a
bounded evidence class, evidence date, and integrity digest. It contains no recipient, account,
message, approval ID, tool-run ID, local path, or payload. `none` is required for a connector that
is not passed. When `recovery_services` is deferred or disabled, both `recovery_network` and
`recovery_reboot` must be `none`. A passed connector or recovery/services lane must have its
lane-specific attestations updated in the same reviewed evidence change; editing only the status
cannot close the preview. Each recovery line binds its evidence class, the validated receipt's UTC
finalization date, finalized receipt-file SHA-256, source commit, active contract-set SHA-256, and
line digest. The finalization date must match the validated receipt exactly and cannot be in the
future. The receipt and line digests provide corruption detection while the reviewed files remain
under honest-operator custody. They are recomputable checksums, not signatures, authenticated
timestamps, tamper-proof authority, or protection from an actor able to rewrite the files and
recompute their digests. They do not replace the supervised observation and independent review.
Calls are explicitly disabled for the August 15 preview and have no call-success attestation;
enabling or live-testing a call channel is post-preview work requiring a separate supervised gate.
The two passed-line schemas are exact and contain no receipt path:

```text
recovery_network=passed|supervised_route_and_process_transition_observed|FINALIZED_UTC_DATE|RECEIPT_SHA256|SOURCE_COMMIT|CONTRACT_SET_SHA256|LINE_DIGEST
recovery_reboot=passed|supervised_boot_and_process_transition_observed|FINALIZED_UTC_DATE|RECEIPT_SHA256|SOURCE_COMMIT|CONTRACT_SET_SHA256|LINE_DIGEST
```

`LINE_DIGEST` is SHA-256 over the verifier's versioned recovery domain followed by the line's lane,
`passed`, evidence class, date, receipt hash, source commit, and contract-set hash, each separated by
one newline. Generate it only with the reviewed verifier helper; do not calculate or edit it by hand.
The closure verifier also binds all three machine evidence blocks—the August 15 required-gate
status, supervised-proof status, and proof attestations—to the exact corresponding blocks in
the documents stored at the immutable candidate's source commit. After recording or changing any
gate, proof disposition, or attestation, commit that content-free evidence state and rebuild the
candidate. A later status edit, even with a correctly regenerated digest, cannot close an older
candidate.

## Safety rules

- Use only fixed, non-private test phrases and freshly created test artifacts.
- Do not send, call, write calendar data, approve an action, or read an account until the exact
  target and expected outcome have been reviewed.
- A send counts only after recipient confirmation. A call request is not proof that a call rang.
- Treat uncertain outcomes as unknown, inspect the target state, and never retry automatically.
- Stop after the first unexpected external effect or private-data exposure.

## Already proved without account or production access

- Fresh V3 terminal startup and the read-only `status dashboard` route.
- Authenticated loopback dashboard lifecycle: unauthenticated status returned `401`, authenticated
  status returned `200`, and shutdown completed cleanly.
- Local Ollama conversation, clean-restart persistence, and a ten-turn mixed session.
- Offline connector contracts for Telegram, iMessage, Instagram, KakaoTalk, Google Calendar, and
  Apple Reminders.
- Voice/OCR dependency, routing, privacy, recovery, and fail-closed smoke coverage.

## 2026-08-14 content-free proof refresh

- The shared checkout was observed clean at
  `a95b9885f65c7c4af740888419d3213de42d348f` immediately after these supervised runs. The
  content-free receipts do not themselves encode a source commit, so this records honest-operator
  checkout custody rather than cryptographic execution-to-commit binding.
- Configured-model chat passed with provider `ollama`, model alias `llama3.1`, validated loopback
  routing, a nonempty response in 6099.8ms, and no fallback, tool handler, pending approval,
  stored context, conversation history, external model request, or storage fallback.
- The isolated mixed session passed 10 assistant turns across chat, research, calendar, and task
  lanes, including three model-backed chat turns and three latency samples. Chat p95 was 1139.2ms;
  all coherence and routing checks passed, and the connector fixtures used no network or account.
- Both receipts retained no conversation content and removed temporary state. Loopback routing and
  `OLLAMA_NO_CLOUD` express policy but do not attest the daemon's execution location; both runs
  reported `model_execution_locality_verified=false`.
- Fresh iMessage and Instagram sends each have a valid one-shot approval chain, an `ok` approved
  rerun with no unknown outcome, a passing automatic verification receipt, and operator-confirmed
  recipient visibility. Their content-free machine attestations are refreshed to 2026-08-14.
  No target, account, message, approval ID, tool-run ID, local path, or payload is retained here.

## Proof 1 — Local push-to-talk and spoken reply

**Status: passed 2026-08-03 on commit `c468063`.** The operator confirmed exact English recognition and
audible English output, followed by meaning-preserving Korean recognition, Korean-localized time
output, and audible Korean speech. The failed preliminary Korean transcripts were discarded and
did not count as proof.

**Where:** macOS Terminal, from the Jarvis V3 project folder. Enter the commands at the normal shell
prompt, not inside Telegram or at Jarvis's `You:` prompt.

Preflight sends nothing and does not request microphone access:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --check
```

With the operator present, use one fixed benign phrase per session:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --speak --language en
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --speak --language ko
```

Pass only if the microphone start and stop are visible, the transcript preserves the phrase's
meaning, the route is safe, the reply is audible, and the temporary clip disappears after the
turn. Discard rather than route an ambiguous transcript.

## Proof 2 — On-device OCR

**Status: passed 2026-08-03 on commit `c285063`.** A generated non-private English/Korean image
was read through the real terminal launcher and macOS Vision. Jarvis returned both lines and the
numeric marker exactly, and treated the recognized text only as content.

**Where:** macOS Terminal, at the normal shell prompt in the Jarvis V3 project folder.

Create a new non-private image containing known English and Korean test text, then run:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "ocr_image path=/absolute/path/to/the/test-image.png"
```

Pass only if the expected text is recognized, the result is presented as image content, and none
of the extracted text is treated as a Jarvis command. Do not use screenshots, receipts, contacts,
messages, or documents from a real account for this proof.

## Proof 3 — Read-only personal connectors

**Status: Google Calendar availability passed 2026-08-05; Apple Reminders passed 2026-08-06.** In
supervised terminal sessions, V3 used its separate read-only Calendar token for one bounded
availability check. It completed without
calendar mutation, mutation-capable scope, V2 state, or retention of query text, results, event
content, or account identifiers. An initial 2026-08-06 Apple Reminders attempt
did not count: the already-approved read failed with the known `app_not_running` precheck before a
verified result, the one-shot approval was consumed, and no launch or automatic retry was requested.
A later fresh request, after the operator manually opened Reminders and created the exact test list,
returned exactly one bounded sentinel from that list, verified the exact list identity, produced a
valid one-shot approval chain, and received a passing audit-backed verification receipt. Apple Event
collection materialization remains explicitly unattested.

Run one deliberately bounded V3 read from Google Calendar or Apple Reminders. Review the exact
time window or list boundary before starting. Pass only if V3 shows real current data without
writing, leaking identifiers, or using V2 state. Stop if authentication or macOS permission is not
clearly scoped to the intended V3 process.

**Where:** At the normal macOS Terminal shell prompt—not in Telegram and not at Jarvis's `You:`
prompt—start the V3 chat with:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_chat.py
```

Only after the `You:` prompt appears, enter the bounded Calendar or Reminders request there.

Google Calendar reads use the separate `JARVIS_GOOGLE_READONLY_TOKEN` and the exact read-only
event/calendar-list scopes. A human must create it with
`JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_calendar_auth.py readonly`
at the normal Terminal shell prompt; ordinary Jarvis commands never open OAuth or fall back to the
full-access mutation token. The supervised read proof explicitly forbids the launcher's
`full-access` mode: do not run it for this proof.

For Apple Reminders, first create a fresh list named `Jarvis V3 Proof` containing one harmless,
non-private sentinel item. Keep Reminders open yourself. At Jarvis's `You:` prompt in Terminal,
enter exactly, then review and complete the one-shot approval chain before the exact rerun:

```text
show my Apple reminders from list: Jarvis V3 Proof; limit: 1
```

Pass only if Jarvis shows that one sentinel after approval, reports that returned content is bounded,
and durable audit rows retain only the fixed runtime-owned content-free receipt. The connector does
not attest that Apple Event collection materialization itself is bounded. It checks that Reminders is already
running and never intentionally requests a launch, but the AppleScript boundary cannot attest that
an app-exit race makes launch impossible. Stop if another list appears, the result is oversized or
malformed, or permission/app state is unclear. The private result is transient; durable messages,
tool audits, runtime traces, and later chat grounding retain only the content-free receipt. Delete
the temporary sentinel/list manually after the proof; Jarvis must not mutate it.

## Proof 4 — Messaging channels

**Status: Telegram passed 2026-08-05; iMessage and Instagram were refreshed 2026-08-14; KakaoTalk
passed 2026-08-07.** Each
passed channel used one reviewed send through the one-shot approval chain and received
recipient-confirmed arrival. Telegram, iMessage, and Instagram also produced successful sender
results; KakaoTalk retained its more conservative outcome-unknown automation truth. Evidence
retains only the channel, approval disposition, bounded sender or target evidence, and recipient
confirmation—not recipient identity or message content.

For KakaoTalk, the approval-bound GUI execution truthfully remained outcome-unknown after reaching
the send-key step because automation could not independently prove one-to-one identity or delivery.
The operator then inspected the already-open exact conversation, confirmed the fixed test message
appeared in the intended chat, and obtained recipient confirmation. The consumed approval was not
replayed; the supervised operator observations supply the identity and delivery evidence that GUI
automation cannot attest.

**Where:** At the normal macOS Terminal shell prompt—not in Telegram and not at Jarvis's `You:`
prompt—start the same explicitly selected V3 chat:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_chat.py
```

Enter the send request only at Jarvis's `You:` prompt in that same session. Jarvis prints the
readiness and execution packets, then asks
`Approve approval #ID after this packet? [y/N]`. Type `y` only at that separate terminal prompt
after the exact stored recipient and fixed harmless message have been reviewed. Check the
destination app only to confirm delivery. Do not paste Terminal commands into Telegram or send the
test from the destination app itself.

Test only the channels the operator wants to call supported: Telegram, iMessage, Instagram, and
KakaoTalk. For each channel, use one harmless fixed message and one reviewed recipient. Preserve
the normal approval chain. Pass only when the sender receipt is successful and the recipient
confirms arrival. Record only content-free evidence; do not paste recipients or message contents
into public history.

For KakaoTalk, use only the supervised `preopened_exact_chat` mode; the ordinary search mode does
not count as this proof. In KakaoTalk, open and visually verify the exact one-to-one conversation,
including its participants, and leave that chat as KakaoTalk's front window. Switch directly to the
Terminal chat session. At Jarvis's `You:` prompt—not the normal shell prompt and not in KakaoTalk—
enter the following single line after replacing `EXACT DISPLAYED NAME` with the exact private title
shown by KakaoTalk. Keep the fixed harmless message unchanged:

```text
kakao preopened_exact_chat EXACT DISPLAYED NAME: Jarvis V3 Kakao supervised proof 8605
```

Use the exact displayed name, including capitalization and diacritics—not an `@handle`, phone
number, email address, alias, or partial name.
This mode binds that untouched name, fixed message, and `preopened_exact_chat` mode before approval;
it does not read Contacts, navigate, search, press Command-2, or press Command-F. At execution it
brings the already-running KakaoTalk process forward, requires the existing front window to have the
exact reviewed title, refuses multiple same-title windows, and rechecks the title before paste and
again before Enter. A same-named group can share the same visible title, so the operator's visual
one-to-one check remains required. Stop before typing `y` if the review packet does not show the
intended exact name, fixed message, `target_mode: preopened_exact_chat`, and a binding to those exact
values. After the one-shot run, require all three:
the approval and attempted execution are durably linked with the execution outcome explicitly
reported as unknown (never `APPROVAL_CHAIN_PROVEN`), the connector reports that the visible window
title was verified while one-to-one identity and delivery remain unverified, and the intended
recipient confirms arrival. Inspect the already-open thread and do not replay the consumed approval.

## Proof 5 — Recovery and optional services

**Status: deferred for the August 15 functional preview.**

The deferred status is authoritative: no recovery receipt path is required and both recovery
attestations remain `none`. Do not change this lane to `passed` based on smoke tests, documentation,
or generated contracts. A future passed disposition requires two distinct, finalized content-bound
receipts—one for the exact network down/up sequence and one for a real preboot/postboot sequence—
plus the owner-only active-contract manifest used by both observations. The closure verifier must
validate both receipts through `v3_recovery_receipt`, match their source and contract set to each
other and the reviewed candidate, and require each recorded date to equal the UTC finalization date
inside the validated receipt and not be in the future. It must receive all three local paths through
its explicit optional CLI flags. Those paths are required only for a passed recovery lane and are
never written into this runbook or closure output.

Even a future `passed` disposition is deliberately narrow. The observer records macOS route status
for a fixed probe target, boot-identity change, `launchctl` running state plus stable PID and exact
loaded program/arguments/working-directory/environment bindings for the expected processes,
scheduler absence, and source/contract continuity. It does
not exercise a dashboard request, bot update, message delivery, model response, Internet transfer,
scheduled workload, or other end-to-end service function. Therefore it cannot by itself prove
general network availability, functional daemon recovery, delivery recovery, or production
readiness; each claimed service workflow needs its own separately defined end-to-end proof.

After the functional checks are stable, separately test dashboard recovery, deliberate network
loss, reboot recovery, and any V3 daemon the operator may want. These require explicit approval and run
last. GitHub publication on August 17 does not authorize service installation, scheduler
activation, or V3 production cutover.

Telegram or iMessage daemon approval does not authorize scheduler execution. Scheduler activation
requires its own separately reviewed enable and proof after every enabled job and delivery target
has been inspected.

**Where:** macOS Terminal only, with the operator present and following a separately reviewed exact
command. Never start these checks from Telegram or a scheduled Jarvis job.

### Offline contract preparation — no activation

First, in macOS Terminal, generate a new owner-only contract set under `/private/tmp`. This reads the
selected owner-only V3 environment file only to validate its custody; it does not read its contents,
install a plist, call `launchctl`, start a service, change the network, or reboot:

```bash
python3 -B -m jarvis_v2.scripts.v3_active_launchagent_contracts \
  --env-file /absolute/path/to/the/reviewed-v3.env \
  --output-dir /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE
```

Stop here unless the operator separately reviews and approves the exact private install and service
activation procedure. This runbook intentionally contains no install, load, kickstart, restart,
network-change, or reboot command. Offline generation is not service proof. Before recording either
receipt, the separately installed contracts must exactly match this generated set, all three expected
V3 processes must satisfy the bounded `launchctl`/PID/module-presence check, the V3 scheduler must
remain disabled, and the source worktree must be clean. Process presence is not end-to-end service
health. The generated manifest also binds a content-free log-custody contract: a future installer
must resolve the account home from the account database, create or validate only
`Library/Logs/JarvisV3` without following symlinks, require that directory to be owner-only `0700`,
and precreate the three named regular log files as owner-only `0600` files before activation. The
offline generator does not create that directory or those files. A missing, linked, shared,
hard-linked, wrongly owned, wrongly permissioned, or unstable log target blocks recovery evidence.

### Network down/up receipt

Create an owner-only receipt directory once, then record the healthy connected baseline:

```bash
mkdir -m 700 /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt prepare \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/network.json \
  --kind network \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
```

Only after a separate, explicit operator approval changes the network to the deliberate offline
state, record the observation. This command observes; it does not change the network:

```bash
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt observe \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/network.json \
  --phase network-down \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
```

After the operator separately restores the network and confirms the same expected processes still
pass the bounded presence check, record the route/process observation and inspect the finalized
receipt:

```bash
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt observe \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/network.json \
  --phase network-up \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt inspect \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/network.json \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
```

### Reboot receipt

While the same source, contract set, disabled scheduler, expected-process presence, and route status
are still verified, record the preboot baseline:

```bash
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt prepare \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/reboot.json \
  --kind reboot \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
```

The reboot itself requires separate explicit approval and is not performed by Jarvis or by any
command in this runbook. After the operator returns to the same macOS Terminal workflow and verifies
the expected V3 processes pass the bounded presence check and the scheduler is still disabled,
record and inspect the postboot observation:

```bash
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt observe \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/reboot.json \
  --phase postboot \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
python3 -B -m jarvis_v2.scripts.v3_recovery_receipt inspect \
  --state /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/reboot.json \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json
```

Both inspect receipts must say `proof_state: finalized`, `verdict: proven`, and retain the same
source and contract-set bindings. Each attestation date must equal that validated receipt's UTC
`finalized_at` date. A pending, failed, unknown, mismatched, replayed, future-dated,
process-presence-failed, scheduler-enabled, or changed-file result does not count. Do not retry or
edit a finalized receipt; diagnose the state and begin a new reviewed proof with new paths. A valid
receipt still proves only the bounded observer facts listed above, not end-to-end service recovery.

## Candidate closure

After any fix, rerun its focused tests. On August 15, freeze the candidate, run the complete
isolated regression suite, and record the exact commit and limitations. Prepare only the sanitized
public export on August 16; do not publish or push before August 17 KST.

Record and commit the final content-free required-gate statuses, supervised-proof statuses, and
connector attestations before building the immutable candidate. Any later evidence-state change
requires a new candidate; do not reuse an older candidate marker.

After every connector lane is recorded as passed or explicitly disabled, and recovery/services is
recorded as passed, disabled, or deferred, run the offline closure verifier against the new
immutable candidate. First create an empty, owner-only external evidence directory; it must be
outside both the source and candidate trees. The verifier exclusively creates the final receipt
only after the same invocation's aggregate run and all postflight checks pass:

In the same private shell, load a reviewed nonempty JSON array of owner/contact deny literals with
hidden input and keep it only in the environment. Create a fresh 256-bit anchor inside the external
owner-only evidence directory and load it into the runtime environment. Reuse the exact deny list
and anchor for creation and verification, then unset the runtime variables. Never put the literals or
anchor in source files, the receipt, command arguments, shell history, or ordinary command output.

```bash
IFS= read -r -s 'JARVIS_PUBLIC_RELEASE_DENY_LITERALS_JSON?Private deny JSON: '
export JARVIS_PUBLIC_RELEASE_DENY_LITERALS_JSON
umask 077
JARVIS_V3_CLOSURE_EVIDENCE_DIR="$(mktemp -d /private/tmp/jarvis-v3-closure-evidence.XXXXXXXX)" || exit 1
chmod 700 "$JARVIS_V3_CLOSURE_EVIDENCE_DIR" || exit 1
export JARVIS_V3_CLOSURE_EVIDENCE_DIR
python3 - <<'PY'
import os
import secrets

root = os.environ["JARVIS_V3_CLOSURE_EVIDENCE_DIR"]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
fd = os.open(os.path.join(root, "anchor.txt"), flags, 0o600)
payload = (secrets.token_hex(32) + "\n").encode("ascii")
try:
    while payload:
        written = os.write(fd, payload)
        if written <= 0:
            raise OSError("anchor write made no progress")
        payload = payload[written:]
    os.fsync(fd)
finally:
    os.close(fd)
PY
IFS= read -r JARVIS_V3_CLOSURE_RECEIPT_ANCHOR \
  < "$JARVIS_V3_CLOSURE_EVIDENCE_DIR/anchor.txt"
export JARVIS_V3_CLOSURE_RECEIPT_ANCHOR
python3 -m jarvis_v2.scripts.v3_preview_closure \
  --source . \
  --candidate /private/tmp/<reviewed-candidate-directory> \
  --run-aggregate \
  --receipt-output "$JARVIS_V3_CLOSURE_EVIDENCE_DIR/closure.json"
python3 -m jarvis_v2.scripts.v3_preview_closure \
  --source . \
  --candidate /private/tmp/<reviewed-candidate-directory> \
  --verify-receipt "$JARVIS_V3_CLOSURE_EVIDENCE_DIR/closure.json"
unset JARVIS_PUBLIC_RELEASE_DENY_LITERALS_JSON JARVIS_V3_CLOSURE_RECEIPT_ANCHOR
```

The external receipt is canonical, content/path-free, owner-only, and self-consistent. It stores a
domain-separated HMAC over its full payload and a separate keyed deny-review commitment, never the
anchor. Its embedded digest proves internal self-consistency only: verification without the runtime
anchor deliberately returns
`self_consistent:true`,
`aggregate_execution_authenticated:false`, `valid:false`, and a nonzero status. Only an exact match
to the separately retained anchor returns authenticated `valid:true`. The receipt's own
`receipt_sha256`, a wrong anchor, or a changed payload do not authenticate the run. If receipt
creation reports
`closure_receipt_outcome_unknown`, inspect the destination before any retry because a linked final
receipt may remain after failed cleanup. Never reconstruct a receipt for an older aggregate run,
place it inside the candidate, overwrite it, or treat it as publication or cutover authority. After
verification, unset both secret runtime variables. Keep `anchor.txt` private and outside every public
candidate or repository.

Only if `recovery_services=passed`, add the separately reviewed owner-only active-contract manifest,
network receipt, and reboot receipt paths using the closure verifier's dedicated optional flags.
Never add those local paths while recovery remains deferred or disabled.

```bash
python3 -B -m jarvis_v2.scripts.v3_preview_closure \
  --source . \
  --candidate /private/tmp/REVIEWED-IMMUTABLE-CANDIDATE \
  --run-aggregate \
  --receipt-output "$JARVIS_V3_CLOSURE_EVIDENCE_DIR/closure.json" \
  --active-contract-manifest /private/tmp/jarvis-v3-active-contracts-REVIEWED-NONCE/manifest.json \
  --network-recovery-receipt /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/network.json \
  --reboot-recovery-receipt /private/tmp/jarvis-v3-recovery-proof-REVIEWED-NONCE/reboot.json
```

When `recovery_services=passed`, repeat those same three evidence flags on the separate
`--verify-receipt` command with the same runtime-only deny list and receipt anchor. Receipt
creation and verification both revalidate the exact bounded recovery evidence; omitting or
substituting any of the three inputs fails closed. When recovery is deferred or disabled, omit all
three flags.

Only `"ready": true` plus read-only receipt verification that reports
`aggregate_execution_authenticated:true` and `valid:true` closes the preview gate. The receipt never
authorizes GitHub publication, V3 cutover, service installation, schedule changes, or live actions.
