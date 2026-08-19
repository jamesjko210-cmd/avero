# Avero V4

> **Current release:** Avero is the product identity of the frozen and tested V4 preview.
> V2 is frozen and remains the running rollback generation until a separately supervised cutover.
> The internal Python package remains `jarvis_v2` temporarily for compatibility.

Avero is the new public and owner-facing name for this personal AI agent. The repository, docs,
and future releases use **Avero**. Existing `jarvis_v2` imports, `launch_jarvis_v3*` launchers,
`JARVIS_*` settings, local storage paths, tool names, and historical evidence remain stable
compatibility interfaces; changing those identifiers without a migration would break existing
installations. See [`BRANDING.md`](BRANDING.md) for the exact boundary.

This version promotion does not rename or migrate the tested runtime. The `launch_jarvis_v3*`
launchers, `JARVIS_V3_*` settings, `~/.jarvis_v3` storage, V3-named evidence, and the current local
workspace remain compatibility identities. New feature development belongs to the Avero V4.x
point-release series.

The detailed guide below was carried forward from the V2 implementation and is being migrated in
place. Any V2 service label, path, or live-proof reference is historical unless a V3-specific
readiness check confirms it. The private development checkout records the August 15 evidence in
`AUGUST_15_FUNCTIONAL_PREVIEW.md`; that private operational file is intentionally excluded from
the sanitized public candidate.

In the public candidate, `QUICKSTART.md` is the authoritative fresh-start and daily-use path and
`CAPABILITIES.md` is the authoritative supported-capability boundary. Complete the owner-only V3
environment setup before using any launcher. This README describes development checks; it does not
authorize scheduler or service activation.

Compatibility-safe development of a personal AI agent.

This is the Avero V4 release source. If another assistant is looking for the compatibility V3
dashboard or launcher, use this folder, not `jarvis-ollama`.

This version keeps the useful parts of the original Jarvis but separates the core systems:

- durable memory in SQLite
- human-readable memory in Obsidian
- skills as Markdown procedures
- explicit tool permissions
- reliable computer-control loops
- scheduled automations

Set `JARVIS_OBSIDIAN_VAULT` to point Jarvis at the exact vault you want. Sanitized releases should
configure this explicitly instead of relying on a machine-local development fallback.
Use `.env.example` only as a placeholder template. Follow `QUICKSTART.md` to create the real
owner-only environment outside the project checkout; never put populated environment files in the
repository. Keep local env variants, Google credential/token files, key files,
`.jarvis_v3_runtime/`, and the legacy `.jarvis_v2_runtime/` out of git.

## Development checks after Quickstart

Run these only after completing `QUICKSTART.md`. Keep `JARVIS_V3_ENV` pointed at the owner-only V3
environment created there so development commands cannot inherit another generation's state.

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" .venv/bin/python3 -m jarvis_v2.scripts.bootstrap_memory
```

That creates the local database and Obsidian folder structure without touching the old Jarvis.

Run the executive-core smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_core
```

Send one message to Jarvis from the terminal. The V3 launcher selects an available compatible
Python when the current `python3` lacks required launcher support:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "what time is it"
```

Pipeline input also works when no message arguments are provided:

```bash
printf "what time is it" | JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py --json
```

Preview how a command will route before any tool execution:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py --diagnose --json -- "run command python3 --version"
```

Risky commands sent through this one-shot path still use the normal Jarvis approval queue; `--diagnose` is read-only and does not queue approvals.
Approval readiness, last-look, and approval commands must run in one continuous runtime because
their short-lived review receipt intentionally does not survive a restart. Use the V3 chat launcher
for that flow:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_chat.py
```

Then enter `approval readiness ID`, `approval packet ID`, and `approve approval ID` as separate
Jarvis messages in that same session. Do not paste shell commands at the `You:` prompt.
Use `--json` when another script needs a receipt-style response with top-level verification, risk levels, handler/approval state, the runtime trace, planned actions, pending approvals, queued approval ids, and tool-result metadata.
Use `--` before the Jarvis message when it contains CLI-looking flags, as shown above.
If Jarvis cannot open its SQLite store, `--json` returns a startup receipt instead of a traceback,
including path-redacted storage labels and safe recovery steps. For a storage failure, point
`JARVIS_DATA_DIR` and `JARVIS_DB_PATH` at writable local paths in the selected V3 environment, run
`JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" .venv/bin/python3 -m jarvis_v2.scripts.bootstrap_memory --check`,
and run the same command without `--check` only after that read-only check reports configured
storage ready.

Run the one-shot command smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_ask_cli
```

Run the architecture-map smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_architecture
```

Run the agent-harness status smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_harness
```

Run the roadmap smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_roadmap
```

Run the model-routing status smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_model_status
```

Run the function/tool smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_functions
```

Run the Hermes-style skill smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_skills
```

Run the full-assistant foundation smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_agi_slice
```

Run the capability-map smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_capabilities
```

Run the browser/computer/scheduler layer smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_next_layer
```

Run the read-only computer-control readiness and task planning smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_computer_plan
```

Run the read-only action rehearsal smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_action_rehearsal
```

Run the read-only assistant-turn rehearsal smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_assistant_turn_rehearsal
```

Run the conversation continuity smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_conversation
```

Run the recent-activity catch-up smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_continuity
```

Run the work-session focus brief smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_focus
```

Run the brain-dump organizer smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_organize
```

Run the goals/projects smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_goals
```

Run the lightweight task queue smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_tasks
```

Run the tactical daily plan smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_daily_plan
```

Run the durable decision log smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_decisions
```

Run the people-memory smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_people
```

Run the structured-preferences smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_preferences
```

Run the proactive review smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_proactive
```

Run the voice-output smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_voice
```

Run the approval-gated shell/code smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_shell
```

Run the Obsidian Inbox/recent-file ingest smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_ingest
```

Run the Inbox source-to-memory projection custody smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_ingested_source_projection_store
```

Run the Jarvis Obsidian note search smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_notes
```

Run the default automation bundle smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_scheduler_basics
```

Run the tool-audit smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_audit
```

Run the pending-approval queue smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_approvals
```

Run the safety-boundaries smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_safety
```

Run the approval-review smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_approval_review
```

Run the privacy-boundary smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_privacy
```

Run the personal-integration boundary smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_personal
```

Run the safe next-action advisor smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_next_step
```

Run the state snapshot smoke test:

```bash
python3 -m jarvis_v2.scripts.smoke_test_state
```

### Scheduler and worker-service boundary

The scheduler and private durable subagent runner are service internals, not fresh-start commands.
Do not start either from the functional-preview checkout or sanitized public candidate. Scheduler
activation, worker-service activation, LaunchAgent installation, and production cutover require a
separately reviewed, operator-present post-cutover procedure. No activation command is provided in
this public guide. V2 remains the running rollback generation until that separate cutover is
approved.

The subagent runner is designed around an owner-only local Unix socket, content-free durable
metadata, and a closed registry of replay-safe handlers. Those design properties do not turn it
into an automatically authorized service.

Use the authenticated loopback-dashboard flow in `QUICKSTART.md`. It creates or validates the
dashboard password inside the selected owner-only V3 environment, starts the root V3 dashboard
launcher, and copies the password without printing it or placing it in shell history. Do not
generate or export a competing dashboard password, and do not store one in the project checkout.

`JARVIS_STATUS_AUTH_TOKEN` is required and must be a unique printable 32-256 character secret.
When the browser prompts, use username `jarvis` and the locally stored password. The launcher fails
closed before loading Jarvis data if the password is missing or invalid, and it is never placed in
the dashboard URL.

On any dashboard `401` response, return to the authenticated-dashboard section of `QUICKSTART.md`.
Stop the dashboard, validate the same selected owner-only V3 environment, copy its password with
the documented helper, and restart the root V3 dashboard launcher. The helper rejects duplicates,
symlinks, non-files, unsafe permissions, and invalid passwords before invoking the local clipboard.
The authentication response is deliberately generic and never distinguishes credential errors or
echoes credentials or request headers.

By default the dashboard binds `127.0.0.1:8766`. To change that locally, put
`JARVIS_STATUS_HOST` and `JARVIS_STATUS_PORT` in the selected owner-only V3 environment, or pass
`--host` / `--port` on the command line. Because HTTP Basic credentials require transport
protection, the server accepts only loopback hosts (`127.0.0.0/8`, `::1`, or `localhost`) and
refuses LAN/all-interface binds until TLS is available.
IPv6 hosts such as `::1` are shown with URL brackets, for example `http://[::1]:8766`.
If the port is already in use, rerun with a different `--port` or set `JARVIS_STATUS_PORT` locally.

Ask Jarvis for the same launch info:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "status dashboard"
```

The root dashboard launcher in `QUICKSTART.md` is the supported preview entrypoint. Module-level
server entrypoints are development internals and are not an alternate fresh-start recipe.

The full private development checkout may contain a dashboard service template, but the
sanitized public candidate deliberately omits every service template. Service installation is unavailable
in that candidate. In the private checkout it remains an explicit, operator-present post-cutover
action governed by the private supervised runbook; this public README intentionally provides no
installation or `launchctl` command. The template is never installed or loaded automatically and
must never replace or disturb the V2 rollback service.

The dashboard exposes `http://127.0.0.1:8766/api/status`, a read-only agent-harness status at `http://127.0.0.1:8766/api/harness-status`, a read-only harness-cycle preview at `http://127.0.0.1:8766/api/harness-cycle?request=...`, a read-only AGI-gates report at `http://127.0.0.1:8766/api/agi-gates`, a read-only safety status at `http://127.0.0.1:8766/api/safety-status`, a read-only autonomy plan at `http://127.0.0.1:8766/api/autonomy-plan?request=...`, read-only computer-control status, readiness, task-plan, and action-packet endpoints at `http://127.0.0.1:8766/api/computer-control-status`, `http://127.0.0.1:8766/api/computer-readiness?objective=...`, `http://127.0.0.1:8766/api/computer-task-plan?objective=...`, and `http://127.0.0.1:8766/api/computer-action-packet?spec=...`, read-only focus brief and work-session packet endpoints at `http://127.0.0.1:8766/api/focus-brief?objective=...` and `http://127.0.0.1:8766/api/work-session-packet?objective=...`, a read-only next-session resume plan at `http://127.0.0.1:8766/api/next-session-plan?objective=...`, a read-only readiness report at `http://127.0.0.1:8766/api/readiness-report`, a read-only capability map at `http://127.0.0.1:8766/api/capabilities?focus=safety`, read-only model routing and model-planner prompt preview endpoints at `http://127.0.0.1:8766/api/model-status` and `http://127.0.0.1:8766/api/model-planner-prompt-preview?request=...`, a read-only build-delta checkpoint at `http://127.0.0.1:8766/api/build-delta`, read-only assistant-turn and action rehearsal endpoints at `http://127.0.0.1:8766/api/assistant-turn-rehearsal?message=...` and `http://127.0.0.1:8766/api/action-rehearsal?request=run%20command%20python3%20--version`, read-only personal integration status and action-preview endpoints at `http://127.0.0.1:8766/api/integration-status` and `http://127.0.0.1:8766/api/integration-action-preview?connector=email&action=send%20draft%20reply`, read-only voice setup, reply preview, spoken-turn rehearsal, audio-file transcription planning, transcript review, confirmation, and lifecycle endpoints at `http://127.0.0.1:8766/api/voice-setup`, `http://127.0.0.1:8766/api/voice-reply-preview?text=Jarvis%20is%20ready`, `http://127.0.0.1:8766/api/spoken-turn-rehearsal?message=run%20command%20python3%20--version`, `http://127.0.0.1:8766/api/voice-file-transcription-plan?path=...`, `http://127.0.0.1:8766/api/voice-transcript-review?transcript=...`, `http://127.0.0.1:8766/api/voice-confirmation?transcript=...`, and `http://127.0.0.1:8766/api/voice-lifecycle?transcript=...`, read-only chat-continuity, chat-response-health, chat-loop, chat-context, and chat-prompt endpoints at `http://127.0.0.1:8766/api/chat-continuity`, `http://127.0.0.1:8766/api/chat-response-health`, `http://127.0.0.1:8766/api/chat-loop-preview?message=...`, `http://127.0.0.1:8766/api/chat-context?prompt=...`, and `http://127.0.0.1:8766/api/chat-prompt-preview?prompt=...`, a read-only chat-safety report at `http://127.0.0.1:8766/api/chat-safety`, a read-only skill match preview at `http://127.0.0.1:8766/api/skill-match-preview?request=...`, read-only brain-loop, learning-review, privacy-report, morning-startup, and weekly-review-context reports at `http://127.0.0.1:8766/api/brain-loop`, `http://127.0.0.1:8766/api/learning-review`, `http://127.0.0.1:8766/api/privacy-report`, `http://127.0.0.1:8766/api/morning-startup`, and `http://127.0.0.1:8766/api/weekly-review-context`, read-only conversation history at `http://127.0.0.1:8766/api/conversation`, a read-only approval review at `http://127.0.0.1:8766/api/approvals`, read-only approval detail and approval execution packet endpoints at `http://127.0.0.1:8766/api/approval-detail?id=1` and `http://127.0.0.1:8766/api/approval-packet?id=1`, read-only work queue and next-action packet endpoints at `http://127.0.0.1:8766/api/work-queue` and `http://127.0.0.1:8766/api/next-action-packet`, read-only resume briefs at `http://127.0.0.1:8766/api/return-brief`, `http://127.0.0.1:8766/api/handoff-brief`, and `http://127.0.0.1:8766/api/session-closeout`, plus a local web chat endpoint at `http://127.0.0.1:8766/api/chat` that still uses Jarvis approval gates and returns `runtime_route` plus read-only `chat_loop_preview` metadata.

Use the dashboard's `Channel Health` panel or `http://127.0.0.1:8766/api/channel-health` for read-only messaging/calling health from audit metadata only; it suppresses recipient names, message text, approval text, and raw tool output, and it does not send messages or start calls.

Use `storage status` or `http://127.0.0.1:8766/api/storage-status` for a read-only, metadata-only storage routing packet that shows configured storage readiness, runtime fallback state, safe recovery env names, and path-redacted display labels.

Install optional computer-control dependencies:

```bash
.venv/bin/python3 -m pip install -r requirements.txt
```

Start the conversational chat loop:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_chat.py
```

Start chat with spoken responses:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_chat.py --speak
```

Jarvis uses Ollama by default: `JARVIS_MODEL_PROVIDER=ollama`, with `JARVIS_CHAT_MODEL` or
`OLLAMA_MODEL` defaulting to `llama3.1`.
Ollama is loopback-only: `OLLAMA_HOST` defaults to `127.0.0.1:11434`, and an explicit `localhost`
host is pinned to `127.0.0.1` before use. Invalid or non-loopback values block Ollama calls and
readiness probes. Ollama requests do not follow redirects or trust proxy environment variables,
and diagnostics and status receipts never show a rejected raw `OLLAMA_HOST` value.
This proves that Jarvis sends request bytes only to the loopback daemon; it does not by itself prove
that the daemon will execute every model on-device. Ollama supports cloud-backed models through a local daemon.
For intended on-device-only inference, configure the Ollama daemon with cloud features disabled
(`OLLAMA_NO_CLOUD=1`, followed by an Ollama restart) and avoid cloud model aliases. To Jarvis,
`OLLAMA_NO_CLOUD=1` records operator intent; it is not daemon attestation or proof of model-execution locality.
Jarvis sends stored profile, preferences, memories, saved skills, and prior chat history to Ollama only when
the destination is validated loopback, the selected model is not a known cloud alias, and both
`OLLAMA_NO_CLOUD=1` and `JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1` are explicitly configured.
The separate stored-context consent defaults to `0`. Otherwise Ollama remains usable for the current message
in stateless mode while all stored personal context stays local. Model execution remains unverified in every
case, including when both flags are enabled.
If the local model is offline, Jarvis falls back to a small built-in conversational mode that still uses profile,
preferences, saved skills, and memory context instead of returning a raw connection error.
Planner model calls are bounded by `JARVIS_MODEL_TIMEOUT_SECONDS`, defaulting to `2.5`, so slow or missing local models do not stall tool routing.
Free-form chat uses `JARVIS_CHAT_TIMEOUT_SECONDS`, defaulting to `20`, so longer conversational answers have room without slowing planner decisions.
Free-form chat replies are capped by `JARVIS_CHAT_MAX_REPLY_TOKENS`, defaulting to `300` and clamped to a maximum of `4096`, to keep local model turns bounded.
The recent conversation history window uses `JARVIS_CHAT_MAX_HISTORY_MESSAGES`, defaulting to `16` and clamped to a maximum of `64`; lower it only when testing the latency/continuity tradeoff.
Ask `chat safety` to inspect the grounding rules for conversational memory and the approval boundary for actions.
Ask `chat response health` to inspect whether recent replies came from the model, grounded memory, or the safe fallback without calling the model.
Ask `chat continuity brief` to catch up on the current conversation thread, latest response path, and safe next checks without acting.
Ask `chat loop preview: <message>` to inspect the perceive -> ground -> classify -> reply path without calling the model or executing tools.
Ask `chat prompt preview: <message>` to inspect the conversational system prompt, grounding packet, reply path, and safety boundary without calling the model.
Ask `session learning preview` to review possible memories, preferences, tasks, and reusable skills from the current session before saving anything.
When a high-risk action is blocked, review `approval packet #ID` before approving and rerunning it.
Set `JARVIS_USE_MODEL_PLANNER=0` to disable model-backed tool planning and use only deterministic routing.

To try GPT-5.6 through the OpenAI Responses API, keep the key in the local environment and opt in explicitly:

```bash
JARVIS_MODEL_PROVIDER=openai
JARVIS_CHAT_MODEL=gpt-5.6-terra
JARVIS_PLANNER_MODEL=gpt-5.6-luna
JARVIS_CHAT_REASONING_EFFORT=low
JARVIS_PLANNER_REASONING_EFFORT=low
JARVIS_OPENAI_MAX_OUTPUT_TOKENS=25000
JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT=0
OPENAI_API_KEY=your-local-key
```

`gpt-5.6` selects Sol for maximum capability; Terra is the balanced Jarvis chat default and Luna keeps planner
calls faster and less expensive. OpenAI routing is opt-in, uses the Responses API with `store: false`, and never
places the API key in Jarvis config objects, status output, audit metadata, or smoke fixtures. `store: false` controls
the Responses persistence request flag; it is not a zero-retention guarantee. OpenAI's default abuse-monitoring logs
may retain prompts and responses for up to 30 days unless the API project has approved retention controls. Check the
project's data-control policy before sending sensitive personal content. `model routing status` reports that this
account-level policy is unverified and checks local configuration without making a paid API request. Ollama remains
unchanged when the provider is omitted.
Ordinary OpenAI chat necessarily sends the current message you explicitly type for that turn. Stored profile,
preferences, memory, saved skills, and prior conversation history are separate: they stay local and are omitted from
OpenAI requests by default. Set `JARVIS_ALLOW_REMOTE_PERSONAL_CONTEXT=1` only to opt in to sending those stored
personal-context sources to the configured OpenAI model. This setting does not make the current typed message local.
Prior-history consent is non-retroactive. Jarvis binds each reusable chat turn to an opaque policy epoch covering the
provider, model, destination, and explicit consent state; changing any of them invalidates older history grants, and
turning consent on later does not release turns created while it was off. Legacy, malformed, stale-process, tool-derived,
and mixed-restriction rows remain local-only. Exact role/content hashes bind message and derived-memory provenance, so
editing or merging content cannot inherit an older disclosure grant. Before eligible stored context crosses a model
boundary, Jarvis records a content-free prepared receipt, acquires the same crash-safe process fence used by policy
rotation, revalidates authority and content inside that fence, and finalizes the receipt as confirmed, blocked, or
uncertain. Conversation compaction uses the same contract: stale/ineligible prefixes advance only through content-free
local receipts with no model call, while eligible digests retain exact source lineage and destination provenance.
To request the intended Ollama-only path, use a validated loopback destination and `OLLAMA_NO_CLOUD=1`; if the turn
should include stored context, also set `JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1`. These are operator
intent and consent, not proof: Jarvis cannot independently attest the daemon's actual execution location.
`model routing status`
reports this policy using content-free flags and source names, without copying message or personal-context content
into status metadata.
Every OpenAI request includes a stable safety identifier as recommended by OpenAI. Because Jarvis is an owner-only
application, it uses one fixed single-owner pseudonym; it is not derived from your name, email, Telegram/iMessage ID,
local username, or API key. `model routing status` discloses that the pseudonym is sent but never displays its value.
If Jarvis later becomes multi-user, this must be replaced with one distinct privacy-preserving identifier per user.
GPT-5.6's `max_output_tokens` covers hidden reasoning, visible output, and formatting together. Jarvis therefore keeps
the local `JARVIS_CHAT_MAX_REPLY_TOKENS=300` behavior for Ollama but uses a separate OpenAI ceiling, configured by
`JARVIS_OPENAI_MAX_OUTPUT_TOKENS` and bounded to 1,000-128,000 tokens. The default is 25,000, following OpenAI's
initial reasoning-model experimentation guidance; it is a maximum rather than expected usage. Routine OpenAI chat
defaults to `low` reasoning to control latency and cost. `model routing status` displays the active ceiling without
making a paid request.
After a real model turn, `chat response health` shows content-free input, cached-input, output, reasoning, and total
token counts when the provider supplies a consistent usage object. Model-planner usage is preserved in the existing
runtime trace receipt. Malformed or inconsistent counters are reported as unavailable; prompt text, response text,
response IDs, and backend metadata are never copied into these usage receipts.
Scheduled conversation compaction is a separate privacy boundary. By default it sends no old conversation text
through OpenAI or an unverified Ollama route and keeps its watermark unchanged. Ollama becomes eligible through
the normal stored-context path only when its loopback/model checks pass and both `OLLAMA_NO_CLOUD=1` and
`JARVIS_ALLOW_UNVERIFIED_OLLAMA_PERSONAL_CONTEXT=1` are set. Alternatively, set
`JARVIS_ALLOW_REMOTE_COMPACTION=1` only when older conversation batches are explicitly approved for the configured
model route. Ollama execution locality remains unverified. Interactive chat, planning, and explicitly invoked model
tools keep using the selected provider.
Set `JARVIS_WATCHED_DIRS` to a colon-separated list of folders for recent-file digests.
Set `JARVIS_SMOKE_MODULE_TIMEOUT_SECONDS`, defaulting to `180`, to tune the aggregate smoke suite's per-module timeout for slow local machines.

## Safety Contract

Jarvis V2 is designed to be useful without harming its operator, the computer, or its data.

- Read-only and local-safe tools can run automatically.
- High-risk actions require explicit approval and are saved to the pending approval queue.
- Personal-data actions, including clipboard reads, focused/running app activity, and computer observation/control, are approval-gated.
- Shell commands and file writes are blocked unless explicitly approved.
- Blocked actions include a safety receipt with the approval ID and exact review/packet/approve/dismiss commands.
- Approval requires a last-look packet first: run `approval packet #ID`, then approve only if the exact stored request is still trusted.
- Pending approvals are mirrored to `Jarvis/Automations/Pending Approvals.md` in Obsidian.
- Brain-dump organizing writes only Jarvis-owned SQLite/Obsidian records.
- Destructive actions such as deleting memories, deleting skills, clearing Inbox, or deleting jobs are high-risk.
- Jarvis should report blocked actions honestly instead of pretending they happened.
- Use `help safety` to see the safety rules inside Jarvis.

## Diagnostics

Two opt-in diagnostics inspect service reachability and feature readiness on this Mac. Both are excluded from
`smoke_test_all`, never run in CI, and always exit 0.

- **Live service check** (free external APIs are reachable):
  ```
  JARVIS_LIVE_SMOKE=1 python3 -m jarvis_v2.scripts.smoke_test_live
  ```
- **Feature readiness check** — a one-command health board for the phone-facing features on
  the real Mac. It covers macOS Contacts resolution, the send-decision pre-flight (what
  `text <name>` would do), voice transcription and warmup, shared dashboard/local/Telegram voice
  wiring, image OCR, research/chat/mixed-conversation readiness, personal integration proof rows,
  daily-brief composition, scheduled morning-brief freshness and 7-day job proof, acceptance
  coverage/gaps/next-proof rows, aggregate-smoke proof, operator-workflow evaluation proof, channel
  health, primary-daemon process/source freshness, guardrail, proof-ledger, completion, and learning control-plane rows, phone approvals,
  phone control, and the Telegram owner channel:
  ```
  # Select the compatible interpreter used for the current V3 environment.
  export JARVIS_PYTHON="${JARVIS_PYTHON:-python3}"
  JARVIS_LIVE_CHECK=1 "$JARVIS_PYTHON" -m jarvis_v2.scripts.live_check --contact NAME
  ```
  Read-only by default; Telegram, Instagram, iMessage, and KakaoTalk have separate recipient-confirmed V3 evidence,
  while calls are disabled for the August 15 preview; any later re-enable and live proof requires a separate, operator-present approval gate. KakaoTalk required operator verification
  after an outcome-unknown GUI attempt. This is not live proof of
  sends, calls, approvals, microphone capture, or reboot survival. To also send one real Telegram test message to the owner, add
  `JARVIS_LIVE_CHECK_SEND=1`. The contact name can also come from `JARVIS_LIVE_CHECK_CONTACT`.
  Note: the first Contacts query triggers the macOS Contacts permission prompt for the Python
  process — grant it once, or the check times out and reports the contact as not found.

### Daemon Recovery

The readiness checks above never restart daemons, call `launchctl`, send messages, or prove reboot survival.
Diagnostics never execute these commands.
The sanitized public candidate contains the offline contract generator and content-free recovery
receipt helpers, but no LaunchAgent templates, installed contracts, installation or restart
commands, or cutover authority. Those helpers cannot activate services, change network state, or
reboot the computer. The candidate must use only the manual V3 launchers documented in
`QUICKSTART.md`; daemon recovery is unavailable from that candidate.

After a separately approved V3 cutover, an operator-present recovery may follow the exact private
supervised runbook from the private checkout. This public README intentionally contains no
`launchctl` or daemon-restart command. A post-cutover recovery must never replace, stop, or disturb
the V2 rollback service unless the separately reviewed cutover procedure explicitly authorizes that
exact change.

The `daemon startup` row reads the local process table and compares each primary daemon's start
time with a bounded set of service source files. A stale result means the running process has not
loaded current source; it reports the affected service but does not grant restart authority.
Live reboot proof still needs the operator present and is not established by documentation or a
readiness row.

## Photo / document text (on-device OCR)

Send a photo or an image document to the Telegram bot and Jarvis reads the text out
of it — receipts, whiteboards, screenshots, business cards — and sends the extracted
text back. OCR runs **on-device via the macOS Vision framework** (no cloud, no API key;
a tiny bundled Swift helper, compiled and cached on first use). Recognizes **Korean and
English** by default (override with `JARVIS_OCR_LANGUAGES=ko-KR,en-US,…`). The text is
surfaced for you to use, not executed as a command. Locally you can also run:

```
ocr_image path=/path/to/image.png
```

## Apple Reminders (native macOS, read-only)

Ask "what's on my apple reminders?" (also "mac reminders", "iphone reminders",
"reminders app") and Jarvis lists a bounded number of open to-dos from the macOS
**Reminders app**, grouped by list. For the smallest exact-list read, use
`show my Apple reminders from list: Jarvis V3 Proof; limit: 1`. This reads native
Apple reminders via `/usr/bin/osascript` (no cloud) and never creates or completes
items. Because this reads personal data, every real query requires a reviewed one-shot
approval; invalid or injected arguments are rejected before approval or query. Returned
content is capped at the requested limit (plus one internal record for truncation), but
Apple Event collection materialization is not independently bounded or attested. It
contains no app-launch request and checks that Reminders is already running,
but the check and later Apple-event send are separate steps; Jarvis does not claim that
this race has been eliminated or that an app launch is impossible. Plain "my reminders" / "list
reminders" still refers to Jarvis's own owner-Telegram reminders/timers. The due
send path can prove Telegram API acceptance; phone-display delivery remains live-user
confirmation. First use
triggers the macOS automation permission prompt for Reminders — grant it once.

## Google Calendar authorization

Jarvis keeps Calendar reads and mutations on separate OAuth tokens. Event, availability, and
calendar-list reads request only `calendar.events.readonly` and
`calendar.calendarlist.readonly`; they never fall back to the full-access token and never open a
browser automatically. Put the Google OAuth desktop-client JSON outside the repository at
`~/.jarvis_v3/google_credentials.json`, make it owner-only with `chmod 600`, then run this yourself
in a normal Mac Terminal:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_calendar_auth.py readonly
```

The read-only token is stored separately at
`~/.jarvis_v3/google_calendar_readonly_token.json` with owner-only permissions. The authorization
check prints no calendar names, event titles, or identifiers. Create/update/delete operations do
not use this token. Their existing full-access authorization remains a separate human-run flow and
is useful only after reviewing the broader Google consent request:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_calendar_auth.py full-access
```

## Talk to Jarvis (local push-to-talk)

Speak to Jarvis directly from the Mac — no Telegram needed. Press Enter to start
recording from the mic, Enter again to stop; the clip is transcribed locally with
the `whisper` CLI and the transcript runs through the normal approval-gated runtime.
The V3 launcher speaks replies by default and keeps the same runtime alive for the
whole session:

```
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --language ko
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --no-speak
```

The module-level commands remain available for development and diagnostics:

```
python3 -m jarvis_v2.scripts.talk --check    # no microphone, recording, or private-data access
python3 -m jarvis_v2.scripts.talk           # print replies
python3 -m jarvis_v2.scripts.talk --speak   # also speak replies aloud
python3 -m jarvis_v2.scripts.talk --language ko  # force Korean ASR for this local session
python3 -m jarvis_v2.scripts.talk --list-mics
```

Needs `ffmpeg` (mic capture) and one configured local Whisper transcriber. These
optional dependencies are not assumed to be installed on a fresh Mac. First run
`JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --check`;
it reports missing or invalid setup without listing/opening microphones, recording or
transcribing audio, starting Jarvis, requesting microphone permission, or reading private data.
Follow its bounded recovery and rerun it until it reports `READY`.

The built-in mic is auto-selected only after the operator starts the real voice session; override
with `--mic N` or `JARVIS_VOICE_MIC=N`. That real session may trigger the macOS Microphone
permission prompt for the terminal application. Review and grant it yourself only if intended.
Use `--language ko` for short Korean commands when automatic language detection is unreliable;
leave the default `auto` mode for multilingual/English speech. (You can also send a voice memo to
the Telegram bot for the same flow.)

## Tunables (environment variables)

Optional knobs for the phone/voice/image features (all have sensible defaults):

| Variable | Default | What it does |
| --- | --- | --- |
| `JARVIS_OWNER_TELEGRAM` | — | Your numeric Telegram chat id (required for the bot/owner channel). |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token (required for the bot). |
| `JARVIS_MORNING_BRIEF` | `09:00` | Daily owner-Telegram morning-brief push time. Set to `HH:MM`; set empty to disable. |
| `JARVIS_MORNING_BRIEF_HHMM` | — | Legacy alias for `JARVIS_MORNING_BRIEF` when the new variable is unset. |
| `JARVIS_VOICE_WHISPER_CLI_MODEL` | `base` | Whisper model for transcription (`small`/`medium` improve Korean accuracy, larger downloads). |
| `JARVIS_VOICE_WHISPER_CLI_LANGUAGE` | auto | Force a transcription language (e.g. `ko`); default auto-detects. |
| `JARVIS_VOICE_WHISPER_CLI` | `which whisper` | Path to the whisper CLI if not on PATH. |
| `JARVIS_VOICE_WARMUP` | off | Set to `1` to warm the local audio transcriber in the background at dashboard UI start using a generated silent WAV; does not request microphone access. |
| `JARVIS_VOICE_MIC` | built-in mic | Mic device index for push-to-talk (`talk --list-mics` to see indices). |
| `JARVIS_VOICE_MAX_SECONDS` | `120` | Hard cap on a single push-to-talk recording (5–600). |
| `JARVIS_OCR_LANGUAGES` | `ko-KR,en-US` | Image-OCR recognition languages (comma-separated). |
| `JARVIS_CACHE_DIR` | `~/.cache/jarvis-v3` | Where the compiled OCR helper is cached. |
| `JARVIS_LIVE_CHECK` / `JARVIS_LIVE_CHECK_SEND` / `JARVIS_LIVE_CHECK_CONTACT` | — | Opt-in live health check (see Diagnostics). Telegram, Instagram, iMessage, and KakaoTalk have recipient-confirmed V3 send proofs; calls are disabled for the August 15 preview, and any later re-enable and live proof requires a separate, operator-present approval gate. KakaoTalk required operator verification after an outcome-unknown GUI attempt. |
| `JARVIS_LIVE_SMOKE` | — | Opt-in live external-API reachability check. |

## Current Core

The first Jarvis V2 brain loop now lives in:

- `jarvis_v2/agent/runtime.py` — orchestrates planning, permissions, execution, verification, and memory logging
- `jarvis_v2/agent/chat.py` — conversational replies using Ollama, with a grounded fallback when Ollama is offline
- `jarvis_v2/agent/planner.py` — deterministic starter planner
- `jarvis_v2/agent/executor.py` — runs planned tool calls through the permission policy
- `jarvis_v2/agent/verifier.py` — checks whether planned actions succeeded
- `jarvis_v2/tools/registry.py` — registers tools with risk levels and toolsets
- `jarvis_v2/tools/permissions.py` — risk-level approval policy
- `jarvis_v2/tools/safety.py` — live safety status, risk inventory, and pending-approval summary
- `jarvis_v2/tools/files.py` — file list/find/read/write handlers
- `jarvis_v2/tools/system.py` — macOS/system handlers
- `jarvis_v2/tools/voice.py` — macOS text-to-speech output
- `jarvis_v2/tools/shell.py` — approval-gated local command runner
- `jarvis_v2/tools/ingest.py` — Obsidian Inbox ingest and watched-file digests
- `jarvis_v2/tools/notes.py` — search and read notes inside the Jarvis Obsidian folder
- `jarvis_v2/tools/audit.py` — recent tool-run audit log

## Six-Layer Assistant Architecture

Jarvis V2 maps the standard assistant architecture into explicit, inspectable pieces:

1. Perception and input handling: CLI/text chat is live; macOS speech output is live; ASR/wake-word remain future migrations.
2. Natural language understanding: deterministic routing plus optional model-backed planning and conversational chat.
3. Reasoning and planning: planner, autonomy plans, focus briefs, safe next actions, and verification.
4. Memory: short-term chat history, SQLite long-term memory, and human-readable Obsidian notes.
5. Action and tool execution: registered tools, risk levels, approval queue, safety receipts, and audit logs.
6. Learning and continuous improvement: skill drafts, reflections, daily/weekly reviews, Memory Tree snapshots, and Mission Control.

Use `architecture map` inside Jarvis to see the live version of this map with safety boundaries and next gaps.

Jarvis completion principle: finish Jarvis as an agent harness, not just an agent. The model is the reasoning core, but the finished assistant needs a dependable surrounding runtime: stable chat loop, perception/input adapters, tool registry, permission policy, approval queue, audit trail, memory/state, scheduler, diagnostics, recovery, and verification. This follows the 2026 agent-harness direction from Aakash Gupta's article, ["2025 was Agents. 2026 is Agent Harnesses. Here's why that changes everything."](https://aakashgupta.medium.com/2025-was-agents-2026-is-agent-harnesses-heres-why-that-changes-everything-073e9877655e)

For Jarvis, that means new features should be judged by whether they strengthen the harness:

- Can Jarvis understand the request and keep enough state to continue later?
- Can Jarvis choose tools through an inspectable registry instead of hidden behavior?
- Can risky actions stop at approvals with exact planned arguments and a last-look packet?
- Can every run leave an audit trail, verification result, and recovery path?
- Can failures degrade to safe chat, preview, or plan mode instead of silently acting?

Use `priority goal` inside Jarvis to keep the top build goal visible: finish Jarvis V2 as an AI agent harness toward a fully functional AGI-like personal assistant. Use `harness status` inside Jarvis or `/api/harness-status` in the dashboard to inspect these harness components, current evidence, and AGI-direction gates. Use `harness cycle: <order>` or `/api/harness-cycle?request=...` to preview the perceive -> route -> gate -> act -> verify -> learn loop before Jarvis executes anything. Use `agi gates` or `/api/agi-gates` to see measurable gates between current Jarvis and stronger future autonomy.
- `jarvis_v2/tools/continuity.py` — recent activity catch-up digest
- `jarvis_v2/tools/organize.py` — deterministic brain-dump organizer for tasks, goals, memories, and decisions
- `jarvis_v2/tools/approvals.py` — blocked-action approval queue
- `jarvis_v2/tools/state.py` — Current Context state snapshots
- `jarvis_v2/tools/utilities.py` — calculator, unit conversion, passwords
- `jarvis_v2/tools/skills.py` — Hermes-style Markdown skill storage
- `jarvis_v2/tools/goals.py` — durable goals/projects and steps
- `jarvis_v2/tools/tasks.py` — lightweight task capture and completion
- `jarvis_v2/tools/decisions.py` — durable decision log with rationale and impact
- `jarvis_v2/tools/people.py` — people profiles and interaction history
- `jarvis_v2/tools/preferences.py` — structured preference capture and chat context
- `jarvis_v2/tools/conversation.py` — session search, export, and reflection summaries
- `jarvis_v2/agent/model_planner.py` — Ollama-backed planner wrapper over deterministic rules
- `jarvis_v2/agent/subagent_runner.py` — custody-verified durable pure-worker runtime with leases and idempotent result publication
- `jarvis_v2/tools/memory_curator.py` — weak-memory review, label-only structured-knowledge custody review, delete, and Memory Tree snapshots
- `jarvis_v2/tools/computer.py` — enable-gated computer-control skeleton
- `jarvis_v2/tools/browser.py` — URL opening, page fetch, link extraction, web search
- `jarvis_v2/tools/personal.py` — Obsidian and Reminders integration footholds
- `jarvis_v2/automations/jobs.py` — daily plan, daily brief, goal nudge, and weekly review builders
- `jarvis_v2/automations/scheduler.py` — SQLite-backed scheduled jobs
- `jarvis_v2/scripts/run_scheduler.py` — simple due-job loop
- `jarvis_v2/scripts/run_subagent_runner.py` — owner-only local subagent worker service
- `jarvis_v2/ui/status_server.py` — local status dashboard, JSON endpoint, chat-health/safety/context previews, and approval-gated web chat endpoint

Supported starter actions:

- remember a fact
- ingest `Jarvis/Inbox.md` notes into memory with duplicate protection
- search and read Markdown notes inside the Jarvis Obsidian folder
- write a metadata-only recent-file digest from watched folders
- search memory
- preview/save chat context for the conversational brain
- rehearse a full assistant turn to preview chat-vs-tool routing, visible context, and approval gates without calling the chat model or executing tools
- show a chat-safety report for grounded memory answers and approval boundaries
- record Jarvis feedback and review safe improvement themes
- run a safe learning review across feedback, memory health, preferences, skills, and category-labeled generic memories that may deserve future structuring; labels remain hints, not facts, and the review never promotes them
- show recent memories
- write an Obsidian daily note
- get current time
- list, find, read, and permission-gated write text files
- run local shell/code commands with explicit approval
- read system info and visible apps
- open apps
- read or set volume
- read clipboard with approval, and set clipboard locally
- speak text aloud and list available macOS voices
- preview spoken-response packets without speaking or recording audio
- rehearse spoken assistant turns, including chat-vs-tool routing and approval gates, without speaking or executing tools
- check voice/ASR setup readiness without requesting microphone access or recording audio
- plan safe user-supplied audio-file transcription without reading audio or saving transcripts
- plan safe voice input, ASR, wake-word, transcript, and microphone privacy boundaries without recording audio
- review supplied voice transcripts and preview routing/approval gates before executing anything
- build voice confirmation packets with exact next commands and approval expectations before acting on speech
- show a full read-only voice command lifecycle from setup through transcript preview, confirmation, rehearsal, approvals, and audit
- calculate math
- convert units
- generate secure passwords
- list registered tools and risk levels
- show Jarvis as an AI agent harness with current components, evidence, blockers, and AGI-direction gates
- show the six-layer assistant architecture map with current implementation status
- show a safe phased roadmap for building Jarvis toward fuller assistant autonomy
- show model-routing readiness for chat/planner models without running a model request
- preview the model-planner prompt, available tool summary, and safety boundaries without calling a model
- show a user-friendly capability map grouped by assistant area and risk level, including everyday category focuses such as `what can Jarvis do info`, `what can Jarvis do productivity`, `what can Jarvis do utilities`, `what can Jarvis do markets`, `what can Jarvis do research`, `what can Jarvis do writing`, and `what can Jarvis do fun`
- show compact Jarvis brain/system status
- show a read-only brain loop report across conversation, memory, skills, planning, execution boundaries, and verification
- show live safety status with risk-gated tools and pending approvals
- show privacy boundaries for local memory, Obsidian, clipboard, computer control, shell/code, browser, and future personal integrations
- show a readiness report for setup, memory, schedules, approvals, and risk gates
- draft a safe autonomy plan with approval-gated steps and verification checkpoints
- run `action readiness: ...` for a read-only go/no-go packet before a proposed action
- run `risk preflight: ...` to classify likely computer-control, shell/code, file, personal-data, and external-side-effect risks before Jarvis acts
- run `second loop: ...` to preview the task/action loop Jarvis would use after chat becomes work
- rehearse a request to preview planner routing, tool risks, and approval requirements without executing tools
- show read-only status dashboard launch info
- status dashboard includes web chat, conversation history, action rehearsal, autonomy planning, computer-control planning, focus briefs, work-session packets, next-session plans, voice safety, capability map, model routing, build progress, build delta, work queue, next action packet, resume briefs, chat response health, chat safety, chat brain context, brain loop, learning review, privacy boundaries, readiness report, safety status, morning startup, weekly review context, architecture layers, and the safety spine
- inspect recent tool-run audit records
- show a build-progress report with recent tool activity, safety posture, blockers, and safe next build moves
- show a build-delta report with the latest checkpoint window, tool activity, conversation delta, safety delta, and safe next checks
- save build-progress reports into Obsidian Reflections
- save approval-review notes into Obsidian Automations
- get a return brief with readiness, recent activity, blockers, and safe next actions
- save return briefs into Obsidian Reflections
- get a handoff brief with readiness, activity, build progress, blockers, safety reminders, and safe next actions
- save handoff briefs into Obsidian Reflections
- get and save a session closeout with activity, build progress, work queue, approval review, and safety reminders
- build a read-only focus brief for a work session from tasks, goals, approvals, decisions, preferences, and schedules
- prepare a read-only work-session packet with a start move, preflight checklist, stop condition, and end verification
- catch up on recent Jarvis activity, blockers, tasks, goals, and memory
- suggest safe next actions from current tasks, goals, approvals, and schedules
- choose one next action packet with rationale, risk, and verification before acting
- show and save a safe work queue ordered by approvals, tasks, goal steps, and scheduled upkeep
- write `Jarvis/Automations/Mission Control.md` with safe next actions and blockers
- organize prefixed brain dumps into tasks, goals, memories, and decisions
- list, inspect, and dismiss pending approval requests
- review pending approval risks before rerunning blocked requests, with extra computer-control preflight checks for screen/control approvals
- export a consolidated Current Context state snapshot to Obsidian
- save, search, list, read, and permission-gated delete reusable Markdown skills
- model-backed tool planning for uncaught actionable requests
- automatic skill context in normal chat
- safer offline chat fallback for capability, safety, memory, autonomy, and next-step questions
- weak-memory inspection and Memory Tree snapshots
- computer-control status, enable/disable, screenshot, mouse/keyboard primitives
- read-only computer-control readiness checks and task plans before any screen observation or control
- browser page fetch, link extraction, URL opening, and search-page extraction
- browser page history stored in SQLite
- search conversations, show recent chat, list sessions, export sessions, inspect chat continuity, chat response health, and chat safety, and write session reflections to Obsidian
- draft reviewable skills from recent sessions
- create/list/inspect goals, add/complete steps, update status, and export project notes to Obsidian
- capture/list/complete/export lightweight tasks in Obsidian
- record/list/inspect/update durable decisions in SQLite, memory, and Obsidian
- save/list/inspect people profiles and log interactions to Obsidian and memory
- save/list/inspect/retire structured preferences and inject active ones into chat
- list next actions across active goals
- summarize fetched pages and save page summaries to Obsidian Sources
- personal integration boundary status, safe connector migration plans, open Jarvis vault, approval-gated reminders
- personal integration boundary contracts for calendar/email/messages/contacts before connector code is added
- read-only personal connector action previews such as `integration action preview: email -> send draft reply`
- read-only personal connector scope packets such as `integration scope packet: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only`
- read-only morning startup brief and saved startup notes for beginning the day safely
- tactical daily plan from tasks, goals, pending approvals, and scheduled jobs
- proactive daily brief written to Obsidian with memories, skills, goals, tasks, approvals, decisions, people, and preferences
- stale-goal nudges and weekly reviews written to Obsidian
- read-only weekly review model-context and prompt-preview packets for safer model-assisted reflection
- schedule/list/run due jobs, including daily briefs, Inbox ingest, file digests, goal nudges, weekly reviews, State Snapshot, Conversation Compaction, and Mission Control refreshes
- schedule default assistant automations with `schedule assistant basics`, resume a paused State Snapshot with `resume job State Snapshot`, or resume paused Memory Trees compaction with `resume job Conversation Compaction`
- pause/resume/delete/run-now scheduled jobs
- manual `run due jobs`, `run job ... now`, Morning Brief scheduling, and job resume requests require explicit approval because the selected job may deliver externally; the scheduler's normal execution of an already authorized schedule is unchanged
- duplicate-memory review and approval-gated merge
- setup/dependency check for optional runtime pieces

Computer-control note:

- mouse, keyboard, screenshots, and full observe-act-verify require `pyautogui`
- if it is missing, Jarvis reports the missing dependency instead of pretending computer control worked

Next major step: replace the deterministic planner with a model-backed planner while keeping the same runtime boundaries.

Permission behavior:

- read-only and local-safe tools run automatically
- high-risk tools, including text-file writes, are blocked unless runtime receives `approved=True`
- blocked high-risk requests are saved into a pending-approval queue
- each blocked request returns a safety receipt with its approval ID
- pending approval records preserve parsed planned arguments, such as exact commands, coordinates, typed text, and screen expectations
- the queue is mirrored to `Jarvis/Automations/Pending Approvals.md` in Obsidian
- use `pending approvals` to list blocked requests, `approval detail #ID` to inspect one, `approval packet #ID` to mark the required last-look preview, `approval resume packet #ID` to preview the one-shot resume contract, `approve approval #ID` to rerun one, and `dismiss approval #ID` to clear one without running it
- use `approval packet #ID` for a read-only last look at what approving one request would rerun; approval is rejected until this packet has been viewed
- future external actions like messages, calls, purchases, and destructive shell commands should stay confirmation-gated

## Full-Assistant Tracks

The V2 foundation now has a working slice in each major assistant/AGI direction:

- Brain: deterministic planner plus optional Ollama-backed model planner
- Architecture map: `architecture map` shows perception, NLP, planning, memory, tools, learning, safety, and gaps
- Brain loop: `brain loop` shows Jarvis' current perception -> memory/skills -> planning -> execution boundary -> verification state without acting
- Roadmap: `roadmap` and `what should we build next` rank the next assistant upgrades without bypassing safety gates
- Model routing: `model routing status` checks chat/planner model readiness while keeping execution behind ToolRegistry and PermissionPolicy
- Model planner prompt preview: `model planner prompt preview: find files README in .` shows the model-planner prompt packet and approval boundaries without calling Ollama
- Capability map: `capability map` and `what can Jarvis do?` show abilities, safety gates, starter commands, and live-proof caveats; category focuses like `what can Jarvis do info`, `what can Jarvis do productivity`, `what can Jarvis do utilities`, `what can Jarvis do markets`, `what can Jarvis do research`, `what can Jarvis do writing`, and `what can Jarvis do fun` list matching tools and read-only boundaries. Telegram, Instagram, iMessage, and KakaoTalk have recipient-confirmed V3 evidence; KakaoTalk required operator chat verification after an outcome-unknown GUI attempt. Calls are disabled for the August 15 preview; any later re-enable and live proof requires a separate, operator-present approval gate. Examples remain approval-gated capability routes.
- Conversation: ChatGPT/Claude-like chat path with memory and skill context, plus `chat continuity brief`, `chat response health`, `chat loop preview: ...`, `assistant turn rehearsal: ...`, `chat safety`, `chat context: ...`, and `save chat context: ...` to inspect what the brain can use before answering or acting
- Offline fallback: grounded conversational answers still explain capabilities, safety, memory, autonomy, and next steps when Ollama is unavailable
- Memory: SQLite index plus Obsidian Memory Tree, weak-memory review, and delete-by-id
- Feedback learning: `feedback: ...` records reviewable user feedback, `feedback report` summarizes themes, `save feedback report` writes the audit trail, `feedback actions` suggests reviewable follow-ups, `failure to test: ...` turns a miss into a read-only regression preview, `failure clusters` groups repeated misses into test/preference/skill candidates, `failure promotion packet` drafts a reviewed path from a cluster into a real test or learning artifact, and `save feedback actions` writes the improvement queue to Obsidian
- Learning review: `learning review` checks feedback, weak/duplicate memories, preferences, skills, and unowned generic memories whose category labels suggest profile, preference, decision, person, or goal review. These are label-only inspection candidates, not semantic proof. `knowledge promotion packet <id>` privately reviews one exact candidate and supplies a content-free opaque review token. Reviewed decision candidates can use `promote memory <id> revision <revision> token <review-token> to decision: <title> | <rationale> | <impact>`. Reviewed preference candidates can use `promote memory <id> revision <revision> token <review-token> to preference: <category> | <key> | <value>` to create a new canonical preference; an existing normalized category/key is refused rather than overwritten or adopted. Reviewed profile candidates can use `promote memory <id> revision <revision> token <review-token> to profile: <heading> | <category> | <body>` to transfer the exact memory ID into an owned Profile.md note. All three transfers are `HIGH_RISK`, bind approval to the reviewed full row, store identity, and exact projection targets, preserve the original memory ID as the structured record's sole canonical memory, and never infer fields from the category label. Profile success additionally requires source-owned profile custody and verified Profile.md publication; person and goal transfer remain review-only. Promotion tokens remain private on public approval/history/trace surfaces, and active HTML/CSS resources plus local path-shaped fields fail before approval. Later ordinary preference updates retain the promoted memory's canonical custody without reopening already-completed no-op projection work. `save learning review` publishes the report to Obsidian and maintains a local lock file for safe publication, while `queue learning tasks` handles established feedback/health signals and excludes label-only candidates.
- Startup projection recovery audit: writable startup now opens one durable, content-free receipt before initializing the Jarvis vault or repairing the bounded person, decision, preference, generic-memory, and skill note projections. A heartbeat protects slow active repairs, every component records counts/status only, malformed or incompatible evidence fails closed, and unreceipted repair stops startup. `startup recovery report` (also `did startup recovery run?`) reads the bounded receipt history without exposing session IDs, paths, content, timestamps, exception messages, or execution authority. Storage-degraded read-only startup skips both projection repair and receipt writes.
- Auto-fetch: Obsidian Inbox ingest atomically links each source to one durable memory and one evidence-backed canonical note projection; missing notes repair without duplicate memory, external edits fail closed, scheduled finalization stays lease-fenced, and partial projection outcomes are reported as pending rather than success. Watched-folder digests remain metadata-only.
- Obsidian access: direct search/read over Jarvis-owned notes
- Skills: Hermes-style Markdown procedures, searchable and injected into chat
- Skill match preview: `skill match preview: import notes into memory` shows which saved procedures apply before Jarvis acts
- Linked skill extraction: `extract linked skills from https://github.com/NousResearch/hermes-agent https://github.com/tinyhumansai/openhuman` previews reusable skill candidates from linked agent docs without fetching pages or saving anything
- Linked skill installation: `install linked skills from hermes-agent openhuman` saves curated Hermes/OpenHuman-inspired skills locally into SQLite and the Obsidian skill vault
- Skill learning: draft reusable skills from session history for human review
- Computer use: enable-gated mouse, keyboard, screenshot, and screen state primitives
- Computer readiness: `computer readiness: ...` checks the preflight boundary before Jarvis observes or acts
- Computer planning: `computer task plan: ...` drafts read-only observe-act-verify steps, stop conditions, and approval checkpoints
- Computer action packet: `computer action packet: click x 100 y 200 expectation settings opens` prepares one approval-ready desktop action without observing or acting
- Computer vision prompt preview: `screen vision prompt preview: expectation settings opens; observation settings window is visible; source approved_screenshot` packages approved screen-observation evidence for a future vision verifier without observing, reading image files, calling a model, or controlling the computer
- Computer vision model review: `screen vision model review: expectation settings opens; observation settings window is visible; source approved_screenshot; observation id obs-7; screenshot path /tmp/jarvis_oav_before.png; captured at 2026-06-09T10:00:00+09:00; redaction reviewed safe; approval 7; consent=true` is PERSONAL_DATA and queues approval before a configured local vision reviewer may analyze the supplied screenshot artifact
- Gated desktop action: `observe act verify action click x 100 y 200 expectation settings opens` routes to the approval queue before any screenshot, click, typing, or mouse movement
- Voice output: local macOS text-to-speech for spoken responses
- Voice setup check: `voice setup check` inventories ASR/capture dependencies and the optional `JARVIS_VOICE_WARMUP=1` silent-WAV warm start without requesting microphone access, recording audio, or starting a listener
- Voice reply preview: `voice reply preview: Jarvis is ready` prepares a spoken response packet without speaking, recording audio, writing memory, or queuing approvals
- Spoken turn rehearsal: `spoken turn rehearsal: run command python3 --version` previews whether a spoken-style turn becomes chat, a low-risk action, or an approval-gated action before speaking or executing tools
- Voice audio-file planning: `voice file transcription plan: /path/to/audio.m4a` checks file metadata, ASR readiness, privacy boundaries, local transcriber configuration, and future transcriber guardrails without opening audio or saving transcripts
- Voice audio-file transcription preview: `voice audio file transcribe: /path/to/audio.m4a consent=true receipt_id=voice-file-...` is PERSONAL_DATA and queues approval before any supplied audio file can be opened or transcribed into a temporary preview; configure `JARVIS_VOICE_WHISPER_MODEL_PATH` or `JARVIS_VOICE_FASTER_WHISPER_MODEL_PATH` to a local model path to make the approved runner available without downloads
- Voice input planning: `voice input plan push-to-talk` defines microphone, ASR, transcript, wake-word, and approval boundaries before any listener exists
- Voice transcript review: `voice transcript review: ...` previews recognition, risk cues, planned tools, and approval requirements before acting on spoken text
- Voice confirmation: `voice confirmation: ...` packages a supplied transcript with risk cues, exact next command, and approval expectations without executing or saving anything
- Voice command lifecycle: `voice command lifecycle: ...` shows the setup -> capture boundary -> transcript preview -> confirmation -> rehearsal -> approval -> audit path without requesting microphone access, recording audio, executing tools, or queuing approvals
- Code control: approval-gated local command execution without shell expansion
- Personal integrations: Obsidian status/opening, connector boundary contracts, and confirmation-gated Reminders foothold
- Proactive work: tactical daily plan and context-rich daily brief generators that write into Obsidian
- Morning startup: `morning startup` and `save morning startup` show blockers, first tasks, goal momentum, preferences, scheduled rhythm, and the safest first move
- Review loop: stale-goal nudges, `weekly review context`, `weekly review prompt preview`, and weekly reviews for continuity
- Goals/projects: durable objectives with steps and Obsidian project notes
- Tasks: lightweight open task queue exported to Obsidian
- Decisions: durable choices with rationale, impact, status, and memory indexing
- People memory: profiles, relationship context, and interaction history in Obsidian
- Preferences: structured active preferences injected into conversation context
- Next-action focus: daily/weekly reviews and `next actions` point at open goal steps
- Safe next-action advisor: `safe next actions` and `what should Jarvis do next?` suggest low-risk progress before autonomy
- Next action packet: `next action packet` chooses one proposed move with rationale, risk, and verification without executing it
- Work queue: `work queue` and `save work queue` order approvals, tasks, goal steps, and upkeep into a safe execution queue
- Next-session plan: `next session plan: continue Jarvis safely` gives Jarvis a read-only resume order without completing tasks, writing notes, or approving anything
- Session closeout: `session closeout` and `save session closeout` record the end-of-session state, blockers, approval review, and resume queue
- Mission Control: `mission control` refreshes `Jarvis/Automations/Mission Control.md` with safe next actions and blockers; the State Snapshot scheduled job refreshes it too
- Reflection: session summaries written to Obsidian Reflections
- Status UI: local dashboard for memory, goals, jobs, sessions, tools, web chat, conversation history, command diagnosis, execution-governor preview, action rehearsal, autonomy planning, computer-control planning, focus briefs, work-session packets, next-session plans, voice safety, capability map, model routing, chat response health, chat safety, chat brain context, brain loop, learning review, privacy boundaries, readiness report, safety status, morning startup, weekly review context, work queue, next action packet, resume briefs, approval review, build progress, build delta, readiness, and safety
- Audit trail: SQLite tool-run log with risk, approval, status, and output snippets
- Approval queue: blocked high-risk requests remain visible until dismissed
- Approval review: `approval review` explains why pending requests are gated and adds computer-control readiness/task-plan checklist items before screen/control approvals
- Approval detail: `approval detail #ID` shows one queued request, blocked output, safer checks, and exact approve/dismiss commands
- Approval execution packet: `approval packet #ID` previews the exact rerun, planned args, and last-look checks before approving; approval commands require this packet first
- Approval resume packet: `approval resume packet #ID` previews the specific one-shot resume contract, proof commands, and next safe command without approving, rerunning, or queuing anything
- Approval-review export: `save approval review` writes the reviewed pending-approval risk summary into Obsidian Automations
- Safety status: `safety status` reports current guardrails, risky tools, and pending approvals
- Privacy report: `privacy report` explains what Jarvis may use by default, what needs approval, which narrow V2 personal connectors are active, and which broader legacy scopes remain blocked
- Action readiness: `action readiness: run a script and email me the result` decides whether a proposed action is ready, needs preflight, or should stop for approval review without executing tools
- Setup diagnostics: `setup check` and `jarvis doctor` check dependency, Google connector package, env-presence including dashboard host/port overrides and hidden invalid host/port fallback, app-presence, credential-file-presence, and storage-fallback readiness without reading clipboard contents, secret values, credential/token file contents, or calling Google
- Personal integrations: `integration status` shows active local integrations, gated V2 personal connectors, and broader legacy account scopes that remain blocked
- Integration contracts: `integration contract: email` defines read-only, personal-data, side-effect, caching, audit, and hard-stop rules before connector work
- Integration scope packets: `integration scope packet: email -> search mailbox metadata; target inbox; time last 7 days; data metadata-only` narrows target, time range, data level, risk, and stop conditions before connector work
- Legacy connector audit: `legacy connector migration audit` reviews old Jarvis browser/calendar/email migration readiness as a group without account access, personal-data reads, side effects, route unlocks, or approval grants
- Integration migration plans: `integration migration plan: email` creates a safe per-connector checklist before adding personal-data or side-effect tools
- Readiness report: `readiness report` checks setup, memory, schedules, Mission Control, pending approvals, and risk gates
- Autonomy planning: `autonomy plan: ...` separates safe prep, approval-gated actions, and verification before Jarvis acts
- Risk preflight: `risk preflight: ...` names likely risky areas and safer preview commands without executing tools or queuing approvals
- Command diagnosis: `command diagnosis: run command python3 --version`, `/api/command-diagnosis?request=...`, and the root launcher's `--diagnose` mode classify chat, auto-safe tools, approval-gated work, and hold states without executing tools or queuing approvals
- One-shot terminal command: `JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "what time is it"` sends one message through the normal Jarvis runtime; risky actions still queue approval receipts instead of running
- Second loop: `second loop: ...` previews the plan -> preflight -> rehearse -> gated execution -> verify -> closeout loop without calling models or running tools
- Action rehearsal: `rehearse: run command python3 --version` previews planned tools, risk levels, and approvals without running anything
- State snapshots: Current Context is periodically exported into the Memory Tree
- Continuity digest: `catch me up` summarizes recent activity and blockers
- Build progress: `build progress` and `what changed in Jarvis` summarize recent tool activity, safety posture, blockers, and safe next build moves
- Build delta: `build delta` and `what changed since checkpoint` summarize the latest checkpoint window without running tools or writing notes
- Build-delta export: `save build delta` writes the latest checkpoint window into Obsidian Reflections
- Build-progress export: `save build progress` writes the current build-progress report into Obsidian Reflections
- Work-block checkpoint: `work block checkpoint: continue Jarvis V2 safely` creates a resumable packet with progress evidence, verification evidence, blockers, and the next safe command
- Work-block checkpoint export: `save work block checkpoint: continue Jarvis V2 safely` writes the resumable packet into Obsidian Reflections
- Checkpoint recovery: `checkpoint recovery: continue Jarvis V2 safely` previews how to resume from the latest saved work-block checkpoint and verify the next safe step
- Checkpoint recovery apply packet: `checkpoint recovery apply: continue Jarvis V2 safely` prepares the approval-gated sequence before applying one checkpointed task
- Checkpoint recovery receipt: `checkpoint recovery receipt: continue Jarvis V2 safely` saves after-action evidence for one reviewed recovery step without granting future approval
- Priority goal: `priority goal` shows the top Jarvis build goal and the guardrail that safety boundaries still override autonomy.
- Harness completion: `harness completion` estimates Jarvis V2 agent-harness prototype completion, strengths, and remaining real-execution gaps without claiming AGI completion
- Saved-note audit: `recent saved notes` lists recent Jarvis Obsidian notes by metadata without opening note contents
- Build target packet: `build target packet: continue Jarvis V2 safely` selects one scoped implementation target with likely files, tests, boundaries, and stop conditions
- Build target export: `save build target packet: continue Jarvis V2 safely` writes that target into `Jarvis/Automations/Build Target Packet.md`
- Return brief: `return brief`, `I'm back`, and `what did I miss` combine readiness, activity, and safe next actions
- Return-brief export: `save return brief` writes the current return brief into Obsidian Reflections
- Handoff brief: `handoff brief` combines readiness, activity, build progress, blockers, safety reminders, and safe next actions into one resume point
- Handoff export: `save handoff brief` writes the current handoff brief into Obsidian Reflections
- Session closeout: `save session closeout` writes the end-of-session activity, build progress, work queue, and approval review into Obsidian Reflections
- Focus brief: `focus brief` and `start work session` create a safe session plan from current context without taking risky actions
- Work-session packet: `work session packet` prepares a safe start move, preflight checks, stop condition, and end verification without acting
- Brain-dump organizer: `organize brain dump: task: ...; remember: ...; goal: ...; decision: ...`

This is still a foundation, not full autonomy. The safe next upgrades are:

- migrate old Jarvis browser/calendar/email tools into V2 risk levels using connector boundary contracts first
- prove the approval-gated local OAV vision reviewer against a real local screenshot model command before using it in live observe-act-verify loops
- prove the approval-gated audio-file transcription runner against a real local Whisper model path before any live microphone or wake-word listener
