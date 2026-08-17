# Jarvis V4 quick start

Use these commands in the macOS Terminal, from the Jarvis V4 project checkout.

The tested runtime intentionally retains V3 compatibility names: `launch_jarvis_v3*`,
`JARVIS_V3_*`, and `~/.jarvis_v3`. Do not rename them during setup; V4 is the public release
identity, not a new runtime-state migration.

## First-time local setup

Jarvis V4 requires Python 3.11 or newer. For a fresh checkout, create one project-local Python
environment so every launcher and setup helper uses the same dependencies. Installing the
requirements uses the network and disk, so run these commands yourself only when intended:

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

`requirements.txt` is the deterministic, alphabetized direct-dependency declaration. Ollama chat
uses Jarvis's bounded standard-library loopback HTTP adapter, so the Ollama, HTTPX, and Pydantic
client closure is not part of the direct runtime declarations.
Validate its syntax, direct imports, and Python 3.11 floor without installing or importing packages:

```bash
python3 -B -m jarvis_v2.scripts.requirements_contract
```

A passing report means the direct declarations are valid; it does not make installation
reproducible. The declarations intentionally contain no guessed version constraints, so dependency
resolution can change over time. Do not guess package versions. Before claiming a reproducible
release, generate and review a separate lock or constraints artifact in a clean supported
environment, test installation from that exact artifact, and explicitly include it in the public
candidate policy.

If `python3 --version` is older than 3.11, install a supported Python first and use that interpreter
to create `.venv`. An absolute `JARVIS_V3_PYTHON` recorded in the selected owner-only
`runtime.env` is authoritative. A shell value is used only when no V3 environment file is selected;
when both exist they must match exactly. Otherwise the launchers inspect the project `.venv`, the
current interpreter, and common Apple Silicon or Intel Homebrew locations; they prefer a compatible
interpreter with the selected model-provider package and fall back to the first core-compatible
interpreter. They stop with a bounded setup message instead of importing Jarvis under an
incompatible Python.

An explicit `JARVIS_V3_PYTHON` may be a normal Homebrew or project-venv symlink. Its spelling and
resolved target must stay outside V2 custody, and the final target must be a root- or owner-owned
regular executable that is not group/world writable—the same target policy used by automatic
launcher discovery. This is an honest-operator setup check, not protection from another same-UID
process racing a pathname after validation.

Create the private V3 environment directory, then let Jarvis create or validate an owner-only
environment file and dashboard password. The password is never printed. This uses the isolated V3
defaults under `~/.jarvis_v3`; it does not read or reuse V2 state.

```bash
mkdir -p "$HOME/.jarvis_v3"
chmod 700 "$HOME/.jarvis_v3"
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" .venv/bin/python3 -m jarvis_v2.scripts.setup_status_auth
```

The setup is safe to run again. It refuses symlinks, duplicate or invalid existing password
entries, files owned by another user, and unsafe file types.

When `JARVIS_V3_ENV` selects this file, every V3 launcher fails closed if the shell already exports
a state path, integration credential, owner identifier, or local executable/model path with a
different value. Unset the conflicting variable or put the exact intended value in this V3 file.
Unrelated shell settings and exact matches are preserved.

### Launcher path recovery

If a launcher stops with exit 78 and names one setting, review only that named setting. Jarvis does
not display its path value. Correct the corresponding line in the selected V3 environment, or
remove that setting if it is optional and you intend to use the isolated V3 default. If the setting
was exported only in the current Terminal, unset that exact named variable instead. Do not change
or disclose unrelated credentials. Then rerun the documented `setup check`; it cannot start until
the named path-custody problem is corrected.

## First launch check

At the normal macOS Terminal prompt—not in Telegram and not at Jarvis's `You:` prompt—run the
deterministic setup check before the first real request:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "setup check"
```

This reports local dependency and integration readiness without approving actions, sending
messages, starting services, or changing schedules. Correct only the bounded item it reports and
run the same check again. Its next steps are complete Terminal commands for the Jarvis V3 project
folder; do not paste them at Jarvis's `You:` prompt. Run
`JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "jarvis doctor"` only when the
setup check asks for deeper diagnostics.

## One-shot request

Use this for one request that does not need a same-session approval follow-up:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3.py "what time is it?"
```

## Typed conversation

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_chat.py
```

At the `You:` prompt, enter only requests for Jarvis. Press Control-C to leave Jarvis and return to
the Terminal shell.

## Push-to-talk with spoken replies

Before the first recording, run the offline readiness check:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py --check
```

This checks `ffmpeg`, the selected local Whisper transcriber, bounded voice settings, and whether
spoken replies are available. It does not list or open microphones, request microphone permission,
record or transcribe audio, start the Jarvis runtime, or read private data. Dependencies are
optional and may not be installed on a fresh Mac. Follow only the bounded recovery shown by the
check, then rerun it until it reports `READY`.

When the check reports `READY`, start push-to-talk yourself:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_voice.py
```

Press Enter to start recording and Enter again to stop. Use `--language ko` for a Korean session,
or `--no-speak` when replies should be printed only. Starting the real voice session—not the
readiness check—may cause macOS to request Microphone access for your terminal application. Review
that prompt yourself; Jarvis does not grant the permission.

## Google Calendar authorization

Calendar authorization is a human-only Terminal flow. For bounded Calendar reads, run:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_calendar_auth.py readonly
```

The separate `full-access` mode is only for approval-gated Calendar mutations after you deliberately
review the broader Google consent request. Neither mode is an ordinary Jarvis command, and neither
may be run detached, in a background job, in Telegram, or at Jarvis's `You:` prompt.

## Authenticated local dashboard

Start the loopback-only dashboard:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" ./launch_jarvis_v3_dashboard.py
```

This root launcher runs only an authenticated loopback dashboard in the current foreground
Terminal process. It does not install or enable a LaunchAgent, background daemon, scheduler, or
messaging listener; pressing Control-C closes the server. Run this exact root wrapper interactively:
the dashboard refuses imported, copied, detached, redirected-input, LaunchAgent, and background-job
starts before it reads the V3 environment or constructs runtime state.

Open `http://127.0.0.1:8766` in your browser and sign in with username `jarvis`. To copy the
locally stored password without printing it or putting the password itself in command history, run:

```bash
JARVIS_V3_ENV="$HOME/.jarvis_v3/runtime.env" .venv/bin/python3 -m jarvis_v2.scripts.copy_status_auth
```

The helper uses the same environment syntax and owner-only custody checks as Jarvis. It refuses
duplicate password entries, symlinks, non-files, unsafe permissions, invalid values, and clipboard
failures without displaying the password or passing it in a process argument.

Paste it only into the browser password field. The clipboard is visible to local applications, so
clear it immediately after sign-in:

```bash
pbcopy </dev/null
```

Never paste the dashboard password into Jarvis, chat, source files, logs, or GitHub. Press
Control-C in the dashboard Terminal when finished.

## Approval flow

Keep one typed-chat session open for the entire approval flow. After a risky request, Jarvis shows
the readiness and approval packets and then asks:

```text
Approve approval #ID after this packet? [y/N]
```

Type `y` only after the stored target, arguments, and expected outcome are exact. Press Enter or
type `n` to leave it pending without execution. For an approval that was already pending, replace
`ID` below with the number Jarvis showed and enter these at the same Jarvis `You:` prompt:

```text
approval readiness ID
approval packet ID
approve approval ID
```

After the approved rerun, the chat automatically prints a privacy-safe approval-chain summary and
verification receipt. A missing or unknown proof is not success and must not be retried
automatically.

## Current boundary

V3 is a supervised functional preview. Private reads, messages, calls, computer control, service
activation, and production cutover still require their documented review or live-proof gates. V2
remains the rollback generation until a separate cutover is approved.
