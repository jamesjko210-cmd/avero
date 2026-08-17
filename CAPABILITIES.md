# Jarvis V4 — Capabilities

> V4 capability inventory promoted from the frozen and tested V3 implementation. A listed integration is
> not a live-proof claim; current readiness, approval state, and recipient-visible evidence remain
> authoritative.

Jarvis is one local personal-agent harness with many visible capabilities. It
routes what you type or say automatically, runs read-only and local-safe work
on its own, and **stops for your ✅ approval before anything risky** (real
messages, calls, emails, calendar writes).

This page mirrors the live **Capability Cockpit** on the dashboard and the
`capability_cockpit` tool — the cockpit is the source of truth (it shows each
lane's real health, last success/failure, and smoke coverage). Type `cockpit`
in Telegram or open the dashboard to see the live version.

Inherited V3 evidence note: Telegram, Instagram, iMessage, and KakaoTalk sends have recipient-confirmed evidence.
That evidence is carried into this unchanged V4 release. KakaoTalk's GUI execution remained
outcome-unknown until the operator verified the exact chat and the recipient
confirmed arrival. Every new send still requires a fresh approval. Calls are explicitly disabled
for the August 15 functional preview; any later re-enable and live proof requires a separate,
operator-present approval gate.

## How safety works
- Every tool has a risk level: `READ_ONLY`, `LOCAL_SAFE`, `PERSONAL_DATA`, `EXTERNAL_SIDE_EFFECT`, `HIGH_RISK`.
- Read-only and local-safe tools run automatically.
- Approval-gated lanes queue a pending approval with ✅/❌ buttons (dashboard or Telegram). Nothing external happens until you approve.
- A queued approval is the system working, not a failure — the cockpit shows it as "awaiting approval".

## Capability lanes

| Lane | Risk | Approval | Try saying |
|------|------|----------|------------|
| **Messaging (KR/EN)** — Telegram, KakaoTalk, Instagram, iMessage | HIGH_RISK | ✅ required | `send my test contact a telegram saying hello` |
| **Calls** — contact / Telegram / Kakao / Instagram | HIGH_RISK | disabled for preview; ✅ required after re-enable | `call my test contact on telegram` |
| **Morning Brief** — weather, calendar, reminders, news | EXTERNAL_SIDE_EFFECT | ✅ required to schedule, resume, or run now | `run job morning brief now` |
| **Contacts & People** — lookup, relations, interactions | LOCAL_SAFE | auto | `who is my test contact` |
| **Calendar reads** — availability and calendar listing | LOCAL_SAFE or PERSONAL_DATA by route | auto only for bounded local-safe reads; otherwise ✅ required | `am I free tomorrow?` |
| **Email** — headers, search/body reads, send | LOCAL_SAFE, PERSONAL_DATA, or HIGH_RISK by operation | private reads and sends require ✅ | `read my recent emails` |
| **Markets & Weather** — crypto/stocks, FX, weather | LOCAL_SAFE | auto | `markets overview` |
| **Memory & Notes** — remember facts, memory stats | LOCAL_SAFE | auto | `remember that I prefer direct answers` |
| **Voice** — voice command cockpit, voices | READ_ONLY | auto | `voice command cockpit` |
| **Diagnostics** — readiness, execution health, channel health, doctor | READ_ONLY | auto | `what broke` |
| **Scheduled Jobs** — list, pause, resume, or run jobs | EXTERNAL_SIDE_EFFECT | auto to list/pause; ✅ required to resume/run | `list scheduled jobs` |

Normal due-job execution continues under the schedule that was already authorized. The approval
boundary above applies to manual run-now/batch-run requests and to creating or re-enabling a
Morning Brief delivery schedule.

## Phone control (Telegram, owner-locked)
Send natural commands, a 🎙 voice note (transcribed on-device), or a 📷 photo
(OCR'd on-device). Read-only status shortcuts:

- `what broke` — recent failures at a glance
- `cockpit` — capability lanes with health + risk
- `brief status` — scheduled jobs incl. the Morning Brief
- `channels` — messaging/calling channel health
- `approvals` — anything waiting for your ✅

Risky actions reply with ✅/❌ approval buttons first.

## Bilingual
Korean and English are supported through command routing, contact names,
approval cards, and mocked eval coverage. Hangul recipient names and message
bodies are preserved before delivery. Telegram, Instagram, iMessage, and KakaoTalk
have recipient-confirmed V3 evidence; KakaoTalk's result required operator chat
verification plus recipient confirmation after an outcome-unknown GUI attempt.
Every new send stays approval-gated. Calls are disabled for the August 15 preview;
any post-preview re-enable and live proof requires a separate supervised gate. Trust
the Capability Cockpit and `channel health` for current live state.

---
*Updated to mirror the live cockpit on 2026-08-07. If a lane here disagrees
with the dashboard cockpit, trust the cockpit — it reads the live registry and
audit trail.*
