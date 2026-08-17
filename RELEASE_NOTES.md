# Release Notes

## 4.0.0-rc.1 — 2026-08-17

Jarvis V4 4.0.0-rc.1 is a local personal-assistant release candidate for macOS. It is intended for
one owner operating one trusted Mac, with explicit review before private-data access or external
side effects. It is not a production service or a multi-user platform.

V4 is the public release identity of the frozen and tested V3 codebase, not an untested runtime
rewrite. The internal `jarvis_v2` package, `launch_jarvis_v3*` launchers, `JARVIS_V3_*` settings,
and `~/.jarvis_v3` storage remain compatibility names. New feature development belongs to V5.

### Included in the preview

- One-shot, typed-chat, push-to-talk, Google Calendar authorization, and authenticated loopback
  dashboard launch flows.
- Local tool routing with permission policy, bounded approval handling, audit metadata, and
  fail-closed treatment of uncertain outcomes.
- Isolated compatibility configuration and storage guidance so a fresh setup does not silently inherit another
  Jarvis generation's runtime state.
- Offline smoke tests and privacy checks for the source-only release candidate.

The internal Python package is still named `jarvis_v2` for compatibility. Public setup and daily
use are documented in `QUICKSTART.md`; supported capability boundaries are documented in
`CAPABILITIES.md`.

### Known limitations

- macOS is the only supported operating system for this preview.
- The design assumes a single trusted local owner. It is not hardened against another malicious
  process running as the same operating-system user.
- Background services, scheduler activation, production cutover, deliberate network-loss testing,
  and reboot recovery require separate operator-present procedures and are not enabled by the
  source package.
- Optional integrations depend on local applications, third-party accounts, operating-system
  permissions, and changing user interfaces. Availability is reported at setup time and is not
  guaranteed.
- GUI-driven actions may reach an uncertain outcome. Inspect the target state before considering a
  new request; never retry automatically.
- Runtime credentials, account identifiers, messages, contacts, local state, operational service
  definitions, and private development evidence are not included in the source-only candidate.
- A fresh checkout still requires the owner to create private local configuration and install the
  documented dependencies.

This GitHub prerelease for V4 is distributed under the MIT License. It does not activate services,
authorize a migration, or claim production readiness.
