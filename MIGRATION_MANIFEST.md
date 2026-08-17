# V3 workspace migration manifest

> Historical lineage record: V4 promotes the frozen and tested V3 implementation as a public
> release identity. It does not perform another runtime-state migration or rename the V3-compatible
> launchers, environment keys, storage paths, or service contracts documented below.

Date: 2026-08-03

## Source lineage

- Source workspace: private Jarvis V2 development tree.
- Source Git head at migration: `0589511faced73a3d7af8c4c007dbffdeb624d84`.
- The V3 Git repository preserves that local lineage and has no remote configured.
- The current V2 working-tree source and tests were overlaid without modifying the V2 workspace.

## Material intentionally excluded

- Populated environment files and credentials.
- Runtime databases, write-ahead logs, locks, caches, and private vault data.
- Machine-specific service definitions and launch state.
- Editor/agent private configuration and compiled Python cache files.

The V3 workspace includes a sanitized `.env.example` only. A future V3 runtime must use new,
isolated `.jarvis_v3_runtime` paths, blank integration credentials, live-send flags disabled, and a
different dashboard port before any command is run against a configured environment.

## Production boundary

Existing V2 dashboard and messaging services still point to the V2 workspace. They were not
renamed, stopped, repointed, or edited during this migration. V3 service cutover requires a
separate approval after regression and supervised acceptance testing.

## Verification gate

Before V3 is treated as runnable or releasable:

1. Confirm no `.env`, runtime database, token, key, plist, or private cache entered the workspace.
2. Create an isolated owner-only V3 environment from scratch.
3. Run focused regression tests.
4. Complete the full V3 smoke suite.
5. From a clean committed worktree, use the offline `public_release_candidate` builder described in
   `PUBLIC_RELEASE_PRECHECK.md`. It copies only allowlisted regular `HEAD` blobs into a new explicit
   destination under `/private/tmp`, includes no repository history, and runs the read-only
   `public_release_preflight` on the result. It does not initialize Git, commit, push, or publish.
