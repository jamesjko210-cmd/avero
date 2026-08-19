# Avero branding and compatibility

**Avero** is the product, portfolio, repository, documentation, and future release identity.

The project was previously developed under the name **Jarvis**. Some internal names remain on
purpose because installed environments, local data, approval evidence, and release proofs depend
on them. They are compatibility identifiers, not the current product name.

## Current identity

- Product name: Avero
- Release line: Avero V4.x
- Public repository: Avero
- Owner-facing documentation: Avero

## Compatibility identifiers

The following names remain supported until a separately tested migration exists:

- Python imports under `jarvis_v2`
- `launch_jarvis_v3*` launchers
- `JARVIS_*` and `JARVIS_V3_*` environment variables
- `~/.jarvis_v3` local state and compatibility paths
- persisted schema names, receipt domains, proof identifiers, and historical release evidence
- legacy `Jarvis` command aliases where existing installations depend on them

Do not globally replace these identifiers. A future migration must be additive, detect conflicting
old and new configuration, preserve rollback, and prove that local state and approval evidence are
not lost or replayed.

Historical V1–V4 release notes may still say Jarvis because rewriting old evidence would make the
record less accurate. New product-facing text should say Avero.
