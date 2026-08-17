# Public release precheck

The release workflow has two offline stages. The candidate builder creates a narrow, history-free
source tree from a clean committed worktree; the preflight then scans that tree. Neither stage
initializes Git, commits, pushes, installs services, changes schedules, or touches runtime data.

1. Commit the reviewed source-only release state and make sure the worktree is completely clean,
   including untracked files. The builder reads regular blobs from `HEAD`, not runtime files or the
   working tree. Its fixed allowlist contains `jarvis_v2/**/*.py` plus reviewed top-level launchers,
   dependency/config templates, and public documentation, including
   `V3_SUPERVISED_PROOF_RUNBOOK.md`. Private operational documents, service plists, runtime state,
   logs, caches, vaults, archives, binaries, and repository history are never selected.
2. Supply any owner- or contact-specific deny markers only for the scan process, as a JSON array in
   `JARVIS_PUBLIC_RELEASE_DENY_LITERALS_JSON`. Do not save those values in source or shell history.
   Matching always retains exact-byte detection. Valid UTF-8 candidate text and paths are also
   compared using bounded Unicode NFKC normalization plus case folding, so case changes and
   canonically equivalent composed/decomposed spellings cannot bypass the private review. Binary or
   malformed UTF-8 remains exact-scanned and independently fails the binary-content policy. A text
   artifact too large for the bounded variant pass fails closed with
   `private_variant_scan_limit`. Scan-wide file, entry, byte, and normalization budgets likewise
   fail closed instead of continuing unbounded work; the scan never prints the deny literal or
   matching text.
   The ordinary/default `preview` profile permits an empty list so the offline preview workflow
   remains available. A public release candidate must instead use the explicit `publication`
   profile described below, which refuses an empty list.
3. Choose a new, explicit destination under `/private/tmp`. The builder refuses an existing
   destination and never deletes or overwrites one. Validate selection without writing:

   ```bash
   python3 -m jarvis_v2.scripts.public_release_candidate \
     --source /path/to/clean/jarvis-v3 \
     --destination /private/tmp/jarvis-v3-public-candidate \
     --dry-run
   ```

4. Build and automatically run the existing preflight:

   ```bash
   python3 -m jarvis_v2.scripts.public_release_candidate \
     --source /path/to/clean/jarvis-v3 \
     --destination /private/tmp/jarvis-v3-public-candidate
   ```

   The path-free summary contains counts, the exact non-sensitive source commit, and a deterministic
   version-2 SHA-256 manifest digest over that source commit plus every selected relative path,
   executable bit, size, and content. It also reports only the count and presence of runtime
   owner/contact deny literals, never their values, so a zero-finding receipt shows whether that
   private scan layer was actually supplied.
   The built tree also contains `PUBLIC_CANDIDATE_MANIFEST.json`, a non-private root marker bound to
   those same values. Its strict validator recomputes the selected-tree digest and rejects malformed
   schema, inconsistent commit or content edits, symlinks, service plists, repository/runtime
   directories, or extra files. The source commit and manifest form an unsigned self-declared origin
   binding, not authenticated provenance: the history-free tree contains neither repository history
   nor a trusted signature, so the marker alone cannot prove that the named commit exists or who
   created it. Marker and manifest version 1 candidates are obsolete and must be rebuilt rather than
   reused.
   A successfully preflighted candidate is finalized as an immutable staging tree: its root and
   directories are owner-readable/executable mode `0500`, regular files are `0400`, and retained
   launchers are `0500`. This prevents nested Python smoke subprocesses from adding bytecode caches.
   These owner mode bits can be reversed with an explicit permission change, so the content-bound
   marker, exact-mode validation, and preflight revalidation remain authoritative. A later checkout
   normalized to ordinary Git modes no longer counts as the reviewed immutable staging candidate,
   even when its content digest still matches.
   A failed candidate remains owner-only and writable for private review, but is never finalized or
   eligible for publication.
   Exit `0` means the candidate was created and its preflight
   passed. Exit `1` means the candidate was created but failed preflight; retain it only for private
   review and do not publish it. Exit `2` means the source, destination, Git state, or private deny
   configuration was unsafe, and no successful candidate may be claimed.
5. To rescan an already-prepared candidate directly:

   ```bash
   python3 -B -m jarvis_v2.scripts.public_release_preflight --root /path/to/candidate --summary-json
   ```

   The summary omits candidate paths. If it fails, use `--json` only in a private terminal after
   supplying the runtime deny markers; that detailed report includes sanitized relative paths.
6. Continue only on `ok:true` and `candidate_finalized_immutable:true`. Keep the immutable staged
   candidate byte-for-byte unchanged while verifying its marker. Run Python with `-B` as
   defense-in-depth so normal bytecode caches are not attempted in the exact candidate tree:

   ```bash
   cd /path/to/candidate
   python3 -B -m jarvis_v2.scripts.smoke_test_all
   ```

   Candidate-aware smoke modules structurally recognize either an exact finalized preview or
   publication tree so intentionally excluded private documents and service templates remain
   unavailable during this run. That structural context check revalidates the marker-bound tree,
   finalized modes, publication artifacts, and ordinary privacy policy, but it does not repeat or
   attest the runtime-only private deny review. Retain the original publication-profile build
   receipt as the evidence for that separate review.

   Review every retained file, then run a separately selected and reviewed local credential scanner
   through the inert adapter. The adapter does not acquire or choose a scanner. The scanner must be
   an absolute, owner-controlled, non-writable Python source file whose second line is exactly
   `# jarvis-independent-credential-scanner:1`; retain its SHA-256 independently:

   ```bash
   python3 -B -m jarvis_v2.scripts.independent_credential_scan \
     --candidate /path/to/immutable-candidate \
     --scanner /absolute/path/to/reviewed-scanner.py \
     --scanner-sha256 <reviewed-scanner-sha256> \
     --profile preview
   ```

   The adapter supplies the exact reviewed scanner source bytes to an isolated Python invocation,
   removes inherited credential and proxy environment variables, bounds output and runtime, and
   revalidates the candidate and scanner source plus the named interpreter path before and after the
   run. The scanner source and candidate traversal are descriptor-bound; the interpreter is launched
   by its trusted current pathname, so the adapter does not attest that interpreter execution was
   bound to the hashed descriptor. It also does not attest kernel-enforced network denial,
   filesystem-write denial, or absence of access through the operator's OS identity; use only a
   scanner and runtime separately reviewed for offline, read-only behavior. A content-free
   `clean:true` receipt means that scanner reported zero findings against the stable candidate; it
   does not make the scanner authoritative or replace manual review. Use `--profile publication`
   only after the publication metadata gate below is satisfied.

   Inspect the new public history before publication. If any command changes the candidate tree,
   discard that staging directory and build a fresh one; do not relax marker validation. A passing
   scan does not authorize daemon installation, scheduler activation, external actions, production
   cutover, or publication before the release date.

7. Prepare the public Git repository separately. This is an explicit operator step: the candidate
   builder and handoff verifier do not initialize Git, copy files, change permissions, stage files,
   create commits, or publish. Copy the exact candidate payload into a new reviewed repository,
   preserving which files are executable. Disable local reflog creation before creating the first
   branch or commit (`git config --local core.logAllRefUpdates false`), then make exactly one reviewed
   root commit. Do not amend, reset, rebase, import, fetch, or write test objects in this repository;
   if preparation goes wrong, discard it and prepare a new repository from the immutable candidate.
   Git stores regular files as `100644` and executables as `100755`, so the read-only verifier
   normalizes only the executable bit when comparing that commit to the candidate's owner-only
   `0400`/`0500` files. The public
   repository must have that exact commit at `HEAD` as its only root commit, exactly one local branch
   ref pointing to it, no other refs or remotes, and a completely clean index and worktree, including
   no ignored or untracked files. Shallow history, grafts, alternates, replacement refs, and partial-
   clone/promisor configuration are rejected; replacement processing and lazy fetching are disabled
   during every bounded Git inspection.

   With the same runtime-only private deny list still loaded, supply both locations and the full
   immutable public commit ID explicitly:

   ```bash
   python3 -B -m jarvis_v2.scripts.public_release_handoff \
     --candidate /path/to/immutable-candidate \
     --repository /path/to/prepared-public-repository \
     --commit <full-public-commit-id>
   ```

   Continue only when the path-free receipt reports `ok:true`,
   `git_tree_matches_candidate:true`, `marker_matches_candidate:true`, and
   `index_and_worktree_match_commit:true`, plus `single_root_commit:true`, `refs_isolated:true`,
   `promisor_disabled:true`, `object_database_exact:true`, `reflogs_absent:true`,
   `git_metadata_no_follow:true`, and `worktree_bytes_match_candidate:true`. The verifier proves the
   complete logical object inventory is exactly the single root commit's reachable commit/tree/blob
   set; unreachable objects, reflogs, external common directories, linked-worktree metadata,
   alternates, and symlinked, hard-linked, or special required Git metadata are rejected. Git object
   and ref trees plus every `.git/info` entry are descriptor-recursively scanned without following
   links; `.git/info/attributes`, external excludes/attributes configuration, all `filter.*`
   configuration, and graft files are rejected before object inspection. The handoff does not run
   `git status`; exact index and no-follow worktree comparisons establish cleanliness without
   activating configured content filters. Git object bytes and modes are checked against the
   candidate, then worktree bytes, file modes, directories, and the exact file set are read
   independently through no-follow descriptors. It also rejects extra, missing, changed, symlinked,
   hard-linked, gitlink, non-regular,
   executable-bit-drifted, or marker-drifted entries. Descriptor-anchored metadata and worktree scans
   hold and revalidate the repository and `.git` identities around each bounded Git subprocess. Git
   launches do not use `preexec_fn`;
   every isolated process group must be proven absent afterward or verification stops with the
   content-free `git_process_cleanup_unknown` result. Public failure packets and translated API
   exceptions suppress path-bearing causes. The verifier also revalidates the immutable publication
   candidate before and after inspection. It does not write or retain the private deny values. The
   verifier does not initialize Git, stage, commit, push, publish, or authorize publication.

8. If the closure workflow writes an external receipt, place it only in an already-existing
   owner-only mode-`0700` directory outside both the source and candidate trees. Final closure also
   requires a nonempty, reviewed `JARVIS_PUBLIC_RELEASE_DENY_LITERALS_JSON` value at runtime; the
   receipt records only its positive count and an anchor-keyed commitment, never the literals or an
   unkeyed hash that could expose low-entropy owner/contact values. The receipt is
   self-consistent, not self-authenticating. Before the aggregate, generate and retain a fresh
   256-bit lowercase-hex anchor outside the source, candidate, receipt, shell history, and ordinary
   command output. Supply it for both creation and verification only through the runtime-only
   `JARVIS_V3_CLOSURE_RECEIPT_ANCHOR` environment variable. The receipt contains a
   domain-separated HMAC over its full payload plus a separate anchor-keyed deny-review commitment;
   it never contains or prints the anchor. Its embedded `receipt_sha256` therefore cannot bootstrap
   authenticated `valid:true`, replay an old anchor onto a changed payload, or authorize a
   same-count deny-list substitution. Verification without the separately retained anchor reports
   self-consistency but not authenticated execution and exits nonzero. A post-link cleanup failure
   is outcome-unknown and must be investigated before retrying. Verification never reconstructs or
   upgrades an older run. Keep the receipt out of the public Git tree unless a separate review
   explicitly selects it.

9. Run the offline dependency declaration contract from the candidate:

   ```bash
   python3 -B -m jarvis_v2.scripts.requirements_contract
   ```

   This proves the deterministic direct declarations are valid, known production imports are
   classified, the stdlib Ollama adapter has no Ollama/HTTPX/Pydantic SDK declaration, and the code
   and guide agree on Python 3.11 or newer. It resolves
   nothing, imports no installed package, contacts no package index, and writes nothing. A pass does
   not make installation reproducible: `requirements.txt` is intentionally unpinned. Do not guess
   package versions. A reproducibility claim requires a separately generated and reviewed lock or
   constraints artifact, a clean-environment installation test using that exact artifact, and an
   explicit candidate-allowlist update. Until then the receipt truthfully reports
   `reproducible_install:false`.

## Publication-readiness profile

The default profile remains `preview` and does not claim publication readiness. After the operator
has separately selected and reviewed a license and a non-development version, add exactly one
tracked UTF-8 `LICENSE` file and set the tracked single-line `VERSION` to a SemVer release or an
`alpha`, `beta`, or `rc` prerelease. The builder does not choose either value.

For the final private staging review only, load the real owner/contact deny literals into the
process environment without committing them or placing them in shell history, then explicitly add
the publication profile:

```bash
python3 -B -m jarvis_v2.scripts.public_release_candidate \
  --source /path/to/clean/jarvis-v3 \
  --destination /private/tmp/jarvis-v3-publication-candidate \
  --profile publication \
  --dry-run
```

Repeat without `--dry-run` only after the path-free dry-run reports
`publication_preconditions_passed:true`, `publication_license_validated:true`, and
`publication_license_included:false`; a dry-run validates committed input but materializes nothing.
A completed publication-profile build must report all of the following:
`candidate_profile:"publication"`, `publication_ready:true`, a nonzero
`extra_private_deny_literal_count`, `publication_license_validated:true`,
`publication_license_included:true`, and `publication_version_kind:"prerelease"` or `"release"`.

The profile fails before destination creation when the runtime-only deny list is empty or malformed,
`LICENSE` is absent, empty, executable, oversized, binary, or control-bearing, or `VERSION` is a
development, snapshot, local-build, malformed, or multiline value. Reports retain only the deny
literal count and presence, never those literals. Even `publication_ready:true` means only that the
offline history-free staging requirements passed. The builder still does not initialize a
repository, choose a license or version, publish, authorize publication, authorize a release date,
install services, activate a scheduler, or cut over V2.
