# Security

This document covers Morph's distributed compute platform (`platform/`)
security model, since that's the part of this project that involves
running code across a network of volunteer machines. Engine-only security
concerns (memory safety in `src/`, etc.) are handled through normal code
review.

## Threat model

Running a public compute-contribution server means accepting connections
from anyone on the internet, some of whom will be careless, some
malicious. The design goal is: a compromised or malicious worker can waste
its own operator's electricity, but cannot corrupt the shared training
data, cannot impersonate another worker, and cannot make the server
execute arbitrary code.

**What the server protects against:**

- **Fabricated training data.** Every submitted position goes through
  structural validation (legal FEN, valid side-to-move, score/depth/nodes
  in sane ranges — `distributed/server/validation.py`) and a plausibility
  check (`platform/server/anti_cheat.py` — e.g. a claimed search depth
  with an implausibly low node count is rejected; nodes=0 is a recognized
  "not tracked" sentinel for bulk-exported data, not treated as
  suspicious). A worker that racks up 25+ rejected records in 30 minutes
  is automatically disabled.
- **Duplicate submissions.** Every position is deduplicated by a
  content hash (fen + eval + result + depth + engine_version); a
  duplicate is silently counted as a duplicate, not double-counted as new
  data. Match results and artifact-completing task submissions are
  similarly guarded — resubmitting against an already-completed task is
  rejected (HTTP 409), not double-applied.
- **Worker impersonation.** Every worker gets a random bearer token at
  registration time; only its SHA-256 hash is stored server-side (same
  pattern used for account API keys and session tokens). Every
  worker-facing endpoint requires this token.
- **Tampered artifacts.** Every artifact (a network file, a training
  dataset) is content-addressed by a SHA-256 the *server* computes from
  the bytes it received — never a client-supplied claim. A worker
  downloading an artifact must independently verify that hash against the
  bytes it receives before using them for anything (see
  `platform/worker/artifacts.py`); a mismatch is treated as fatal, not
  logged-and-ignored.
- **Abandoned/crashed work.** Every task assignment has a lease with an
  expiry. If a worker disconnects, crashes, or is killed mid-task, the
  lease expires and the task is reassigned — no manual intervention
  needed, and no task silently vanishes.
- **Credential stuffing / brute force.** Registration, login, and
  API-key-regeneration endpoints are rate-limited per IP (or per account,
  for key regeneration). Passwords are hashed with PBKDF2-HMAC-SHA256 at
  310,000 iterations (OWASP's 2023 minimum recommendation); API keys and
  worker/session tokens are random and only ever stored as their SHA-256
  hash. Every secret-equality check (admin token, legacy shared
  registration secret) uses a timing-safe comparison
  (`hmac.compare_digest`), not `==`.
- **Arbitrary code execution.** The worker only ever invokes the compiled
  engine binaries it was configured with (`chess`, `chess_train`) as
  subprocesses with a fixed, known argument shape — it does not download
  and execute arbitrary scripts or binaries the server sends it, and does
  not `eval`/deserialize untrusted data into executable form.

**What is explicitly out of scope / accepted limitations:**

- The server cannot *prove* a submitted position came from a real engine
  search — proving that would require re-running the search server-side,
  which defeats the point of distributing the work. The defenses above
  (attribution, structural validation, plausibility checks, rate-based
  auto-suspension) catch cheap and sustained abuse, not a sophisticated
  attacker willing to run a real (but differently-configured) engine.
- A worker's own machine is trusted to run the binaries you gave it
  faithfully; this platform has no remote attestation. If you don't trust
  a specific server operator's build, don't point your worker at it.

## Reporting a vulnerability

Please do not open a public GitHub issue for a security vulnerability.
Instead, open a private security advisory via this repository's GitHub
Security tab ("Security" → "Report a vulnerability"), or contact the
maintainer directly if you don't have GitHub access. Include: what you
found, how to reproduce it, and what you think the impact is. We'll
acknowledge reports and work with you on a fix and disclosure timeline.

## For server operators

- Always set `CHESS_PLATFORM_ADMIN_TOKEN` explicitly in production
  (if unset, one is generated and printed at startup, which is fine for a
  quick local test but means the token isn't persisted across restarts).
- Prefer per-account API keys (`registration_secret` unset) over the
  legacy shared-secret registration mode for any public-facing
  deployment — a shared secret can't be revoked for one bad actor without
  affecting every other worker using it.
- Put the server behind TLS (a reverse proxy — see
  [platform/docs/SERVER.md](platform/docs/SERVER.md)) before exposing it
  publicly; the server itself speaks plain HTTP.
- Review `platform/server/anti_cheat.py`'s thresholds
  (`AUTO_DISABLE_THRESHOLD`, `AUTO_DISABLE_WINDOW_MINUTES`) and
  `platform_config.py`'s rate limits for your deployment's expected scale
  before going public