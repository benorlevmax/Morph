# Contributing to Morph

There are two ways to contribute: code, or compute. Both are welcome and
neither requires the other.

## Contributing compute

The easiest way to help: run the worker client and let it use your
machine's spare CPU (or GPU, if you opt in) time. See the
[README](README.md#run-the-worker-contribute-compute) for the quick-start,
or [platform/docs/WORKER.md](platform/docs/WORKER.md) for the full
reference. No code knowledge required.

## Contributing code

### Engine (`src/`)

- C++20, built via CMake (see [README.md](README.md#build-the-engine)).
- Run `ctest --test-dir build -C Release --output-on-failure` before
  opening a PR — every existing test must still pass. Add a test under
  `tests/` for new behavior where practical (perft/FEN/Zobrist/PGN/eval/
  search tests all follow the same CTest pattern already in that
  directory).
- If your change touches search or evaluation, include a bench comparison
  or SPRT match result in the PR description (see `chess_match --sprt`) —
  strength-affecting changes need evidence, not just "should be an
  improvement."
- Perft correctness is non-negotiable: any movegen change must still pass
  `chess_perft` at standard depths on the standard test positions.

### Platform (`platform/`)

- Python 3.9+. Install `pip install -r platform/requirements.txt`.
- The server (`platform/server/`) and worker (`platform/worker/`) are
  deliberately decoupled from `distributed/` (the original, simpler
  LAN-only coordinator) — `distributed/server/` and
  `distributed/database/schema.sql` should not be modified by platform
  changes; `platform/server/database.py` subclasses and extends instead.
  See [platform/docs/ARCHITECTURE.md](platform/docs/ARCHITECTURE.md) for
  why.
- Test any server change against a fresh SQLite database before opening a
  PR — migrations must be idempotent (safe to run against both a brand
  new database and an existing one from a previous version).
- Test any worker-side task executor change with a real local server +
  real engine binary, not just unit tests against mocked HTTP calls — this
  project's whole point is that task execution is real, and a mocked test
  can't catch a broken CLI invocation or a malformed subprocess call the
  way an actual end-to-end run does.

### General

- Keep commits focused; one logical change per commit/PR makes review
  faster.
- If you're proposing a larger architectural change (a new task type, a
  new evaluation architecture, a change to the distributed protocol),
  open an issue first to discuss the approach before investing in an
  implementation — this avoids wasted work on both sides.
- Be honest in PR descriptions about what's actually tested vs. what's
  believed to work — this project would rather merge a smaller, verified
  change than a larger one with unverified edges.

## Reporting bugs

Open an issue with: what you ran, what you expected, what happened
instead, and (for engine bugs) the FEN/PGN that reproduces it if possible.
For worker/server bugs, include the relevant log lines (worker output is
timestamped and printed to stdout; the server logs via uvicorn).

## Security issues

Do not open a public issue for a security vulnerability — see
[SECURITY.md](SECUR