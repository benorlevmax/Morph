# Syzygy Tablebase Support — Enablement Report

Scope: restore/enable exact ≤7-piece Syzygy probing only. No search
heuristics, NNUE, or unrelated features touched.

## Audit findings (why it was disabled)

The entire integration — UCI `SyzygyPath` option, `Tablebases::init/probe_wdl/
probe_root`, the in-search WDL probe at the top of every non-root node
(`search.cpp`, right before the TT probe), and the root DTZ probe in
`think()` — was **already fully written and correct**. The only problem was
a missing build dependency: `CMakeLists.txt`'s `CHESS_USE_FATHOM` option had
a `FATAL_ERROR` guard requiring Fathom's `tbchess.c` vendored next to the
already-present `tbprobe.c`/`tbprobe.h`/`stdendian.h` — and `tbchess.c`
(plus `tbconfig.h`, referenced by `tbprobe.h` via `#include <tbconfig.h>`,
also missing) simply weren't in the repo. With the flag defaulting `OFF`
and the dependency missing, tablebase support was 100% inert in every build.

## Files changed

| File | Change | Lines | Why |
|---|---|---|---|
| `tbchess.c` (new, repo root) | Vendored from `jdart1/Fathom` (`src/tbchess.c`, MIT), unmodified | +1049 | The missing dependency `tbprobe.c` `#include`s directly (single translation unit) — move generation and position validity logic the prober needs. |
| `tbconfig.h` (new, repo root) | Vendored from `jdart1/Fathom` (`src/tbconfig.h`, MIT), unmodified | +151 | `tbprobe.h` requires this via `#include <tbconfig.h>`; defines scoring constants and optional engine-integration override hooks (none of which this engine uses — it takes Fathom's defaults). |
| `CMakeLists.txt` | `option(CHESS_USE_FATHOM ...)` default flipped `OFF` → `ON`; comment updated | ~9 (1 functional line + 8 comment) | The `FATAL_ERROR` guard now passes (dependency present) and probing is a pure no-op at runtime without a configured `SyzygyPath`, so enabling by default only adds capability — see verification below. |

**No changes to `src/` at all** — not `search.cpp`, not `eval/`, not
`nnue/`, not `uci.cpp`. The architecture the user specified
(piece-count-gated probe → exact result or fall through to normal search)
was already exactly what the existing, dormant code did.

## Build verification

Clean Release build (this sandbox: g++ 11.4, Ninja, `-O3 -march=native`,
same toolchain as prior sessions' baselines):

```
-- Fathom Syzygy probing: ENABLED
...
[42/42] Linking CXX executable bin/test_eval
```

Builds cleanly; `tbprobe.c`/`tbchess.c` compile as C++ (per the existing
`set_source_files_properties(... LANGUAGE CXX)` directive, and the
`#ifdef __cplusplus` namespace-wrapping already present in both vendored
files) with a single pre-existing, harmless sign-compare warning inside
Fathom's own `tb_expand_mate` — not our code, not new.

## Correctness results

- **Unit tests**: 5/5 pass (`perft`, `fen`, `search`, `eval`, `pgn`) — unchanged from before.
- **Perft, start position, depth 6**: 119,060,324 nodes — correct, identical to baseline.
- **Perft, Kiwipete-class position (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1`), divide depth 5**: 193,690,690 total nodes — correct, matches the well-known reference value for this position, identical to baseline.
- **Alpha-beta behavior preserved**: ran fixed-depth (depth 10) search on 6 positions spanning 3, 4, 5, 6, 10, and full-board piece counts, on this build (Fathom compiled in, no `SyzygyPath` configured) vs. the pre-change build (Fathom not compiled in at all). **Every score, node count, and PV was bit-identical**; only NPS/wall-clock time varied, by the same amount as this sandbox's normal run-to-run system noise (see bench comparison below). This directly confirms the "preserve existing alpha-beta behavior" and "do not slow down normal search" requirements — the new code path costs nothing when not engaged.
- **Graceful fallback / no-crash checks**: `setoption name SyzygyPath value <empty-directory>` correctly reports `unavailable (path=..., no files found) -> normal search`; searching and evaluating positions with this option set produces no crashes, hangs, or incorrect output — `tb_init` on a real-but-empty directory is handled cleanly.

## Benchmark impact (before vs. after, built-in `bench 12`)

| | nodes | nps | time |
|---|---|---|---|
| Classical, before | 891,489 | 1,888,747 | 472 ms |
| Classical, after | 891,489 | 1,900,829 | 469 ms |
| NNUE, before | 482,203 | 1,686,024 | 286 ms |
| NNUE, after | 482,203 | 1,668,522 | 289 ms |

**Node counts are exactly identical, before and after, in both eval
modes.** NPS/time differ by ~1%, consistent with ordinary system jitter in
this shared sandbox (the same size of variation seen between repeated runs
of the identical binary elsewhere this session) — not a measurable
regression. This is expected: `Tablebases::available()` is a single cached
boolean check per node when no real tablebase files are loaded, and Syzygy
probing is gated behind `popcount(pos.pieces()) <= max_pieces()` (0 when
unavailable) before any real Fathom call is even attempted.

## What is NOT verified: real WDL/DTZ correctness

**This is the one honest gap.** This sandbox has no internet access from
the shell (the same `blocked-by-allowlist` restriction documented in the
NNUE/Bullet work) and no real Syzygy `.rtbw`/`.rtbz` files exist anywhere
in the repo or could be downloaded here. Everything above proves the
integration is correctly *wired* and *inert-by-default* — it does not prove
that a real WDL/DTZ probe returns the right answer, because no such probe
could actually be executed against real tablebase data in this environment.

**To finish verification** (needs to happen on a machine with internet
access): download the standard Syzygy 3-4-5 piece set (a few hundred MB,
e.g. from `tablebase.sesse.net` or `syzygy-tables.info`) or the full 6-7
piece set if disk space allows, then:

```
setoption name SyzygyPath value <path to your tablebase files>
```

and confirm `info string syzygy ready up to N pieces (path=...)` reports
`available()==true`. Then verify against these representative positions
(chosen to span ≤5, 6, and 7 pieces — construct exact FENs for whichever
piece configurations your downloaded set covers):

- A simple ≤5-piece win (e.g. KRvK, KQvK) — confirm `go` returns a mate
  score / correct WDL-derived value immediately rather than searching to
  full depth, and that `bestmove` matches the DTZ-optimal move Syzygy
  reports via any reference tool (e.g. `python-chess`'s own Syzygy module,
  or `syzygy-tables.info`'s online probe, for cross-checking).
- A 6-piece drawn position (e.g. KRvKR without a clear win) — confirm the
  engine reports a draw score/claims the draw rather than searching for a
  nonexistent win.
- A 7-piece position at the edge of the loaded set's cardinality — confirm
  `Tablebases::available()` and the piece-count gate correctly engage; then
  add one more piece (8-piece) and confirm the probe correctly declines
  (`WDLResult::Fail`) and falls through to normal search, exactly per the
  architecture diagram in the request.
- Confirm no crashes across at least a few dozen probes of each type.

## Expected Elo gain

Small but real and free, consistent with Stockfish's own historical
SPRT-tested experience adding Syzygy support: plausibly low-single-digit to
~10 Elo at normal time controls, from exact (not approximate) play in
positions that reach ≤7-piece endgames — a rare but non-zero event even at
this engine's current strength, and an unambiguous correctness improvement
independent of Elo (the engine now plays these positions *provably
optimally* rather than by search+eval approximation) whenever real
tablebase files are actually configured via `SyzygyPath`. Zero Elo impact
one way or the other for anyone who doesn't configure `SyzygyPath` at all,
per the benchmark results above.
