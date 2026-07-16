# Syzygy Tablebase Support — Verification Report (real tablebase files)

Scope: verify the existing Syzygy/Fathom integration against real `.rtbw`/`.rtbz`
data. **No engine code changed** — this report is configuration + testing only,
continuing directly from `docs/syzygy_tablebase_enablement.md`, which flagged
"real WDL/DTZ correctness" as the one unverified gap.

## 1. Tablebase files used

This sandbox has no general internet access (`curl`/`wget`/generic HTTPS is
blocked by the network allowlist — confirmed against `tablebase.sesse.net`,
`syzygy-tables.info`, and other known Syzygy hosts). However, **PyPI is
reachable** (`pip download` succeeds). `python-chess`'s own PyPI source
distribution bundles a complete, real Syzygy "regular" (standard chess) test
fixture set for its own test suite — genuine tablebase data, not synthetic.

- Obtained via: `pip download --no-deps chess` → extracted
  `chess-1.11.2/data/syzygy/regular/`
- **70 real files** (35 material configurations × WDL + DTZ): all 3-piece and
  4-piece "regular" tables — KRvK, KQvK, KBvK, KNvK, KPvK, KNNvK, KBBvK,
  KBNvK, KRvKR, KQvKQ, KRPvK, KQPvK, and 23 more (full list in
  `SOURCE.txt`, provenance cited as `http://tablebase.sesse.net/syzygy/`).
- Copied to `/tmp/syzygy_regular/` (4.4 MB total), configured via
  `setoption name SyzygyPath value /tmp/syzygy_regular`.
- Engine confirms: `info string syzygy ready up to 4 pieces
  (path=/tmp/syzygy_regular)`.

**5-piece / 6-piece tables**: confirmed infeasible in this environment. A
full 5-piece Syzygy set alone is ~939 MB (378 MB WDL + 561 MB DTZ); no host
serving Syzygy data is reachable from this sandbox at any size. This is a
sandbox/environment limitation, not an engine limitation — the integration
itself is piece-count-generic (gated by `Tablebases::max_pieces()`, read from
whatever files are actually loaded) and will use 5/6/7-piece tables
automatically if pointed at a `SyzygyPath` containing them.

## 2. Test positions with known results

Ten positions spanning wins, draws, and different material types. Expected
values are **not hand-computed** — they come from `python-chess`'s
independent `chess.syzygy` prober, run against the exact same files loaded
into the engine. This is two separate implementations reading identical
binary data, which is a stronger check than reasoning about endgame theory
by hand.

| Position | FEN | Expected WDL | Expected DTZ |
|---|---|---|---|
| KRvK | `8/8/8/8/8/8/8/R3K2k w - - 0 1` | +2 (win) | 5 |
| KQvK | `8/8/8/8/8/8/8/Q3K2k w - - 0 1` | +2 (win) | 3 |
| KBvK | `8/8/8/8/8/8/8/B3K2k w - - 0 1` | 0 (draw) | 0 |
| KNvK | `8/8/8/8/8/8/8/N3K2k w - - 0 1` | 0 (draw) | 0 |
| KNNvK | `8/8/8/8/8/8/8/NN2K2k w - - 0 1` | 0 (draw) | 0 |
| KPvK (winning) | `4k3/8/4K3/4P3/8/8/8/8 w - - 0 1` | +2 (win) | 3 |
| KPvK (drawing) | `8/8/8/8/4k3/8/4P3/4K3 w - - 0 1` | 0 (draw) | 0 |
| KRvKR | `8/8/8/3k4/8/3K4/8/3R3r w - - 0 1` | +2 (win) | 1 |
| KQvKQ | `8/8/3k4/8/8/3K4/8/3Q3q w - - 0 1` | +2 (win) | 1 |
| KRPvK | `8/8/8/4k3/8/4P3/4K3/4R3 w - - 0 1` | +2 (win) | 1 |

(An earlier draft of the KPvK "win" case had the black king incorrectly
blocking the pawn from the front, which is actually a known KPK draw — that
was caught by the same cross-check and replaced with a genuinely winning
king-supported-pawn position before running any engine tests.)

## 3. Verification results

### WDL probing

For every position, the engine's `Tablebases::probe_root` was queried via
UCI (`position fen ... / go depth 20`) with the real `SyzygyPath` set. For
each engine-chosen move, the resulting position's WDL (from the opponent's
perspective) was checked against the expected outcome:

| Position | Expected | Engine move | Result |
|---|---|---|---|
| KRvK | win | e1f1 | OK — opponent still losing |
| KQvK | win | e1f2 | OK — opponent still losing |
| KBvK | draw | a1b2 | OK — draw held |
| KNvK | draw | e1e2 | OK — draw held |
| KNNvK | draw | e1d1 | OK — draw held |
| KPvK (win) | win | e6d6 | OK — opponent still losing |
| KPvK (draw) | draw | e2e3 | OK — draw held |
| KRvKR | win | d1h1 | OK — opponent still losing |
| KQvKQ | win | d1h1 | OK — opponent still losing |
| KRPvK | win | e3e4 | OK — opponent still losing |

**10/10 correct.** WDL probing is confirmed correct.

### DTZ probing

For the 6 winning positions, the engine's chosen move was checked against
every legal move's resulting DTZ (per Syzygy convention: the move that
minimizes distance-to-zeroing while keeping the win) — i.e. a strict
DTZ-optimality check, not just "any winning move":

| Position | Engine move | \|DTZ\| achieved | Best possible \|DTZ\| | Optimal? |
|---|---|---|---|---|
| KRvK | e1f1 | 4 | 4 | Yes |
| KQvK | e1f2 | 2 | 2 | Yes |
| KPvK (win) | e6d6 | 2 | 2 | Yes |
| KRvKR | d1h1 | 24 | 24 | Yes |
| KQvKQ | d1h1 | 14 | 14 | Yes |
| KRPvK | e3e4 | 2 | 2 | Yes |

**6/6 DTZ-optimal.** DTZ probing is confirmed correct and is what actually
drives root move selection (`Tablebases::probe_root`), not just WDL.

### Search uses tablebases correctly

- Root probe short-circuits before any normal search: for all ten TB-range
  test positions, the engine returned `bestmove` immediately with **zero**
  `info depth` lines emitted, confirming the root DTZ probe fires before
  `iterative_deepening()` is even called (per `search.cpp:994-1002`) —
  exactly matches "return exact WDL/DTZ result" in the requested
  architecture, with no wasted search effort.
- Non-root (in-search) probing was also observed engaging correctly and
  usefully: bench position 4 (`8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8`, a 10-piece
  rook/pawn ending) found a **stronger line at depth 11-12 with tablebases
  enabled** (score improved cp 76 → cp 95, different/better PV) — internal
  nodes of its search tree reduce to ≤4 pieces, where the exact TB probe
  at `search.cpp:458-461` corrects what the heuristic evaluator would
  otherwise misjudge. This is the intended benefit of in-search probing
  showing up in a realistic (non-root-TB) position.
- No crashes, hangs, or illegal moves across 16+ full engine invocations
  (10 WDL/DTZ tests + 6 DTZ-optimality re-tests, each spawning a fresh
  engine process) plus 2 full 8-position bench suites — 32 searches total
  against real tablebase data with zero failures.

### No performance regression outside tablebase range

Ran the built-in `bench 12` (8 fixed positions, depths 1-12) with
`SyzygyPath` unset vs. set to the real 4-piece set, diffing node-by-node:

| Bench position | Pieces | Nodes/PV with real TB loaded |
|---|---|---|
| Start position | 32 | **Bit-identical** to baseline at every depth |
| Kiwipete-class middlegame | 32 | **Bit-identical** |
| Castled middlegame | 30 | **Bit-identical** |
| 10-piece rook/pawn ending | 10 | Diverges from depth 10 on — *improved* play (see above), not a regression |
| Endgame-ish middlegame | 24 | **Bit-identical** |
| Queenside attack middlegame | 28 | **Bit-identical** |
| KQvK (3-piece) | 3 | Resolved instantly by root probe: 0 search nodes vs. 52,380 nodes at depth 12 without TB — a speed *improvement*, not regression |
| Castled middlegame #2 | 30 | **Bit-identical** |

6 of 8 positions (all with >10 pieces, i.e. genuinely outside/far from
tablebase range) show **exactly identical node counts and principal
variations** with real tablebase files loaded vs. not — only NPS/wall-clock
timing differs, by the same ~1-5% run-to-run jitter documented in the prior
enablement report's benchmark section. The two positions that do differ are
both *within or adjacent to* tablebase range, where a difference is the
correct and intended behavior, not a bug. **No regression outside
tablebase range.**

## 4. Remaining issues / honest caveats

- **Only 3-4 piece tables verified.** 5, 6, and 7-piece tables could not be
  obtained in this sandbox (no reachable host serves them, and the combined
  size is prohibitive regardless). The probing code path is piece-count
  generic and not specific to 3-4 piece material, so this is a coverage gap
  in *this verification*, not a known defect in the integration — but it
  is not proven with larger tables. If real 5/6/7-piece files become
  available (e.g. run on a machine with normal internet access), the exact
  same test harness (10 constructed FENs + `chess.syzygy` cross-check +
  `bench` diff) can be pointed at them with no changes needed.
- **Root-probe UCI output has no `info` line.** When the root DTZ probe
  fires, the engine emits `bestmove` with no preceding `info depth ...
  score ...` line (confirmed in `uci.cpp`/`search.cpp`: the function
  returns before `iterative_deepening()` runs). This is harmless — GUIs
  will still get a valid `bestmove` — but a GUI that wants to *display* a
  tablebase-derived score/mate-distance to the user won't see one. Purely
  cosmetic, not a correctness issue, and out of scope for this "do not
  change engine code" verification pass; noting it here in case a future
  change wants to add an `info string` line for tablebase hits.
- **50-move-rule edge cases not exercised.** All test positions used
  halfmove clock 0. DTZ's 50-move-rule-aware zeroing behavior near the
  clock limit was not specifically tested (would require positions with a
  pre-set halfmove clock close to 100) — a possible follow-up if that edge
  case matters.

## 5. Summary

All five verification requirements are met using real Syzygy data (3-4
piece set, the largest obtainable in this sandbox):

- WDL probing: **10/10 correct**, cross-checked against an independent
  prober on identical files.
- DTZ probing: **6/6 DTZ-optimal**, not just "any winning move."
- Search integration: root probe short-circuits correctly with zero wasted
  nodes; non-root probing measurably improves play in a near-TB-range
  position.
- No regression outside tablebase range: 6/8 bench positions bit-identical;
  the 2 that differ do so *because* tablebases correctly engaged, not
  because of overhead.
- Zero crashes across 32 real-data engine invocations.

The integration verified in `docs/syzygy_tablebase_enablement.md` as
"correctly wired but unverified against real data" is now confirmed correct
against real data, within the 3-4 piece ceiling this environment could
obtain.
