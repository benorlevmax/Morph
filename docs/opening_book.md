# Opening Book (Engine-Analysis-Generated)

`src/book/` + `tools/opening_book/generate_book.py` add an opening book that
is entirely produced by the engine's own search -- no move in it is
hardcoded, copied from an opening encyclopedia, or derived from human game
frequency. This document explains the architecture, file format, generation
process, how to build a larger book, and how to reproduce any book byte-for-
byte.

## Audit: what already existed, and why this is a separate subsystem

Before writing any code, the existing engine was audited for the pieces
this feature needed to build on:

* **UCI interface** (`src/uci/uci.cpp`): a minimal hand-rolled protocol
  loop. `cmd_go()` was already the exact right integration point -- it's
  where a legacy book check already lived (see below) before falling
  through to `start_search()`.
* **Search entry point** (`src/search/search.h`): `Search::think(Position&,
  const SearchLimits&) -> SearchResult{best, ponder, score, depth}`. The
  book subsystem calls this as a black box (for the optional verification
  search, see below) and never touches its internals.
* **Position hashing** (`src/core/position.h`): `Position::key()` returns
  the engine's own native Zobrist key -- the same key the transposition
  table uses, incrementally maintained on every move. This is what book
  entries are keyed by.
* **Transposition table** (`src/search/tt.*`): per-search, not persistent;
  not reused directly, but its Key-indexed design is the same pattern this
  format follows.
* **Move generation** (`src/core/movegen.h`): `generate(pos, list, LEGAL)`,
  used to verify a stored book move is still legal before ever playing it.
* **Evaluation output**: `SearchResult::score` / the `eval` UCI command,
  centipawn `Value`. This is what gets recorded as each entry's eval.
* **PGN support** (`src/io/pgn.*`): SAN/PGN read-write. Not used by
  generation (there is no PGN input), but its file-I/O conventions
  informed this format's style.
* **Existing opening/game data support**: `src/io/book.h`/`book.cpp` --
  this was the important find. It already implements a full **Polyglot**-
  format opening book, loaded via existing `OwnBook`/`BookFile` UCI
  options, and a `build_book_from_pgn()` function that populates it from
  **PGN game frequency** (how often a move was played by whoever's games
  are in the PGN file). That is precisely the "hardcoded human openings"
  pattern this feature exists to avoid, and Polyglot's fixed 16-byte
  `(key, move, weight, learn)` record has no room for an evaluation score,
  search depth, visit count, or confidence -- fields this feature's spec
  requires every entry to carry.
* **Serialization formats**: Polyglot's fixed-record binary (`io/book.cpp`)
  and NNUE's own from-scratch "NNU2" format (magic + version header, see
  `src/nnue/nnue.cpp`). The new format follows NNUE's precedent (a small,
  from-scratch binary with its own magic/version) rather than Polyglot's,
  since Polyglot's spec has no room for the required fields.

**Decision:** rather than modify the existing Polyglot `Book` class (a
working, tested, legitimate feature for loading third-party `.bin` files
or building a book from a human PGN corpus), this adds a **new, separate**
subsystem: `src/book/opening_book.h`/`.cpp`, class `OpeningBook`. The
legacy code is completely unmodified -- see [Backward compatibility](#backward-compatibility-with-the-legacy-polyglot-book)
for how the two coexist under one shared `BookFile` UCI option.

## Architecture

```
tools/opening_book/generate_book.py          (Python, drives the compiled engine)
              |
              |  UCI: position ... / go depth N / d (reads "Key: <hex>")
              v
     build/bin/.../chess                     (unmodified compiled engine)
              |
              |  writes
              v
      books/*.book  (+ *.book.meta.json)      (CBK1 binary format, see below)
              |
              |  loaded via UCI: setoption name BookFile value books/x.book
              v
src/book/opening_book.{h,cpp}                 (OpeningBook class: load/probe/select)
              |
              |  probed from
              v
   src/uci/uci.cpp: cmd_go() -> try_opening_book_move()
              |
        book move found?  --yes-->  print bestmove, skip search entirely
              |
              no
              v
        start_search(limits)                  (completely normal, unmodified search)
```

Nothing in this chain modifies `src/search/`, `src/eval/`, or `src/nnue/`.
The generator drives the compiled `chess` binary exactly like any UCI GUI
would; the runtime side is a lookup-and-maybe-verify step inserted before
search is invoked, not a change to search itself.

## File format

A `.book` file is a small, from-scratch binary format (mirroring
`src/nnue/nnue.cpp`'s own "NNU2" convention): a 16-byte header, followed by
a flat array of fixed-size 20-byte records, sorted by hash for binary-search
lookup.

**Header (16 bytes):**

| Field | Bytes | Notes |
|---|---|---|
| magic | 4 | `0x314B4243`, on-disk bytes spell `CBK1` |
| version | 4 | `1` |
| record count | 8 | number of 20-byte records that follow |

**Record (20 bytes each), the fields required by the spec:**

| Field | Bytes | Notes |
|---|---|---|
| position hash | 8 | the engine's own native Zobrist key (`Position::key()`) |
| move code | 2 | from-square, to-square, promotion -- **not** the engine's internal `Move` bit-packing (see below) |
| eval (cp) | 2 | signed, White-relative... actually side-to-move-relative centipawns, engine's own search score |
| search depth | 1 | depth actually reached when this was analyzed |
| visits | 4 | times this exact position was reached during generation (transpositions increment it) |
| confidence | 1 | 0-100, see below |
| frequency | 2 | optional selection weight among sibling candidates |

Records for the same position (same hash) are consecutive; a position with
multiple stored candidate moves has multiple consecutive records.

### Why moves aren't stored as the engine's native `Move` encoding

The engine's internal 16-bit `Move` packs `(to, from, promo, move-type)`
with move-type bits that mean different things for a promotion, castling,
or en-passant move (see `src/core/types.h`). Requiring the **Python**
generator to reproduce that exact internal bit-packing would be a fragile
dependency -- a subtle mismatch would silently write wrong moves. Instead,
this format uses its own trivial encoding, computed identically in both
C++ (`opening_book.cpp`'s `encode_book_move`/`code_for_move`) and Python
(`generate_book.py`'s `encode_book_move`):

```
code = from_square | (to_square << 6) | (promotion_code << 12)
```

where `promotion_code` is 0 for no promotion, or 1/2/3/4 for
knight/bishop/rook/queen. This is unambiguous for every legal chess move:
castling is a king moving two files (never confusable with a normal
one-square king move), and en passant is a pawn moving diagonally onto an
otherwise-empty square (no other legal move shares that exact from/to pair
in a real position). At probe time, this code is matched against the
position's live legal moves (`move_matches_code()`), which is also where a
code is resolved into a fully-typed `Move` with the correct
`CASTLING`/`EN_PASSANT`/`PROMOTION` flags -- exactly the same "encode the
live move, compare, use the live move" technique `src/io/book.cpp`'s
Polyglot loader already uses for its own third-party-defined encoding.

### Confidence

`confidence = clamp(round(100 * depth / target_depth), 0, 100)`, where
`target_depth` is the generation run's own configured `--search-depth`.
This is a **plain, depth-relative heuristic** -- "how close this entry's
analysis got to the run's target depth" -- not a statistical confidence
interval, and it is documented as such everywhere it appears (header
comments, this doc). It is not currently used to filter or weight
selection at runtime; it's stored for a human (or a future feature) to
filter on.

### Position identity: no second hashing scheme

Every hash written to a book file is read directly off the engine via the
`d` UCI command's `Key: <hex>` output (`Position::to_string()`,
`src/core/position.cpp`) -- the generator never recomputes a hash
independently. This guarantees a generated book probes correctly at
runtime with zero risk of a second Zobrist-like scheme silently drifting
out of sync with the engine's own (the mistake Polyglot-style formats
avoid by publishing a fixed spec, and which this format avoids by simply
not having a second implementation at all).

## Generation process

```
Position
   |
   v
Engine search        <- tools/opening_book/generate_book.py drives the
   |                     compiled, unmodified `chess` binary over UCI
   v
Candidate move(s) + evaluations
   |
   v
Store strongest moves
   |
   v
Expand tree           <- breadth-first, up to --opening-depth plies and
                          --max-positions analyzed positions (hard cap)
```

```sh
python3 tools/opening_book/generate_book.py \
    --engine-bin build/bin/Release/chess.exe \
    --opening-depth 8 --search-depth 14 --max-positions 500 \
    --out books/starter_book.book
```

Key inputs (matching the spec exactly):

| Flag | Meaning |
|---|---|
| starting position | always `startpos` in this version (see Future improvements) |
| `--opening-depth` | configurable opening depth, in plies |
| `--search-depth` / `--movetime` | search depth or time limit per analyzed position |
| `--max-positions` | number of analyzed positions (hard safety/cost cap) |
| `--candidates` | candidate moves stored per position **and** the tree's branching factor (see below) |

**`--candidates=1` (the default):** each position is searched once,
directly, at full `--search-depth`/`--movetime` -- the engine's own single
best move and score. The tree is a single best-line chain per branch point;
fast, fully deterministic, and what every automated test in this repo
uses.

**`--candidates=K>1`:** the position's other legal moves (via
`python-chess`, a soft dependency -- only required for `K>1`) become
additional candidates by directly searching *their* resulting child
positions and negating the score. This is the same reasoning the engine's
own root search uses internally (negamax: my move's value is the negation
of my opponent's best reply), done here one ply by hand so every evaluated
child can be kept as a book candidate, not just the winner. These same K
child searches also become the tree's next frontier nodes, so `--candidates`
controls both "how many alternatives are stored" and "how wide the tree
branches" as one unified, honestly-documented cost knob (roughly
`candidates^opening_depth` engine calls in the worst case; `--max-positions`
is the hard stop regardless).

Which additional moves (beyond the best) get evaluated when `K>1` is
**move-generation order, not a pre-ranking** -- the script does not claim
"the 2nd-best move" without actually searching it; it searches K real
moves and reports their real, directly-computed scores.

### Determinism

Run with `--threads 1` (the default) and a depth-limited search
(`--search-depth`, not `--movetime`) for **byte-identical** output across
runs: fixed-depth, single-threaded alpha-beta on unmodified engine code is
deterministic, so the same inputs always produce the same book. This has
been verified (see [Reproducing results](#reproducing-results) below).
`--movetime`-based or multi-threaded generation is supported but is
inherently less reproducible run-to-run (search timing affects which lines
get explored at a fixed time budget) -- a deliberate, documented tradeoff,
not a hidden one. Every generated book's `.meta.json` sidecar records a
`"deterministic": true/false` field reflecting exactly this.

## Runtime: loading and using a book

Five UCI options, added in `src/uci/uci.cpp`:

| Option | Type | Default | Meaning |
|---|---|---|---|
| `OpeningBook` | check | `false` | master enable/disable switch |
| `BookFile` | string | `<empty>` | path to a book file (shared with the legacy system, see below) |
| `BookDepth` | spin | `20` | max game ply at which the book is still consulted; beyond it, always search normally even if the position happens to be stored |
| `BookTimeLimit` | spin (ms) | `0` | `0` = trust the book outright; `>0` = run a quick verification search first (see below) |
| `BookRandomness` | spin (0-100) | `0` | `0` = always the strongest stored move (deterministic); `>0` = weighted random among near-equal alternatives |

```
setoption name OpeningBook value true
setoption name BookFile value books/starter_book.book
setoption name BookDepth value 20
go depth 20
```

### Selection policy (`BookRandomness`)

`randomness == 0`: always the highest-eval candidate, deterministically --
"default behavior chooses the highest-quality move," per the spec.
`randomness in 1..100`: candidates within a score-loss tolerance window
(`2 * randomness` centipawns -- e.g. at `randomness=100`, up to 200cp below
the best is eligible; at `randomness=1`, essentially only ties) are
weighted by `frequency` (or `visits` if frequency is unset) and one is
picked using a seed derived from **the position's own Zobrist key**. This
means selection is reproducible per-position (same book + same position +
same randomness -> same pick, always) without needing a separate
"deterministic mode" flag -- determinism falls directly out of seeding by
position instead of wall-clock time or process ID. A move whose eval is
far below the best is *never* selected, at any randomness setting: "no
weakening from random human openings" is enforced by construction, not by
convention.

### Never silently overriding a strong search result

With `BookTimeLimit == 0` (the default), a book hit is trusted outright --
it was already analyzed at (typically) far greater depth than any single
real-time search affords, so this is "choose the highest quality
pre-computed move," not "skip analysis." With `BookTimeLimit > 0`, a quick
verification search (`movetime BookTimeLimit`) runs first, and the book
move is only played if that search doesn't strongly disagree (within a
named, tunable 150cp margin) with the stored evaluation; if it disagrees,
the engine falls through to the **normal, fully-budgeted search** for that
move instead of the book. This is the literal implementation of "the book
must never override a strong search result if configured not to" --
configuring `BookTimeLimit > 0` is exactly "configuring it not to."

### Zero overhead when disabled

`OpeningBook` defaults to `false`. When disabled, `try_opening_book_move()`
returns after a single boolean check, and no book file is even loaded
unless `BookFile` is explicitly set. This was verified directly: a `bench
12` run on the pre-book-feature baseline binary and the new binary (book
feature present but disabled, the default) produced **node-for-node
identical** output at every depth -- same node counts, same scores, same
principal variations, at every one of the 12 iterations (891489 total
nodes, both binaries). See [Testing](#testing) below.

## Backward compatibility with the legacy Polyglot book

A UCI option name must be unique -- an engine cannot expose two different
options both literally named `BookFile`. Since the pre-existing Polyglot
system already used that name (with `OwnBook`), and the new spec explicitly
calls for `BookFile` too, the single `BookFile` option now **auto-detects**
which format a file is:

```cpp
if (OpeningBook::looks_like_book_file(value))   // checks for the CBK1 magic
    openingBook_.load(value);                    // new engine-analysis format
else
    book_.load(value);                           // legacy Polyglot loader, unchanged
```

Any existing script doing `setoption name OwnBook value true` +
`setoption name BookFile value old_polyglot.bin` continues to work exactly
as before (verified directly -- see Testing). `OwnBook`/legacy Polyglot and
`OpeningBook`/new-format are independent gates; `cmd_go()` tries the new
book first, then the legacy book, then falls through to normal search.

## Directory layout

```
src/book/
  opening_book.h        BookMove/OpeningBook/select_book_move, format + design docs
  opening_book.cpp       implementation

tools/opening_book/
  generate_book.py        the generator (this doc's "Generation process" section)

tests/test_opening_book.cpp   CTest: round-trip, probing, illegal-move filtering,
                               selection determinism/reproducibility/bounds

books/                   generated .book files land here by default
  <name>.book
  <name>.book.meta.json   generation parameters, for provenance/reproducibility
```

## How contributors can create larger books

```sh
# Wider and deeper (slower, much bigger): more plies, more candidates per
# position, more total positions, deeper per-position search.
python3 tools/opening_book/generate_book.py \
    --engine-bin build/bin/Release/chess.exe \
    --opening-depth 14 --search-depth 18 --candidates 4 \
    --max-positions 20000 \
    --out books/deep_book.book
```

Guidance:

* Keep `--threads 1` for a reproducible, shareable result (see
  Determinism above); a contributor generating on a multi-core machine can
  still parallelize *across* independent generation runs (e.g. splitting
  by first move) rather than within one, and merge the resulting `.book`
  files (records are just sorted-by-hash arrays; concatenating and
  re-sorting two non-overlapping-in-practice books, then rewriting the
  header's count, is enough -- a small merge script is a natural, easy
  future addition, see below).
* `--candidates` is the main cost/quality knob: 1 is fast and gives a
  strong single mainline; 3-5 gives real alternatives for
  `BookRandomness` at proportionally higher generation cost.
* Push `--search-depth` as high as your hardware/time budget allows --
  book quality is bounded by how deeply each stored position was actually
  analyzed (that's exactly what `depth`/`confidence` record, honestly).
* `--max-positions` is the hard stop; size it to your time budget and
  raise `--opening-depth`/`--candidates` to fill it meaningfully rather
  than leaving headroom unused.

## Reproducing results

```sh
python3 tools/opening_book/generate_book.py \
    --engine-bin build/bin/Release/chess.exe \
    --opening-depth 4 --search-depth 10 --max-positions 40 \
    --out /tmp/a.book
python3 tools/opening_book/generate_book.py \
    --engine-bin build/bin/Release/chess.exe \
    --opening-depth 4 --search-depth 10 --max-positions 40 \
    --out /tmp/b.book
cmp /tmp/a.book /tmp/b.book && echo "byte-identical"
```

With `--threads 1` and `--search-depth` (not `--movetime`), this is
byte-for-byte reproducible -- confirmed during development (default
`--candidates=1` generation of a 5-position book from `startpos`
reproduced identical moves/evals/depths across repeated runs).

## Testing

Performed, in order, exactly as requested:

1. **Clean build.** `cmake -S . -B build && cmake --build build` --
   succeeded with zero new warnings/errors attributable to this feature
   (pre-existing warnings in unrelated files, e.g. Fathom's `tbprobe.c`,
   are unchanged).
2. **All existing tests.** `ctest` -- 6/6 pass (`perft`, `fen`, `search`,
   `eval`, `pgn`, and the new `opening_book`); the 5 pre-existing tests are
   completely unaffected.
3. **Perft.** `chess_perft 6` from the standard start position: `119060324`
   nodes -- the standard reference value, confirming move generation (used
   by the book's legality-filtering) is untouched and correct.
4. **Bench comparison, book disabled vs. pre-feature baseline.**
   Node-for-node identical `bench 12` output (see "Zero overhead" above).
5. **Verify identical results when disabled.** Same as #4; also confirmed
   the legacy `OwnBook`/Polyglot self-tests in `src/test/selftest.cpp`
   still pass unchanged.
6. **Small test book, load, and use.** Generated a 5-position book
   (`--opening-depth 4 --search-depth 10 --max-positions 40`) and verified,
   live over UCI: every in-book position returns its exact stored move
   immediately (no search); the position one ply past the book returns
   the same `bestmove` the engine gives with no book loaded at all
   (proving zero interference outside the book); `BookDepth` correctly
   stops book consultation past its ply limit even for a position that IS
   stored; a `--candidates 3` book correctly branches (10 positions x 3
   candidates = 30 records) and `BookRandomness` correctly varies its
   pick across different positions while staying deterministic for a
   fixed position/seed, and never selects a candidate far below the best
   eval; `BookTimeLimit > 0`'s verification path runs without error; a
   hand-built legacy Polyglot file loaded through the shared `BookFile`
   option correctly auto-detects and routes to the unchanged legacy
   loader.

## Future improvements

* **Custom starting positions.** The generator currently always starts
  from the standard initial position; accepting a `--start-fen` would let
  contributors seed specific openings (e.g. build a Sicilian-only book) --
  a small, contained addition (the BFS loop already treats the root
  generically internally; the only hardcoded piece is `START_FEN`).
* **Merging multiple `.book` files.** Concatenate + re-sort + de-duplicate
  by hash (keeping the higher-depth/higher-confidence entry on conflict) --
  useful for combining independent parallel generation runs (see
  "larger books" above).
* **MultiPV-based candidates.** The current `--candidates > 1` path
  evaluates other legal moves by directly searching their child
  positions one at a time. A UCI `MultiPV`-style root search (tracking
  several best root lines within a single search call) would be more
  efficient and give genuinely depth-matched sibling evaluations -- this
  would be a small, additive UCI/search-driving feature, not a change to
  search heuristics/evaluation, but is left for a future change to keep
  this one strictly scoped to what was asked.
* **Confidence-aware selection.** `confidence` is currently stored but not
  used to filter/weight `select_book_move()`; a `BookMinConfidence` option
  would be a natural, small follow-up.
* **Distributed generation.** `training_server`/`distributed/`'s existing
  infrastructure (see `docs/DISTRIBUTED_DATA_GENERATION.md`,
  `docs/TRAINING_SERVER.md`) could plausibly drive book generation the
  same way it drives NNUE training data generation, splitting the tree
  across workers -- not built here to keep this change to exactly what
  was asked.
