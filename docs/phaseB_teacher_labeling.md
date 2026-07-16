# Phase B: Transformer-Teacher Labeling — Design, Constraints, and Partial Execution

Status: **design complete, code written, only Strategy A executed.**
Strategies B/C/D (anything requiring the DeepMind transformer checkpoint)
could not be run in this sandbox — see §1. This is an environment
limitation, not an architecture or feasibility finding; the design and code
for all four strategies are ready to run on a properly provisioned machine.

## 1. Why B/C/D couldn't run here

- **No internet from the shell.** `curl`/`wget`/Python `urllib` to any host
  outside a small allowlist return `403 Forbidden, X-Proxy-Error:
  blocked-by-allowlist` (confirmed against `static.rust-lang.org`,
  `crates.io`; the `web_fetch` tool has a *different*, broader allowlist for
  reading web pages, but has no mechanism to download and save a multi-GB
  binary checkpoint file into the sandbox). DeepMind's `searchless_chess`
  checkpoints (9M/136M/270M params) are hosted on Google Cloud Storage and
  need `gsutil`/`gcloud`/direct HTTPS GET — none reachable here.
- **No GPU.** `nvidia-smi`: not found. Even the smallest (9M-param)
  checkpoint would run inference on CPU only, which is survivable for a
  *tiny* label-experiment batch but was deprioritized given the network
  block already makes the checkpoint unobtainable.
- **No Stockfish binary with real NNUE weights.** The repo vendors full
  Stockfish *source* (`stockfish/src/`, buildable with the `g++`/`make`
  already confirmed present), but Stockfish's own network weights are
  fetched by `make net` from `tests.stockfishchess.org` at build time —
  blocked for the same reason. A source-only build without weights isn't a
  meaningful "Stockfish evaluation" baseline for a modern Stockfish version
  (pre-NNUE classical eval was removed years ago). Not attempted, given the
  network block makes it moot for Strategy A too in its literal form (see
  below).

## 2. Label strategies — exact mathematical targets (as specified)

Let `s_stock(pos)` = Stockfish's centipawn eval (white-relative),
`s_engine(pos)` = this engine's own search eval (already available, used as
the best available local substitute for `s_stock` — see §3),
`s_teacher(pos)` = DeepMind searchless_chess's output for `pos` (see §4 for
what signal that actually is), `r(pos)` = game result white-relative
(+1/0/-1), and `σ(x) = sigmoid(x/400)` (this project's existing convention,
`encoding.h::sigmoid_eval`).

- **A. Stockfish-only**: `target = σ(s_stock(pos))`. (No result blending —
  pure positional-eval imitation.)
- **B. Transformer-only**: `target = teacher_win_prob(pos)` — see §4 for
  exactly which searchless_chess output this should be (action-value at the
  played move vs. a full position value; they are not the same signal and
  the choice matters, discussed below).
- **C. Blended**: `target = α·σ(s_stock(pos)) + (1-α)·teacher_win_prob(pos)`,
  α a tunable weight (start at 0.5, same spirit as `encoding.h`'s existing
  `λ=0.5` eval/result blend).
- **D. Stockfish + transformer-confidence weighting**: `target =
  σ(s_stock(pos))`, but each sample's training-loss weight is scaled by
  `agreement(pos) = 1 - |σ(s_stock(pos)) - teacher_win_prob(pos)|` — i.e.
  positions where the transformer and Stockfish *disagree* are down-weighted
  (treated as noisier/harder labels) rather than discarded, so the transformer
  acts as a confidence filter on Stockfish labels rather than a label source
  itself.

None of these assume the transformer label is automatically superior — that
is exactly what the strategy comparison in §5/Phase C is for.

## 3. What §2 actually used here: Strategy A, substituting our own engine's search eval for Stockfish

Since neither a weighted Stockfish binary nor the DeepMind checkpoint could
be obtained, the only strategy locally executable end-to-end is a **variant
of Strategy A** using this engine's own alpha-beta search score in place of
`s_stock` — which is precisely what `chess_train gen`'s self-play already
produces (`Sample.eval`, from `Search::think()`), and precisely what Phase
A's smoke run already trained on. **No new experiment was needed for this
variant — it's the same run reported in Phase A §5.** This is explicitly
flagged as a substitute, not a true "Stockfish-only" baseline: our own
engine's classical eval is far weaker and more biased than Stockfish's, so
this variant's usefulness as a Phase-C comparison point is limited to
"proves the pipeline trains on *some* real eval signal," not "establishes
what a true Stockfish-only baseline would score."

## 4. Teacher signal audit (DeepMind `searchless_chess`) — what's actually available to extract

From arXiv:2402.04494 and the `google-deepmind/searchless_chess` repo
(Apache-2.0 code / CC-BY-4.0 weights, already cited with full detail in the
earlier architecture-investigation report, `docs/transformer_architecture_investigation.md`
§2.2 — not re-researched from scratch here):

- The model's **native training target is action-value**: given a position
  and a candidate move, it predicts a win-probability-like scalar for that
  specific move (this is how ChessBench's 15B datapoints are structured —
  `(position, action, Stockfish-16 value)` triples, not `(position, value)`
  pairs).
- A **position value** can be derived by evaluating all legal moves' action-values
  and taking the max (white-to-move) / min (black-to-move) — i.e. a
  1-ply search using the transformer as a leaf evaluator over the legal move
  list, not a single forward pass on the position alone. This is the
  correct way to get a `teacher_win_prob(pos)` comparable to a Stockfish
  static eval, and it's more expensive than a single forward pass
  (num_legal_moves forward passes per position, or one if the model exposes
  a "value head" variant — the smaller 9M checkpoint may be more
  suitable for cheap batch labeling for exactly this reason).
- **Policy signal** (a probability distribution over legal moves) is also
  available and could feed a *different* project (move-ordering/policy
  distillation) but is out of scope for Phase B, which is about NNUE
  *evaluation* target labels specifically per the user's instruction.
- **Expected outcome**: the paper also reports the model can be trained/
  queried for win/draw/loss classification directly; if the checkpoint
  exposes this head separately it would be the cheapest, most directly
  comparable signal to Stockfish's centipawn-derived win probability.

**Practical recommendation for whoever runs this next** (not yet
implementable here): use the smallest (9M-param) checkpoint for the initial
label-generation pass — it's dramatically cheaper for the "evaluate every
legal move" position-value derivation above, and the paper reports it is
already strong (the strength gain from 9M→270M is real but the 9M model is
far more than adequate for this "label quality/style" comparison, especially
against a lightly-trained small NNUE where the bottleneck is data volume,
not label precision).

## 5. Tiny experimental NNUE per strategy — what ran, what didn't

`tools/nnue_training/train_reference.py` (Phase A's numpy stand-in trainer)
already supports arbitrary per-sample targets — extending it to strategies
B/C/D is a one-line change (replace the `win_prob_target` computation in
`encode_sample()` with the blended formula from §2), not new machinery. This
was **not done**, because strategies B/C/D need `teacher_win_prob(pos)`
values that don't exist without the checkpoint. Only the Strategy-A-variant
net (`smoke_A.nnue`, Phase A §5) exists and was compared against nothing
else, since there is nothing else to compare it against yet.

## 6. Honest bottom line for Phase C

**Phase C's requested comparison (training loss, validation loss, reference
inference agreement, engine NPS, nodes searched, head-to-head Elo, tactical/
positional/endgame test behavior, ranked by label strategy) cannot be
produced honestly right now, because three of the four label strategies have
no data to train on.** Fabricating comparative numbers for strategies that
never ran would be actively misleading. What CAN be honestly reported: the
Strategy-A-variant network trains, quantizes, exports, and verifies
correctly (Phase A §5); its Elo is not meaningful at this data volume (6
sanity games, `+0 −6 =0`, exactly as expected for 3,000 positions / 2
epochs); and no evidence exists yet, in either direction, on whether the
transformer teacher would improve on it, because that experiment could not
be run in this environment.
