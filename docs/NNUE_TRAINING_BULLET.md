# NNUE Training with Bullet

How to train a network for this engine using [bullet](https://github.com/jw1912/bullet),
the trainer used by Reckless, Berserk, and most modern engines.

> **Status / blockers (this machine):** bullet **cannot run here** — see
> [§6 Blockers](#6-blockers). The data-export step (§2) is implemented and tested;
> the training + conversion steps (§3–§5) are documented to run on a machine with
> a Rust toolchain and (ideally) an NVIDIA GPU.

---

## 1. Our target architecture

From `src/nnue/nnue.{h,cpp}` — **do not change these; the trainer must match them.**

| Property | Value |
|---|---|
| Feature set | **HalfKP** — `(king_bucket, oriented_piece_square, own/opp piece type)`, **kings excluded** from the piece set |
| Perspectives | **Dual** (White + Black accumulators; output concatenates `[stm, ~stm]`) |
| King buckets | **16** — `(rank/2)*4 + (file/2)` on the perspective-oriented square (4×4 grid) |
| Piece-rel types | 10 — `(PNBRQ) × {own, opp}` |
| Input features | `16 × 64 × 10 = 10240` |
| Hidden width (HL) | **512** |
| Activation | **Clipped ReLU** `[0, CR_MAX]` |
| Output buckets | **8** — `(piece_count - 1) / 4`, clamped 0..7 |
| Orientation | White: `s`; Black: `s ^ 56` |

`feature_index(persp, kingSq, pc, s) = kb*(64*10) + orient(persp,s)*10 + pieceRel`,
`pieceRel = (type-1)*2 + (color==persp ? 0 : 1)`.

### Our `.nnue` on-disk format ("NNU2")
Header then payload, little-endian:

```
u32 MAGIC
u32 VERSION
u32 features   (= 10240)
u32 hl         (= 512)
u32 out_buckets(= 8)
i32 scale
i16 ftBias[512]
i16 ftWeights[10240][512]           # feature-major
i16 outWeights[8][1024]             # per bucket: [0..511]=stm persp, [512..1023]=other persp
i32 outBias[8]
```
Final eval (cp) = `(bias + Σ clippedReLU(acc_stm)·W_own + Σ clippedReLU(acc_opp)·W_opp) / scale`.
The loader (`NNUE::load`) validates `features/hl/out_buckets` match the compiled engine.

---

## 2. Generate self-play data in Bullet format  ✅ implemented

`chess_train gen` now accepts `--format bullet`, emitting bullet's text ingestion
format (one position per line):

```
<FEN> | <score> | <wdl>
```
* `score` — White-relative centipawns (`Sample::eval`).
* `wdl` — `1.0` White win, `0.5` draw, `0.0` Black win (White-relative).

This is exactly what `bulletformat::ChessBoard`'s parser (`FromStr`) expects; it
internally flips to side-to-move orientation and negates score/result for Black.

```sh
# 100k-ish samples, depth-6 labels, 6 random opening plies for diversity
chess_train gen --games 500 --depth 6 --randomplies 6 --format bullet --out data/run1.txt
```
Example output line (verified):
```
r1bqkbnr/pp2pp1p/n2p2p1/2p5/3P1P2/P6P/1PP1P1P1/RNBQKBNR w KQkq c6 0 5 | -9 | 0.0
```

For serious training, generate **tens of millions** of positions (concatenate many
`gen` runs). Higher `--depth` / `--nodes` gives better labels but is slower
(~650 samples/s at depth 6 single-thread here).

### Bullet's packed binary (reference)
For speed, bullet trains on the 32-byte packed `bulletformat::ChessBoard`
(`repr(C)`): `occ:u64, pcs:[u8;16], score:i16, result:u8, ksq:u8, opp_ksq:u8,
extra:[u8;3]`. `pcs` holds 4-bit piece codes (`colour<<3 | pieceType`,
`pieceType 0..5 = PNBRQK`) for each occupied square in `occ` bit order. `train.py
--engine bullet` (and `main.rs`'s `DirectSequentialDataLoader`) reads the text
format directly -- no separate conversion step needed for that path.

---

## 3. (historical) Convert text → packed binary

Some bullet versions require a separate `convert` step to pack the text format
into `bulletformat::ChessBoard`'s binary layout before training. As wired up
today (`tools/nnue_training/bullet_trainer/src/main.rs`'s `DirectSequentialDataLoader`),
the text format from §2 is read directly, so this step is not required for
the code path this project actually uses -- kept here for reference in case a
future bullet_lib version changes that.

---

## 4. Train the network

Bullet is a **library**: you write a small Rust binary describing the net.
`tools/nnue_training/bullet_trainer/src/main.rs` is that binary, and it is
now a **real, complete implementation** (Phase 3 of the distributed-compute
buildout) — not a pseudocode skeleton. It implements:

* a custom `SparseInputType` (`ProductionHalfKp`) emitting feature indices
  in exactly `feature_index()`'s order (§1), written directly against
  bullet_lib's real trait definitions (fetched from
  github.com/jw1912/bullet's `crates/bullet_lib/src/game/inputs.rs`) and
  modeled on the crate's own `Chess768` example, not guessed;
* a custom `OutputBuckets<ChessBoard>` (`ProductionOutputBuckets`)
  implementing our exact `(popcount-1)/4` boundary — deliberately **not**
  bullet's built-in `MaterialCount<N>`, whose `(popcount-2)/ceil(32/N)`
  formula disagrees with ours at several piece counts (see the file's own
  comments for the exact mismatch table);
* real CLI flags (`--data`, `--out`, `--epochs`, `--net-id`, `--threads`,
  `--batch-size`, plus `--features cuda` at the Cargo level) so
  `platform/trainer/train_network.py` can invoke it the same way it invokes
  the CPU reference trainer.

Run it directly with:
```sh
cd tools/nnue_training/bullet_trainer
cargo run --release -- --data data/run1.txt --out checkpoints/run1 --epochs 40
# on a CUDA box:
cargo run --release --features cuda -- --data data/run1.txt --out checkpoints/run1 --epochs 40
```

**Still not compiled or run anywhere in this project** (no Rust toolchain,
no GPU in any environment this has been developed in so far) — see
`main.rs`'s own module doc for the one specific unverified assumption
(whether `pos.our_ksq()`/`opp_ksq()`/`into_iter()` use an absolute or
side-to-move-relative frame) and the exact verification steps to run before
trusting a real training run: `cargo check` first, then cross-check a
handful of encoded positions against `tools/nnue_pipeline/nnue_format.py`
(the same reference implementation `test.py` already uses to verify the CPU
path) before trusting any output for real.

> **Activation/quantisation matches the engine.** Plain **CReLU** (not
> SCReLU) and a post-hoc `/scale`, with `QA=400`/`QB=128` chosen so the
> on-disk `scale` field comes out to a clean integer (`QB`) with no
> conversion-time rounding — see `main.rs`'s own derivation comment.

---

## 5. Convert bullet output → our `.nnue`

Bullet saves its own layout; our engine needs the `NNU2` layout from §1.
`tools/nnue_pipeline/export.py --bullet-quantised <path>/quantised.bin --out
<net>.nnue --scale 128` does this conversion (Phase 3) -- it's the same
export.py already used for the CPU reference path's checkpoints, just fed a
real bullet `quantised.bin` instead. It:

* copies `ftWeights`/`ftBias` directly (same feature-major order, since
  `ProductionHalfKp` emits indices in our `feature_index` order already);
* copies `outWeights` per bucket (`[own(512) | opp(512)]`, matching bullet's
  `[stm(512) | ntm(512)]` per-bucket layout from `main.rs`'s `save_format`);
* uses the `--scale` argument (128 = `QB`, per `main.rs`'s derivation) for
  the on-disk `scale` field;
* round-trips the file it just wrote and prints two sanity evaluations
  before exiting (same as the CPU-path export), so a badly-formed bullet
  export is caught immediately rather than silently producing a broken net.

### 5.1 Sanity check the converted net
```sh
# The bundled/untrained net evaluates start pos ~0 and is material-linear.
# A trained net should give a small non-zero start eval and react to material.
printf 'setoption name EvalFile value net.nnue\nsetoption name Use NNUE value true\nposition startpos\neval\nquit\n' | chess.exe
```
Better: run `tools/nnue_pipeline/test.py --net net.nnue`, which does this
same check plus a full 8-position exact-match verify against the Python
reference implementation, a benchmark, and an Elo match -- the same
discipline used to validate every CPU-path network in this project.

---

## 6. Load and test in the engine

```
setoption name EvalFile value C:\path\to\net.nnue     # loads via NNUE::load (validates arch)
setoption name Use NNUE value true                    # switch eval to NNUE
```
Then A/B the trained net vs. Classical with the built-in match harness / SPRT:
```sh
chess_match --configA nnue --evalA net.nnue --configB current \
            --sprt 0 5 --nodesA 100000 --nodesB 100000 --pgn out.pgn
```
Ship the net as default only after a positive SPRT vs. the classical eval.

---

## 7. Blockers

| Blocker | Impact | Resolution |
|---|---|---|
| **No Rust toolchain** (`rustc`/`cargo` absent) | Cannot build/run bullet here | Install `rustup` on the training machine (or let `platform/trainer/train_network.py`'s automatic GPU-capability check fall back to the CPU reference trainer, which it already does) |
| **No NVIDIA GPU / CUDA** (`nvidia-smi` absent) | Bullet is GPU-oriented; CPU training a 10240×512 net is impractically slow | Train on a CUDA box (cloud GPU is fine), or accept the CPU reference trainer's slower throughput |
| **`main.rs` unverified against real bullet_lib** | The `SparseInputType`/`OutputBuckets` impls are written against real, fetched trait source but have never been through `cargo check` | Run `cargo check`, then cross-check encoded positions against `nnue_format.py` before trusting a real training run (see §4) |

What **is** done and verified on this machine:
* `chess_train gen --format bullet` emits valid `FEN | score | wdl` text (tested).
* The CPU reference path (`tools/nnue_pipeline/train.py --engine reference`)
  is fully implemented, tested end-to-end against the compiled engine, and
  wired into the distributed platform's TRAIN_NETWORK task (see
  `platform/trainer/train_network.py`) with automatic GPU-detection and
  fallback to this same CPU path when no GPU/toolchain is present.
* Engine build clean, `ctest` 6/6.

Nothing in `src/search`, `src/eval`, or NNUE **inference** (`nnue.cpp` accumulator/
output) was modified — only the training data exporter (`train/dataset.*`,
`apps/train_main.cpp`) and the separate, external training tooling.
