# Morph (C++)

## Project goal
From-scratch classical chess engine (C++20) with a dual evaluation architecture:
classical eval + PSQT (current default) and NNUE (HalfKP, dual-perspective,
designed to eventually replace it as the default). NNUE is trained by the
distributed compute platform's `TRAIN_NETWORK` task, which wraps
`tools/nnue_pipeline/` (CPU reference) or `tools/nnue_training/bullet_trainer`
(Rust/GPU via bullet_lib, CUDA or ROCm) -- both produce the engine's real
binary `.nnue` format directly. See `README.md` for the phase-by-phase status
table.

## Note: repo split
This repo used to also contain a separate Python AlphaZero-style research
pipeline (self-play + MCTS + PyTorch net). That code, its checkpoints, and its
training data now live in `python-engine/` (own `CLAUDE.md` there). The two
projects are independent — no code or model sharing between them. Shared
reference/opponent binaries (`stockfish/`, `lc0.exe`, `berserk.exe`,
`baseline.exe`, `cutechess-1.5.1-win64/`, the `791556.pb*` Lc0 net) stay here
at the repo root since both sides' benchmarking scripts reference them by
this absolute path.

## Key files
| Path | Purpose |
|------|---------|
| `src/core/` | bitboard, position, movegen, Zobrist, perft |
| `src/search/` | iterative deepening, PVS, null-move, LMR, probcut, qsearch, TT, lazy SMP |
| `src/eval/` | classical evaluation + PSQT |
| `src/nnue/` | NNUE evaluator (stub — untrained net, not the default) |
| `src/uci/` | UCI protocol loop |
| `src/syzygy/` | tablebase probing |
| `src/train/` | Self-contained reference trainer (`trainer.cpp`, flat MLP) -- a dependency-free correctness check for the encode/train/checkpoint plumbing, not the production NNUE trainer. The real one is `tools/nnue_pipeline/` + `tools/nnue_training/bullet_trainer/` (see `platform/docs/TRAINING.md`) |
| `src/match/` | engine-vs-engine match runner |
| `src/apps/` | CLI entry points (perft, train, etc.) |
| `tests/*.cpp` | perft / FEN / Zobrist / PGN / eval / search tests (CTest) |
| `CMakeLists.txt` | build config |

## 