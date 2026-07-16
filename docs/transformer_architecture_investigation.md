# Architecture Investigation: Chessformer / Transformers vs. the Fastest Route to ~3200 Elo

**Status:** Research only. No code modified, no training run. Elo-optimization queue paused per instruction after recommendation #2 (TT-in-qsearch), checkpointed at 55 games, Elo −6.3 ± 41.0 vs. the recommendation-#1 baseline (statistically flat — see prior session).

**Scope note on "3200 Elo":** this almost certainly means an engine-vs-engine rating (CCRL/CEGT/OpenBench-style self-play Elo), not the Lichess human-comparison scale — the two are not interchangeable. On CCRL, 3200 is a strong-but-not-elite level (many established open-source engines sit in the 3200–3400 band; Stockfish/Lc0-class engines are 3600+). On Lichess's human-blitz scale, DeepMind's searchless-chess transformer already hits ~2895 *without any search*, which sounds close to 3200 but is a different, non-comparable scale. Everything below assumes the CCRL-style engine-rating reading, since that's the scale the engine has been benchmarked on all session (vs. Stockfish/Lc0/berserk/baseline binaries). If the human-scale reading is actually intended, flag it — the recommendation changes.

---

## 1. Audit of the current engine

### 1.1 Evaluation architecture

- **Production default: classical PSQT + material eval** (`src/eval/evaluate.cpp`, `psqt.cpp`). This is what every Elo test this session has actually been measuring.
- **NNUE (`src/nnue/`): implemented but untrained.** HalfKP-style feature set, king-bucketed (16 buckets, 4×4 grid), 10,240 input features (`16 buckets × 64 squares × 10 piece-relative types`), 512-wide dual-perspective accumulator, 8 output buckets by piece count. Weights are currently a **PSQT-equivalent hand-init**, not the product of any training run — the comment in `nnue.h` says this explicitly. This is the single most important audit finding: *the engine's fast evaluator has never been trained on real data.*
- This session's work (Finny-table king-bucket accumulator cache) made NNUE inference ~3x cheaper (NNUE penalty vs. classical dropped from 25.6% to 5.6% at 1 thread) but did nothing to its actual playing strength, because there is no trained signal in it yet.

### 1.2 Search / evaluation call frequency, NPS

- Alpha-beta PVS with qsearch, Lazy SMP (1–256 threads, no shared mutable per-thread state except the TT).
- From this session's profiling pass (gprof, depth-14 bench suite): qsearch accounts for **64–72% of all nodes**; `evaluate()` is called roughly once per node (~1.5–1.7M calls at that bench depth).
- Measured NPS in this sandbox (2-core cloud VM, not representative of real hardware): roughly 1–2M nodes/sec single-threaded, classical eval; NNUE currently costs ~5–10% on top of that post-Finny-cache. Real hardware would be higher; treat these as relative, not absolute, numbers.
- For calibration: Stockfish NNUE runs sub-microsecond per eval and reaches tens of millions of NPS on modern CPUs; Lc0's deep conv/transformer nets need GPU batching and run at far lower raw NPS (thousands, not millions) but are efficient *per node* because MCTS extracts far more value from each expensive evaluation than alpha-beta extracts from a cheap one.

### 1.3 Batching limitations

- **There is no batching anywhere in this engine.** `NNUE::evaluate()` is called exactly once per visited node, inline, in a single serial per-thread traversal. This is by design — it's what makes NNUE-style CPU alpha-beta fast — but it also means the engine has *zero* existing infrastructure for the batched inference that any transformer-scale model needs to be efficient.

### 1.4 Threading model

- Lazy SMP: each thread owns a full deep-copied `Position` (including its own NNUE accumulator and Finny cache) and its own history tables; only the TT is shared, accessed lock-free with 16-bit key verification. Threads are otherwise fully independent — there is no natural "batch point" where multiple threads' evaluation requests could be pooled without adding new synchronization machinery.

### 1.5 Training pipeline (this is the second major finding)

- `src/train/` contains a **separate, disconnected research trainer**: a plain 768-feature (`2 colors × 6 piece types × 64 squares`) dense MLP (`768 → 256 → 32 → 1`, sigmoid win-probability output), trained via a small reference trainer (`trainer.cpp`, always available) or a LibTorch production trainer (`trainer_torch.cpp`, gated on `CHESS_HAVE_TORCH`, **not installed in this sandbox**).
- Its own header comment says it's "kept separate per the modularity constraint" and exists "so a trained net can later be distilled into NNUE" — that distillation step **does not exist yet**. Training this MLP would not train the actual production NNUE.
- Self-play data generation (`src/train/selfplay.cpp`) exists and works — it plays games with the engine's own search and records `(fen, eval, result)` triples — but it's single-machine, not distributed, and again feeds the disconnected 768-feature trainer, not NNUE.
- **One genuinely useful, underused piece of existing infrastructure**: `Dataset::save_bullet()` already exports training samples in the text format consumed by **Bullet** (`jw1912/bullet`), the de facto standard open-source NNUE trainer used by nearly every strong current open-source engine. This means there is already a clean path from "engine-generated or downloaded labeled positions" to "a properly trained net in the engine's actual NNUE format" — it just hasn't been wired up or run.

### 1.6 Inference path

- `NNUE::evaluate()` → `refresh_perspective_cached()` (Finny-table-accelerated incremental accumulator maintenance) → `output()` (SIMD dot product + output-bucket selection). Fully incremental, no batching, no GPU — this is architecturally identical in spirit to Stockfish's NNUE inference path, just untrained.

---

## 2. Research: Chessformer and chess-specific transformers

### 2.1 NNUE lineage (for contrast)

NNUE ("Efficiently Updatable Neural Network") was invented by Yu Nasu in 2018 for shogi (engine YaneuraOu), built on Kunihito Hoki's king-relative piece-square-table idea from Bonanza. Hisayori Noda ported it to a Stockfish dev branch in early 2020; the official Stockfish team merged it in August 2020 (Stockfish 12), producing one of the largest single-version strength jumps in the engine's history. Every strong open-source engine has used some NNUE variant since. Current Stockfish uses **two** nets selected by material imbalance: a big net (1024-wide hidden layer, `HalfKAv2_hm` + "FullThreats" features) and a small net (128-wide, `HalfKAv2_hm` only) for clearly won/lost positions. *(Sources: Wikipedia "Efficiently updatable neural network"; chessprogramming.org/NNUE and /Stockfish_NNUE; Stockfish blog "Introducing NNUE Evaluation"; DeepWiki official-stockfish/Stockfish NNUE page.)*

### 2.2 DeepMind: "Grandmaster-Level Chess Without Search" / "Amortized Planning with Large-Scale Transformers: A Case Study on Chess" (NeurIPS 2024; arXiv:2402.04494)

- A **270M-parameter decoder-only transformer**, trained purely by supervised learning (no self-play, no search at inference) on **ChessBench**: 10M human games, 15 billion `(position, action-value)` datapoints labeled by **Stockfish 16**.
- Achieves a **Lichess blitz Elo of 2895 against humans**, playing with *zero search* — pure feed-forward policy. Outperforms AlphaZero's raw policy/value nets and GPT-3.5-turbo-instruct on chess tasks.
- The paper's own conclusion is important: "a remarkably good approximation of Stockfish's search-based algorithm can be distilled into large-scale transformers via supervised learning, [but] perfect distillation is still beyond reach." I.e. even DeepMind, at this scale, could not fully replace search — they approximated it.
- **Fully open**: code Apache-2.0, model weights (9M/136M/270M checkpoints) CC-BY-4.0, dataset partly CC0 (Lichess-derived) / partly CC-BY. Repo: `google-deepmind/searchless_chess`, checkpoints and full ChessBench dataset downloadable today via a script in the repo. This is directly, legally usable by this project right now, at zero training cost.

### 2.3 Chessformer: A Unified Architecture for Chess Modeling (ICLR 2026; arXiv:2605.19091)

- Authors: Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson — **CSSLab, University of Toronto** (the Maia-chess team), not the Lc0 core team or DeepMind.
- **Encoder-only transformer**: the 64 board squares are tokens; adds a novel positional scheme called **Geometric Attention Bias (GAB)** that encodes chess-specific square geometry into attention, plus an attention-based source→destination policy head (naturally suited to representing chess moves as square-pairs).
- Three demonstrated results: (1) integrated into **Leela Chess Zero**, added **>100 Elo** over the prior architecture and contributed to tournament wins over Stockfish in computer chess events — but this is Chessformer *running inside Lc0's GPU-batched MCTS*, not inside an alpha-beta engine; (2) powers **Maia-3**, reaching 57.1% human-move-prediction accuracy with under a quarter of the parameters of the prior state of the art; (3) its square-token design gives cleaner interpretability than convolutional nets.
- License/availability of trained Chessformer/Maia-3 weights specifically was not confirmed from public sources in this pass (Maia-1 weights are GPL-3.0, Maia-2 is MIT — Maia-3/Chessformer weight licensing should be checked directly against the CSSLab repos, `CSSLab/maia3`, before any use).

### 2.4 Leela Chess Zero's transformer nets (BT2/BT3/BT4)

- BT4: 191M parameters, 15 transformer layers, 1024-wide hidden state, 32 attention heads, 64-token input (one per square), plus a "smolgen" module for dynamic per-position attention biasing, and auxiliary heads (categorical value distribution, "future move" prediction 2 plies out) borrowed from BT3.
- **Runs exclusively inside Lc0's GPU-accelerated MCTS.** Lc0 was built from day one around GPU batched inference — MCTS naturally produces many simultaneous leaf-expansion requests that can be batched into one GPU forward pass; this is precisely what alpha-beta's single-threaded-per-node traversal does *not* provide.
- License: Lc0 itself is GPL-3.0; published network weights have historically been released openly but under separate terms from the engine code — check per-network before using as a prior/oracle in a differently-licensed project.

### 2.5 The core technical finding

**No chess transformer architecture in the literature has an NNUE-equivalent efficient-update mechanism.** NNUE's speed comes specifically from its first layer being sparse and linear, so a single piece move only requires a handful of add/subtract operations against a small accumulator. Self-attention is inherently dense and global — every token attends to every other token — so there is no known cheap incremental update when one square changes; a transformer eval is a full forward pass, every time, for every position. Combined with the batching point above, this is why every published strong chess transformer (searchless_chess, Chessformer/Maia-3, Lc0's BT-series) is either **used without any search at all** (pure feed-forward policy) or **paired with GPU-batched MCTS** — never plugged into a serial, millions-of-nodes-per-second CPU alpha-beta loop like this engine's.

---

## 3. Research: crowdsourced / distributed compute infrastructure

Available now, no volunteers needed:

- **Lichess open database** (`database.lichess.org`, CC0): 394M+ Stockfish-evaluated positions, 6M+ rated puzzles, monthly PGN game archives. Free, immediate, no attribution required.
- **DeepMind's `searchless_chess` checkpoints + ChessBench** (Apache-2.0 code / CC-BY weights & dataset): 9M/136M/270M pretrained transformers and 15B Stockfish-16-labeled datapoints, downloadable today.
- **Bullet** (`jw1912/bullet`, MIT, Rust): the standard fast NNUE trainer used by most current top open-source engines. GPU-accelerated, runs fine on a single rented consumer/cloud GPU; this engine already exports its native format via `Dataset::save_bullet()`.
- Your own machine(s), or a handful of rented cloud CPU/GPU instances — sufficient to run all of Phase 0–3 below without any outside contributors.

Requires attracting contributors (not available today, explicitly separated per your instruction):

- **Fishtest** (`official-stockfish/fishtest`): Stockfish's own volunteer-CPU distributed SPRT-testing framework, running since 2013 (Gary Linscott). FastAPI server + worker client model, GSPRT statistics, χ² anomalous-worker detection. Tightly coupled to Stockfish's own fork workflow — the *software pattern* is copyable, the *infrastructure* is not directly reusable for a different engine.
- **OpenBench** (`AndyGrant/OpenBench`, GPL, Django + Cutechess): a general-purpose Fishtest-alike that already hosts SPRT testing for many independent open-source engines (Berserk, Ethereal, Koivisto, and others) on a public instance at `chess.grantnet.us`. This is the most directly reusable option: either (a) self-host your own instance today with zero volunteers (immediately available, just your own compute), or (b) apply to the public instance once the engine meets its "Requirements for Public Engines" (existing published engine, some public track record) to inherit real volunteer CPU time — that path requires the engine to already look credible externally, i.e. it's a *later* step, not a Day-1 one.
- **Leela Chess Zero's distributed training network**: client downloads the latest net, self-plays on volunteer CPU/GPU, uploads games to a central server that periodically retrains and republishes weights. At peak this generated on the order of 1M games/day and has produced 2.5B+ cumulative self-play games since inception. This is Lc0's *own* infra for training Lc0's *own* nets — not something this project can plug into directly, though the client/server self-play pattern is a proven, copyable blueprint if this project ever wanted its own volunteer self-play network.
- **BOINC**: a general volunteer-computing platform (SETI@home's lineage), used for scientific compute (math, climate, medicine, astrophysics). No existing chess-engine project was found on it, and its embarrassingly-parallel batch-job model is a poor fit for head-to-head SPRT testing compared to purpose-built tools like Fishtest/OpenBench. **Recommendation: don't build on BOINC for this** — it would mean building chess-specific work distribution and validation from scratch, when OpenBench already does this and already has a chess-engine-testing community around it.

**Bottom line on distributed compute**: everything needed for Phases 0–3 below is available today on your own hardware or a few rented cloud instances. Real crowdsourced scale (Fishtest/OpenBench-public/Lc0-style) is a genuine multiplier on *iteration throughput*, but growing a volunteer base takes weeks-to-months of community engagement — that's a parallel, non-blocking track, not something on the critical path to 3200.

---

## 4. Evaluation of the six options

| # | Route | Elo upside | NPS impact | Training compute | Testing compute | Impl. time | Integration difficulty | Batching req. | CPU inference practicality | AB compatibility | Biggest risk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **Train the existing (untrained) NNUE properly** | **High** (+150 to +400 est., precedented by Stockfish's own 2020 NNUE jump) | ~0 (already measured: 5–10% vs. classical, post-Finny-cache) | Low-moderate (Bullet, hours on one GPU or longer on CPU) | Normal SPRT | **Low** (days) | **Low** (inference code already works; only weights are missing) | None | Excellent (already proven in this exact codebase) | Perfect (native fit) | Data/label pipeline bugs, training-config tuning |
| 4/5 | **Offline transformer teacher → distilled into NNUE** (using free DeepMind checkpoints) | Medium-high, additive on top of the above (+50–150 est.) | **Zero** (transformer never runs at inference) | Moderate (batched offline GPU inference over millions of positions — no training of a new transformer required, checkpoints already exist) | Normal SPRT | Low-medium (days) | **Low** (production engine code doesn't change at all) | Ideal — this is the one place batching is free and natural | N/A (doesn't run in production) | **Perfect** (zero interference with alpha-beta) | Label/domain mismatch between transformer-teacher style and this engine's own search; easily mitigated by blending with self-play/Lichess data |
| — | **Continue classical search/eval tuning** (this session's queue: pin-aware legality, etc.) | Low-medium per item (+10–40 each, as measured this session) | Positive (that's the point) | None | Normal SPRT | Low per item, but many items needed | Low | None | Excellent | Perfect | Diminishing returns per item; needs many iterations to add up |
| 3 | **Transformer policy for move ordering, NNUE eval unchanged** | **Low** — this session's own profiling already showed 90–92% first-move cutoff rate; ordering is near-ceiling already | Negative (per-node inference cost) | Moderate | Normal SPRT, but slower games | Medium | Medium-high (same batching mismatch, narrower scope) | Needed for internal nodes to matter | Poor without batching | Moderate — ordering errors are "soft" (worse pruning, not wrong values), more forgiving than eval replacement, but still bottlenecked | Cost likely exceeds the small remaining headroom |
| 2 | **NNUE + transformer only at selected high-value nodes** | Medium, unproven for this engine | Small-to-moderate if genuinely rare, but any non-cached call costs 10³–10⁶× an NNUE call | Moderate-high (needs a working transformer) | Normal SPRT, higher variance | Medium-high | Medium-high (new inference-queue architecture across Lazy SMP threads) | Cross-thread batching is possible but nontrivial to build | Workable only if very rare | Partial — real risk of search instability from eval discontinuities between neighboring nodes | No established recipe for "when is a node worth it"; genuine research problem |
| 1 | **Replace NNUE entirely with a tiny Chessformer at every node** | **Uncertain-to-negative at matched time controls** — eval quality gain is real but NPS collapse (est. 10³–10⁶×, since nothing incrementally updates a transformer) likely costs more depth than it buys in eval quality | **Catastrophic** | High (need a working small transformer, more data/GPU than NNUE) | Normal SPRT but each game is drastically slower, ballooning wall-clock test time | **High** (months; no NNUE-equivalent incremental trick exists to invent) | **High** (accumulator model, threading model, TT-eval-caching all need rethinking, and still wouldn't close a 3–6 order-of-magnitude gap) | Fundamentally required, fundamentally incompatible with serial per-node AB traversal | Poor beyond toy sizes | **Poor** — this is the core architectural mismatch | Ending up *weaker* than a properly trained NNUE at the same time control, because alpha-beta strength scales strongly with depth/NPS |
| 6 | **Any better modern architecture that fits high-NPS CPU alpha-beta** | The honest answer from the evidence: **iterate on NNUE itself**, not a new family — bigger/deeper nets, richer feature sets (e.g. Stockfish's `HalfKAv2_hm`), dual big/small net selection, WDL-head training (Bullet supports this natively) | Small, tunable | Low-moderate | Normal SPRT | Low-medium | Low | None | Excellent | Perfect | None new — this is proven, incremental work in the same family that's already fast |

### Ranked by expected Elo gain ÷ wall-clock development time

1. **Train the existing untrained NNUE** (Bullet + Lichess CC0 + self-play) — highest ratio by a wide margin; this is free strength sitting unused in the repo today.
2. **Offline transformer-teacher distillation into that same NNUE** (Option 4/5, using the already-published, free DeepMind checkpoints) — nearly as good a ratio, stacks directly on #1, zero production-code risk.
3. **Continue the classical search/eval optimization queue** (this session's approach) — solid, proven, low-risk, but each item is a small increment; still worth doing, just not the fastest single lever.
4. **NNUE architecture upgrades** (Option 6: bigger net, richer features, dual net, WDL training) — good ratio, but logically comes *after* #1 (train what exists before growing it).
5. **Transformer policy for move ordering** (Option 3) — low ratio; the thing it improves is already near-optimal in this engine.
6. **Transformer at selected high-value nodes** (Option 2) — low-to-moderate ratio, real research risk, no established recipe.
7. **Replace NNUE with an online transformer** (Option 1) — lowest, likely negative, ratio. Evidence-backed conclusion: **wrong route for this engine and this goal.**

---

## 5. Final recommendation

**Chessformer and transformers are the wrong tool for the online, per-node evaluation role in this engine — the evidence is unambiguous on that point (Section 2.5, and every published strong chess transformer either runs search-free or is paired with GPU-batched MCTS, never serial CPU alpha-beta). They are, however, an excellent and essentially free tool for improving this engine's training data offline.**

**The fastest technically plausible route to ~3200 Elo is:** finish training the NNUE evaluator that already exists in this codebase but has never been trained, using the industry-standard open trainer (Bullet) this repo already has an export path for, enriched with free labels from DeepMind's already-published, permissively-licensed transformer checkpoint — then keep running the classical search/eval optimization loop already underway this session in parallel. No new architecture, no new search paradigm, no volunteers required to start.

### Phased execution plan

**Phase 0 — Testing infrastructure & baseline (1–3 days)**
- Task: stand up self-hosted OpenBench (or scale the custom Elo harness already built this session) for repeatable SPRT testing; establish a tight-margin baseline for the current engine (classical eval, recommendations #1+#2 applied).
- Compute workload: several thousand fast games (e.g. 5–10s+0.1s) for a ±10–15 Elo baseline margin.
- Compute source: your own machine(s) / rented cloud CPU instances. No volunteers.
- Success metric: reproducible SPRT results; baseline Elo established with a tight confidence interval.
- Failure metric: results not reproducible, or game throughput too low to reach adequate sample size.
- Trigger to next phase: infra operational and baseline recorded.

**Phase 1 — Train the existing NNUE for real (the big lever)**
- Task: wire self-play (`generate_selfplay`) + downloaded Lichess CC0 evaluations into `Dataset::save_bullet()`'s existing export; configure Bullet for this engine's exact HalfKP feature layout (10,240 features, 16 king buckets, 512-wide, 8 output buckets); train; load the result via the existing `NNUE::load()`; verify correctness with the debug sampled-assertion path already built during the Finny-cache work.
- Compute workload: data collection (Lichess CC0 is a direct download; self-play runs on your own CPU); training (hours on one GPU, or longer CPU-only — this net is Stockfish-small-net scale, not big-net scale).
- Compute source: Lichess CC0 database (free, immediate) + one rented cloud GPU for a few hours, or CPU-only if budget-constrained.
- Wall-clock: 3–7 days.
- Success metric: SPRT-confirmed Elo gain over the classical-only baseline (precedent suggests +150 to +400); NNUE enabled by default with no correctness regressions.
- Failure metric: trained net performs at or below the classical baseline, or introduces non-determinism.
- Trigger to next phase: NNUE beats classical eval by a statistically significant SPRT margin.

**Phase 2 — Offline transformer-teacher label enrichment**
- Task: run the free DeepMind `searchless_chess` checkpoint (270M, or a smaller one if inference time matters) as a **batched, offline** oracle over a large, diverse position set (opening books + Phase-1 self-play positions + Lichess positions); blend the resulting labels into the Phase-1 training set; retrain via Bullet.
- Compute workload: batched GPU inference over ~10–50M positions — no transformer *training*, just inference on an already-trained, already-downloadable model.
- Compute source: one rented cloud GPU, a few hours to a day.
- Wall-clock: 3–5 days.
- Success metric: SPRT-confirmed further Elo gain over the Phase-1 net.
- Failure metric: no measurable gain, or a regression — in that case, revert to the Phase-1 net and record this as a documented negative result; don't let it block further progress.
- Trigger to next phase: gain confirmed or conclusively ruled out within the time-box.

**Phase 3 — Resume the classical search/eval optimization queue (parallelizable with 1–2)**
- Task: continue exactly the process already running this session (recommendation #3 and beyond: pin-aware legality/check-info, further profiling-driven changes), now using Phase 0's larger-scale testing infra.
- Compute workload: thousands of SPRT games per candidate.
- Compute source: same self-hosted testing infra.
- Wall-clock: ongoing.
- Success metric: each change SPRT-confirmed positive before merging (same discipline as this session).
- Failure metric: a change fails to show gain — revert, as already practiced.
- Trigger: continuous; feeds the same testing loop as Phases 0–2 rather than gating on them.

**Phase 4 — Community/distributed scale-up (optional, explicitly not on the critical path)**
- Task: once solo/rented-cloud SPRT throughput — not model architecture — becomes the binding constraint, stand up a public-facing OpenBench instance and begin outreach (engine-programming Discord, TalkChess, r/chess) to attract volunteer CPU testers, mirroring the OpenBench/Fishtest model.
- Compute workload: N/A — this is community building, not a compute task.
- Compute source: volunteers (not available today; must be recruited).
- Wall-clock: weeks to months to reach meaningful scale.
- Success metric: multiple independent, verified contributors submitting test workers.
- Failure metric: no organic uptake — continue on solo/rented compute; this does not block reaching 3200, it only accelerates iteration once the bigger levers (Phases 1–3) are exhausted.
- Trigger: pursue only after Phases 1–3 gains plateau.

---

## Sources

- [Grandmaster-Level Chess Without Search (arXiv:2402.04494)](https://arxiv.org/html/2402.04494v1)
- [google-deepmind/searchless_chess (GitHub — code, weights, ChessBench dataset, licenses)](https://github.com/google-deepmind/searchless_chess)
- [Amortized Planning with Large-Scale Transformers: A Case Study on Chess — NeurIPS 2024 paper PDF](https://arxiv.org/pdf/2402.04494)
- [Chessformer: A Unified Architecture for Chess Modeling (arXiv:2605.19091)](https://arxiv.org/abs/2605.19091)
- [Chessformer — ICLR 2026 poster page](https://iclr.cc/virtual/2026/poster/10011702)
- [Chessformer — OpenReview](https://openreview.net/forum?id=2ltBRzEHyd)
- [CSSLab/maia-chess, maia2, maia3 (GitHub — licenses, model families)](https://github.com/CSSLab)
- [Transformer Progress — Leela Chess Zero blog (BT3/BT4 architecture)](https://lczero.org/blog/2024/02/transformer-progress/)
- [How well do Lc0 networks compare to the greatest transformer network from DeepMind? — Leela Chess Zero blog](https://lczero.org/blog/2024/02/how-well-do-lc0-networks-compare-to-the-greatest-transformer-network-from-deepmind/)
- [LeelaChessZero/lc0 (GitHub — GPL-3.0, network downloads)](https://github.com/LeelaChessZero/lc0)
- [Efficiently updatable neural network — Wikipedia](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)
- [NNUE — Chessprogramming wiki](https://www.chessprogramming.org/NNUE)
- [Stockfish NNUE — Chessprogramming wiki](https://www.chessprogramming.org/Stockfish_NNUE)
- [Introducing NNUE Evaluation — Stockfish blog](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)
- [NNUE Neural Network Evaluation — official-stockfish/Stockfish DeepWiki](https://deepwiki.com/official-stockfish/Stockfish/5.1-nnue-neural-network-evaluation)
- [official-stockfish/fishtest (GitHub — Stockfish's distributed SPRT framework)](https://github.com/official-stockfish/fishtest)
- [Statistical Methods and Algorithms in Fishtest — Stockfish docs](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
- [AndyGrant/OpenBench (GitHub — GPL, distributed SPRT framework)](https://github.com/AndyGrant/OpenBench)
- [OpenBench — Chessprogramming wiki](https://www.chessprogramming.org/OpenBench)
- [Leela Chess Zero — Wikipedia (distributed training history)](https://en.wikipedia.org/wiki/Leela_Chess_Zero)
- [lichess.org open database (CC0 games, evaluations, puzzles)](https://database.lichess.org/)
- [lichess-org/database (GitHub)](https://github.com/lichess-org/database)
- [jw1912/bullet (GitHub — MIT, standard open NNUE trainer)](https://github.com/jw1912/bullet)
- [BOINC: A Platform for Volunteer Computing (arXiv:1903.01699)](https://arxiv.org/pdf/1903.01699)
