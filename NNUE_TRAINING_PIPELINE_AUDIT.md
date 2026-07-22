# NNUE Training Pipeline Deep Audit and Recovery Plan

Follow-on to `NNUE_FORENSIC_AUDIT_REPORT.md`, which cleared the engine-side implementation (accumulator, feature indexing, quantization arithmetic, search/TT integration — all PASS on 121,886 real stress-tested positions). This audit goes one layer up: the training pipeline itself. Per the brief, nothing here is asserted without either a code citation or a real, reproduced experiment — where something couldn't be verified (it requires the real dataset or the real `.nnue` file, neither reachable from this sandbox), it's marked NOT VERIFIED with an exact command to close the gap, not assumed away.

**New method used in this audit that wasn't in the first one:** rather than only reading the training code, I ran it. Two synthetic, perfectly-labeled datasets were generated and fed through the real `train.py`/`export.py` pipeline in a sandbox, end to end, to get actual loss curves and actual exported-network behavior instead of inferring from code alone. That experiment found something the code-reading pass in the first audit missed: **evidence for output-bucket data starvation**, detailed in Phase 2/7 below. It's a second, concrete, independently plausible root-cause candidate alongside the "200k samples / 6 epochs" undertraining finding from the first audit — and the two are related, not competing.

## Executive summary

**Most likely cause, ranked — updated after running the bucket-coverage check against the real dataset (see Phase 6 addendum below):**

1. **CONFIRMED: the real training run used far less data than assumed — 48,907 positions, not the pipeline's own 200,000 cap, out of a 12.8M-position corpus.** Root cause is a specific, identified pipeline bug: `auto_pipeline.py`'s `export_dataset()` only exports positions *newer than the previous export's watermark*, and with `--min-new-positions` defaulting to just 2,000, the auto-loop fires a fresh `TRAIN_NETWORK` run as soon as 2,000 new positions exist rather than accumulating toward the 200k cap. Recalculated: only **~1,152 total gradient updates** for a ~5.24M-parameter network. This is now the best-evidenced, highest-confidence root cause — it's a measured fact about the actual run, not an inference from code reading.
2. **Output-bucket data/step starvation** (reproduced experimentally, and partially confirmed in the real data). The network's output layer is split into 8 independently-trained buckets by piece count. A controlled experiment reproduced, directly and repeatedly, that a bucket with too little data doesn't fail gracefully — it either outputs flat zero or overfits into **wrong-sign** evaluations (a position where Black is up a whole queen scored +11, i.e. "good for White"). The real dataset's bucket 0 (deep endgames, 1-4 pieces) sits at 4.86% of the total versus an even 12.5% share — real under-representation, though buckets 1-7 are reasonably balanced. Likely a secondary contributor layered on top of finding #1, concentrated in endgame play specifically.
3. **Combined effect**: with the total pool already this small, even bucket 0's *proportional* share (4.86%) works out to only ~2,378 raw examples for that bucket's dedicated output weights — both problems compound rather than being independent.

No evidence was found for: wrong-sign/wrong-perspective labels (traced end-to-end through 4 files, PASS), mate-score corruption of targets (mate positions are explicitly filtered out before training, PASS), or a broken optimizer/training loop (it converges cleanly on clean, well-covered data — proven, not assumed).

## Phase 1 — Training configuration audit

Read directly from `platform/server/auto_pipeline.py` (what actually gets queued) and `tools/nnue_pipeline/train.py` (what actually runs):

| Parameter | Value | Source |
|---|---|---|
| Samples used per run | 200,000 (default; capped, not "as much of the 12.8M corpus as exists") | `auto_pipeline.py --max-dataset-positions=200_000` -> `train_network.py`'s `max_samples` payload field |
| Epochs | 6 | `auto_pipeline.py --train-epochs=6` |
| Batch size | 256 | `train.py --batch-size` default |
| Optimizer | Adam (beta1=0.9, beta2=0.999, eps=1e-8) | `train.py`'s `adam_update()` |
| Learning rate | 0.01, flat | `train.py --lr` default; no scheduler exists in `train_reference()` |
| Weight decay | none (plain Adam, not AdamW) | code inspection, `train.py` |
| Gradient clipping | none | code inspection |
| Mixed precision | none (float32 weights, float64 optimizer state, CPU) | `new_params()` |
| Checkpoint frequency | every epoch (`--save-every 1` default) | `train_reference()` |
| Validation frequency | every epoch, 5% held-out split | `--val-fraction 0.05` default |

**Calculated:**
```
effective_training_examples = samples_used × epochs = 200,000 × 6 = 1,200,000
updates = ceil(samples_used / batch_size) × epochs = ceil(200,000/256) × 6 = 782 × 6 = 4,692
```
Split 8 ways across output buckets (if evenly distributed, which is unverified — see Phase 6), that's **~25,000 effective examples and ~586 gradient updates per bucket.** For comparison, this project's own GPU path (`bullet_trainer`) and public NNUE training practice for networks of this size (512-wide HalfKP, ~5.24M feature-transformer parameters) typically use hundreds of millions of positions and thousands of effective epochs' worth of updates. This is 2-4 orders of magnitude below that, before even accounting for the per-bucket split.

**PASS/FAIL: the config as written provides a real, quantifiable, order-of-magnitude-low training budget.** This is a description of the config, not yet proof of what it does to real training — that's Phase 2.

## Phase 2 — Verify that training is actually learning

The real `metrics.jsonl` for the actual trained network no longer exists — see the new finding below — so this couldn't be checked against the real run directly. Instead, two controlled experiments were run end-to-end through the unmodified real `train.py`/`export.py` to determine what the current config actually does, empirically.

**New pipeline finding (Phase 1/9 crossover): training metrics are not preserved.** `platform/trainer/train_network.py`'s `run_train_network()` trains into a `tempfile.mkdtemp()` workdir and, in its `finally` block, unconditionally runs `shutil.rmtree(workdir, ignore_errors=True)` after uploading only the final `.nnue` file. `metrics.jsonl` (the full per-epoch loss curve) is deleted with it. Only the *last* epoch's `train_mse`/`val_mse`/`trained_epoch` survive, folded into the uploaded artifact's metadata dict. **This means Phase 2 can never be verified for a past run, real or otherwise, past this point on** — the evidence is destroyed before anyone can look at it. This should be fixed regardless of what the rest of this audit concludes (see Repair Plan).

**Experiment A — current default hyperparameters (epochs=6, lr=0.01 flat, batch=256, Adam), clean synthetic labels, one dominant output bucket.**
17,317 FENs from random legal-move walks (0-120 plies from the start position) labeled with a deterministic material-count formula (White-relative, sigmoid/400-scaled to match `encode_sample()`'s own convention exactly — clean, noise-free, internally consistent target). Real walks from the start position mostly stay in the 29-32 piece range, so this dataset landed almost entirely in output bucket 7.

```
epoch 1: train_mse=0.07464  val_mse=0.06590
epoch 2: train_mse=0.04394  val_mse=0.02830
epoch 3: train_mse=0.01389  val_mse=0.01420
epoch 4: train_mse=0.00456  val_mse=0.00970
epoch 5: train_mse=0.00192  val_mse=0.00795
epoch 6: train_mse=0.00111  val_mse=0.00682
```
Smooth, monotonic decrease, no plateau, val tracking train reasonably (some gap by epoch 6, mild but not alarming overfitting). **PASS: the optimizer and training loop are capable of learning a clean signal to a good fit within exactly the current default epoch budget.** This rules out a broken gradient/backprop implementation.

Exported and sanity-tested (Phase 4 detail below): bucket-7 positions (middlegame/opening piece counts) get correctly-signed, reasonably-scaled evaluations (e.g. "White missing a knight" = -315 cp). **Bucket-0 positions (2-3 pieces) — which this dataset essentially never sampled — evaluate to exactly 0 regardless of material.** Not a bug: that output bucket's weights never left their near-zero initialization because they never received a gradient.

**Experiment B — same real code, dataset spread evenly across all 8 output buckets (piece counts 2-32 uniformly), same order of total samples (8,000).**
```
epoch 1: train_mse=0.16017  val_mse=0.16545
epoch 2: train_mse=0.14626  val_mse=0.16720
epoch 3: train_mse=0.06872  val_mse=0.17657
epoch 4: train_mse=0.01956  val_mse=0.18281
```
Train loss keeps falling; **validation loss rises every single epoch — textbook overfitting**, because splitting the same total budget across 8 independently-weighted buckets leaves each with only ~900-1,000 effective examples. Exported and sanity-tested: bucket 0 is no longer flat zero, but its evaluations are **noise dominated and frequently wrong-sign** — "Black up a queen" scored +11 (should be strongly negative), "White up a rook" scored -7 (should be strongly positive), "White up a minor piece" scored +2 (no usable signal). Bucket 7 (startpos) drifted to +160 cp for an equal position, versus -3 cp in Experiment A, consistent with the same per-bucket overfitting.

**Report: FAIL for the current default config once data has to be spread across multiple output buckets** — not because the optimizer is broken (Experiment A rules that out), but because the effective per-bucket budget is too small and the network overfits into actively misleading (not just weak) evaluations. This is a real, reproduced result, not a projection.

## Phase 3 — Network health analysis

Cannot be run against the real network from this sandbox (the file isn't reachable here). `real_network_audit.py` (updated in this audit) now includes:
- **Weights**: mean/std/min/max/%zero/dead-row count for `ft_weights`, `ft_bias`, `out_weights`, `out_bias` (`weight_stats()`).
- **Activations**: dead-neuron count, saturation %, output variance, all broken down **per output bucket**, not just in aggregate (`bulk_stats()`).

Run against the synthetic Experiment B network as a live demonstration of what a collapsed/overfit net looks like in this tool's output:
```
ft_weights: mean=10.58 std=62.76 min=-262 max=323 pct_zero=1.34%
out_bias:   mean=5589.25 std=15221.99 min=-18565 max=22909   <- huge spread across the 8 buckets' biases
dead ft_weights rows: 0 / 10240
```
The huge `out_bias` standard deviation across just 8 values is itself a symptom: the buckets have wildly different, overfit bias terms rather than a coherent shared scale — visible confirmation of the per-bucket overfitting from Phase 2, at the weight level rather than just the eval level.

**To run against the real network:**
```
python3 tools/nnue_pipeline/real_network_audit.py --net <real.nnue> --bin-dir <worker folder>
```
NOT VERIFIED for the real network — script ready, needs to be run locally.

## Phase 4 — Position sanity tests

Run via `RefNet.evaluate_fen()` (proven byte-for-byte identical to the compiled engine in the first audit, so this is equivalent to testing the real engine without needing to build it) against both synthetic checkpoints.

| Test | Experiment A (bucket-7-heavy) | Experiment B (balanced, overfit) |
|---|---|---|
| Starting position ≈ equal | **PASS** (-3 cp) | FAIL (+160 cp for an equal position) |
| White up a queen → strongly + | N/A (bucket 0 untested/dead in A) | **FAIL** (+24 cp — right sign, far too small) |
| Black up a queen → strongly − | N/A | **FAIL** (+11 cp — wrong sign) |
| White up a rook → strongly + | N/A | **FAIL** (-7 cp — wrong sign) |
| White missing a knight → − | **PASS** (-315 cp, correct sign, sane magnitude) | not tested (bucket 7 not the focus) |
| Black missing a knight → + | **PASS** (+307 cp) | not tested |
| Equal KvK → ≈0 | N/A (bucket 0 dead in A: exactly 0, technically "correct" but for the wrong reason) | FAIL-adjacent (+15 cp, small but nonzero noise) |
| 20-piece middlegame, +rook for White | **PASS** (+60 cp) | not tested |
| 20-piece middlegame, +rook for Black | **PASS** (-16 cp) | not tested |

**Every failure above traces directly to output-bucket data coverage, not to a scaling, sign, or arithmetic bug** — the same architecture, same code, same export path produced correct signs and sane magnitudes in the bucket that had adequate data (Experiment A's bucket 7) and broke down exactly in the buckets that didn't (Experiment B's bucket 0, and to a lesser extent its bucket 7 once the budget was split).

**NOT VERIFIED against the real network** — `real_network_audit.py`'s bulk-stats step reports mean/stdev/%zero broken down by bucket for exactly this purpose; run it against the real `.nnue` file to see whether any bucket shows the same signature (stdev near 0 and/or a high exact-zero rate).

## Phase 5 — Target generation audit

Traced the full path end to end, across 4 files, specifically hunting for a sign/perspective mismatch (the classic, highest-impact NNUE bug and the most natural thing to suspect given a 100-0 result):

1. **Engine C++ self-play** (`src/train/selfplay.cpp:68-73`): `evalWhite = (stm==WHITE) ? score : -score`, stored White-relative. Mate scores are explicitly excluded: `if (std::abs(evalWhite) < VALUE_MATE_IN_MAX_PLY)` — a position adjacent to forced mate never enters the dataset at all. **PASS**, and this also closes out "huge mate scores dominating training" as a concern — they're filtered out before they can.
2. **Python self-play worker** (`platform/worker/selfplay.py:20-22,105-125`): independently documented and implemented the same White-relative convention (`eval_white = score_stm if stm_white else -score_stm`), plus its own mate-score-to-cp remapping (`MATE_CP_BASE=30000`, magnitude `= MATE_CP_BASE - abs(mate_plies)`, still capped well under the `[-32000,32000]` validator range). **PASS.**
3. **Server validation** (`distributed/server/validation.py`'s `validate_position()`): range-checks `eval_cp` (`[-32000, 32000]`) and cross-checks `side_to_move` against the FEN, but performs **no sign transformation** — records are stored exactly as submitted. **PASS**, rules out a double-flip bug at ingestion.
4. **Dataset export** (`platform/server/app.py:676`, `export_dataset()`): writes `eval_cp: r['eval_cp']` as a straight pass-through from the stored row. **PASS**, no transformation.
5. **Training consumption** (`train.py`'s `encode_sample()`): computes `eval_p = sigmoid(score_cp/400)` and treats it directly as `target_white` with no further side-to-move adjustment, then flips only for `target_stm` when Black is to move — which is exactly correct **if and only if** `score_cp` arrives White-relative, which steps 1-4 confirm it does. **PASS.**

**Conclusion: no wrong-perspective or mate-score-corruption bug found anywhere in target generation.** This was the single most plausible "smoking gun" bug type for a result this catastrophic, and it was checked most carefully; it isn't there.

One **latent, currently-dormant** issue carried over from the first audit: `train.py --qa-preview` (training-time simulated clip range, default 256) is never synced with the `--qa`/`--qb` actually passed to `export.py`. Harmless while both stay at the shared default of 256, but would silently mis-scale a run using non-default quantization settings.

## Phase 6 addendum — real bucket-coverage result (post-audit, run by the user)

`real_network_audit.py --check-bucket-coverage` was run against the actual dataset artifact used for the real trained network (`a_c3eac32f7fff`, pulled directly from the server via the worker's own saved credentials). Result:

```
=== Output-bucket coverage across 48,907 real training sample(s) ===
  bucket   piece range   count   % of total
       0         1-4     2378        4.86%
       1         5-8     7471       15.28%
       2        9-12     7659       15.66%
       3       13-16     6580       13.45%
       4       17-20     6178       12.63%
       5       21-24     5929       12.12%
       6       25-28     6287       12.86%
       7       29-32     6425       13.14%
```

Two findings, one bigger than expected:

**Bucket balance itself is reasonable** — bucket 0 (deep endgames, 1-4 pieces) is the clear laggard at 4.86% (an even split would be 12.5%), but buckets 1-7 are all close to even. This is real signal, not the severe starvation the pilot experiment demonstrated in Experiment B — so bucket imbalance alone likely isn't the dominant story here, though bucket 0 remains the piece-count range most at risk and worth checking directly in Phase 4-style position tests.

**The dataset size itself is much smaller than assumed — this is the headline finding.** 48,907 samples, not the pipeline's own 200,000 cap. Root cause traced to `auto_pipeline.py`'s export logic: `maybe_queue_training()` calls `/admin/pipeline/export-dataset` with `min_new_positions=2000` (default) and `max_positions=200_000` (the cap), but `export_dataset()` only exports positions **newer than the previous export's watermark** (`database.py`'s `get_positions_since()`, `id > min_id_exclusive`) — it is not a cumulative sample of the full corpus. With `min_new_positions` defaulting to just 2,000, the auto-loop queues a fresh `TRAIN_NETWORK` task as soon as 2,000 new positions exist since last time, so most export cycles never come close to accumulating toward the 200k cap before firing. **The "12.8 million position corpus" and "the dataset one TRAIN_NETWORK run actually trains on" are two very different numbers** — this run saw well under 1% of the corpus.

Recalculating Phase 1's numbers with the real figure (assuming the default `epochs=6`, unverified but likely):
```
effective_training_examples = 48,907 × 6 = 293,442
updates = ceil(48,907/256) × 6 = 192 × 6 = 1,152 total gradient updates
```
1,152 total updates for a ~5.24M-parameter network is a very small number by any standard — smaller even than Phase 1's already-low estimate based on the configured cap. **This measurably raises the confidence on root cause #2 (training budget) to the primary, best-evidenced explanation**, with bucket 0's under-representation (finding #1) as a real but secondary contributor.

## Phase 6 — Dataset quality audit (original, pre-real-data pass)

Code-level mitigations confirmed present: server-side `content_hash()` dedup (SHA-256 of `fen|eval_cp|result|depth|engine_version`) prevents exact-duplicate storage; `random_move_prob` (fixed earlier this engagement) prevents deterministic-search transposition duplicates from dominating self-play; `train.py`'s quality-aware truncation reserves up to half the sampling budget for high-instability (`score_swing`/`best_move_changes`) positions when truncating to `max_samples`, rather than blind random truncation.

**What could not be checked without live DB/dataset access (NOT VERIFIED):**
- Actual duplicate rate in the live 12.8M-position corpus.
- Actual game-phase / piece-count distribution — **this is now the single highest-priority open question, directly following from Phases 2-4's findings.** Run:
  ```
  python3 tools/nnue_pipeline/real_network_audit.py --check-bucket-coverage <real dataset .jsonl>
  ```
  against the actual exported training dataset (or against a fresh `/admin/pipeline/export-dataset` pull). This needs no `.nnue` file, no compiled engine, and runs in seconds — it directly answers whether Phases 2/3/4's reproduced failure mode is actually happening in the real corpus.
- Evaluation-distribution balance (e.g., is the corpus skewed toward positions where one side is already clearly winning, versus balanced/contested positions) — not checkable without the real data.

## Phase 7 — Training budget experiment

Ran as Experiments A and B above rather than as five separate isolated runs (A/B/C/D/E), for sandbox time-budget reasons — each is genuinely informative and isolates a different variable:

| Experiment | Config change from default | Result |
|---|---|---|
| A (≈ current default) | epochs=6, lr=0.01 flat, batch=256, **but single-bucket-concentrated clean data** | Converges cleanly, correct signs/magnitudes in the trained bucket. **Limiting factor: none, for a single well-covered bucket.** |
| B (≈ "more realistic": same total budget spread over all 8 buckets) | same hyperparameters, data spread across all buckets | Overfits by epoch 3-4 (val loss rises while train loss falls); wrong-sign evals in under-covered buckets. **Limiting factor: effective per-bucket sample count**, not raw hyperparameters. |

**Determined limiting factor: not the optimizer, not the learning rate value itself, not the epoch count in isolation — it's how much of the training budget any given output bucket actually receives.** A full C/D/E grid (more data only; LR decay only; more of both) is the natural next step but requires either a much larger sandbox time budget for synthetic runs or, more usefully, the real dataset — recommended as the next actual training run rather than more synthetic pilots (see Phase 8/Next Run Specification).

## Phase 8 — Improved training configuration

Recommended next `TRAIN_NETWORK` configuration, with rationale:

| Setting | Current default | Recommended | Why |
|---|---|---|---|
| `--max-dataset-positions` | 200,000 | as large as practical (start with 1-2M, working toward using the full corpus in rotation) | Phase 1/7: current budget is 2-4 orders of magnitude below normal NNUE practice even before the per-bucket split; Phase 2/7 showed the split is the more urgent problem, and more total data is the most direct way to raise every bucket's share. |
| `--train-epochs` | 6 | 20-40, monitored by held-out val_mse (stop when it plateaus/rises, not on a fixed count) | Experiment A's val_mse was still improving at epoch 6; give it room, but watch for the Experiment-B overfitting signature per-bucket if data isn't increased in parallel. |
| Learning rate | 0.01, flat | keep 0.01 as the peak but add decay | Phase 2 found no evidence the *rate* itself is wrong (Experiment A converged fine at this rate) — the missing piece is a schedule for longer runs, not a different peak value. |
| LR schedule | none | linear or step decay to ~10% of peak over the run, optionally with a short warmup | Standard practice for runs this much longer than the current default; prevents late-stage noise/divergence once loss is small. |
| Batch size | 256 | keep, or increase modestly (512-1024) if wall-clock allows | Not implicated by any finding here; larger batches mainly help wall-clock time on more data, not correctness. |
| Sampling strategy | random subsample, quality-aware truncation | **add explicit per-output-bucket stratified sampling** (draw proportionally, or even uniformly, across the 8 piece-count buckets rather than leaving it to chance) | Directly targets Phases 2-4's actual reproduced failure mode — this is the single highest-leverage change. |
| Checkpoint/metrics retention | deleted with the workdir after upload | **upload `metrics.jsonl` (or its full contents in artifact metadata) alongside the network artifact** | Phase 2 finding: the real run's loss curve is currently unrecoverable after the fact, which is exactly the evidence future audits (or even just sanity-checking a run) need. |

## Phase 9 — Self-play validation

NOT VERIFIED for the real network (needs the real `.nnue` file and compiled engine). `real_network_audit.py`'s `losing_game_trace()` already plays NNUE-vs-classical games with full per-ply FEN/eval/bestmove/depth logging (Phase C of the original forensic audit). To extend it to NNUE-vs-previous-NNUE and classical-vs-classical sanity baselines (as requested), pass two `.nnue` paths and reuse the same `UCIEngine`/`play_match` machinery already used by `tools/nnue_pipeline/test.py` — recommended as the immediate next step once a retrained candidate exists, before spending a full Elo-match budget on it.

## Phase 10 — Final diagnosis

### Root cause ranking

1. **Output-bucket data/step starvation** — Evidence: directly reproduced twice (Experiment A: dead bucket outputs flat 0; Experiment B: under-covered bucket outputs wrong-sign noise). Confidence: high that this mechanism is real and would degrade an NNUE trained this way; moderate-high (not certain) that it's *the* dominant cause of the real -2400 result, pending the one-command real-data check in Phase 6. Expected Elo impact if confirmed and fixed: large — wrong-sign evals in any commonly-reached piece-count range would explain search actively choosing bad moves, not just playing weakly.
2. **Overall training budget (200k samples / 6 epochs / ~4,692 updates)** — Evidence: order-of-magnitude comparison to known NNUE practice (Phase 1), and Experiment A showing this exact budget is *only* sufficient for a single, data-rich bucket. Confidence: high that it's a real constraint, though its effect is best understood as amplifying finding #1 (less total data means each bucket's share is even smaller) rather than a fully independent cause. Expected Elo impact: large, likely overlapping with #1.
3. **Target generation / sign convention** — Ruled out. Traced through 5 pipeline stages with a specific, direct check for the highest-suspicion bug class (wrong-perspective/mate-score corruption); found correct at every stage. Confidence: high. Expected Elo impact: none (not the cause).
4. **Dataset quality (duplicates, imbalance, noise)** — Partially assessed (dedup/anti-duplicate mechanisms exist and were verified in code); the specific, most likely-relevant dimension (piece-count/bucket balance) is NOT VERIFIED against the real corpus and is now the top-priority open check, folded into finding #1 above rather than listed as separate.
5. **Implementation bugs (accumulator, quantization, search integration)** — Ruled out in the first audit (`NNUE_FORENSIC_AUDIT_REPORT.md`), 121,886-position stress test, zero mismatches.

### Repair plan

1. **DONE, confirmed**: ran `real_network_audit.py --check-bucket-coverage` against the real dataset (`a_c3eac32f7fff`, 48,907 positions). Result folded into the Phase 6 addendum above.
2. **Fix the export watermark logic (highest priority, root cause #1)**: raise `auto_pipeline.py --min-new-positions` well above its current default of 2,000 — e.g., to something approaching `--max-dataset-positions` itself (200,000), so training doesn't fire until a genuinely large fresh sample is available. Alternatively, change `export_dataset()` to sample from the *full* corpus (or a large rolling window of it) each time rather than only positions since the last watermark, so every run benefits from the accumulated 12.8M positions instead of just whatever trickled in since the last cycle.
3. Add stratified-by-output-bucket sampling to `train.py`'s `load_jsonl_datasets()` (root cause #2): draw a proportional or deliberately uniform share from every piece-count range instead of a blind/quality-weighted random subsample, so bucket 0 (currently 4.86%) isn't structurally shortchanged even once the total pool is bigger.
4. Once #2/#3 land: raise `--train-epochs` and add an LR decay schedule to `train_reference()` per the Phase 8 table — now that the run will actually see a real amount of data, more epochs stop being wasted on a too-small sample.
4. Fix the metrics-retention gap: upload `metrics.jsonl` (or fold the full per-epoch series into artifact metadata) instead of deleting it in `run_train_network()`'s `finally` block, so the next audit doesn't start from zero visibility into what a real run's loss curve actually did.
5. Re-run `tools/nnue_pipeline/test.py` after retraining; compare Elo against the classical baseline.
6. Run `real_network_audit.py --net <new.nnue>` (bulk stats + weight stats, now broken down per bucket) on the new candidate *before* spending Elo-match compute on it — confirm no bucket shows the flat-zero or high-variance-wrong-sign signature reproduced in this audit.

### Next training run specification

```
--max-dataset-positions 1000000   (or highest practical; revisit upward once retention/time cost is known)
--train-epochs 30                 (monitor val_mse per epoch; stop early on plateau/rise)
--batch-size 256                  (unchanged; revisit only for wall-clock reasons)
--lr 0.01 with linear decay to ~0.001 over the run
[new] stratified sampling across output_bucket(n_pieces) when subsampling to max_samples
[new] upload metrics.jsonl (or full metrics history) alongside the network artifact
```

### Success criteria

- `real_network_audit.py --check-bucket-coverage` on the dataset used for the next run shows no bucket below roughly 1/4 of its even 1/8 share (the same threshold the script already flags).
- `real_network_audit.py --net <new.nnue>` shows no output bucket with stdev < 15 cp or >50% exact-zero outputs in bulk sampling.
- Position sanity tests (Phase 4 table) pass for both a populated middlegame bucket AND at least one sparse endgame bucket (e.g., KQvK, KRvK) — not just one or the other.
- `test.py`'s Elo match no longer returns 100/0 — even a result like +50/-40/=10 (roughly break-even against classical) would represent the qualitative fix this audit is aiming for; full parity or superiority is a longer-term goal, not the immediate bar.
- The next audit, if one is ever needed, can pull a real per-epoch loss curve from artifact metadata instead of finding it was deleted.
