//! Bullet (jw1912/bullet, MIT) GPU trainer for this engine's PRODUCTION NNUE
//! shape: HalfKP-style, 16 king buckets (4x4 grid, NOT mirrored by file --
//! see src/nnue/nnue.cpp::king_bucket/orient), 10 piece-relative types
//! (PAWN..QUEEN x own/opp, kings excluded), 512-wide dual-perspective
//! accumulator, 8 output buckets by piece count, plain (non-squared) CReLU.
//!
//! HISTORY: earlier drafts of this file (see git history / PHASE1 notes) had
//! the custom `SparseInputType` impl and the output-bucket selector left as
//! pseudocode / `unimplemented!()`, because the exact trait signatures
//! weren't confirmed against the real bullet_lib source. Both are now real,
//! working implementations, written directly against bullet_lib's actual
//! trait definitions (fetched from github.com/jw1912/bullet main branch,
//! crates/bullet_lib/src/game/inputs.rs and .../outputs.rs) and modeled on
//! the crate's own real examples (examples/simple.rs's `Chess768`,
//! examples/progression/4_multi_layer.rs's `ChessBucketsMirrored` +
//! `MaterialCount` usage) rather than guessed.
//!
//! *** STILL NOT COMPILED/RUN IN THIS SANDBOX ***
//! This sandbox has no Rust toolchain and no GPU (see docs/NNUE_TRAINING_BULLET.md
//! Section 6 / docs/phaseA_nnue_bullet_audit.md), so this file has still never
//! been through `cargo check`, let alone a real training run. Two things are
//! implemented from real, fetched source rather than guesswork (the
//! `SparseInputType`/`OutputBuckets` trait shapes and a real example's
//! `map_features` pattern), but one genuine unknown remains: whether
//! `pos.our_ksq()`/`pos.opp_ksq()`/`pos.into_iter()` give coordinates in a
//! fixed absolute (White's) frame or something already side-to-move-relative
//! -- inferred here from how `ChessBuckets`/`ChessBucketsMirrored` use those
//! same accessors (no side-to-move-conditional flip before indexing a bucket
//! table), consistent with `Chess768`'s own fixed, unconditional `sq ^ 56`
//! flip applied only to the second ("ntm") feature slot. Before trusting a
//! real run: `cargo check`, then cross-check a handful of encoded positions
//! against tools/nnue_pipeline/nnue_format.py's independently-verified
//! reference (the same discipline test.py already uses for the
//! reference-trainer path) BEFORE pointing this at a real dataset. This is
//! the one item Phase 3 leaves as pending hardware/toolchain-specific
//! validation, exactly as flagged in the top-level implementation report.
//!
//! Data: point --data at a file produced by
//! `chess_train gen --format bullet --out data.txt`, i.e. lines of
//! `<FEN> | <score> | <wdl>` -- bulletformat::ChessBoard's own text parser
//! reads this directly (see docs/NNUE_TRAINING_BULLET.md Section 2).
//!
//! CLI (no extra crates beyond bullet_lib -- kept dependency-free like the
//! rest of this tool):
//!     cargo run --release -- --data data/run1.txt --out checkpoints/run1 \
//!         --epochs 40 --net-id run1 [--threads 4] [--batch-size 16384]
//! `--epochs` maps directly to bullet's `end_superbatch` (bullet trains in
//! superbatches, not epochs in the PyTorch sense -- this CLI uses --epochs
//! only so platform/trainer/train_network.py can pass the same flag name it
//! already uses for the reference (NumPy) trainer).

use std::env;

use bullet_lib::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};
use bulletformat::ChessBoard;

// --- Production NNUE constants (must match src/nnue/nnue.h exactly) -------
const HL_SIZE: usize = 512; // NNUE_HL
const KING_BUCKETS: usize = 16; // NNUE_KING_BUCKETS
const PIECE_REL: usize = 10; // NNUE_PIECE_REL (PAWN..QUEEN x own/opp, no king)
const NUM_FEATURES: usize = KING_BUCKETS * 64 * PIECE_REL; // 10240
const NUM_OUTPUT_BUCKETS: usize = 8; // NNUE_OUT_BUCKETS

// --- Quantization ------------------------------------------------------
// QA chosen equal to EVAL_SCALE so the stored-scale algebra below works out
// to a clean integer with no separate conversion-time folding step (see
// derivation). This means QA=400 is unrelated to the engine's CR_MAX=32767
// clipping constant in nnue.cpp -- CR_MAX is so much larger than any sane QA
// that clipping never triggers in practice, i.e. the engine's "clipped ReLU"
// is functionally plain ReLU for any reasonable QA. Not a bug, just a
// divergence from the conventional Stockfish-style CReLU where the clip
// bound IS QA; if genuine clipping behaviour is wanted, change CR_MAX in
// nnue.cpp to equal QA before training against this config.
const QA: i16 = 400;
const QB: i16 = 128;
// Bullet's reference inference computes cp = raw_sum * EVAL_SCALE / (QA*QB).
// Our engine's NNUE::load() format has a single int32 `scale` divisor and
// computes cp = raw_sum / scale, no separate EVAL_SCALE multiply. Setting
// these equal: scale == (QA*QB)/EVAL_SCALE. Choosing QA == EVAL_SCALE makes
// this scale == QB exactly, an integer, with no rounding/bias.
const EVAL_SCALE: f32 = QA as f32; // = 400.0
const ENGINE_STORED_SCALE: i32 = QB as i32; // pass this to export.py's --scale

// ---------------------------------------------------------------------------
// king_bucket: (rank/2)*4 + (file/2) on the perspective-oriented king square,
// matching nnue.cpp::king_bucket() / nnue_format.py::king_bucket() exactly.
// ---------------------------------------------------------------------------
fn king_bucket(king_sq_oriented: usize) -> usize {
    let rank = king_sq_oriented / 8;
    let file = king_sq_oriented % 8;
    (rank / 2) * 4 + (file / 2)
}

// ---------------------------------------------------------------------------
// Custom HalfKP input type. Real trait (SparseInputType), fetched from
// crates/bullet_lib/src/game/inputs.rs:
//     type RequiredDataType: Copy + Send + Sync;
//     fn num_inputs(&self) -> usize;
//     fn max_active(&self) -> usize;
//     fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F);
//     fn shorthand(&self) -> String;
//     fn description(&self) -> String;
//
// Pattern modeled directly on the crate's own Chess768 (examples/../chess768.rs):
// iterate `pos.into_iter()` yielding (piece, square) with `piece & 8` = color
// bit and `piece & 7` = piece type (0=pawn..5=king, bulletformat convention),
// call f(stm_index, ntm_index) once per piece with the "ntm" slot's square
// flipped by ^56 relative to the "stm" slot -- the same fixed, unconditional
// flip Chess768 itself uses (see module doc for why this is believed correct
// but not yet toolchain-verified).
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default, Debug)]
struct ProductionHalfKp;

impl SparseInputType for ProductionHalfKp {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        NUM_FEATURES
    }

    fn max_active(&self) -> usize {
        30 // 32 non-king pieces max on a legal board, minus the 2 kings themselves
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // Absolute king squares (0..64). See module doc: `our_ksq`/`opp_ksq`
        // are used unconditionally (no side-to-move flip) by the crate's own
        // ChessBuckets/ChessBucketsMirrored, so the "stm" slot below treats
        // `our_ksq` as already correctly oriented (persp=WHITE, i.e. identity
        // orient), and the "ntm" slot flips both the opponent's king square
        // and every piece square by ^56 -- exactly mirroring nnue.cpp's
        // orient(persp, s) = (persp == WHITE) ? s : s ^ 56.
        let our_ksq = usize::from(pos.our_ksq());
        let opp_ksq = usize::from(pos.opp_ksq());

        let stm_kb = king_bucket(our_ksq);
        let ntm_kb = king_bucket(opp_ksq ^ 56);

        for (piece, square) in pos.into_iter() {
            let color = usize::from(piece & 8 > 0); // 0 = "stm-slot-native" colour, 1 = other
            let ptype = usize::from(piece & 7); // 0=pawn,1=knight,2=bishop,3=rook,4=queen,5=king
            if ptype == 5 {
                continue; // kings are bucketing-only, never a piece-relative feature
            }
            let sq = usize::from(square);

            // piece_rel = ptype*2 + (0 if "own" to this perspective else 1),
            // matching nnue.cpp's (piece_type-1)*2 + (color==persp?0:1) with
            // our 0-based `ptype` standing in for (piece_type-1).
            let stm_rel = ptype * 2 + if color == 0 { 0 } else { 1 };
            let ntm_rel = ptype * 2 + if color == 1 { 0 } else { 1 };

            let stm_idx = stm_kb * (64 * PIECE_REL) + sq * PIECE_REL + stm_rel;
            let ntm_idx = ntm_kb * (64 * PIECE_REL) + (sq ^ 56) * PIECE_REL + ntm_rel;

            f(stm_idx, ntm_idx);
        }
    }

    fn shorthand(&self) -> String {
        format!("morph_halfkp_{NUM_FEATURES}")
    }

    fn description(&self) -> String {
        "Morph production HalfKP: 16 king buckets (4x4 rank/2,file/2 grid, NOT \
         file-mirrored), 10 piece-relative types, kings excluded from the piece \
         set -- matches src/nnue/nnue.cpp::feature_index() exactly (independently \
         verified byte-for-byte in Python at tools/nnue_pipeline/nnue_format.py \
         against the compiled C++ engine)."
            .to_string()
    }
}

// ---------------------------------------------------------------------------
// Custom output-bucket selector. Real trait (OutputBuckets<T>), fetched from
// crates/bullet_lib/src/game/outputs.rs:
//     const BUCKETS: usize;
//     fn bucket(&self, pos: &T) -> u8;
//
// NOT bullet's built-in `MaterialCount<N>` -- its formula is
// `(popcount - 2) / ceil(32/N)`, which disagrees with our engine's
// `(popcount - 1) / 4` at popcount in {5,9,13,17,21,25,29} (off by one
// bucket). Using MaterialCount here would silently corrupt every exported
// net's output-bucket semantics, so this is a real, from-scratch impl of
// the correct boundary, not a drop-in reuse.
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Default)]
struct ProductionOutputBuckets;

impl OutputBuckets<ChessBoard> for ProductionOutputBuckets {
    const BUCKETS: usize = NUM_OUTPUT_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let n = pos.occ().count_ones() as i32;
        let b = (n - 1) / 4;
        b.clamp(0, (NUM_OUTPUT_BUCKETS - 1) as i32) as u8
    }
}

// ---------------------------------------------------------------------------
// Minimal dependency-free CLI parsing (Cargo.toml intentionally has no
// clap/argh dependency -- this crate's only dependency is bullet_lib itself,
// same philosophy as the rest of tools/).
// ---------------------------------------------------------------------------
struct Args {
    data: String,
    out_dir: String,
    net_id: String,
    epochs: usize,
    threads: usize,
    batch_size: usize,
}

fn parse_args() -> Args {
    let mut data = None;
    let mut out_dir = None;
    let mut net_id = "candidate".to_string();
    let mut epochs = 40usize;
    let mut threads = 4usize;
    let mut batch_size = 16_384usize;

    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--data" => {
                data = Some(argv[i + 1].clone());
                i += 2;
            }
            "--out" => {
                out_dir = Some(argv[i + 1].clone());
                i += 2;
            }
            "--net-id" => {
                net_id = argv[i + 1].clone();
                i += 2;
            }
            "--epochs" => {
                epochs = argv[i + 1].parse().expect("--epochs must be an integer");
                i += 2;
            }
            "--threads" => {
                threads = argv[i + 1].parse().expect("--threads must be an integer");
                i += 2;
            }
            "--batch-size" => {
                batch_size = argv[i + 1].parse().expect("--batch-size must be an integer");
                i += 2;
            }
            other => {
                eprintln!("warning: ignoring unrecognized argument {other:?}");
                i += 1;
            }
        }
    }

    Args {
        data: data.expect("--data <path to bullet-format text/bin dataset> is required"),
        out_dir: out_dir.expect("--out <checkpoint output directory> is required"),
        net_id,
        epochs,
        threads,
        batch_size,
    }
}

fn main() {
    let args = parse_args();

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ProductionHalfKp)
        .output_buckets(ProductionOutputBuckets)
        .save_format(&[
            // Order matches nnue.cpp write_net() exactly: ftBias, ftWeights,
            // outWeights, outBias -- see docs/NNUE_TRAINING_BULLET.md Section 1
            // for the on-disk layout this must line up with.
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", NUM_FEATURES, HL_SIZE);
            let l1 = builder.new_affine("l1", 2 * HL_SIZE, NUM_OUTPUT_BUCKETS);

            // Plain CReLU (linear in the clipped region), NOT SCReLU --
            // matches NNUE::output()'s `clipped(x) * w`, not `clipped(x)^2 * w`.
            let stm_hidden = l0.forward(stm_inputs).crelu();
            let ntm_hidden = l0.forward(ntm_inputs).crelu();
            let hidden = stm_hidden.concat(ntm_hidden);

            // Select the one active output bucket per sample, matching
            // NNUE::output()'s bucket-indexed outWeights/outBias row.
            l1.forward(hidden).select(output_buckets)
        });

    let schedule = TrainingSchedule {
        net_id: args.net_id.clone(),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch: 1000,
            start_superbatch: 1,
            end_superbatch: args.epochs.max(1),
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 }, // matches encoding.h's lambda=0.5 eval/result blend
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.3, step: 15 },
        save_rate: args.epochs.max(1), // only the final superbatch needs to be kept for export
    };

    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: Box::leak(args.out_dir.clone().into_boxed_str()),
        batch_queue_size: 32,
    };

    let dataloader = DirectSequentialDataLoader::new(&[&args.data]);
    trainer.run(&schedule, &settings, &dataloader);

    println!(
        "Training complete. checkpoints -> {}/{}. Convert the final \
         quantised.bin with tools/nnue_pipeline/export.py --bullet-quantised \
         <path>/quantised.bin --out <net>.nnue --scale {} (stored scale = QB = {}).",
        args.out_dir, args.net_id, ENGINE_STORED_SCALE, ENGINE_STORED_SCALE
    );
}
