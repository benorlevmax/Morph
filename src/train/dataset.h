// dataset.h - Training dataset: self-play samples (position, eval, outcome).
//
// This subsystem is research/training only and is fully decoupled from NNUE and
// search (it merely *uses* the engine to label positions).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace chess::train {

// One training example. `eval` and `result` are from White's point of view.
struct Sample {
    std::string fen;          // position
    std::int16_t eval = 0;    // engine search score in centipawns (White POV)
    std::int8_t  result = 0;  // game outcome: +1 White win, 0 draw, -1 Black win
    // Search-instability signals (see search.h's SearchResult::scoreSwing /
    // bestMoveChanges) -- optional, additive training-data quality/difficulty
    // signal. Both default to 0, which is indistinguishable from "a position
    // resolved in a single iteration" -- callers that care about "was this
    // signal actually recorded" should track that separately (e.g. dataset
    // format version); this struct itself makes no such claim.
    std::int16_t scoreSwing      = 0;
    std::uint8_t bestMoveChanges = 0;
};

class Dataset {
public:
    void add(const Sample& s) { samples_.push_back(s); }
    void clear() { samples_.clear(); }

    std::size_t size() const { return samples_.size(); }
    const Sample& operator[](std::size_t i) const { return samples_[i]; }
    const std::vector<Sample>& samples() const { return samples_; }

    bool save(const std::string& path) const;   // binary, versioned
    bool load(const std::string& path);
    bool append_to(const std::string& path) const;

    // Export in Bullet's text ingestion format, one sample per line:
    //   <FEN> | <score> | <wdl>
    // where score is White-relative centipawns and wdl is 1.0 (White win),
    // 0.5 (draw) or 0.0 (Black win). This text is consumed by bullet's
    // `convert` utility to produce the packed `bulletformat::ChessBoard` binary.
    bool save_bullet(const std::string& path) const;

    // Extended text export for distributed data-generation/mining specifically
    // (NOT consumed by bullet's `convert` utility -- that tool expects exactly
    // the 3-field format `save_bullet` produces, so this is a separate,
    // additive method rather than a change to save_bullet's format). Same
    // 3 fields plus two trailing columns carrying the instability signal:
    //   <FEN> | <score> | <wdl> | <scoreSwing> | <bestMoveChanges>
    bool save_bullet_ext(const std::string& path) const;

private:
    std::vector<Sample> samples_;
};

} // namespace chess::train
