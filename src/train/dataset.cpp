// dataset.cpp - Binary dataset serialization.
#include "train/dataset.h"

#include <fstream>

namespace chess::train {

namespace {
constexpr std::uint32_t MAGIC   = 0x44415431;   // "DAT1"
// VERSION 1: fen/eval/result only (original format).
// VERSION 2: adds scoreSwing/bestMoveChanges (search-instability signal,
// additive -- see Sample in dataset.h). load() still reads VERSION 1 files
// unchanged (defaulting the new fields to 0), so no existing dataset is
// invalidated by this change.
constexpr std::uint32_t VERSION = 2;

void w32(std::ostream& o, std::uint32_t v) { o.write(reinterpret_cast<char*>(&v), 4); }
bool r32(std::istream& i, std::uint32_t& v) { return bool(i.read(reinterpret_cast<char*>(&v), 4)); }

void write_record(std::ostream& o, const Sample& s) {
    std::uint16_t len = std::uint16_t(s.fen.size());
    o.write(reinterpret_cast<char*>(&len), 2);
    o.write(s.fen.data(), len);
    o.write(reinterpret_cast<const char*>(&s.eval), 2);
    o.write(reinterpret_cast<const char*>(&s.result), 1);
    o.write(reinterpret_cast<const char*>(&s.scoreSwing), 2);
    o.write(reinterpret_cast<const char*>(&s.bestMoveChanges), 1);
}
} // namespace

bool Dataset::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    w32(f, MAGIC); w32(f, VERSION);
    w32(f, std::uint32_t(samples_.size()));
    for (const Sample& s : samples_) write_record(f, s);
    return bool(f);
}

bool Dataset::save_bullet(const std::string& path) const {
    std::ofstream f(path, std::ios::trunc);   // text
    if (!f) return false;
    for (const Sample& s : samples_) {
        // result is +1 White win / 0 draw / -1 Black win (White POV); bullet
        // wants White-relative wdl in {1.0, 0.5, 0.0}. score is already White POV.
        const char* wdl = s.result > 0 ? "1.0" : (s.result < 0 ? "0.0" : "0.5");
        f << s.fen << " | " << int(s.eval) << " | " << wdl << '\n';
    }
    return bool(f);
}

bool Dataset::save_bullet_ext(const std::string& path) const {
    std::ofstream f(path, std::ios::trunc);   // text
    if (!f) return false;
    for (const Sample& s : samples_) {
        const char* wdl = s.result > 0 ? "1.0" : (s.result < 0 ? "0.0" : "0.5");
        f << s.fen << " | " << int(s.eval) << " | " << wdl << " | "
          << int(s.scoreSwing) << " | " << int(s.bestMoveChanges) << '\n';
    }
    return bool(f);
}

bool Dataset::append_to(const std::string& path) const {
    // Read existing count, append records, rewrite header count.
    std::vector<Sample> existing;
    Dataset tmp;
    if (tmp.load(path)) existing = tmp.samples_;

    Dataset merged;
    merged.samples_ = existing;
    for (const Sample& s : samples_) merged.samples_.push_back(s);
    return merged.save(path);
}

bool Dataset::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::uint32_t magic, version, count;
    if (!r32(f, magic) || !r32(f, version) || !r32(f, count)) return false;
    // Accept VERSION 1 (original, no instability fields) and VERSION 2
    // (adds scoreSwing/bestMoveChanges) -- every existing .dat file on disk
    // is a VERSION 1 file, and this must keep loading it unchanged rather
    // than invalidating previously-generated training data.
    if (magic != MAGIC || (version != 1 && version != 2)) return false;

    samples_.clear();
    samples_.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint16_t len;
        if (!f.read(reinterpret_cast<char*>(&len), 2)) return false;
        Sample s;
        s.fen.resize(len);
        if (!f.read(s.fen.data(), len)) return false;
        if (!f.read(reinterpret_cast<char*>(&s.eval), 2)) return false;
        if (!f.read(reinterpret_cast<char*>(&s.result), 1)) return false;
        if (version >= 2) {
            if (!f.read(reinterpret_cast<char*>(&s.scoreSwing), 2)) return false;
            if (!f.read(reinterpret_cast<char*>(&s.bestMoveChanges), 1)) return false;
        }
        // version == 1: scoreSwing/bestMoveChanges keep their default (0),
        // which is the correct "not recorded" value for old data.
        samples_.push_back(std::move(s));
    }
    return true;
}

} // namespace chess::train
