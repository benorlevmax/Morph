// opening_book.cpp - see opening_book.h for the format/design rationale.
#include "book/opening_book.h"
#include "core/movegen.h"

#include <algorithm>
#include <fstream>
#include <random>

namespace chess {

namespace {

// Big-endian byte-level readers/writers, matching the style already used by
// src/io/book.cpp and src/nnue/nnue.cpp -- explicit byte I/O means the file
// format never depends on struct packing, alignment, or host endianness.
void wr16(std::ostream& os, std::uint16_t v) { os.put(char(v >> 8)); os.put(char(v & 0xFF)); }
void wr32(std::ostream& os, std::uint32_t v) {
    os.put(char(v >> 24)); os.put(char(v >> 16)); os.put(char(v >> 8)); os.put(char(v));
}
void wr64(std::ostream& os, std::uint64_t v) {
    for (int i = 7; i >= 0; --i) os.put(char((v >> (8 * i)) & 0xFF));
}
std::uint16_t rd16(const unsigned char* p) { return std::uint16_t((p[0] << 8) | p[1]); }
std::uint32_t rd32(const unsigned char* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16)
         | (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}
std::uint64_t rd64(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}

std::uint16_t code_for_move(Move m) {
    const int promoCode = (m.type_of() == PROMOTION) ? (int(m.promotion_type()) - int(KNIGHT) + 1) : 0;
    return encode_book_move(m.from_sq(), m.to_sq(), promoCode);
}

} // namespace

std::uint16_t encode_book_move(Square from, Square to, int promoCode) {
    return std::uint16_t((from & 0x3F) | ((to & 0x3F) << 6) | ((promoCode & 0x7) << 12));
}

bool move_matches_code(Move m, std::uint16_t code) {
    // Castling and en passant need no dedicated flag here: castling is a
    // king moving two files (from/to alone is never ambiguous with a normal
    // king move, which can only be one square), and en passant is a pawn
    // moving diagonally onto an otherwise-empty square (no other legal move
    // shares that exact from/to pair in that position). Promotion is the
    // only case that needs an explicit extra field, since e.g. e7e8 with a
    // pawn could promote to four different pieces.
    return code_for_move(m) == code;
}

bool OpeningBook::looks_like_book_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    unsigned char hdr[4];
    if (!f.read(reinterpret_cast<char*>(hdr), 4)) return false;
    return rd32(hdr) == BOOK_MAGIC;
}

bool OpeningBook::load(const std::string& path) {
    entries_.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    unsigned char hdr[16];
    if (!f.read(reinterpret_cast<char*>(hdr), 16)) return false;
    const std::uint32_t magic   = rd32(hdr);
    const std::uint32_t version = rd32(hdr + 4);
    const std::uint64_t count   = rd64(hdr + 8);
    if (magic != BOOK_MAGIC || version != BOOK_VERSION) return false;

    entries_.reserve(std::size_t(count));
    unsigned char rec[BOOK_RECORD_BYTES];
    for (std::uint64_t i = 0; i < count; ++i) {
        if (!f.read(reinterpret_cast<char*>(rec), BOOK_RECORD_BYTES)) {
            entries_.clear();
            return false;   // truncated/corrupt file: fail cleanly, don't half-load
        }
        Entry e;
        e.hash       = rd64(rec);
        e.code       = rd16(rec + 8);
        e.evalCp     = int(std::int16_t(rd16(rec + 10)));
        e.depth      = int(rec[12]);
        e.visits     = rd32(rec + 13);
        e.confidence = int(rec[17]);
        e.frequency  = int(rd16(rec + 18));
        entries_.push_back(e);
    }
    std::stable_sort(entries_.begin(), entries_.end(),
                     [](const Entry& a, const Entry& b) { return a.hash < b.hash; });
    return loaded();
}

bool OpeningBook::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    wr32(f, BOOK_MAGIC);
    wr32(f, BOOK_VERSION);
    wr64(f, std::uint64_t(entries_.size()));

    for (const Entry& e : entries_) {
        wr64(f, e.hash);
        wr16(f, e.code);
        wr16(f, std::uint16_t(std::int16_t(e.evalCp)));
        f.put(char(std::uint8_t(std::clamp(e.depth, 0, 255))));
        wr32(f, e.visits);
        f.put(char(std::uint8_t(std::clamp(e.confidence, 0, 100))));
        wr16(f, std::uint16_t(std::clamp(e.frequency, 0, 65535)));
    }
    return bool(f);
}

void OpeningBook::add(Key hash, const BookMove& mv) {
    Entry e;
    e.hash       = hash;
    e.code       = code_for_move(mv.move);
    e.evalCp     = mv.evalCp;
    e.depth      = mv.depth;
    e.visits     = mv.visits;
    e.confidence = mv.confidence;
    e.frequency  = mv.frequency;
    entries_.push_back(e);
}

void OpeningBook::finalize() {
    std::stable_sort(entries_.begin(), entries_.end(),
                     [](const Entry& a, const Entry& b) { return a.hash < b.hash; });
}

std::vector<BookMove> OpeningBook::probe(const Position& pos) const {
    std::vector<BookMove> out;
    if (entries_.empty()) return out;

    const Key key = pos.key();
    auto lo = std::lower_bound(entries_.begin(), entries_.end(), key,
                               [](const Entry& e, Key k) { return e.hash < k; });
    if (lo == entries_.end() || lo->hash != key) return out;

    // Resolve each stored (from,to,promo) code against this position's real
    // legal moves -- this is where a stored code becomes a fully-typed Move
    // (CASTLING/EN_PASSANT/PROMOTION flags correctly set), and also where an
    // entry that is no longer legal here (shouldn't happen for a correctly
    // generated book, but a hand-edited/corrupt file could) is silently
    // dropped rather than returned, exactly like src/io/book.cpp's
    // Book::probe() already does for the legacy format.
    MoveList legal;
    generate(pos, legal, LEGAL);

    for (auto it = lo; it != entries_.end() && it->hash == key; ++it) {
        for (const auto& sm : legal) {
            if (move_matches_code(sm.move, it->code)) {
                BookMove bm;
                bm.move       = sm.move;
                bm.evalCp     = it->evalCp;
                bm.depth      = it->depth;
                bm.visits     = it->visits;
                bm.confidence = it->confidence;
                bm.frequency  = it->frequency;
                out.push_back(bm);
                break;
            }
        }
    }

    // Strongest eval first (this is what randomness==0 / "pick best" relies on).
    std::stable_sort(out.begin(), out.end(),
                     [](const BookMove& a, const BookMove& b) { return a.evalCp > b.evalCp; });
    return out;
}

Move select_book_move(const std::vector<BookMove>& candidates, int randomness, std::uint64_t seed) {
    if (candidates.empty()) return Move::none();
    if (randomness <= 0) return candidates.front().move;

    randomness = std::min(randomness, 100);
    // Tolerance window grows with randomness: at randomness=100, moves up to
    // 200cp worse than the best are eligible; at randomness=1, essentially
    // only moves tied with the best are eligible. This keeps "some
    // randomness" from ever meaning "might pick a genuinely bad move".
    const int toleranceCp = 2 * randomness;
    const int bestEval = candidates.front().evalCp;

    std::vector<const BookMove*> pool;
    std::uint64_t totalWeight = 0;
    for (const auto& c : candidates) {
        if (bestEval - c.evalCp > toleranceCp) continue;
        const std::uint64_t w = c.frequency > 0 ? std::uint64_t(c.frequency)
                                                 : std::max<std::uint64_t>(1, c.visits);
        pool.push_back(&c);
        totalWeight += w;
    }
    if (pool.empty()) return candidates.front().move;   // shouldn't happen (best is always in-window)

    // Seeded, not global: the same (book, position, randomness) always
    // reproduces the same pick, satisfying the "deterministic mode must
    // reproduce the same book" requirement without needing a separate mode --
    // determinism falls out of seeding by position rather than wall-clock/PID.
    std::mt19937_64 rng(seed ^ 0x9E3779B97F4A7C15ULL);
    std::uint64_t r = totalWeight > 0
        ? std::uniform_int_distribution<std::uint64_t>(0, totalWeight - 1)(rng)
        : 0;
    for (const BookMove* c : pool) {
        const std::uint64_t w = c->frequency > 0 ? std::uint64_t(c->frequency)
                                                  : std::max<std::uint64_t>(1, c->visits);
        if (r < w) return c->move;
        r -= w;
    }
    return pool.back()->move;
}

} // namespace chess
