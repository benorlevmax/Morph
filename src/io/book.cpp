// book.cpp - Opening book implementation.
#include "io/book.h"
#include "io/pgn.h"
#include "core/movegen.h"
#include "core/bitboard.h"

#include <algorithm>
#include <fstream>
#include <istream>
#include <map>
#include <random>

namespace chess {

namespace {

// 781-entry random array for Polyglot-style hashing. Filled deterministically;
// replace with the canonical Polyglot constants for third-party .bin interop.
Key Random64[781];
bool randomReady = false;

// Polyglot index layout.
constexpr int RandomPiece    = 0;     // [0,768)
constexpr int RandomCastle   = 768;   // [768,772)
constexpr int RandomEnPassant= 772;   // [772,780)
constexpr int RandomTurn     = 780;

int polyglot_kind(Piece pc) {
    // pawn b/w=0/1, knight 2/3, bishop 4/5, rook 6/7, queen 8/9, king 10/11.
    return 2 * (int(type_of(pc)) - 1) + (color_of(pc) == WHITE ? 1 : 0);
}

// Big-endian readers.
std::uint16_t rd16(const unsigned char* p) { return std::uint16_t((p[0] << 8) | p[1]); }
std::uint32_t rd32(const unsigned char* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16)
         | (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}
Key rd64(const unsigned char* p) {
    Key v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}
void wr16(std::ostream& os, std::uint16_t v) { os.put(char(v >> 8)); os.put(char(v & 0xFF)); }
void wr32(std::ostream& os, std::uint32_t v) {
    os.put(char(v >> 24)); os.put(char(v >> 16)); os.put(char(v >> 8)); os.put(char(v));
}
void wr64(std::ostream& os, Key v) {
    for (int i = 7; i >= 0; --i) os.put(char((v >> (8 * i)) & 0xFF));
}

// Can the side to move legally capture en passant (Polyglot includes the ep
// file only in that case)?
bool ep_capturable(const Position& pos) {
    const Square ep = pos.ep_square();
    if (ep == SQ_NONE) return false;
    const Color us = pos.side_to_move();
    return pawn_attacks(~us, ep) & pos.pieces(us, PAWN);
}

} // namespace

void book_init() {
    if (randomReady) return;
    std::mt19937_64 rng(0xB00C1234ABCD5678ULL);
    for (Key& k : Random64) k = rng();
    randomReady = true;
}

Key book_key(const Position& pos) {
    Key key = 0;
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        const Piece pc = pos.piece_on(s);
        if (pc == NO_PIECE) continue;
        const int idx = 64 * polyglot_kind(pc) + 8 * rank_of(s) + file_of(s);
        key ^= Random64[RandomPiece + idx];
    }

    const int cr = pos.castling_rights();
    if (cr & WHITE_OO)  key ^= Random64[RandomCastle + 0];
    if (cr & WHITE_OOO) key ^= Random64[RandomCastle + 1];
    if (cr & BLACK_OO)  key ^= Random64[RandomCastle + 2];
    if (cr & BLACK_OOO) key ^= Random64[RandomCastle + 3];

    if (ep_capturable(pos))
        key ^= Random64[RandomEnPassant + file_of(pos.ep_square())];

    if (pos.side_to_move() == WHITE)
        key ^= Random64[RandomTurn];

    return key;
}

std::uint16_t encode_polyglot(const Position& pos, Move m) {
    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Polyglot encodes castling as the king moving onto its own rook's square.
    if (m.type_of() == CASTLING) {
        const Rank r = rank_of(from);
        to = (file_of(to) == FILE_G) ? make_square(FILE_H, r)
                                     : make_square(FILE_A, r);
    }

    int promo = 0;   // 0 none, 1 N, 2 B, 3 R, 4 Q
    if (m.type_of() == PROMOTION)
        promo = int(m.promotion_type()) - int(KNIGHT) + 1;

    return std::uint16_t(file_of(to) | (rank_of(to) << 3)
                       | (file_of(from) << 6) | (rank_of(from) << 9)
                       | (promo << 12));
}

bool Book::load(const std::string& path) {
    entries_.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    unsigned char buf[16];
    while (f.read(reinterpret_cast<char*>(buf), 16)) {
        BookEntry e;
        e.key    = rd64(buf);
        e.move   = rd16(buf + 8);
        e.weight = rd16(buf + 10);
        e.learn  = rd32(buf + 12);
        entries_.push_back(e);
    }
    // Polyglot books are sorted by key; ensure it for binary search.
    std::stable_sort(entries_.begin(), entries_.end(),
                     [](const BookEntry& a, const BookEntry& b) { return a.key < b.key; });
    return loaded();
}

Move Book::probe(const Position& pos, bool pickBest) const {
    if (entries_.empty()) return Move::none();

    const Key key = book_key(pos);
    auto lo = std::lower_bound(entries_.begin(), entries_.end(), key,
                               [](const BookEntry& e, Key k) { return e.key < k; });

    // Gather candidate entries with the matching key.
    std::vector<const BookEntry*> cand;
    for (auto it = lo; it != entries_.end() && it->key == key; ++it)
        cand.push_back(&*it);
    if (cand.empty()) return Move::none();

    const BookEntry* chosen = cand.front();
    if (pickBest) {
        for (const BookEntry* e : cand)
            if (e->weight > chosen->weight) chosen = e;
    } else {
        std::uint32_t total = 0;
        for (const BookEntry* e : cand) total += e->weight ? e->weight : 1;
        static std::mt19937 rng(0xC0FFEE);
        std::uint32_t r = std::uniform_int_distribution<std::uint32_t>(0, total - 1)(rng);
        for (const BookEntry* e : cand) {
            std::uint32_t w = e->weight ? e->weight : 1;
            if (r < w) { chosen = e; break; }
            r -= w;
        }
    }

    // Match the encoded move back to a legal move (handles castling/promotion).
    MoveList list;
    generate(pos, list, LEGAL);
    for (const auto& sm : list)
        if (encode_polyglot(pos, sm.move) == chosen->move)
            return sm.move;
    return Move::none();   // book move illegal here: fall back to search
}

bool build_book_from_pgn(std::istream& pgn, const std::string& path, int maxPlies) {
    // (key, move) -> count.
    std::map<std::pair<Key, std::uint16_t>, std::uint32_t> counts;

    GameRecord g;
    while (read_pgn(pgn, g)) {
        Position pos;
        pos.set(g.startFen);
        int plies = 0;
        for (Move m : g.moves) {
            if (plies++ >= maxPlies) break;
            const Key k = book_key(pos);
            const std::uint16_t mv = encode_polyglot(pos, m);
            ++counts[{k, mv}];
            pos.do_move(m);
        }
        g = GameRecord{};
    }

    std::vector<BookEntry> out;
    out.reserve(counts.size());
    for (const auto& [km, c] : counts)
        out.push_back(BookEntry{km.first, km.second,
                                std::uint16_t(std::min<std::uint32_t>(c, 0xFFFF)), 0});
    std::stable_sort(out.begin(), out.end(),
                     [](const BookEntry& a, const BookEntry& b) {
                         return a.key < b.key || (a.key == b.key && a.move < b.move);
                     });

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    for (const BookEntry& e : out) {
        wr64(f, e.key); wr16(f, e.move); wr16(f, e.weight); wr32(f, e.learn);
    }
    return true;
}

} // namespace chess
