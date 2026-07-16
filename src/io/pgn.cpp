// pgn.cpp - SAN notation and PGN serialization.
#include "io/pgn.h"
#include "core/movegen.h"

#include <cctype>
#include <istream>
#include <ostream>
#include <sstream>

namespace chess {

namespace {

constexpr char PieceLetter[PIECE_TYPE_NB] = {' ', 'P', 'N', 'B', 'R', 'Q', 'K'};

std::string sq_str(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

// '+' / '#' suffix for a move (mutates and restores pos).
std::string check_suffix(Position& pos, Move m) {
    pos.do_move(m);
    bool chk = pos.in_check();
    MoveList replies;
    generate(pos, replies, LEGAL);
    bool noMoves = replies.empty();
    pos.undo_move(m);
    if (chk) return noMoves ? "#" : "+";
    return "";
}

// Disambiguation string for a piece move (file, rank, or both as needed).
std::string disambiguation(Position& pos, PieceType pt, Square from, Square to) {
    MoveList list;
    generate(pos, list, LEGAL);
    bool any = false, sameFile = false, sameRank = false;
    for (const auto& sm : list) {
        Move mv = sm.move;
        if (mv.from_sq() == from) continue;
        if (mv.to_sq() != to) continue;
        if (type_of(pos.piece_on(mv.from_sq())) != pt) continue;
        any = true;
        if (file_of(mv.from_sq()) == file_of(from)) sameFile = true;
        if (rank_of(mv.from_sq()) == rank_of(from)) sameRank = true;
    }
    if (!any) return "";
    if (!sameFile) return std::string{char('a' + file_of(from))};
    if (!sameRank) return std::string{char('1' + rank_of(from))};
    return sq_str(from);
}

// Normalize a SAN token for tolerant comparison.
std::string normalize(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '+' || c == '#' || c == '!' || c == '?' || c == '=')
            continue;
        if (c == '0') c = 'O';   // 0-0 -> O-O
        if (std::isspace((unsigned char)c)) continue;
        r += c;
    }
    return r;
}

} // namespace

std::string move_to_san(Position& pos, Move m) {
    if (m == Move::none()) return "(none)";
    if (m == Move::null()) return "--";

    const Square from = m.from_sq();
    const Square to   = m.to_sq();

    if (m.type_of() == CASTLING) {
        std::string s = (file_of(to) == FILE_G) ? "O-O" : "O-O-O";
        return s + check_suffix(pos, m);
    }

    const PieceType pt = type_of(pos.piece_on(from));
    const bool capture = (m.type_of() == EN_PASSANT) || !pos.empty(to);
    std::string s;

    if (pt == PAWN) {
        if (capture) { s += char('a' + file_of(from)); s += 'x'; }
        s += sq_str(to);
        if (m.type_of() == PROMOTION) { s += '='; s += PieceLetter[m.promotion_type()]; }
    } else {
        s += PieceLetter[pt];
        s += disambiguation(pos, pt, from, to);
        if (capture) s += 'x';
        s += sq_str(to);
    }
    return s + check_suffix(pos, m);
}

Move san_to_move(Position& pos, const std::string& san) {
    const std::string want = normalize(san);
    if (want.empty()) return Move::none();
    MoveList list;
    generate(pos, list, LEGAL);
    for (const auto& sm : list)
        if (normalize(move_to_san(pos, sm.move)) == want)
            return sm.move;
    return Move::none();
}

void write_pgn(std::ostream& os, const GameRecord& game) {
    static const char* order[] = {"Event", "Site", "Date", "Round", "White", "Black"};
    auto emit = [&](const std::string& k, const std::string& dflt) {
        auto it = game.tags.find(k);
        os << "[" << k << " \"" << (it != game.tags.end() ? it->second : dflt) << "\"]\n";
    };
    emit("Event", "?");
    emit("Site", "?");
    emit("Date", "????.??.??");
    emit("Round", "-");
    emit("White", "?");
    emit("Black", "?");
    os << "[Result \"" << game.result << "\"]\n";

    const std::string start =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    if (game.startFen != start)
        os << "[SetUp \"1\"]\n[FEN \"" << game.startFen << "\"]\n";
    // Emit any extra tags not already written.
    for (const auto& [k, v] : game.tags) {
        bool known = (k == "Result");
        for (const char* o : order) if (k == o) known = true;
        if (!known) os << "[" << k << " \"" << v << "\"]\n";
    }
    os << "\n";

    Position pos;
    pos.set(game.startFen);
    std::ostringstream line;
    int col = 0;
    auto flush_token = [&](const std::string& tok) {
        if (col + int(tok.size()) + 1 > 80) { os << line.str() << "\n"; line.str(""); col = 0; }
        if (col) { line << ' '; ++col; }
        line << tok; col += int(tok.size());
    };

    int moveNo = pos.game_ply() / 2 + 1;
    bool whiteToMove = pos.side_to_move() == WHITE;
    if (!whiteToMove) flush_token(std::to_string(moveNo) + "...");

    for (Move m : game.moves) {
        if (whiteToMove) flush_token(std::to_string(moveNo) + ".");
        flush_token(move_to_san(pos, m));
        pos.do_move(m);
        if (!whiteToMove) ++moveNo;
        whiteToMove = !whiteToMove;
    }
    flush_token(game.result);
    os << line.str() << "\n\n";
}

bool read_pgn(std::istream& is, GameRecord& out) {
    out = GameRecord{};
    std::string line;
    bool sawTag = false, sawAny = false;

    // Tag section.
    while (std::getline(is, line)) {
        // Trim leading whitespace.
        std::size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) { if (sawTag) break; else continue; }
        if (line[b] != '[') { /* movetext starts here */ break; }
        sawTag = sawAny = true;
        std::size_t key0 = b + 1;
        std::size_t key1 = line.find(' ', key0);
        std::size_t q0 = line.find('"', key1);
        std::size_t q1 = line.find('"', q0 + 1);
        if (key1 != std::string::npos && q0 != std::string::npos && q1 != std::string::npos) {
            std::string key = line.substr(key0, key1 - key0);
            std::string val = line.substr(q0 + 1, q1 - q0 - 1);
            out.tags[key] = val;
            if (key == "FEN") out.startFen = val;
            if (key == "Result") out.result = val;
        }
    }

    // Movetext section: accumulate remaining lines until a blank line after we
    // have started, or EOF. `line` currently holds the first movetext line.
    std::string text = line + "\n";
    while (std::getline(is, line)) {
        std::size_t b = line.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) break;     // blank line ends the game
        text += line + "\n";
    }

    // Tokenize movetext, skipping comments, variations, NAGs, move numbers.
    Position pos;
    pos.set(out.startFen);
    std::size_t i = 0;
    auto skip_balanced = [&](char open, char close) {
        int depth = 0;
        for (; i < text.size(); ++i) {
            if (text[i] == open) ++depth;
            else if (text[i] == close) { if (--depth == 0) { ++i; break; } }
        }
    };

    while (i < text.size()) {
        char c = text[i];
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        if (c == '{') { skip_balanced('{', '}'); continue; }
        if (c == '(') { skip_balanced('(', ')'); continue; }
        if (c == '$') { while (i < text.size() && !std::isspace((unsigned char)text[i])) ++i; continue; }

        // Read a token up to whitespace.
        std::size_t j = i;
        while (j < text.size() && !std::isspace((unsigned char)text[j])
               && text[j] != '{' && text[j] != '(')
            ++j;
        std::string tok = text.substr(i, j - i);
        i = j;

        if (tok.empty()) continue;
        if (tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*") {
            out.result = tok;
            sawAny = true;
            break;
        }
        // Strip leading move number like "12." or "12..." possibly fused to SAN.
        std::size_t k = 0;
        while (k < tok.size() && std::isdigit((unsigned char)tok[k])) ++k;
        if (k > 0 && k < tok.size() && tok[k] == '.') {
            while (k < tok.size() && tok[k] == '.') ++k;
            tok = tok.substr(k);
            if (tok.empty()) continue;
        } else if (k == tok.size()) {
            continue;   // pure number
        }

        Move m = san_to_move(pos, tok);
        if (m == Move::none()) continue;   // unparsable token: skip defensively
        out.moves.push_back(m);
        pos.do_move(m);
        sawAny = true;
    }

    return sawAny;
}

} // namespace chess
