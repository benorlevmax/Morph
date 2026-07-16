// match.cpp - Match runner / self-play.
#include "match/match.h"
#include "core/movegen.h"
#include "eval/evaluate.h"

#include <cstdlib>

namespace chess {

std::string play_game(EngineConfig white, EngineConfig black,
                      const std::string& startFen, const MatchSettings& s,
                      GameRecord* out) {
    Search searchW, searchB;
    searchW.set_hash_size(white.hashMB);
    searchB.set_hash_size(black.hashMB);
    searchW.set_config(white.cfg);
    searchB.set_config(black.cfg);
    searchW.set_threads(white.threads);
    searchB.set_threads(black.threads);
    searchW.set_quiet(true);
    searchB.set_quiet(true);
    searchW.clear();
    searchB.clear();

    Position pos;
    pos.set(startFen);

    GameRecord rec;
    rec.startFen = startFen;
    rec.tags["White"] = white.name;
    rec.tags["Black"] = black.name;

    int resignCountW = 0, resignCountB = 0;
    std::string result = "*";

    for (int ply = 0; ply < s.maxPlies; ++ply) {
        // Terminal checks before moving.
        MoveList legal;
        generate(pos, legal, LEGAL);
        if (legal.empty()) {
            if (pos.in_check())
                result = (pos.side_to_move() == WHITE) ? "0-1" : "1-0";  // mated
            else
                result = "1/2-1/2";   // stalemate
            break;
        }
        if (pos.is_draw()) { result = "1/2-1/2"; break; }

        const bool whiteToMove = pos.side_to_move() == WHITE;
        Search& eng = whiteToMove ? searchW : searchB;
        const EngineConfig& cfg = whiteToMove ? white : black;

        // Evaluation mode is a global; set it for the side about to move
        // (moves are played strictly sequentially, so this is safe).
        Eval::set_mode(cfg.evalMode);
        Eval::set_material_only(cfg.materialOnly);
        eng.arm();
        SearchResult r = eng.think(pos, cfg.limits);
        if (r.best == Move::none()) { result = "1/2-1/2"; break; }

        // Resign adjudication (score is from side-to-move's perspective).
        if (s.resignScore > 0) {
            int sc = int(r.score);
            int& cnt = whiteToMove ? resignCountW : resignCountB;
            int& opp = whiteToMove ? resignCountB : resignCountW;
            if (sc <= -s.resignScore) { if (++cnt >= s.resignPlies) {
                result = whiteToMove ? "0-1" : "1-0"; break; } }
            else cnt = 0;
            (void)opp;
        }

        rec.moves.push_back(r.best);
        pos.do_move(r.best);
    }

    if (result == "*") result = "1/2-1/2";   // hit ply cap
    rec.result = result;
    if (out) *out = rec;
    return result;
}

MatchResult play_match(const EngineConfig& a, const EngineConfig& b,
                       const MatchSettings& s) {
    MatchResult mr;
    const std::vector<std::string> defOpen = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"};
    const std::vector<std::string>& openings = s.openings.empty() ? defOpen : s.openings;

    for (int g = 0; g < s.games; ++g) {
        // Each opening is played twice: once with A as White, once as Black.
        const std::string& fen = openings[(g / 2) % openings.size()];
        const bool aIsWhite = !s.alternate || (g % 2 == 0);

        GameRecord rec;
        std::string res = aIsWhite ? play_game(a, b, fen, s, &rec)
                                   : play_game(b, a, fen, s, &rec);

        // Translate result to A's perspective.
        if (res == "1/2-1/2") ++mr.draws;
        else if ((res == "1-0") == aIsWhite) ++mr.aWins;
        else ++mr.bWins;

        if (s.collectPgn) {
            rec.tags["Round"] = std::to_string(g + 1);
            mr.games.push_back(rec);
        }
    }
    return mr;
}

} // namespace chess
