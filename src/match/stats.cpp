// stats.cpp - Elo and SPRT computations.
#include "match/stats.h"

#include <algorithm>
#include <cmath>

namespace chess {

namespace {
double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }
double elo_to_score(double elo) { return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0)); }
} // namespace

double elo_diff(double score) {
    score = clampd(score, 1e-6, 1.0 - 1e-6);
    return -400.0 * std::log10(1.0 / score - 1.0);
}

EloEstimate elo_estimate(int wins, int losses, int draws) {
    const int n = wins + losses + draws;
    EloEstimate e;
    if (n == 0) return e;

    const double pw = double(wins) / n;
    const double pd = double(draws) / n;
    const double pl = double(losses) / n;
    const double score = pw + 0.5 * pd;

    e.elo = elo_diff(score);

    // Variance of the per-game score; standard error of the mean score.
    const double var = pw * 1.0 + pd * 0.25 + pl * 0.0 - score * score;
    const double stderrScore = std::sqrt(std::max(var, 1e-12) / n);

    // Propagate the score CI to Elo via the logistic derivative.
    const double lo = clampd(score - 1.96 * stderrScore, 1e-6, 1.0 - 1e-6);
    const double hi = clampd(score + 1.96 * stderrScore, 1e-6, 1.0 - 1e-6);
    e.margin = (elo_diff(hi) - elo_diff(lo)) / 2.0;
    return e;
}

SprtResult sprt(int wins, int losses, int draws,
                double elo0, double elo1, double alpha, double beta) {
    SprtResult r;
    r.lowerBound = std::log(beta / (1.0 - alpha));
    r.upperBound = std::log((1.0 - beta) / alpha);

    const int n = wins + losses + draws;
    if (n == 0) return r;

    const double pw = double(wins) / n;
    const double pd = double(draws) / n;
    const double pl = double(losses) / n;
    const double score = pw + 0.5 * pd;

    double var = pw * 1.0 + pd * 0.25 + pl * 0.0 - score * score;
    var = std::max(var, 1e-9);

    const double s0 = elo_to_score(elo0);
    const double s1 = elo_to_score(elo1);

    // GSPRT (Gaussian) LLR for the mean score under H1 vs H0:
    //   LLR = (N / 2var) * [ (score-s0)^2 - (score-s1)^2 ]
    //       = (N / 2var) * (s1-s0)*(2*score - s0 - s1)
    r.llr = (double(n) / (2.0 * var)) *
            ((score - s0) * (score - s0) - (score - s1) * (score - s1));

    if (r.llr >= r.upperBound)      r.verdict = SprtVerdict::AcceptH1;
    else if (r.llr <= r.lowerBound) r.verdict = SprtVerdict::AcceptH0;
    else                            r.verdict = SprtVerdict::Continue;
    return r;
}

} // namespace chess
