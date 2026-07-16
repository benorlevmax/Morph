// stats.h - Elo estimation and SPRT for engine match results.
#pragma once

namespace chess {

struct EloEstimate {
    double elo    = 0.0;   // point estimate of Elo difference (A relative to B)
    double margin = 0.0;   // +/- 95% confidence margin
};

// Elo difference implied by an average score in [0,1].
double elo_diff(double score);

// Elo estimate with a 95% confidence margin from W/L/D counts.
EloEstimate elo_estimate(int wins, int losses, int draws);

enum class SprtVerdict { Continue, AcceptH1, AcceptH0 };

struct SprtResult {
    double      llr        = 0.0;   // log-likelihood ratio
    double      lowerBound = 0.0;   // accept H0 at/below
    double      upperBound = 0.0;   // accept H1 at/above
    SprtVerdict verdict    = SprtVerdict::Continue;
};

// Generalized SPRT (GSPRT) over trinomial W/L/D results.
// H0: elo == elo0,  H1: elo == elo1.  Type I/II error rates alpha/beta.
SprtResult sprt(int wins, int losses, int draws,
                double elo0, double elo1,
                double alpha = 0.05, double beta = 0.05);

} // namespace chess
