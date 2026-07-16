// selftest.h - In-engine self-tests (runnable via the `selftest` UCI command).
//
// These run inside the trusted engine binary so Phase 5 infrastructure can be
// verified without spawning separate test executables.
#pragma once

#include <iosfwd>

namespace chess {

// Runs all registered self-tests, printing PASS/FAIL lines to `os`.
// Returns the number of failures (0 == all passed).
int run_selftests(std::ostream& os);

} // namespace chess
