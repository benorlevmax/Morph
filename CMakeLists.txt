cmake_minimum_required(VERSION 3.20)

project(Morph
        VERSION 0.1.0
        DESCRIPTION "Morph -- a from-scratch chess engine (NNUE + deep-net research)"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put all binaries in a predictable place.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Compiler flags (MSVC primary, with portable fallbacks)
# ---------------------------------------------------------------------------
# CHESS_NATIVE_ARCH: OFF by default so a normal Release build (and every
# GitHub Actions release binary) is portable -- it runs on any machine that
# supports the compiler's own default baseline ISA, not just the exact CPU
# that built it. Turn this ON only for a build you compile and run yourself
# on the same machine (never for a binary you hand or ship to someone
# else): it enables -march=native / MSVC's AVX2 baseline for the extra
# speed those give NNUE's SIMD paths (src/nnue/nnue.cpp already falls back
# to portable scalar code at compile time when the target ISA lacks AVX2,
# via its own `#if defined(__AVX2__)` guards -- no source changes needed
# either way).
option(CHESS_NATIVE_ARCH
       "Optimize for the exact build machine's CPU (-march=native / MSVC AVX2). \
Fast, but the resulting binary is NOT portable -- only for local builds you \
compile and run yourself, never for binaries distributed to other people."
       OFF)
if(MSVC)
    add_compile_options(/W4 /permissive- /EHsc)
    add_compile_options($<$<CONFIG:Release>:/O2>)
    if(CHESS_NATIVE_ARCH)
        add_compile_options($<$<CONFIG:Release>:/arch:AVX2>)
    endif()
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    if(CHESS_NATIVE_ARCH)
        add_compile_options($<$<CONFIG:Release>:-march=native>)
    endif()
endif()
if(CHESS_NATIVE_ARCH)
    message(STATUS "CHESS_NATIVE_ARCH=ON: binaries optimized for THIS machine's CPU -- do not distribute them")
else()
    message(STATUS "CHESS_NATIVE_ARCH=OFF (default): portable binaries, safe to distribute")
endif()

# ---------------------------------------------------------------------------
# Core library (every subsystem links against this)
# ---------------------------------------------------------------------------
add_library(engine_core STATIC
    src/core/bitboard.cpp
    src/core/zobrist.cpp
    src/core/position.cpp
    src/core/movegen.cpp
    src/core/perft.cpp
    src/eval/psqt.cpp
    src/eval/evaluate.cpp
    src/nnue/nnue.cpp
    src/search/tt.cpp
    src/search/search.cpp
    src/io/pgn.cpp
    src/io/book.cpp
    src/book/opening_book.cpp
    src/syzygy/tablebases.cpp
    src/match/stats.cpp
    src/match/match.cpp
    src/train/dataset.cpp
    src/train/encoding.cpp
    src/train/selfplay.cpp
    src/train/trainer.cpp
    src/test/selftest.cpp
    src/uci/uci.cpp
)
target_include_directories(engine_core PUBLIC ${CMAKE_SOURCE_DIR}/src)

find_package(Threads REQUIRED)
target_link_libraries(engine_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Syzygy tablebases via Fathom.
# Requires Fathom's source vendored at the project root: tbprobe.c, tbprobe.h,
# tbchess.c, tbconfig.h and stdendian.h (all present as of this change).
# Probing itself is still a no-op at runtime unless a UCI `SyzygyPath` is
# configured with real .rtbw/.rtbz files, so enabling this by default only
# adds capability -- it changes nothing for anyone not pointing at real
# tablebase files. Disable with -DCHESS_USE_FATHOM=OFF if Fathom's vendored
# sources are ever removed.
# ---------------------------------------------------------------------------
option(CHESS_USE_FATHOM "Build with Fathom Syzygy tablebase probing" ON)
if(CHESS_USE_FATHOM)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/tbchess.c")
        message(FATAL_ERROR
            "CHESS_USE_FATHOM=ON but tbchess.c is missing. Copy Fathom's "
            "src/tbchess.c (and stdendian.h) next to tbprobe.c, then re-run cmake.")
    endif()
    add_library(fathom STATIC tbprobe.c)
    target_include_directories(fathom PUBLIC ${CMAKE_SOURCE_DIR})
    set_source_files_properties(tbprobe.c PROPERTIES LANGUAGE CXX)  # Fathom builds as C++
    if(MSVC)
        target_compile_options(fathom PRIVATE /TP /wd4244 /wd4267 /wd4146)
    endif()
    target_link_libraries(engine_core PUBLIC fathom)
    target_compile_definitions(engine_core PUBLIC CHESS_USE_FATHOM)
    target_include_directories(engine_core PUBLIC ${CMAKE_SOURCE_DIR})
    message(STATUS "Fathom Syzygy probing: ENABLED")
else()
    message(STATUS "Fathom Syzygy probing: disabled (graceful fallback)")
endif()

# ---------------------------------------------------------------------------
# Perft CLI application
# ---------------------------------------------------------------------------
add_executable(chess_perft src/apps/main.cpp)
target_link_libraries(chess_perft PRIVATE engine_core)

# Main UCI engine
add_executable(chess src/apps/uci_main.cpp)
target_link_libraries(chess PRIVATE engine_core)

# Match / self-play driver
add_executable(chess_match src/apps/match_main.cpp)
target_link_libraries(chess_match PRIVATE engine_core)

# Training CLI (self-play data generation + reference CPU trainer)
add_executable(chess_train src/apps/train_main.cpp)
target_link_libraries(chess_train PRIVATE engine_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_perft tests/test_perft.cpp)
target_link_libraries(test_perft PRIVATE engine_core)
add_test(NAME perft COMMAND test_perft)

add_executable(test_fen tests/test_fen.cpp)
target_link_libraries(test_fen PRIVATE engine_core)
add_test(NAME fen COMMAND test_fen)

add_executable(test_search tests/test_search.cpp)
target_link_libraries(test_search PRIVATE engine_core)
add_test(NAME search COMMAND test_search)

add_executable(test_eval tests/test_eval.cpp)
target_link_libraries(test_eval PRIVATE engine_core)
add_test(NAME eval COMMAND test_eval)

add_executable(test_pgn tests/test_pgn.cpp)
target_link_libraries(test_pgn PRIVATE engine_core)
add_test(NAME pgn COMMAND test_pgn)

add_executable(test_opening_book tests/test_opening_book.cpp)
target_link_libraries(test_opening_book PRIVATE engine_core)
add_test(NAME opening_book COMMAND test_opening_book)
