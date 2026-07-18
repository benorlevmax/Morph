# platform/docker/Dockerfile - Container image for the community-compute
# server (platform/server/app.py). Does NOT include the C++ engine or any
# training pipeline -- those run on the maintainer's own machine / on
# volunteer worker machines, never centrally in this container. This image
# is just the FastAPI server plus the (unmodified) distributed/server/
# module it reuses.
#
# Build from the REPO ROOT (not this directory), because the server imports
# sibling code from distributed/server/ via a relative sys.path insert --
# see platform/server/database.py's docstring:
#
#   docker build -f platform/docker/Dockerfile -t chess-platform-server .
#
# or just use docker-compose.yml in this directory, which sets the build
# context correctly.
FROM python:3.11-slim AS base

# System deps: none beyond what python:3.11-slim already has -- this image
# deliberately has no compiler/build-essential, no C++ toolchain, and no
# copy of src/ -- it never builds or runs the chess engine itself.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin platform

WORKDIR /app

# Copy only what the server actually imports: distributed/server (reused
# via sys.path, see database.py) and platform/ itself. Explicitly NOT
# copying src/, tools/, python-engine/, stockfish/, lc0.exe, etc. -- keeps
# the image small and makes it obvious this container has no engine binary.
COPY distributed/server/ /app/distributed/server/
COPY distributed/database/ /app/distributed/database/
COPY platform/ /app/platform/

# One deliberate, narrow exception to "no tools/" above:
# platform/server/auto_pipeline.py imports uci_match.py (tools/nnue_pipeline/)
# for Elo/SPRT math -- both a human-readable log line (elo_estimate) and,
# critically, the actual promotion decision itself (sprt(), used by
# maybe_promote_candidates()/_sprt_verdict() to decide whether a candidate
# network gets accepted). Without this file present, auto_pipeline.py
# crash-loops the instant it runs (ModuleNotFoundError: uci_match) and
# candidate promotion silently never happens -- a real bug found via a live
# crash-loop on the deployed server, not caught by tests because the
# in-process test suite runs with the full repo on sys.path and never sees
# what actually ships in the image. uci_match.py only imports stdlib
# (math/subprocess/time -- see its own module docstring), so this adds zero
# new pip dependencies; copying just this one file, not all of
# tools/nnue_pipeline/, keeps the "no engine/training toolchain in this
# image" property intact for everything else in that directory (train.py,
# nnue_format.py, etc. pull in torch/numpy this image deliberately doesn't
# install).
COPY tools/nnue_pipeline/uci_match.py /app/tools/nnue_pipeline/uci_match.py

RUN pip install --no-cache-dir -r /app/platform/requirements.txt

# Default DB path lives on a mounted volume so it survives container
# recreation (docker-compose.yml declares this as a named volume).
ENV CHESS_PLATFORM_DB_PATH=/data/platform.sqlite3
RUN mkdir -p /data && chown platform:platform /data
VOLUME /data

USER platform
WORKDIR /app/platform/server
EXPOSE 8000

# No CHESS_PLATFORM_ADMIN_TOKEN default -- if the operator doesn't set one,
# platform_config.py generates and logs a random one at startup (see
# print_startup_banner()), same fail-safe behavior as distributed/server/.
CMD ["python3", "run_server.py", "--host", "0.0.0.0", "--port", "8000"]
