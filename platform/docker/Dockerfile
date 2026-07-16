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
