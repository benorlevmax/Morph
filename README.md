# Morph

Morph is a free chess engine — a computer program that plays and analyzes
chess. You use it together with a chess app (called a "GUI") that shows
the board; Morph does the thinking.

Morph is also a community project: anyone can help make future versions
of Morph stronger by letting their computer help out in the background,
even if they've never written a line of code.

## Features

- ♟️ **Plays and analyzes chess** — works with standard chess GUIs on
  Windows and Linux.
- 🧠 **Learns from experience** — Morph is developing a neural-network way
  of judging positions, trained on millions of practice games. It's still
  learning, so it isn't switched on by default yet.
- 🌍 **Community-powered** — volunteers' computers generate practice games
  and test new versions, so Morph keeps improving over time.
- 💻 **Help out with your CPU** — run a small helper program in the
  background and it takes care of the rest.
- 🚀 **Help out with your GPU** — if you have a supported NVIDIA or AMD
  graphics card, it can help train new versions faster.
- 📈 **No unproven changes** — a new version is only adopted after playing
  enough real test games to prove it's actually better, not on a hunch.

## Getting started

### "I just want to play chess against Morph"

1. **Get a chess GUI.** This is the app with the board you click on — Morph
   itself has no board or graphics. Any GUI that supports the "UCI"
   engine format works (most do); a couple of common free ones are Arena
   and CuteChess.
2. **Download Morph.** Go to [Releases](../../releases) and download the
   archive for your operating system (Windows or Linux).
3. **Unzip it.** Inside you'll find a file named `chess.exe` (Windows) or
   `chess` (Linux) — that's Morph.
4. **Add it to your GUI.** Look for a menu option like "Install engine" or
   "Add engine" and point it at that file.
5. **Play or analyze.** Your GUI takes it from there — you never need to
   type commands yourself.

### "I want to help make Morph stronger"

This is completely optional and separate from playing. Volunteers run a
small helper program, called **the worker**, that asks a server for a
small task (like playing some practice games), does it, and sends back
the result. You don't need to understand chess programming to help.

1. **Download the same release archive** from step 2 above — it already
   includes the worker alongside the engine.
2. **Get an API key.** The official community server is at
   `http://64.181.243.154:8000` — sign up for a free account and get your
   own key in three commands (replace `yourname`/`yourpassword`):

   ```bash
   curl -X POST http://64.181.243.154:8000/accounts/register \
     -H "Content-Type: application/json" \
     -d '{"username": "yourname", "password": "yourpassword"}'

   SESSION=$(curl -s -X POST http://64.181.243.154:8000/accounts/login \
     -H "Content-Type: application/json" \
     -d '{"username": "yourname", "password": "yourpassword"}' \
     | python3 -c "import sys,json; print(json.load(sys.stdin)['session_token'])")

   curl -X POST http://64.181.243.154:8000/accounts/api-key/regenerate \
     -H "Authorization: Bearer $SESSION"
   ```

   The last command prints your API key (`cek_...`) once — save it, it
   can't be shown again (only regenerated, which invalidates the old one).
   This is only needed to contribute compute — if you just want to play,
   you can skip this whole section. Note the server is plain HTTP for now
   (no TLS yet), so don't reuse a password you care about elsewhere.
3. **Run the worker:**

   ```bash
   # Linux
   ./worker --server http://64.181.243.154:8000 --engine-bin ./chess --api-key <your key>

   # Windows
   worker.exe --server http://64.181.243.154:8000 --engine-bin chess.exe --api-key <your key>
   ```
4. **Leave it running.** It reports what your computer can do, requests
   real work, and keeps going in the background until you close it.

For every option the worker supports — limiting how much CPU/memory it
uses, opting in to help train new versions with `--trainer-capable`, and
more — see [platform/docs/WORKER.md](platform/docs/WORKER.md), the full
worker reference guide. [SECURITY.md](SECURITY.md) explains exactly what
the worker does and doesn't do on your computer.

## Building from source

Most people don't need this — see [Getting started](#getting-started)
above to just download a release. Build from source only if you want to
modify the engine yourself.

Windows (MSVC + CMake), from a *Developer Command Prompt for VS*:

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Linux (CMake + a C++20 compiler):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

The engine (`chess`) and a few developer/testing tools land in
`build/bin/Release/` (Windows) or `build/bin/` (Linux).

Run the test suite: `ctest --test-dir build -C Release --output-on-failure`

## Repository layout

- `src/` — the chess engine itself
- `tests/` — automated tests that check the engine still works correctly
- `tools/` — programs used to train new versions of Morph's neural network
- `platform/` — the server and worker programs that coordinate community help
- `distributed/` — an older, simpler version of the community system, kept
  for local testing
- `docs/` — technical notes for developers

## Contributing

There are two ways to help: donate some computer time (see
[Getting started](#getting-started) above), or contribute code. For bug
reports and code contributions, see [CONTRIBUTING.md](CONTRIBUTING.md) —
it explains how the project is organized and what a good pull request
looks like. Running your own server is covered in
[platform/docs/SERVER.md](platform/docs/SERVER.md). Security issues:
please see [SECURITY.md](SECURITY.md) rather than opening a public issue.

## License

Morph is free and open source, licensed under the
[GPLv3](COPYING). It also includes one small piece of third-party code
(Fathom, for endgame tablebase lookups, MIT-licensed) — see
[FATHOM-LICENSE](FATHOM-LICENSE) for details.

## Documentation

The sections above cover everything most people need. If you want to go
deeper:

- [docs/SEARCH_ARCHITECTURE.md](docs/SEARCH_ARCHITECTURE.md) — how the
  engine decides on a move
- [platform/docs/ARCHITECTURE.md](platform/docs/ARCHITECTURE.md) — how the
  server and worker fit together
- [platform/docs/TRAINING.md](platform/docs/TRAINING.md) — how new
  versions are trained and tested automatically
- [platform/docs/SERVER.md](platform/docs/SERVER.md) — running your own
  server
- [SECURITY.md](SECURITY.md) — the security and privacy mo