#!/usr/bin/env python3
"""test_docker_image_contents.py - Regression test for a real bug found via
a live crash-loop on the deployed server: platform/server/auto_pipeline.py
imports uci_match.py from tools/nnue_pipeline/ (for Elo/SPRT math -- both
the log-line elo_estimate() and, critically, the actual promotion decision
sprt() used by maybe_promote_candidates()), but platform/docker/Dockerfile
deliberately does NOT copy tools/ into the server image (by design, to keep
the image free of the engine/training toolchain -- see the Dockerfile's own
comment). Nothing in this gap showed up as a test failure, because
test_promotion.py and friends import auto_pipeline with the FULL repo on
sys.path (tools/nnue_pipeline physically present on disk), which is exactly
what the deployed container does NOT have. The server crash-looped in
production before this was caught.

This test doesn't run the server in a real container (no docker available
in most dev/test environments, and that would be a much heavier, slower
check). Instead it does the cheap, fast, static thing that would have
caught this specific bug immediately: parse the Dockerfile's COPY
instructions, then scan platform/server/*.py for cross-directory imports of
anything living under tools/, and assert every such import has a matching
COPY line. If someone adds a new "from tools/whatever import X" to
auto_pipeline.py (or any platform/server/*.py) without also updating the
Dockerfile, this test fails loudly instead of only surfacing as a
production crash-loop days or weeks later.
"""
import os
import re
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DOCKERFILE_PATH = os.path.join(REPO_ROOT, 'platform', 'docker', 'Dockerfile')
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(REPO_ROOT, 'tools')


def _dockerfile_copied_paths():
    """Returns the set of repo-relative source paths (files or directory
    prefixes) that the Dockerfile's COPY instructions bring into the image."""
    copied = set()
    with open(DOCKERFILE_PATH) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('COPY '):
                continue
            parts = line.split()
            # COPY <src> <dest>  (a plain two-arg COPY, which is all this
            # Dockerfile uses -- no --from=, no multi-src forms)
            if len(parts) == 3:
                copied.add(parts[1])
    return copied


def _modules_imported_from_tools():
    """Scans every platform/server/*.py file for 'from <name> import ...' or
    'import <name>' where <name> matches a .py filename that actually exists
    somewhere under tools/ -- i.e. a real cross-directory dependency on
    tools/, not a coincidentally-matching stdlib/pip package name."""
    tools_module_names = set()
    for root, _dirs, files in os.walk(TOOLS_DIR):
        for fn in files:
            if fn.endswith('.py'):
                tools_module_names.add(fn[:-3])

    found = {}  # module_name -> set of (server_file, tools_file_relpath)
    import_re = re.compile(r'^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)\b')
    for fn in os.listdir(SERVER_DIR):
        if not fn.endswith('.py'):
            continue
        path = os.path.join(SERVER_DIR, fn)
        with open(path) as f:
            for line in f:
                m = import_re.match(line)
                if not m:
                    continue
                name = m.group(1)
                if name in tools_module_names:
                    # locate the actual file(s) under tools/ with this name
                    for root, _dirs, files in os.walk(TOOLS_DIR):
                        if f'{name}.py' in files:
                            relpath = os.path.relpath(os.path.join(root, f'{name}.py'), REPO_ROOT)
                            found.setdefault(name, set()).add((fn, relpath.replace(os.sep, '/')))
    return found


class DockerImageContainsEverythingItImportsTests(unittest.TestCase):
    def test_every_tools_import_in_platform_server_is_copied_into_the_image(self):
        copied = _dockerfile_copied_paths()
        imports = _modules_imported_from_tools()

        missing = []
        for module_name, usages in imports.items():
            for server_file, tools_relpath in usages:
                # Satisfied if the Dockerfile copies the exact file, or a
                # directory prefix that contains it (e.g. "tools/nnue_pipeline/").
                covered = any(
                    tools_relpath == src or tools_relpath.startswith(src.rstrip('/') + '/')
                    for src in copied
                )
                if not covered:
                    missing.append((server_file, module_name, tools_relpath))

        self.assertEqual(
            missing, [],
            'platform/server/*.py imports these tools/ modules, but the '
            'Dockerfile does not COPY them into the image -- this is exactly '
            'the bug that crash-looped auto_pipeline.py in production '
            '(ModuleNotFoundError: uci_match). Add a COPY line for each: '
            + repr(missing))

    def test_known_uci_match_dependency_is_explicitly_covered(self):
        # Belt-and-suspenders direct check for the specific bug that was
        # actually observed in production, independent of the general
        # scanner above (which could theoretically have its own bugs).
        copied = _dockerfile_copied_paths()
        self.assertIn(
            'tools/nnue_pipeline/uci_match.py', copied,
            'auto_pipeline.py imports uci_match for the real SPRT promotion '
            'decision (_sprt_verdict/maybe_promote_candidates), not just a '
            'log line -- without this COPY, candidate networks can never '
            'be promoted in the deployed server, and the process '
            'crash-loops the moment maybe_promote_candidates() runs.')


if __name__ == '__main__':
    unittest.main()
