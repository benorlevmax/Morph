#!/usr/bin/env python3
"""test_platform_config.py - Regression test for get_install_dir(), the
frozen-aware path-resolution helper this module exports.

Context: under a normal `python3 platform_worker.py` source run, __file__
correctly points at a stable, permanent location. Under a PyInstaller
--onefile freeze (what every downloadable worker.exe/worker release
actually is -- see .github/workflows/release.yml), __file__ instead
resolves into a fresh temporary extraction directory created fresh on
every single run and deleted again on exit -- so anything derived from it
(the default --state-file location, --artifacts-cache-dir, and
train_network.py's bundled pipeline-binary lookup) silently breaks: saved
worker credentials vanish the moment the process exits, forcing
re-registration on every run. This was found via a live contributor
hitting exactly that symptom.

get_install_dir() fixes this by branching on sys.frozen (set by
PyInstaller) to use sys.executable's directory instead, which correctly
points at the real, on-disk .exe location in that case.

Run directly:  python3 test_platform_config.py
Run via pytest: pytest test_platform_config.py
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import platform_config as pc


class GetInstallDirTests(unittest.TestCase):
    def test_source_mode_uses_file_dir(self):
        # Not frozen (the normal case when running from source) -- should
        # resolve to this package's own directory (platform_config.py's
        # __file__ dir), matching the pre-fix behavior for a source run.
        with patch.object(sys, 'frozen', False, create=True):
            result = pc.get_install_dir()
        expected = os.path.dirname(os.path.abspath(pc.__file__))
        self.assertEqual(result, expected)

    def test_frozen_mode_uses_executable_dir_not_file_dir(self):
        # This is the actual bug fix: under a frozen worker.exe, __file__
        # resolves into a temp _MEIxxxxxx extraction directory, but
        # sys.executable correctly points at the real, permanent .exe
        # location -- get_install_dir() must prefer the latter when frozen.
        fake_exe_dir = os.path.abspath(os.path.join(os.sep, 'some', 'fake', 'install', 'dir'))
        fake_exe_path = os.path.join(fake_exe_dir, 'worker.exe')
        with patch.object(sys, 'frozen', True, create=True), \
             patch.object(sys, 'executable', fake_exe_path):
            result = pc.get_install_dir()
        self.assertEqual(os.path.normpath(result), os.path.normpath(fake_exe_dir))

    def test_frozen_flag_absent_behaves_like_not_frozen(self):
        # getattr(sys, 'frozen', False) must default to False when the
        # attribute doesn't exist at all (the normal case outside any
        # PyInstaller build -- sys.frozen is never set by plain CPython).
        if hasattr(sys, 'frozen'):
            delattr(sys, 'frozen')
        try:
            result = pc.get_install_dir()
        finally:
            pass  # nothing to restore -- plain CPython never has sys.frozen
        expected = os.path.dirname(os.path.abspath(pc.__file__))
        self.assertEqual(result, expected)


class StateFileDefaultTests(unittest.TestCase):
    def test_state_file_default_derives_from_install_dir(self):
        with patch.object(sys, 'frozen', False, create=True):
            args = pc.parse_args(['--server', 'http://x', '--engine-bin', '/x'])
        expected_dir = pc.get_install_dir()
        self.assertEqual(os.path.dirname(args.state_file), expected_dir)

    def test_artifacts_cache_dir_default_derives_from_install_dir(self):
        with patch.object(sys, 'frozen', False, create=True):
            args = pc.parse_args(['--server', 'http://x', '--engine-bin', '/x'])
        expected_dir = pc.get_install_dir()
        self.assertEqual(os.path.dirname(args.artifacts_cache_dir), expected_dir)


if __name__ == '__main__':
    unittest.main()
