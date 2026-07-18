#!/usr/bin/env python3
"""test_train_network.py - Regression test for _run_pipeline_script()'s
frozen-vs-source dispatch logic (see its own docstring for the full story).

Context: this executor used to unconditionally build
[sys.executable, script_path] to run tools/nnue_pipeline/train.py or
export.py as a subprocess. That's correct when running from source
(`python3 platform_worker.py ...`), but under a PyInstaller-frozen
worker.exe, sys.executable points back at worker.exe itself, not a real
Python interpreter -- so the resulting command actually re-invoked
worker.exe's own CLI with train.py's arguments, which immediately failed
on worker.exe's own argparse ('the following arguments are required:
--server, --engine-bin'). This is exactly what a live contributor's
TRAIN_NETWORK task hit. The release archive also never shipped the .py
source files at all, so even a correct interpreter path wouldn't have
found anything to run.

The fix: when frozen (sys.frozen), run the script's own frozen,
standalone binary (nnue_train/nnue_export, shipped bundled next to
worker.exe -- see .github/workflows/release.yml) directly, no interpreter
needed. These tests mock subprocess.run so nothing is ever actually
executed -- they only prove which command gets built and that the
missing-binary case fails loudly and clearly instead of some confusing
downstream error.

Run directly:  python3 test_train_network.py
Run via pytest: pytest test_train_network.py
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worker'))

import train_network as tn


def _fake_completed_process(stdout='', stderr='', returncode=0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class FrozenBinaryPathTests(unittest.TestCase):
    def test_train_py_maps_to_nnue_train(self):
        with patch.object(tn, 'get_install_dir', return_value='/install/dir'), \
             patch('os.name', 'posix'):
            path = tn._frozen_pipeline_binary_path('train.py')
        self.assertEqual(path, os.path.join('/install/dir', 'nnue_train'))

    def test_export_py_maps_to_nnue_export(self):
        with patch.object(tn, 'get_install_dir', return_value='/install/dir'), \
             patch('os.name', 'posix'):
            path = tn._frozen_pipeline_binary_path('export.py')
        self.assertEqual(path, os.path.join('/install/dir', 'nnue_export'))

    def test_windows_gets_exe_suffix(self):
        with patch.object(tn, 'get_install_dir', return_value='C:\\install'), \
             patch('os.name', 'nt'):
            path = tn._frozen_pipeline_binary_path('train.py')
        self.assertTrue(path.endswith('nnue_train.exe'))


class RunPipelineScriptDispatchTests(unittest.TestCase):
    def test_source_mode_uses_interpreter_and_script_path(self):
        # Regression check: unchanged behavior for a `python3
        # platform_worker.py ...` source run.
        with patch.object(sys, 'frozen', False, create=True), \
             patch.object(tn.subprocess, 'run',
                          return_value=_fake_completed_process()) as mock_run:
            tn._run_pipeline_script('train.py', ['--data', 'x'], log=lambda m: None,
                                     timeout=10)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith('train.py'))
        self.assertIn('--data', cmd)

    def test_frozen_mode_invokes_bundled_binary_directly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = os.path.join(tmpdir, 'nnue_train')
            open(binary_path, 'w').close()  # just needs to exist for isfile()
            with patch.object(sys, 'frozen', True, create=True), \
                 patch.object(tn, 'get_install_dir', return_value=tmpdir), \
                 patch('os.name', 'posix'), \
                 patch.object(tn.subprocess, 'run',
                              return_value=_fake_completed_process()) as mock_run:
                tn._run_pipeline_script('train.py', ['--data', 'x'], log=lambda m: None,
                                         timeout=10)
            cmd = mock_run.call_args[0][0]
            # No interpreter in front -- the binary itself is cmd[0].
            self.assertEqual(cmd[0], binary_path)
            self.assertIn('--data', cmd)
            self.assertNotIn(sys.executable, cmd)

    def test_frozen_mode_missing_binary_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Deliberately don't create nnue_train in tmpdir.
            with patch.object(sys, 'frozen', True, create=True), \
                 patch.object(tn, 'get_install_dir', return_value=tmpdir), \
                 patch('os.name', 'posix'):
                with self.assertRaises(tn.TrainNetworkError) as ctx:
                    tn._run_pipeline_script('train.py', [], log=lambda m: None, timeout=10)
            self.assertIn('nnue_train', str(ctx.exception))
            self.assertIn(tmpdir, str(ctx.exception))

    def test_frozen_mode_export_py_also_dispatches_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = os.path.join(tmpdir, 'nnue_export')
            open(binary_path, 'w').close()
            with patch.object(sys, 'frozen', True, create=True), \
                 patch.object(tn, 'get_install_dir', return_value=tmpdir), \
                 patch('os.name', 'posix'), \
                 patch.object(tn.subprocess, 'run',
                              return_value=_fake_completed_process()) as mock_run:
                tn._run_pipeline_script('export.py', ['--checkpoint', 'x'], log=lambda m: None,
                                         timeout=10)
            cmd = mock_run.call_args[0][0]
            self.assertEqual(cmd[0], binary_path)

    def test_nonzero_returncode_raises_with_stderr(self):
        with patch.object(sys, 'frozen', False, create=True), \
             patch.object(tn.subprocess, 'run',
                          return_value=_fake_completed_process(stderr='boom', returncode=2)):
            with self.assertRaises(tn.TrainNetworkError) as ctx:
                tn._run_pipeline_script('train.py', [], log=lambda m: None, timeout=10)
        self.assertIn('boom', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
