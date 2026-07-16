#!/usr/bin/env python3
"""logging_setup.py - Rotating file + console logging for the controller.

Everything under the 'automation' logger name (pipeline_controller, notify,
models_registry) shares this configuration once setup_logging() is called
from main(). Rotates at 10MB x 5 backups so a long-running daemon never
fills the disk with logs.
"""
import logging
import logging.handlers
import os

import config


def setup_logging(verbose=False):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    root = logging.getLogger('automation')
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.handlers.RotatingFileHandler(
        config.CONTROLLER_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    return root
