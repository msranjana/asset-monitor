"""Printing helper for a program where several threads write to the console.

`print` writes the text and the newline separately, so two threads can end up on the
same line. `say` writes each message in one go instead, and flushes right away so the
output is not held back in a buffer.
"""

import sys
import threading

_lock = threading.Lock()


def say(message):
    with _lock:
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()
