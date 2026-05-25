from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager


@contextmanager
def decrypted_file_path(file_field, *, suffix: str = ""):
    """Expose a possibly encrypted Django FileField as a temporary plaintext path."""

    with file_field.open("rb") as source:
        data = source.read()

    fd, path = tempfile.mkstemp(prefix="hatirlaf-audio-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as target:
            target.write(data)
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
