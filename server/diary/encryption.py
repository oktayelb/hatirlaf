from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import models


TEXT_PREFIX = "hatirlaf:v1:"
FILE_PREFIX = b"hatirlaf-file:v1:"


def encrypt_text(value: str) -> str:
    if value.startswith(TEXT_PREFIX):
        return value
    token = _fernet().encrypt(value.encode("utf-8")).decode("ascii")
    return TEXT_PREFIX + token


def decrypt_text(value: str) -> str:
    if not value.startswith(TEXT_PREFIX):
        return value
    token = value[len(TEXT_PREFIX):].encode("ascii")
    try:
        return _fernet().decrypt(token).decode("utf-8")
    except InvalidToken as exc:
        raise ImproperlyConfigured("Hatırlaf encryption key cannot decrypt stored data.") from exc


def encrypt_bytes(value: bytes) -> bytes:
    if value.startswith(FILE_PREFIX):
        return value
    return FILE_PREFIX + _fernet().encrypt(value)


def decrypt_bytes(value: bytes) -> bytes:
    if not value.startswith(FILE_PREFIX):
        return value
    try:
        return _fernet().decrypt(value[len(FILE_PREFIX):])
    except InvalidToken as exc:
        raise ImproperlyConfigured("Hatırlaf encryption key cannot decrypt stored media.") from exc


def is_encrypted_text(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TEXT_PREFIX)


def is_encrypted_bytes(value: bytes) -> bool:
    return value.startswith(FILE_PREFIX)


class EncryptedTextField(models.TextField):
    """TextField that stores ciphertext and returns plaintext to Django code."""

    description = "Encrypted text"

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

    def to_python(self, value):
        if value is None:
            return value
        value = str(value)
        return decrypt_text(value) if is_encrypted_text(value) else value

    def get_prep_value(self, value):
        value = super().get_prep_value(value)
        if value is None:
            return value
        value = str(value)
        return value if is_encrypted_text(value) else encrypt_text(value)


class EncryptedJSONField(models.TextField):
    """JSON-compatible field stored as encrypted text."""

    description = "Encrypted JSON"

    def __init__(self, *args, **kwargs):
        self._fallback_default = kwargs.get("default", None)
        super().__init__(*args, **kwargs)

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

    def to_python(self, value):
        if value is None:
            return self._default_value()
        if isinstance(value, (dict, list, int, float, bool)):
            return value
        raw = str(value)
        if is_encrypted_text(raw):
            raw = decrypt_text(raw)
        if raw == "":
            return self._default_value()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return self._default_value()

    def get_prep_value(self, value):
        if value is None:
            value = self._default_value()
        if is_encrypted_text(value):
            return value
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return encrypt_text(raw)

    def value_to_string(self, obj):
        return json.dumps(self.value_from_object(obj), ensure_ascii=False)

    def _default_value(self):
        if callable(self._fallback_default):
            return self._fallback_default()
        if self._fallback_default is not None:
            return self._fallback_default
        return None


_cached_fernet: Fernet | None = None


def _fernet() -> Fernet:
    global _cached_fernet
    if _cached_fernet is None:
        _cached_fernet = Fernet(_load_key())
    return _cached_fernet


def _load_key() -> bytes:
    configured = os.environ.get("HATIRLAF_ENCRYPTION_KEY") or getattr(
        settings, "HATIRLAF_ENCRYPTION_KEY", ""
    )
    if configured:
        return _normalise_key(configured)

    key_path = Path(getattr(settings, "HATIRLAF_ENCRYPTION_KEY_FILE"))
    if key_path.exists():
        return _normalise_key(key_path.read_text(encoding="ascii").strip())

    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
        handle.write(b"\n")
    return key


def _normalise_key(value: str | bytes) -> bytes:
    raw = value.encode("ascii") if isinstance(value, str) else value
    try:
        Fernet(raw)
        return raw
    except Exception:
        pass
    digest = base64.urlsafe_b64encode(raw[:32].ljust(32, b"\0"))
    Fernet(digest)
    return digest
