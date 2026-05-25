from __future__ import annotations

from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage

from .encryption import decrypt_bytes, encrypt_bytes, is_encrypted_bytes


class EncryptedFileSystemStorage(FileSystemStorage):
    """FileSystemStorage that encrypts media contents at rest."""

    def _save(self, name, content):
        chunks = []
        for chunk in content.chunks():
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
        encrypted = ContentFile(encrypt_bytes(b"".join(chunks)))
        return super()._save(name, encrypted)

    def _open(self, name, mode="rb"):
        encrypted_file = super()._open(name, "rb")
        try:
            data = encrypted_file.read()
        finally:
            encrypted_file.close()
        return ContentFile(decrypt_bytes(data), name=name)

    def encrypt_existing(self, name: str) -> bool:
        path = self.path(name)
        with open(path, "rb") as handle:
            data = handle.read()
        if is_encrypted_bytes(data):
            return False
        with open(path, "wb") as handle:
            handle.write(encrypt_bytes(data))
        return True
