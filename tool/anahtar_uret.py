#!/usr/bin/env python3
"""Yedek sifreleme anahtar cifti uretir.

    tool/anahtar_uret.py ~/hatirlaf-yedek-anahtari.gizli

ACIK anahtar APK'ya gomulur (yedek.json icine). GIZLI anahtar yalnizca
sizde kalir ve YEDEKLENMELIDIR: kaybolursa butun yedekler bir daha
acilamaz. Telefonlardaki asil kayitlar durdugu icin felaket degil, ama
her telefonu tek tek toplamak demek.

Ayrintilar: docs/yedekleme.md
"""
import base64
import hashlib
import os
import stat
import sys

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    hedef = os.path.expanduser(sys.argv[1])

    if os.path.exists(hedef):
        print(f"hata: {hedef} zaten var.", file=sys.stderr)
        print("Ustune yazmak eski yedekleri acilamaz hale getirir.", file=sys.stderr)
        return 1

    gizli = X25519PrivateKey.generate()
    gizli_ham = gizli.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    acik_ham = gizli.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    # Once 0600 ile olustur, sonra yaz: dosya bir an bile herkese
    # okunabilir durumda olmasin.
    bayrak = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(hedef, bayrak, 0o600), "wb") as f:
        f.write(gizli_ham)
    os.chmod(hedef, stat.S_IRUSR | stat.S_IWUSR)

    iz = hashlib.sha256(acik_ham).hexdigest()[:16]
    print(f"gizli anahtar : {hedef}  (yalnizca sizde kalmali)")
    print(f"parmak izi    : {iz}")
    print()
    print("yedek.json icine:")
    print(f'  "YEDEK_ALICI_ANAHTARI": "{base64.b64encode(acik_ham).decode()}"')
    print()
    print("Gizli anahtari SIMDI yedekleyin; tek kopya tek arıza demek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
