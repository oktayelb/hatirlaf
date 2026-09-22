#!/usr/bin/env python3
"""Telefondan gelen sifreli yedekleri cozer.

    tool/yedek_coz.py [gizli-anahtar] <girdi.hyz> [cikti]
    tool/yedek_coz.py [gizli-anahtar] --klasor <dizin>

Anahtar verilmezse depo kokundeki .env icindeki YEDEK_GIZLI_ANAHTAR
kullanilir. Ikinci bicim bir dizindeki butun .hyz dosyalarini cozer;
uzantisi atilarak yanina yazilir.

Bicim tanimi: lib/services/backup_crypto.dart
"""
import hashlib
import os
import sys

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization

SIHIRLI = b"HTRLF1"
SURUM = 1
ETIKET = 16
IZ = 8
BASLIK = 6 + 1 + 4 + IZ + 32  # 51
BAGLAM = b"hatirlaf-yedek-v1"


def _dosya_anahtari(gizli: X25519PrivateKey, gecici_acik: bytes) -> bytes:
    acik = gizli.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    paylasilan = gizli.exchange(X25519PublicKey.from_public_bytes(gecici_acik))
    # Tuzun sirasi Dart tarafiyla ayni olmali: gecici || alici.
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=gecici_acik + acik,
        info=BAGLAM,
    ).derive(paylasilan)


def _nonce(sira: int) -> bytes:
    return b"\x00" * 8 + sira.to_bytes(4, "big")


def _ek(baslik: bytes, sira: int, son_mu: bool) -> bytes:
    return baslik + sira.to_bytes(4, "big") + (b"\x01" if son_mu else b"\x00")


def coz(gizli: X25519PrivateKey, ham: bytes) -> bytes:
    if len(ham) < BASLIK:
        raise ValueError("Dosya baslik icin bile kisa.")
    if ham[:6] != SIHIRLI:
        raise ValueError("Bu bir hatirlaf yedegi degil.")
    if ham[6] != SURUM:
        raise ValueError(f"Bilinmeyen yedek surumu: {ham[6]}")

    baslik = ham[:BASLIK]
    parca_boyu = int.from_bytes(baslik[7:11], "big")
    beklenen_iz = baslik[11 : 11 + IZ]
    gecici_acik = baslik[11 + IZ : BASLIK]

    acik = gizli.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if hashlib.sha256(acik).digest()[:IZ] != beklenen_iz:
        raise ValueError("Bu yedek baska bir anahtara sifrelenmis.")

    aes = AESGCM(_dosya_anahtari(gizli, gecici_acik))
    tam = parca_boyu + ETIKET
    cikti = bytearray()
    konum, sira = BASLIK, 0
    while True:
        kalan = len(ham) - konum
        if kalan < ETIKET:
            raise ValueError("Yedek budanmis.")
        son_mu = kalan <= tam
        bu = kalan if son_mu else tam
        cikti += aes.decrypt(
            _nonce(sira), ham[konum : konum + bu], _ek(baslik, sira, son_mu)
        )
        konum += bu
        sira += 1
        if son_mu:
            return bytes(cikti)


def _env_oku(ad: str) -> str | None:
    """Depo kokundeki .env'den tek bir deger okur."""
    kok = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    yol = os.path.join(kok, ".env")
    if not os.path.exists(yol):
        return None
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir or satir.startswith("#") or "=" not in satir:
                continue
            k, _, v = satir.partition("=")
            if k.strip() == ad:
                return v.strip().strip('"').strip("'") or None
    return None


def _anahtar_yolu(verilen: str | None) -> str:
    """Once komut satiri, sonra .env."""
    if verilen:
        return verilen
    envden = _env_oku("YEDEK_GIZLI_ANAHTAR")
    if not envden:
        raise SystemExit(
            "hata: gizli anahtar verilmedi ve .env icinde "
            "YEDEK_GIZLI_ANAHTAR yok."
        )
    return envden


def _anahtar_mi(yol: str) -> bool:
    """32 baytlik ham bir dosya mi? Ilk argumanin anahtar mi yoksa
    girdi dosyasi mi oldugunu ayirt etmek icin."""
    try:
        return os.path.isfile(os.path.expanduser(yol)) and \
            os.path.getsize(os.path.expanduser(yol)) == 32
    except OSError:
        return False


def _anahtari_oku(yol: str) -> X25519PrivateKey:
    with open(os.path.expanduser(yol), "rb") as f:
        ham = f.read()
    if len(ham) != 32:
        raise ValueError(f"Gizli anahtar 32 bayt olmali, {len(ham)} bayt okundu.")
    return X25519PrivateKey.from_private_bytes(ham)


def _tek(gizli: X25519PrivateKey, girdi: str, cikti: str) -> None:
    with open(girdi, "rb") as f:
        duz = coz(gizli, f.read())
    # Once gecici dosyaya yaz, sonra tasi: yarim cozulmus bir dosya
    # asil adi almasin.
    gec = cikti + ".yarim"
    with open(gec, "wb") as f:
        f.write(duz)
    os.replace(gec, cikti)
    print(f"  {os.path.basename(girdi)} -> {os.path.basename(cikti)}  ({len(duz)} bayt)")


def main() -> int:
    arg = sys.argv[1:]
    if not arg:
        print(__doc__, file=sys.stderr)
        return 1

    # Ilk arguman ancak 32 baytlik bir dosyaysa anahtardir; degilse
    # girdi sayilir ve anahtar .env'den gelir.
    if _anahtar_mi(arg[0]):
        anahtar, arg = arg[0], arg[1:]
    else:
        anahtar = None
    if not arg:
        print(__doc__, file=sys.stderr)
        return 1

    gizli = _anahtari_oku(_anahtar_yolu(anahtar))

    if arg[0] == "--klasor":
        if len(arg) != 2:
            print(__doc__, file=sys.stderr)
            return 1
        kok = os.path.expanduser(arg[1])
        sayi, hata = 0, 0
        for dizin, _, dosyalar in os.walk(kok):
            for ad in sorted(dosyalar):
                if not ad.endswith(".hyz"):
                    continue
                girdi = os.path.join(dizin, ad)
                try:
                    _tek(gizli, girdi, girdi[: -len(".hyz")])
                    sayi += 1
                except Exception as e:  # noqa: BLE001
                    # Tek bozuk dosya butun klasoru durdurmasin.
                    print(f"  HATA {ad}: {e}", file=sys.stderr)
                    hata += 1
        print(f"\n{sayi} dosya cozuldu, {hata} hata.")
        return 1 if hata else 0

    girdi = os.path.expanduser(arg[0])
    if len(arg) >= 2:
        cikti = os.path.expanduser(arg[1])
    elif girdi.endswith(".hyz"):
        cikti = girdi[: -len(".hyz")]
    else:
        cikti = girdi + ".cozulmus"
    _tek(gizli, girdi, cikti)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
