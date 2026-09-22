#!/usr/bin/env python3
"""Deneme nesnesini master anahtarla indirip cozer ve karsilastirir.

    tool/b2_indir_coz.py <dizin>

Dizinde `nesne.txt` (B2 dosya adi) ve `kaynak.m4a` (duz metin) bekler;
ikisini de tool/b2_deneme.dart birakir. Indirdigini cozup kaynakla bayt
bayt karsilastirir, sonra deneme nesnesini B2'den siler.

Bu, yedegin gercekten ACILABILDIGININ kanitidir: sifrelemenin calistigini
gormek yetmez, geri donebildigini de gormek gerekir.
"""
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yedek_coz import coz, _anahtari_oku  # noqa: E402

KOK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
YETKI = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"


def env_oku() -> dict:
    d = {}
    with open(os.path.join(KOK, ".env"), encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir or satir.startswith("#") or "=" not in satir:
                continue
            k, _, v = satir.partition("=")
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def _yeniden_dene(islem, ad: str, kez: int = 6):
    """Gecici ag/TLS hatalarinda tekrar dener.

    B2'ye giderken ara sira TLS el sikismasi yarida kopuyor
    (WRONG_VERSION_NUMBER / UNEXPECTED_EOF). Kalici hatalari
    (HTTPError) oldugu gibi birakir: onlari tekrar denemek anlamsiz.
    """
    import ssl
    import time
    for deneme in range(1, kez + 1):
        try:
            return islem()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ssl.SSLError, OSError) as e:
            if deneme == kez:
                raise
            bekle = min(15, 2 ** (deneme - 1))
            print(f"  {ad}: gecici hata ({type(e).__name__}), "
                  f"{bekle} sn sonra tekrar ({deneme}/{kez - 1})",
                  file=sys.stderr)
            time.sleep(bekle)


def istek(url, govde=None, basliklar=None) -> dict:
    ham = None
    b = dict(basliklar or {})
    if govde is not None:
        ham = json.dumps(govde).encode()
        b["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=ham, headers=b)

    def calistir():
        with urllib.request.urlopen(r, timeout=60) as y:
            return json.loads(y.read().decode())

    return _yeniden_dene(calistir, url.rsplit("/", 1)[-1])


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1
    dizin = sys.argv[1]

    with open(os.path.join(dizin, "nesne.txt"), encoding="utf-8") as f:
        nesne = f.read().strip()
    with open(os.path.join(dizin, "kaynak.m4a"), "rb") as f:
        kaynak = f.read()

    env = env_oku()
    temel = base64.b64encode(
        f"{env['B2_MASTER_API_KEYID']}:{env['B2_MASTER_API_KEY']}".encode()
    ).decode()
    o = istek(YETKI, basliklar={"Authorization": f"Basic {temel}"})

    with open(os.path.join(KOK, "yedek.json"), encoding="utf-8") as f:
        ayar = json.load(f)
    kova_id = ayar["B2_BUCKET_ID"]

    kovalar = istek(
        f"{o['apiUrl']}/b2api/v2/b2_list_buckets",
        {"accountId": o["accountId"], "bucketId": kova_id},
        {"Authorization": o["authorizationToken"]},
    )
    kova_adi = kovalar["buckets"][0]["bucketName"]

    url = f"{o['downloadUrl']}/file/{kova_adi}/{urllib.parse.quote(nesne)}"
    r = urllib.request.Request(url, headers={"Authorization": o["authorizationToken"]})

    def indir():
        with urllib.request.urlopen(r, timeout=120) as y:
            return y.read()

    sifreli = _yeniden_dene(indir, "indirme")
    print(f"  indirildi  : {len(sifreli)} bayt")

    gizli = _anahtari_oku(env["YEDEK_GIZLI_ANAHTAR"])
    duz = coz(gizli, sifreli)
    print(f"  cozuldu    : {len(duz)} bayt")

    if duz != kaynak:
        print("  KARSILASTIRMA BASARISIZ", file=sys.stderr)
        print(f"    kaynak sha256 : {hashlib.sha256(kaynak).hexdigest()}",
              file=sys.stderr)
        print(f"    cozulen sha256: {hashlib.sha256(duz).hexdigest()}",
              file=sys.stderr)
        return 1
    print(f"  karsilastirma ✓ ({len(duz)} bayt, "
          f"sha256={hashlib.sha256(duz).hexdigest()[:16]}…)")

    # Deneme nesnesini temizle (master anahtar gerekir; uygulamanin
    # anahtari silemez, zaten silememeli).
    liste = istek(
        f"{o['apiUrl']}/b2api/v2/b2_list_file_names",
        {"bucketId": kova_id, "prefix": "_deneme/", "maxFileCount": 100},
        {"Authorization": o["authorizationToken"]},
    )
    for d in liste.get("files", []):
        istek(
            f"{o['apiUrl']}/b2api/v2/b2_delete_file_version",
            {"fileName": d["fileName"], "fileId": d["fileId"]},
            {"Authorization": o["authorizationToken"]},
        )
    if liste.get("files"):
        print(f"  {len(liste['files'])} deneme nesnesi silindi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
