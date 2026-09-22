#!/usr/bin/env python3
"""B2'deki yedekleri indirir ve ister cozer.

    tool/yedek_indir.py                 indir + coz (varsayilan)
    tool/yedek_indir.py --sadece-indir  cozme, sifreli birak
    tool/yedek_indir.py --dizin YOL     baska bir hedef
    tool/yedek_indir.py --liste         hicbir sey indirme, ne var yaz

Hedef VARSAYILAN OLARAK DEPONUN DISINDA (.env icindeki
YEDEK_INDIRME_DIZINI, ontanimli ~/hatirlaf-yedekler). Bu bilerek boyle:
cozulmus kayitlar insanlarin hayat hikayeleri ve depo herkese acik.
Depo icine indirilirse tek bir `git add -A` hepsini yayinlar.

    <dizin>/sifreli/<cihaz>/<hatira>/ses.m4a.hyz    B2'den geldigi gibi
    <dizin>/cozulmus/<cihaz>/<hatira>/ses.m4a       cozulmus hali
                                      bilgi.json

Sifreli kopyalar saklanir: cozme hatasinda ya da anahtar degisiminde
tekrar denenebilsin. B2'den HICBIR SEY SILINMEZ.
"""
import base64
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yedek_coz import coz, _anahtari_oku  # noqa: E402

KOK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
YETKI = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"
VARSAYILAN_DIZIN = "~/hatirlaf-yedekler"


def env_oku() -> dict:
    yol = os.path.join(KOK, ".env")
    if not os.path.exists(yol):
        raise SystemExit("hata: .env yok.")
    d = {}
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if not satir or satir.startswith("#") or "=" not in satir:
                continue
            k, _, v = satir.partition("=")
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d


def _yeniden_dene(islem, ad: str, kez: int = 6):
    """Gecici ag/TLS hatalarinda tekrar dener; kalicilari birakir."""
    for deneme in range(1, kez + 1):
        try:
            return islem()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ssl.SSLError, OSError) as e:
            if deneme == kez:
                raise
            bekle = min(15, 2 ** (deneme - 1))
            print(f"  {ad}: gecici hata ({type(e).__name__}), {bekle} sn sonra tekrar",
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


def guvenli_yol(kok: str, ad: str) -> str:
    """Kova adini yerel yola cevirir; kokun disina cikamaz.

    Adlar telefondan geliyor. Uygulama zaten temizliyor ama indirme
    tarafi ona guvenmemeli: tek bir ".." bileseni ev dizinine yazardi.
    """
    parcalar = []
    for p in ad.split("/"):
        p = p.strip()
        if not p or p == "." or p == "..":
            continue
        parcalar.append(p.replace(os.sep, "_"))
    if not parcalar:
        parcalar = ["adsiz"]
    tam = os.path.abspath(os.path.join(kok, *parcalar))
    if not tam.startswith(os.path.abspath(kok) + os.sep):
        raise ValueError(f"guvensiz ad: {ad}")
    return tam


def main() -> int:
    arg = sys.argv[1:]
    sadece_indir = "--sadece-indir" in arg
    sadece_liste = "--liste" in arg
    dizin = None
    if "--dizin" in arg:
        dizin = arg[arg.index("--dizin") + 1]

    env = env_oku()
    kok = os.path.expanduser(
        dizin or env.get("YEDEK_INDIRME_DIZINI") or VARSAYILAN_DIZIN
    )

    # Depo icine indirmeyi reddet: cozulmus kayitlar oraya ait degil.
    if os.path.abspath(kok).startswith(os.path.abspath(KOK) + os.sep) or \
            os.path.abspath(kok) == os.path.abspath(KOK):
        raise SystemExit(
            f"hata: hedef depo icinde ({kok}).\n"
            "      Cozulmus kayitlar depoya girmemeli. Depo disinda bir\n"
            "      yol secin (.env icindeki YEDEK_INDIRME_DIZINI)."
        )

    temel = base64.b64encode(
        f"{env['B2_MASTER_API_KEYID']}:{env['B2_MASTER_API_KEY']}".encode()
    ).decode()
    o = istek(YETKI, basliklar={"Authorization": f"Basic {temel}"})

    with open(os.path.join(KOK, "yedek.json"), encoding="utf-8") as f:
        kova_id = json.load(f)["B2_BUCKET_ID"]
    kovalar = istek(
        f"{o['apiUrl']}/b2api/v2/b2_list_buckets",
        {"accountId": o["accountId"], "bucketId": kova_id},
        {"Authorization": o["authorizationToken"]},
    )
    kova_adi = kovalar["buckets"][0]["bucketName"]
    print(f"kova   : {kova_adi}")
    print(f"hedef  : {kok}")

    # Butun dosyalari say (sayfali).
    dosyalar, sonraki = [], None
    while True:
        y = istek(
            f"{o['apiUrl']}/b2api/v2/b2_list_file_names",
            {"bucketId": kova_id, "maxFileCount": 1000,
             **({"startFileName": sonraki} if sonraki else {})},
            {"Authorization": o["authorizationToken"]},
        )
        dosyalar += [d for d in y.get("files", [])
                     if not d["fileName"].startswith("_deneme/")]
        sonraki = y.get("nextFileName")
        if not sonraki:
            break

    if not dosyalar:
        print("\nKovada hic yedek yok.")
        return 0

    print(f"dosya  : {len(dosyalar)}")
    if sadece_liste:
        toplam = sum(d["contentLength"] for d in dosyalar)
        for d in sorted(dosyalar, key=lambda x: x["fileName"]):
            print(f"  {d['contentLength']:>12,}  {d['fileName']}")
        print(f"\ntoplam {toplam:,} bayt")
        return 0

    sifreli_kok = os.path.join(kok, "sifreli")
    cozulmus_kok = os.path.join(kok, "cozulmus")
    os.makedirs(sifreli_kok, exist_ok=True)
    # 0700: kisisel kayitlar, baska kullanicilar okumasin.
    os.chmod(kok, 0o700)

    gizli = None
    if not sadece_indir:
        gizli = _anahtari_oku(env["YEDEK_GIZLI_ANAHTAR"])

    yeni = atlanan = cozulen = hata = 0
    for d in sorted(dosyalar, key=lambda x: x["fileName"]):
        ad = d["fileName"]
        hedef = guvenli_yol(sifreli_kok, ad)
        os.makedirs(os.path.dirname(hedef), exist_ok=True)

        if os.path.exists(hedef) and os.path.getsize(hedef) == d["contentLength"]:
            atlanan += 1
        else:
            url = f"{o['downloadUrl']}/file/{kova_adi}/{urllib.parse.quote(ad)}"
            r = urllib.request.Request(
                url, headers={"Authorization": o["authorizationToken"]})

            def indir():
                with urllib.request.urlopen(r, timeout=300) as y:
                    return y.read()

            try:
                ham = _yeniden_dene(indir, ad)
            except Exception as e:  # noqa: BLE001
                print(f"  HATA indirme {ad}: {e}", file=sys.stderr)
                hata += 1
                continue
            # Once gecici ada yaz: yarim inen dosya tam sanilmasin.
            with open(hedef + ".yarim", "wb") as f:
                f.write(ham)
            os.replace(hedef + ".yarim", hedef)
            yeni += 1
            print(f"  indi  {d['contentLength']:>12,}  {ad}")

        if gizli is not None and ad.endswith(".hyz"):
            c_hedef = guvenli_yol(cozulmus_kok, ad[: -len(".hyz")])
            if os.path.exists(c_hedef):
                continue
            os.makedirs(os.path.dirname(c_hedef), exist_ok=True)
            try:
                with open(hedef, "rb") as f:
                    duz = coz(gizli, f.read())
                with open(c_hedef + ".yarim", "wb") as f:
                    f.write(duz)
                os.replace(c_hedef + ".yarim", c_hedef)
                cozulen += 1
            except Exception as e:  # noqa: BLE001
                # Tek bozuk dosya butun indirmeyi durdurmasin.
                print(f"  HATA cozme {ad}: {e}", file=sys.stderr)
                hata += 1

    print(f"\n{yeni} yeni, {atlanan} zaten vardi, {cozulen} cozuldu, {hata} hata.")
    print(f"Cozulmus kayitlar: {cozulmus_kok}")
    if hata:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
