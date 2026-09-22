#!/usr/bin/env python3
"""B2 tarafinin yayina hazir olup olmadigini denetler.

    tool/b2_saglik.py

Hicbir sey degistirmez (tek istisna: yazma yetkisini sinamak icin kucuk
bir nesne yazip master anahtarla siler). Yayindan once calistirin:
burada gecmeyen bir sey telefonda sessiz bir arizaya donusur.
"""
import base64
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yedek_indir import env_oku, istek, _yeniden_dene, YETKI, KOK  # noqa: E402

TAMAM, UYARI, HATA = "✓", "!", "✗"
sonuclar = []


def yaz(durum: str, baslik: str, ayrinti: str = "") -> None:
    sonuclar.append(durum)
    print(f"  {durum} {baslik}" + (f"  —  {ayrinti}" if ayrinti else ""))


def main() -> int:
    env = env_oku()
    with open(os.path.join(KOK, "yedek.json"), encoding="utf-8") as f:
        ayar = json.load(f)

    print("uygulama anahtari (APK'ya gomulecek olan)")
    temel = base64.b64encode(
        f"{ayar['B2_KEY_ID']}:{ayar['B2_APP_KEY']}".encode()).decode()
    r = urllib.request.Request(YETKI, headers={"Authorization": f"Basic {temel}"})
    try:
        o = _yeniden_dene(
            lambda: json.loads(urllib.request.urlopen(r, timeout=60).read()),
            "yetki")
    except urllib.error.HTTPError as e:
        yaz(HATA, "yetkilendirme", f"HTTP {e.code} — anahtar gecersiz mi?")
        return 1
    yaz(TAMAM, "yetkilendirme", o["apiUrl"])

    izin = o["allowed"]
    yetkiler = sorted(izin["capabilities"])
    if yetkiler == ["writeFiles"]:
        yaz(TAMAM, "yetkiler", "yalnizca writeFiles")
    else:
        yaz(HATA, "yetkiler", f"{yetkiler} — fazla yetkili!")

    if izin.get("bucketId") == ayar["B2_BUCKET_ID"]:
        yaz(TAMAM, "kova kisiti", izin.get("bucketName", ""))
    else:
        yaz(HATA, "kova kisiti",
            f"anahtar {izin.get('bucketId')} kovasina bagli, "
            f"yedek.json {ayar['B2_BUCKET_ID']} diyor")

    kova_adi = izin.get("bucketName")
    onerilen = o.get("recommendedPartSize", 0)
    enkucuk = o.get("absoluteMinimumPartSize", 0)
    yaz(TAMAM, "parca esigi",
        f"{onerilen / 1e6:.0f} MB ustu parcali gider "
        f"(en kucuk {enkucuk / 1e6:.0f} MB)")

    # Yazma gecmeli.
    ad = f"_saglik-{secrets.token_hex(4)}.txt"
    govde = b"saglik denetimi"
    import hashlib
    try:
        u = istek(f"{o['apiUrl']}/b2api/v2/b2_get_upload_url",
                  {"bucketId": izin["bucketId"]},
                  {"Authorization": o["authorizationToken"]})
        ry = urllib.request.Request(
            u["uploadUrl"], data=govde,
            headers={"Authorization": u["authorizationToken"],
                     "X-Bz-File-Name": urllib.parse.quote(ad),
                     "Content-Type": "text/plain",
                     "X-Bz-Content-Sha1": hashlib.sha1(govde).hexdigest()})
        _yeniden_dene(lambda: urllib.request.urlopen(ry, timeout=60).read(),
                      "yazma")
        yaz(TAMAM, "yazma", "telefon yukleyebiliyor")
    except Exception as e:  # noqa: BLE001
        yaz(HATA, "yazma", f"{type(e).__name__}: {e}")
        return 1

    # Okuma GECMEMELI.
    r2 = urllib.request.Request(
        f"{o['downloadUrl']}/file/{kova_adi}/{urllib.parse.quote(ad)}",
        headers={"Authorization": o["authorizationToken"]})
    try:
        _yeniden_dene(lambda: urllib.request.urlopen(r2, timeout=60).read(1),
                      "okuma")
        yaz(HATA, "okuma reddi", "anahtar OKUYABILIYOR — sizinca arsiv acilir")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            yaz(TAMAM, "okuma reddi", f"HTTP {e.code}")
        else:
            yaz(HATA, "okuma reddi", f"beklenmedik HTTP {e.code}")
    except Exception as e:  # noqa: BLE001
        yaz(UYARI, "okuma reddi", f"BELIRSIZ — baglanilamadi ({type(e).__name__})")

    print("\nmaster anahtar (yalnizca bu PC)")
    temel2 = base64.b64encode(
        f"{env['B2_MASTER_API_KEYID']}:{env['B2_MASTER_API_KEY']}".encode()).decode()
    rm = urllib.request.Request(YETKI, headers={"Authorization": f"Basic {temel2}"})
    try:
        om = _yeniden_dene(
            lambda: json.loads(urllib.request.urlopen(rm, timeout=60).read()),
            "master yetki")
        yaz(TAMAM, "yetkilendirme", f"hesap {om['accountId']}")
    except Exception as e:  # noqa: BLE001
        yaz(HATA, "yetkilendirme", f"{type(e).__name__} — kayitlari INDIREMEZSINIZ")
        return 1

    kovalar = istek(f"{om['apiUrl']}/b2api/v2/b2_list_buckets",
                    {"accountId": om["accountId"], "bucketId": ayar["B2_BUCKET_ID"]},
                    {"Authorization": om["authorizationToken"]})
    if not kovalar.get("buckets"):
        yaz(HATA, "kova", "bulunamadi")
        return 1
    kova = kovalar["buckets"][0]
    if kova["bucketType"] == "allPrivate":
        yaz(TAMAM, "kova gizliligi", "allPrivate")
    else:
        yaz(HATA, "kova gizliligi", f"{kova['bucketType']} — herkese acik!")

    # Temizlik + mevcut durum.
    liste = istek(f"{om['apiUrl']}/b2api/v2/b2_list_file_names",
                  {"bucketId": ayar["B2_BUCKET_ID"], "maxFileCount": 1000},
                  {"Authorization": om["authorizationToken"]})
    dosyalar = liste.get("files", [])
    silinen = 0
    for d in dosyalar:
        if d["fileName"].startswith("_saglik-"):
            istek(f"{om['apiUrl']}/b2api/v2/b2_delete_file_version",
                  {"fileName": d["fileName"], "fileId": d["fileId"]},
                  {"Authorization": om["authorizationToken"]})
            silinen += 1
    yaz(TAMAM, "silme (master)", f"{silinen} denetim nesnesi temizlendi")

    gercek = [d for d in dosyalar if not d["fileName"].startswith("_saglik-")]
    toplam = sum(d["contentLength"] for d in gercek)
    yaz(TAMAM, "kovadaki yedek",
        f"{len(gercek)} dosya, {toplam / 1e6:.1f} MB "
        f"(10 GB ucretsizin %{toplam / 1e10 * 100:.2f}'i)")

    # Cozme anahtari yerinde mi? Kaybolursa arsiv acilamaz.
    gizli = os.path.expanduser(env.get("YEDEK_GIZLI_ANAHTAR", ""))
    if os.path.exists(gizli) and os.path.getsize(gizli) == 32:
        yaz(TAMAM, "gizli anahtar", gizli)
    else:
        yaz(HATA, "gizli anahtar", f"{gizli} yok/bozuk — yedekler ACILAMAZ")

    print()
    if HATA in sonuclar:
        print("YAYINLAMAYIN: yukarida ✗ var.")
        return 1
    if UYARI in sonuclar:
        print("Gecti, ama ! isaretli satirlar dogrulanamadi.")
        return 0
    print("B2 tarafi yayina hazir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
