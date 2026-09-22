#!/usr/bin/env python3
"""B2 tarafini bastan kurar ve yedek.json'u yazar.

    tool/b2_kur.py [kova-adi]          (varsayilan: hatirlaf-yedek)
    tool/b2_kur.py --deneme            hicbir sey olusturmaz, durumu yazar

Sirayla:

  1. X25519 anahtar cifti yoksa uretir (.env'deki YEDEK_GIZLI_ANAHTAR).
  2. Master anahtarla B2'ye baglanir.
  3. Kova yoksa olusturur (allPrivate).
  4. YALNIZCA writeFiles yetkili, o kovaya kisitli yeni bir anahtar uretir.
  5. yedek.json'u yazar.
  6. Yeni anahtarla yazmayi dener (gecmeli) ve okumayi dener (GECMEMELI).

Master anahtar .env'de kalir ve derlemeye ASLA girmez. yedek.json'a
yalnizca kisitli anahtar yazilir; o zaten APK'nin icine gomulecek.

Bu script kendi kendine yeter: baska bir makineye tek basina kopyalanip
calistirilabilsin diye .env okuyucusu yedek_coz.py'dekiyle ayni.
"""
import base64
import json
import os
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization

KOK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
YETKI = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"

# Uygulamadaki anahtar yalnizca bunu tasir. "readFiles" ya da
# "deleteFiles" eklemeyin: APK'yi parcalayan herkes o yetkiyi kazanir.
UYGULAMA_YETKILERI = ["writeFiles"]


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


def istek(url: str, govde=None, basliklar=None) -> dict:
    ham = None
    b = dict(basliklar or {})
    if govde is not None:
        ham = json.dumps(govde).encode()
        b["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=ham, headers=b)
    try:
        with urllib.request.urlopen(r, timeout=60) as y:
            return json.loads(y.read().decode())
    except urllib.error.HTTPError as e:
        govde_metni = e.read().decode(errors="replace")
        try:
            j = json.loads(govde_metni)
            mesaj = f"{j.get('message', govde_metni)} ({j.get('code', '')})"
        except json.JSONDecodeError:
            mesaj = govde_metni[:300]
        raise SystemExit(f"hata: B2 {e.code}: {mesaj}") from e


def yetkilen(key_id: str, key: str) -> dict:
    temel = base64.b64encode(f"{key_id}:{key}".encode()).decode()
    return istek(YETKI, basliklar={"Authorization": f"Basic {temel}"})


def anahtari_hazirla(yol: str, deneme: bool) -> bytes:
    """Gizli anahtar yoksa uretir; acik esini doner."""
    tam = os.path.expanduser(yol)
    if os.path.exists(tam):
        with open(tam, "rb") as f:
            ham = f.read()
        if len(ham) != 32:
            raise SystemExit(f"hata: {tam} 32 bayt degil ({len(ham)}).")
        gizli = X25519PrivateKey.from_private_bytes(ham)
        print(f"  gizli anahtar zaten var: {tam}")
    else:
        if deneme:
            print(f"  [deneme] gizli anahtar uretilecekti: {tam}")
            return b"\x00" * 32
        gizli = X25519PrivateKey.generate()
        ham = gizli.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        os.makedirs(os.path.dirname(tam) or ".", exist_ok=True)
        bayrak = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        with os.fdopen(os.open(tam, bayrak, 0o600), "wb") as f:
            f.write(ham)
        print(f"  gizli anahtar URETILDI: {tam}")
        print("  >>> BUNU SIMDI YEDEKLEYIN. Kaybolursa yedekler acilamaz.")
    return gizli.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def kovayi_hazirla(o: dict, ad: str, deneme: bool) -> str:
    mevcut = istek(
        f"{o['apiUrl']}/b2api/v2/b2_list_buckets",
        {"accountId": o["accountId"], "bucketName": ad},
        {"Authorization": o["authorizationToken"]},
    )
    for k in mevcut.get("buckets", []):
        if k["bucketName"] == ad:
            print(f"  kova zaten var: {ad} ({k['bucketId']}, {k['bucketType']})")
            if k["bucketType"] != "allPrivate":
                print("  UYARI: kova ozel degil! Konsoldan allPrivate yapin.")
            return k["bucketId"]

    if deneme:
        print(f"  [deneme] kova olusturulacakti: {ad}")
        return "DENEME"
    yeni = istek(
        f"{o['apiUrl']}/b2api/v2/b2_create_bucket",
        {
            "accountId": o["accountId"],
            "bucketName": ad,
            "bucketType": "allPrivate",
        },
        {"Authorization": o["authorizationToken"]},
    )
    print(f"  kova OLUSTURULDU: {ad} ({yeni['bucketId']})")
    return yeni["bucketId"]


def eski_anahtarlari_goster(o: dict, kova: str) -> None:
    y = istek(
        f"{o['apiUrl']}/b2api/v2/b2_list_keys",
        {"accountId": o["accountId"], "maxKeyCount": 100},
        {"Authorization": o["authorizationToken"]},
    )
    eskiler = [
        k for k in y.get("keys", [])
        if k.get("keyName", "").startswith("hatirlaf-yaz")
    ]
    if eskiler:
        print(f"  not: {len(eskiler)} eski 'hatirlaf-yaz' anahtari duruyor.")
        print("       Silmiyorum: sahadaki eski APK'lar hala onlari kullaniyor.")


def anahtar_uret(o: dict, kova: str, deneme: bool) -> tuple:
    ad = "hatirlaf-yaz-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if deneme:
        print(f"  [deneme] anahtar uretilecekti: {ad} {UYGULAMA_YETKILERI}")
        return "DENEME", "DENEME"
    y = istek(
        f"{o['apiUrl']}/b2api/v2/b2_create_key",
        {
            "accountId": o["accountId"],
            "capabilities": UYGULAMA_YETKILERI,
            "keyName": ad,
            "bucketId": kova,
        },
        {"Authorization": o["authorizationToken"]},
    )
    print(f"  anahtar URETILDI: {ad}  yetkiler={y['capabilities']}")
    if set(y["capabilities"]) - set(UYGULAMA_YETKILERI):
        raise SystemExit("hata: anahtar beklenenden fazla yetkili, durduruldu.")
    return y["applicationKeyId"], y["applicationKey"]


def dogrula(key_id: str, key: str, kova_adi: str) -> bool:
    """Yeni anahtar yazabilmeli, OKUYAMAMALI."""
    o = yetkilen(key_id, key)
    print(f"  yetkiler (sunucunun dedigi): {o['allowed']['capabilities']}")

    ad = f"_kurulum-denemesi-{secrets.token_hex(4)}.txt"
    govde = b"hatirlaf kurulum denemesi"
    import hashlib

    u = istek(
        f"{o['apiUrl']}/b2api/v2/b2_get_upload_url",
        {"bucketId": o["allowed"]["bucketId"]},
        {"Authorization": o["authorizationToken"]},
    )
    r = urllib.request.Request(
        u["uploadUrl"],
        data=govde,
        headers={
            "Authorization": u["authorizationToken"],
            "X-Bz-File-Name": urllib.parse.quote(ad),
            "Content-Type": "text/plain",
            "X-Bz-Content-Sha1": hashlib.sha1(govde).hexdigest(),
        },
    )
    with urllib.request.urlopen(r, timeout=60) as y:
        json.loads(y.read().decode())
    print("  yazma  : gecti ✓")

    # Okuma GECMEMELI. Gecerse anahtar fazla yetkili demektir.
    okundu = False
    try:
        r2 = urllib.request.Request(
            f"{o['downloadUrl']}/file/{kova_adi}/{urllib.parse.quote(ad)}",
            headers={"Authorization": o["authorizationToken"]},
        )
        with urllib.request.urlopen(r2, timeout=60):
            okundu = True
    except urllib.error.HTTPError as e:
        print(f"  okuma  : reddedildi ✓ (HTTP {e.code})")
    if okundu:
        print("  okuma  : GECTI ✗  ANAHTAR FAZLA YETKILI!")
        return False
    return True


def deneme_dosyasini_sil(o: dict, kova: str) -> None:
    """Master anahtarla kurulum denemesi dosyalarini temizler."""
    y = istek(
        f"{o['apiUrl']}/b2api/v2/b2_list_file_names",
        {"bucketId": kova, "prefix": "_kurulum-denemesi-", "maxFileCount": 100},
        {"Authorization": o["authorizationToken"]},
    )
    for d in y.get("files", []):
        istek(
            f"{o['apiUrl']}/b2api/v2/b2_delete_file_version",
            {"fileName": d["fileName"], "fileId": d["fileId"]},
            {"Authorization": o["authorizationToken"]},
        )
    if y.get("files"):
        print(f"  {len(y['files'])} deneme dosyasi silindi")


def main() -> int:
    arg = [a for a in sys.argv[1:] if a != "--deneme"]
    deneme = "--deneme" in sys.argv[1:]
    kova_adi = arg[0] if arg else "hatirlaf-yedek"

    env = env_oku()
    key_id = env.get("B2_MASTER_API_KEYID", "")
    key = env.get("B2_MASTER_API_KEY", "")
    if not key_id or not key:
        raise SystemExit(
            "hata: .env icinde B2_MASTER_API_KEYID ve B2_MASTER_API_KEY olmali.\n"
            "      Backblaze konsolu -> Application Keys -> Master Application Key.\n"
            "      Gizli kisim yalnizca uretildigi anda bir kez gosterilir."
        )

    print("1/6 sifreleme anahtari")
    acik = anahtari_hazirla(env.get("YEDEK_GIZLI_ANAHTAR", ""), deneme)

    print("2/6 B2 baglantisi")
    o = yetkilen(key_id, key)
    yetki = o["allowed"]["capabilities"]
    print(f"  hesap {o['accountId']}, {len(yetki)} yetki")
    if "writeKeys" not in yetki:
        raise SystemExit("hata: bu anahtar yeni anahtar uretemez (writeKeys yok).")

    print("3/6 kova")
    kova = kovayi_hazirla(o, kova_adi, deneme)
    eski_anahtarlari_goster(o, kova)

    print("4/6 uygulama anahtari (yalnizca writeFiles)")
    yeni_id, yeni_key = anahtar_uret(o, kova, deneme)

    print("5/6 yedek.json")
    veri = {
        "B2_KEY_ID": yeni_id,
        "B2_APP_KEY": yeni_key,
        "B2_BUCKET_ID": kova,
        "YEDEK_ALICI_ANAHTARI": base64.b64encode(acik).decode(),
    }
    if deneme:
        print("  [deneme] yazilacakti:")
        print("   ", json.dumps({**veri, "B2_APP_KEY": "<gizli>"}, indent=2))
        print("\n--- DENEME: hicbir sey olusturulmadi ---")
        return 0

    yol = os.path.join(KOK, "yedek.json")
    bayrak = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(yol, bayrak, 0o600), "w", encoding="utf-8") as f:
        json.dump(veri, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  yazildi: {yol} (0600)")

    print("6/6 dogrulama")
    tamam = dogrula(yeni_id, yeni_key, kova_adi)
    deneme_dosyasini_sil(o, kova)

    if not tamam:
        print("\nKURULUM GUVENLI DEGIL: anahtar okuyabiliyor.", file=sys.stderr)
        return 1

    print("\nB2 hazir. Artik `tool/yayinla.sh feature \"...\"` calistirilabilir.")
    print("Gizli anahtari yedeklediginizden emin olun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
