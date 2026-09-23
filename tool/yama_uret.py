#!/usr/bin/env python3
"""Iki APK arasinda fark yamasi uretir.

APK bir zip dosyasi ve iki surum arasinda girdilerin neredeyse hepsi bayt
bayt ayni kaliyor (1.1.0 -> 1.2.0'da 560 girdinin 554'u). Degismeyen kisim
telefonda zaten duruyor: kurulu APK. Yama bu yuzden "sunu kurulu APK'nin
su ofsetinden kopyala" komutlariyla, degisenler de govdede ham olarak
tasiniyor; 22 MB yerine ~3 MB.

Telefondaki karsiligi: lib/services/yama.dart. Bicim oradaki sinif
belgesinde yaziyor; ikisi birlikte degismeli.

    tool/yama_uret.py eski.apk yeni.apk cikti.yama
"""

import hashlib
import json
import os
import struct
import sys
import zipfile
import zlib

SIHIR = b"HTRLYAMA1"
KOPYALA = 1
YENI = 2

# Bundan kucuk girdi icin komut yazmaya degmez: komutun kendisi ~10 bayt
# ve zlib zaten benzer baytlari sikistiriyor.
EN_KUCUK_KOPYA = 1024


def _varint(n: int) -> bytes:
    b = bytearray()
    while True:
        p = n & 0x7F
        n >>= 7
        b.append(p | (0x80 if n else 0))
        if not n:
            return bytes(b)


def _veri_araligi(z: zipfile.ZipFile, girdi: zipfile.ZipInfo):
    """Girdinin sikistirilmis verisinin dosyadaki [bas, son) araligi.

    Merkezi dizindeki `header_offset` yerel baslığı gosteriyor; verinin
    yeri ancak yerel basliktaki ad ve ek alan uzunluklari okunarak
    bulunabiliyor (ikisi merkezi dizindekinden farkli olabilir).
    """
    z.fp.seek(girdi.header_offset)
    yerel = z.fp.read(30)
    if yerel[:4] != b"PK\x03\x04":
        raise ValueError(f"{girdi.filename}: yerel baslik bulunamadi")
    ad_uzunluk, ek_uzunluk = struct.unpack("<HH", yerel[26:30])
    bas = girdi.header_offset + 30 + ad_uzunluk + ek_uzunluk
    return bas, bas + girdi.compress_size


def uret(eski_yol: str, yeni_yol: str, cikti_yol: str) -> dict:
    with open(eski_yol, "rb") as f:
        eski = f.read()
    with open(yeni_yol, "rb") as f:
        yeni = f.read()

    with zipfile.ZipFile(eski_yol) as ze, zipfile.ZipFile(yeni_yol) as zy:
        # Eski APK'daki girdiler: ayni ad + ayni CRC + ayni sikistirilmis
        # boyut demek, neredeyse kesinlikle ayni baytlar demek. "Neredeyse"
        # yetmez, asagida birebir karsilastiriliyor.
        eski_ind = {}
        for g in ze.infolist():
            eski_ind[(g.filename, g.CRC, g.compress_size)] = _veri_araligi(ze, g)

        komutlar = bytearray()
        govde = bytearray()
        imlec = 0
        kopyalanan = 0

        def govdeye_al(bas: int, son: int) -> None:
            nonlocal imlec
            if son > bas:
                komutlar.append(YENI)
                komutlar.extend(_varint(son - bas))
                govde.extend(yeni[bas:son])
                imlec = son

        for g in sorted(zy.infolist(), key=lambda x: x.header_offset):
            anahtar = (g.filename, g.CRC, g.compress_size)
            if anahtar not in eski_ind or g.compress_size < EN_KUCUK_KOPYA:
                continue
            y_bas, y_son = _veri_araligi(zy, g)
            e_bas, e_son = eski_ind[anahtar]
            if eski[e_bas:e_son] != yeni[y_bas:y_son]:
                continue

            # Girdiler arasinda kalan her sey (yerel basliklar, hizalama
            # dolgusu, imza blogu, merkezi dizin) govdeye gidiyor.
            govdeye_al(imlec, y_bas)
            komutlar.append(KOPYALA)
            komutlar.extend(_varint(e_bas))
            komutlar.extend(_varint(y_son - y_bas))
            imlec = y_son
            kopyalanan += y_son - y_bas

        govdeye_al(imlec, len(yeni))

    sikistirilmis = zlib.compress(bytes(govde), 9)
    with open(cikti_yol, "wb") as f:
        f.write(SIHIR)
        f.write(struct.pack("<Q", len(yeni)))
        f.write(hashlib.sha256(eski).digest())
        f.write(struct.pack("<I", len(komutlar)))
        f.write(bytes(komutlar))
        f.write(struct.pack("<Q", len(govde)))
        f.write(sikistirilmis)

    return {
        "yama": cikti_yol,
        "boyut": os.path.getsize(cikti_yol),
        "sha256": hashlib.sha256(open(cikti_yol, "rb").read()).hexdigest(),
        "kaynakSha256": hashlib.sha256(eski).hexdigest(),
        "hedefSha256": hashlib.sha256(yeni).hexdigest(),
        "hedefBoyut": len(yeni),
        "kopyalanan": kopyalanan,
        "govde": len(govde),
    }


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    bilgi = uret(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(bilgi, ensure_ascii=False))


if __name__ == "__main__":
    main()
