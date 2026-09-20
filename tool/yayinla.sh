#!/usr/bin/env bash
# Yeni bir surum yayinlar.
#
#   tool/yayinla.sh 1.0.1 "Kayıt düğmesi büyütüldü."
#   tool/yayinla.sh 1.0.1 "Veri kaybı düzeltildi." --zorunlu
#
# Yaptigi sirayla:
#   1. pubspec.yaml'daki surumu yukseltir (surum adi + surum kodu).
#   2. Yayin anahtariyla imzali, mimariye ozel APK'lar derler.
#   3. Her APK'nin sha256 ozetini hesaplar, guncelleme.json'u yazar.
#   4. GitHub'da surum (release) olusturup APK'lari ekler.
#   5. guncelleme.json'u depoya iter.
#
# Telefonlardaki uygulamalar 4. ve 5. adimdan sonra, internete ciktiklari
# ilk acilista guncellemeyi kendiliginden indirir.
set -euo pipefail

cd "$(dirname "$0")/.."

SURUM="${1:-}"
NOTLAR="${2:-}"
ZORUNLU="false"
[[ "${3:-}" == "--zorunlu" ]] && ZORUNLU="true"

if [[ -z "$SURUM" ]]; then
  echo "kullanim: tool/yayinla.sh <surum> \"<neler degisti>\" [--zorunlu]" >&2
  echo "ornek  : tool/yayinla.sh 1.0.1 \"Kayıt düğmesi büyütüldü.\"" >&2
  exit 1
fi

if [[ ! "$SURUM" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "hata: surum 1.2.3 bicinde olmali (girilen: $SURUM)" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "hata: calisma dizininde kaydedilmemis degisiklik var." >&2
  echo "Once commit edin; yayinlanan surum depoda izlenebilir olmali." >&2
  exit 1
fi

# --- 1. imza anahtari yerinde mi? ---------------------------------------
#
# Bu kontrol sondan basa en onemlisi: debug anahtariyla imzalanmis bir APK
# telefonlara guncelleme olarak KURULAMAZ ve bu ancak kullanicinin
# telefonunda, sessizce fark edilir.
if [[ ! -f android/key.properties || ! -f android/hatirlaf.jks ]]; then
  echo "hata: yayin imza anahtari yok (android/key.properties + hatirlaf.jks)." >&2
  echo "Yedekten geri koyun. Bu anahtar olmadan guncelleme yayinlanamaz." >&2
  exit 1
fi

# --- 2. surum kodunu yukselt --------------------------------------------
ESKI_SATIR="$(grep -m1 '^version:' pubspec.yaml)"
ESKI_KOD="${ESKI_SATIR##*+}"
YENI_KOD=$((ESKI_KOD + 1))

echo "surum   : $SURUM+$YENI_KOD  (onceki: ${ESKI_SATIR#version: })"
sed -i "s|^version:.*|version: $SURUM+$YENI_KOD|" pubspec.yaml

# --- 3. derle ------------------------------------------------------------
#
# Mimariye ozel APK'lar: tek parca (universal) APK 60 MB, arm64'e ozel
# olan 22 MB. Guncelleme her surumde yeniden indirilecegi icin aradaki
# 38 MB her seferinde tekrar odenirdi.
echo "derleniyor…"
tool/flutter.sh build apk --release --split-per-abi

ABILER=(arm64-v8a armeabi-v7a x86_64)
for A in "${ABILER[@]}"; do
  [[ -f "build/app/outputs/flutter-apk/app-$A-release.apk" ]] || {
    echo "hata: app-$A-release.apk olusmadi" >&2; exit 1; }
done

# Surum kodunu APK'dan okumak icin aapt2 sart.
#
# Mimariye ozel derlemede Flutter surum kodunu kaydiriyor (armeabi-v7a
# +1000, arm64-v8a +2000, x86_64 +4000). Telefondaki kurulu kod bu yuzden
# pubspec'teki sayi degil. Kaydirmayi burada varsaymak yerine derlenmis
# APK'ya sorup ogreniyoruz; Flutter yarin kurali degistirse de dogru kalir.
AAPT="$(ls "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"/build-tools/*/aapt2 2>/dev/null | sort -V | tail -1)"
if [[ -z "$AAPT" ]]; then
  echo "hata: aapt2 bulunamadi (Android SDK build-tools)." >&2
  exit 1
fi

# --- 4. guncelleme.json --------------------------------------------------
#
# Her mimari icin ayri adres + ozet + boyut. Telefon kendi mimarisini
# (Build.SUPPORTED_ABIS) bilip dogru satiri seciyor.
python3 - "$AAPT" "$SURUM" "$NOTLAR" "$ZORUNLU" "${ABILER[@]}" <<'PY'
import hashlib, io, json, os, re, subprocess, sys

aapt, surum, notlar, zorunlu = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
abiler = sys.argv[5:]
KOK = "https://github.com/oktayelb/hatirlaf/releases/latest/download"

paketler = {}
for a in abiler:
    yol = f"build/app/outputs/flutter-apk/app-{a}-release.apk"

    basligi = subprocess.run(
        [aapt, "dump", "badging", yol],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[0]
    esles = re.search(r"versionCode='(\d+)'", basligi)
    if not esles:
        sys.exit(f"hata: {yol} icindeki versionCode okunamadi")
    kod = int(esles.group(1))

    ozet = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            ozet.update(parca)
    boyut = os.path.getsize(yol)

    paketler[a] = {
        "surumKodu": kod,
        "apkUrl": f"{KOK}/hatirlaf-{a}.apk",
        "sha256": ozet.hexdigest(),
        "boyut": boyut,
    }
    print(f"  {a:<14} kod={kod:<6} {boyut / 1048576:5.1f} MB  {ozet.hexdigest()[:16]}…")

veri = {
    "surumAdi": surum,
    "notlar": notlar,
    "zorunlu": zorunlu == "true",
    "paketler": paketler,
}
io.open("guncelleme.json", "w", encoding="utf-8").write(
    json.dumps(veri, ensure_ascii=False, indent=2) + "\n"
)
PY

# --- 5. GitHub surumu ----------------------------------------------------
#
# SIRA ONEMLI: once APK yuklenir, sonra guncelleme.json itilir. Ters sirada
# olsaydi telefonlar "yeni surum var" deyip henuz var olmayan bir dosyayi
# indirmeye calisirdi.
YUKLENECEK=()
for A in "${ABILER[@]}"; do
  cp "build/app/outputs/flutter-apk/app-$A-release.apk" "/tmp/hatirlaf-$A.apk"
  YUKLENECEK+=("/tmp/hatirlaf-$A.apk")
done

if command -v gh >/dev/null; then
  echo "GitHub surumu olusturuluyor…"
  git add pubspec.yaml guncelleme.json
  git commit -m "surum $SURUM+$YENI_KOD"
  git tag "v$SURUM"
  git push origin HEAD --tags

  gh release create "v$SURUM" "${YUKLENECEK[@]}" \
    --title "hatırlaf $SURUM" \
    --notes "${NOTLAR:-Küçük iyileştirmeler.}"

  echo
  echo "yayinlandi. Telefonlar bir sonraki acilislarinda alacak."
else
  echo
  echo "gh (GitHub CLI) kurulu degil. Kalan iki adim elle:"
  echo
  echo "  1. https://github.com/oktayelb/hatirlaf/releases/new adresinde"
  echo "     v$SURUM etiketiyle bir surum olusturun ve su dosyalari"
  echo "     ADLARINI DEGISTIRMEDEN ekleyin:"
  for A in "${ABILER[@]}"; do echo "       /tmp/hatirlaf-$A.apk"; done
  echo
  echo "  2. Dosya yuklendikten SONRA:"
  echo "       git add pubspec.yaml guncelleme.json"
  echo "       git commit -m 'surum $SURUM+$YENI_KOD'"
  echo "       git push"
  echo
  echo "Sirayi bozmayin: once APK, sonra guncelleme.json."
fi
