#!/usr/bin/env bash
# Yeni bir surum yayinlar.
#
#   tool/yayinla.sh bug     "Kayıt düğmesi bazen çalışmıyordu."
#   tool/yayinla.sh feature "Fotoğraf eklenebiliyor."
#   tool/yayinla.sh version "Yeni hatıra defteri."
#   tool/yayinla.sh feature "Deneme." --deneme     (hicbir sey yayinlanmaz)
#
# Surum numarasi elle yazilmaz; degisikligin turunu soylersiniz, numarayi
# script pubspec.yaml'dan hesaplar:
#
#   bug      1.4.2 -> 1.4.3    en sagdaki artar
#   feature  1.4.2 -> 1.5.0    ortadaki artar, sagdaki sifirlanir
#   version  1.4.2 -> 2.0.0    soldaki artar, digerleri sifirlanir
#
# Gerekirse acik numara da verilebilir: tool/yayinla.sh 3.0.0 "..."
#
# Yaptigi sirayla:
#   1. On kontroller (imza anahtari, temiz dizin, arac ve yetki).
#   2. pubspec.yaml'daki surumu yukseltir (surum adi + surum kodu).
#   3. Mimariye ozel, imzali APK'lari derler.
#   4. Her APK'nin gercek versionCode'unu, sha256'sini ve boyutunu okuyup
#      guncelleme.json'u yazar.
#   5. Surum commit'ini ve etiketini iter, GitHub surumunu olusturup
#      APK'lari yukler.
#   6. Yuklenen dosyalarin gercekten indirilebildigini dogrular.
#   7. **Ancak bundan sonra** guncelleme.json'u iter.
#
# 5-6-7 sirasi pazarlik konusu degil: guncelleme.json "yeni surum var"
# demektir. Once itilseydi telefonlar henuz yuklenmemis bir dosyayi
# indirmeye calisir, basarisiz olur ve 20 saat boyunca bir daha
# denemezdi. Bkz. docs/guncelleme.md, 2.2.
set -euo pipefail

cd "$(dirname "$0")/.."

ARTIS=""
NOTLAR=""
DENEME="false"

for ARG in "$@"; do
  case "$ARG" in
    --deneme)  DENEME="true" ;;
    -*) echo "hata: bilinmeyen secenek: $ARG" >&2; exit 1 ;;
    *)
      if [[ -z "$ARTIS" ]]; then ARTIS="$ARG"
      elif [[ -z "$NOTLAR" ]]; then NOTLAR="$ARG"
      else echo "hata: fazladan argüman: $ARG" >&2; exit 1
      fi
      ;;
  esac
done

# --- surum numarasini hesapla -------------------------------------------
#
# Numarayi elle yazmak, yazilan sayinin pubspec'tekiyle ilgisiz olmasi
# demekti: once "hangi numaradaydik?" diye bakmak, sonra dogru yeri
# artirmak gerekiyordu. Artik degisikligin *turunu* soyluyoruz.
MEVCUT_AD="$(grep -m1 '^version:' pubspec.yaml \
  | sed 's/^version:[[:space:]]*//' | cut -d'+' -f1)"
if [[ ! "$MEVCUT_AD" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "hata: pubspec.yaml icindeki surum okunamadi: '$MEVCUT_AD'" >&2
  exit 1
fi
IFS='.' read -r BUYUK ORTA KUCUK <<< "$MEVCUT_AD"

kullanim() {
  echo "kullanim: tool/yayinla.sh <bug|feature|version> \"<neler degisti>\" [--deneme]" >&2
  echo >&2
  echo "Su an $MEVCUT_AD. Buradan:" >&2
  echo "  bug      -> $BUYUK.$ORTA.$((KUCUK + 1))  (hata duzeltmesi)" >&2
  echo "  feature  -> $BUYUK.$((ORTA + 1)).0  (yeni ozellik)" >&2
  echo "  version  -> $((BUYUK + 1)).0.0  (buyuk degisiklik)" >&2
  echo >&2
  echo "Acik numara da verilebilir: tool/yayinla.sh 3.0.0 \"...\"" >&2
  exit 1
}

[[ -z "$ARTIS" ]] && kullanim

case "$ARTIS" in
  bug)
    KUCUK=$((KUCUK + 1)) ;;
  feature)
    ORTA=$((ORTA + 1)); KUCUK=0 ;;
  version)
    BUYUK=$((BUYUK + 1)); ORTA=0; KUCUK=0 ;;
  *)
    # Acik numara: kacis kapisi. Numarayi atlamak ya da geri almak
    # gerekirse diye duruyor.
    if [[ ! "$ARTIS" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "hata: '$ARTIS' anlasilmadi." >&2
      echo >&2
      kullanim
    fi
    IFS='.' read -r BUYUK ORTA KUCUK <<< "$ARTIS" ;;
esac

SURUM="$BUYUK.$ORTA.$KUCUK"

DEPO="oktayelb/hatirlaf"
# APK adresleri ETIKETE sabitleniyor, "latest"e degil.
#
# `releases/latest/download/...` hareketli bir hedef: yeni bir surum
# olusturuldugu anda eski guncelleme.json'un isaret ettigi adres yeni
# dosyaya kayardi. Telefon o dosyayi indirir, sha256 tutmaz, atar ve
# sonsuza kadar yeniden denerdi. Etiketli adres hic degismez.
KOK="https://github.com/$DEPO/releases/download/v$SURUM"
ABILER=(arm64-v8a armeabi-v7a x86_64)

# --- 1. on kontroller ----------------------------------------------------
#
# Hepsi burada, derlemeye baslamadan once: 10 dakikalik bir derlemenin
# sonunda "gh yok" demek kotu bir saka olurdu.

if [[ -n "$(git status --porcelain)" ]]; then
  echo "hata: calisma dizininde kaydedilmemis degisiklik var." >&2
  echo "Once commit edin; yayinlanan surum depoda izlenebilir olmali." >&2
  exit 1
fi

# Imza anahtari en kritik kontrol: debug anahtariyla imzalanmis bir APK
# telefonlara guncelleme olarak KURULAMAZ ve bu ancak kullanicinin
# telefonunda, sessizce fark edilir.
if [[ ! -f android/key.properties || ! -f android/hatirlaf.jks ]]; then
  echo "hata: yayin imza anahtari yok (android/key.properties + hatirlaf.jks)." >&2
  echo "Yedekten geri koyun. Bu anahtar olmadan guncelleme yayinlanamaz." >&2
  exit 1
fi

# Surum kodunu APK'dan okumak icin aapt2 sart. Mimariye ozel derlemede
# Flutter surum kodunu kaydiriyor (armeabi-v7a +1000, arm64-v8a +2000,
# x86_64 +4000); kaydirmayi varsaymak yerine APK'ya soruyoruz.
AAPT="$(ls "${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"/build-tools/*/aapt2 2>/dev/null | sort -V | tail -1)"
if [[ -z "$AAPT" ]]; then
  echo "hata: aapt2 bulunamadi (Android SDK build-tools)." >&2
  exit 1
fi

if [[ "$DENEME" == "false" ]]; then
  if ! command -v gh >/dev/null; then
    echo "hata: gh (GitHub CLI) kurulu degil." >&2
    echo "  sudo dnf install gh && gh auth login" >&2
    exit 1
  fi
  if ! gh auth status >/dev/null 2>&1; then
    echo "hata: gh oturumu yok. 'gh auth login' calistirin." >&2
    exit 1
  fi
  if gh release view "v$SURUM" --repo "$DEPO" >/dev/null 2>&1; then
    echo "hata: v$SURUM surumu zaten var. Baska bir surum numarasi secin." >&2
    exit 1
  fi
  git fetch --quiet origin main
  if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "hata: yerel main ile origin/main ayni degil. Once pull/push edin." >&2
    exit 1
  fi
fi

# --- 2. surum kodunu yukselt --------------------------------------------
ESKI_SATIR="$(grep -m1 '^version:' pubspec.yaml)"
ESKI_KOD="${ESKI_SATIR##*+}"
YENI_KOD=$((ESKI_KOD + 1))

# Buradan sonra bir sey patlarsa pubspec.yaml'i geri al: yarim kalmis bir
# surum yukseltmesi bir sonraki denemede numarayi sessizce kaydirirdi.
GERI_AL="evet"
geri_al() {
  if [[ "$GERI_AL" == "evet" ]]; then
    git checkout -- pubspec.yaml guncelleme.json 2>/dev/null || true
    echo >&2
    echo "yayinlama yarida kaldi; pubspec.yaml geri alindi." >&2
  fi
}
trap geri_al EXIT INT TERM

echo "surum   : $MEVCUT_AD -> $SURUM+$YENI_KOD  ($ARTIS)"
sed -i "s|^version:.*|version: $SURUM+$YENI_KOD|" pubspec.yaml

# --- 3. derle ------------------------------------------------------------
#
# Mimariye ozel APK'lar: tek parca (universal) APK 60 MB, arm64'e ozel
# olan 22 MB. Guncelleme her surumde yeniden indirilecegi icin aradaki
# 38 MB her seferinde tekrar odenirdi.
echo "derleniyor…"
tool/flutter.sh build apk --release --split-per-abi

for A in "${ABILER[@]}"; do
  [[ -f "build/app/outputs/flutter-apk/app-$A-release.apk" ]] || {
    echo "hata: app-$A-release.apk olusmadi" >&2; exit 1; }
done

# --- 4. guncelleme.json --------------------------------------------------
#
# Her mimari icin ayri adres + surum kodu + ozet + boyut. Telefon kendi
# mimarisini (Build.SUPPORTED_ABIS) bilip dogru satiri seciyor.
python3 - "$AAPT" "$SURUM" "$NOTLAR" "$KOK" "${ABILER[@]}" <<'PY'
import hashlib, io, json, os, re, subprocess, sys

aapt, surum, notlar, kok = sys.argv[1:5]
abiler = sys.argv[5:]

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
        "apkUrl": f"{kok}/hatirlaf-{a}.apk",
        "sha256": ozet.hexdigest(),
        "boyut": boyut,
    }
    print(f"  {a:<14} kod={kod:<6} {boyut / 1048576:5.1f} MB  {ozet.hexdigest()[:16]}…")

veri = {
    "surumAdi": surum,
    "notlar": notlar,
    "paketler": paketler,
}
io.open("guncelleme.json", "w", encoding="utf-8").write(
    json.dumps(veri, ensure_ascii=False, indent=2) + "\n"
)
PY

YUKLENECEK=()
for A in "${ABILER[@]}"; do
  cp "build/app/outputs/flutter-apk/app-$A-release.apk" "/tmp/hatirlaf-$A.apk"
  YUKLENECEK+=("/tmp/hatirlaf-$A.apk")
done

if [[ "$DENEME" == "true" ]]; then
  echo
  echo "--- DENEME: hicbir sey yayinlanmadi ---"
  echo "Uretilecek guncelleme.json:"
  sed 's/^/  /' guncelleme.json
  echo
  echo "Yuklenecek dosyalar:"
  printf '  %s\n' "${YUKLENECEK[@]}"
  git checkout -- pubspec.yaml guncelleme.json
  GERI_AL="hayir"
  trap - EXIT INT TERM
  echo
  echo "pubspec.yaml ve guncelleme.json geri alindi."
  exit 0
fi

# --- 5. surum commit'i + GitHub surumu -----------------------------------
#
# guncelleme.json bilerek DISARIDA birakiliyor; o en sona kaliyor.
echo
echo "surum commit'i itiliyor…"
git add pubspec.yaml
git commit -q -m "surum $SURUM+$YENI_KOD"
GERI_AL="hayir"
git tag "v$SURUM"
git push --quiet origin main
git push --quiet origin "v$SURUM"

echo "GitHub surumu olusturuluyor ve APK'lar yukleniyor…"
gh release create "v$SURUM" "${YUKLENECEK[@]}" \
  --repo "$DEPO" \
  --title "hatırlaf $SURUM" \
  --notes "${NOTLAR:-Küçük iyileştirmeler.}"

# --- 6. yuklenenler gercekten inebiliyor mu? -----------------------------
#
# Guvenlik kemeri: guncelleme.json'u itmeden once dosyalarin telefonun
# kullanacagi ADRESTEN indirilebildigini dogruluyoruz. Burada durursak
# telefonlar eski surumde kalir, yani kimse zarar gormez.
echo "yuklenen dosyalar dogrulaniyor…"
for A in "${ABILER[@]}"; do
  BEKLENEN="$(stat -c%s "/tmp/hatirlaf-$A.apk")"
  GORULEN=""
  for _ in 1 2 3 4 5; do
    GORULEN="$(curl -sIL "$KOK/hatirlaf-$A.apk" \
      | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')"
    [[ "$GORULEN" == "$BEKLENEN" ]] && break
    sleep 3
  done
  if [[ "$GORULEN" != "$BEKLENEN" ]]; then
    echo >&2
    echo "hata: $A dosyasi adresinden dogrulanamadi." >&2
    echo "  beklenen $BEKLENEN bayt, gorulen '${GORULEN:-yok}'" >&2
    echo >&2
    echo "guncelleme.json ITILMEDI - telefonlar eski surumde kaliyor," >&2
    echo "yani kimse zarar gormedi. GitHub surumunu kontrol edip eksik" >&2
    echo "dosyayi yukleyin, sonra:" >&2
    echo "  git add guncelleme.json && git commit -m 'guncelleme $SURUM' && git push" >&2
    exit 1
  fi
  echo "  $A ✓"
done

# --- 7. guncelleme.json: telefonlara "yeni surum var" diyen adim ---------
echo "guncelleme.json itiliyor…"
git add guncelleme.json
git commit -q -m "guncelleme $SURUM"
git push --quiet origin main

trap - EXIT INT TERM
echo
echo "yayinlandi: hatırlaf $SURUM ($YENI_KOD)"
echo "Telefonlar internete ciktiklari ilk acilista indirecek,"
echo "bir sonraki acilista soracak."
