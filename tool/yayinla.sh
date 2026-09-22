#!/usr/bin/env bash
# Yeni bir surum yayinlar. Ayrintilar: docs/guncelleme.md
#
#   tool/yayinla.sh bug|feature|version "Not"  [--deneme]
#   tool/yayinla.sh 3.0.0 "Not"                (acik numara)
#
# Surum numarasini pubspec.yaml'dan hesaplar. 5-6-7. adimlarin sirasi
# pazarlik konusu degil: guncelleme.json "yeni surum var" demektir, once
# itilseydi telefonlar henuz yuklenmemis bir dosyayi indirmeye calisirdi.
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
    # Acik numara: kacis kapisi.
    if [[ ! "$ARTIS" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "hata: '$ARTIS' anlasilmadi." >&2
      echo >&2
      kullanim
    fi
    IFS='.' read -r BUYUK ORTA KUCUK <<< "$ARTIS" ;;
esac

SURUM="$BUYUK.$ORTA.$KUCUK"

DEPO="oktayelb/hatirlaf"
# APK adresleri ETIKETE sabitleniyor: "latest" hareketli bir hedef, yeni
# surumde eski manifest'in adresi kayar ve sha256 tutmaz.
KOK="https://github.com/$DEPO/releases/download/v$SURUM"
ABILER=(arm64-v8a armeabi-v7a x86_64)

# --- 1. on kontroller (derlemeye baslamadan once) ------------------------

if [[ -n "$(git status --porcelain)" ]]; then
  echo "hata: calisma dizininde kaydedilmemis degisiklik var." >&2
  echo "Once commit edin; yayinlanan surum depoda izlenebilir olmali." >&2
  exit 1
fi

# En kritik kontrol: debug anahtariyla imzalanmis APK telefonlara
# guncelleme olarak kurulamaz ve bu ancak kullanicinin telefonunda anlasilir.
IMZA_EKSIK=()
if [[ -f .env ]]; then
  # Degerleri kabuga almadan yalnizca varliklarini kontrol et.
  for A in ANDROID_KEYSTORE ANDROID_KEYSTORE_PASSWORD ANDROID_KEY_ALIAS ANDROID_KEY_PASSWORD; do
    grep -qE "^[[:space:]]*$A[[:space:]]*=[[:space:]]*[^[:space:]]" .env || IMZA_EKSIK+=("$A")
  done
  KS="$(sed -nE 's/^[[:space:]]*ANDROID_KEYSTORE[[:space:]]*=[[:space:]]*//p' .env | head -1 | tr -d '"'"'"'"')"
  # Acik `if`: kisa devre yapan `[[ ]] && ...` kaliba 1 donduruyor ve
  # blogun son satiri olursa `set -e` scripti sessizce bitirir.
  if [[ -n "$KS" && ! -f "$KS" ]]; then
    IMZA_EKSIK+=("$KS (dosya yok)")
  fi
else
  IMZA_EKSIK+=(".env")
fi

if (( ${#IMZA_EKSIK[@]} )); then
  echo "hata: yayin imza ayarlari eksik: ${IMZA_EKSIK[*]}" >&2
  echo "Yedekten geri koyun. Bu anahtar olmadan guncelleme yayinlanamaz;" >&2
  echo "debug anahtariyla imzali APK telefonlara KURULAMAZ." >&2
  exit 1
fi

# Flutter surum kodunu mimariye gore kaydiriyor; varsaymak yerine
# aapt2 ile APK'ya soruyoruz.
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
# yukseltme sonraki denemede numarayi kaydirirdi.
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
# Mimariye ozel APK'lar: universal 60 MB, arm64'e ozel olan 22 MB.
#
# yedek.json gitignore'da: B2 yazma anahtarini tasiyor, depo herkese acik.
# Yoksa derleme gecerlidir ama yedekleme KAPALI cikar; sessiz gecmeyelim.
YEDEK_TANIM=()
if [[ -f yedek.json ]]; then
  YEDEK_TANIM=(--dart-define-from-file=yedek.json)
  echo "yedek    : yedek.json bulundu, yedekleme acik"
else
  echo "UYARI: yedek.json yok -> bu surumde aile yedegi KAPALI olacak." >&2
  echo "       Acik olmasi gerekiyorsa yedek.json.ornek'i kopyalayin." >&2
fi

echo "derleniyor…"
tool/flutter.sh build apk --release --split-per-abi "${YEDEK_TANIM[@]+"${YEDEK_TANIM[@]}"}"

for A in "${ABILER[@]}"; do
  [[ -f "build/app/outputs/flutter-apk/app-$A-release.apk" ]] || {
    echo "hata: app-$A-release.apk olusmadi" >&2; exit 1; }
done

# --- 4. guncelleme.json --------------------------------------------------
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

if not paketler:
    sys.exit("hata: hicbir paket uretilmedi")

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
# guncelleme.json bilerek disarida; o en sona kaliyor.
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
# Burada durursak telefonlar eski surumde kalir, kimse zarar gormez.
echo "yuklenen dosyalar dogrulaniyor…"
for A in "${ABILER[@]}"; do
  BEKLENEN="$(stat -c%s "/tmp/hatirlaf-$A.apk")"
  GORULEN=""
  for _ in 1 2 3 4 5; do
    GORULEN="$(curl -sIL "$KOK/hatirlaf-$A.apk" \
      | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' \
      || true)"
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
