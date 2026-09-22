#!/usr/bin/env bash
# Dart sifreler, Python cozer, sonuc bayt bayt karsilastirilir.
#
# Yedek bicimi her degistiginde calistirin. Iki taraf birbirini
# anlamazsa bu ancak veriye ihtiyac duyuldugu gun fark edilir -- yani
# en kotu gun.
set -euo pipefail

cd "$(dirname "$0")/.."

DIZIN="$(mktemp -d)"
temizle() { rm -rf "$DIZIN"; }
trap temizle EXIT

echo "1/3 Dart sifreliyor…"
dart run tool/yedek_interop.dart "$DIZIN" | sed 's/^/  /'

echo
echo "2/3 Python cozuyor…"
python3 tool/yedek_coz.py "$DIZIN/alici.gizli" --klasor "$DIZIN" | sed 's/^/  /'

echo
echo "3/3 karsilastiriliyor…"
HATA=0
for KAYNAK in "$DIZIN"/*.kaynak; do
  AD="$(basename "$KAYNAK" .kaynak)"
  COZULEN="$DIZIN/$AD.bin"
  if [[ ! -f "$COZULEN" ]]; then
    echo "  EKSIK $AD: cozulmus dosya yok" >&2
    HATA=1
    continue
  fi
  if cmp -s "$KAYNAK" "$COZULEN"; then
    echo "  $AD ✓ $(stat -c%s "$KAYNAK") bayt"
  else
    echo "  FARKLI $AD" >&2
    HATA=1
  fi
done

if [[ "$HATA" != "0" ]]; then
  echo >&2
  echo "Dart ve Python ayni bicimi KONUSMUYOR. Yedekler acilamaz." >&2
  exit 1
fi

echo
echo "Dart ve Python ayni bicimi konusuyor."
