#!/usr/bin/env bash
# Gercek B2'ye karsi uctan uca deneme. Telefona dokunmadan, uretimdeki
# kodun ta kendisiyle:
#
#   Dart: sifrele -> yukle      (uygulamanin kisitli anahtariyla)
#   Python: indir -> coz -> karsilastir  (master anahtarla)
#
#   tool/b2_deneme.sh [bayt]     (varsayilan 3 MiB)
#
# Yuklenen deneme nesnesi sonunda silinir.
set -euo pipefail

cd "$(dirname "$0")/.."

BOYUT="${1:-3145728}"

if [[ ! -f yedek.json ]]; then
  echo "hata: yedek.json yok. Once tool/b2_kur.py calistirin." >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "hata: .env yok." >&2
  exit 1
fi

DIZIN="$(mktemp -d)"
temizle() { rm -rf "$DIZIN"; }
trap temizle EXIT

echo "1/2 Dart: sifreleyip B2'ye yukluyor…"
dart run tool/b2_deneme.dart "$DIZIN" "$BOYUT"

echo
echo "2/2 Python: indirip cozuyor…"
python3 tool/b2_indir_coz.py "$DIZIN"

echo
echo "Uctan uca calisiyor: telefonun yazdigini PC cozebiliyor."
