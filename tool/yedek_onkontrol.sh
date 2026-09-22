#!/usr/bin/env bash
# Yayindan once her seyi denetler. Burada gecmeyen bir sey telefonda
# sessiz bir arizaya donusur: kimse hata gormez, hicbir kayit yukselmez.
#
#   tool/yedek_onkontrol.sh          hizli (uctan uca denemesiz)
#   tool/yedek_onkontrol.sh --tam    3 MB'lik gercek gidis-donus de yapar
set -euo pipefail

cd "$(dirname "$0")/.."

TAM="false"
[[ "${1:-}" == "--tam" ]] && TAM="true"

echo "=== 1/4 dosyalar ==="
EKSIK=0
for D in .env yedek.json android/hatirlaf.jks; do
  if [[ -f "$D" ]]; then echo "  ✓ $D"; else echo "  ✗ $D yok" >&2; EKSIK=1; fi
done
GIZLI="$(sed -nE 's/^[[:space:]]*YEDEK_GIZLI_ANAHTAR[[:space:]]*=[[:space:]]*//p' .env \
  | head -1 | tr -d "\"'")"
GIZLI="${GIZLI/#\~/$HOME}"
if [[ -f "$GIZLI" && "$(stat -c%s "$GIZLI")" == "32" ]]; then
  echo "  ✓ $GIZLI (32 bayt)"
else
  echo "  ✗ gizli anahtar yok/bozuk: $GIZLI" >&2
  EKSIK=1
fi
(( EKSIK )) && { echo; echo "Eksikler var, durdu." >&2; exit 1; }

echo
echo "=== 2/4 sirlar depoya sizmis mi ==="
SIZAN="$(git ls-files | grep -E '^\.env$|^yedek\.json$|\.gizli$|\.jks$|\.m4a$|\.wav$' || true)"
if [[ -n "$SIZAN" ]]; then
  echo "  ✗ DEPODA IZLENIYOR:" >&2
  echo "$SIZAN" | sed 's/^/      /' >&2
  exit 1
fi
echo "  ✓ hicbir sir izlenmiyor"

echo
echo "=== 3/4 ayar adlari uygulamaya ulasiyor mu ==="
# Adlar tutmazsa yedekleme KAPALI derlenir ve kimse fark etmez.
DEFS="$(python3 -c "
import json
d = json.load(open('yedek.json'))
print(' '.join(f'--define={k}={v}' for k, v in d.items()))
")"
# shellcheck disable=SC2086
dart run $DEFS tool/yedek_ayar_dogrula.dart 2>&1 | grep -vE "build hooks|^$"

echo
echo "=== 4/4 B2 ==="
tool/b2_saglik.py

if [[ "$TAM" == "true" ]]; then
  echo
  echo "=== ek: uctan uca gidis-donus ==="
  tool/b2_deneme.sh
fi

echo
echo "Her sey yerinde. tool/yayinla.sh feature \"...\" calistirilabilir."
