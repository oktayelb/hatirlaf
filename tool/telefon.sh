#!/usr/bin/env bash
# APK'yi USB kablosuyla bagli gercek telefona kurar.
#
#   tool/telefon.sh            release kur (varsayilan)
#   tool/telefon.sh --debug    debug kur (hot reload icin degil, sadece kurulum)
#
# Telefonda once: Ayarlar > Telefon hakkinda > Yapi numarasina 7 kez dokun,
# sonra Ayarlar > Gelistirici secenekleri > USB hata ayiklama = acik.
set -euo pipefail

SDK="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
ADB="$SDK/platform-tools/adb"
KIP="release"
[[ "${1:-}" == "--debug" ]] && KIP="debug"

cd "$(dirname "$0")/.."

mapfile -t CIHAZ < <("$ADB" devices | awk '/\tdevice$/{print $1}' | grep -v '^emulator-' || true)

if [[ ${#CIHAZ[@]} -eq 0 ]]; then
  echo "Telefon gorunmuyor. Sirasiyla:"
  echo "  1. USB kablosunu tak (sarj kablosu degil, veri kablosu olmali)."
  echo "  2. Telefonda 'USB hata ayiklamaya izin ver' penceresine Tamam de."
  echo "  3. USB modunu 'Dosya aktarimi (MTP)' yap; 'Sadece sarj' calismaz."
  echo
  "$ADB" devices -l
  exit 1
fi

TEL="${CIHAZ[0]}"
ABI="$("$ADB" -s "$TEL" shell getprop ro.product.cpu.abi | tr -d '\r')"
APK="build/app/outputs/flutter-apk/app-$ABI-$KIP.apk"

# --split-per-abi kullanilmadiysa tek bir birlesik APK vardir.
[[ -f "$APK" ]] || APK="build/app/outputs/flutter-apk/app-$KIP.apk"

if [[ ! -f "$APK" ]]; then
  echo "hata: $APK yok. Once derleyin:" >&2
  echo "  tool/flutter.sh build apk --$KIP --split-per-abi" >&2
  exit 1
fi

echo "telefon : $TEL ($ABI)"
echo "paket   : $APK ($(du -h "$APK" | cut -f1))"

# -r: uzerine yaz. Imza degistiyse once silmek gerekir.
if ! "$ADB" -s "$TEL" install -r "$APK"; then
  echo
  echo "Kurulum basarisiz. Telefonda eski bir surum baska bir anahtarla"
  echo "imzalanmis olabilir. Silip tekrar deneyin:"
  echo "  $ADB -s $TEL uninstall com.hatirla.hatirla"
  exit 1
fi

echo "kuruldu."
