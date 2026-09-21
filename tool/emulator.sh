#!/usr/bin/env bash
# Test emulatorunu kurar (yoksa) ve baslatir.
#
#   tool/emulator.sh          arka planda baslat, hazir olunca don
#   tool/emulator.sh --sil    AVD'yi sifirla (bastan kur)
#
# Acildiktan sonra:  tool/flutter.sh run
set -euo pipefail

SDK="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
AVD="${AVD_NAME:-hatirla}"
IMAGE="system-images;android-36;google_apis;x86_64"
export JAVA_HOME="${JAVA_HOME:-$HOME/jdk/jdk-21.0.12.1+1}"
export PATH="$JAVA_HOME/bin:$SDK/platform-tools:$SDK/emulator:$PATH"

if [[ "${1:-}" == "--sil" ]]; then
  "$SDK/cmdline-tools/latest/bin/avdmanager" delete avd -n "$AVD" 2>/dev/null || true
  shift
fi

if [[ ! -d /dev/kvm && ! -c /dev/kvm ]]; then
  echo "hata: /dev/kvm yok. BIOS'ta sanallastirma kapali olabilir." >&2
  exit 1
fi

if ! "$SDK/cmdline-tools/latest/bin/avdmanager" list avd -c 2>/dev/null | grep -qx "$AVD"; then
  echo "AVD kuruluyor: $AVD"
  echo no | "$SDK/cmdline-tools/latest/bin/avdmanager" create avd \
    -n "$AVD" -k "$IMAGE" -d pixel_6 --force

  CFG="$HOME/.android/avd/$AVD.avd/config.ini"
  # Varsayilanlar fazla comert; derleme de ayni RAM'de donecek.
  # Mikrofon acik olmali.
  sed -i -e '/^hw\.ramSize=/d' -e '/^vm\.heapSize=/d' \
         -e '/^hw\.audioInput=/d' -e '/^hw\.keyboard=/d' \
         -e '/^disk\.dataPartition\.size=/d' -e '/^hw\.lcd\.density=/d' "$CFG"
  cat >> "$CFG" <<'INI'
hw.ramSize=2048
vm.heapSize=512
hw.audioInput=yes
hw.keyboard=yes
disk.dataPartition.size=6G
INI
fi

if "$SDK/platform-tools/adb" devices | grep -q "^emulator-.*device$"; then
  echo "emulator zaten calisiyor"
  "$SDK/platform-tools/adb" devices
  exit 0
fi

# Wayland'de takilirsa: EMU_GPU=swiftshader_indirect (yavas ama calisir).
GPU="${EMU_GPU:-host}"

echo "emulator baslatiliyor ($AVD, gpu=$GPU)..."
nohup "$SDK/emulator/emulator" -avd "$AVD" \
  -gpu "$GPU" -memory 2048 -no-boot-anim \
  >/tmp/emulator-$AVD.log 2>&1 &

"$SDK/platform-tools/adb" wait-for-device
# wait-for-device sadece soketi bekler.
until [[ "$("$SDK/platform-tools/adb" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; do
  sleep 2
done

echo "hazir."
"$SDK/platform-tools/adb" devices
