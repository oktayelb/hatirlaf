#!/usr/bin/env bash
# Bellek tavani konmus flutter. "flutter" yerine bunu cagirin:
#
#   tool/flutter.sh run -d emulator-5554
#   tool/flutter.sh build apk --debug
#
# Derleyici kacarsa makineyi degil kendini oldurur. Gecen sefer web
# derlemesi butun RAM'i yedi ve sistem kitlendi; bu makinede systemd-oomd
# kapali oldugu icin kimse araya girmedi.
set -euo pipefail

# MemoryHigh'a gelince cekirdek yavaslatip sayfa geri alir (yumusak),
# MemoryMax'ta surec oldurulur (sert). Ikisinin arasi manevra alani.
HIGH="${FLUTTER_MEM_HIGH:-6G}"
MAX="${FLUTTER_MEM_MAX:-8G}"

# dart2js ve frontend_server Dart VM uzerinde kosuyor. VM'e ayri bir tavan
# koyuyoruz ki cgroup limitine carpip SIGKILL yemek yerine anlasilir bir
# "out of memory" hatasi versin.
export DART_VM_OPTIONS="--old_gen_heap_size=4096 ${DART_VM_OPTIONS:-}"

if ! command -v flutter >/dev/null; then
  echo "hata: flutter PATH'te yok" >&2
  exit 1
fi

# Onceden acik kalmis bir Gradle daemon'u bu scope'un disinda kalir, yani
# tavan ona islemez. Onun sinirlari android/gradle.properties icinde.
if systemd-run --user --scope -q -p MemoryMax=64M -- true >/dev/null 2>&1; then
  exec systemd-run --user --scope -q \
    --unit="flutter-$$" \
    -p MemoryHigh="$HIGH" \
    -p MemoryMax="$MAX" \
    -- flutter "$@"
fi

echo "uyari: cgroup bellek siniri kurulamadi, flutter sinirsiz calisiyor" >&2
exec flutter "$@"
