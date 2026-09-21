#!/usr/bin/env bash
# Bellek tavani konmus flutter; "flutter" yerine bunu cagirin. Bu makinede
# systemd-oomd kapali, kacan bir derleyiciyi kimse durdurmuyor.
set -euo pipefail

# MemoryHigh: cekirdek yavaslatir. MemoryMax: surec oldurulur.
HIGH="${FLUTTER_MEM_HIGH:-6G}"
MAX="${FLUTTER_MEM_MAX:-8G}"

# VM'e ayri tavan: cgroup limitine carpip SIGKILL yemek yerine anlasilir
# bir "out of memory" hatasi versin.
export DART_VM_OPTIONS="--old_gen_heap_size=4096 ${DART_VM_OPTIONS:-}"

if ! command -v flutter >/dev/null; then
  echo "hata: flutter PATH'te yok" >&2
  exit 1
fi

# Gradle daemon'u bu scope'un disinda kalabilir; sinirlari
# android/gradle.properties icinde.
if systemd-run --user --scope -q -p MemoryMax=64M -- true >/dev/null 2>&1; then
  exec systemd-run --user --scope -q \
    --unit="flutter-$$" \
    -p MemoryHigh="$HIGH" \
    -p MemoryMax="$MAX" \
    -- flutter "$@"
fi

echo "uyari: cgroup bellek siniri kurulamadi, flutter sinirsiz calisiyor" >&2
exec flutter "$@"
