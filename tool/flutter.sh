#!/usr/bin/env bash
# Bellek VE CPU tavani konmus flutter; "flutter" yerine bunu cagirin.
# Bu makinede systemd-oomd kapali, kacan bir derleyiciyi kimse durdurmuyor
# ve 12 cekirdegin hepsini yiyen bir derleme makineyi kullanilmaz yapiyor.
#
# Amac hiz degil: derleme yavaslasa da makine kullanilabilir kalmali.
#
#   FLUTTER_CPU=400   yuzde cinsinden tavan (400 = 4 cekirdek)
#   FLUTTER_MEM_MAX=8G
set -euo pipefail

HIGH="${FLUTTER_MEM_HIGH:-6G}"
MAX="${FLUTTER_MEM_MAX:-8G}"

# Varsayilan: cekirdeklerin ucte biri, en az 2. 12 cekirdekte 4.
CEKIRDEK="$(nproc 2>/dev/null || echo 4)"
VARSAYILAN_CPU=$(( CEKIRDEK / 3 * 100 ))
(( VARSAYILAN_CPU < 200 )) && VARSAYILAN_CPU=200
CPU="${FLUTTER_CPU:-$VARSAYILAN_CPU}"

# VM'e ayri tavan: cgroup limitine carpip SIGKILL yemek yerine anlasilir
# bir "out of memory" hatasi versin.
export DART_VM_OPTIONS="--old_gen_heap_size=4096 ${DART_VM_OPTIONS:-}"

if ! command -v flutter >/dev/null; then
  echo "hata: flutter PATH'te yok" >&2
  exit 1
fi

# Derleme ise, once eski Gradle daemon'ini kapat. Daemon kendini bizim
# kapsamimizin disina dogurabiliyor; yenisini bu kapsam icinde
# baslatirsak cgroup sinirlari ona da isler. (Ikinci kusak savunma
# android/gradle.properties icindeki ActiveProcessorCount.)
for ARG in "$@"; do
  if [[ "$ARG" == "build" || "$ARG" == "apk" ]]; then
    if pgrep -f "org.gradle.launcher.daemon.bootstrap.GradleDaemon" >/dev/null 2>&1; then
      echo "not: eski Gradle daemon kapatiliyor (sinirlar icinde yenisi acilsin)" >&2
      pkill -f "org.gradle.launcher.daemon.bootstrap.GradleDaemon" || true
    fi
    break
  fi
done

if systemd-run --user --scope -q -p MemoryMax=64M -- true >/dev/null 2>&1; then
  echo "sinirlar: CPU %${CPU} (${CEKIRDEK} cekirdekten), bellek $MAX" >&2
  # CPUWeight dusuk: cekisme oldugunda kullanicinin uygulamalari kazansin.
  # nice da ayni isi zamanlayici tarafinda yapiyor.
  exec systemd-run --user --scope -q \
    --unit="flutter-$$" \
    -p MemoryHigh="$HIGH" \
    -p MemoryMax="$MAX" \
    -p CPUQuota="${CPU}%" \
    -p CPUWeight=20 \
    -- nice -n 10 flutter "$@"
fi

echo "uyari: cgroup sinirlari kurulamadi; yalnizca nice ile calisiyor" >&2
exec nice -n 10 flutter "$@"
