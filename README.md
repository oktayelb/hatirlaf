# hatırlaf

Büyüklerin hayat hikâyelerini kendi sesleriyle saklamak için bir sesli
hatıra defteri. Kayıt alınır, `whisper.cpp` ile telefonun içinde,
internetsiz olarak yazıya dökülür. Hatıralar telefondan çıkmaz.

Arayüz yaşlı kullanıcı için: büyük yazı, her ikonun yanında etiket, en
fazla iki seviye gezinme, teknik olmayan hata mesajları.

## Kurulum (telefona)

APK'yi telefona aktarıp dokunun; "bilinmeyen kaynak" uyarısına izin
verin. Karşılama ekranı mikrofon iznini ve yazıya çevirme paketini
(~142 MB, Wi-Fi'de) halleder. Paket indirilmezse kayıt yine alınır,
sadece yazıya çevrilmez; sonradan Ayarlar'dan indirilebilir.

## Geliştirme

```bash
tool/flutter.sh build apk --debug              # sideload
tool/flutter.sh build apk --release --split-per-abi
tool/telefon.sh                                # USB'deki telefona kur
tool/emulator.sh                               # AVD kur/başlat (--sil: sıfırla)
```

`flutter` yerine `tool/flutter.sh` kullanın: bu makinede `systemd-oomd`
kapalı, kaçan bir derleyici sistemi kilitliyor. Script bellek tavanı
koyuyor (`FLUTTER_MEM_HIGH` / `FLUTTER_MEM_MAX`). Gradle'ın kendi
sınırları `android/gradle.properties` içinde.

Ortam: Flutter 3.44.1, JDK 21 (`~/jdk/jdk-21.0.12.1+1`), Android SDK 36
(`~/Android/Sdk`, NDK 29). Sistemdeki JDK 25 AGP ile uyumsuz.

İlk derleme ~10-15 dk sürer: whisper.cpp kaynaktan derleniyor.

Release derlemesi `android/key.properties` + `android/hatirlaf.jks`
ister; bu dosyalar depoda yok, yedekten gelir. Yoksa Gradle debug
anahtarına düşer ve o APK telefonlara güncelleme olarak kurulamaz.

## Yayınlama

```bash
tool/yayinla.sh bug     "Kayıt düğmesi bazen çalışmıyordu."
tool/yayinla.sh feature "Fotoğraf eklenebiliyor."
```

Sürüm numarasını script hesaplar (`bug`/`feature`/`version`).
Ayrıntılar ve tuzaklar: [docs/guncelleme.md](docs/guncelleme.md).

## Yapı

```
lib/
├── main.dart, theme.dart
├── data/prompts.dart      soru kütüphanesi
├── models/memory.dart
├── services/              store, recorder, player, transcriber,
│                          whisper_model_manager, cover_photo,
│                          updater, update_info, network, permissions
├── screens/               welcome, home, record, memory, question,
│                          settings, help, update
├── widgets/
└── utils/format.dart
```

Durum yönetimi için ek paket yok: servisler `ChangeNotifier` tekilleri.

Veriler `<app documents>/` altında: `hatiralar.json` (+ `.yedek`),
`kapak.jpg`, `hatiralar/<uuid>/ses.m4a`. Dosya yolları göreceli tutulur;
Android yedekten geri yükleme sonrası mutlak yol değişebiliyor.
`hatiralar.json` bozulursa store önce yedeği, sonra klasörleri tarayarak
kurtarır.

## Bilinen sınırlar

- Web derlenemez: `whisper_ggml` ve `path_provider` web'i desteklemiyor.
- iOS derlenmedi (Mac gerekiyor).
- Uzun kayıtlar yavaş çevrilir; kuyruk arka planda çalışır ama uygulama
  açık kalmalı.
- Kayıt arka plana alınırsa Android 14+ süreci öldürebilir; wakelock var,
  foreground service yok.
- Proje klasöründeki `ı` karakteri Dart analiz sunucusunu çökertiyor;
  `flutter analyze` ve IDE tamamlama bu klasörde çalışmaz. Derleme
  etkilenmez.
