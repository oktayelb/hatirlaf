# hatırlaf

Büyüklerimizin hayat hikâyelerini kendi sesleriyle saklamak için yapılmış
bir sesli hatıra defteri.

Yaşlı bir kullanıcı telefonu eline alır, kırmızı düğmeye basar ve anlatır.
Uygulama hem sesi kaydeder hem de **telefonun içinde**, internetsiz olarak
konuşmayı yazıya döker. Çocukları ve torunları yıllar sonra hem sesi
dinleyebilir hem de yazısını okuyabilir.

---

## Neler var

| Özellik | Açıklama |
|---|---|
| Sesli kayıt | Tek dokunuşla başlar, ara verilebilir, kaydedilir |
| Cihaz üstü yazıya çevirme | `whisper.cpp` ile Türkçe, tamamen çevrimdışı |
| Soru kütüphanesi | 10 konuda ~70 hazır hayat hikâyesi sorusu |
| Fotoğraf | Ana sayfada tek kapak fotoğrafı (kameradan veya galeriden) |
| Hatıra defteri | Tüm yazıları tek metin dosyası olarak dışa aktarma |
| Kendi kendine güncelleme | Yeni sürümü sessizce indirir, hazır olunca bir kez sorar |

**Hatıralar telefondan çıkmaz.** Ses kayıtları ve yazıya çevrilmiş
metinler hiçbir yere gönderilmiyor; hesap, giriş, bulut yok. Uygulamanın
internete çıktığı tek iki yer var ve ikisi de yalnızca **indirme**
yönünde: yazıya çevirme paketi ve uygulamanın kendi güncellemesi.

---

## Yaşlı kullanıcı için tasarım kararları

Bunlar keyfi değil; her biri bilinen bir takılma noktasını kapatıyor:

- **20 puntonun altında yazı yok**, dokunulabilir alanlar en az 72 piksel.
- **Her ikonun yanında yazı var.** Simge tek başına bırakılmıyor.
- **Gezinme derinliği en fazla iki.** Sekme, alt menü, hamburger menü yok.
- **Ana buton her zaman ekranda**, listeyle birlikte kaymıyor.
- **Boş ekran yok.** Ne anlatacağını bilemeyene uygulama soruyu kendisi sorar.
- **İzin pencereleri habersiz açılmaz**; önce sade bir cümleyle anlatılır.
  (Habersiz çıkan izin penceresinde refleksle "Reddet"e basılıyor.)
- **Silme işlemlerinde onay butonu ikinci sırada** ve kırmızı.
- **Hata mesajları teknik değil**: "Sunucu 503" değil, "İnternete
  bağlanılamadı. Wi-Fi'nizi kontrol edip tekrar deneyin."
- **Ekran kayıt sırasında kapanmaz** (wakelock), kayıt yarıda kesilmesin.
- **Telefon karanlık moddayken bile** uygulama açık temada kalır; kontrast
  düşmesin.

---

## Kurulum (telefona)

Kuracak kişi genellikle çocuğu/torunu olacak. Sıra:

1. `hatirla.apk` dosyasını telefona aktarın (kablo, WhatsApp, e-posta).
2. Dosyaya dokunun. Android "bilinmeyen kaynak" uyarısı verirse
   *İzin ver* deyin.
3. Uygulamayı açın. Karşılama ekranı 4 adımda her şeyi halleder:
   - Mikrofon izni
   - Yazıya çevirme paketinin indirilmesi (**Wi-Fi'de yapın**, ~142 MB)
   - Hazır
4. Telefonun ayarlarından yazı boyutunu büyütmeye gerek yok; uygulama
   zaten büyük. (Sistemden büyütülmüşse de bozulmaz.)

**Önemli:** Yazıya çevirme paketi indirilmezse uygulama yine çalışır, ses
kayıtları alınır — sadece yazıya çevrilmez. Paket sonradan Ayarlar'dan
indirilebilir; bekleyen kayıtlar o zaman otomatik olarak çevrilir.

---

## Güncelleme nasıl çalışıyor

Uygulama mağazası yok; güncelleme depodaki `guncelleme.json` dosyası ve
GitHub sürüm (release) ekleri üzerinden yürüyor. Yeni sürüm yayınlamak
tek komut:

```bash
tool/yayinla.sh bug     "Kayıt düğmesi bazen çalışmıyordu."
tool/yayinla.sh feature "Fotoğraf eklenebiliyor."
```

Sürüm numarasını siz yazmazsınız; değişikliğin türünü söylersiniz,
numarayı script hesaplar (`bug` → 1.4.3, `feature` → 1.5.0,
`version` → 2.0.0).

Telefon tarafında **denetim ve indirme görünmezdir**: yaşlı kullanıcı ne
ilerleme çubuğu ne de hata uyarısı görür. Yalnızca APK inip özeti
doğrulandıktan sonra, bir sonraki açılışta bir kez sorulur.

> **Yayınlamadan önce [docs/guncelleme.md](docs/guncelleme.md) okuyun.**
> Üç şey yanlış yapılırsa hata sizin makinenizde değil, akrabanızın
> telefonunda sessizce ortaya çıkar: imza anahtarının kaybolması,
> `guncelleme.json`'un APK'lardan önce itilmesi ve `versionCode`'un
> mimariye göre kayması.

---

## Geliştirme

### Ortam

Bu makinede kurulu olanlar:

```
Flutter 3.44.1        /home/oktay/flutter
JDK 21 (Temurin)      ~/jdk/jdk-21.0.12.1+1
Android SDK 36        ~/Android/Sdk
  ├── build-tools 36.0.0
  ├── platform-tools
  ├── ndk 29.0.13113456   (whisper.cpp bununla derleniyor)
  └── cmake 3.22.1
```

Flutter bu yolları kalıcı olarak biliyor:

```bash
flutter config --android-sdk ~/Android/Sdk --jdk-dir ~/jdk/jdk-21.0.12.1+1
```

> Sistemdeki JDK 25, Android Gradle Plugin ile uyumlu değil. Bu yüzden
> ayrı bir JDK 21 kuruldu ve Flutter'a gösterildi. `JAVA_HOME`'u global
> olarak değiştirmeye gerek yok.

### Derleme

```bash
flutter build apk --debug      # sideload için
flutter build apk --release    # yayın anahtarıyla imzalı
```

Release derlemesi `android/key.properties` + `android/hatirlaf.jks`
ikilisini kullanır. Bu dosyalar depoda **yok** ve olmamalı; yedekten
gelirler. Yoksa Gradle uyarı basıp debug anahtarına düşer — o APK
telefonlara **güncelleme olarak kurulamaz**.

İlk derleme **uzun sürer** (~10-15 dk): whisper.cpp dört ABI için
kaynaktan derleniyor. Sonraki derlemeler hızlı.

### Bellek — `tool/flutter.sh`

Bu makinede 15 GB RAM var, `systemd-oomd` ve `earlyoom` **kapalı**. Yani
bir derleyici kaçarsa araya girip onu öldürecek kimse yok; sistem
kilitleniyor. Bir kez web derlemesinde oldu.

Bu yüzden `flutter` doğrudan değil, tavan konmuş hâliyle çağrılıyor:

```bash
tool/flutter.sh run
tool/flutter.sh build apk --debug
```

Script işi bir `systemd` scope'una sokup cgroup sınırı koyuyor: 6 GB'da
çekirdek yavaşlatmaya başlıyor, 8 GB'da süreç ölüyor. Ayrıca
`DART_VM_OPTIONS=--old_gen_heap_size=4096` ile `dart2js`/`frontend_server`
kendi tavanına çarpıp anlaşılır bir hata veriyor. Sınırlar
`FLUTTER_MEM_HIGH` / `FLUTTER_MEM_MAX` ile değiştirilebilir.

Gradle daemon'u bu scope'un dışında kalabildiği için onun sınırları ayrı,
`android/gradle.properties` içinde: heap 3 GB, metaspace 768 MB, Kotlin
daemon 1 GB, `org.gradle.workers.max=4`. (Önceki değerler `-Xmx4G` +
2 GB metaspace idi; tek başına Gradle 6 GB'ı geçebiliyordu. 12 çekirdeğin
hepsiyle whisper.cpp derlemek de ayrı bir kaynak yiyicisi.)

Kalıcı çözüm işletim sistemi tarafında — bir kez, `sudo` ile:

```bash
sudo systemctl enable --now systemd-oomd
```

O zaman RAM biterse çekirdek doğru süreci öldürür, masaüstü ayakta kalır.

### Web derlenmiyor — ve derlenemez

`flutter build web` bu projede zaten anında hata veriyor (`web/` klasörü
yok). Oluşturmaya da çalışmayın: uygulamanın çekirdeği web'de **yok**.

| Paket | Web |
|---|---|
| `whisper_ggml` | ✗ — whisper.cpp native, sadece android/ios/linux/macos/windows |
| `path_provider` | ✗ — tarayıcıda dosya sistemi yok |
| `record`, `just_audio`, `image_picker`, `share_plus` | ✓ |

Yani yazıya çevirme de, hatıraların diske yazılması da düşer; geriye
uygulama kalmaz. `flutter create . --platforms web` demek, saatlerce
derleyip sonuçta çalışmayan bir şey elde etmek olur.

Web'i makine genelinde kapatmadım, çünkü `boşanmakul` projesinin `web/`
klasörü var ve `flutter config --no-enable-web` hepsini birden etkilerdi.

### Gerçek telefona kurma

İki yol var.

**Kablo varsa** (geliştirirken bunu kullanın):

```bash
tool/flutter.sh build apk --release --split-per-abi
tool/telefon.sh
```

`tool/telefon.sh` telefonun ABI'sini kendisi okuyup doğru APK'yi kuruyor.
Telefonda önce geliştirici seçenekleri açılmalı: *Ayarlar > Telefon
hakkında > Yapı numarası*'na 7 kez dokunun, sonra *Geliştirici seçenekleri
> USB hata ayıklama*'yı açın. Kablo veri kablosu olmalı; birçok şarj
kablosunda veri hattı yok, telefon hiç görünmez.

Kod değiştirirken hot reload için kablo takılıyken:

```bash
tool/flutter.sh run --release   # ya da hot reload icin --debug
```

**Kablo yoksa** (telefonu kuracak kişi uzaktaysa): APK'yi WhatsApp,
e-posta veya bir bulut klasörüyle gönderin. Kurulum adımları yukarıdaki
*Kurulum (telefona)* bölümünde.

**`--split-per-abi` neden:** whisper.cpp üç ABI için ayrı ayrı derleniyor,
hepsi tek APK'de olunca dosya 123 MB oluyor. Ayrılınca telefonun ihtiyacı
olan tek APK'yi gönderiyorsunuz. Hangisi olduğundan emin değilseniz
`arm64-v8a`; son ~8 yılın neredeyse bütün telefonları bu.

> Release APK **debug anahtarıyla** imzalanıyor. Sideload için sorun değil,
> ama telefonda başka bir anahtarla imzalanmış eski bir sürüm varsa
> kurulum reddedilir — önce onu kaldırın. Play Store'a çıkılacaksa gerçek
> bir `signingConfig` gerekir.

### Emülatörde test (PC)

```bash
tool/emulator.sh     # ilk seferde AVD'yi kurar, sonra başlatır
tool/flutter.sh run  # emülatöre yükler
```

`tool/emulator.sh` `hatirla` adında bir AVD kuruyor: Android 36
(`google_apis`, x86_64), Pixel 6 profili, 2 GB RAM, 6 GB depolama.
Mikrofon açık (`hw.audioInput=yes`) — bu uygulamada şart, PC'nin mikrofonu
emülatöre geçiyor, kayıt gerçekten test edilebiliyor.

`--sil` AVD'yi silip sıfırdan kurar. Grafik takılırsa:

```bash
EMU_GPU=swiftshader_indirect tool/emulator.sh
```

Emülatör logu `/tmp/emulator-hatirla.log`.

Emülatörde **çalışmayan** tek şey pratikte kamera: sanal kamera var ama
gerçek fotoğraf vermiyor, galeriden seçmek daha kolay. Whisper modeli
emülatöre de ~142 MB inecek, ilk açılışta Wi-Fi hızında bekleyin.

### Klasör adındaki Türkçe karakter

Proje klasörünün adında `ı` var (`hatırlaf`). Gradle bununla sorun
yaşamıyor ama **Dart analiz sunucusu çöküyor** (LSP çerçevelemesi
karakter/bayt karıştırıyor):

```
FormatException: Unexpected end of input
```

Yani `flutter analyze` ve IDE'deki kod tamamlama bu klasörde çalışmaz.
Çözüm klasörü ASCII bir isme almak:

```bash
mv ~/Masaüstü/code/flutter/hatırlaf ~/Masaüstü/code/flutter/hatirla
```

Kod ve derleme bundan etkilenmez.

### Yapı

```
lib/
├── main.dart                     açılış, yerelleştirme, yazı ölçeği sınırı
├── theme.dart                    renkler, ölçüler, buton/yazı temaları
├── data/prompts.dart             soru kütüphanesi
├── models/memory.dart            Memory + JSON
├── services/
│   ├── store.dart                hatıraların tek kaynağı (JSON + dosyalar)
│   ├── cover_photo.dart          ana sayfadaki tek kapak fotoğrafı
│   ├── recorder.dart             mikrofon kaydı
│   ├── player.dart               tek oynatıcı
│   ├── whisper_model_manager.dart model indirme / kalite
│   └── transcriber.dart          yazıya çevirme kuyruğu
├── screens/                      welcome, home, record, memory, question,
│                                 settings, help
├── widgets/                      BuyukButon, MemoryCard, dialoglar
└── utils/format.dart             tarih/süre metinleri
```

Durum yönetimi için ek paket yok: servisler `ChangeNotifier` tekilleri,
arayüz `ListenableBuilder` ile dinliyor.

### Veri nerede duruyor

```
<app documents>/
├── hatiralar.json           dizin (atomik yazılır)
├── hatiralar.json.yedek     bir önceki sürüm
├── kapak.jpg                ana sayfadaki tek fotoğraf
└── hatiralar/<uuid>/
    └── ses.m4a
```

Kapak fotoğrafı dizine girmiyor; varlığı doğrudan dosyadan okunuyor.
Hatıralara bağlı olmadığı için `hatiralar.json` bozulsa bile yerinde
kalır. Hep aynı ada yazıldığından Flutter'ın resim önbelleği eski kareyi
gösterirdi; `CoverPhoto` her değişimde önbelleği boşaltıp bir sürüm
sayacı artırıyor.

Dosya yolları **göreceli** tutulur. Android'de uygulama klasörünün mutlak
yolu yedekten geri yükleme sonrası değişebiliyor; mutlak yol kaydetmek
eski hatıraları "kayıp" gösterirdi.

`hatiralar.json` bozulursa `store.dart` önce yedeği, o da olmazsa
klasörleri tarayarak hatıraları geri kurar. Ses dosyası duruyorsa hatıra
kaybolmaz.

Whisper modeli `getApplicationSupportDirectory()` altında tutulur
(`ggml-base.bin`). Yarım inen dosya `.yarim` uzantısıyla yazılır, ancak
tamamlanınca asıl adına taşınır — yarım model yüklemek whisper.cpp'yi
çökertiyor.

---

## Bilinen sınırlar

- **Web sürümü mümkün değil.** `whisper_ggml` ve `path_provider` web'i
  desteklemiyor; ayrıntı yukarıda.
- **iOS derlenmedi.** Kod iOS'a hazır ama `flutter create --platforms ios`
  ve bir Mac gerekiyor.
- **Uzun kayıtlar yavaş çevrilir.** Eski bir telefonda 10 dakikalık bir
  hatıra `base` modelle ~5-10 dakika sürebilir. Kuyruk arka planda
  çalışır, kullanıcı beklemek zorunda değil — ama uygulama açık kalmalı.
- **Release APK debug anahtarıyla imzalanıyor.** Play Store'a çıkılacaksa
  `android/app/build.gradle.kts` içine gerçek bir `signingConfig` gerekir.
- **Kayıt arka plana alınırsa** (uygulamadan çıkılırsa) Android 14+ süreci
  öldürebilir. Ekran wakelock ile açık tutuluyor ama foreground service
  eklenmedi.

---

## whisper_ggml'in compileSdk çakışması

`whisper_ggml` 2.6.0 kendi `android/build.gradle` dosyasında `compileSdk 34`
yazıyor, ama bağımlılığı `ffmpeg_kit_flutter_new_min` 2.1.0 kendisine bağlı
modüllerin **35+** ile derlenmesini şart koşuyor. İkisi çakışınca derleme
şurada kırılıyor:

```
Execution failed for task ':whisper_ggml:checkDebugAarMetadata'
> Dependency ':ffmpeg_kit_flutter_new_min' requires ... version 35 or later
  :whisper_ggml is currently compiled against android-34.
```

Uygulamanın kendi `compileSdk = 36` değeri eklenti modüllerine geçmiyor.
`android/build.gradle.kts` içindeki `subprojects` bloğu 36'nın altında kalan
modülleri yukarı çekiyor.

Bu blok **`evaluationDependsOn(":app")` bloğundan önce** durmalı. Sonra
konursa projeler çoktan değerlendirilmiş oluyor ve Gradle şunu diyor:

```
Cannot run Project.afterEvaluate(Action) when the project is already evaluated.
```

---

## Neden `permission_handler` yok

`permission_handler_android` 14.1.0 `compileSdk = 37` istiyor. Google bu
platformu `platforms/android-37.0` adıyla yayınlıyor, AGP 9 ise
`android-37` arıyor ve derleme şu hatayla kırılıyor:

```
Failed to find target with hash string 'android-37' in: ~/Android/Sdk
```

Mikrofon iznini zaten `record` paketi istiyor. Geriye kalan iki şey —
"uygulama ayarlarını aç" ve "bir daha sorma seçilmiş mi" — `MainActivity.kt`
içinde ~40 satırlık bir MethodChannel ile çözüldü
(`lib/services/permissions.dart`). Böylece koca bir eklenti ve onunla gelen
derleme kırılganlığı projeden çıktı.

`kaliciReddedildiMi()` yalnızca bir izin isteği **reddedildikten sonra**
çağrılmalı: `shouldShowRequestPermissionRationale()` ilk istekten önce de
`false` döndüğü için aksi halde yanlış pozitif verir.
