# Kayıt ve kurtarma

Bu uygulamada kaybolan bir kayıt, kaybolan bir dosya değil: bir daha
anlatılmayacak bir hikâye. Kayıt yolundaki her karar buna göre alındı.

## İki koruma

**1. Kayıt sürerken süreç öldürülmesin.** Android 12'den beri arka plana
alınan bir uygulamanın toplanması normaldir. `wakelock` ekranı açık
tutar, **süreci tutmaz** — onu ancak bir ön plan servisi yapabilir.
Kayıt başlarken `KayitServisi` açılıyor, kayıt biterken kapanıyor.

**2. Süreç yine de ölürse kayıt kurtarılsın.** Ses, MPEG-4 (`.m4a`)
yerine **ADTS akışı** (`.aac`) olarak yazılıyor ve her iki saniyede bir
diske boşaltılıyor.

Biçim seçimi meselenin tamamı:

| | `.m4a` (MPEG-4) | `.aac` (ADTS) |
|---|---|---|
| Süre ve çerçeve tablosu | `moov` atomunda, **dosyanın sonunda** | her çerçevenin kendi başlığında |
| Yarıda kesilirse | dosya hiç açılmaz | o ana kadarki kısım geçerli |
| Boyut | 64 kbps | 64 kbps (aynı kodlayıcı) |

`record` paketi dosyaya yazarken hep MPEG-4 muxer kullanıyor; ADTS'i
yalnızca `startStream()` veriyor. Bu yüzden çerçeveleri akıştan alıp
diske kendimiz yazıyoruz (`Recorder._akisaBasla`).

Akış açılamayan bir cihaz olursa eski yola, dosyaya yazmaya düşülüyor —
kayıt hiç alınamamaktansa kurtarılamaz olsun.

## Kurtarma nasıl işliyor

Kayıt başlarken hatıra klasörüne bir işaret bırakılıyor
(`hatiralar/<id>/kayit.json`): id, başlık, soru, ses dosyasının adı,
başlangıç zamanı. Normal akışta bu dosya siliniyor. Kalmışsa kayıt
yarıda kesilmiş demektir.

Açılışta `Kurtarma.tara()` işaretli klasörleri geçiyor:

| Durum | Ne olur |
|---|---|
| Hatıra zaten dizinde | İşaret bayat; yalnızca işaret silinir |
| İşarette `bitti: true` | Ses tamamdı, dizine yazılamamış; olduğu gibi kurtarılır |
| Ses ADTS olarak okunuyor | Son tam çerçeveye kadar kırpılır, süresi çerçevelerden hesaplanır, kurtarılır |
| Ses okunamıyor (yarım `.m4a`) | Klasör silinir: açılmayan bir hatıra listede durmasın |
| Ses yok ya da 1 KB'den küçük | Klasör silinir |

Kurtarma **sessiz**: kullanıcıya pencere açılmaz, hatıra listede kendi
adıyla belirir. Yedekleme ve güncelleme de böyle çalışıyor; yaşlı
kullanıcıya açıklaması zor bir uyarı çıkarmaktansa hatırayı yerine koyup
susmak doğru.

Kaybedilen en fazla diske boşaltılmamış **son iki saniye** oluyor.
Elektrik kesintisi gibi işletim sisteminin de çöktüğü durumlarda daha
fazlası gidebilir; `flush()` veriyi çekirdeğe geçirir, diske indiğini
garanti etmez.

## Bildirim görünmeyebilir

Android 13'ten beri bildirim göstermek ayrı bir izin istiyor
(`POST_NOTIFICATIONS`) ve uygulama onu **bilerek istemiyor**: yaşlı
kullanıcıya bir izin penceresi daha çıkarmamak için. İzin yoksa bildirim
gizli kalır, **servis yine çalışır** — korunan şey bildirim değil,
süreç.

## Telefonda denemek

```bash
adb install -r hatirlaf-arm64-v8a.apk
```

**Ön plan servisi duruyor mu?** Kayda başlayın, sonra:

```bash
adb shell dumpsys activity services com.hatirla.hatirla | grep -A3 KayitServisi
```

`isForeground=true` ve `foregroundServiceType=...MICROPHONE` görünmeli.

**Kurtarma çalışıyor mu?** Kayda başlayın, bir dakika konuşun, sonra
süreci öldürün (uygulamayı durdurmak değil — gerçek bir ölüm):

```bash
adb shell am kill com.hatirla.hatirla      # arka plandayken
# ya da daha sert:
adb shell killall -9 com.hatirla.hatirla
```

Uygulamayı yeniden açın: hatıra listede, konuştuğunuz süreyle durmalı ve
açıp dinleyebilmelisiniz. Logda:

```bash
adb logcat -s flutter:V | grep -i kurtar
```

**Dosya gerçekten ADTS mi?**

```bash
adb shell run-as com.hatirla.hatirla ls files/hatiralar/<id>/
```

`ses.aac` görünmeli (`ses.m4a` görünüyorsa cihaz akışı desteklememiş,
yedek yola düşülmüş demektir).

## Dosyalar

| Dosya | İşi |
|---|---|
| `lib/services/recorder.dart` | Kayıt: akış modu, yedek dosya modu, diske boşaltma |
| `lib/services/adts.dart` | Çerçeve sayma: süre ve dosyanın nerede bittiği |
| `lib/services/kurtarma.dart` | İşaret dosyası ve açılıştaki tarama |
| `lib/services/kayit_servisi.dart` | Ön plan servisinin Dart tarafı |
| `android/.../KayitServisi.kt` | Ön plan servisi |
| `lib/screens/record_screen.dart` | İşareti bırakan ve silen ekran |

Kanal: `hatirla/kayit` (MethodChannel).
