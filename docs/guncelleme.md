# Kendi kendine güncelleme

Uygulama mağazada değil. Yine de akrabaların telefonundaki uygulama
kendi kendini güncelliyor. Bu belge o akışın tamamını anlatıyor:
yayınlarken ne yapmanız gerektiğini, telefonda ne olduğunu ve
**yanlış yapılırsa nelerin sessizce bozulacağını**.

---

## 1. Fikir

İki bedava parça var, arada sunucu yok:

| Parça | Nerede |
|---|---|
| Sürüm bilgisi | Deponun `main` dalındaki `guncelleme.json` |
| APK'lar | GitHub sürüm (release) ekleri |

Telefon `raw.githubusercontent.com` üzerinden `guncelleme.json`'u okur,
kendi mimarisine ait APK'yı indirir, sha256 özetini doğrular ve
kullanıcıya **bir kez** sorar.

---

## 2. Üç altın kural

Bu üçü yanlış yapılırsa hata kullanıcının telefonunda, sessizce ortaya
çıkar. Sırasıyla en yıkıcıdan:

### 2.1. İmza anahtarını kaybetmeyin

`android/hatirlaf.jks` + `android/key.properties`. **Depoda yoklar**
(`android/.gitignore`), yedekten gelirler.

Android, imzası farklı bir APK'yı güncelleme olarak kurmaz. Anahtar
kaybolursa tek çıkış yolu uygulamayı silip yeniden kurmaktır — o da
**akrabaların bütün hatıralarını siler**. Git'ten geri getiremezsiniz.

Sertifika parmak izi (doğrulamak için):

```
961fc05bef683bb9944e867c95aa413c86f4198920766c885c6e53860829f1fc
```

```bash
apksigner verify --print-certs build/app/outputs/flutter-apk/app-arm64-v8a-release.apk
```

Anahtar yoksa Gradle uyarı basıp **debug anahtarına düşer** ve ürettiği
APK telefonlara kurulamaz. `tool/yayinla.sh` bunu baştan kontrol edip
durur.

### 2.2. Sıra: önce APK, sonra `guncelleme.json`

`guncelleme.json` "yeni sürüm var" demektir. Telefonlar onu görür görmez
APK'yı indirmeye çalışır. Manifest'i önce iterseniz telefonlar henüz var
olmayan bir dosyayı indirmeye çalışır ve hata döngüsüne girer.

`tool/yayinla.sh` doğru sırayı uygular. Elle yapıyorsanız siz uygulayın.

### 2.3. `versionCode` mimariye göre kayar

Mimariye özel derlemede (`--split-per-abi`) Flutter sürüm kodunu
kaydırıyor:

| Mimari | Kaydırma | `pubspec: 1.0.0+1` → |
|---|---|---|
| `armeabi-v7a` | +1000 | 1001 |
| `arm64-v8a` | +2000 | 2001 |
| `x86_64` | +4000 | 4001 |

Yani telefondaki kurulu `versionCode`, `pubspec.yaml`'daki sayı
**değildir**. Bu yüzden `guncelleme.json` içinde her paket kendi gerçek
kodunu taşır ve `yayinla.sh` bu sayıyı derlenmiş APK'dan okur:

```bash
aapt2 dump badging app-arm64-v8a-release.apk | head -1
```

Elle manifest yazarsanız bu sayıyı uydurmayın; APK'dan okuyun. Yanlış
olursa güncelleme ya hiç önerilmez ya da sonsuza kadar önerilir.

---

## 3. Yeni sürüm yayınlama

```bash
tool/yayinla.sh 1.0.1 "Kayıt düğmesi büyütüldü."
tool/yayinla.sh 1.0.2 "Veri kaybı düzeltildi." --zorunlu
tool/yayinla.sh 1.0.3 "Deneme." --deneme      # hiçbir şey yayınlanmaz
```

Script sırasıyla:

1. **Ön kontroller**, derlemeye başlamadan önce: çalışma dizini temiz mi,
   imza anahtarı yerinde mi, `aapt2`/`gh` var mı, `gh` oturumu açık mı,
   bu sürüm zaten yayınlanmış mı, yerel `main` `origin/main` ile aynı mı.
2. `pubspec.yaml`'daki sürümü yükseltir (`1.0.1+2`).
3. `--split-per-abi` ile üç imzalı APK derler.
4. Her APK'nın **gerçek** `versionCode`'unu, sha256 özetini ve boyutunu
   okuyup `guncelleme.json`'u yazar.
5. **Yalnızca sürüm commit'ini** ve etiketi iter, `gh release create` ile
   APK'ları yükler. `guncelleme.json` bilerek dışarıda bırakılır.
6. Yüklenen her dosyanın, telefonun kullanacağı adresten gerçekten
   indirilebildiğini doğrular (boyut karşılaştırır, 5 kez dener).
7. **Ancak bundan sonra** `guncelleme.json`'u iter.

5-6-7 sırası bu belgenin 2.2'sindeki kuralın koda dökülmüş hâli. 6. adım
başarısız olursa script durur ve `guncelleme.json` **itilmez** —
telefonlar eski sürümde kalır, yani kimse zarar görmez.

Derleme bittikten sonra bir şey ters giderse `pubspec.yaml` otomatik geri
alınır; yarım kalmış bir yayın sürüm numarasını sessizce kaydırmaz.

`--deneme` her şeyi yapar ama hiçbir şey yayınlamaz: üretilecek
`guncelleme.json`'u ve yüklenecek dosyaları gösterir, sonra çalışma
dizinini eski hâline döndürür. Yeni bir şey denerken bunu kullanın.

> `aapt2` (Android SDK `build-tools`) ve `gh` gerekir. Yoksa script
> **derlemeye başlamadan** durur.

### Elle yayınlamak

Script'in yapamadığı bir durum çıkarsa sıra aynı kalmalı:

1. `pubspec.yaml` içindeki `version:` satırını yükseltin (kod **artmalı**).
2. `tool/flutter.sh build apk --release --split-per-abi`
3. Üç APK'yı `hatirlaf-<mimari>.apk` adıyla GitHub sürümüne ekleyin.
4. Yüklendiklerini doğrulayın (adresten indirilebiliyor mu?).
5. **Sonra** `guncelleme.json`'u güncelleyip `main`'e itin.

---

## 4. `guncelleme.json` biçimi

Her yayında **yeniden yazılır**. Deponun kökünde, `main` dalında durur.

```json
{
  "surumAdi": "1.0.1",
  "notlar": "Kayıt düğmesi büyütüldü.",
  "zorunlu": false,
  "paketler": {
    "arm64-v8a": {
      "surumKodu": 2002,
      "apkUrl": "https://github.com/oktayelb/hatirlaf/releases/latest/download/hatirlaf-arm64-v8a.apk",
      "sha256": "…64 hane…",
      "boyut": 23404032
    },
    "armeabi-v7a": { "surumKodu": 1002, "…": "…" },
    "x86_64":      { "surumKodu": 4002, "…": "…" }
  }
}
```

| Alan | Anlamı |
|---|---|
| `surumAdi` | Kullanıcıya gösterilen etiket. Karar verirken kullanılmaz |
| `notlar` | "Neler değişti" kutusunda görünür. Sade, tek iki cümle |
| `zorunlu` | `true` ise erteleme 3 gün yerine 20 saat. Kurulum yine zorlanamaz |
| `paketler.<mimari>.surumKodu` | **O APK'nın** gerçek `versionCode`'u. Tek karşılaştırma ölçütü |
| `paketler.<mimari>.sha256` | İnen dosya bununla doğrulanır. Tutmazsa kurulmaz |
| `paketler.<mimari>.boyut` | Bayt. Hem ilerleme hem erken doğrulama için |

**Adres kısıtı:** `apkUrl` yalnızca
`https://github.com/oktayelb/hatirlaf/releases/…` altında olabilir.
Manifest bir şekilde değiştirilse bile uygulama başka bir yerden APK
indirip kuramaz. Depo adını değiştirirseniz `GuncellemeKaynagi` içindeki
`sahip`/`depo` sabitlerini de değiştirin.

`latest/download/…` adresi her zaman en son sürüme gider; dosya adresleri
sürümden sürüme değişmez.

Bozuk, eksik ya da şüpheli bir manifest **sessizce yok sayılır** —
uygulama çökmez, kullanıcı bir şey görmez.

---

## 5. Telefonda ne oluyor

### 5.1. Tasarımın özü

**Denetim ve indirme tamamen görünmezdir.** Yaşlı kullanıcı ne ilerleme
çubuğu, ne "internete bağlanılamadı" uyarısı, ne de iptal edebileceği bir
işlem görür. İnternet yoksa hiçbir şey olmaz ve hiçbir şey söylenmez.

Soru yalnızca APK inip **özeti doğrulandıktan sonra**, yani geriye iki
dokunuş kaldığında sorulur. Sebebi basit: bu kullanıcılar yarıda kalan
bir işlemi kendileri kurtaramaz. Her ilerleme çubuğu, iptal edilebilen
her işlem ve anlaşılmayan her hata bir telefon görüşmesi demek.

### 5.2. Akış

1. **Açılışta** (en fazla 20 saatte bir) `guncelleme.json` okunur.
2. Kendi mimarisine ait paket seçilir; `surumKodu` kuruludan büyükse APK
   iner (sessizce, kesilirse `Range` ile devam ederek).
3. sha256 doğrulanır. Tutmazsa dosya atılır, bir daha denenir.
4. **Oturum başında** hazırsa tek ekran gösterilir.
5. "Güncelle" → `PackageInstaller` oturumu → Android'in onay penceresi.
6. Kullanıcı onaylar, Android APK'yı yerinde değiştirir, süreç ölür.
7. Uygulama yeni sürümle açılır. Hatıralar aynı yerde durur (aynı paket,
   aynı imza). Sonraki açılışta artık gereksiz APK silinir.

### 5.3. Ne zaman sorulur, ne zaman sorulmaz

| Durum | Davranış |
|---|---|
| İnternet yok (günlerce) | Sessizlik. **Başarısız denetim zaman damgası bırakmaz**, ilk bağlantıda hemen denenir |
| Mobil veri | İner. ~22 MB için mobil veriyi beklemeye değmiyor |
| Uygulama açıkken ağa girildi | Ağ dinlendiği için indirme kendiliğinden başlar |
| İndirme yarıda kesildi | Yarım dosya saklanır, `Range` ile kaldığı yerden devam eder |
| İnen dosya bozuk | sha256 tutmazsa atılır, asla kurulmaz |
| **Kullanım sırasında indi** | Ekran **basmaz**. Soru bir sonraki açılışta sorulur |
| Kayıt ya da yazıya çevirme sürüyor | Ekran açılmaz; kurulum süreci öldüreceği için iş yarıda kalmaz |
| Ana ekranda değil | Ekran açılmaz, kullanıcının işi bölünmez |
| "Vazgeç" dendi | 3 gün (`zorunlu` ise 20 saat) ertelenir, ısrar edilmez |
| Cihazın mimarisi manifest'te yok | Güncelleme önerilmez |
| İmza uyuşmuyor | "Aileden biri yardım etsin" denir; **asla** "silip yeniden kurun" denmez |

### 5.4. Diskte ne tutuluyor

```
<app support>/guncelleme/hatirlaf-<surumKodu>.apk        tamamlanmış
<app support>/guncelleme/hatirlaf-<surumKodu>.apk.yarim  yarım inen
```

`SharedPreferences` anahtarları:

| Anahtar | İşi |
|---|---|
| `guncelleme_son_denetim` | Son **başarılı** denetim zamanı |
| `guncelleme_son_bilgi` | Son manifest, ham JSON |
| `guncelleme_ertelenen_surum` | "Sonra" denen sürüm kodu |
| `guncelleme_erteleme_bitisi` | Ertelemenin bittiği an |

`guncelleme_son_bilgi` diskte tutulduğu için dün inmiş bir güncelleme,
bugün internet hiç olmasa bile kurulabilir. İnternet yalnızca
*indirmek* için gerekli.

Telefonun saati elle değiştirilebildiği için hem denetim aralığı hem
erteleme, geleceğe ait damgalara karşı korumalı — yoksa yanlış bir saat
güncellemeyi sonsuza kadar kilitleyebilirdi.

---

## 6. Android'in dayattığı sınırlar

Bunlar tercih değil, kısıt. Değiştirilemezler.

**Sessiz kurulum yok.** Cihaz sahibi (device owner) olarak sağlanmamış
bir uygulama, sistemin *"Bu uygulamayı güncellemek istiyor musunuz?"*
penceresini göstermek zorunda. Device owner, fabrika ayarlarına dönüp
her telefonu tek tek kaydetmek demek. Yapabildiğimiz en iyi şey o
pencereye kadar olan her şeyi görünmez halletmek.

**"Bilinmeyen kaynak" izni bir kez gerekir.** `REQUEST_INSTALL_PACKAGES`
manifest'te var ama kullanıcının bir kez "bu kaynağa izin ver" demesi
gerekiyor. **Telefonu teslim ederken bu ayarı kendiniz açın**; sonrasında
kullanıcıya tek dokunuş + sistem penceresi kalır. Açık değilse uygulama
onu anlatan bir ekran gösterir.

**Örtük yayınlar manifest'teki alıcılara ulaşmaz.** Android 8'den beri.
Kurulum sonucu `Intent(context, KurulumAlicisi::class.java)` ile
**açıkça** gönderiliyor; eylem adı + `setPackage()` yetmiyor, oturum
commit ediliyor ama sonuç hiç gelmiyordu.

**`addPostFrameCallback` yalnızca kare çizilirse çalışır.** Durgun bir
ana ekranda Flutter kare üretmez. Güncelleme ekranını açan geri çağırma
sonsuza kadar bekliyordu; `ensureVisualUpdate()` gerekirse bir kare
planlıyor.

**`PopScope(canPop: false)` içinde `maybePop()` çağırmayın.** Geri
çağırmayı yeniden tetikler, o da tekrar kapatmayı dener — sonsuz döngü.
Güncelleme ekranı `canPop: true` kullanıyor ve ertelemeyi çıkışta
kaydediyor.

---

## 7. Dosyalar

| Dosya | İşi |
|---|---|
| `lib/services/update_info.dart` | Manifest çözümleme, mimari seçimi, adres kısıtı, erteleme politikası (saf, test edilebilir) |
| `lib/services/updater.dart` | Durum makinesi: denetle → indir → doğrula → kur |
| `lib/services/network.dart` | Ağ durumu akışı |
| `lib/screens/update_screen.dart` | Kullanıcının gördüğü tek ekran |
| `lib/screens/home_screen.dart` | `_GuncellemeGozcusu`: ekranı **doğru anda** açan görünmez gözcü |
| `lib/screens/settings_screen.dart` | Sürüm/durum bölümü (kuran kişi için) |
| `android/.../Guncelleyici.kt` | `PackageInstaller` oturumu, sürüm/mimari, izin |
| `android/.../KurulumAlicisi.kt` | Kurulum sonucu alıcısı |
| `android/.../AgGozcusu.kt` | Ağ durumu (`EventChannel`) |
| `tool/yayinla.sh` | Yayınlama |
| `guncelleme.json` | Yayınlanan sürüm bilgisi |

Kanallar: `hatirla/guncelleme` (MethodChannel), `hatirla/ag` (EventChannel).

---

## 8. Yerel test

Gerçek bir yayın yapmadan, emülatörde baştan sona denemek için. GitHub'a
hiçbir şey gitmez.

**1. Geçici yama** — `lib/services/update_info.dart`:

```dart
static Uri bilgiAdresi() => Uri.parse('http://10.0.2.2:8000/guncelleme.json'
    '?t=${DateTime.now().millisecondsSinceEpoch}');
static const String izinliSunucu = '10.0.2.2';
static String get izinliYolBasi => '/';
// _adresGuvenliMi içinde http'ye de izin verin
```

`AndroidManifest.xml` içindeki `<application>` etiketine:

```xml
android:usesCleartextTraffic="true"
```

**2. İki APK derleyin** (biri `+1`, diğeri `+2`), ikincisini bir klasöre
`hatirlaf-x86_64.apk` olarak koyun, yanına `guncelleme.json` yazın
(`surumKodu` = `aapt2 dump badging` çıktısındaki sayı, emülatörde 4002).

**3. `Range` destekli bir sunucu** çalıştırın (`python3 -m http.server`
`Range` desteklemez; kesilen indirmeyi test etmek için gerekir).

**4. Kurun ve izleyin:**

```bash
adb install -r v1.apk
adb shell appops set com.hatirla.hatirla REQUEST_INSTALL_PACKAGES allow
adb logcat -s KurulumAlicisi:V
```

**5. Bitince geçici yamaları geri alın.** Testler `github.com` beklediği
için yama duruyorken `flutter test` kırmızı yanar — bu normal.

---

## 9. Sorun giderme

**Ayarlar → Uygulama Sürümü**, kuran kişi için: kurulu sürüm, son
denetim zamanı, bekleyen sürüm, indirilecek boyut + mimari, teknik hata
metni ve elle denetim düğmesi. "Neden güncellenmedi?" sorusunun cevabı
telefonla değil buradan okunur.

```bash
adb logcat -s KurulumAlicisi:V   # kurulum durum kodu ve sistem mesajı
```

| Belirti | Muhtemel sebep |
|---|---|
| Hiç güncelleme önerilmiyor | `surumKodu` kaydırmayı hesaba katmıyor (bkz. 2.3); ya da cihazın mimarisi manifest'te yok |
| "İmza uyuşmuyor" | Kurulu sürüm debug anahtarıyla imzalanmış. Kullanıcı çözemez |
| İndiriyor ama kurmuyor | sha256 tutmuyor — manifest APK'dan önce mi itildi? |
| Ekran hiç açılmıyor | Güncelleme kullanım sırasında indi; bir sonraki açılışta sorulur |
| Sistem penceresi çıkmıyor | "Bilinmeyen kaynak" izni kapalı |
