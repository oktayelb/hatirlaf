# Kendi kendine güncelleme

Sunucu yok: sürüm bilgisi `main` dalındaki `guncelleme.json`, APK'lar
GitHub sürüm ekleri. Telefon manifest'i okur, kendi mimarisinin APK'sını
sessizce indirir, sha256 doğrular ve sonraki açılışta bir kez sorar.
Güncelleme zorunludur.

Tam APK inmiyor: bir önceki sürümden geliyorsa telefon ~3 MB'lık bir
fark yaması indirip kendi kurulu APK'sına uyguluyor (aşağıda
[Fark güncellemesi](#fark-güncellemesi)).

## Üç kural

**1. İmza anahtarını kaybetmeyin.** `android/hatirlaf.jks` dosyası ve
parolaları (`.env` içinde `ANDROID_*`), ikisi de depoda yok. Android farklı imzalı bir APK'yı
güncelleme olarak kurmaz; anahtar kaybolursa tek çıkış uygulamayı silmek
ve bu bütün hatıraları siler. Parmak izi:

```
961fc05bef683bb9944e867c95aa413c86f4198920766c885c6e53860829f1fc
```

**2. Önce APK, sonra `guncelleme.json`.** Manifest "yeni sürüm var"
demektir; önce itilirse telefonlar olmayan bir dosyayı indirmeye çalışır.

**3. `versionCode` mimariye göre kayar.** `--split-per-abi` ile
armeabi-v7a +1000, arm64-v8a +2000, x86_64 +4000. Manifest'teki sayıyı
uydurmayın, APK'dan okuyun (`aapt2 dump badging`).

## Yayınlama

```bash
tool/yayinla.sh bug     "Kayıt düğmesi bazen çalışmıyordu."
tool/yayinla.sh feature "Fotoğraf eklenebiliyor."
tool/yayinla.sh version "Yeni hatıra defteri."
tool/yayinla.sh feature "Deneme." --deneme   # hiçbir şey yayınlanmaz
tool/yayinla.sh 3.0.0   "..."                # açık numara
```

Sürüm numarasını script `pubspec.yaml`'dan hesaplar: `bug` 1.4.2 → 1.4.3,
`feature` → 1.5.0, `version` → 2.0.0. Argümansız çalıştırırsanız
seçenekleri yazar.

Sırasıyla: ön kontroller (temiz dizin, imza anahtarı, `aapt2`/`gh`,
sürüm çakışması) → `pubspec.yaml` yükseltme → üç imzalı APK →
`guncelleme.json` yazma → sürüm commit'i + etiket + APK yükleme →
yüklenenlerin indirilebildiğini doğrulama → **ancak sonra**
`guncelleme.json`'u itme. Son adımdan önce durursa telefonlar eski
sürümde kalır. Hata olursa `pubspec.yaml` geri alınır.

Elle yayınlarken sıra aynı kalmalı.

## `guncelleme.json`

```json
{
  "surumAdi": "1.0.1",
  "notlar": "Kayıt düğmesi büyütüldü.",
  "paketler": {
    "arm64-v8a": {
      "surumKodu": 2002,
      "apkUrl": "https://github.com/oktayelb/hatirlaf/releases/download/v1.0.1/hatirlaf-arm64-v8a.apk",
      "sha256": "…64 hane…",
      "boyut": 23404032,
      "yamalar": [
        {
          "kaynakSurumKodu": 2001,
          "kaynakSha256": "…kurulu APK'nın özeti…",
          "url": "https://github.com/oktayelb/hatirlaf/releases/download/v1.0.1/hatirlaf-arm64-v8a-2001.yama",
          "sha256": "…64 hane…",
          "boyut": 3041692
        }
      ]
    }
  }
}
```

`yamalar` isteğe bağlı: bozuk ya da eksik bir kayıt güncellemeyi iptal
ettirmez, o telefon tam APK indirir. Eski sürümler alanı zaten yok
sayar.

Karar **yalnızca** `surumKodu`'na bakar; `surumAdi` sadece etikettir.
`apkUrl` yalnızca `https://github.com/oktayelb/hatirlaf/releases/…`
altında olabilir (depo adı değişirse `GuncellemeKaynagi` sabitleri de
değişmeli). Adresler `latest`'e değil etikete sabitlenir: `latest`
hareketli bir hedef, yeni sürümde eski manifest'in adresi kayar ve sha256
tutmaz. Bozuk manifest sessizce yok sayılır.

## Fark güncellemesi

APK bir zip ve iki sürüm arasında girdilerin neredeyse hepsi bayt bayt
aynı kalıyor — 1.1.0 → 1.2.0'da 560 girdinin 554'ü, 19.7 MB. Değişmeyen
kısım telefonda zaten duruyor: kurulu APK'nın kendisi
(`applicationInfo.sourceDir`, uygulama kendi APK'sını okuyabiliyor).

Yama bu yüzden iki komuttan ibaret: "şunu kurulu APK'nın şu ofsetinden
kopyala", "şunu benden al". Ölçülen: **2.90 MB yama, 22.6 MB APK
yerine** — %87 az. Yamanın kendisi neredeyse tamamen `libapp.so`
(Dart kodu); o her sürümde baştan derlendiği için yama boyutu
değişikliğin büyüklüğünden çok etkilenmiyor.

Bicim ve komutlar: `lib/services/yama.dart` sınıf belgesinde.

### Yanlış giderse

Her aşamada sessizce tam APK'ya düşülür; kullanıcı ikisini ayırt edemez:

| Durum | Ne olur |
|---|---|
| Manifest'te yama yok / kurulu sürüme uymuyor | Tam APK |
| Kurulu APK yamanın beklediği dosya değil (`kaynakSha256`) | Tam APK |
| İnen yama bozuk (`sha256`) | Tam APK |
| Yama uygulanamadı (bozuk komut, sınır dışı ofset) | Tam APK |
| Çıkan APK manifest'in özetini tutmuyor | Tam APK |

Son satır önemli: yamadan çıkan APK da tam inen APK kadar doğrulanıyor.
Bozuk bir yamanın maliyeti 3 MB boşa trafik, asla hatalı kurulum değil.

### Üretimi

`yayinla.sh` bir önceki etiketin APK'larını indirip (`build/eski-apk/`
altında saklanır) her mimari için yama üretir, sonra **telefondaki kodla
doğrular**: `tool/yama_dogrula.dart` yamayı `lib/services/yama.dart` ile
uygulayıp çıkan dosyanın sha256'sını karşılaştırır. Tutmazsa yayınlama
orada durur — üretici Python ile uygulayıcı Dart birbirinden ayrılırsa
bunu kullanıcının telefonunda değil burada öğrenelim.

Yalnızca bir önceki sürümden yama üretiliyor: güncelleme zorunlu ve 20
saatte bir denetleniyor, telefonlar neredeyse her zaman N-1'de. Sürüm
atlayan telefon tam APK indirir.

## Telefonda

Denetim ve indirme görünmezdir: ilerleme çubuğu, hata uyarısı, iptal
edilebilir işlem yok. Soru yalnızca APK inip özeti doğrulandıktan sonra,
**oturum başında**, ana ekrandayken ve kayıt/çevirme sürmüyorken sorulur.

| Durum | Davranış |
|---|---|
| İnternet yok | Sessizlik; başarısız denetim damga bırakmaz |
| Mobil veri | İner (yama ~3 MB, tam APK ~22 MB) |
| Bir önceki sürümden geliyor | Yama iner, kurulu APK'ya uygulanır |
| Sürüm atlamış / yama tutmadı | Sessizce tam APK'ya düşer |
| İndirme kesildi | `Range` ile kaldığı yerden devam |
| sha256 tutmuyor | Dosya atılır, asla kurulmaz |
| Kullanım sırasında indi | Sonraki açılışta sorulur |
| Mimari manifest'te yok | Önerilmez |
| İmza uyuşmuyor | "Aileden biri yardım etsin"; asla "silip kurun" denmez |

Denetim aralığı en az 20 saat. Diskte:
`<app support>/guncelleme/hatirlaf-<surumKodu>.apk` (yarım inen `.yarim`,
inen yama `.yama`)
ve `SharedPreferences`: `guncelleme_son_denetim`, `guncelleme_son_bilgi`. Manifest
diskte tutulduğu için dün inmiş bir güncelleme bugün internetsiz de
kurulabilir. Telefonun saati geri alınabildiği için zaman damgaları
geleceğe karşı korumalı.

## Android kısıtları

- **Sessiz kurulum yok.** Device owner olmayan uygulama sistemin onay
  penceresini göstermek zorunda.
- **"Bilinmeyen kaynak" izni bir kez gerekir.** Telefonu teslim ederken
  siz açın; kapalıysa uygulama anlatan bir ekran gösterir.
- **Örtük yayınlar manifest alıcılarına ulaşmaz** (Android 8+); kurulum
  sonucu açık `Intent` ile gönderiliyor.
- **`addPostFrameCallback` kare çizilmezse çalışmaz**; durgun ekranda
  `ensureVisualUpdate()` gerekiyor.
- **`PopScope(canPop: false)` içinde `maybePop()` çağırmayın** — sonsuz
  döngü.

## Dosyalar

| Dosya | İşi |
|---|---|
| `lib/services/update_info.dart` | Manifest çözümleme, mimari seçimi, adres kısıtı (saf) |
| `lib/services/updater.dart` | Durum makinesi: denetle → (yama\|tam) indir → doğrula → kur |
| `lib/services/yama.dart` | Yamayı kurulu APK'ya uygular (Flutter'a bağımlı değil) |
| `lib/services/network.dart` | Ağ durumu akışı |
| `lib/screens/update_screen.dart` | Kullanıcının gördüğü tek ekran |
| `lib/screens/home_screen.dart` | `_GuncellemeGozcusu` |
| `android/.../Guncelleyici.kt` | `PackageInstaller` oturumu |
| `android/.../KurulumAlicisi.kt` | Kurulum sonucu |
| `android/.../AgGozcusu.kt` | Ağ durumu (`EventChannel`) |
| `tool/yayinla.sh` | Yayınlama |
| `tool/yama_uret.py` | İki APK arasında yama üretir |
| `tool/yama_dogrula.dart` | Yamayı telefondaki kodla dener, yayını durdurabilir |

Kanallar: `hatirla/guncelleme` (MethodChannel), `hatirla/ag`
(EventChannel).

## Yerel test

`update_info.dart` içindeki adres/sunucu sabitlerini `10.0.2.2:8000`'e
çevirin, `_adresGuvenliMi`'de http'ye izin verin, manifest'e
`android:usesCleartextTraffic="true"` ekleyin. İki APK derleyip (`+1`,
`+2`) ikincisini `hatirlaf-x86_64.apk` olarak, yanına `guncelleme.json`
ile birlikte `Range` destekleyen bir sunucuda yayınlayın
(`python3 -m http.server` desteklemez).

```bash
adb install -r v1.apk
adb shell appops set com.hatirla.hatirla REQUEST_INSTALL_PACKAGES allow
adb logcat -s KurulumAlicisi:V
```

Bitince bu değişiklikleri geri alın; duruyorken `flutter test`
kırmızı yanar.

## Sorun giderme

**Ayarlar → Uygulama Sürümü** kuran kişi için kurulu sürümü, son denetimi,
bekleyen sürümü, teknik hata metnini ve elle denetim düğmesini gösterir.

| Belirti | Sebep |
|---|---|
| Hiç önerilmiyor | `surumKodu` kayması hesaba katılmamış ya da mimari yok |
| "İmza uyuşmuyor" | Kurulu sürüm debug anahtarıyla imzalı |
| İndiriyor ama kurmuyor | sha256 tutmuyor — manifest APK'lardan önce mi itildi? |
| Ekran açılmıyor | Güncelleme kullanım sırasında indi; sonraki açılışta |
| Sistem penceresi çıkmıyor | "Bilinmeyen kaynak" izni kapalı |
