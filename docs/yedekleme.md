# Aile yedeği

Kayıtlar telefonda şifrelenip Backblaze B2'ye kopyalanır. Sunucu yok,
kullanıcıya soru sorulmaz. Şifreleme **asimetrik**: APK'ya yalnızca açık
anahtar girer, çözmek yalnızca sizdeki gizli anahtarla mümkündür.

## Üç kural

**1. Gizli anahtarı kaybetmeyin.** Kaybolursa B2'deki her şey çöp olur.
Felaket değil — telefonlardaki asıl kayıtlar duruyor — ama 20-30 telefonu
tek tek toplamak demek. Birden fazla yerde saklayın.

**2. Telefondaki kayıt asla silinmez.** Yükleme bir kopyadır. Gizli
anahtar kaybolursa tek çıkış kapısı odur, ve kod bu varsayıma dayanıyor.

**3. B2 anahtarı yalnızca `writeFiles` olmalı.** APK'yı parçalayan biri
kimsenin kaydını indiremesin, silemesin. Uygulama açılışta fazla yetkili
anahtarı fark edip logda uyarır.

## Hangi sır nereye

İki dosya var ve aradaki fark güvenliğin tamamı:

| Dosya | İçindekiler | APK'ya girer mi? |
|---|---|---|
| `.env` | Keystore parolaları, X25519 **gizli** anahtar yolu, B2 hesap parolası | **Hayır.** Yalnızca bu PC'de |
| `yedek.json` | `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET_ID`, `YEDEK_ALICI_ANAHTARI` | **Evet.** Derlemeye gömülür |

İkisi de `.gitignore`'da. Şablonları: `.env.ornek`, `yedek.json.ornek`.

`yedek.json` içindekiler APK'nın içine gömülür, yani **gizli tutulamaz**
— parçalayan herkes okur. Bu bir kusur değil, tasarım: bu yüzden oradaki
B2 anahtarı yalnızca `writeFiles` yetkili olmalı ve oradaki X25519
anahtarı **açık** olan olmalı. Sızdığında yapılabilecek en kötü şey
kovaya çöp yüklemektir.

`.env` içindekiler hiçbir derlemeye girmez. Keystore parolaları APK'yı
*imzalamakta* kullanılır, içine konmaz; gizli X25519 anahtarı yalnızca
sizin PC'nizde çözmek için durur.

## Kurulum

```bash
# 1. Anahtar çifti
tool/anahtar_uret.py ~/hatirlaf-yedek-anahtari.gizli

# 2. B2'de kova + YALNIZCA writeFiles yetkili uygulama anahtarı
#    (Backblaze konsolu -> Application Keys -> Add a New Application Key)

# 3. Bu PC'de kalacaklar
cp .env.ornek .env
$EDITOR .env          # ANDROID_*, YEDEK_GIZLI_ANAHTAR, B2_ACCOUNT_PASSWORD

# 4. Derlemeye girecekler
cp yedek.json.ornek yedek.json
$EDITOR yedek.json    # B2_KEY_ID, B2_APP_KEY, B2_BUCKET_ID, YEDEK_ALICI_ANAHTARI
```

`yedek.json`'daki dört alan dolu değilse yedekleme tamamen kapalıdır ve
uygulama eskisi gibi çalışır. `tool/yayinla.sh` dosya varsa
`--dart-define-from-file` ile derler, yoksa uyarıp yedeklemesiz sürüm
üretir.

`android/key.properties` artık yok; imza parolaları `.env`'e taşındı.
`android/app/build.gradle.kts` önce `.env`'e, sonra gerçek ortam
değişkenlerine bakar.

## Kayıtları alma

```bash
# B2'den indirin (b2 CLI, rclone, konsol — nasıl isterseniz)
b2 sync b2://kovaniz ./gelenler

# Toptan çözün. Anahtar yolu .env'deki YEDEK_GIZLI_ANAHTAR'dan gelir;
# istersen ilk argüman olarak da verebilirsin.
tool/yedek_coz.py --klasor ./gelenler
```

Her hatıra iki dosyadır: `<cihaz>/<hatıraId>/ses.m4a.hyz` ve
`bilgi.json.hyz` (başlık, tarih, süre, soru, **metin dökümü**). Metin de
şifrelenir: aranabilir olduğu için sesten daha hassas.

Çözücü tek bozuk dosyada durmaz, atlar ve sonunda kaç hata olduğunu
söyler.

## Şifreleme biçimi

`lib/services/backup_crypto.dart`. Dosya başına rastgele bir X25519
geçici anahtar üretilir, alıcının açık anahtarıyla ECDH yapılır, HKDF-
SHA256 ile o dosyaya özel bir AES-256-GCM anahtarı türetilir.

```
"HTRLF1"   6    sihirli sayı
sürüm      1    0x01
parçaBoyu  4    düz metin parça boyu (varsayılan 1 MiB)
alıcıİz    8    sha256(alıcıAçıkAnahtarı)[0..8]
geçici     32   bu dosyaya özel X25519 açık anahtarı
--------------- 51 bayt başlık
her parça: şifreliMetin + 16 bayt etiket
```

Parçalı olmasının sebebi bellek: 300 MB'lik bir kayıt tek seferde RAM'e
alınamaz. Her parçanın nonce'ı sıra numarasıdır (dosya anahtarı her
dosyada farklı olduğu için sayaç güvenli), AAD'si ise **başlık + sıra +
son mu**. Bu üçü şunları imkânsız kılar:

| Saldırı | Neden tutmaz |
|---|---|
| Gövdede bit çevirme | GCM etiketi tutmaz |
| Parçaları yer değiştirme | AAD'deki sıra numarası tutmaz |
| Dosyayı budama | Son parça "son" işaretli değil |
| Başlığı kurcalama | Başlık AAD'nin içinde |
| Başka anahtarla açma | Parmak izi tutmaz |

Bu vakaların hepsi `test/backup_crypto_test.dart` içinde.

**Biçim değişirse `tool/yedek_interop.sh` çalıştırın.** Dart şifreler,
Python çözer, sonuç bayt bayt karşılaştırılır. İki taraf birbirini
anlamazsa bu ancak veriye ihtiyaç duyduğunuz gün ortaya çıkar — yani en
kötü gün.

## Telefonda

| Durum | Davranış |
|---|---|
| Wi-Fi yok | Sessizce bekler (mobil veriyle yüklemez) |
| Yükleme yarıda kesildi | O hatıra işaretlenmez, baştan denenir |
| Ses gitti, bilgi gitmedi | Yarım sayılır, tekrar denenir |
| Üst üste hata | 2 saat bekler |
| Ayarlar eksik | Yedekleme tamamen kapalı |

Tetikleyiciler: açılış, ön plana dönüş, ağın gelmesi (3 sn sakinleşmeyle),
yeni kayıt ya da biten çevirme. Şifreleme ayrı bir isolate'te; 300 MB'lik
bir kayıt ana isolate'te şifrelenseydi ekran saniyelerce donardı.

Diskte: `<app support>/yedek/` (geçici şifreli kopyalar, yükleme sonrası
silinir) ve `SharedPreferences`: `yedek_cihaz`, `yedek_yuklenen`,
`yedek_son_deneme`.

**Ayarlar → Aile Yedeği** kuran kişiye durumu, gönderilen/bekleyen
sayısını, cihaz kimliğini ve teknik hata metnini gösterir. Kullanıcıya
hiçbir hata gösterilmez: yaşlı bir kullanıcının "yükleme başarısız"
uyarısıyla yapabileceği bir şey yok.

## Rıza

Yedekleme açıkken Ayarlar ekranındaki kapanış cümlesi değişir: "sesiniz
telefonunuzdan dışarı çıkmaz" yerine kayıtların şifrelenip aileden bir
kişiye ulaştığı yazar. Kapalıyken eski cümle geçerli.

Bu bir incelik değil: kapalı bir aile çemberinde bile insanların kendi
hayat hikâyelerinin nereye gittiğini bilmeye hakkı var, ve bunu sonradan
eklemek baştan yazmaktan çok daha zor.

## Dosyalar

| Dosya | İşi |
|---|---|
| `lib/services/backup_crypto.dart` | Şifreleme biçimi (saf) |
| `lib/services/b2_client.dart` | B2 REST istemcisi |
| `lib/services/backup_info.dart` | Derleme zamanı ayarları, dosya adları |
| `lib/services/uploader.dart` | Kuyruk: şifrele → yükle → işaretle |
| `tool/anahtar_uret.py` | Anahtar çifti üretir |
| `tool/yedek_coz.py` | PC'de çözer |
| `tool/yedek_interop.sh` | Dart ↔ Python biçim doğrulaması |

## Maliyet

B2'de ilk 10 GB ücretsiz, sonrası ~$7/TB/ay. Kayıtlar 64 kbps AAC, yani
saatte ~29 MB: 10 GB ≈ 350 saat. Aşarsanız aylık kuruşlar konuşulur.

Sert bir harcama sınırı koymayın: sınır dolduğunda yüklemeler
`storage_cap_exceeded` ile **sessizce** başarısız olur ve arşiv siz fark
etmeden büyümeyi bırakır. Bunun yerine %75 uyarısı kurun.
