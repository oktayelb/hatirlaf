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
| `.env` | Keystore parolaları, X25519 **gizli** anahtar yolu, B2 master anahtarı, indirme dizini | **Hayır.** Yalnızca bu PC'de |
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
cp .env.ornek .env
$EDITOR .env    # B2_MASTER_API_KEYID + B2_MASTER_API_KEY, ANDROID_*, YEDEK_GIZLI_ANAHTAR

tool/b2_kur.py --deneme    # ne yapacagini yazar, hicbir sey olusturmaz
tool/b2_kur.py             # gercekten kurar
```

`tool/b2_kur.py` master anahtarla bağlanıp sırayla: X25519 çiftini
üretir (yoksa), kovayı `allPrivate` olarak açar, **yalnızca
`writeFiles`** yetkili ve o kovaya kısıtlı yeni bir uygulama anahtarı
üretir, `yedek.json`'u yazar, sonra o anahtarla **yazmayı dener
(geçmeli) ve okumayı dener (geçmemeli)**.

Son adım önemli: anahtarın okuyamadığını varsaymak yetmez, kanıtlamak
gerekir. Okuma geçerse script hata verip durur.

Eski `hatirlaf-yaz-*` anahtarlarını **silmez** — sahadaki eski APK'lar
hâlâ onları kullanıyor olabilir.

Anahtar çiftini tek başına üretmek isterseniz `tool/anahtar_uret.py`
hâlâ duruyor.

`yedek.json`'daki dört alan dolu değilse yedekleme tamamen kapalıdır ve
uygulama eskisi gibi çalışır. `tool/yayinla.sh` dosya varsa
`--dart-define-from-file` ile derler, yoksa uyarıp yedeklemesiz sürüm
üretir.

`android/key.properties` artık yok; imza parolaları `.env`'e taşındı.
`android/app/build.gradle.kts` önce `.env`'e, sonra gerçek ortam
değişkenlerine bakar.

## Yayından önce

```bash
tool/yedek_onkontrol.sh          # hızlı
tool/yedek_onkontrol.sh --tam    # 3 MB'lık gerçek gidiş-dönüş de yapar
```

Dört adım: dosyalar yerinde mi, sırlar depoya sızmış mı, **ayar adları
uygulamaya ulaşıyor mu**, B2 sağlıklı mı.

Üçüncüsü en sinsisi: `yedek.json`'daki bir ad uygulamanın beklediğiyle
tutmazsa yedekleme **kapalı** derlenir. Hata yok, uyarı yok, uygulama
normal çalışır — sadece hiçbir kayıt yükselmez, ve bunu aylar sonra
kovaya bakınca fark edersiniz. `tool/yedek_ayar_dogrula.dart` aynı
`--define` değerleriyle `YedekAyarlari`'nı gerçekten okur.

`tool/b2_saglik.py` tek başına da çalışır: anahtarın hâlâ yalnızca
`writeFiles` olduğunu, okumanın reddedildiğini, kovanın `allPrivate`
kaldığını, gizli anahtarın yerinde durduğunu ve kovanın ne kadar
dolduğunu söyler.

## Telefona dokunmadan denemek

```bash
tool/b2_deneme.sh          # 3 MiB
tool/b2_deneme.sh 30000000 # ~30 MB, bir saatlik kayıt kadar
```

Üretimdeki kodun ta kendisiyle (benzetim değil): Dart şifreler ve
uygulamanın kısıtlı anahtarıyla B2'ye yükler, Python master anahtarla
indirip çözer ve kaynakla bayt bayt karşılaştırır. Deneme nesnesi
sonunda silinir.

Şifrelemenin çalıştığını görmek yetmez; **geri dönebildiğini** görmek
gerekir. Bu script onu gösterir.

## Kayıtları alma

```bash
tool/yedek_indir.py --liste        # ne var, ne kadar yer tutuyor
tool/yedek_indir.py                # indir + çöz
tool/yedek_indir.py --sadece-indir # şifreli bırak
```

İnen dosyalar **deponun dışına**, `.env`'deki `YEDEK_INDIRME_DIZINI`
altına gider (öntanımlı `~/hatirlaf-yedekler`, `0700`):

```
~/hatirlaf-yedekler/
  sifreli/<cihaz>/<hatıra>/ses.m4a.hyz     B2'den geldiği gibi
  cozulmus/<cihaz>/<hatıra>/ses.m4a        çözülmüş
                            bilgi.json
```

Şifreli kopyalar **saklanır**: çözme hatasında ya da anahtar değişiminde
tekrar denenebilsin diye. B2'den hiçbir şey silinmez, ve ikinci kez
çalıştırmak var olanı yeniden indirmez.

### Neden depo dışına

Çözülmüş bir kayıt birinin hayat hikâyesi; depo ise herkese açık. Repo
içine inseydi tek bir `git add -A` hepsini yayımlardı — `*.hyz`
`.gitignore`'da ama çözülmüş `ses.m4a` ve `bilgi.json` değildi.

Üç katman:

1. `tool/yedek_indir.py` depo içine yazmayı **reddeder**.
2. B2'den gelen adlar `..` içerse bile hedef klasörün dışına çıkamaz —
   adlar telefondan geliyor, indirme tarafı onlara güvenmemeli.
3. `.gitignore` yine de `/gelenler/`, `/yedekler/`, `/indirilenler/`,
   `/hatirlaf-yedekler/`, `*.m4a` ve `*.wav` yakalar; `b2 sync` ya da
   rclone kullanırsanız bu ağ devrede olsun.

Tek tek çözmek için `tool/yedek_coz.py` hâlâ duruyor.

Her hatıra iki dosyadır: `<cihaz>/<hatıraId>/ses.m4a.hyz` ve
`bilgi.json.hyz` (başlık, tarih, süre, soru, **metin dökümü**). Metin de
şifrelenir: aranabilir olduğu için sesten daha hassas.

### Kimin kaydı hangisi

Kovadaki klasör adı `<ad>-<kimlik>` (ör. `Dedem Ahmet-05e24e66`). Ad
telefonu teslim ederken **Ayarlar → Aile Yedeği → Telefonu Adlandır**
ile bir kez girilir; rastgele kimlik yanında kalır çünkü aynı adı
taşıyan iki telefon birbirinin üzerine yazardı.

Ad girilmemişse klasör yalnız kimliktir. Sonradan eşlemek için indirme
dizinine `cihazlar.json` koyun:

```json
{ "05e24e66": "Dedem Ahmet" }
```

İsim çözme sırası: `cihazlar.json` → telefonda girilen ad → ham kimlik.
Eşleme telefondakini ezer, yani adı burada da düzeltebilirsiniz.

Çözülmüş kayıtlar kişiye göre klasörlenir (`cozulmus/<kişi>/...`). Adı
sonradan değiştirirseniz eski klasör yerinde kalır; `--yeniden-coz`
`cozulmus/`'u silip baştan kurar — `sifreli/` durduğu için veri kaybı
değildir.

**Uygulama verisi silinirse** telefon yeni bir rastgele kimlik üretir ve
o kişinin kayıtları iki klasöre bölünür. Ad aynı kaldığı için hangisinin
kim olduğu yine bellidir.

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
| `tool/b2_kur.py` | B2'yi kurar, `yedek.json`'u yazar, yetkiyi doğrular |
| `tool/b2_deneme.sh` | Gerçek B2'ye karşı uçtan uca deneme |
| `tool/yedek_indir.py` | B2'den indirir ve çözer (depo dışına) |
| `tool/yedek_onkontrol.sh` | Yayından önce her şeyi denetler |
| `tool/b2_saglik.py` | B2 tarafının sağlık kontrolü |
| `tool/anahtar_uret.py` | Anahtar çifti üretir (tek başına) |
| `tool/yedek_coz.py` | PC'de çözer |
| `tool/yedek_interop.sh` | Dart ↔ Python biçim doğrulaması |

## Derleme kaynak sınırları

`tool/flutter.sh` derlemeyi CPU ve bellek tavanı altında çalıştırır:
12 çekirdekli makinede varsayılan **%400 (4 çekirdek)**, `CPUWeight=20`
(çekişmede kullanıcının uygulamaları kazanır) ve `nice -n 10`.

```bash
FLUTTER_CPU=200 tool/yayinla.sh feature "..."   # daha da yavas, daha sakin
```

Gradle daemon systemd kapsamının dışına kaçabildiği için ikinci bir
savunma var: `android/gradle.properties` içindeki
`-XX:ActiveProcessorCount=4` JVM'e daha az çekirdeği varmış gibi
gösterir, böylece GC/JIT/worker havuzları küçülür. `flutter.sh` derleme
öncesi eski daemon'ı da kapatır ki yenisi sınırların içinde doğsun.

Amaç hız değil: derleme yavaşlasa da makine kullanılabilir kalmalı.

## Maliyet

B2'de ilk 10 GB ücretsiz, sonrası ~$7/TB/ay. Kayıtlar 64 kbps AAC, yani
saatte ~29 MB: 10 GB ≈ 350 saat. Aşarsanız aylık kuruşlar konuşulur.

Sert bir harcama sınırı koymayın: sınır dolduğunda yüklemeler
`storage_cap_exceeded` ile **sessizce** başarısız olur ve arşiv siz fark
etmeden büyümeyi bırakır. Bunun yerine %75 uyarısı kurun.
