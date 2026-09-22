import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cryptography/cryptography.dart';
import 'package:crypto/crypto.dart' as ozet;

/// Yedeklerin telefonda sifrelenmesi.
///
/// Asimetrik, bilerek: APK'ya yalnizca ACIK anahtar konur. Telefon
/// sifreler, yalnizca gelistiricinin PC'sindeki GIZLI anahtar cozer.
/// APK'yi parcalayan biri kimsenin kaydini okuyamaz -- simetrik bir
/// anahtar konsaydi tek bir APK butun aileyi acardi.
///
/// Gizli anahtar kaybolursa yedekler bir daha ACILAMAZ. Bu yuzden
/// telefondaki asil kayit asla silinmez: son cikis kapisi odur.
///
/// Bicim (sayilar big-endian):
///
///     "HTRLF1"   6    sihirli sayi
///     surum      1    0x01
///     parcaBoyu  4    duz metin parca boyu
///     aliciIz    8    sha256(aliciAcikAnahtari)[0..8]
///     gecici     32   bu dosyaya ozel X25519 acik anahtari
///     ----------------  (toplam 51 bayt baslik)
///     her parca icin: sifreliMetin + 16 bayt etiket
///
/// Son parca disindaki her parca tam olarak `parcaBoyu + 16` bayttir;
/// son parca kalanı kadardir. Bos dosya bile bir parca uretir, yoksa
/// "sifir parca" ile "budanmis dosya" ayirt edilemezdi.
class YedekSifreleme {
  const YedekSifreleme._();

  static const List<int> sihirliSayi = <int>[0x48, 0x54, 0x52, 0x4C, 0x46, 0x31];
  static const int surum = 1;

  /// 1 MiB: telefonda bellek derdi yok, parca basina 16 bayt etiket
  /// yuku de binde birin altinda kaliyor.
  static const int varsayilanParcaBoyu = 1 << 20;

  static const int etiketBoyu = 16;
  static const int acikAnahtarBoyu = 32;
  static const int izBoyu = 8;
  static const int basliktakiBayt =
      6 + 1 + 4 + izBoyu + acikAnahtarBoyu; // 51

  /// HKDF'in `info` alani. Degistirilirse eski yedekler acilamaz.
  static final List<int> _baglam = utf8.encode('hatirlaf-yedek-v1');

  static final X25519 _x25519 = X25519();
  static final AesGcm _aes = AesGcm.with256bits();

  static bool _esit(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Alicinin acik anahtarindan kisa, insan okuyabilir bir iz.
  static Uint8List parmakIzi(List<int> acikAnahtar) =>
      Uint8List.fromList(ozet.sha256.convert(acikAnahtar).bytes.sublist(0, izBoyu));

  /// Bu dosyaya ozel anahtari turetir.
  ///
  /// Tuzun icine iki acik anahtar da giriyor: ayni paylasilan sirdan
  /// baska bir baglamda ayni anahtar cikmasin.
  ///
  /// [geciciAcik] ve [aliciAcik] acikca veriliyor cunku tuzun SIRASI
  /// onemli: sifrelerken elimizde gecici gizli anahtar, cozerken alici
  /// gizli anahtari var. Sirayi cagiranin anahtarindan cikarsaydik iki
  /// taraf tuzu ters sirada kurar ve hicbir dosya acilmazdi.
  static Future<SecretKey> _dosyaAnahtari({
    required KeyPair kendiAnahtari,
    required SimplePublicKey karsiAcik,
    required List<int> geciciAcik,
    required List<int> aliciAcik,
  }) async {
    final SecretKey paylasilan = await _x25519.sharedSecretKey(
      keyPair: kendiAnahtari,
      remotePublicKey: karsiAcik,
    );
    return Hkdf(hmac: Hmac.sha256(), outputLength: 32).deriveKey(
      secretKey: paylasilan,
      nonce: <int>[...geciciAcik, ...aliciAcik],
      info: _baglam,
    );
  }

  static Uint8List _baslik({
    required int parcaBoyu,
    required List<int> aliciIz,
    required List<int> geciciAcik,
  }) {
    final BytesBuilder b = BytesBuilder();
    b.add(sihirliSayi);
    b.addByte(surum);
    b.add(_dort(parcaBoyu));
    b.add(aliciIz);
    b.add(geciciAcik);
    final Uint8List sonuc = b.toBytes();
    assert(sonuc.length == basliktakiBayt);
    return sonuc;
  }

  static Uint8List _dort(int deger) {
    final ByteData d = ByteData(4);
    d.setUint32(0, deger);
    return d.buffer.asUint8List();
  }

  /// Parca sirasindan nonce. Dosya anahtari her dosyada farkli oldugu
  /// icin sayac guvenli: ayni (anahtar, nonce) cifti iki kez olusmaz.
  static Uint8List _nonce(int sira) {
    final Uint8List n = Uint8List(12);
    ByteData.view(n.buffer).setUint32(8, sira);
    return n;
  }

  /// Parcayi baslikla ve sirasiyla baglar: baslik kurcalanirsa,
  /// parcalar yer degistirirse ya da dosya budanirsa cozme basarisiz olur.
  static Uint8List _ek(Uint8List baslik, int sira, bool sonMu) {
    final BytesBuilder b = BytesBuilder();
    b.add(baslik);
    b.add(_dort(sira));
    b.addByte(sonMu ? 1 : 0);
    return b.toBytes();
  }

  /// [kaynak] dosyasini sifreleyip [hedef] dosyasina yazar.
  ///
  /// [geciciAnahtar] yalnizca testler icindir: verilirse cikti
  /// belirlenimci olur. Uretimde her zaman `null` birakilmali, yoksa
  /// nonce'lar tekrar eder ve AES-GCM'in guvenligi coker.
  static Future<void> dosyayiSifrele({
    required File kaynak,
    required File hedef,
    required List<int> aliciAcikAnahtari,
    int parcaBoyu = varsayilanParcaBoyu,
    SimpleKeyPair? geciciAnahtar,
  }) async {
    if (aliciAcikAnahtari.length != acikAnahtarBoyu) {
      throw ArgumentError('Alici acik anahtari $acikAnahtarBoyu bayt olmali.');
    }
    if (parcaBoyu <= 0) {
      throw ArgumentError('Parca boyu pozitif olmali.');
    }

    final SimplePublicKey alici = SimplePublicKey(
      List<int>.unmodifiable(aliciAcikAnahtari),
      type: KeyPairType.x25519,
    );
    final SimpleKeyPair gecici = geciciAnahtar ?? await _x25519.newKeyPair();
    final SimplePublicKey geciciAcik = await gecici.extractPublicKey();
    final SecretKey anahtar = await _dosyaAnahtari(
      kendiAnahtari: gecici,
      karsiAcik: alici,
      geciciAcik: geciciAcik.bytes,
      aliciAcik: aliciAcikAnahtari,
    );

    final Uint8List baslik = _baslik(
      parcaBoyu: parcaBoyu,
      aliciIz: parmakIzi(aliciAcikAnahtari),
      geciciAcik: geciciAcik.bytes,
    );

    final RandomAccessFile girdi = await kaynak.open();
    final IOSink cikti = hedef.openWrite();
    try {
      cikti.add(baslik);
      final int toplam = await kaynak.length();
      int okunan = 0;
      int sira = 0;

      // Bos dosya da bir parca uretsin: dongu en az bir kez donmeli.
      do {
        final Uint8List duz = await girdi.read(parcaBoyu);
        if (duz.isEmpty && okunan < toplam) {
          // Dosya sifreleme sirasinda kisaldi. Devam etseydik dongu
          // hic bitmezdi; yarim bir yedek yazmaktansa durmak dogru.
          throw const FileSystemException('Kaynak dosya sifreleme sirasinda degisti');
        }
        okunan += duz.length;
        final bool sonMu = okunan >= toplam;
        final SecretBox kutu = await _aes.encrypt(
          duz,
          secretKey: anahtar,
          nonce: _nonce(sira),
          aad: _ek(baslik, sira, sonMu),
        );
        cikti.add(kutu.cipherText);
        cikti.add(kutu.mac.bytes);
        sira++;
        if (sonMu) break;
      } while (true);

      await cikti.flush();
    } finally {
      await girdi.close();
      await cikti.close();
    }
  }

  /// Sifrelenmis dosyanin boyutunu onceden hesaplar.
  ///
  /// B2'ye yuklemeden once `Content-Length` gerektigi icin lazim.
  static int sifreliBoyut(int duzBoyut, {int parcaBoyu = varsayilanParcaBoyu}) {
    final int parcaSayisi =
        duzBoyut == 0 ? 1 : (duzBoyut + parcaBoyu - 1) ~/ parcaBoyu;
    return basliktakiBayt + duzBoyut + parcaSayisi * etiketBoyu;
  }

  /// Yalnizca testler ve yerel dogrulama icin: PC'deki cozucunun
  /// karsiligi. Uretimde telefon hicbir seyi cozmez.
  static Future<void> dosyayiCoz({
    required File kaynak,
    required File hedef,
    required List<int> aliciGizliAnahtari,
  }) async {
    final Uint8List ham = await kaynak.readAsBytes();
    if (ham.length < basliktakiBayt) {
      throw const FormatException('Dosya baslik icin bile kisa.');
    }
    for (int i = 0; i < sihirliSayi.length; i++) {
      if (ham[i] != sihirliSayi[i]) {
        throw const FormatException('Bu bir hatirlaf yedegi degil.');
      }
    }
    if (ham[6] != surum) {
      throw FormatException('Bilinmeyen yedek surumu: ${ham[6]}');
    }

    final Uint8List baslik = ham.sublist(0, basliktakiBayt);
    final int parcaBoyu = ByteData.view(baslik.buffer).getUint32(7);
    final Uint8List geciciAcik = baslik.sublist(
      6 + 1 + 4 + izBoyu,
      basliktakiBayt,
    );

    final SimpleKeyPair alici = await _x25519.newKeyPairFromSeed(
      List<int>.from(aliciGizliAnahtari),
    );
    final SimplePublicKey aliciAcik = await alici.extractPublicKey();
    if (!_esit(parmakIzi(aliciAcik.bytes), baslik.sublist(11, 11 + izBoyu))) {
      throw const FormatException(
        'Bu yedek baska bir anahtara sifrelenmis.',
      );
    }
    final SecretKey anahtar = await _dosyaAnahtari(
      kendiAnahtari: alici,
      karsiAcik: SimplePublicKey(geciciAcik, type: KeyPairType.x25519),
      geciciAcik: geciciAcik,
      aliciAcik: aliciAcik.bytes,
    );

    final IOSink cikti = hedef.openWrite();
    try {
      int konum = basliktakiBayt;
      int sira = 0;
      final int tamParcaBoyu = parcaBoyu + etiketBoyu;
      while (true) {
        final int kalan = ham.length - konum;
        if (kalan < etiketBoyu) {
          throw const FormatException('Yedek budanmis.');
        }
        final bool sonMu = kalan <= tamParcaBoyu;
        final int buParca = sonMu ? kalan : tamParcaBoyu;
        final Uint8List govde =
            ham.sublist(konum, konum + buParca - etiketBoyu);
        final Uint8List etiket =
            ham.sublist(konum + buParca - etiketBoyu, konum + buParca);

        final List<int> duz = await _aes.decrypt(
          SecretBox(govde, nonce: _nonce(sira), mac: Mac(etiket)),
          secretKey: anahtar,
          aad: _ek(baslik, sira, sonMu),
        );
        cikti.add(duz);

        konum += buParca;
        sira++;
        if (sonMu) break;
      }
      await cikti.flush();
    } finally {
      await cikti.close();
    }
  }
}
