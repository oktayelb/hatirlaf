import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:cryptography/cryptography.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/backup_crypto.dart';

/// Testlerin belirlenimci olmasi icin sabit tohumlar. Uretimde anahtarlar
/// her zaman rastgeledir.
final List<int> _aliciTohum = List<int>.filled(32, 7);
final List<int> _geciciTohum = List<int>.filled(32, 9);

late Directory _gecici;

Future<SimpleKeyPair> _aliciAnahtari() =>
    X25519().newKeyPairFromSeed(_aliciTohum);

Future<Uint8List> _aliciAcik() async {
  final SimplePublicKey p = await (await _aliciAnahtari()).extractPublicKey();
  return Uint8List.fromList(p.bytes);
}

File _dosya(String ad) => File('${_gecici.path}/$ad');

Future<File> _yaz(String ad, List<int> icerik) async {
  final File f = _dosya(ad);
  await f.writeAsBytes(icerik, flush: true);
  return f;
}

Uint8List _rastgele(int n) {
  final Random r = Random(42);
  return Uint8List.fromList(
    List<int>.generate(n, (_) => r.nextInt(256)),
  );
}

/// Sifrele, coz, geri geleni karsilastir.
Future<void> _gidisDonus(String ad, List<int> veri, {int? parcaBoyu}) async {
  final File duz = await _yaz('$ad.duz', veri);
  final File sifreli = _dosya('$ad.sifreli');
  final File cozulen = _dosya('$ad.cozulen');

  await YedekSifreleme.dosyayiSifrele(
    kaynak: duz,
    hedef: sifreli,
    aliciAcikAnahtari: await _aliciAcik(),
    parcaBoyu: parcaBoyu ?? YedekSifreleme.varsayilanParcaBoyu,
  );
  await YedekSifreleme.dosyayiCoz(
    kaynak: sifreli,
    hedef: cozulen,
    aliciGizliAnahtari: _aliciTohum,
  );

  expect(await cozulen.readAsBytes(), veri);
}

void main() {
  setUpAll(() {
    _gecici = Directory.systemTemp.createTempSync('hatirlaf_sifre_');
  });
  tearDownAll(() {
    if (_gecici.existsSync()) _gecici.deleteSync(recursive: true);
  });

  group('gidis-donus', () {
    test('kucuk dosya', () async {
      await _gidisDonus('kucuk', _rastgele(100));
    });

    test('bos dosya da bir parca uretir', () async {
      await _gidisDonus('bos', <int>[]);
      // Bos dosya: baslik + tek bos parcanin etiketi.
      final File s = _dosya('bos.sifreli');
      expect(
        await s.length(),
        YedekSifreleme.basliktakiBayt + YedekSifreleme.etiketBoyu,
      );
    });

    test('tam parca boyunda dosya', () async {
      await _gidisDonus('tam', _rastgele(256), parcaBoyu: 256);
    });

    test('parca sinirinin bir altinda', () async {
      await _gidisDonus('alt', _rastgele(255), parcaBoyu: 256);
    });

    test('parca sinirinin bir ustunde', () async {
      await _gidisDonus('ust', _rastgele(257), parcaBoyu: 256);
    });

    test('cok parcali dosya', () async {
      await _gidisDonus('cok', _rastgele(4096), parcaBoyu: 256);
    });
  });

  group('boyut hesabi', () {
    test('onceden hesaplanan boyut gercekle birebir tutuyor', () async {
      for (final int n in <int>[0, 1, 255, 256, 257, 1000]) {
        final File duz = await _yaz('boyut_$n.duz', _rastgele(n));
        final File sifreli = _dosya('boyut_$n.sifreli');
        await YedekSifreleme.dosyayiSifrele(
          kaynak: duz,
          hedef: sifreli,
          aliciAcikAnahtari: await _aliciAcik(),
          parcaBoyu: 256,
        );
        expect(
          await sifreli.length(),
          YedekSifreleme.sifreliBoyut(n, parcaBoyu: 256),
          reason: '$n baytlik dosya',
        );
      }
    });
  });

  group('kurcalama', () {
    late File sifreli;

    setUp(() async {
      final File duz = await _yaz('k.duz', _rastgele(700));
      sifreli = _dosya('k.sifreli');
      await YedekSifreleme.dosyayiSifrele(
        kaynak: duz,
        hedef: sifreli,
        aliciAcikAnahtari: await _aliciAcik(),
        parcaBoyu: 256,
      );
    });

    Future<void> beklenenHata(Uint8List bozuk) async {
      final File b = await _yaz('k.bozuk', bozuk);
      await expectLater(
        YedekSifreleme.dosyayiCoz(
          kaynak: b,
          hedef: _dosya('k.cikti'),
          aliciGizliAnahtari: _aliciTohum,
        ),
        throwsA(anything),
      );
    }

    test('govdede tek bit degisirse cozulmez', () async {
      final Uint8List ham = await sifreli.readAsBytes();
      ham[YedekSifreleme.basliktakiBayt + 10] ^= 1;
      await beklenenHata(ham);
    });

    test('baslik kurcalanirsa cozulmez', () async {
      final Uint8List ham = await sifreli.readAsBytes();
      // Gecici acik anahtarin son bayti.
      ham[YedekSifreleme.basliktakiBayt - 1] ^= 1;
      await beklenenHata(ham);
    });

    test('dosya budanirsa cozulmez', () async {
      final Uint8List ham = await sifreli.readAsBytes();
      // Son parcayi tamamen at: kalan parcalar kendi baslarina saglam,
      // ama sonuncusu "son" olarak isaretli degil.
      await beklenenHata(
        Uint8List.fromList(ham.sublist(0, 256 + YedekSifreleme.etiketBoyu * 1 + YedekSifreleme.basliktakiBayt)),
      );
    });

    test('parcalar yer degistirirse cozulmez', () async {
      final Uint8List ham = await sifreli.readAsBytes();
      const int tam = 256 + YedekSifreleme.etiketBoyu;
      final int b = YedekSifreleme.basliktakiBayt;
      final Uint8List birinci = ham.sublist(b, b + tam);
      final Uint8List ikinci = ham.sublist(b + tam, b + 2 * tam);
      ham.setRange(b, b + tam, ikinci);
      ham.setRange(b + tam, b + 2 * tam, birinci);
      await beklenenHata(ham);
    });

    test('sihirli sayi yanlissa anlasilir hata', () async {
      final Uint8List ham = await sifreli.readAsBytes();
      ham[0] = 0x00;
      await expectLater(
        YedekSifreleme.dosyayiCoz(
          kaynak: await _yaz('k.sihir', ham),
          hedef: _dosya('k.cikti2'),
          aliciGizliAnahtari: _aliciTohum,
        ),
        throwsA(isA<FormatException>()),
      );
    });

    test('baska bir gizli anahtar cozemez', () async {
      await expectLater(
        YedekSifreleme.dosyayiCoz(
          kaynak: sifreli,
          hedef: _dosya('k.cikti3'),
          aliciGizliAnahtari: List<int>.filled(32, 3),
        ),
        throwsA(anything),
      );
    });
  });

  group('bicim', () {
    test('baslik beklenen alanlari tasiyor', () async {
      final File duz = await _yaz('b.duz', _rastgele(10));
      final File sifreli = _dosya('b.sifreli');
      final Uint8List acik = await _aliciAcik();
      await YedekSifreleme.dosyayiSifrele(
        kaynak: duz,
        hedef: sifreli,
        aliciAcikAnahtari: acik,
        parcaBoyu: 4096,
        geciciAnahtar: await X25519().newKeyPairFromSeed(_geciciTohum),
      );

      final Uint8List ham = await sifreli.readAsBytes();
      expect(ham.sublist(0, 6), YedekSifreleme.sihirliSayi);
      expect(ham[6], YedekSifreleme.surum);
      expect(ByteData.view(ham.buffer).getUint32(7), 4096);
      expect(ham.sublist(11, 19), YedekSifreleme.parmakIzi(acik));

      final SimplePublicKey geciciAcik =
          await (await X25519().newKeyPairFromSeed(_geciciTohum))
              .extractPublicKey();
      expect(ham.sublist(19, 51), geciciAcik.bytes);
    });

    test('ayni girdi iki kez sifrelenince farkli cikti verir', () async {
      final File duz = await _yaz('t.duz', _rastgele(64));
      final Uint8List acik = await _aliciAcik();
      final List<Uint8List> ciktilar = <Uint8List>[];
      for (int i = 0; i < 2; i++) {
        final File s = _dosya('t$i.sifreli');
        await YedekSifreleme.dosyayiSifrele(
          kaynak: duz,
          hedef: s,
          aliciAcikAnahtari: acik,
        );
        ciktilar.add(await s.readAsBytes());
      }
      // Gecici anahtar her seferinde yeni: govdeler esit olmamali.
      expect(ciktilar[0], isNot(ciktilar[1]));
    });

    test('yanlis uzunlukta acik anahtar reddedilir', () async {
      final File duz = await _yaz('y.duz', <int>[1, 2, 3]);
      expect(
        () => YedekSifreleme.dosyayiSifrele(
          kaynak: duz,
          hedef: _dosya('y.sifreli'),
          aliciAcikAnahtari: List<int>.filled(31, 0),
        ),
        throwsA(isA<ArgumentError>()),
      );
    });
  });
}
