import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/yama.dart';

/// Testte yama uretmek icin kucuk bir yazici. `tool/yama_uret.py` ile ayni
/// bicimi yaziyor; bozuk yamalari elde kurmak icin de gerekiyor.
class _Yazici {
  final BytesBuilder _komutlar = BytesBuilder();
  final BytesBuilder _govde = BytesBuilder();

  void kopyala(int ofset, int uzunluk) {
    _komutlar.addByte(1);
    _varint(ofset);
    _varint(uzunluk);
  }

  void yeni(List<int> baytlar) {
    _komutlar.addByte(2);
    _varint(baytlar.length);
    _govde.add(baytlar);
  }

  /// Bilinmeyen komut uretmek icin.
  void hamKomut(int kod) => _komutlar.addByte(kod);

  void _varint(int n) {
    while (true) {
      final int p = n & 0x7F;
      n >>= 7;
      _komutlar.addByte(p | (n != 0 ? 0x80 : 0));
      if (n == 0) return;
    }
  }

  Uint8List yaz({
    required int hedefUzunluk,
    required List<int> kaynakOzet,
    String sihir = 'HTRLYAMA1',
  }) {
    final Uint8List komutlar = _komutlar.toBytes();
    final Uint8List govde = _govde.toBytes();
    final BytesBuilder b = BytesBuilder()
      ..add(utf8.encode(sihir))
      ..add(_u64(hedefUzunluk))
      ..add(kaynakOzet)
      ..add(_u32(komutlar.length))
      ..add(komutlar)
      ..add(_u64(govde.length))
      ..add(zlib.encode(govde));
    return b.toBytes();
  }

  static Uint8List _u32(int n) =>
      Uint8List(4)..buffer.asByteData().setUint32(0, n, Endian.little);

  static Uint8List _u64(int n) =>
      Uint8List(8)..buffer.asByteData().setUint64(0, n, Endian.little);
}

void main() {
  late Directory gecici;
  late File kaynak;
  late File yama;
  late File hedef;

  /// 4 KB'lik taninabilir bir "eski dosya".
  final Uint8List eskiVeri = Uint8List.fromList(
    List<int>.generate(4096, (int i) => (i * 7) & 0xFF),
  );

  setUp(() async {
    gecici = await Directory.systemTemp.createTemp('yama-testi');
    kaynak = File('${gecici.path}/eski.bin');
    yama = File('${gecici.path}/fark.yama');
    hedef = File('${gecici.path}/yeni.bin');
    await kaynak.writeAsBytes(eskiVeri);
  });

  tearDown(() async {
    if (gecici.existsSync()) await gecici.delete(recursive: true);
  });

  List<int> ozetiniAl(List<int> veri) => sha256.convert(veri).bytes;

  group('Yama.uygula', () {
    test('kopyala ve yeni komutlarini birlestirir', () async {
      final List<int> eklenen = utf8.encode('yeni baytlar');
      final _Yazici y = _Yazici()
        ..kopyala(0, 1000)
        ..yeni(eklenen)
        ..kopyala(2000, 96);

      await yama.writeAsBytes(
        y.yaz(
          hedefUzunluk: 1000 + eklenen.length + 96,
          kaynakOzet: ozetiniAl(eskiVeri),
        ),
      );

      await Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef);

      final Uint8List cikan = await hedef.readAsBytes();
      expect(
        cikan,
        <int>[
          ...eskiVeri.sublist(0, 1000),
          ...eklenen,
          ...eskiVeri.sublist(2000, 2096),
        ],
      );
    });

    // Telefondaki APK baska bir dosyaysa uretilecek sey copten ibaret olur.
    test('kaynak beklenenden farkliysa reddeder', () async {
      final _Yazici y = _Yazici()..kopyala(0, 4096);
      await yama.writeAsBytes(
        y.yaz(
          hedefUzunluk: 4096,
          kaynakOzet: ozetiniAl(<int>[1, 2, 3]),
        ),
      );

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
      expect(hedef.existsSync(), isFalse);
    });

    test('taninmayan bicim reddedilir', () async {
      final _Yazici y = _Yazici()..kopyala(0, 4096);
      await yama.writeAsBytes(
        y.yaz(
          hedefUzunluk: 4096,
          kaynakOzet: ozetiniAl(eskiVeri),
          sihir: 'BASKASI!!',
        ),
      );

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
    });

    test('kaynagin disini gosteren kopya reddedilir', () async {
      final _Yazici y = _Yazici()..kopyala(4000, 1000);
      await yama.writeAsBytes(
        y.yaz(hedefUzunluk: 1000, kaynakOzet: ozetiniAl(eskiVeri)),
      );

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
      // Yarim kalan cikti birakilmamali: sonraki acilista "hazir" sanilir.
      expect(hedef.existsSync(), isFalse);
    });

    test('bilinmeyen komut reddedilir', () async {
      final _Yazici y = _Yazici()
        ..kopyala(0, 10)
        ..hamKomut(9);
      await yama.writeAsBytes(
        y.yaz(hedefUzunluk: 10, kaynakOzet: ozetiniAl(eskiVeri)),
      );

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
    });

    test('beklenenden az bayt ureten yama reddedilir', () async {
      final _Yazici y = _Yazici()..kopyala(0, 100);
      await yama.writeAsBytes(
        y.yaz(hedefUzunluk: 4096, kaynakOzet: ozetiniAl(eskiVeri)),
      );

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
      expect(hedef.existsSync(), isFalse);
    });

    test('beklenenden cok bayt ureten yama reddedilir', () async {
      final _Yazici y = _Yazici()..kopyala(0, 4096);
      await yama.writeAsBytes(
        y.yaz(hedefUzunluk: 100, kaynakOzet: ozetiniAl(eskiVeri)),
      );

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
    });

    test('kirpilmis yama reddedilir', () async {
      final _Yazici y = _Yazici()..kopyala(0, 4096);
      final Uint8List tam =
          y.yaz(hedefUzunluk: 4096, kaynakOzet: ozetiniAl(eskiVeri));
      await yama.writeAsBytes(tam.sublist(0, tam.length ~/ 2));

      await expectLater(
        Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef),
        throwsA(isA<FormatException>()),
      );
    });
  });

  // Yamayi ureten Python ile uygulayan Dart ayri dosyalarda; bicim
  // birbirinden ayrilirsa yayinlanan yama telefonda ise yaramaz.
  group('tool/yama_uret.py ile uctan uca', () {
    test('uretilen yama ayni dosyayi geri verir', () async {
      final File eskiZip = File('${gecici.path}/eski.zip');
      final File yeniZip = File('${gecici.path}/yeni.zip');

      // Iki "APK": buyuk ortak bir girdi (kopyalanacak) ve degisen kucuk
      // bir girdi (govdeye girecek).
      final ProcessResult kur = await Process.run('python3', <String>[
        '-c',
        '''
import sys, zipfile
ortak = bytes(i % 251 for i in range(200000))
for yol, metin in ((sys.argv[1], b"eski"), (sys.argv[2], b"yeni metin")):
    with zipfile.ZipFile(yol, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("buyuk.bin", ortak)
        z.writestr("kucuk.txt", metin * 40)
''',
        eskiZip.path,
        yeniZip.path,
      ]);
      expect(kur.exitCode, 0, reason: kur.stderr.toString());

      final ProcessResult uret = await Process.run('python3', <String>[
        'tool/yama_uret.py',
        eskiZip.path,
        yeniZip.path,
        yama.path,
      ]);
      expect(uret.exitCode, 0, reason: uret.stderr.toString());

      // Ortak girdi kopyalanmali: yama iki dosyanin toplamindan cok kucuk.
      expect(await yama.length(), lessThan(await yeniZip.length() ~/ 2));

      await Yama.uygula(kaynak: eskiZip, yama: yama, hedef: hedef);
      expect(await hedef.readAsBytes(), await yeniZip.readAsBytes());
    });
  });
}
