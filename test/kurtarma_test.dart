import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/models/memory.dart';
import 'package:hatirla/services/kurtarma.dart';

import 'adts_ornek.dart';

void main() {
  late Directory hatiralar;

  setUp(() async {
    hatiralar = await Directory.systemTemp.createTemp('kurtarma-testi');
  });

  tearDown(() async {
    if (hatiralar.existsSync()) await hatiralar.delete(recursive: true);
  });

  /// Yarida kesilmis bir kaydin diskte biraktigi hali kurar.
  Future<Directory> kayitKlasoru(
    String id, {
    Map<String, dynamic>? isaret,
    Uint8List? ses,
    String sesAdi = 'ses.aac',
    bool isaretYaz = true,
  }) async {
    final Directory d = Directory('${hatiralar.path}/$id');
    d.createSync(recursive: true);
    if (ses != null) {
      await File('${d.path}/$sesAdi').writeAsBytes(ses);
    }
    if (isaretYaz) {
      await File('${d.path}/${Kurtarma.isaretAdi}').writeAsString(
        json.encode(<String, dynamic>{
          'id': id,
          'baslik': 'Çocukluğum',
          'soru': 'Çocukluğunuz nerede geçti?',
          'ses': sesAdi,
          'baslangic': '2026-03-04T12:00:00.000',
          ...?isaret,
        }),
      );
    }
    return d;
  }

  File isaretDosyasi(Directory d) => File('${d.path}/${Kurtarma.isaretAdi}');

  group('yarida kesilmis kayit', () {
    test('kurtarilir ve isaret silinir', () async {
      final Directory d = await kayitKlasoru('abc', ses: adtsAkisi(200));

      final List<Memory> sonuc =
          await Kurtarma.klasorleriTara(hatiralar, <String>{});

      expect(sonuc, hasLength(1));
      final Memory m = sonuc.single;
      expect(m.id, 'abc');
      expect(m.title, 'Çocukluğum');
      expect(m.question, 'Çocukluğunuz nerede geçti?');
      expect(m.audioRelPath, 'hatiralar/abc/ses.aac');
      expect(m.createdAt, DateTime.parse('2026-03-04T12:00:00.000'));
      expect(m.status, TranscriptStatus.bekliyor);
      // 200 cerceve * 1024 ornek / 44100
      expect(m.durationMs, (200 * 1024 * 1000) ~/ 44100);

      expect(isaretDosyasi(d).existsSync(), isFalse);
      expect(File('${d.path}/ses.aac').existsSync(), isTrue);
    });

    // Surec cercevenin ortasinda oldurulmus olabilir; dosya bastan sona
    // gecerli kalsin diye kalinti kirpiliyor.
    test('sondaki yarim cerceve kirpilir', () async {
      final Uint8List tam = adtsAkisi(50);
      final Directory d = await kayitKlasoru(
        'yarim',
        ses: Uint8List.fromList(<int>[...tam.sublist(0, tam.length - 80)]),
      );

      final List<Memory> sonuc =
          await Kurtarma.klasorleriTara(hatiralar, <String>{});

      expect(sonuc, hasLength(1));
      expect(await File('${d.path}/ses.aac').length(), 49 * 200);
      expect(sonuc.single.durationMs, (49 * 1024 * 1000) ~/ 44100);
    });

    test('okunamayan ses klasoruyle birlikte atilir', () async {
      final Directory d = await kayitKlasoru(
        'bozuk',
        // Yarim kalmis bir m4a: acilmaz, kurtarilacak bir sey yok.
        ses: Uint8List.fromList(List<int>.generate(8192, (int i) => i & 0xFF)),
        sesAdi: 'ses.m4a',
      );

      final List<Memory> sonuc =
          await Kurtarma.klasorleriTara(hatiralar, <String>{});

      expect(sonuc, isEmpty);
      expect(d.existsSync(), isFalse);
    });

    test('ses dosyasi yoksa klasor silinir', () async {
      final Directory d = await kayitKlasoru('sessiz');

      expect(await Kurtarma.klasorleriTara(hatiralar, <String>{}), isEmpty);
      expect(d.existsSync(), isFalse);
    });

    test('duyulmayacak kadar kisa kayit silinir', () async {
      final Directory d = await kayitKlasoru('kisa', ses: adtsAkisi(2));

      expect(await Kurtarma.klasorleriTara(hatiralar, <String>{}), isEmpty);
      expect(d.existsSync(), isFalse);
    });
  });

  group('kayit bitmisti', () {
    // Ses tamam, yalnizca dizine yazilamadan kapanilmis: bicim ne olursa
    // olsun kurtarilir ve sure isaretten gelir.
    test('bitti isaretli kayit bicimine bakilmadan kurtarilir', () async {
      await kayitKlasoru(
        'tamam',
        sesAdi: 'ses.m4a',
        ses: Uint8List.fromList(List<int>.generate(4096, (int i) => i & 0xFF)),
        isaret: <String, dynamic>{'bitti': true, 'sureMs': 123456},
      );

      final List<Memory> sonuc =
          await Kurtarma.klasorleriTara(hatiralar, <String>{});

      expect(sonuc, hasLength(1));
      expect(sonuc.single.durationMs, 123456);
      expect(sonuc.single.audioRelPath, 'hatiralar/tamam/ses.m4a');
    });
  });

  group('isaret bayat', () {
    test('hatira zaten dizindeyse yalnizca isaret silinir', () async {
      final Directory d = await kayitKlasoru('var', ses: adtsAkisi(100));

      final List<Memory> sonuc =
          await Kurtarma.klasorleriTara(hatiralar, <String>{'var'});

      expect(sonuc, isEmpty);
      expect(isaretDosyasi(d).existsSync(), isFalse);
      // Hatira duruyor: sesi silmek olmaz.
      expect(File('${d.path}/ses.aac').existsSync(), isTrue);
    });

    test('bozuk isaret klasoru ucurmaz, kendini siler', () async {
      final Directory d = await kayitKlasoru('bozukisaret', ses: adtsAkisi(100));
      await isaretDosyasi(d).writeAsString('{bu json degil');

      expect(await Kurtarma.klasorleriTara(hatiralar, <String>{}), isEmpty);
      expect(isaretDosyasi(d).existsSync(), isFalse);
      expect(File('${d.path}/ses.aac').existsSync(), isTrue);
    });
  });

  group('dokunulmayanlar', () {
    test('isaretsiz klasorlere karisilmaz', () async {
      final Directory d =
          await kayitKlasoru('duzgun', ses: adtsAkisi(100), isaretYaz: false);

      expect(await Kurtarma.klasorleriTara(hatiralar, <String>{}), isEmpty);
      expect(d.existsSync(), isTrue);
      expect(File('${d.path}/ses.aac').existsSync(), isTrue);
    });

    test('olmayan dizin sorun cikarmaz', () async {
      final Directory yok = Directory('${hatiralar.path}/yok');
      expect(await Kurtarma.klasorleriTara(yok, <String>{}), isEmpty);
    });

    test('birden fazla yarim kayit birlikte kurtarilir', () async {
      await kayitKlasoru('bir', ses: adtsAkisi(100));
      await kayitKlasoru('iki', ses: adtsAkisi(150));

      final List<Memory> sonuc =
          await Kurtarma.klasorleriTara(hatiralar, <String>{});
      expect(sonuc.map((Memory m) => m.id).toSet(), <String>{'bir', 'iki'});
    });
  });
}
