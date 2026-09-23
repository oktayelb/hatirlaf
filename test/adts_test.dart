import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/adts.dart';

import 'adts_ornek.dart';

void main() {
  late Directory gecici;
  late File ses;

  setUp(() async {
    gecici = await Directory.systemTemp.createTemp('adts-testi');
    ses = File('${gecici.path}/ses.aac');
  });

  tearDown(() async {
    if (gecici.existsSync()) await gecici.delete(recursive: true);
  });

  group('AdtsBilgi.oku', () {
    test('cerceveleri sayar ve sureyi hesaplar', () async {
      await ses.writeAsBytes(adtsAkisi(100));

      final AdtsBilgi? b = await AdtsBilgi.oku(ses);
      expect(b, isNotNull);
      expect(b!.cerceveSayisi, 100);
      expect(b.orneklemeHizi, 44100);
      expect(b.gecerliBayt, 100 * 200);
      // 100 * 1024 ornek / 44100 = 2.32 sn
      expect(b.sure.inMilliseconds, (100 * 1024 * 1000) ~/ 44100);
    });

    test('baska ornekleme hizini okur', () async {
      // Dizin 8 -> 16000
      await ses.writeAsBytes(adtsAkisi(10, hizDizini: 8));

      final AdtsBilgi? b = await AdtsBilgi.oku(ses);
      expect(b!.orneklemeHizi, 16000);
      expect(b.sure.inMilliseconds, (10 * 1024 * 1000) ~/ 16000);
    });

    // Asil mesele bu: surec oldurulurse dosya cercevenin ortasinda biter.
    test('sondaki yarim cerceve sayilmaz', () async {
      final Uint8List tam = adtsAkisi(50);
      // Son cercevenin yarisini kes.
      await ses.writeAsBytes(tam.sublist(0, tam.length - 100));

      final AdtsBilgi? b = await AdtsBilgi.oku(ses);
      expect(b, isNotNull);
      expect(b!.cerceveSayisi, 49);
      expect(b.gecerliBayt, 49 * 200);
      expect(b.gecerliBayt, lessThan(await ses.length()));
    });

    test('tek baytlik kalinti da sorun cikarmaz', () async {
      final Uint8List tam = adtsAkisi(5);
      await ses.writeAsBytes(<int>[...tam, 0xFF]);

      final AdtsBilgi? b = await AdtsBilgi.oku(ses);
      expect(b!.cerceveSayisi, 5);
      expect(b.gecerliBayt, 5 * 200);
    });

    // Okuma 256 KB'lik bloklar halinde; cerceveler blok sinirina denk
    // gelince sayim kaymamali.
    test('blok sinirini asan dosyayi dogru sayar', () async {
      // 256 KB'yi asacak kadar, boyutu blok boyutuna bolunmeyen cerceveler.
      const int sayi = 4000;
      const int uzunluk = 173;
      await ses.writeAsBytes(adtsAkisi(sayi, uzunluk: uzunluk));
      expect(await ses.length(), greaterThan(256 * 1024));

      final AdtsBilgi? b = await AdtsBilgi.oku(ses);
      expect(b, isNotNull);
      expect(b!.cerceveSayisi, sayi);
      expect(b.gecerliBayt, sayi * uzunluk);
    });

    test('bir cercevede birden fazla ham blok sayilir', () async {
      await ses.writeAsBytes(adtsCercevesi(hamBlok: 3));

      final AdtsBilgi? b = await AdtsBilgi.oku(ses);
      expect(b!.cerceveSayisi, 3);
    });

    test('ADTS olmayan dosya null doner', () async {
      await ses.writeAsBytes(
        Uint8List.fromList(List<int>.generate(4096, (int i) => i & 0xFF)),
      );
      expect(await AdtsBilgi.oku(ses), isNull);
    });

    test('bos ya da cok kisa dosya null doner', () async {
      await ses.writeAsBytes(Uint8List(0));
      expect(await AdtsBilgi.oku(ses), isNull);

      await ses.writeAsBytes(Uint8List.fromList(<int>[0xFF, 0xF1, 0x50]));
      expect(await AdtsBilgi.oku(ses), isNull);
    });

    test('olmayan dosya null doner', () async {
      expect(await AdtsBilgi.oku(File('${gecici.path}/yok.aac')), isNull);
    });

    test('gecersiz hiz dizini tasiyan ilk cerceve reddedilir', () async {
      // Dizin 13-15 ayrilmis; ilk cercevede olursa dosya ADTS sayilmaz.
      await ses.writeAsBytes(adtsAkisi(3, hizDizini: 13));
      expect(await AdtsBilgi.oku(ses), isNull);
    });
  });
}
