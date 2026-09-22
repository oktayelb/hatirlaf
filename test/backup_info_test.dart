import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/backup_info.dart';

void main() {
  group('acik anahtar cozumleme', () {
    test('32 baytlik base64 kabul ediliyor', () {
      final String b64 = base64.encode(List<int>.filled(32, 5));
      expect(YedekAyarlari.acikAnahtariCoz(b64), List<int>.filled(32, 5));
    });

    test('bastaki ve sondaki bosluk sorun cikarmiyor', () {
      final String b64 = base64.encode(List<int>.filled(32, 1));
      expect(YedekAyarlari.acikAnahtariCoz('  $b64\n'), isNotNull);
    });

    test('kisa anahtar reddediliyor', () {
      expect(
        YedekAyarlari.acikAnahtariCoz(base64.encode(List<int>.filled(31, 0))),
        isNull,
      );
    });

    test('uzun anahtar reddediliyor', () {
      expect(
        YedekAyarlari.acikAnahtariCoz(base64.encode(List<int>.filled(33, 0))),
        isNull,
      );
    });

    test('base64 olmayan metin reddediliyor', () {
      expect(YedekAyarlari.acikAnahtariCoz('bu base64 degil!!'), isNull);
    });

    test('bos metin reddediliyor', () {
      expect(YedekAyarlari.acikAnahtariCoz(''), isNull);
      expect(YedekAyarlari.acikAnahtariCoz('   '), isNull);
    });
  });

  group('kurulum', () {
    test('dart-define verilmeden yedekleme kapali', () {
      // Testler --dart-define olmadan kosuyor; kapali olmasi DOGRU
      // davranis: anahtarsiz derleme de gecerli bir derleme.
      expect(YedekAyarlari.kurulu, isFalse);
      expect(YedekAyarlari.eksiklik, isNotNull);
      expect(YedekAyarlari.eksiklik, contains('B2_KEY_ID'));
      expect(YedekAyarlari.eksiklik, contains('YEDEK_ALICI_ANAHTARI'));
    });
  });

  group('dosya adi', () {
    test('cihaz/hatira/dosya duzeninde ve .hyz uzantili', () {
      expect(
        YedekAyarlari.dosyaAdi(
          cihaz: 'telefon1',
          hatiraId: 'abc123',
          dosya: 'ses.m4a',
        ),
        'telefon1/abc123/ses.m4a.hyz',
      );
    });

    test('guvensiz karakterler temizleniyor', () {
      // Kullanicinin yazdigi baslik buraya hic girmiyor ama yine de
      // ".." ile kovada yukari cikmak mumkun olmamali.
      expect(
        YedekAyarlari.dosyaAdi(
          cihaz: '../../kok',
          hatiraId: 'a b/c',
          dosya: 'ses.m4a',
        ),
        // "/" once "_" oluyor, sonra bastaki noktalar atiliyor.
        '_.._kok/a_b_c/ses.m4a.hyz',
      );
    });

    test('tek basina ".." bilesenine izin verilmiyor', () {
      // Asil onemli olan bu: bir bilesen tam olarak ".." olursa
      // indirilen arsiv acilirken hedef klasorun disina yazilabilirdi.
      expect(
        YedekAyarlari.dosyaAdi(cihaz: '..', hatiraId: '.', dosya: '...'),
        'adsiz/adsiz/adsiz.hyz',
      );
    });

    test('bastaki noktalar atiliyor', () {
      expect(
        YedekAyarlari.dosyaAdi(cihaz: '...gizli', hatiraId: 'x', dosya: 'y'),
        'gizli/x/y.hyz',
      );
    });

    test('tamamen guvensiz ad "adsiz" oluyor', () {
      expect(
        YedekAyarlari.dosyaAdi(cihaz: '', hatiraId: '', dosya: ''),
        'adsiz/adsiz/adsiz.hyz',
      );
    });

    test('turkce harfler guvenli karsiliga donuyor', () {
      expect(
        YedekAyarlari.dosyaAdi(cihaz: 'dedemin telefonu', hatiraId: 'ü', dosya: 'ses'),
        'dedemin_telefonu/_/ses.hyz',
      );
    });
  });
}
