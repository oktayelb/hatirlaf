import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/data/prompts.dart';
import 'package:hatirla/utils/format.dart';
import 'package:intl/date_symbol_data_local.dart';

void main() {
  setUpAll(() async {
    await initializeDateFormatting('tr_TR', null);
  });

  group('Bicim.sayac', () {
    test('dakika:saniye olarak yazar', () {
      expect(Bicim.sayac(const Duration(seconds: 7)), '0:07');
      expect(Bicim.sayac(const Duration(minutes: 3, seconds: 12)), '3:12');
      // 60 dakikayi asinca da dakika olarak saymaya devam eder: yaslilar
      // "1:02:30" formatini okumakta zorlaniyor.
      expect(Bicim.sayac(const Duration(hours: 1, minutes: 2)), '62:00');
    });
  });

  group('Bicim.okunurSure', () {
    test('sadece saniye', () {
      expect(Bicim.okunurSure(const Duration(seconds: 45)), '45 saniye');
    });
    test('tam dakika saniyeyi yazmaz', () {
      expect(Bicim.okunurSure(const Duration(minutes: 2)), '2 dakika');
    });
    test('dakika ve saniye', () {
      expect(
        Bicim.okunurSure(const Duration(minutes: 2, seconds: 5)),
        '2 dakika 5 saniye',
      );
    });
  });

  group('Bicim.gunlukTarih', () {
    test('bugun ve dun kelimeyle yazilir', () {
      final DateTime simdi = DateTime.now();
      expect(Bicim.gunlukTarih(simdi), 'Bugün');
      expect(
        Bicim.gunlukTarih(simdi.subtract(const Duration(days: 1))),
        'Dün',
      );
    });

    test('daha eski tarihler Turkce ay adiyla yazilir', () {
      expect(Bicim.gunlukTarih(DateTime(2025, 9, 12)), '12 Eylül 2025');
    });

    test('gece yarisina yakin saatlerde gun kaymaz', () {
      // 23:55'te kaydedilen bir hatira "Bugün" olmali; saat farki gun
      // farkina donusmemeli.
      final DateTime simdi = DateTime.now();
      final DateTime gecVakit =
          DateTime(simdi.year, simdi.month, simdi.day, 23, 55);
      expect(Bicim.gunlukTarih(gecVakit), 'Bugün');
    });
  });

  group('Bicim.onizleme', () {
    test('kisa metni oldugu gibi birakir', () {
      expect(Bicim.onizleme('Merhaba dünya'), 'Merhaba dünya');
    });
    test('satir sonlarini tek bosluga cevirir', () {
      expect(Bicim.onizleme('bir\n\n  iki   üç'), 'bir iki üç');
    });
    test('uzun metni keser ve uc nokta koyar', () {
      final String uzun = 'a' * 200;
      final String sonuc = Bicim.onizleme(uzun, enFazla: 20);
      expect(sonuc.length, 21); // 20 karakter + '…'
      expect(sonuc.endsWith('…'), isTrue);
    });
  });

  group('Bicim.boyut', () {
    test('MB olarak yuvarlar', () {
      expect(Bicim.boyut(142 * 1024 * 1024), '142 MB');
      expect(Bicim.boyut(0), '0 MB');
      expect(Bicim.boyut(500), '1 MB’den az');
    });
  });

  group('Sorular', () {
    test('her konunun en az bir sorusu var', () {
      expect(Sorular.konular, isNotEmpty);
      for (final SoruKonusu k in Sorular.konular) {
        expect(k.sorular, isNotEmpty, reason: '${k.ad} bos');
      }
    });

    test('sorular benzersiz', () {
      final List<String> hepsi = Sorular.tumSorular;
      expect(hepsi.toSet().length, hepsi.length);
    });

    test('rastgeleSoru verilen soruyu tekrar etmez', () {
      final String ilk = Sorular.tumSorular.first;
      for (int i = 0; i < 50; i++) {
        expect(Sorular.rastgeleSoru(haric: ilk), isNot(ilk));
      }
    });

    test('baslik soru isaretini ve uzunlugu kirpar', () {
      expect(
        Sorular.soruyuBasligaCevir('Çocukluğunuz nerede geçti? Anlatın.'),
        'Çocukluğunuz nerede geçti',
      );
      final String basliksiz = Sorular.soruyuBasligaCevir('x' * 100);
      expect(basliksiz.length, lessThanOrEqualTo(49));
      expect(basliksiz.endsWith('…'), isTrue);
    });
  });
}
