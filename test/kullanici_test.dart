import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/data/akrabalar.dart';
import 'package:hatirla/services/kullanici.dart';
import 'package:hatirla/services/uploader.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Yedekleyici tekili Recorder'a bakiyor; Recorder kurulurken yerli
  // kanala gidiyor ve testte eklenti yok.
  const MethodChannel kayitKanali =
      MethodChannel('com.llfbandit.record/messages');

  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    Kullanici.instance.testIcinUnut();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(kayitKanali, (MethodCall c) async => null);
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(kayitKanali, null);
  });

  group('Akraba', () {
    test('kimlikler benzersiz ve adlar dolu', () {
      final Set<String> kimlikler =
          Akraba.values.map((Akraba a) => a.kimlik).toSet();
      expect(kimlikler.length, Akraba.values.length);
      for (final Akraba a in Akraba.values) {
        expect(a.ad.trim(), isNotEmpty);
      }
    });

    test('kaydedilmis kimlik geri cevriliyor', () {
      for (final Akraba a in Akraba.values) {
        expect(Akraba.kimlikten(a.kimlik), a);
      }
    });

    // Listeden bir ad kalkarsa ya da kaliciya cop yazilirsa uygulama
    // cokmemeli, yeniden sormali.
    test('taninmayan kimlik null', () {
      expect(Akraba.kimlikten(null), isNull);
      expect(Akraba.kimlikten(''), isNull);
      expect(Akraba.kimlikten('halam'), isNull);
    });
  });

  group('Kullanici', () {
    test('secim yapilmadan secildi false', () async {
      await Kullanici.instance.yukle();
      expect(Kullanici.instance.secildi, isFalse);
      expect(Kullanici.instance.ad, isEmpty);
    });

    test('secim kaliciya yaziliyor ve geri okunuyor', () async {
      await Kullanici.instance.sec(Akraba.sukruDede);

      final SharedPreferences p = await SharedPreferences.getInstance();
      expect(p.getString(Kullanici.pAkraba), Akraba.sukruDede.kimlik);

      await Kullanici.instance.yukle();
      expect(Kullanici.instance.akraba, Akraba.sukruDede);
      expect(Kullanici.instance.ad, 'Şükrü Dedem');
    });

    // Kovadaki klasorun adi secilen akrabadan geliyor: kuran kisi
    // otuz rastgele kimlige bakmak zorunda kalmasin.
    test('secim yedek sahibini de belirliyor', () async {
      await Kullanici.instance.sec(Akraba.anneanne);
      expect(Yedekleyici.instance.sahip, 'Anneannem');
    });
  });

  group('klasor adi', () {
    test('turkce harfler ASCII karsiligina iniyor', () {
      expect(Yedekleyici.asciiyeIndir('Şükrü Dedem'), 'Sukru Dedem');
      expect(Yedekleyici.asciiyeIndir('Çiğdem Ağabey'), 'Cigdem Agabey');
    });

    // Sahadaki telefonlarin klasoru yerinden oynamasin.
    test('ASCII adlar oldugu gibi kaliyor', () {
      expect(Yedekleyici.asciiyeIndir('Dedem Ahmet'), 'Dedem Ahmet');
    });

    test('sahip secilince klasor adi ada gore olusuyor', () async {
      Yedekleyici.instance.testIcinKur(cihaz: 'abc12345');
      await Kullanici.instance.sec(Akraba.sukruDede);
      expect(Yedekleyici.instance.klasorAdi, 'Sukru Dedem-abc12345');
    });
  });
}
