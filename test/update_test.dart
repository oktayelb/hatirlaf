import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/network.dart';
import 'package:hatirla/services/update_info.dart';

/// 64 hanelik gecerli ozetler. Varsayilan parametre olarak kullanildigi
/// icin sabit olmak zorunda; `'a' * 64` derlenmiyor.
const String _ozetArm64 =
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const String _ozetArm32 =
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

const String _kok =
    'https://github.com/oktayelb/hatirlaf/releases/download/v1.0.1';

/// Telefonlarin bildirdigi tipik mimari listeleri.
const List<String> _yeniTelefon = <String>['arm64-v8a', 'armeabi-v7a'];
const List<String> _eskiTelefon = <String>['armeabi-v7a'];

Map<String, Object?> _paket({
  Object? surumKodu = 2002,
  Object? apkUrl = '$_kok/hatirlaf-arm64-v8a.apk',
  Object? sha256 = _ozetArm64,
  Object? boyut = 23000000,
}) {
  return <String, Object?>{
    'surumKodu': surumKodu,
    'apkUrl': apkUrl,
    'sha256': sha256,
    'boyut': boyut,
  };
}

/// Gecerli bir `guncelleme.json` uretir; testler yalnizca inceledikleri
/// alani degistirir.
String _json({
  Object? surumAdi = '1.0.1',
  Object? notlar = 'Kayıt düğmesi büyütüldü.',
  Object? paketler,
}) {
  return json.encode(<String, Object?>{
    'surumAdi': surumAdi,
    'notlar': notlar,
    'paketler': paketler ??
        <String, Object?>{
          'arm64-v8a': _paket(),
          'armeabi-v7a': _paket(
            surumKodu: 1002,
            apkUrl: '$_kok/hatirlaf-armeabi-v7a.apk',
            sha256: _ozetArm32,
            boyut: 29000000,
          ),
        },
  });
}

void main() {
  group('GuncellemeBilgisi.cozumle', () {
    test('gecerli dosyayi okur', () {
      final GuncellemeBilgisi? b =
          GuncellemeBilgisi.cozumle(_json(), _yeniTelefon);
      expect(b, isNotNull);
      expect(b!.surumKodu, 2002);
      expect(b.surumAdi, '1.0.1');
    });

    test('bozuk JSON uygulamayi coktermez, null doner', () {
      expect(GuncellemeBilgisi.cozumle('{', _yeniTelefon), isNull);
      expect(GuncellemeBilgisi.cozumle('', _yeniTelefon), isNull);
      expect(GuncellemeBilgisi.cozumle('[]', _yeniTelefon), isNull);
      expect(GuncellemeBilgisi.cozumle('null', _yeniTelefon), isNull);
    });

    test('paketler yoksa null doner', () {
      expect(GuncellemeBilgisi.cozumle(_json(paketler: 'olmaz'), _yeniTelefon),
          isNull);
      expect(
        GuncellemeBilgisi.cozumle(
          _json(paketler: <String, Object?>{}),
          _yeniTelefon,
        ),
        isNull,
      );
    });

    test('surum adi yoksa surum koduna duser', () {
      final GuncellemeBilgisi? b =
          GuncellemeBilgisi.cozumle(_json(surumAdi: '  '), _yeniTelefon);
      expect(b?.surumAdi, '2002');
    });
  });

  group('mimari secimi', () {
    // 64 bit telefon iki APK'yi da calistirabilir; 22 MB olani secmeli.
    test('64 bit telefon arm64 paketini secer', () {
      final GuncellemeBilgisi? b =
          GuncellemeBilgisi.cozumle(_json(), _yeniTelefon);
      expect(b!.abi, 'arm64-v8a');
      expect(b.sha256, _ozetArm64);
      expect(b.boyut, 23000000);
    });

    test('32 bit telefon armeabi-v7a paketini secer', () {
      final GuncellemeBilgisi? b =
          GuncellemeBilgisi.cozumle(_json(), _eskiTelefon);
      expect(b!.abi, 'armeabi-v7a');
      expect(b.sha256, _ozetArm32);
      expect(b.apkUrl, endsWith('hatirlaf-armeabi-v7a.apk'));
    });

    // Flutter surum kodunu mimariye gore kaydiriyor; her paket kendi
    // gercek kodunu tasimazsa karsilastirma yanlis cikar.
    test('her mimari kendi gercek surum kodunu tasir', () {
      expect(
        GuncellemeBilgisi.cozumle(_json(), _yeniTelefon)!.surumKodu,
        2002,
      );
      expect(
        GuncellemeBilgisi.cozumle(_json(), _eskiTelefon)!.surumKodu,
        1002,
      );
    });

    test('surum kodu olmayan ya da gecersiz paket atlanir', () {
      expect(
        GuncellemeBilgisi.cozumle(
          _json(paketler: <String, Object?>{'arm64-v8a': _paket(surumKodu: null)}),
          _yeniTelefon,
        ),
        isNull,
      );
      expect(
        GuncellemeBilgisi.cozumle(
          _json(paketler: <String, Object?>{'arm64-v8a': _paket(surumKodu: 0)}),
          _yeniTelefon,
        ),
        isNull,
      );
    });

    // Yanlis mimarideki APK bos veri harcar ve "uyumsuz" hatasi verir.
    test('cihazin mimarisi yoksa guncelleme onerilmez', () {
      expect(
        GuncellemeBilgisi.cozumle(_json(), const <String>['riscv64']),
        isNull,
      );
      expect(GuncellemeBilgisi.cozumle(_json(), const <String>[]), isNull);
    });

    test('yalnizca bozuk paket varsa o mimari atlanir', () {
      final String ham = _json(
        paketler: <String, Object?>{
          'arm64-v8a': _paket(sha256: 'kisa'), // bozuk
          'armeabi-v7a': _paket(
            surumKodu: 1002,
            apkUrl: '$_kok/hatirlaf-armeabi-v7a.apk',
            sha256: _ozetArm32,
            boyut: 29000000,
          ),
        },
      );
      final GuncellemeBilgisi? b =
          GuncellemeBilgisi.cozumle(ham, _yeniTelefon);
      expect(b!.abi, 'armeabi-v7a');
    });
  });

  group('paket dogrulamasi', () {
    GuncellemeBilgisi? tekPaketle(Map<String, Object?> p) =>
        GuncellemeBilgisi.cozumle(
          _json(paketler: <String, Object?>{'arm64-v8a': p}),
          _yeniTelefon,
        );

    // Manifest degistirilse bile rastgele bir adresten APK inmemeli.
    test('baska sunucudaki APK reddedilir', () {
      expect(tekPaketle(_paket(apkUrl: 'https://kotuadam.example/x.apk')),
          isNull);
      expect(
        tekPaketle(
          _paket(apkUrl: 'https://github.com.kotuadam.example/a/releases/x.apk'),
        ),
        isNull,
      );
    });

    test('baska bir GitHub deposu reddedilir', () {
      expect(
        tekPaketle(
          _paket(
            apkUrl: 'https://github.com/baskasi/uygulama/releases/'
                'latest/download/x.apk',
          ),
        ),
        isNull,
      );
    });

    test('sifresiz (http) adres reddedilir', () {
      expect(
        tekPaketle(
          _paket(
            apkUrl: 'http://github.com/oktayelb/hatirlaf/releases/'
                'latest/download/hatirlaf-arm64-v8a.apk',
          ),
        ),
        isNull,
      );
    });

    test('gecersiz sha256 reddedilir', () {
      expect(tekPaketle(_paket(sha256: 'kisa')), isNull);
      expect(tekPaketle(_paket(sha256: 'z' * 64)), isNull);
      expect(tekPaketle(_paket(sha256: null)), isNull);
    });

    test('sha256 buyuk harfle yazilmissa da kabul edilir', () {
      final GuncellemeBilgisi? b =
          tekPaketle(_paket(sha256: _ozetArm64.toUpperCase()));
      expect(b?.sha256, _ozetArm64);
    });

    test('akil disi buyuklukteki APK reddedilir', () {
      expect(tekPaketle(_paket(boyut: 0)), isNull);
      expect(tekPaketle(_paket(boyut: -1)), isNull);
      expect(tekPaketle(_paket(boyut: null)), isNull);
      expect(
        tekPaketle(_paket(boyut: GuncellemeKaynagi.enBuyukApkBayt + 1)),
        isNull,
      );
    });
  });

  group('denetim araligi', () {
    final DateTime simdi = DateTime(2026, 3, 4, 12);

    test('hic denetlenmediyse denetlenir', () {
      expect(
        GuncellemePolitikasi.denetimZamaniGeldiMi(
          sonDenetim: null,
          simdi: simdi,
        ),
        isTrue,
      );
    });

    test('az once denetlendiyse tekrar denetlenmez', () {
      expect(
        GuncellemePolitikasi.denetimZamaniGeldiMi(
          sonDenetim: simdi.subtract(const Duration(hours: 2)),
          simdi: simdi,
        ),
        isFalse,
      );
    });

    test('sure dolduysa denetlenir', () {
      expect(
        GuncellemePolitikasi.denetimZamaniGeldiMi(
          sonDenetim: simdi.subtract(const Duration(hours: 21)),
          simdi: simdi,
        ),
        isTrue,
      );
    });

    // Gelecege ait bir damga denetimi sonsuza kadar kilitlememeli.
    test('gelecege ait damga denetimi kilitlemez', () {
      expect(
        GuncellemePolitikasi.denetimZamaniGeldiMi(
          sonDenetim: simdi.add(const Duration(days: 400)),
          simdi: simdi,
        ),
        isTrue,
      );
    });
  });

  group('AgDurumu', () {
    // Mobil veri artik engel degil: mimariye ozel APK ~22 MB.
    test('mobil veride de indirme yapilir', () {
      expect(AgDurumu.sayacli.internetVar, isTrue);
      expect(AgDurumu.serbest.internetVar, isTrue);
      expect(AgDurumu.yok.internetVar, isFalse);
    });

    test('sayacsiz ayrimi bilgi olarak duruyor', () {
      expect(AgDurumu.serbest.sayacsiz, isTrue);
      expect(AgDurumu.sayacli.sayacsiz, isFalse);
    });

    test('taninmayan durum guvenli tarafa duser', () {
      expect(AgDurumu.adindan('boyle_bir_sey_yok'), AgDurumu.yok);
      expect(AgDurumu.adindan(null), AgDurumu.yok);
    });
  });
}
