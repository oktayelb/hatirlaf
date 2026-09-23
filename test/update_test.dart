import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/network.dart';
import 'package:hatirla/services/update_info.dart';
import 'package:hatirla/services/yama.dart';

/// 64 hanelik gecerli ozetler. Varsayilan parametre olarak kullanildigi
/// icin sabit olmak zorunda; `'a' * 64` derlenmiyor.
const String _ozetArm64 =
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const String _ozetArm32 =
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
const String _ozetYama =
    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc';

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
  Object? yamalar,
}) {
  return <String, Object?>{
    'surumKodu': surumKodu,
    'apkUrl': apkUrl,
    'sha256': sha256,
    'boyut': boyut,
    if (yamalar != null) 'yamalar': yamalar,
  };
}

/// Bir onceki surumden bu surume gecerli bir yama kaydi.
Map<String, Object?> _yama({
  Object? kaynakSurumKodu = 2001,
  Object? kaynakSha256 = _ozetArm32,
  Object? url = '$_kok/hatirlaf-arm64-v8a-2001.yama',
  Object? sha256 = _ozetYama,
  Object? boyut = 3000000,
}) {
  return <String, Object?>{
    'kaynakSurumKodu': kaynakSurumKodu,
    'kaynakSha256': kaynakSha256,
    'url': url,
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

  group('yama kayitlari', () {
    GuncellemeBilgisi? yamayla(Object? yamalar) => GuncellemeBilgisi.cozumle(
          _json(
            paketler: <String, Object?>{
              'arm64-v8a': _paket(yamalar: yamalar),
            },
          ),
          _yeniTelefon,
        );

    test('kurulu surume uyan yama bulunur', () {
      final GuncellemeBilgisi? b = yamayla(<Object?>[_yama()]);
      expect(b!.paket.yamalar, hasLength(1));

      final GuncellemeYamasi? y = b.yamaBul(2001);
      expect(y, isNotNull);
      expect(y!.kaynakSurumKodu, 2001);
      expect(y.kaynakSha256, _ozetArm32);
      expect(y.boyut, 3000000);
    });

    // Bir surum atlayan telefon yamayi kullanamaz, tam APK indirmeli.
    test('baska surumden gelen telefon icin yama yok', () {
      expect(yamayla(<Object?>[_yama()])!.yamaBul(1999), isNull);
    });

    test('yama alani hic yoksa sorun cikmaz', () {
      final GuncellemeBilgisi? b = yamayla(null);
      expect(b, isNotNull);
      expect(b!.paket.yamalar, isEmpty);
      expect(b.yamaBul(2001), isNull);
    });

    // Bozuk bir yama kaydi guncellemeyi iptal ettirmemeli: o telefon
    // eskisi gibi tam APK indirir.
    test('bozuk yama kaydi atlanir, paket gecerli kalir', () {
      for (final Object? bozuk in <Object?>[
        'liste degil',
        <Object?>['kayit degil'],
        <Object?>[_yama(kaynakSurumKodu: null)],
        <Object?>[_yama(kaynakSha256: 'kisa')],
        <Object?>[_yama(sha256: null)],
        <Object?>[_yama(boyut: 0)],
        <Object?>[_yama(boyut: -1)],
        <Object?>[_yama(url: 'https://kotuadam.example/x.yama')],
        <Object?>[_yama(url: 'http://github.com/oktayelb/hatirlaf/releases/x')],
        // Kendisinden kendisine ya da ileriden geriye yama anlamsiz.
        <Object?>[_yama(kaynakSurumKodu: 2002)],
        <Object?>[_yama(kaynakSurumKodu: 2003)],
      ]) {
        final GuncellemeBilgisi? b = yamayla(bozuk);
        expect(b, isNotNull, reason: '$bozuk');
        expect(b!.surumKodu, 2002, reason: '$bozuk');
        expect(b.paket.yamalar, isEmpty, reason: '$bozuk');
      }
    });

    test('akil disi buyuklukteki yama reddedilir', () {
      expect(
        yamayla(<Object?>[_yama(boyut: Yama.enBuyukYamaBayt + 1)])!
            .paket
            .yamalar,
        isEmpty,
      );
    });

    test('birden fazla yama arasindan dogru olani secilir', () {
      final GuncellemeBilgisi? b = yamayla(<Object?>[
        _yama(kaynakSurumKodu: 2000),
        _yama(kaynakSurumKodu: 2001),
      ]);
      expect(b!.paket.yamalar, hasLength(2));
      expect(b.yamaBul(2000)!.kaynakSurumKodu, 2000);
      expect(b.yamaBul(2001)!.kaynakSurumKodu, 2001);
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
