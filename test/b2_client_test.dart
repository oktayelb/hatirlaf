import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart' as ozet;
import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/b2_client.dart';

/// B2 yerine gecen kucuk sunucu. Gercek hesaba dokunmadan istemcinin
/// dogru basliklari, dogru sirayi ve dogru sha1'leri urettigini gormek
/// icin: yanlis bir sha1 ancak uretimde, sessiz bozuk dosya olarak
/// ortaya cikardi.
class SahteB2 {
  SahteB2({
    this.onerilenParcaBoyu = 1000000,
    this.enKucukParcaBoyu = 100,
    this.yetkiler = const <String>['writeFiles'],
  });

  final int onerilenParcaBoyu;
  final int enKucukParcaBoyu;
  final List<String> yetkiler;

  late HttpServer _sunucu;
  String get adres => 'http://127.0.0.1:${_sunucu.port}';
  String get yetkiAdresi => '$adres/b2api/v2/b2_authorize_account';

  final List<String> cagrilar = <String>[];
  final Map<String, List<int>> yuklenenler = <String, List<int>>{};
  final Map<String, List<String>> parcaOzetleri = <String, List<String>>{};
  final Map<String, Map<String, String>> ustBilgiler =
      <String, Map<String, String>>{};

  int yetkiSayisi = 0;

  /// Bir sonraki yukleme istegine donulecek hata kodu; sonra sifirlanir.
  int? birKerelikHata;

  /// Yalnizca bu token kabul edilir; her yetkilendirmede degisir.
  String gecerliToken = '';

  Future<void> baslat() async {
    _sunucu = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _sunucu.listen(_karsila);
  }

  Future<void> durdur() => _sunucu.close(force: true);

  Future<void> _karsila(HttpRequest r) async {
    final String yol = r.uri.path;
    final List<int> govde = await _oku(r);

    Future<void> yanitla(int kod, Object icerik) async {
      r.response.statusCode = kod;
      r.response.headers.contentType = ContentType.json;
      r.response.write(json.encode(icerik));
      await r.response.close();
    }

    if (yol.endsWith('b2_authorize_account')) {
      cagrilar.add('yetki');
      yetkiSayisi++;
      gecerliToken = 'token-$yetkiSayisi';
      await yanitla(200, <String, dynamic>{
        'authorizationToken': gecerliToken,
        'apiUrl': adres,
        'downloadUrl': adres,
        'recommendedPartSize': onerilenParcaBoyu,
        'absoluteMinimumPartSize': enKucukParcaBoyu,
        'allowed': <String, dynamic>{
          'bucketId': 'kova-izinli',
          'capabilities': yetkiler,
        },
      });
      return;
    }

    if (r.headers.value('Authorization') != gecerliToken) {
      cagrilar.add('401');
      await yanitla(401, <String, dynamic>{
        'code': 'expired_auth_token',
        'message': 'token eskimis',
      });
      return;
    }

    if (yol.endsWith('b2_get_upload_url')) {
      cagrilar.add('yuklemeAdresi');
      final Map<String, dynamic> g =
          json.decode(utf8.decode(govde)) as Map<String, dynamic>;
      // Istemci izinli kova kimligini kullanmali.
      expect(g['bucketId'], 'kova-izinli');
      await yanitla(200, <String, dynamic>{
        'uploadUrl': '$adres/yukle',
        'authorizationToken': gecerliToken,
      });
      return;
    }

    if (yol.endsWith('b2_get_upload_part_url')) {
      cagrilar.add('parcaAdresi');
      await yanitla(200, <String, dynamic>{
        'uploadUrl': '$adres/yukleparca',
        'authorizationToken': gecerliToken,
      });
      return;
    }

    if (yol.endsWith('b2_start_large_file')) {
      cagrilar.add('buyukBaslat');
      final Map<String, dynamic> g =
          json.decode(utf8.decode(govde)) as Map<String, dynamic>;
      ustBilgiler['buyuk-1'] = ((g['fileInfo'] as Map<String, dynamic>?) ??
              <String, dynamic>{})
          .map((String k, dynamic v) => MapEntry<String, String>(k, '$v'));
      await yanitla(200, <String, dynamic>{'fileId': 'buyuk-1'});
      return;
    }

    if (yol.endsWith('b2_finish_large_file')) {
      cagrilar.add('buyukBitir');
      final Map<String, dynamic> g =
          json.decode(utf8.decode(govde)) as Map<String, dynamic>;
      parcaOzetleri[g['fileId'] as String] =
          (g['partSha1Array'] as List<dynamic>)
              .map((dynamic e) => e.toString())
              .toList();
      await yanitla(200, <String, dynamic>{
        'fileId': g['fileId'],
        'fileName': 'buyuk',
      });
      return;
    }

    if (yol == '/yukle' || yol == '/yukleparca') {
      final int? hata = birKerelikHata;
      if (hata != null) {
        birKerelikHata = null;
        cagrilar.add('hata$hata');
        await yanitla(hata, <String, dynamic>{'message': 'gecici'});
        return;
      }

      // Istemcinin gonderdigi sha1 gercekten govdeyi tanimliyor mu?
      final String beklenen = r.headers.value('X-Bz-Content-Sha1') ?? '';
      final String gercek = ozet.sha1.convert(govde).toString();
      if (beklenen != gercek) {
        await yanitla(400, <String, dynamic>{'message': 'sha1 tutmadi'});
        return;
      }

      if (yol == '/yukle') {
        cagrilar.add('yukle');
        final String ad =
            Uri.decodeComponent(r.headers.value('X-Bz-File-Name') ?? '');
        yuklenenler[ad] = govde;
        ustBilgiler[ad] = <String, String>{
          for (final String k in <String>['hatira', 'cihaz'])
            if (r.headers.value('X-Bz-Info-$k') != null)
              k: Uri.decodeComponent(r.headers.value('X-Bz-Info-$k')!),
        };
        await yanitla(200, <String, dynamic>{
          'fileId': 'dosya-${yuklenenler.length}',
          'fileName': ad,
        });
      } else {
        final String no = r.headers.value('X-Bz-Part-Number') ?? '?';
        cagrilar.add('parca$no');
        yuklenenler['parca-$no'] = govde;
        await yanitla(200, <String, dynamic>{'partNumber': int.parse(no)});
      }
      return;
    }

    await yanitla(404, <String, dynamic>{'message': 'bilinmeyen yol: $yol'});
  }

  static Future<List<int>> _oku(HttpRequest r) async {
    final List<int> hepsi = <int>[];
    await for (final List<int> p in r) {
      hepsi.addAll(p);
    }
    return hepsi;
  }
}

void main() {
  late SahteB2 sunucu;
  late Directory gecici;
  B2Istemcisi? istemci;

  Future<File> dosyaYaz(String ad, List<int> veri) async {
    final File f = File('${gecici.path}/$ad');
    await f.writeAsBytes(veri, flush: true);
    return f;
  }

  Future<B2Istemcisi> kur(SahteB2 s) async {
    sunucu = s;
    await s.baslat();
    istemci = B2Istemcisi(
      anahtarKimligi: 'k',
      anahtar: 'a',
      kovaKimligi: 'ayarlardaki-kova',
      yetkiAdresi: s.yetkiAdresi,
    );
    return istemci!;
  }

  setUp(() {
    gecici = Directory.systemTemp.createTempSync('hatirlaf_b2_');
  });

  tearDown(() async {
    istemci?.kapat();
    istemci = null;
    await sunucu.durdur();
    if (gecici.existsSync()) gecici.deleteSync(recursive: true);
  });

  group('oturum', () {
    test('alanlar yanittan okunuyor', () async {
      final B2Istemcisi c = await kur(SahteB2(onerilenParcaBoyu: 4242));
      final B2Oturumu o = await c.oturum();
      expect(o.token, 'token-1');
      expect(o.onerilenParcaBoyu, 4242);
      expect(o.yetkiler, <String>['writeFiles']);
      // Anahtar bir kovaya kisitliysa ayarlardaki degil O kimlik gecerli.
      expect(o.kovaKimligi, 'kova-izinli');
      expect(o.fazlaYetkiliMi, isFalse);
    });

    test('oturum onbellege aliniyor, her istekte yetkilenmiyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      await c.oturum();
      await c.oturum();
      await c.oturum();
      expect(sunucu.yetkiSayisi, 1);
    });

    test('fazla yetkili anahtar farkediliyor', () async {
      final B2Istemcisi c = await kur(
        SahteB2(yetkiler: const <String>['writeFiles', 'readFiles']),
      );
      expect((await c.oturum()).fazlaYetkiliMi, isTrue);
    });
  });

  group('tek parca yukleme', () {
    test('icerik ve sha1 sunucuda dogrulaniyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      final List<int> veri = List<int>.generate(5000, (int i) => i % 251);
      final B2Dosyasi d = await c.yukle(
        dosya: await dosyaYaz('a.bin', veri),
        ad: 'cihaz1/hatira9/ses.m4a.hyz',
      );
      expect(d.dosyaKimligi, 'dosya-1');
      expect(sunucu.yuklenenler['cihaz1/hatira9/ses.m4a.hyz'], veri);
    });

    test('dosya adindaki bosluk ve turkce harf kodlaniyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      await c.yukle(
        dosya: await dosyaYaz('b.bin', <int>[1, 2, 3]),
        ad: 'cihaz 1/hatıra ü/ses.m4a.hyz',
      );
      expect(
        sunucu.yuklenenler.keys,
        contains('cihaz 1/hatıra ü/ses.m4a.hyz'),
      );
    });

    test('ek bilgiler X-Bz-Info olarak gidiyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      await c.yukle(
        dosya: await dosyaYaz('c.bin', <int>[9]),
        ad: 'x.hyz',
        bilgiler: const <String, String>{'hatira': 'abc', 'cihaz': 'telefon 2'},
      );
      expect(sunucu.ustBilgiler['x.hyz'],
          <String, String>{'hatira': 'abc', 'cihaz': 'telefon 2'});
    });

    test('bos dosya da yuklenebiliyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      await c.yukle(dosya: await dosyaYaz('bos.bin', <int>[]), ad: 'bos.hyz');
      expect(sunucu.yuklenenler['bos.hyz'], <int>[]);
    });
  });

  group('parcali yukleme', () {
    test('parcalara bolunup ozetleriyle bitiriliyor', () async {
      final B2Istemcisi c = await kur(
        SahteB2(onerilenParcaBoyu: 100, enKucukParcaBoyu: 100),
      );
      final List<int> veri = List<int>.generate(250, (int i) => i % 256);
      await c.yukle(dosya: await dosyaYaz('buyuk.bin', veri), ad: 'buyuk.hyz');

      expect(sunucu.cagrilar, contains('buyukBaslat'));
      expect(sunucu.cagrilar, contains('buyukBitir'));
      expect(sunucu.yuklenenler['parca-1'], veri.sublist(0, 100));
      expect(sunucu.yuklenenler['parca-2'], veri.sublist(100, 200));
      expect(sunucu.yuklenenler['parca-3'], veri.sublist(200, 250));

      // Bitirme cagrisindaki ozetler parcalarla birebir eslesmeli;
      // eslesmezse B2 dosyayi sessizce bozuk birlestirir.
      final List<String> beklenen = <String>[
        ozet.sha1.convert(veri.sublist(0, 100)).toString(),
        ozet.sha1.convert(veri.sublist(100, 200)).toString(),
        ozet.sha1.convert(veri.sublist(200, 250)).toString(),
      ];
      expect(sunucu.parcaOzetleri['buyuk-1'], beklenen);
    });

    test('ilerleme geri cagrimi toplama ulasiyor', () async {
      final B2Istemcisi c = await kur(
        SahteB2(onerilenParcaBoyu: 100, enKucukParcaBoyu: 100),
      );
      final List<int> gorulen = <int>[];
      await c.yukle(
        dosya: await dosyaYaz('p.bin', List<int>.filled(250, 7)),
        ad: 'p.hyz',
        ilerleme: (int y, int t) => gorulen.add(y),
      );
      expect(gorulen, <int>[100, 200, 250]);
    });

    test('durdurma istegi yuklemeyi kesiyor', () async {
      final B2Istemcisi c = await kur(
        SahteB2(onerilenParcaBoyu: 100, enKucukParcaBoyu: 100),
      );
      await expectLater(
        c.yukle(
          dosya: await dosyaYaz('d.bin', List<int>.filled(250, 1)),
          ad: 'd.hyz',
          devamEdilsinMi: () async => false,
        ),
        throwsA(isA<B2Hatasi>()),
      );
      expect(sunucu.cagrilar, isNot(contains('buyukBitir')));
    });
  });

  group('hatalar', () {
    test('401 alinca bir kez yetkilenip tekrar deniyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      await c.oturum();
      // Sunucunun token'ini degistir: istemcinin elindeki artik gecersiz.
      sunucu.gecerliToken = 'baska-token';
      await c.yukle(dosya: await dosyaYaz('e.bin', <int>[5]), ad: 'e.hyz');

      expect(sunucu.cagrilar, contains('401'));
      expect(sunucu.yetkiSayisi, 2);
      expect(sunucu.yuklenenler['e.hyz'], <int>[5]);
    });

    test('503 gecici olarak isaretleniyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      sunucu.birKerelikHata = 503;
      await expectLater(
        c.yukle(dosya: await dosyaYaz('f.bin', <int>[1]), ad: 'f.hyz'),
        throwsA(
          isA<B2Hatasi>()
              .having((B2Hatasi h) => h.gecici, 'gecici', isTrue)
              .having((B2Hatasi h) => h.kod, 'kod', 503),
        ),
      );
    });

    test('400 kalici sayiliyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      sunucu.birKerelikHata = 400;
      await expectLater(
        c.yukle(dosya: await dosyaYaz('g.bin', <int>[1]), ad: 'g.hyz'),
        throwsA(
          isA<B2Hatasi>().having((B2Hatasi h) => h.gecici, 'gecici', isFalse),
        ),
      );
    });

    test('sunucu mesaji hataya tasiniyor', () async {
      final B2Istemcisi c = await kur(SahteB2());
      sunucu.birKerelikHata = 400;
      try {
        await c.yukle(dosya: await dosyaYaz('h.bin', <int>[1]), ad: 'h.hyz');
        fail('hata beklenirdi');
      } on B2Hatasi catch (e) {
        expect(e.mesaj, contains('gecici'));
      }
    });
  });
}
