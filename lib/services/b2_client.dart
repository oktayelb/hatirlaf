import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart' as ozet;
import 'package:meta/meta.dart';

/// Backblaze B2'ye yazan kucuk istemci.
///
/// Uygulamadaki anahtar YALNIZCA `writeFiles` yetkisine sahip olmali:
/// APK'yi parcalayan biri kimsenin kaydini indiremesin, silemesin.
/// Bunun bir bedeli var: `b2_list_parts` cagrilamadigi icin yarim kalmis
/// buyuk dosyalarin durumu sunucuya sorulamaz, yerelde tutulur.
/// B2 istemcisinin uyari kanali.
///
/// `flutter/foundation` yerine bu: istemci Flutter'a baglanmayinca
/// `dart run` ile gercek B2'ye karsi PC'den denenebiliyor. Uygulama
/// acilista bunu [debugPrint]'e baglar.
void Function(String mesaj) b2Uyari = (String m) {
  // ignore: avoid_print
  print(m);
};

class B2Hatasi implements Exception {
  const B2Hatasi(this.mesaj, {this.kod, this.gecici = false});

  final String mesaj;
  final int? kod;

  /// Tekrar denemeye deger mi? Ag hatasi, 429, 5xx.
  final bool gecici;

  @override
  String toString() => 'B2Hatasi($kod): $mesaj';
}

/// `b2_authorize_account` yaniti.
@immutable
class B2Oturumu {
  const B2Oturumu({
    required this.token,
    required this.apiAdresi,
    required this.kovaKimligi,
    required this.onerilenParcaBoyu,
    required this.enKucukParcaBoyu,
    required this.yetkiler,
  });

  final String token;
  final String apiAdresi;
  final String kovaKimligi;
  final int onerilenParcaBoyu;
  final int enKucukParcaBoyu;
  final List<String> yetkiler;

  /// Anahtar okuma yetkisi de tasiyor mu? Tasiyorsa kurulum yanlis:
  /// sizan APK butun arsivi acar.
  bool get fazlaYetkiliMi =>
      yetkiler.any((String y) => y == 'readFiles' || y == 'deleteFiles');
}

/// Tek bir yuklemenin sonucu.
@immutable
class B2Dosyasi {
  const B2Dosyasi({required this.dosyaKimligi, required this.ad});

  final String dosyaKimligi;
  final String ad;
}

class B2Istemcisi {
  B2Istemcisi({
    required this.anahtarKimligi,
    required this.anahtar,
    required this.kovaKimligi,
    HttpClient? istemci,
    String? yetkiAdresi,
  })  : _istemci = istemci ?? HttpClient(),
        _yetkiAdresi = yetkiAdresi ?? varsayilanYetkiAdresi;

  final String anahtarKimligi;
  final String anahtar;
  final String kovaKimligi;

  final HttpClient _istemci;

  static const String varsayilanYetkiAdresi =
      'https://api.backblazeb2.com/b2api/v2/b2_authorize_account';

  /// Testlerde sahte sunucuya yonlendirilebilsin diye alan; uretimde
  /// her zaman [varsayilanYetkiAdresi].
  final String _yetkiAdresi;

  /// B2'nin kesin alt siniri; daha kucuk parca kabul edilmiyor.
  static const int mutlakEnKucukParca = 5 * 1000 * 1000;

  B2Oturumu? _oturumOnbellegi;

  void kapat() => _istemci.close(force: true);

  /// Gecerli oturum; yoksa yetkilendirir.
  Future<B2Oturumu> oturum({bool tazele = false}) async {
    final B2Oturumu? mevcut = _oturumOnbellegi;
    if (!tazele && mevcut != null) return mevcut;

    final String temel =
        base64.encode(utf8.encode('$anahtarKimligi:$anahtar'));
    final Map<String, dynamic> y = await _istek(
      yontem: 'GET',
      adres: Uri.parse(_yetkiAdresi),
      basliklar: <String, String>{'Authorization': 'Basic $temel'},
    );

    final Map<String, dynamic> izin =
        (y['allowed'] as Map<String, dynamic>?) ?? <String, dynamic>{};
    // Anahtar tek bir kovaya kisitlanmissa onun kimligi buradan gelir;
    // kisitlanmamissa ayarlardaki kimligi kullaniriz.
    final String kova =
        (izin['bucketId'] as String?) ?? kovaKimligi;

    final B2Oturumu yeni = B2Oturumu(
      token: y['authorizationToken'] as String,
      apiAdresi: y['apiUrl'] as String,
      kovaKimligi: kova,
      onerilenParcaBoyu:
          (y['recommendedPartSize'] as num?)?.toInt() ?? 100 * 1000 * 1000,
      enKucukParcaBoyu:
          (y['absoluteMinimumPartSize'] as num?)?.toInt() ?? mutlakEnKucukParca,
      yetkiler: ((izin['capabilities'] as List<dynamic>?) ?? <dynamic>[])
          .map((dynamic e) => e.toString())
          .toList(growable: false),
    );

    if (yeni.fazlaYetkiliMi) {
      // Calismayi engellemiyoruz ama sessiz de gecmiyoruz: bu bir
      // kurulum hatasi ve ancak loglarda goruluyor.
      b2Uyari(
        'UYARI: B2 anahtari okuma/silme yetkisi tasiyor. '
        'Yalnizca writeFiles olan bir anahtar uretin.',
      );
    }

    _oturumOnbellegi = yeni;
    return yeni;
  }

  /// Dosyayi kovaya yazar. Buyukse parcali yukleme kullanilir.
  ///
  /// [ilerleme] 0..1 arasi; uzun kayitlarda kullaniciya gosterilebilir.
  Future<B2Dosyasi> yukle({
    required File dosya,
    required String ad,
    String icerikTuru = 'application/octet-stream',
    Map<String, String> bilgiler = const <String, String>{},
    void Function(int yuklenen, int toplam)? ilerleme,
    Future<bool> Function()? devamEdilsinMi,
  }) async {
    final int boyut = await dosya.length();
    final B2Oturumu o = await oturum();

    // Tek parcaya sigan her seyi tek istekte gonderiyoruz: parcali
    // yukleme uc ekstra tur demek.
    if (boyut < o.onerilenParcaBoyu) {
      return _tekParca(
        dosya: dosya,
        ad: ad,
        icerikTuru: icerikTuru,
        bilgiler: bilgiler,
        ilerleme: ilerleme,
      );
    }
    return _parcali(
      dosya: dosya,
      ad: ad,
      icerikTuru: icerikTuru,
      bilgiler: bilgiler,
      ilerleme: ilerleme,
      devamEdilsinMi: devamEdilsinMi,
    );
  }

  // --- tek parca ----------------------------------------------------------

  Future<B2Dosyasi> _tekParca({
    required File dosya,
    required String ad,
    required String icerikTuru,
    required Map<String, String> bilgiler,
    void Function(int, int)? ilerleme,
  }) async {
    final int boyut = await dosya.length();
    final String sha1 = await _sha1(dosya);

    return _yetkiTazeleyerek(() async {
      final B2Oturumu o = await oturum();
      final Map<String, dynamic> u = await _istek(
        yontem: 'POST',
        adres: Uri.parse('${o.apiAdresi}/b2api/v2/b2_get_upload_url'),
        basliklar: <String, String>{'Authorization': o.token},
        govde: <String, dynamic>{'bucketId': o.kovaKimligi},
      );

      final Map<String, String> basliklar = <String, String>{
        'Authorization': u['authorizationToken'] as String,
        'X-Bz-File-Name': _adiKodla(ad),
        'Content-Type': icerikTuru,
        'X-Bz-Content-Sha1': sha1,
      };
      bilgiler.forEach((String k, String v) {
        basliklar['X-Bz-Info-$k'] = Uri.encodeComponent(v);
      });

      final Map<String, dynamic> y = await _istek(
        yontem: 'POST',
        adres: Uri.parse(u['uploadUrl'] as String),
        basliklar: basliklar,
        akis: dosya.openRead(),
        uzunluk: boyut,
        ilerleme: ilerleme == null ? null : (int n) => ilerleme(n, boyut),
      );
      return B2Dosyasi(
        dosyaKimligi: y['fileId'] as String,
        ad: y['fileName'] as String? ?? ad,
      );
    });
  }

  // --- parcali ------------------------------------------------------------

  Future<B2Dosyasi> _parcali({
    required File dosya,
    required String ad,
    required String icerikTuru,
    required Map<String, String> bilgiler,
    void Function(int, int)? ilerleme,
    Future<bool> Function()? devamEdilsinMi,
  }) async {
    final int boyut = await dosya.length();
    final B2Oturumu o = await oturum();
    final int parcaBoyu = max(o.onerilenParcaBoyu, o.enKucukParcaBoyu);

    final Map<String, dynamic> bas = await _yetkiTazeleyerek(() async {
      final B2Oturumu s = await oturum();
      final Map<String, dynamic> ek = <String, dynamic>{};
      bilgiler.forEach((String k, String v) => ek[k] = v);
      return _istek(
        yontem: 'POST',
        adres: Uri.parse('${s.apiAdresi}/b2api/v2/b2_start_large_file'),
        basliklar: <String, String>{'Authorization': s.token},
        govde: <String, dynamic>{
          'bucketId': s.kovaKimligi,
          'fileName': ad,
          'contentType': icerikTuru,
          if (ek.isNotEmpty) 'fileInfo': ek,
        },
      );
    });
    final String dosyaKimligi = bas['fileId'] as String;

    final List<String> ozetler = <String>[];
    int gonderilen = 0;
    int parcaNo = 1;

    final RandomAccessFile giris = await dosya.open();
    try {
      while (gonderilen < boyut) {
        if (devamEdilsinMi != null && !await devamEdilsinMi()) {
          throw const B2Hatasi('Yukleme durduruldu', gecici: true);
        }
        final Uint8List govde = await giris.read(parcaBoyu);
        if (govde.isEmpty) break;
        final String sha1 = ozet.sha1.convert(govde).toString();

        await _yetkiTazeleyerek(() async {
          final B2Oturumu s = await oturum();
          final Map<String, dynamic> pu = await _istek(
            yontem: 'POST',
            adres:
                Uri.parse('${s.apiAdresi}/b2api/v2/b2_get_upload_part_url'),
            basliklar: <String, String>{'Authorization': s.token},
            govde: <String, dynamic>{'fileId': dosyaKimligi},
          );
          return _istek(
            yontem: 'POST',
            adres: Uri.parse(pu['uploadUrl'] as String),
            basliklar: <String, String>{
              'Authorization': pu['authorizationToken'] as String,
              'X-Bz-Part-Number': '$parcaNo',
              'X-Bz-Content-Sha1': sha1,
            },
            akis: Stream<List<int>>.value(govde),
            uzunluk: govde.length,
          );
        });

        ozetler.add(sha1);
        gonderilen += govde.length;
        parcaNo++;
        ilerleme?.call(gonderilen, boyut);
      }
    } finally {
      await giris.close();
    }

    final Map<String, dynamic> son = await _yetkiTazeleyerek(() async {
      final B2Oturumu s = await oturum();
      return _istek(
        yontem: 'POST',
        adres: Uri.parse('${s.apiAdresi}/b2api/v2/b2_finish_large_file'),
        basliklar: <String, String>{'Authorization': s.token},
        govde: <String, dynamic>{
          'fileId': dosyaKimligi,
          'partSha1Array': ozetler,
        },
      );
    });
    return B2Dosyasi(
      dosyaKimligi: son['fileId'] as String,
      ad: son['fileName'] as String? ?? ad,
    );
  }

  // --- ortak --------------------------------------------------------------

  /// Token suresi dolduysa (401) bir kez yetkilenip tekrar dener.
  Future<T> _yetkiTazeleyerek<T>(Future<T> Function() is_) async {
    try {
      return await is_();
    } on B2Hatasi catch (e) {
      if (e.kod != HttpStatus.unauthorized) rethrow;
      await oturum(tazele: true);
      return is_();
    }
  }

  /// B2 dosya adlarini yuzde kodlamasiyla ister; `/` oldugu gibi kalabilir.
  static String _adiKodla(String ad) =>
      ad.split('/').map(Uri.encodeComponent).join('/');

  static Future<String> _sha1(File dosya) async =>
      (await ozet.sha1.bind(dosya.openRead()).first).toString();

  Future<Map<String, dynamic>> _istek({
    required String yontem,
    required Uri adres,
    Map<String, String> basliklar = const <String, String>{},
    Map<String, dynamic>? govde,
    Stream<List<int>>? akis,
    int? uzunluk,
    void Function(int)? ilerleme,
  }) async {
    HttpClientRequest istek;
    try {
      istek = await _istemci.openUrl(yontem, adres);
    } on SocketException catch (e) {
      throw B2Hatasi('Baglanti kurulamadi: ${e.message}', gecici: true);
    } on HttpException catch (e) {
      throw B2Hatasi('Baglanti hatasi: ${e.message}', gecici: true);
    }

    basliklar.forEach(istek.headers.set);

    try {
      if (govde != null) {
        final List<int> ham = utf8.encode(json.encode(govde));
        istek.headers.contentType = ContentType.json;
        istek.headers.contentLength = ham.length;
        istek.add(ham);
      } else if (akis != null) {
        istek.headers.contentLength = uzunluk ?? -1;
        int gonderilen = 0;
        await for (final List<int> parca in akis) {
          istek.add(parca);
          gonderilen += parca.length;
          ilerleme?.call(gonderilen);
        }
      }

      final HttpClientResponse yanit = await istek.close();
      final String metin = await yanit.transform(utf8.decoder).join();

      if (yanit.statusCode >= 200 && yanit.statusCode < 300) {
        if (metin.trim().isEmpty) return <String, dynamic>{};
        final dynamic cozulen = json.decode(metin);
        if (cozulen is! Map<String, dynamic>) {
          throw const B2Hatasi('Beklenmeyen yanit bicimi');
        }
        return cozulen;
      }

      // 401 yetki tazelemeyi tetikler; 408/429/5xx tekrar denemeye deger.
      final bool gecici = yanit.statusCode == HttpStatus.requestTimeout ||
          yanit.statusCode == HttpStatus.tooManyRequests ||
          yanit.statusCode >= 500;
      throw B2Hatasi(
        _hataMetni(metin, yanit.statusCode),
        kod: yanit.statusCode,
        gecici: gecici,
      );
    } on B2Hatasi {
      rethrow;
    } on SocketException catch (e) {
      throw B2Hatasi('Baglanti koptu: ${e.message}', gecici: true);
    } on HttpException catch (e) {
      throw B2Hatasi('Istek yarida kaldi: ${e.message}', gecici: true);
    }
  }

  static String _hataMetni(String govde, int kod) {
    try {
      final dynamic c = json.decode(govde);
      if (c is Map<String, dynamic>) {
        final String? m = c['message'] as String?;
        final String? k = c['code'] as String?;
        if (m != null) return k == null ? m : '$m ($k)';
      }
    } catch (_) {
      // JSON degilse ham metni kullan.
    }
    final String kisa = govde.trim();
    return kisa.isEmpty ? 'HTTP $kod' : kisa.substring(0, min(200, kisa.length));
  }
}
