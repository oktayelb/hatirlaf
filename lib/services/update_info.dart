import 'dart:convert';

import 'package:flutter/foundation.dart';

/// Guncellemelerin gelecegi yer.
///
/// Sunucu yok: `guncelleme.json` deponun ana dalinda duran sade bir dosya,
/// APK'lar ise GitHub surum (release) ekleri. Ikisi de bedava, kimliksiz
/// ve kalici. `releases/latest/download/...` adresi **her zaman** en son
/// surume gittigi icin dosya adresleri surumden surume degismiyor.
class GuncellemeKaynagi {
  const GuncellemeKaynagi._();

  static const String sahip = 'oktayelb';
  static const String depo = 'hatirlaf';

  /// Surum bilgisinin okundugu adres.
  ///
  /// `raw.githubusercontent.com` CDN'i birkac dakika onbellekliyor; buna
  /// takilmamak icin adrese her seferinde degisen bir parametre ekliyoruz.
  static Uri bilgiAdresi() => Uri.parse(
        'https://raw.githubusercontent.com/$sahip/$depo/main/guncelleme.json'
        '?t=${DateTime.now().millisecondsSinceEpoch}',
      );

  /// APK adresinin gitmesine izin verilen tek yer.
  ///
  /// `guncelleme.json` bir sekilde degistirilse bile uygulamanin rastgele
  /// bir adresten APK indirip kurmasini istemiyoruz. Yonlendirmeler
  /// (GitHub CDN'e gider) bu kontrolden sonra izleniyor; onemli olan
  /// baslangic adresinin bize ait olmasi.
  static const String izinliSunucu = 'github.com';
  static String get izinliYolBasi => '/$sahip/$depo/releases/';

  /// Makul bir APK ust siniri. Bunun uzerindeki bir "guncelleme" ya yanlis
  /// ya kotu niyetli; yasli kullanicinin hattini doldurmadan duralim.
  static const int enBuyukApkBayt = 300 * 1024 * 1024;
}

/// Tek bir islemci mimarisine ait APK.
@immutable
class GuncellemePaketi {
  const GuncellemePaketi({
    required this.surumKodu,
    required this.apkUrl,
    required this.sha256,
    required this.boyut,
  });

  /// **Bu APK'nin** gercek `versionCode`'u.
  ///
  /// Mimariye ozel derlemede Flutter surum kodunu mimariye gore kaydiriyor
  /// (armeabi-v7a +1000, arm64-v8a +2000, x86_64 +4000). Yani telefondaki
  /// kurulu kod `pubspec.yaml`'daki sayi degil. Karsilastirma yanlis
  /// olmasin diye her paket kendi gercek kodunu tasiyor; `yayinla.sh` bu
  /// sayiyi derlenmis APK'dan (`aapt2 dump badging`) okuyor.
  final int surumKodu;

  final String apkUrl;

  /// Inen dosyanin dogrulanacagi ozet (64 hanelik onaltilik).
  final String sha256;

  /// Beklenen dosya boyutu (bayt).
  final int boyut;
}

/// `guncelleme.json` icerigi - **bu cihaz icin** cozumlenmis hali.
@immutable
class GuncellemeBilgisi {
  const GuncellemeBilgisi({
    required this.surumKodu,
    required this.surumAdi,
    required this.abi,
    required this.paket,
    required this.notlar,
  });

  /// Secilen APK'nin gercek `versionCode`'u. Karsilastirma **yalniz**
  /// buna bakar; bkz. [GuncellemePaketi.surumKodu].
  final int surumKodu;

  /// Kullaniciya gosterilen etiket: "1.0.1".
  final String surumAdi;

  /// Bu cihaz icin secilen mimari ("arm64-v8a" gibi).
  final String abi;

  final GuncellemePaketi paket;

  /// "Neler degisti" - kullaniciya gosterilen sade bir iki cumle.
  final String notlar;

  String get apkUrl => paket.apkUrl;
  String get sha256 => paket.sha256;
  int get boyut => paket.boyut;

  /// Bozuk/eksik/supheli JSON'da `null` doner.
  ///
  /// [abiler] cihazin calistirabilecegi mimariler, **tercih sirasiyla**
  /// (`Build.SUPPORTED_ABIS`). Her mimari icin ayri APK yayinlaniyor:
  /// tek parca APK 60 MB, arm64'e ozel olan 22 MB. Guncelleme her surumde
  /// yeniden indirilecegi icin aradaki fark her seferinde odenirdi.
  ///
  /// Cihazin hicbir mimarisi `paketler` icinde yoksa `null` doner: yanlis
  /// mimarideki bir APK'yi indirmek bos yere veri harcamak ve kurulumda
  /// "uyumsuz" hatasi almak demek.
  ///
  /// Hicbir kosulda firlatmiyor: guncelleme denetimi yasli kullanicinin
  /// gormemesi gereken bir arka plan isi, bir yazim hatasi yuzunden
  /// uygulamanin acilmamasi kabul edilemez.
  static GuncellemeBilgisi? cozumle(String ham, List<String> abiler) {
    try {
      final dynamic kok = json.decode(ham);
      if (kok is! Map<String, dynamic>) return null;

      final dynamic paketler = kok['paketler'];
      if (paketler is! Map<String, dynamic>) return null;

      // Cihazin tercih sirasina gore ilk eslesen mimariyi al: 64 bit bir
      // telefon armeabi-v7a da calistirabilir ama arm64-v8a tercih edilmeli.
      String? secilenAbi;
      GuncellemePaketi? secilen;
      for (final String abi in abiler) {
        final GuncellemePaketi? p = _paketCozumle(paketler[abi]);
        if (p != null) {
          secilenAbi = abi;
          secilen = p;
          break;
        }
      }
      if (secilen == null || secilenAbi == null) return null;

      return GuncellemeBilgisi(
        surumKodu: secilen.surumKodu,
        surumAdi: (kok['surumAdi'] as String?)?.trim().isNotEmpty == true
            ? (kok['surumAdi'] as String).trim()
            : secilen.surumKodu.toString(),
        abi: secilenAbi,
        paket: secilen,
        notlar: (kok['notlar'] as String?)?.trim() ?? '',
      );
    } catch (e) {
      debugPrint('Guncelleme bilgisi cozumlenemedi: $e');
      return null;
    }
  }

  static GuncellemePaketi? _paketCozumle(dynamic ham) {
    if (ham is! Map<String, dynamic>) return null;

    final int? kod = (ham['surumKodu'] as num?)?.toInt();
    final String? url = ham['apkUrl'] as String?;
    final String? ozet = (ham['sha256'] as String?)?.trim().toLowerCase();
    final int? boyut = (ham['boyut'] as num?)?.toInt();

    if (kod == null || kod <= 0) return null;
    if (url == null || !_adresGuvenliMi(url)) return null;
    if (ozet == null || !_ozetGecerliMi(ozet)) return null;
    if (boyut == null ||
        boyut <= 0 ||
        boyut > GuncellemeKaynagi.enBuyukApkBayt) {
      return null;
    }

    return GuncellemePaketi(
      surumKodu: kod,
      apkUrl: url,
      sha256: ozet,
      boyut: boyut,
    );
  }

  static bool _adresGuvenliMi(String url) {
    final Uri? u = Uri.tryParse(url);
    if (u == null) return false;
    return u.scheme == 'https' &&
        u.host == GuncellemeKaynagi.izinliSunucu &&
        u.path.startsWith(GuncellemeKaynagi.izinliYolBasi);
  }

  static bool _ozetGecerliMi(String ozet) =>
      ozet.length == 64 && RegExp(r'^[0-9a-f]{64}$').hasMatch(ozet);

  @override
  String toString() => 'GuncellemeBilgisi($surumAdi / $surumKodu / $abi)';
}

/// Ne zaman denetlenir.
///
/// Guncelleme denetimi ve indirme tamamen gorunmezdir; kullanici yalnizca
/// her sey hazir oldugunda, tek dokunusla bitecek noktada bir ekranla
/// karsilasir. O ekran **atlanamaz**: guncelleme zorunludur.
class GuncellemePolitikasi {
  const GuncellemePolitikasi._();

  /// Iki denetim arasi en az sure.
  ///
  /// 24 degil 20 saat: her gun ayni saatte uygulamayi acan biri 24 saatlik
  /// pencereye hep birkac dakikayla yetisemez ve gunlerce denetim yapilmaz.
  static const Duration denetimAraligi = Duration(hours: 20);

  static bool denetimZamaniGeldiMi({
    required DateTime? sonDenetim,
    required DateTime simdi,
  }) {
    if (sonDenetim == null) return true;
    // Telefonun saati geri alinmis olabilir; gelecege ait bir "son denetim"
    // damgasi denetimi sonsuza kadar kilitlemesin.
    if (sonDenetim.isAfter(simdi)) return true;
    return simdi.difference(sonDenetim) >= denetimAraligi;
  }

}
