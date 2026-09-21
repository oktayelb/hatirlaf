import 'dart:convert';

import 'package:flutter/foundation.dart';

/// Guncellemelerin gelecegi yer: `main` dalindaki `guncelleme.json` ve
/// GitHub surum ekleri. Sunucu yok.
class GuncellemeKaynagi {
  const GuncellemeKaynagi._();

  static const String sahip = 'oktayelb';
  static const String depo = 'hatirlaf';

  /// `raw.githubusercontent.com` birkac dakika onbellekliyor; adrese her
  /// seferinde degisen bir parametre ekliyoruz.
  static Uri bilgiAdresi() => Uri.parse(
        'https://raw.githubusercontent.com/$sahip/$depo/main/guncelleme.json'
        '?t=${DateTime.now().millisecondsSinceEpoch}',
      );

  /// APK adresinin gitmesine izin verilen tek yer: manifest degistirilse
  /// bile rastgele bir adresten APK inmesin. Yonlendirmeler bu kontrolden
  /// sonra izleniyor.
  static const String izinliSunucu = 'github.com';
  static String get izinliYolBasi => '/$sahip/$depo/releases/';

  /// Ust sinir: bunun otesi ya yanlis ya kotu niyetli.
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

  /// Bu APK'nin gercek `versionCode`'u. Mimariye ozel derlemede Flutter
  /// kodu kaydiriyor (+1000/+2000/+4000), bu yuzden her paket kendi
  /// kodunu tasiyor; `yayinla.sh` APK'dan okuyor.
  final int surumKodu;

  final String apkUrl;

  /// 64 hanelik onaltilik sha256.
  final String sha256;

  /// Bayt.
  final int boyut;
}

/// `guncelleme.json` icerigi, bu cihaz icin cozumlenmis hali.
@immutable
class GuncellemeBilgisi {
  const GuncellemeBilgisi({
    required this.surumKodu,
    required this.surumAdi,
    required this.abi,
    required this.paket,
    required this.notlar,
  });

  /// Karsilastirma yalniz buna bakar; bkz. [GuncellemePaketi.surumKodu].
  final int surumKodu;

  /// Kullaniciya gosterilen etiket.
  final String surumAdi;

  /// Bu cihaz icin secilen mimari.
  final String abi;

  final GuncellemePaketi paket;

  /// "Neler degisti" metni.
  final String notlar;

  String get apkUrl => paket.apkUrl;
  String get sha256 => paket.sha256;
  int get boyut => paket.boyut;

  /// Bozuk, eksik ya da supheli JSON'da `null` doner; hicbir kosulda
  /// firlatmaz. [abiler] cihazin mimarileri, tercih sirasiyla
  /// (`Build.SUPPORTED_ABIS`); hicbiri manifest'te yoksa `null`.
  static GuncellemeBilgisi? cozumle(String ham, List<String> abiler) {
    try {
      final dynamic kok = json.decode(ham);
      if (kok is! Map<String, dynamic>) return null;

      final dynamic paketler = kok['paketler'];
      if (paketler is! Map<String, dynamic>) return null;

      // Ilk eslesen mimari: 64 bit telefon armeabi-v7a'yi da calistirir
      // ama arm64-v8a tercih edilmeli.
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
class GuncellemePolitikasi {
  const GuncellemePolitikasi._();

  /// 24 degil 20 saat: her gun ayni saatte acan biri 24 saatlik pencereye
  /// hep birkac dakikayla yetisemezdi.
  static const Duration denetimAraligi = Duration(hours: 20);

  static bool denetimZamaniGeldiMi({
    required DateTime? sonDenetim,
    required DateTime simdi,
  }) {
    if (sonDenetim == null) return true;
    // Telefonun saati geri alinmis olabilir; gelecege ait bir damga
    // denetimi sonsuza kadar kilitlemesin.
    if (sonDenetim.isAfter(simdi)) return true;
    return simdi.difference(sonDenetim) >= denetimAraligi;
  }

}
