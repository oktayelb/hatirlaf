import 'dart:convert';

import 'package:flutter/foundation.dart';

import 'yama.dart';

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

/// Kurulu APK'dan yeni APK uretmeye yarayan fark dosyasi.
///
/// Bir yama yalnizca [kaynakSurumKodu] surumunden gecerlidir: telefonun
/// kurulu APK'si tam olarak o dosya olmak zorunda, cunku degismeyen
/// baytlar oradan kopyalaniyor. [kaynakSha256] bunu kesinlestirir.
@immutable
class GuncellemeYamasi {
  const GuncellemeYamasi({
    required this.kaynakSurumKodu,
    required this.kaynakSha256,
    required this.url,
    required this.sha256,
    required this.boyut,
  });

  /// Hangi surumden gecerli.
  final int kaynakSurumKodu;

  /// O surumun APK'sinin ozeti.
  final String kaynakSha256;

  final String url;

  /// Yama dosyasinin kendi ozeti.
  final String sha256;

  /// Bayt.
  final int boyut;
}

/// Tek bir islemci mimarisine ait APK.
@immutable
class GuncellemePaketi {
  const GuncellemePaketi({
    required this.surumKodu,
    required this.apkUrl,
    required this.sha256,
    required this.boyut,
    this.yamalar = const <GuncellemeYamasi>[],
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

  /// Bu APK'ya goturen fark dosyalari; bos olabilir.
  final List<GuncellemeYamasi> yamalar;
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

  /// [kuruluSurumKodu] surumunden gecerli yama; yoksa `null` ve tam APK
  /// inmeli. Surum atlayan telefon (ornegin bir surum boyunca internetsiz
  /// kalmis) burada bos doner, bu bir hata degil.
  GuncellemeYamasi? yamaBul(int kuruluSurumKodu) {
    for (final GuncellemeYamasi y in paket.yamalar) {
      if (y.kaynakSurumKodu == kuruluSurumKodu) return y;
    }
    return null;
  }

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
      yamalar: _yamalariCozumle(ham['yamalar'], kod),
    );
  }

  /// Bozuk bir yama kaydi guncellemeyi iptal ettirmez; yalnizca atlanir ve
  /// o telefon tam APK'yi indirir.
  static List<GuncellemeYamasi> _yamalariCozumle(dynamic ham, int hedefKod) {
    if (ham is! List<dynamic>) return const <GuncellemeYamasi>[];

    final List<GuncellemeYamasi> sonuc = <GuncellemeYamasi>[];
    for (final dynamic e in ham) {
      if (e is! Map<String, dynamic>) continue;

      final int? kaynakKod = (e['kaynakSurumKodu'] as num?)?.toInt();
      final String? kaynakOzet =
          (e['kaynakSha256'] as String?)?.trim().toLowerCase();
      final String? url = e['url'] as String?;
      final String? ozet = (e['sha256'] as String?)?.trim().toLowerCase();
      final int? boyut = (e['boyut'] as num?)?.toInt();

      // Kendisinden kendisine yama anlamsiz; ileri giden bir sey olmali.
      if (kaynakKod == null || kaynakKod <= 0 || kaynakKod >= hedefKod) {
        continue;
      }
      if (kaynakOzet == null || !_ozetGecerliMi(kaynakOzet)) continue;
      if (url == null || !_adresGuvenliMi(url)) continue;
      if (ozet == null || !_ozetGecerliMi(ozet)) continue;
      if (boyut == null || boyut <= 0 || boyut > Yama.enBuyukYamaBayt) {
        continue;
      }

      sonuc.add(
        GuncellemeYamasi(
          kaynakSurumKodu: kaynakKod,
          kaynakSha256: kaynakOzet,
          url: url,
          sha256: ozet,
          boyut: boyut,
        ),
      );
    }
    return List<GuncellemeYamasi>.unmodifiable(sonuc);
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
