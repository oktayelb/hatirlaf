import 'dart:convert';

/// Yedeklemenin derleme zamani ayarlari.
///
/// Degerler `--dart-define-from-file=yedek.json` ile gelir; `yedek.json`
/// gitignore'da. Depo herkese acik oldugu icin B2 yazma anahtari
/// kaynaga GIRMEMELI: sizarsa yabanci biri kovaya yazabilir.
///
/// Ayarlar eksikse yedekleme tamamen kapalidir; uygulama eskisi gibi
/// calisir. Boylece anahtarsiz derleme de gecerli bir derlemedir.
class YedekAyarlari {
  const YedekAyarlari._();

  static const String b2AnahtarKimligi =
      String.fromEnvironment('B2_KEY_ID');
  static const String b2Anahtari = String.fromEnvironment('B2_APP_KEY');
  static const String b2KovaKimligi =
      String.fromEnvironment('B2_BUCKET_ID');

  /// Alicinin X25519 acik anahtari, base64. Acik anahtar gizli degildir;
  /// APK'ya girmesi sorun olmaz, zaten amaci budur.
  static const String aliciAcikAnahtariB64 =
      String.fromEnvironment('YEDEK_ALICI_ANAHTARI');

  /// Yalnizca kablosuz agda yukle. Kayitlar guncellemeden cok daha
  /// buyuk; kimsenin mobil veri paketi bunun icin harcanmamali.
  static const bool yalnizcaKablosuz = true;

  static List<int>? _cozulmus;
  static bool _cozuldu = false;

  /// Acik anahtarin cozulmus hali; bicim bozuksa `null`.
  static List<int>? get aliciAcikAnahtari {
    if (_cozuldu) return _cozulmus;
    _cozuldu = true;
    _cozulmus = acikAnahtariCoz(aliciAcikAnahtariB64);
    return _cozulmus;
  }

  /// Ayrik cozumleyici: testlerden dogrudan cagrilabilsin diye.
  static List<int>? acikAnahtariCoz(String b64) {
    final String kirpik = b64.trim();
    if (kirpik.isEmpty) return null;
    try {
      final List<int> ham = base64.decode(kirpik);
      // X25519 acik anahtari her zaman 32 bayttir. Yanlis uzunluktaki
      // bir deger genellikle yanlis yapistirmadir; sessizce kabul
      // edersek sifreleme calisma aninda patlardi.
      if (ham.length != 32) return null;
      return List<int>.unmodifiable(ham);
    } on FormatException {
      return null;
    }
  }

  /// Hepsi yerinde mi? Biri bile eksikse yedekleme kapali.
  static bool get kurulu =>
      b2AnahtarKimligi.isNotEmpty &&
      b2Anahtari.isNotEmpty &&
      b2KovaKimligi.isNotEmpty &&
      aliciAcikAnahtari != null;

  /// Kurulum neden eksik? Ayarlar ekraninda kuran kisiye gosterilir.
  static String? get eksiklik {
    if (kurulu) return null;
    final List<String> eksik = <String>[
      if (b2AnahtarKimligi.isEmpty) 'B2_KEY_ID',
      if (b2Anahtari.isEmpty) 'B2_APP_KEY',
      if (b2KovaKimligi.isEmpty) 'B2_BUCKET_ID',
      if (aliciAcikAnahtariB64.trim().isEmpty)
        'YEDEK_ALICI_ANAHTARI'
      else if (aliciAcikAnahtari == null)
        'YEDEK_ALICI_ANAHTARI (32 bayt base64 olmali)',
    ];
    return eksik.join(', ');
  }

  /// Kovadaki dosya adi. Cihaz basina klasor: hangi telefondan geldigi
  /// belli olsun ve iki telefon birbirinin dosyasini ezmesin.
  ///
  /// [cihaz] ve [hatiraId] yalnizca guvenli karakterlere indirgenir;
  /// kullanicinin yazdigi baslik buraya hic girmez.
  static String dosyaAdi({
    required String cihaz,
    required String hatiraId,
    required String dosya,
  }) =>
      '${_temiz(cihaz)}/${_temiz(hatiraId)}/${_temiz(dosya)}.hyz';

  static final RegExp _guvensiz = RegExp(r'[^A-Za-z0-9._-]');

  static String _temiz(String s) {
    final String t = s.replaceAll(_guvensiz, '_');
    // Bastaki nokta ve bos ad ise yarar dosya adlari uretmiyor.
    final String u = t.replaceAll(RegExp(r'^\.+'), '');
    return u.isEmpty ? 'adsiz' : u;
  }
}
