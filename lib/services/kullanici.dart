import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../data/akrabalar.dart';
import 'uploader.dart';

/// Telefonu kimin kullandigi.
///
/// Ilk acilista bir kez sorulur ([Akraba]), sonra degismez. Iki isi var:
/// ekranda kisiye adiyla seslenmek ve yedekleri kime ait olduklari belli
/// olacak sekilde adlandirmak.
class Kullanici extends ChangeNotifier {
  Kullanici._();

  static final Kullanici instance = Kullanici._();

  static const String pAkraba = 'kullanici_akraba';

  Akraba? _akraba;

  /// Secim yapilmadiysa `null`.
  Akraba? get akraba => _akraba;

  /// Ilk acilista sorulacak mi?
  bool get secildi => _akraba != null;

  /// "Anneannem" gibi. Secilmediyse bos.
  String get ad => _akraba?.ad ?? '';

  /// Acilista bir kere cagrilir.
  Future<void> yukle() async {
    try {
      final SharedPreferences ayarlar = await SharedPreferences.getInstance();
      _akraba = Akraba.kimlikten(ayarlar.getString(pAkraba));
    } catch (e) {
      // Okunamazsa soru bir daha sorulur; cokmesinden iyidir.
      debugPrint('Kullanici okunamadi: $e');
    }
    notifyListeners();
  }

  /// Secimi kaydeder.
  ///
  /// Ayni adi [Yedekleyici]'ye de yaziyoruz: kovada rastgele kimlikli
  /// otuz klasor yerine dogrudan kimin anlattigi yazsin. Kuran kisi
  /// Ayarlar'dan yine degistirebilir.
  Future<void> sec(Akraba secim) async {
    _akraba = secim;
    notifyListeners();
    try {
      final SharedPreferences ayarlar = await SharedPreferences.getInstance();
      await ayarlar.setString(pAkraba, secim.kimlik);
    } catch (e) {
      // Yazilamazsa soru bir dahaki acilista tekrar cikar.
      debugPrint('Kullanici kaydedilemedi: $e');
    }
    await Yedekleyici.instance.sahibiKaydet(secim.ad);
  }

  /// Tekil nesne testler arasinda yasiyor; her test temiz baslasin.
  @visibleForTesting
  void testIcinUnut() => _akraba = null;
}
