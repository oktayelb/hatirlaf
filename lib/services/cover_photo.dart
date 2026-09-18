import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/painting.dart';
import 'package:path_provider/path_provider.dart';

/// Ana sayfanin en ustundeki tek kapak fotografi.
///
/// Hatiralara bagli degil: anlatan kisinin portresi ya da ailece sevilen
/// bir kare. Tek dosya oldugu icin `hatiralar.json` dizinine girmiyor;
/// varligi dogrudan dosyanin kendisinden okunuyor. Dizin bozulsa bile
/// fotograf yerinde kalir.
class CoverPhoto extends ChangeNotifier {
  CoverPhoto._();

  static final CoverPhoto instance = CoverPhoto._();

  static const String fileName = 'kapak.jpg';

  String _rootPath = '';
  bool _varMi = false;

  /// Fotograf hep ayni dosya adina yaziliyor. Flutter'in resim onbellegi
  /// dosya yolunu anahtar aldigi icin yeni fotograf gorunmezdi; bu sayac
  /// her degisimde artip [ValueKey] olarak widget'i yeniden kurduruyor.
  int _surum = 0;
  int get surum => _surum;

  bool get varMi => _varMi;

  /// Dosyanin mutlak yolu. [load] cagrilmadan once bos klasore isaret eder.
  String get yol => '$_rootPath/$fileName';

  /// Uygulama acilisinda bir kere cagrilir.
  Future<void> load() async {
    try {
      final Directory docs = await getApplicationDocumentsDirectory();
      _rootPath = docs.path;
      _varMi = File(yol).existsSync();
    } catch (e) {
      debugPrint('Kapak fotografi okunamadi: $e');
      _varMi = false;
    }
    notifyListeners();
  }

  /// [kaynakYolu] gecici bir dosya olabilir; icerik uygulama klasorune
  /// kopyalanir. Basarisizsa `false` doner, eski fotograf yerinde kalir.
  Future<bool> ayarla(String kaynakYolu) async {
    if (_rootPath.isEmpty) return false;
    try {
      // Once yana yazip sonra yerine tasiyoruz: kopyalama yarida kesilirse
      // eski fotograf bozulmasin.
      final File gecici = File('$yol.gecici');
      await File(kaynakYolu).copy(gecici.path);
      await gecici.rename(yol);
      await _onbellegiTemizle();
      _varMi = true;
      _surum++;
      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('Kapak fotografi kaydedilemedi: $e');
      return false;
    }
  }

  Future<void> sil() async {
    try {
      final File f = File(yol);
      if (f.existsSync()) await f.delete();
      await _onbellegiTemizle();
    } catch (e) {
      debugPrint('Kapak fotografi silinemedi: $e');
    }
    _varMi = false;
    _surum++;
    notifyListeners();
  }

  Future<void> _onbellegiTemizle() async {
    try {
      await FileImage(File(yol)).evict();
    } catch (e) {
      debugPrint('Resim onbellegi temizlenemedi: $e');
    }
  }
}
