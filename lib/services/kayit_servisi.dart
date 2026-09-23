import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Kayit boyunca calisan on plan servisinin Dart tarafi.
///
/// Servis kayda hicbir sey katmiyor; tek isi telefonun sureci
/// oldurmesini engellemek. Baslatilamazsa kayit yine de alinir, yalnizca
/// arka planda oldurulme riski geri gelir -- bu yuzden hicbir cagri
/// firlatmaz.
class KayitServisi {
  const KayitServisi._();

  static const MethodChannel _kanal = MethodChannel('hatirla/kayit');

  static Future<bool> basla() => _cagir('basla');

  static Future<bool> bitir() => _cagir('bitir');

  static Future<bool> _cagir(String yontem) async {
    try {
      return await _kanal.invokeMethod<bool>(yontem) ?? false;
    } on MissingPluginException {
      // Android disi platform ya da test ortami.
      return false;
    } catch (e) {
      debugPrint('Kayit servisi "$yontem" basarisiz: $e');
      return false;
    }
  }
}
