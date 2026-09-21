import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Mikrofon izniyle ilgili iki soru icin ince bir yerli kopru. Izin
/// istemek bu sinifin isi degil; onu `record` paketi yapiyor.
class Izinler {
  const Izinler._();

  static const MethodChannel _kanal = MethodChannel('hatirla/izin');

  /// Telefonun "uygulama bilgisi" ekranini acar.
  static Future<bool> ayarlariAc() async {
    try {
      return await _kanal.invokeMethod<bool>('ayarlariAc') ?? false;
    } on PlatformException catch (e) {
      debugPrint('Ayarlar acilamadi: $e');
      return false;
    } on MissingPluginException {
      // Android disi bir platformda calisiyorsak sessizce gec.
      return false;
    }
  }

  /// "Bir daha sorma" secilmis mi? Yalnizca bir reddedilmeden SONRA
  /// cagrilmali: ilk istekten once de `true` doner.
  static Future<bool> kaliciReddedildiMi() async {
    try {
      return await _kanal.invokeMethod<bool>('kaliciReddedildiMi') ?? false;
    } on PlatformException catch (e) {
      debugPrint('Izin durumu okunamadi: $e');
      return false;
    } on MissingPluginException {
      return false;
    }
  }
}
