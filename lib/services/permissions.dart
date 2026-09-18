import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Mikrofon izniyle ilgili iki soru icin ince bir yerli kopru.
///
/// Izin **istemek** bu sinifin isi degil; onu `record` paketi yapiyor
/// (`Recorder.izinVarMi`). Burada sadece reddedildikten sonra ne
/// yapacagimizi bilmek icin gereken iki sey var.
class Izinler {
  const Izinler._();

  static const MethodChannel _kanal = MethodChannel('hatirla/izin');

  /// Telefonun "uygulama bilgisi" ekranini acar; kullanici izni oradan
  /// acabilir. Acilamazsa `false` doner.
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

  /// Kullanici izni "bir daha sorma" diyerek reddetmis mi?
  ///
  /// Yalnizca bir izin istegi reddedildikten **sonra** cagrilmali: ilk
  /// istekten once de `true` doner ve kullaniciyi bosuna ayarlara yollariz.
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
