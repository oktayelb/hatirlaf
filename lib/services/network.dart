import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Telefonun internet durumu.
enum AgDurumu {
  /// Internet yok ya da baglanti dogrulanmadi (kafe giris sayfasi gibi).
  yok,

  /// Mobil veri ya da sinirli hotspot.
  sayacli,

  /// Wi-Fi, ethernet.
  serbest;

  static AgDurumu adindan(String? ad) => AgDurumu.values.firstWhere(
        (AgDurumu d) => d.name == ad,
        orElse: () => AgDurumu.yok,
      );

  /// Guncelleme icin tek kosul. Sayacli/sayacsiz ayrimina bakilmiyor:
  /// APK ~22 MB, mobil veriyi beklemek aylarca eski surumde kalmak demekti.
  bool get internetVar => this != AgDurumu.yok;

  /// Yalnizca Ayarlar'da bilgi amacli gosterilir.
  bool get sayacsiz => this == AgDurumu.serbest;
}

/// Ag durumunu calisma boyunca izleyen tekil servis: tek seferlik bir
/// "internet var mi?" sorusu menzil degisimlerini kacirirdi.
class Ag extends ChangeNotifier {
  Ag._();

  static final Ag instance = Ag._();

  static const EventChannel _kanal = EventChannel('hatirla/ag');

  /// Wi-Fi'ye baglanirken sistem arka arkaya birkac durum bildiriyor;
  /// her birinde indirme baslatmayalim.
  static const Duration _sakinlesmeSuresi = Duration(seconds: 3);

  AgDurumu _durum = AgDurumu.yok;
  AgDurumu get durum => _durum;

  StreamSubscription<dynamic>? _abonelik;
  Timer? _sakinlesme;

  /// [notifyListeners]'dan farki: yalnizca gercek gecislerde ve
  /// sakinlestikten sonra atesleniyor.
  final StreamController<AgDurumu> _degisim =
      StreamController<AgDurumu>.broadcast();
  Stream<AgDurumu> get degisim => _degisim.stream;

  bool _basladi = false;

  void basla() {
    if (_basladi) return;
    _basladi = true;
    try {
      _abonelik = _kanal.receiveBroadcastStream().listen(
        _geldi,
        onError: (Object e) {
          // Ag bilgisi alinamiyorsa guncelleme sessizce devre disi kalir.
          debugPrint('Ag durumu dinlenemedi: $e');
        },
      );
    } on MissingPluginException {
      // Android disi bir platform ya da test ortami.
      debugPrint('Ag kanali yok; guncelleme denetimi kapali.');
    }
  }

  void _geldi(dynamic olay) {
    final AgDurumu yeni = AgDurumu.adindan(olay as String?);
    if (yeni == _durum) return;
    _durum = yeni;
    notifyListeners();

    _sakinlesme?.cancel();
    _sakinlesme = Timer(_sakinlesmeSuresi, () {
      if (!_degisim.isClosed) _degisim.add(_durum);
    });
  }

  @override
  void dispose() {
    _sakinlesme?.cancel();
    _abonelik?.cancel();
    _degisim.close();
    super.dispose();
  }

  /// Yalnizca testler icin.
  @visibleForTesting
  void testDurumuAyarla(AgDurumu d) {
    _durum = d;
    notifyListeners();
  }
}
