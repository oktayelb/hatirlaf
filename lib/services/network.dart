import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Telefonun internet durumu.
enum AgDurumu {
  /// Internet yok ya da baglanti dogrulanmadi (kafe giris sayfasi, modem
  /// acik ama hat yok gibi).
  yok,

  /// Internet var ama sayacli: mobil veri veya sinirli hotspot.
  sayacli,

  /// Internet var ve sayacsiz: Wi-Fi, ethernet.
  serbest;

  static AgDurumu adindan(String? ad) => AgDurumu.values.firstWhere(
        (AgDurumu d) => d.name == ad,
        orElse: () => AgDurumu.yok,
      );

  /// Internete cikilabiliyor mu? Guncelleme icin tek kosul bu.
  ///
  /// Sayacli/sayacsiz ayrimina **bakilmiyor**: mimariye ozel APK'lar
  /// ~22 MB ve guncelleme yilda birkac kez. Mobil veriyi beklemek,
  /// eve gelen misafir torunun Wi-Fi'sini bekleyip aylarca eski surumde
  /// kalmak demekti. Ayrim yine de Ayarlar'da gosterilmek uzere duruyor.
  bool get internetVar => this != AgDurumu.yok;

  /// Sayacsiz (Wi-Fi/ethernet) mi? Yalnizca bilgi amacli gosterilir.
  bool get sayacsiz => this == AgDurumu.serbest;
}

/// Ag durumunu izleyen tekil servis.
///
/// Uygulama acilinca dinlemeye baslar ve **calisma boyunca** dinler: yasli
/// kullanici uygulamayi acip Wi-Fi menzilinden cikabilir, ya da tam tersi,
/// uygulama acikken eve girip Wi-Fi'ye baglanabilir. Ikisi de tek seferlik
/// bir "internet var mi?" sorusuyla yakalanamaz.
class Ag extends ChangeNotifier {
  Ag._();

  static final Ag instance = Ag._();

  static const EventChannel _kanal = EventChannel('hatirla/ag');

  /// Wi-Fi'ye baglanirken sistem birkac saniye icinde birden fazla durum
  /// bildiriyor (once dogrulanmamis, sonra dogrulanmis). Her birinde
  /// indirme baslatmayalim.
  static const Duration _sakinlesmeSuresi = Duration(seconds: 3);

  AgDurumu _durum = AgDurumu.yok;
  AgDurumu get durum => _durum;

  StreamSubscription<dynamic>? _abonelik;
  Timer? _sakinlesme;

  /// Durum *degistiginde* haber verir. [notifyListeners]'dan farki: bu akis
  /// yalnizca gercek gecislerde ve sakinlestikten sonra atesleniyor.
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
          // Ag bilgisi alinamiyorsa guncelleme sessizce devre disi kalir;
          // uygulamanin geri kalani bundan etkilenmemeli.
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
