import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:whisper_ggml/whisper_ggml.dart';

/// Kullaniciya gosterilen ses tanima kalitesi secenekleri.
///
/// Whisper modelleri buyudukce Turkce dogrulugu artar ama yavaslar.
/// Varsayilan [normal]: eski telefonlarda bile makul surede biter ve
/// Turkce'de anlasilir metin uretir. `tiny` Turkce'de cok zayif oldugu
/// icin secenek olarak sunulmuyor.
enum SesKalitesi {
  normal(
    model: WhisperModel.base,
    baslik: 'Normal',
    aciklama: 'Her telefonda hızlı çalışır. Çoğu hatıra için yeterli.',
    yaklasikMb: 142,
  ),
  yuksek(
    model: WhisperModel.small,
    baslik: 'Yüksek',
    aciklama: 'Daha doğru yazar, daha uzun sürer. Yeni telefonlar için.',
    yaklasikMb: 466,
  );

  const SesKalitesi({
    required this.model,
    required this.baslik,
    required this.aciklama,
    required this.yaklasikMb,
  });

  final WhisperModel model;
  final String baslik;
  final String aciklama;
  final int yaklasikMb;

  static SesKalitesi fromName(String? name) => SesKalitesi.values.firstWhere(
        (SesKalitesi q) => q.name == name,
        orElse: () => SesKalitesi.normal,
      );
}

/// Indirme durumu.
enum IndirmeDurumu { yok, iniyor, hazir, hata }

/// Whisper model dosyasini indirir, saklar ve durumunu bildirir.
class WhisperModelManager extends ChangeNotifier {
  WhisperModelManager._();

  static final WhisperModelManager instance = WhisperModelManager._();

  static const String _prefsKey = 'ses_kalitesi';

  final WhisperController _controller = WhisperController();

  SesKalitesi _kalite = SesKalitesi.normal;
  SesKalitesi get kalite => _kalite;

  IndirmeDurumu _durum = IndirmeDurumu.yok;
  IndirmeDurumu get durum => _durum;

  /// 0.0 - 1.0 arasi. Toplam boyut bilinmiyorsa null.
  double? _ilerleme;
  double? get ilerleme => _ilerleme;

  int _inenBayt = 0;
  int get inenBayt => _inenBayt;
  int _toplamBayt = 0;
  int get toplamBayt => _toplamBayt;

  String? _hata;
  String? get hata => _hata;

  HttpClient? _client;
  bool _iptal = false;

  WhisperModel get model => _kalite.model;

  Future<void> init() async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    _kalite = SesKalitesi.fromName(prefs.getString(_prefsKey));
    _durum = await isReady() ? IndirmeDurumu.hazir : IndirmeDurumu.yok;
    notifyListeners();
  }

  Future<String> modelPath([SesKalitesi? q]) =>
      _controller.getPath((q ?? _kalite).model);

  /// Model dosyasi diskte ve makul buyuklukte mi?
  ///
  /// Yarim inmis dosyayi "hazir" saymamak icin boyut da kontrol edilir;
  /// aksi halde whisper.cpp acilista cokerdi.
  Future<bool> isReady([SesKalitesi? q]) async {
    try {
      final File f = File(await modelPath(q));
      if (!f.existsSync()) return false;
      final int len = await f.length();
      // Beklenen boyutun %90'indan kucukse dosya yarim demektir.
      final int minimum = ((q ?? _kalite).yaklasikMb * 1024 * 1024 * 0.9).round();
      return len >= minimum;
    } catch (e) {
      debugPrint('Model kontrolu basarisiz: $e');
      return false;
    }
  }

  Future<void> setKalite(SesKalitesi q) async {
    if (q == _kalite) return;
    _kalite = q;
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsKey, q.name);
    _durum = await isReady() ? IndirmeDurumu.hazir : IndirmeDurumu.yok;
    _ilerleme = null;
    _hata = null;
    notifyListeners();
  }

  void iptalEt() {
    _iptal = true;
    _client?.close(force: true);
  }

  /// Modeli indirir. Zaten varsa hemen doner.
  ///
  /// Once `.yarim` uzantili gecici dosyaya yazilir, ancak tamamen inince
  /// asil ada tasinir. Boylece internet kesilirse bir sonraki acilista
  /// yarim dosya "hazir" sanilmaz.
  Future<bool> download() async {
    if (_durum == IndirmeDurumu.iniyor) return false;
    if (await isReady()) {
      _durum = IndirmeDurumu.hazir;
      notifyListeners();
      return true;
    }

    _iptal = false;
    _hata = null;
    _inenBayt = 0;
    _toplamBayt = 0;
    _ilerleme = null;
    _durum = IndirmeDurumu.iniyor;
    notifyListeners();

    final String hedef = await modelPath();
    final File gecici = File('$hedef.yarim');
    IOSink? sink;
    try {
      final Directory parent = gecici.parent;
      if (!parent.existsSync()) parent.createSync(recursive: true);
      if (gecici.existsSync()) await gecici.delete();

      _client = HttpClient()
        ..connectionTimeout = const Duration(seconds: 30)
        ..idleTimeout = const Duration(seconds: 30);

      final HttpClientRequest req = await _client!.getUrl(model.modelUri);
      final HttpClientResponse res = await req.close();

      if (res.statusCode != HttpStatus.ok) {
        throw HttpException('Sunucu ${res.statusCode} yanıtı verdi');
      }

      _toplamBayt = res.contentLength > 0 ? res.contentLength : 0;
      sink = gecici.openWrite();

      DateTime sonBildirim = DateTime.fromMillisecondsSinceEpoch(0);
      await for (final List<int> parca in res) {
        if (_iptal) throw const _IptalEdildi();
        sink.add(parca);
        _inenBayt += parca.length;
        if (_toplamBayt > 0) _ilerleme = _inenBayt / _toplamBayt;
        // Saniyede ~10 kez guncelle; her pakette setState cagirmak
        // eski telefonlarda arayuzu kasiyor.
        final DateTime now = DateTime.now();
        if (now.difference(sonBildirim).inMilliseconds > 100) {
          sonBildirim = now;
          notifyListeners();
        }
      }
      await sink.flush();
      await sink.close();
      sink = null;

      final int inen = await gecici.length();
      if (_toplamBayt > 0 && inen != _toplamBayt) {
        throw const HttpException('İndirme yarıda kesildi');
      }

      await gecici.rename(hedef);
      _durum = IndirmeDurumu.hazir;
      _ilerleme = 1;
      notifyListeners();
      return true;
    } catch (e) {
      try {
        await sink?.close();
      } catch (_) {}
      try {
        if (gecici.existsSync()) await gecici.delete();
      } catch (_) {}
      _durum = IndirmeDurumu.hata;
      _hata = e is _IptalEdildi ? null : _kullaniciyaMesaj(e);
      notifyListeners();
      return false;
    } finally {
      _client?.close();
      _client = null;
    }
  }

  /// Teknik hatayi yaslilarin anlayacagi bir cumleye cevirir.
  static String _kullaniciyaMesaj(Object e) {
    if (e is SocketException || e is HttpException) {
      return 'İnternete bağlanılamadı. Wi-Fi bağlantınızı kontrol edip '
          'tekrar deneyin.';
    }
    if (e is FileSystemException) {
      return 'Telefonda yer kalmamış olabilir. Biraz yer açıp tekrar deneyin.';
    }
    return 'Bir sorun oldu. Tekrar deneyin.';
  }

  /// Indirilen modeli siler (Ayarlar ekranindan yer acmak icin).
  Future<void> modelSil(SesKalitesi q) async {
    try {
      final File f = File(await modelPath(q));
      if (f.existsSync()) await f.delete();
    } catch (e) {
      debugPrint('Model silinemedi: $e');
    }
    if (q == _kalite) {
      _durum = IndirmeDurumu.yok;
      _ilerleme = null;
    }
    notifyListeners();
  }
}

class _IptalEdildi implements Exception {
  const _IptalEdildi();
}
