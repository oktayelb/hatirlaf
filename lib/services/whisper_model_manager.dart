import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:whisper_ggml/whisper_ggml.dart';

/// Indirme durumu.
enum IndirmeDurumu { yok, iniyor, hazir, hata }

/// Yaziya cevirme paketini indirir, saklar ve durumunu bildirir.
///
/// Tek model var, kullaniciya secenek sorulmuyor: `small` modelinin q5_1
/// nicemlenmis hali. Gerekce:
///
/// - Turkce'de model buyudukce kazanc buyuk. Whisper makalesinin FLEURS
///   olcumunde `base` %27,5 WER yapiyor, `small` %15,9 — yani eski
///   "Normal" secenegi her dort kelimeden birini yanlis yaziyordu.
/// - q5_1 nicemleme `small`'in dogrulugunu pratikte degistirmiyor ama
///   dosyayi 466 MB'dan 190 MB'a indiriyor: eski "Normal"den (142 MB)
///   biraz buyuk, eski "Yuksek"ten cok kucuk.
/// - Geriye tek bir makul secenek kalinca secim ekrani da gereksizdi.
class WhisperModelManager extends ChangeNotifier {
  WhisperModelManager._();

  static final WhisperModelManager instance = WhisperModelManager._();

  /// whisper_ggml'in `WhisperModel` enum'unda nicemlenmis varyantlar yok.
  /// Dosyayi kendimiz indirip alt seviye API'ye yol olarak veriyoruz;
  /// bkz. transcriber.dart.
  static const String modelDosya = 'ggml-small-q5_1.bin';

  static final Uri modelUri = Uri.parse(
    'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$modelDosya',
  );

  /// Sunucudaki dosyanin tam boyutu (2026-09'da dogrulandi).
  static const int modelBayt = 190085487;

  /// Kullaniciya gosterilecek kaba boyut.
  static const int yaklasikMb = 190;

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
  String? _dizin;

  Future<void> init() async {
    _durum = await isReady() ? IndirmeDurumu.hazir : IndirmeDurumu.yok;
    notifyListeners();
  }

  /// Model dosyasinin telefondaki yolu. Klasor whisper_ggml'inkiyle ayni
  /// kalsin diye paketin kendi cozumunu kullaniyoruz.
  Future<String> modelPath() async {
    _dizin ??= await WhisperController.getModelDir();
    return '$_dizin/$modelDosya';
  }

  /// Boyut da kontrol edilir: yarim dosyayi "hazir" saymak whisper.cpp'yi
  /// acilista cokertiyor. Indirme zaten `.yarim` uzerinden gidiyor, bu
  /// ikinci savunma. Tam esitlik aranmiyor ki dosya sunucuda bir gun
  /// yeniden yuklenirse uygulama kilitlenmesin.
  Future<bool> isReady() async {
    try {
      final File f = File(await modelPath());
      if (!f.existsSync()) return false;
      return await f.length() >= modelBayt * 0.95;
    } catch (e) {
      debugPrint('Model kontrolu basarisiz: $e');
      return false;
    }
  }

  void iptalEt() {
    _iptal = true;
    _client?.close(force: true);
  }

  /// Modeli indirir. Once `.yarim` uzantiyla yazilir, tamamlaninca asil
  /// ada tasinir.
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

      final HttpClientRequest req = await _client!.getUrl(modelUri);
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
        // Saniyede ~10 kez: her pakette setState eski telefonlari kasiyor.
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

  /// Teknik hatayi anlasilir bir cumleye cevirir.
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

  /// Indirilen modeli siler (Ayarlar'dan yer acmak icin).
  Future<void> modelSil() async {
    try {
      final File f = File(await modelPath());
      if (f.existsSync()) await f.delete();
    } catch (e) {
      debugPrint('Model silinemedi: $e');
    }
    _durum = IndirmeDurumu.yok;
    _ilerleme = null;
    notifyListeners();
  }
}

class _IptalEdildi implements Exception {
  const _IptalEdildi();
}
