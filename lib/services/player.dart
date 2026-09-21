import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:just_audio/just_audio.dart';

/// Hatiralari dinletmek icin tek oynatici: yeni bir hatira calinca eskisi
/// durur.
class Player extends ChangeNotifier {
  Player._() {
    _player.playerStateStream.listen((PlayerState s) {
      _caliyor = s.playing && s.processingState != ProcessingState.completed;
      if (s.processingState == ProcessingState.completed) {
        // Bitince basa sar: "tekrar dinle" tek dokunus olsun.
        unawaited(_player.seek(Duration.zero));
        unawaited(_player.pause());
      }
      notifyListeners();
    }, onError: (Object e) => debugPrint('Oynatici durumu: $e'));

    _player.positionStream.listen((Duration p) {
      _konum = p;
      notifyListeners();
    }, onError: (Object e) => debugPrint('Oynatici konumu: $e'));

    _player.durationStream.listen((Duration? d) {
      if (d != null) _uzunluk = d;
      notifyListeners();
    }, onError: (Object e) => debugPrint('Oynatici uzunlugu: $e'));
  }

  static final Player instance = Player._();

  final AudioPlayer _player = AudioPlayer();

  String? _aktifId;
  String? get aktifId => _aktifId;

  bool _caliyor = false;
  bool get caliyor => _caliyor;

  bool _yukleniyor = false;
  bool get yukleniyor => _yukleniyor;

  Duration _konum = Duration.zero;
  Duration get konum => _konum;

  Duration _uzunluk = Duration.zero;
  Duration get uzunluk => _uzunluk;

  bool aktifMi(String id) => _aktifId == id;

  /// Verilen hatirayi calar; zaten caliyorsa duraklatir.
  Future<String?> calDurdur(String memoryId, String dosyaYolu) async {
    try {
      if (_aktifId == memoryId) {
        if (_caliyor) {
          await _player.pause();
        } else {
          await _player.play();
        }
        return null;
      }

      if (!File(dosyaYolu).existsSync()) {
        return 'Ses kaydı bulunamadı.';
      }

      _yukleniyor = true;
      _aktifId = memoryId;
      _konum = Duration.zero;
      _uzunluk = Duration.zero;
      notifyListeners();

      await _player.stop();
      await _player.setFilePath(dosyaYolu);
      _yukleniyor = false;
      notifyListeners();
      await _player.play();
      return null;
    } catch (e) {
      debugPrint('Ses calinamadi: $e');
      _yukleniyor = false;
      _aktifId = null;
      notifyListeners();
      return 'Ses çalınamadı. Kayıt bozulmuş olabilir.';
    }
  }

  Future<void> duraklat() async {
    try {
      await _player.pause();
    } catch (e) {
      debugPrint('Duraklatilamadi: $e');
    }
  }

  Future<void> durdur() async {
    try {
      await _player.stop();
    } catch (e) {
      debugPrint('Durdurulamadi: $e');
    }
    _aktifId = null;
    _caliyor = false;
    _konum = Duration.zero;
    notifyListeners();
  }

  Future<void> sar(Duration hedef) async {
    try {
      final Duration sinirli = hedef < Duration.zero
          ? Duration.zero
          : (_uzunluk > Duration.zero && hedef > _uzunluk ? _uzunluk : hedef);
      await _player.seek(sinirli);
    } catch (e) {
      debugPrint('Sarilamadi: $e');
    }
  }

  Future<void> geriSar([Duration miktar = const Duration(seconds: 10)]) =>
      sar(_konum - miktar);

  Future<void> ileriSar([Duration miktar = const Duration(seconds: 10)]) =>
      sar(_konum + miktar);

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }
}
