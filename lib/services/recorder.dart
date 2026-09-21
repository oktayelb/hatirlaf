import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

/// Kayit ekraninin durumu.
enum KayitDurumu { bos, kaydediyor, duraklatildi }

/// Mikrofon kaydini yoneten tekil servis.
class Recorder extends ChangeNotifier {
  Recorder._();

  static final Recorder instance = Recorder._();

  final AudioRecorder _recorder = AudioRecorder();

  KayitDurumu _durum = KayitDurumu.bos;
  KayitDurumu get durum => _durum;

  String? _dosyaYolu;
  String? get dosyaYolu => _dosyaYolu;

  Duration _sure = Duration.zero;
  Duration get sure => _sure;

  /// 0.0 - 1.0 arasi, dalga animasyonu icin. Sessizlikte 0'a yaklasir.
  double _seviye = 0;
  double get seviye => _seviye;

  Timer? _sayac;
  StreamSubscription<Amplitude>? _genlik;
  DateTime? _baslangic;
  Duration _birikmis = Duration.zero;

  /// AAC/m4a: whisper donusumu kendisi yapiyor, sikistirilmis saklamak
  /// 1 saatlik hatirayi 500 MB yerine ~30 MB'a indiriyor.
  static const RecordConfig _config = RecordConfig(
    encoder: AudioEncoder.aacLc,
    bitRate: 64000,
    sampleRate: 44100,
    numChannels: 1,
    autoGain: true,
    noiseSuppress: true,
    androidConfig: AndroidRecordConfig(
      audioSource: AndroidAudioSource.voiceRecognition,
    ),
  );

  /// Bazi cihazlarda `voiceRecognition` acilmiyor; varsayilana dusuyoruz.
  static const RecordConfig _yedekConfig = RecordConfig(
    encoder: AudioEncoder.aacLc,
    bitRate: 64000,
    sampleRate: 44100,
    numChannels: 1,
  );

  Future<bool> izinVarMi({bool iste = true}) async {
    try {
      return await _recorder.hasPermission(request: iste);
    } catch (e) {
      debugPrint('Mikrofon izni sorgulanamadi: $e');
      return false;
    }
  }

  /// Kaydi baslatir. Basarisiz olursa `false` doner.
  Future<bool> basla(String hedefYol) async {
    if (_durum != KayitDurumu.bos) return false;
    try {
      final Directory parent = File(hedefYol).parent;
      if (!parent.existsSync()) parent.createSync(recursive: true);

      try {
        await _recorder.start(_config, path: hedefYol);
      } catch (e) {
        debugPrint('voiceRecognition kaynagi acilmadi, varsayilana geciliyor: $e');
        await _recorder.start(_yedekConfig, path: hedefYol);
      }

      _dosyaYolu = hedefYol;
      _durum = KayitDurumu.kaydediyor;
      _sure = Duration.zero;
      _birikmis = Duration.zero;
      _baslangic = DateTime.now();
      _sayacBasla();
      _genlikDinle();
      // Ekran kapanirsa kayit kesilebiliyor.
      unawaited(WakelockPlus.enable());
      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('Kayit baslatilamadi: $e');
      await _temizle();
      return false;
    }
  }

  Future<void> duraklat() async {
    if (_durum != KayitDurumu.kaydediyor) return;
    try {
      await _recorder.pause();
      _birikmis = _sure;
      _baslangic = null;
      _sayac?.cancel();
      _seviye = 0;
      _durum = KayitDurumu.duraklatildi;
      notifyListeners();
    } catch (e) {
      debugPrint('Kayit duraklatilamadi: $e');
    }
  }

  Future<void> devamEt() async {
    if (_durum != KayitDurumu.duraklatildi) return;
    try {
      await _recorder.resume();
      _baslangic = DateTime.now();
      _sayacBasla();
      _durum = KayitDurumu.kaydediyor;
      notifyListeners();
    } catch (e) {
      debugPrint('Kayda devam edilemedi: $e');
    }
  }

  /// Kaydi bitirir ve `(dosyaYolu, sure)` doner. Basarisizsa null.
  Future<({String yol, Duration sure})?> bitir() async {
    if (_durum == KayitDurumu.bos) return null;
    final Duration kayitSuresi = _sure;
    try {
      final String? yol = await _recorder.stop();
      final String? sonuc = yol ?? _dosyaYolu;
      await _temizle();
      if (sonuc == null || !File(sonuc).existsSync()) return null;
      // Cok kisa ya da bos dosya.
      if (await File(sonuc).length() < 1024) {
        try {
          await File(sonuc).delete();
        } catch (_) {}
        return null;
      }
      return (yol: sonuc, sure: kayitSuresi);
    } catch (e) {
      debugPrint('Kayit bitirilemedi: $e');
      await _temizle();
      return null;
    }
  }

  /// Kaydi iptal eder ve dosyayi siler.
  Future<void> iptal() async {
    if (_durum == KayitDurumu.bos) return;
    final String? yol = _dosyaYolu;
    try {
      await _recorder.cancel();
    } catch (e) {
      debugPrint('Kayit iptal edilemedi: $e');
    }
    await _temizle();
    if (yol != null) {
      try {
        final File f = File(yol);
        if (f.existsSync()) await f.delete();
      } catch (_) {}
    }
  }

  void _sayacBasla() {
    _sayac?.cancel();
    _sayac = Timer.periodic(const Duration(milliseconds: 200), (_) {
      if (_baslangic == null) return;
      _sure = _birikmis + DateTime.now().difference(_baslangic!);
      notifyListeners();
    });
  }

  void _genlikDinle() {
    _genlik?.cancel();
    _genlik = _recorder
        .onAmplitudeChanged(const Duration(milliseconds: 120))
        .listen(
      (Amplitude a) {
        // dBFS (-60 .. 0) araligini 0..1'e tasi.
        final double db = a.current.isFinite ? a.current : -60;
        final double norm = ((db + 50) / 50).clamp(0.0, 1.0);
        // Yumusat: cubuk zipzip etmesin.
        _seviye = _seviye + (math.pow(norm, 0.7).toDouble() - _seviye) * 0.45;
        notifyListeners();
      },
      onError: (Object e) => debugPrint('Genlik dinlenemedi: $e'),
    );
  }

  Future<void> _temizle() async {
    _sayac?.cancel();
    _sayac = null;
    await _genlik?.cancel();
    _genlik = null;
    _durum = KayitDurumu.bos;
    _seviye = 0;
    _baslangic = null;
    _birikmis = Duration.zero;
    _dosyaYolu = null;
    unawaited(WakelockPlus.disable());
    notifyListeners();
  }

  @override
  void dispose() {
    _sayac?.cancel();
    _genlik?.cancel();
    _recorder.dispose();
    super.dispose();
  }
}
