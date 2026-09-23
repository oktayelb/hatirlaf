import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import 'kayit_servisi.dart';

/// Kayit ekraninin durumu.
enum KayitDurumu { bos, kaydediyor, duraklatildi }

/// Mikrofon kaydini yoneten tekil servis.
class Recorder extends ChangeNotifier {
  Recorder._();

  static final Recorder instance = Recorder._();

  final AudioRecorder _recorder = AudioRecorder();

  KayitDurumu _durum = KayitDurumu.bos;
  KayitDurumu get durum => _durum;

  /// Yalnizca testler icin: gercek mikrofona dokunmadan durumu kurar.
  @visibleForTesting
  void testIcinDurum(KayitDurumu d) {
    _durum = d;
    notifyListeners();
  }

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

  /// Akis modunda sesi diske yazan uclu.
  IOSink? _cikti;
  StreamSubscription<Uint8List>? _sesAbone;
  Timer? _bosaltma;
  bool _ciktiKapaniyor = false;

  /// Diske ne siklikta bosaltilir. Surec oldurulurse kaybedilen, en fazla
  /// bu kadarlik ses olur.
  static const Duration _bosaltmaAraligi = Duration(seconds: 2);

  /// AAC 64 kbps: whisper donusumu kendisi yapiyor, sikistirilmis
  /// saklamak 1 saatlik hatirayi 500 MB yerine ~30 MB'a indiriyor.
  ///
  /// Ses dosyaya degil akisa aliniyor ([startStream]) ve ADTS cerceveleri
  /// diske biz yaziyoruz. Sebep bicimde: `record` dosyaya yazarken MPEG-4
  /// kullaniyor, orada sure ve cerceve tablosu `moov` atomunda ve en sona
  /// yaziliyor -- kayit yarida kesilirse dosya hic acilmiyor. ADTS'te her
  /// cerceve kendi uzunlugunu tasir, yani yarim dosya da gecerlidir.
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

  /// Akis modunda ADTS, dosya modunda MPEG-4 yazilir; ad bicimi belli
  /// etsin ki sonradan bakan biri hangisi oldugunu bilsin.
  static const String akisDosyaAdi = 'ses.aac';
  static const String dosyaDosyaAdi = 'ses.m4a';

  /// Kaydi baslatir. Basarisiz olursa `false` doner.
  ///
  /// [klasorYolu] hatiranin klasoru; dosya adini kayit bicimi belirledigi
  /// icin yolu cagiran degil buradaki kod seciyor ([dosyaYolu]).
  Future<bool> basla(String klasorYolu) async {
    if (_durum != KayitDurumu.bos) return false;

    // Servis mikrofondan once aciliyor: Android 14 mikrofon turundeki on
    // plan servisini uygulama on plandayken istiyor.
    await KayitServisi.basla();

    try {
      final Directory klasor = Directory(klasorYolu);
      if (!klasor.existsSync()) klasor.createSync(recursive: true);

      String? yol = await _akisaBasla(klasorYolu);
      // Akis desteklenmeyen bir cihaz olursa kayit hic alinamamaktansa
      // eski yoldan, dosyaya alinsin.
      yol ??= await _dosyayaBasla(klasorYolu);
      if (yol == null) {
        await KayitServisi.bitir();
        return false;
      }

      _dosyaYolu = yol;
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

  /// Akis modu: `record` ADTS cerceveleri veriyor, diske biz yaziyoruz.
  /// Cihaz desteklemiyorsa `null` doner.
  Future<String?> _akisaBasla(String klasorYolu) async {
    final String yol = '$klasorYolu/$akisDosyaAdi';
    try {
      Stream<Uint8List> akis;
      try {
        akis = await _recorder.startStream(_config);
      } catch (e) {
        debugPrint('voiceRecognition kaynagi acilmadi, varsayilana geciliyor: $e');
        akis = await _recorder.startStream(_yedekConfig);
      }

      _ciktiKapaniyor = false;
      _cikti = File(yol).openWrite();
      _sesAbone = akis.listen(
        (Uint8List parca) {
          if (_ciktiKapaniyor) return;
          _cikti?.add(parca);
        },
        onError: (Object e) => debugPrint('Ses akisi hatasi: $e'),
        cancelOnError: false,
      );
      _bosaltma = Timer.periodic(_bosaltmaAraligi, (_) => _diskeBosalt());
      return yol;
    } catch (e) {
      debugPrint('Akis modunda kayit baslatilamadi: $e');
      await _akisiKapat();
      try {
        await _recorder.cancel();
      } catch (_) {
        // Zaten baslamamis olabilir.
      }
      try {
        final File yarim = File(yol);
        if (yarim.existsSync()) await yarim.delete();
      } catch (_) {
        // Onemli degil: dosya moduna gecerken adi da degisiyor.
      }
      return null;
    }
  }

  /// Eski yol: `record` dogrudan dosyaya yazar. Yarida kesilirse dosya
  /// acilmaz, o yuzden yalnizca akis calismadiginda kullanilir.
  Future<String?> _dosyayaBasla(String klasorYolu) async {
    final String yol = '$klasorYolu/$dosyaDosyaAdi';
    try {
      try {
        await _recorder.start(_config, path: yol);
      } catch (e) {
        debugPrint('voiceRecognition kaynagi acilmadi, varsayilana geciliyor: $e');
        await _recorder.start(_yedekConfig, path: yol);
      }
      return yol;
    } catch (e) {
      debugPrint('Dosya modunda kayit baslatilamadi: $e');
      return null;
    }
  }

  /// Yazilanlari isletim sistemine gecirir. Surec oldurulurse buraya
  /// kadari saglam kalir.
  void _diskeBosalt() {
    final IOSink? cikti = _cikti;
    if (cikti == null || _ciktiKapaniyor) return;
    unawaited(
      cikti.flush().catchError(
        (Object e) => debugPrint('Ses diske bosaltilamadi: $e'),
      ),
    );
  }

  Future<void> _akisiKapat() async {
    _bosaltma?.cancel();
    _bosaltma = null;
    _ciktiKapaniyor = true;

    try {
      await _sesAbone?.cancel();
    } catch (e) {
      debugPrint('Ses akisi kapatilamadi: $e');
    }
    _sesAbone = null;

    final IOSink? cikti = _cikti;
    _cikti = null;
    if (cikti == null) return;
    try {
      await cikti.flush();
      await cikti.close();
    } catch (e) {
      debugPrint('Ses dosyasi kapatilamadi: $e');
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
    final String? beklenen = _dosyaYolu;
    try {
      // Once kayit durur, sonra dosya kapanir: son cerceveler de insin.
      final String? yol = await _recorder.stop();
      final String? sonuc = yol ?? beklenen;
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
    await _akisiKapat();
    await KayitServisi.bitir();
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
    _bosaltma?.cancel();
    _genlik?.cancel();
    _sesAbone?.cancel();
    _recorder.dispose();
    super.dispose();
  }
}
