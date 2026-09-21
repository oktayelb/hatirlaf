import 'dart:async';
import 'dart:collection';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:whisper_ggml/whisper_ggml.dart';

import '../models/memory.dart';
import 'store.dart';
import 'whisper_model_manager.dart';

/// Ses kayitlarini sirayla yaziya ceviren arka plan servisi. Tek seferde
/// tek kayit: whisper.cpp butun cekirdekleri kullaniyor.
class Transcriber extends ChangeNotifier {
  Transcriber._();

  static final Transcriber instance = Transcriber._();

  final WhisperController _whisper = WhisperController();
  final Queue<String> _kuyruk = Queue<String>();
  final Set<String> _kuyruktakiler = <String>{};

  bool _calisiyor = false;

  String? _aktifId;
  String? get aktifId => _aktifId;

  int _yuzde = 0;
  int get yuzde => _yuzde;

  bool get mesgul => _calisiyor;
  int get bekleyenSayisi => _kuyruk.length + (_calisiyor ? 1 : 0);

  /// Acilista yarim kalmis islere devam eder: "cevriliyor" durumunda
  /// kalmis bir kayit uygulama kapandigi icin yarida kalmis demektir.
  void resumePending() {
    for (final Memory m in MemoryStore.instance.memories) {
      if (m.status == TranscriptStatus.bekliyor ||
          m.status == TranscriptStatus.cevriliyor) {
        enqueue(m.id);
      }
    }
  }

  void enqueue(String memoryId) {
    if (_aktifId == memoryId || _kuyruktakiler.contains(memoryId)) return;
    _kuyruk.add(memoryId);
    _kuyruktakiler.add(memoryId);
    notifyListeners();
    unawaited(_pump());
  }

  /// Basarisiz bir kaydi kullanici istegiyle tekrar dener.
  Future<void> retry(String memoryId) async {
    final Memory? m = MemoryStore.instance.byId(memoryId);
    if (m == null) return;
    await MemoryStore.instance.update(
      m.copyWith(status: TranscriptStatus.bekliyor, clearError: true),
    );
    enqueue(memoryId);
  }

  Future<void> _pump() async {
    if (_calisiyor) return;
    _calisiyor = true;
    notifyListeners();

    try {
      while (_kuyruk.isNotEmpty) {
        final String id = _kuyruk.removeFirst();
        _kuyruktakiler.remove(id);
        await _islet(id);
      }
    } finally {
      _calisiyor = false;
      _aktifId = null;
      _yuzde = 0;
      notifyListeners();
      // Modeli birak: 150-500 MB RAM'i bosa tutmayalim.
      try {
        await _whisper.releaseModel();
      } catch (_) {}
    }
  }

  Future<void> _islet(String id) async {
    final MemoryStore store = MemoryStore.instance;
    Memory? memory = store.byId(id);
    if (memory == null) return;

    _aktifId = id;
    _yuzde = 0;
    notifyListeners();

    // 1) Model hazir mi?
    final WhisperModelManager mm = WhisperModelManager.instance;
    if (!await mm.isReady()) {
      await store.update(
        memory.copyWith(
          status: TranscriptStatus.hata,
          errorMessage: 'Yazıya çevirme paketi telefonda yok. '
              'Ayarlar’dan indirebilirsiniz.',
        ),
      );
      return;
    }

    // 2) Ses dosyasi yerinde mi?
    final String sesYolu = store.absolute(memory.audioRelPath);
    if (!File(sesYolu).existsSync()) {
      await store.update(
        memory.copyWith(
          status: TranscriptStatus.hata,
          errorMessage: 'Ses kaydı bulunamadı.',
        ),
      );
      return;
    }

    await store.update(
      memory.copyWith(status: TranscriptStatus.cevriliyor, clearError: true),
    );

    try {
      // TranscribeResult disariya aciklanmadigi icin tip cikarimi.
      final sonuc = await _whisper.transcribe(
        model: mm.model,
        audioPath: sesYolu,
        lang: 'tr',
        // Noktalamali bir ornek, whisper'in ciktisini da noktalamali yapiyor.
        initialPrompt:
            'Aşağıda Türkçe anlatılmış bir hatıra var. Noktalama işaretleriyle yazalım.',
        suppressNonSpeechTokens: true,
        keepModelLoaded: true,
        onProgress: (int p) {
          _yuzde = p.clamp(0, 100);
          notifyListeners();
        },
      );

      // Islem sirasinda hatira silinmis olabilir.
      memory = store.byId(id);
      if (memory == null) return;

      final String metin = (sonuc?.transcription.text ?? '').trim();

      if (sonuc == null) {
        await store.update(
          memory.copyWith(
            status: TranscriptStatus.hata,
            errorMessage: 'Konuşma yazıya çevrilemedi. Tekrar deneyebilirsiniz.',
          ),
        );
        return;
      }

      await store.update(
        memory.copyWith(
          transcript: metin,
          status: TranscriptStatus.hazir,
          clearError: true,
        ),
      );
    } catch (e) {
      debugPrint('Yaziya cevirme hatasi: $e');
      memory = store.byId(id);
      if (memory == null) return;
      await store.update(
        memory.copyWith(
          status: TranscriptStatus.hata,
          errorMessage: 'Konuşma yazıya çevrilemedi. Tekrar deneyebilirsiniz.',
        ),
      );
    } finally {
      // whisper_ggml "<ses>.m4a.wav" uretiyor (~2 MB/dakika); birikmesin.
      try {
        final File wav = File('$sesYolu.wav');
        if (wav.existsSync()) await wav.delete();
      } catch (_) {}
    }
  }
}
