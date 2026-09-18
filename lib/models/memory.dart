import 'dart:io';

/// Bir hatiranin yaziya cevrilme durumu.
enum TranscriptStatus {
  /// Sirada bekliyor.
  bekliyor,

  /// Su anda yaziya cevriliyor.
  cevriliyor,

  /// Basariyla bitti.
  hazir,

  /// Hata oldu; kullanici "tekrar dene" diyebilir.
  hata;

  static TranscriptStatus fromName(String? name) {
    return TranscriptStatus.values.firstWhere(
      (TranscriptStatus s) => s.name == name,
      orElse: () => TranscriptStatus.bekliyor,
    );
  }
}

/// Tek bir sesli hatira.
///
/// Dosya yollari **gorece** tutulur (`hatiralar/<id>/ses.m4a`). Android
/// uygulama klasorunun mutlak yolu surum/yedek sonrasi degisebildigi icin
/// mutlak yol kaydetmek eski hatiralari "kayip" gosterir; bu yuzden mutlak
/// yol yalnizca calisma aninda [Memory.absolutePath] ile uretilir.
class Memory {
  const Memory({
    required this.id,
    required this.title,
    required this.createdAt,
    required this.audioRelPath,
    required this.durationMs,
    this.question,
    this.transcript = '',
    this.status = TranscriptStatus.bekliyor,
    this.errorMessage,
  });

  final String id;

  /// Kullaniciya gosterilen baslik. Bos birakilmaz.
  final String title;

  /// Kayit tarihi.
  final DateTime createdAt;

  /// `hatiralar/<id>/ses.m4a`
  final String audioRelPath;

  /// Ses uzunlugu (milisaniye). Bilinmiyorsa 0.
  final int durationMs;

  /// Kayit sirasinda ekranda duran soru (varsa).
  final String? question;

  /// Yaziya cevrilmis metin.
  final String transcript;

  final TranscriptStatus status;

  /// Hata durumunda kullaniciya gosterilecek sade aciklama.
  final String? errorMessage;

  Duration get duration => Duration(milliseconds: durationMs);

  bool get hasTranscript => transcript.trim().isNotEmpty;

  Memory copyWith({
    String? title,
    String? audioRelPath,
    int? durationMs,
    String? question,
    String? transcript,
    TranscriptStatus? status,
    String? errorMessage,
    bool clearError = false,
  }) {
    return Memory(
      id: id,
      title: title ?? this.title,
      createdAt: createdAt,
      audioRelPath: audioRelPath ?? this.audioRelPath,
      durationMs: durationMs ?? this.durationMs,
      question: question ?? this.question,
      transcript: transcript ?? this.transcript,
      status: status ?? this.status,
      errorMessage: clearError ? null : (errorMessage ?? this.errorMessage),
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
        'id': id,
        'title': title,
        'createdAt': createdAt.toIso8601String(),
        'audio': audioRelPath,
        'durationMs': durationMs,
        'question': question,
        'transcript': transcript,
        'status': status.name,
        'error': errorMessage,
      };

  static Memory fromJson(Map<String, dynamic> json) {
    return Memory(
      id: json['id'] as String,
      title: (json['title'] as String?)?.trim().isNotEmpty == true
          ? json['title'] as String
          : 'Hatıra',
      createdAt:
          DateTime.tryParse(json['createdAt'] as String? ?? '') ?? DateTime.now(),
      audioRelPath: json['audio'] as String? ?? '',
      durationMs: (json['durationMs'] as num?)?.toInt() ?? 0,
      question: json['question'] as String?,
      transcript: json['transcript'] as String? ?? '',
      status: TranscriptStatus.fromName(json['status'] as String?),
      errorMessage: json['error'] as String?,
    );
  }

  /// Gorece yolu, uygulama klasorunun [root] mutlak yoluyla birlestirir.
  static String absolutePath(String root, String relPath) =>
      '$root${Platform.pathSeparator}$relPath';
}
