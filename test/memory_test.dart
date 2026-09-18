import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/models/memory.dart';

void main() {
  final Memory ornek = Memory(
    id: 'abc-123',
    title: 'Çocukluğum',
    createdAt: DateTime(2025, 9, 12, 14, 30),
    audioRelPath: 'hatiralar/abc-123/ses.m4a',
    durationMs: 195000,
    question: 'Çocukluğunuz nerede geçti?',
    transcript: 'Köyde geçti.',
    status: TranscriptStatus.hazir,
  );

  test('JSON gidis donus bilgi kaybetmez', () {
    final Memory geri = Memory.fromJson(ornek.toJson());
    expect(geri.id, ornek.id);
    expect(geri.title, ornek.title);
    expect(geri.createdAt, ornek.createdAt);
    expect(geri.audioRelPath, ornek.audioRelPath);
    expect(geri.durationMs, ornek.durationMs);
    expect(geri.question, ornek.question);
    expect(geri.transcript, ornek.transcript);
    expect(geri.status, ornek.status);
  });

  test('bozuk/eksik JSON cokmez, makul varsayilanlara duser', () {
    final Memory geri = Memory.fromJson(<String, dynamic>{'id': 'x'});
    expect(geri.id, 'x');
    expect(geri.title, 'Hatıra');
    expect(geri.durationMs, 0);
    expect(geri.status, TranscriptStatus.bekliyor);
  });

  test('eski surumden kalan photos anahtari yok sayilir, cokmez', () {
    final Memory geri = Memory.fromJson(<String, dynamic>{
      'id': 'x',
      'photos': <String>['hatiralar/x/foto_1.jpg'],
    });
    expect(geri.id, 'x');
  });

  test('bos baslik kaydedilmis olsa bile bos gosterilmez', () {
    final Memory geri =
        Memory.fromJson(<String, dynamic>{'id': 'x', 'title': '   '});
    expect(geri.title, 'Hatıra');
  });

  test('taninmayan durum adi bekliyor olarak okunur', () {
    expect(TranscriptStatus.fromName('uydurma'), TranscriptStatus.bekliyor);
    expect(TranscriptStatus.fromName(null), TranscriptStatus.bekliyor);
    expect(TranscriptStatus.fromName('hazir'), TranscriptStatus.hazir);
  });

  test('copyWith clearError hata mesajini gercekten siler', () {
    final Memory hatali = ornek.copyWith(
      status: TranscriptStatus.hata,
      errorMessage: 'bir sorun',
    );
    expect(hatali.errorMessage, 'bir sorun');

    final Memory temiz = hatali.copyWith(
      status: TranscriptStatus.bekliyor,
      clearError: true,
    );
    expect(temiz.errorMessage, isNull);
  });

  test('copyWith id ve tarihi degistirmez', () {
    final Memory yeni = ornek.copyWith(title: 'Başka');
    expect(yeni.id, ornek.id);
    expect(yeni.createdAt, ornek.createdAt);
  });

  test('hasTranscript sadece bosluk iceren metni saymaz', () {
    expect(ornek.copyWith(transcript: '   \n ').hasTranscript, isFalse);
    expect(ornek.hasTranscript, isTrue);
  });
}
