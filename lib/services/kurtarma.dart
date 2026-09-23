import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';

import '../models/memory.dart';
import 'adts.dart';
import 'store.dart';

/// Yarida kalmis kayitlari kurtarir.
///
/// Kayit basladigi anda hatira klasorune bir isaret dosyasi yaziliyor.
/// Uygulama normal yoldan biterse isaret siliniyor; telefon uygulamayi
/// oldurur ya da uygulama cokerse isaret yerinde kaliyor ve bir sonraki
/// acilista buradan bulunuyor.
///
/// Ses ADTS akisi olarak yazildigi icin yarim dosya da gecerli: son tam
/// cerceveye kadar olan kisim oynatilir ve yaziya cevrilir. Kaybedilen,
/// diske yetismemis son bir iki saniyedir.
class Kurtarma {
  const Kurtarma._();

  static const String isaretAdi = 'kayit.json';

  /// Bundan kisa dosyada isitilecek bir sey yok.
  static const int _enKucukSesBayt = 1024;

  /// Kayda baslarken cagrilir; hatira klasorune isareti birakir.
  static Future<void> basladi({
    required String klasor,
    required String id,
    required String baslik,
    required String sesDosyasi,
    String? soru,
  }) async {
    try {
      await File('$klasor/$isaretAdi').writeAsString(
        json.encode(<String, dynamic>{
          'id': id,
          'baslik': baslik,
          'soru': soru,
          'ses': sesDosyasi,
          'baslangic': DateTime.now().toIso8601String(),
        }),
        flush: true,
      );
    } catch (e) {
      // Isaret yazilamadiysa kayit yine de alinsin; yalnizca kurtarma
      // agi yok demektir.
      debugPrint('Kayit isareti yazilamadi: $e');
    }
  }

  /// Kayit duzgun bittiginde, hatira daha kaydedilmeden cagrilir. Ses
  /// dosyasi artik tamam: bicimi ne olursa olsun kurtarilabilir.
  static Future<void> tamamlandi({
    required String klasor,
    required Duration sure,
  }) async {
    try {
      final File isaret = File('$klasor/$isaretAdi');
      if (!isaret.existsSync()) return;
      final Map<String, dynamic> veri =
          json.decode(await isaret.readAsString()) as Map<String, dynamic>;
      veri['bitti'] = true;
      veri['sureMs'] = sure.inMilliseconds;
      await isaret.writeAsString(json.encode(veri), flush: true);
    } catch (e) {
      debugPrint('Kayit isareti guncellenemedi: $e');
    }
  }

  /// Hatira kaydedildikten (ya da kayit iptal edildikten) sonra cagrilir.
  static Future<void> bitti(String klasor) async {
    try {
      final File isaret = File('$klasor/$isaretAdi');
      if (isaret.existsSync()) await isaret.delete();
    } catch (e) {
      debugPrint('Kayit isareti silinemedi: $e');
    }
  }

  /// Acilista bir kez calisir. Kurtarilan hatiralari dizine ekler ve
  /// listeyi doner.
  static Future<List<Memory>> tara() async {
    final MemoryStore store = MemoryStore.instance;
    final List<Memory> kurtarilan = await klasorleriTara(
      store.memoriesDir,
      store.memories.map((Memory m) => m.id).toSet(),
    );
    for (final Memory m in kurtarilan) {
      await store.add(m);
    }
    if (kurtarilan.isNotEmpty) {
      debugPrint('${kurtarilan.length} yarim kayit kurtarildi.');
    }
    return kurtarilan;
  }

  /// [klasor] altindaki isaretli kayitlari degerlendirir. Dizinde zaten
  /// olan ([mevcutIdler]) bir kayit icin isaret bayattir, silinir.
  @visibleForTesting
  static Future<List<Memory>> klasorleriTara(
    Directory klasor,
    Set<String> mevcutIdler,
  ) async {
    final List<Memory> sonuc = <Memory>[];
    try {
      if (!klasor.existsSync()) return sonuc;

      for (final FileSystemEntity girdi in klasor.listSync()) {
        if (girdi is! Directory) continue;
        final File isaret = File('${girdi.path}/$isaretAdi');
        if (!isaret.existsSync()) continue;

        try {
          final Memory? m = await _birKlasor(girdi, isaret, mevcutIdler);
          if (m != null) sonuc.add(m);
        } catch (e) {
          debugPrint('Yarim kayit degerlendirilemedi (${girdi.path}): $e');
          await _sessizSil(isaret);
        }
      }
    } catch (e) {
      debugPrint('Yarim kayitlar taranamadi: $e');
    }
    return sonuc;
  }

  static Future<Memory?> _birKlasor(
    Directory klasor,
    File isaret,
    Set<String> mevcutIdler,
  ) async {
    final dynamic cozulen = json.decode(await isaret.readAsString());
    if (cozulen is! Map<String, dynamic>) {
      await _sessizSil(isaret);
      return null;
    }

    final String? id = cozulen['id'] as String?;
    final String? sesAdi = cozulen['ses'] as String?;
    if (id == null || id.isEmpty || sesAdi == null || sesAdi.isEmpty) {
      await _sessizSil(isaret);
      return null;
    }

    // Hatira zaten dizinde: kayit bitmis, yalnizca isaret silinememis.
    if (mevcutIdler.contains(id)) {
      await _sessizSil(isaret);
      return null;
    }

    final File ses = File('${klasor.path}/$sesAdi');
    if (!ses.existsSync() || await ses.length() < _enKucukSesBayt) {
      // Duyulacak bir sey yok; klasoru oldugu gibi kaldiralim.
      await _sessizSilKlasor(klasor);
      return null;
    }

    final bool bitti = cozulen['bitti'] == true;
    Duration sure;

    if (bitti) {
      // Ses dosyasi tamamdi, yalnizca dizine yazilamadan kapanmis.
      sure = Duration(milliseconds: (cozulen['sureMs'] as num?)?.toInt() ?? 0);
    } else {
      // Yarida kesilmis. ADTS ise son tam cerceveye kadari saglamdir.
      final AdtsBilgi? bilgi = await AdtsBilgi.oku(ses);
      if (bilgi == null || bilgi.bos) {
        // Okunamiyor (ornegin yarim kalmis bir m4a): kurtarilacak bir sey
        // yok, yarim dosya "acilmayan hatira" olarak durmasin.
        debugPrint('Yarim kayit okunamadi, atiliyor: ${ses.path}');
        await _sessizSilKlasor(klasor);
        return null;
      }
      // Sondaki yarim cerceveyi kirp: dosya bastan sona gecerli kalsin.
      final int uzunluk = await ses.length();
      if (bilgi.gecerliBayt > 0 && bilgi.gecerliBayt < uzunluk) {
        await _kirp(ses, bilgi.gecerliBayt);
      }
      sure = bilgi.sure;
    }

    await _sessizSil(isaret);

    return Memory(
      id: id,
      title: (cozulen['baslik'] as String?)?.trim().isNotEmpty == true
          ? (cozulen['baslik'] as String).trim()
          : 'Hatıra',
      createdAt:
          DateTime.tryParse(cozulen['baslangic'] as String? ?? '') ??
              ses.statSync().modified,
      audioRelPath:
          '${MemoryStore.memoriesDirName}/$id/$sesAdi',
      durationMs: sure.inMilliseconds,
      question: cozulen['soru'] as String?,
    );
  }

  static Future<void> _kirp(File dosya, int uzunluk) async {
    RandomAccessFile? f;
    try {
      f = await dosya.open(mode: FileMode.append);
      await f.truncate(uzunluk);
    } catch (e) {
      debugPrint('Yarim cerceve kirpilamadi: $e');
    } finally {
      try {
        await f?.close();
      } catch (_) {
        // Zaten kapanmis olabilir.
      }
    }
  }

  static Future<void> _sessizSil(File dosya) async {
    try {
      if (dosya.existsSync()) await dosya.delete();
    } catch (e) {
      debugPrint('Silinemedi (${dosya.path}): $e');
    }
  }

  static Future<void> _sessizSilKlasor(Directory klasor) async {
    try {
      if (klasor.existsSync()) await klasor.delete(recursive: true);
    } catch (e) {
      debugPrint('Klasor silinemedi (${klasor.path}): $e');
    }
  }
}
