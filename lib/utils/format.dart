import 'package:intl/intl.dart';

/// Tarih ve sure metinleri.
///
/// Kural: rakam yiginindan kacinilir. "02:14" yerine "2 dakika 14 saniye"
/// gibi okunabilir ifadeler tercih edilir; sayac gibi surekli degisen
/// yerlerde ise buyuk rakamlar kullanilir.
class Bicim {
  const Bicim._();

  static final DateFormat _uzunTarih = DateFormat('d MMMM y, EEEE', 'tr_TR');
  static final DateFormat _kisaTarih = DateFormat('d MMMM y', 'tr_TR');
  static final DateFormat _saat = DateFormat('HH:mm', 'tr_TR');

  static String uzunTarih(DateTime t) => _uzunTarih.format(t);

  static String saat(DateTime t) => _saat.format(t);

  /// "Bugün", "Dün" ya da "12 Eylül 2025".
  static String gunlukTarih(DateTime t) {
    final DateTime simdi = DateTime.now();
    final DateTime bugun = DateTime(simdi.year, simdi.month, simdi.day);
    final DateTime gun = DateTime(t.year, t.month, t.day);
    final int fark = bugun.difference(gun).inDays;
    if (fark == 0) return 'Bugün';
    if (fark == 1) return 'Dün';
    return _kisaTarih.format(t);
  }

  /// Sayac icin: "5:07"
  static String sayac(Duration d) {
    final int dakika = d.inMinutes;
    final String saniye = (d.inSeconds % 60).toString().padLeft(2, '0');
    return '$dakika:$saniye';
  }

  /// Liste icin: "3 dakika 12 saniye"
  static String okunurSure(Duration d) {
    final int dakika = d.inMinutes;
    final int saniye = d.inSeconds % 60;
    if (dakika == 0) return '$saniye saniye';
    if (saniye == 0) return '$dakika dakika';
    return '$dakika dakika $saniye saniye';
  }

  /// "142 MB"
  static String boyut(int bayt) {
    if (bayt <= 0) return '0 MB';
    final double mb = bayt / (1024 * 1024);
    if (mb < 1) return '1 MB’den az';
    return '${mb.toStringAsFixed(0)} MB';
  }

  /// Metnin ilk birkac kelimesi (kart onizlemesi icin).
  static String onizleme(String metin, {int enFazla = 110}) {
    final String tek = metin.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (tek.length <= enFazla) return tek;
    return '${tek.substring(0, enFazla).trimRight()}…';
  }
}
