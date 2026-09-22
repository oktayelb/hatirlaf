// yedek.json'daki adlar uygulamanin bekledigi adlarla ortusuyor mu?
//
// Ortusmezse yedekleme SESSIZCE kapali kalir: uygulama calisir, kimse
// bir sey fark etmez, hicbir kayit yukselmez. Derlemeden once burada
// yakalanmali.
//
//     dart run --define=... tool/yedek_ayar_dogrula.dart
//
// Cagrilma bicimi: tool/yedek_onkontrol.sh
import 'dart:io';

import 'package:hatirla/services/backup_info.dart';

void main() {
  stdout.writeln('  kurulu        : ${YedekAyarlari.kurulu}');
  stdout.writeln('  eksiklik      : ${YedekAyarlari.eksiklik ?? "yok"}');
  stdout.writeln('  anahtar uzunl : '
      '${YedekAyarlari.aliciAcikAnahtari?.length ?? 0} bayt (32 olmali)');
  stdout.writeln('  kova          : ${YedekAyarlari.b2KovaKimligi}');
  stdout.writeln('  ornek ad      : ${YedekAyarlari.dosyaAdi(
    cihaz: 'abc12345',
    hatiraId: 'hatira-1',
    dosya: 'ses.m4a',
  )}');
  if (!YedekAyarlari.kurulu) {
    stderr.writeln('HATA: yedekleme KAPALI cikardi.');
    exit(1);
  }
}
