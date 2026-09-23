// Uretilen yamanin telefonda gercekten calisacagini yayindan ONCE
// kanitlar: yamayi telefondaki koda -- lib/services/yama.dart -- uygulatip
// cikan APK'nin ozetini bekleneni ile karsilastirir. Python uretici ile
// Dart uygulayici birbirinden ayrilirsa yayinlama burada durur.
//
//   dart run tool/yama_dogrula.dart <eski.apk> <yama> <beklenenSha256>

import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:hatirla/services/yama.dart';

Future<void> main(List<String> argumanlar) async {
  if (argumanlar.length != 3) {
    stderr.writeln(
      'kullanim: dart run tool/yama_dogrula.dart '
      '<eski.apk> <yama> <beklenenSha256>',
    );
    exit(2);
  }

  final File kaynak = File(argumanlar[0]);
  final File yama = File(argumanlar[1]);
  final String beklenen = argumanlar[2].trim().toLowerCase();

  final Directory gecici = await Directory.systemTemp.createTemp('hatirlaf-yama');
  final File hedef = File('${gecici.path}/uretilen.apk');

  try {
    final Stopwatch sure = Stopwatch()..start();
    await Yama.uygula(kaynak: kaynak, yama: yama, hedef: hedef);
    sure.stop();

    final Digest ozet = await sha256.bind(hedef.openRead()).first;
    if (ozet.toString() != beklenen) {
      stderr.writeln('hata: yamadan cikan APK beklenen dosya degil');
      stderr.writeln('  beklenen: $beklenen');
      stderr.writeln('  cikan   : $ozet');
      exit(1);
    }

    final int boyut = await hedef.length();
    stdout.writeln(
      '    dogrulandi: ${(boyut / 1048576).toStringAsFixed(1)} MB, '
      '${sure.elapsedMilliseconds} ms',
    );
  } on FormatException catch (e) {
    stderr.writeln('hata: yama uygulanamadi: ${e.message}');
    exit(1);
  } finally {
    try {
      await gecici.delete(recursive: true);
    } catch (_) {
      // Gecici dizin kalsa da yayinlamayi engellemez.
    }
  }
}
