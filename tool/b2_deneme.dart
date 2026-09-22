// Gercek B2'ye karsi uctan uca deneme: telefonun yaptigi isin aynisi.
//
//     dart run tool/b2_deneme.dart <cikti-dizini> [bayt]
//
// Uretimdeki kodun ta kendisini kullanir (YedekSifreleme + B2Istemcisi),
// benzetim degil. tool/b2_deneme.sh bunu cagirip sonra Python tarafiyla
// indirip cozuyor.

import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:hatirla/services/b2_client.dart';
import 'package:hatirla/services/backup_crypto.dart';
import 'package:hatirla/services/backup_info.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln(
      'kullanim: dart run tool/b2_deneme.dart <dizin> [bayt] [cihaz]',
    );
    exit(1);
  }
  final Directory kok = Directory(args[0]);
  if (!kok.existsSync()) kok.createSync(recursive: true);
  final int boyut = args.length > 1 ? int.parse(args[1]) : 3 * 1024 * 1024;

  // yedek.json: telefona gidecek ayarlarin aynisi.
  final File ayarDosyasi = File('yedek.json');
  if (!ayarDosyasi.existsSync()) {
    stderr.writeln('hata: yedek.json yok. Once tool/b2_kur.py calistirin.');
    exit(1);
  }
  final Map<String, dynamic> ayar =
      json.decode(await ayarDosyasi.readAsString()) as Map<String, dynamic>;

  final List<int>? acikAnahtar =
      YedekAyarlari.acikAnahtariCoz(ayar['YEDEK_ALICI_ANAHTARI'] as String);
  if (acikAnahtar == null) {
    stderr.writeln('hata: YEDEK_ALICI_ANAHTARI 32 baytlik base64 degil.');
    exit(1);
  }

  // Ornek "kayit": rastgele ama belirlenimci.
  final Random r = Random(7);
  final Uint8List veri =
      Uint8List.fromList(List<int>.generate(boyut, (_) => r.nextInt(256)));
  final File duz = File('${kok.path}/kaynak.m4a');
  await duz.writeAsBytes(veri, flush: true);
  stdout.writeln('  ornek kayit: ${veri.length} bayt');

  final File sifreli = File('${kok.path}/kaynak.m4a.hyz');
  final Stopwatch kronometre = Stopwatch()..start();
  await YedekSifreleme.dosyayiSifrele(
    kaynak: duz,
    hedef: sifreli,
    aliciAcikAnahtari: acikAnahtar,
  );
  kronometre.stop();
  stdout.writeln(
    '  sifrelendi : ${await sifreli.length()} bayt, '
    '${kronometre.elapsedMilliseconds} ms',
  );
  // Onceden hesaplanan boyut gerceklesenle tutmali.
  final int beklenen = YedekSifreleme.sifreliBoyut(veri.length);
  if (await sifreli.length() != beklenen) {
    stderr.writeln('hata: boyut tutmadi ($beklenen beklenmisti)');
    exit(1);
  }

  final B2Istemcisi istemci = B2Istemcisi(
    anahtarKimligi: ayar['B2_KEY_ID'] as String,
    anahtar: ayar['B2_APP_KEY'] as String,
    kovaKimligi: ayar['B2_BUCKET_ID'] as String,
  );

  try {
    final B2Oturumu o = await istemci.oturum();
    stdout.writeln('  yetkiler   : ${o.yetkiler}');
    if (o.fazlaYetkiliMi) {
      stderr.writeln('hata: anahtar okuma/silme yetkisi tasiyor, durduruldu.');
      exit(1);
    }

    // Ontanimli "_deneme": yedek_indir.py bu oneki atlar, yani normal
    // indirmeler deneme cop'uyle karismaz. Indirme yolunu sinamak icin
    // gercekci bir cihaz adi verilebilir.
    final String ad = YedekAyarlari.dosyaAdi(
      cihaz: args.length > 2 ? args[2] : '_deneme',
      hatiraId: DateTime.now().millisecondsSinceEpoch.toString(),
      dosya: 'ses.m4a',
    );

    final Stopwatch yuklemeSuresi = Stopwatch()..start();
    final B2Dosyasi d = await istemci.yukle(
      dosya: sifreli,
      ad: ad,
      bilgiler: <String, String>{'deneme': 'evet'},
      ilerleme: (int y, int t) {
        if (y == t) stdout.writeln('  yuklendi   : $y/$t bayt');
      },
    );
    yuklemeSuresi.stop();
    stdout.writeln(
      '  B2 dosyasi : ${d.ad}  (${yuklemeSuresi.elapsedMilliseconds} ms)',
    );

    // Kabuk tarafinin indirebilmesi icin adi yaz.
    await File('${kok.path}/nesne.txt').writeAsString('${d.ad}\n');
  } finally {
    istemci.kapat();
  }
}
