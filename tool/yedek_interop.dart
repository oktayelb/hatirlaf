// Dart tarafinin urettigi yedekleri diske yazar; tool/yedek_interop.sh
// bunlari Python cozucusuyle acip karsilastirir.
//
// Iki taraf ayni bicimi konusmazsa yedekler acilamaz hale gelir ve bu
// ancak yillar sonra, veriye ihtiyac duyuldugunda anlasilir. O yuzden
// bicim her degistiginde bu calistirilmali.
//
//     dart run tool/yedek_interop.dart <cikti-dizini>

import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:cryptography/cryptography.dart';
import 'package:hatirla/services/backup_crypto.dart';

/// Sabit tohum: cikti belirlenimci olsun.
final List<int> aliciTohum = List<int>.filled(32, 7);

Uint8List rastgele(int n, int tohum) {
  final Random r = Random(tohum);
  return Uint8List.fromList(List<int>.generate(n, (_) => r.nextInt(256)));
}

Future<void> main(List<String> args) async {
  if (args.length != 1) {
    stderr.writeln('kullanim: dart run tool/yedek_interop.dart <cikti-dizini>');
    exit(1);
  }
  final Directory kok = Directory(args.single);
  if (!kok.existsSync()) kok.createSync(recursive: true);

  final SimpleKeyPair alici = await X25519().newKeyPairFromSeed(aliciTohum);
  final SimplePublicKey acik = await alici.extractPublicKey();

  // Gizli anahtari Python'un okuyacagi ham bicimde yaz.
  await File('${kok.path}/alici.gizli').writeAsBytes(aliciTohum);
  await File('${kok.path}/alici.acik').writeAsBytes(acik.bytes);

  // Sinir durumlari: bos, tek parca, tam parca, parca+1, cok parca.
  final Map<String, Uint8List> ornekler = <String, Uint8List>{
    'bos': Uint8List(0),
    'kucuk': rastgele(100, 1),
    'tamparca': rastgele(256, 2),
    'parcaarti': rastgele(257, 3),
    'cokparca': rastgele(4096, 4),
    'buyuk': rastgele(300000, 5),
  };

  for (final MapEntry<String, Uint8List> e in ornekler.entries) {
    // Duz metin ".kaynak" olarak duruyor; Python cozdugunde ".bin"
    // yazacak, boylece ikisini bayt bayt karsilastirabiliyoruz.
    final File duz = File('${kok.path}/${e.key}.kaynak');
    await duz.writeAsBytes(e.value, flush: true);
    await YedekSifreleme.dosyayiSifrele(
      kaynak: duz,
      hedef: File('${kok.path}/${e.key}.bin.hyz'),
      aliciAcikAnahtari: acik.bytes,
      // Kucuk parca boyu: az veriyle cok parca sinirini zorlamak icin.
      parcaBoyu: 256,
    );
    stdout.writeln('yazildi: ${e.key} (${e.value.length} bayt)');
  }
}
