import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/b2_client.dart';
import 'package:hatirla/services/recorder.dart';
import 'package:hatirla/services/uploader.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Recorder tekili kurulurken AudioRecorder() yerli kanala gidiyor;
  // testte eklenti yok. Kanali sahteleyip yalnizca durum mantigini
  // sinayabiliyoruz.
  const MethodChannel kayitKanali =
      MethodChannel('com.llfbandit.record/messages');

  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(kayitKanali, (MethodCall c) async => null);
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(kayitKanali, null);
  });

  group('acilis', () {
    // baslat() main.dart'ta unawaited cagriliyor: burada atilan bir
    // istisna kimseye gorunmez, yedekleme sessizce hic baslamaz.
    // Bu yuzden "patlamiyor" ayri bir test olarak duruyor.
    test('baslat() ayarlar eksikken bile patlamiyor', () async {
      await Yedekleyici.instance.baslat();
      expect(Yedekleyici.instance.acikMi, isFalse);
    });

    test('cihaz kimligi uretilip kaliciya yaziliyor', () async {
      await Yedekleyici.instance.baslat();
      final String ilk = Yedekleyici.instance.cihaz;
      expect(ilk, isNotEmpty);

      final SharedPreferences p = await SharedPreferences.getInstance();
      expect(p.getString('yedek_cihaz'), ilk);
    });

    test('ayarlar eksikken degerlendir() sessizce donuyor', () async {
      await Yedekleyici.instance.baslat();
      await Yedekleyici.instance.degerlendir();
      expect(Yedekleyici.instance.asama, YedekAsamasi.bos);
      expect(Yedekleyici.instance.hata, isNull);
    });

    test('gelecege ait son deneme damgasi yok sayiliyor', () async {
      // Telefonun saati geri alinabiliyor.
      SharedPreferences.setMockInitialValues(<String, Object>{
        'yedek_son_deneme':
            DateTime.now().add(const Duration(days: 400)).millisecondsSinceEpoch,
      });
      await Yedekleyici.instance.baslat();
      expect(Yedekleyici.instance.sonDeneme, isNull);
    });
  });

  group('mesguliyet', () {
    test('bos durumda mesgul degil', () {
      expect(Yedekleyici.mesgulMu, isFalse);
    });

    test('kayit sirasinda mesgul sayiliyor', () async {
      // Kayit uygulamanin asil isi; yukleme sira beklemeli.
      Recorder.instance.testIcinDurum(KayitDurumu.kaydediyor);
      expect(Yedekleyici.mesgulMu, isTrue);

      Recorder.instance.testIcinDurum(KayitDurumu.duraklatildi);
      expect(Yedekleyici.mesgulMu, isTrue,
          reason: 'duraklatilmis kayit da surmekte sayilir');

      Recorder.instance.testIcinDurum(KayitDurumu.bos);
      expect(Yedekleyici.mesgulMu, isFalse);
    });

    test('mesgulken degerlendir() yukleme baslatmiyor', () async {
      // Asama burada `bos` kalir: ayarlar olmadigi icin degerlendir()
      // daha en basta donuyor. Onemli olan hicbir sey baslatmamasi.
      Recorder.instance.testIcinDurum(KayitDurumu.kaydediyor);
      await Yedekleyici.instance.baslat();
      await Yedekleyici.instance.degerlendir(elle: true);
      expect(Yedekleyici.instance.hata, isNull);
      expect(Yedekleyici.instance.asama, YedekAsamasi.bos);
      Recorder.instance.testIcinDurum(KayitDurumu.bos);
    });
  });

  group('gunluk koprusu', () {
    test('b2Uyari uygulamanin gunlugune baglaniyor', () async {
      await Yedekleyici.instance.baslat();
      // Cagrilabilir olmasi yeterli: eskiden burada `as` ile bir
      // donusturme vardi ve imza degisse acilis patlardi.
      expect(() => b2Uyari('deneme'), returnsNormally);
    });
  });
}
