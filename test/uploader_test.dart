import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/services/b2_client.dart';
import 'package:hatirla/services/uploader.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues(<String, Object>{}));

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

  group('gunluk koprusu', () {
    test('b2Uyari uygulamanin gunlugune baglaniyor', () async {
      await Yedekleyici.instance.baslat();
      // Cagrilabilir olmasi yeterli: eskiden burada `as` ile bir
      // donusturme vardi ve imza degisse acilis patlardi.
      expect(() => b2Uyari('deneme'), returnsNormally);
    });
  });
}
