import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:uuid/uuid.dart';

import '../models/memory.dart';
import 'b2_client.dart';
import 'backup_crypto.dart';
import 'backup_info.dart';
import 'network.dart';
import 'recorder.dart';
import 'store.dart';
import 'transcriber.dart';

enum YedekAsamasi { bos, sifreliyor, yukleniyor, bekliyor, hata }

/// Sifrelemeyi ayri bir isolate'te yapar.
///
/// 300 MB'lik bir kaydin saf Dart AES-GCM ile sifrelenmesi saniyeler
/// suruyor; ana isolate'te yapilsaydi kullanici bu sure boyunca donmus
/// bir ekran gorurdu.
///
/// Yalnizca `String` ve `List<int>` yakalaniyor: [Isolate.run] kapanista
/// gonderilemeyen bir nesne (ornegin [File]) yakalanirsa calisma
/// aninda patlar.
Future<void> _sifreleAyri({
  required String kaynak,
  required String hedef,
  required List<int> anahtar,
}) {
  return Isolate.run(
    () => YedekSifreleme.dosyayiSifrele(
      kaynak: File(kaynak),
      hedef: File(hedef),
      aliciAcikAnahtari: anahtar,
    ),
  );
}

/// Hatiralari sifreleyip B2'ye kopyalar.
///
/// Uc kural:
///
/// 1. Telefondaki asil kayit ASLA silinmez. Yukleme bir kopyadir;
///    gizli anahtar kaybolursa tek kurtulus odur.
/// 2. Kullaniciya hicbir sey sorulmaz, hicbir hata gosterilmez. Yasli
///    bir kullanicinin "yukleme basarisiz" uyarisiyla yapabilecegi bir
///    sey yok; kuran kisi Ayarlar'dan gorur.
/// 3. Varsayilan olarak yalnizca kablosuz agda calisir. Kayitlar
///    guncellemeden cok daha buyuk.
class Yedekleyici extends ChangeNotifier {
  Yedekleyici._();

  static final Yedekleyici instance = Yedekleyici._();

  static const String _pCihaz = 'yedek_cihaz';
  static const String _pSahip = 'yedek_sahip';
  static const String _pYuklenen = 'yedek_yuklenen';
  static const String _pSonDeneme = 'yedek_son_deneme';

  /// Ust uste hata alindiginda beklenecek sure. Sunucuyu da telefonun
  /// pilini de bosuna yormayalim.
  static const Duration _hataBeklemesi = Duration(hours: 2);

  YedekAsamasi _asama = YedekAsamasi.bos;
  String? _hata;
  String _cihaz = '';
  String _sahip = '';
  Set<String> _yuklenen = <String>{};
  bool _mesgul = false;
  int _kalan = 0;
  double? _ilerleme;
  DateTime? _sonDeneme;

  YedekAsamasi get asama => _asama;
  String? get hata => _hata;
  String get cihaz => _cihaz;

  /// Telefonu kuran kisinin yazdigi ad ("Dedem Ahmet"). Bos olabilir.
  String get sahip => _sahip;

  /// Kovadaki klasor adi: okunabilir ad + rastgele kimlik.
  ///
  /// Ikisi birlikte: ad olmadan 30 tane anlamsiz klasor olurdu, kimlik
  /// olmadan ayni adi tasiyan iki telefon birbirinin uzerine yazardi.
  String get klasorAdi => _sahip.trim().isEmpty
      ? _cihaz
      : '${asciiyeIndir(_sahip.trim())}-$_cihaz';

  /// Turkce harfleri ASCII karsiliklarina indirir.
  ///
  /// B2 dosya adinda guvensiz her karakter alt cizgiye donuyor
  /// ([YedekAyarlari.dosyaAdi]); dokunmasak "Şükrü Dedem" kovada
  /// "_ukr__Dedem" olurdu. Yalnizca bu harfleri degistiriyoruz: zaten
  /// ASCII olan adlar aynen kaliyor, yani sahadaki telefonlarin klasoru
  /// yerinden oynamiyor.
  static String asciiyeIndir(String s) {
    const Map<String, String> harfler = <String, String>{
      'ç': 'c', 'Ç': 'C',
      'ğ': 'g', 'Ğ': 'G',
      'ı': 'i', 'İ': 'I',
      'ö': 'o', 'Ö': 'O',
      'ş': 's', 'Ş': 'S',
      'ü': 'u', 'Ü': 'U',
    };
    final StringBuffer b = StringBuffer();
    for (final String harf in s.split('')) {
      b.write(harfler[harf] ?? harf);
    }
    return b.toString();
  }

  /// Kuran kisi telefonu teslim ederken bir kez yazar.
  Future<void> sahibiKaydet(String ad) async {
    _sahip = ad.trim();
    final SharedPreferences ayarlar = await SharedPreferences.getInstance();
    await ayarlar.setString(_pSahip, _sahip);
    notifyListeners();
  }
  int get bekleyenSayisi => _kalan;
  double? get ilerleme => _ilerleme;
  DateTime? get sonDeneme => _sonDeneme;
  int get yuklenenSayisi => _yuklenen.length;

  /// Ayarlar eksikse yedekleme tamamen kapali.
  bool get acikMi => YedekAyarlari.kurulu;

  /// Telefon su an daha onemli bir isle mesgul mu?
  static bool get mesgulMu =>
      Recorder.instance.durum != KayitDurumu.bos || Transcriber.instance.mesgul;

  B2Istemcisi? _istemci;

  Future<void> baslat() async {
    // Istemci Flutter'a bagli degil (PC'den denenebilsin diye);
    // uyarilarini burada uygulamanin gunlugune bagliyoruz.
    //
    // Sarmalayici, `as` ile donusturmek yerine: debugPrint'in imzasi
    // `void Function(String?, {int? wrapWidth})`. Bugun cast calisiyor
    // ama imza degisirse acilista patlar ve baslat() unawaited
    // cagrildigi icin yedekleme SESSIZCE hic baslamaz.
    b2Uyari = (String mesaj) => debugPrint(mesaj);

    final SharedPreferences ayarlar = await SharedPreferences.getInstance();
    _cihaz = ayarlar.getString(_pCihaz) ?? '';
    if (_cihaz.isEmpty) {
      // Cihaz kimligi rastgele: telefonun gercek kimligini toplamak
      // gerekmiyor, yalnizca dosyalarin karismamasi gerekiyor.
      _cihaz = const Uuid().v4().substring(0, 8);
      await ayarlar.setString(_pCihaz, _cihaz);
    }
    _sahip = ayarlar.getString(_pSahip) ?? '';
    _yuklenen = (ayarlar.getStringList(_pYuklenen) ?? <String>[]).toSet();
    final int? damga = ayarlar.getInt(_pSonDeneme);
    if (damga != null) {
      final DateTime d = DateTime.fromMillisecondsSinceEpoch(damga);
      // Telefonun saati geri alinabiliyor; gelecege ait damgaya guvenme.
      _sonDeneme = d.isAfter(DateTime.now()) ? null : d;
    }
    _kalan = _bekleyenler().length;
    notifyListeners();

    if (!acikMi) return;

    Ag.instance.basla();
    // Sakinlesmis gecisler: Wi-Fi'ye baglanirken gelen art arda
    // bildirimlerin her biri yukleme baslatmasin.
    _agAbonelik ??= Ag.instance.degisim.listen((AgDurumu d) {
      if (d.internetVar) unawaited(degerlendir());
    });
    // Yeni kayit ya da biten cevirme de sirayi buyutuyor.
    MemoryStore.instance.addListener(_depoDegisti);

    unawaited(degerlendir());
  }

  StreamSubscription<AgDurumu>? _agAbonelik;

  void _depoDegisti() {
    if (_mesgul) return;
    unawaited(degerlendir());
  }

  /// Henuz yuklenmemis, sesi diskte duran hatiralar.
  List<Memory> _bekleyenler() {
    return MemoryStore.instance.memories.where((Memory m) {
      if (_yuklenen.contains(m.id)) return false;
      final String yol = MemoryStore.instance.absolute(m.audioRelPath);
      return File(yol).existsSync();
    }).toList();
  }

  /// Kosullar uygunsa bekleyenleri yukler. Cagrilmasi ucuzdur;
  /// uygun degilse sessizce doner.
  Future<void> degerlendir({bool elle = false}) async {
    if (!acikMi || _mesgul) return;

    // Uygulamanin asil isi kayit almak. Yukleme (isolate'te sifreleme +
    // ag) yasli bir telefonda kaydi kekeletebilir; sira bekleyebilir,
    // kayit bekleyemez.
    if (mesgulMu) {
      _durumaGec(YedekAsamasi.bekliyor);
      return;
    }

    // Ucuz kontroller once: depo her cevirme adiminda haber veriyor ve
    // _bekleyenler() hatira basina bir disk erisimi demek.
    if (!elle) {
      final AgDurumu ag = Ag.instance.durum;
      if (!ag.internetVar) return;
      if (YedekAyarlari.yalnizcaKablosuz && !ag.sayacsiz) {
        _durumaGec(YedekAsamasi.bekliyor);
        return;
      }
      final DateTime? son = _sonDeneme;
      if (_asama == YedekAsamasi.hata &&
          son != null &&
          DateTime.now().difference(son) < _hataBeklemesi) {
        return;
      }
    }

    final List<Memory> sira = _bekleyenler();
    _kalan = sira.length;
    if (sira.isEmpty) {
      _durumaGec(YedekAsamasi.bos);
      return;
    }

    _mesgul = true;
    try {
      for (final Memory m in sira) {
        if (mesgulMu) break;
        if (!elle && !Ag.instance.durum.internetVar) break;
        final bool oldu = await _hatirayiYukle(m);
        if (!oldu) break;
        _kalan = _bekleyenler().length;
        notifyListeners();
      }
      if (_asama != YedekAsamasi.hata) {
        _durumaGec(_kalan == 0 ? YedekAsamasi.bos : YedekAsamasi.bekliyor);
      }
    } finally {
      _mesgul = false;
      _ilerleme = null;
      await _damgala();
      notifyListeners();
    }
  }

  /// Tek bir hatirayi sifreleyip yukler. Basarili olduysa `true`.
  Future<bool> _hatirayiYukle(Memory m) async {
    final Directory gecici = await _geciciKlasor();
    final File ses = File(MemoryStore.instance.absolute(m.audioRelPath));
    final File sesSifreli = File('${gecici.path}/${m.id}-ses.hyz');
    final File bilgiSifreli = File('${gecici.path}/${m.id}-bilgi.hyz');
    final File bilgiDuz = File('${gecici.path}/${m.id}-bilgi.json');

    try {
      final List<int>? anahtar = YedekAyarlari.aliciAcikAnahtari;
      if (anahtar == null) {
        _hataylaDur('Alici acik anahtari gecersiz.');
        return false;
      }

      _durumaGec(YedekAsamasi.sifreliyor);

      // Metin de sifrelenir: bir hayat hikayesinin yaziya dokulmus hali
      // sesinden daha az hassas degil, aranabilir oldugu icin daha fazla.
      await bilgiDuz.writeAsString(
        json.encode(<String, dynamic>{
          ...m.toJson(),
          'cihaz': _cihaz,
          'sahip': _sahip,
          'yuklendi': DateTime.now().toIso8601String(),
        }),
        flush: true,
      );

      await _sifreleAyri(
        kaynak: ses.path,
        hedef: sesSifreli.path,
        anahtar: anahtar,
      );
      await _sifreleAyri(
        kaynak: bilgiDuz.path,
        hedef: bilgiSifreli.path,
        anahtar: anahtar,
      );

      _durumaGec(YedekAsamasi.yukleniyor);
      final B2Istemcisi c = _istemciAl();

      await c.yukle(
        dosya: sesSifreli,
        ad: YedekAyarlari.dosyaAdi(
          cihaz: klasorAdi,
          hatiraId: m.id,
          dosya: 'ses.m4a',
        ),
        bilgiler: <String, String>{'hatira': m.id, 'cihaz': klasorAdi},
        ilerleme: (int y, int t) {
          _ilerleme = t > 0 ? y / t : null;
          notifyListeners();
        },
        // Yukleme sirasinda kayit baslarsa parcalar arasinda birakip
        // cikiyoruz; yarim kalan hatira isaretlenmedigi icin sonra
        // bastan denenir.
        devamEdilsinMi: () async =>
            !mesgulMu &&
            Ag.instance.durum.internetVar &&
            (!YedekAyarlari.yalnizcaKablosuz || Ag.instance.durum.sayacsiz),
      );

      await c.yukle(
        dosya: bilgiSifreli,
        ad: YedekAyarlari.dosyaAdi(
          cihaz: klasorAdi,
          hatiraId: m.id,
          dosya: 'bilgi.json',
        ),
        bilgiler: <String, String>{'hatira': m.id, 'cihaz': klasorAdi},
      );

      // Ses VE bilgi gectikten sonra isaretle: yarim yuklenmis bir
      // hatira bir daha denenmeli.
      await _isaretle(m.id);
      _hata = null;
      return true;
    } on B2Hatasi catch (e) {
      _hataylaDur('B2: ${e.mesaj}');
      return false;
    } catch (e) {
      _hataylaDur('$e');
      return false;
    } finally {
      // Sifreli kopyalar gecici; asil kayit yerinde duruyor.
      for (final File f in <File>[sesSifreli, bilgiSifreli, bilgiDuz]) {
        try {
          if (f.existsSync()) await f.delete();
        } catch (_) {
          // Silinemezse bir sonraki temizlikte gider.
        }
      }
    }
  }

  B2Istemcisi _istemciAl() {
    return _istemci ??= B2Istemcisi(
      anahtarKimligi: YedekAyarlari.b2AnahtarKimligi,
      anahtar: YedekAyarlari.b2Anahtari,
      kovaKimligi: YedekAyarlari.b2KovaKimligi,
    );
  }

  Future<Directory> _geciciKlasor() async {
    final Directory kok = await getApplicationSupportDirectory();
    final Directory d = Directory('${kok.path}/yedek');
    if (!d.existsSync()) d.createSync(recursive: true);
    return d;
  }

  Future<void> _isaretle(String id) async {
    _yuklenen.add(id);
    final SharedPreferences ayarlar = await SharedPreferences.getInstance();
    await ayarlar.setStringList(_pYuklenen, _yuklenen.toList());
  }

  Future<void> _damgala() async {
    _sonDeneme = DateTime.now();
    final SharedPreferences ayarlar = await SharedPreferences.getInstance();
    await ayarlar.setInt(_pSonDeneme, _sonDeneme!.millisecondsSinceEpoch);
  }

  void _durumaGec(YedekAsamasi a) {
    if (_asama == a) return;
    _asama = a;
    notifyListeners();
  }

  void _hataylaDur(String mesaj) {
    // Kullaniciya gosterilmez; Ayarlar'da kuran kisi icin duruyor.
    debugPrint('Yedekleme hatasi: $mesaj');
    _hata = mesaj;
    _asama = YedekAsamasi.hata;
    notifyListeners();
  }

  /// Ayarlar ekranindaki "yeniden dene" icin.
  Future<void> tekrarDene() async {
    _hata = null;
    _asama = YedekAsamasi.bos;
    await degerlendir(elle: true);
  }

  @visibleForTesting
  void testIcinKur({
    required String cihaz,
    Set<String>? yuklenen,
    B2Istemcisi? istemci,
  }) {
    _cihaz = cihaz;
    _yuklenen = yuklenen ?? <String>{};
    _istemci = istemci;
  }
}
