import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'network.dart';
import 'update_info.dart';

/// Guncelleme surecinin bulundugu asama.
enum GuncellemeAsamasi {
  /// Bilinen bir guncelleme yok. Kullaniciya hicbir sey gosterilmez.
  bos,

  /// `guncelleme.json` okunuyor.
  denetleniyor,

  /// Yeni surum var, APK iniyor. Yasli kullaniciya **gosterilmez**.
  indiriliyor,

  /// APK indi, ozeti dogrulandi, kurulmaya hazir.
  hazir,

  /// Sistemin kurulum penceresi acildi.
  kuruluyor,

  /// Bir sey ters gitti. Yasli kullaniciya gosterilmez; Ayarlar'da durur.
  hata,
}

/// Uygulamayi kendi kendine gunceller.
///
/// ## Tasarim
///
/// Uygulama magazasi yok; guncelleme GitHub'daki bir JSON dosyasi ve bir
/// surum ekinden ibaret. Onemli olan kisim teknik degil, davranissal:
///
/// **Denetim ve indirme tamamen gorunmezdir.** Yasli kullanici ne bir
/// ilerleme cubugu, ne "internete bağlanılamadı" uyarisi, ne de iptal
/// edebilecegi bir islem gorur. Internet yoksa hicbir sey olmaz ve hicbir
/// sey soylenmez. Ancak APK inip ozeti dogrulandiktan **sonra**, yani
/// geriye yalnizca iki dokunus kaldiginda bir kez soru sorulur. "Sonra"
/// denirse gunlerce bir daha sorulmaz.
///
/// Sebebi basit: bu kullanicilar yarida kalan bir islemi kendileri
/// kurtaramaz. Gosterilen her ilerleme cubugu, iptal edilebilen her islem
/// ve anlasilmayan her hata bir telefon gorusmesi demek.
///
/// ## Kurulumun siniri
///
/// Sessiz kurulum mumkun degil: cihaz sahibi (device owner) olarak
/// saglanmamis bir uygulama sistemin onay penceresini gostermek zorunda.
/// Elimizden gelen, o pencereye kadar olan her seyi halletmek.
class Guncelleyici extends ChangeNotifier {
  Guncelleyici._();

  static final Guncelleyici instance = Guncelleyici._();

  static const MethodChannel _kanal = MethodChannel('hatirla/guncelleme');

  // Ayarlarda saklananlar.
  static const String _pSonDenetim = 'guncelleme_son_denetim';
  static const String _pSonBilgi = 'guncelleme_son_bilgi';

  static const String _klasorAdi = 'guncelleme';

  /// `guncelleme.json` icin ust sinir. Dosya birkac yuz bayt; bunun
  /// otesi bir yanlislik demek.
  static const int _enBuyukBilgiBayt = 16 * 1024;

  // --------------------------------------------------------------- durum

  int _mevcutSurumKodu = 0;
  int get mevcutSurumKodu => _mevcutSurumKodu;

  String _mevcutSurumAdi = '';
  String get mevcutSurumAdi => _mevcutSurumAdi;

  /// Cihazin calistirabilecegi mimariler, tercih sirasiyla. Hangi APK'nin
  /// indirilecegini bu belirliyor.
  List<String> _abiler = const <String>[];
  List<String> get abiler => _abiler;

  GuncellemeAsamasi _asama = GuncellemeAsamasi.bos;
  GuncellemeAsamasi get asama => _asama;

  GuncellemeBilgisi? _bilgi;
  GuncellemeBilgisi? get bilgi => _bilgi;

  double? _ilerleme;
  double? get ilerleme => _ilerleme;

  int _inenBayt = 0;
  int get inenBayt => _inenBayt;

  int _toplamBayt = 0;
  int get toplamBayt => _toplamBayt;

  /// Teknik hata metni. Yalnizca Ayarlar'da, kuran kisi icin gosterilir.
  String? _hata;
  String? get hata => _hata;

  /// Kurulum bir imza uyusmazligiyla reddedildi mi?
  ///
  /// Yasli kullanicinin cozemeyecegi tek hata: tek cikis yolu uygulamayi
  /// silmek, o da hatiralari siler. Bu yuzden ona asla "silip yeniden
  /// kurun" denmez, "aileden biri yardım etsin" denir.
  bool _imzaUyusmazligi = false;
  bool get imzaUyusmazligi => _imzaUyusmazligi;

  /// Kullanici sistemin kurulum penceresinde "Vazgeç" dedi mi?
  ///
  /// Guncelleme zorunlu oldugu icin ekran yerinde kaliyor; bu bayrak
  /// olmasa kullanici ayni ekrana hicbir aciklama olmadan geri donerdi.
  bool _iptalEdildi = false;
  bool get iptalEdildi => _iptalEdildi;

  DateTime? _sonDenetim;
  DateTime? get sonDenetim => _sonDenetim;

  bool _kurulumIzniVar = true;
  bool get kurulumIzniVar => _kurulumIzniVar;

  bool _basladi = false;
  bool _mesgul = false;
  bool _indirmeIptal = false;
  HttpClient? _istemci;
  StreamSubscription<AgDurumu>? _agAbonelik;
  String? _klasorYolu;

  /// Guncelleme, kullanici uygulamayi actiginda **zaten** hazir miydi?
  ///
  /// Kullanimin ortasinda inen bir guncelleme ekrani basmasin diye var:
  /// hatirasina bakan biri, birden tam ekran bir soruyla karsilasmasin.
  /// Oyle bir durumda APK diskte bekler, soru bir sonraki acilista sorulur.
  bool _acilistaHazirdi = false;
  bool get acilistaHazirdi => _acilistaHazirdi;

  /// Uygulama yeniden on plana geldi: bu da "yeni bir oturum" sayilir,
  /// yani bekleyen guncelleme artik sorulabilir.
  void oturumaGirildi() {
    if (_asama == GuncellemeAsamasi.hazir && !_acilistaHazirdi) {
      _acilistaHazirdi = true;
      notifyListeners();
    }
  }

  /// Kurulmaya hazir, dogrulanmis bir APK var mi?
  bool get hazirApkVar =>
      _asama == GuncellemeAsamasi.hazir ||
      _asama == GuncellemeAsamasi.kuruluyor;

  /// Kullaniciya su anda guncelleme gosterilmeli mi?
  ///
  /// Guncelleme zorunlu: erteleme yok, "sonra" yok. Hazirsa gosterilir.
  bool get sorulabilir =>
      _asama == GuncellemeAsamasi.hazir && _bilgi != null;

  // -------------------------------------------------------------- baslat

  /// Uygulama acilisinda bir kez cagrilir. Hicbir kosulda firlatmaz ve
  /// acilisi bekletmez.
  Future<void> baslat() async {
    if (_basladi) return;
    _basladi = true;

    try {
      _kanal.setMethodCallHandler(_yerliCagri);

      final Map<Object?, Object?>? surum =
          await _kanal.invokeMapMethod<Object?, Object?>('surum');
      _mevcutSurumKodu = (surum?['surumKodu'] as num?)?.toInt() ?? 0;
      _mevcutSurumAdi = (surum?['surumAdi'] as String?) ?? '';
      _abiler = (surum?['abiler'] as List<Object?>?)
              ?.whereType<String>()
              .toList(growable: false) ??
          const <String>[];
      if (_mevcutSurumKodu <= 0 || _abiler.isEmpty) return;

      final SharedPreferences ayarlar = await SharedPreferences.getInstance();
      final int? denetimMs = ayarlar.getInt(_pSonDenetim);
      _sonDenetim = denetimMs == null
          ? null
          : DateTime.fromMillisecondsSinceEpoch(denetimMs);
      // Son bilinen surum bilgisi diskten geri yuklenir. Boylece dun
      // Wi-Fi'de inmis bir guncelleme, bugun internet hic olmasa bile
      // kurulabilir; internet yalnizca *indirmek* icin gerekli.
      final String? sonBilgi = ayarlar.getString(_pSonBilgi);
      if (sonBilgi != null) {
        final GuncellemeBilgisi? b =
            GuncellemeBilgisi.cozumle(sonBilgi, _abiler);
        if (b != null && b.surumKodu > _mevcutSurumKodu) _bilgi = b;
      }

      await _eskiDosyalariTemizle();

      // Onceki oturumda inmis ve dogrulanmis bir APK var mi?
      final GuncellemeBilgisi? b = _bilgi;
      if (b != null && await _diskteHazirMi(b)) {
        // Onceki oturumdan kalmis: kullanici uygulamayi actiginda zaten
        // hazirdi, yani sormak icin dogru an.
        _hazirla(b, acilista: true);
      }

      Ag.instance.basla();
      // Uygulama acikken eve girip Wi-Fi'ye baglanan kullanici da
      // guncellemeyi alabilsin.
      _agAbonelik = Ag.instance.degisim.listen((AgDurumu d) {
        if (d.internetVar) unawaited(degerlendir());
      });

      notifyListeners();
      unawaited(degerlendir());
    } on MissingPluginException {
      // Android disi platform ya da test ortami: guncelleme kapali kalsin.
      debugPrint('Guncelleme kanali yok; guncelleme kapali.');
    } catch (e, s) {
      debugPrint('Guncelleyici baslatilamadi: $e\n$s');
    }
  }

  // --------------------------------------------------------- degerlendir

  /// "Su anda ne yapmaliyim?" sorusunun tek cevap yeri.
  ///
  /// Acilista, ag degisiminde ve Ayarlar'daki elle denetimde cagrilir.
  /// [elle] `true` ise ag ve sure kisitlarini atlar: Ayarlar'a girip
  /// dugmeye basan kisi zaten yasli kullanici degil, ona yardim eden biri.
  Future<void> degerlendir({bool elle = false}) async {
    if (_mesgul || _mevcutSurumKodu <= 0) return;

    _mesgul = true;
    try {
      // Kurulmayi bekleyen ya da kurulmakta olan bir sey varsa karismayalim.
      if (_asama == GuncellemeAsamasi.kuruluyor) return;
      if (_asama == GuncellemeAsamasi.hazir && !elle) return;

      final AgDurumu ag = Ag.instance.durum;

      final bool denetimGerek = elle ||
          _bilgi == null ||
          GuncellemePolitikasi.denetimZamaniGeldiMi(
            sonDenetim: _sonDenetim,
            simdi: DateTime.now(),
          );

      if (denetimGerek) {
        if (elle && !ag.internetVar) {
          _hataKur('İnternet bağlantısı yok.');
          return;
        }
        if (!elle && !ag.internetVar) return;
        if (!await _denetle()) return;
      }

      final GuncellemeBilgisi? b = _bilgi;
      if (b == null || b.surumKodu <= _mevcutSurumKodu) {
        _bosaAl();
        return;
      }

      if (await _diskteHazirMi(b)) {
        _hazirla(b);
        return;
      }

      if (!elle && !ag.internetVar) return;
      await _indir(b, elle: elle);
    } finally {
      _mesgul = false;
    }
  }

  // ------------------------------------------------------------- denetle

  /// `guncelleme.json`'u okur. Basarisizsa **sessizce** `false` doner.
  Future<bool> _denetle() async {
    if (_asama != GuncellemeAsamasi.hazir) {
      _asama = GuncellemeAsamasi.denetleniyor;
      _hata = null;
      notifyListeners();
    }

    HttpClient? istemci;
    try {
      istemci = HttpClient()
        ..connectionTimeout = const Duration(seconds: 20)
        ..idleTimeout = const Duration(seconds: 20);

      final HttpClientRequest istek =
          await istemci.getUrl(GuncellemeKaynagi.bilgiAdresi());
      istek.headers.set(HttpHeaders.cacheControlHeader, 'no-cache');
      final HttpClientResponse yanit = await istek.close();

      if (yanit.statusCode != HttpStatus.ok) {
        throw HttpException('Sunucu ${yanit.statusCode} yanıtı verdi');
      }

      final List<int> bayt = <int>[];
      await for (final List<int> parca in yanit) {
        bayt.addAll(parca);
        if (bayt.length > _enBuyukBilgiBayt) {
          throw const FormatException('guncelleme.json beklenenden büyük');
        }
      }

      final GuncellemeBilgisi? okunan = GuncellemeBilgisi.cozumle(
        utf8.decode(bayt, allowMalformed: true),
        _abiler,
      );
      if (okunan == null) {
        throw const FormatException('guncelleme.json okunamadı');
      }

      _bilgi = okunan;
      // Yalnizca **basarili** denetim zaman damgasi birakir; boylece
      // internetsiz gecen gunlerin ardindan ilk baglantida hemen denenir.
      _sonDenetim = DateTime.now();
      final SharedPreferences ayarlar = await SharedPreferences.getInstance();
      await ayarlar.setInt(_pSonDenetim, _sonDenetim!.millisecondsSinceEpoch);
      await ayarlar.setString(_pSonBilgi, utf8.decode(bayt, allowMalformed: true));

      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('Guncelleme denetlenemedi: $e');
      _hataKur(_teknikMesaj(e));
      return false;
    } finally {
      istemci?.close();
    }
  }

  // --------------------------------------------------------------- indir

  /// APK'yi indirir, dogrular ve [GuncellemeAsamasi.hazir]'a gecer.
  ///
  /// Yarim inen dosya silinmez: `Range` basligiyla kaldigi yerden devam
  /// eder. Wi-Fi menzilinden cikan bir telefon 45 MB'i bastan indirmez.
  Future<void> _indir(GuncellemeBilgisi b, {required bool elle}) async {
    final Directory klasor = Directory(await _klasor());
    final File hedef = File(_apkYolu(klasor.path, b.surumKodu));
    final File yarim = File('${hedef.path}.yarim');

    _indirmeIptal = false;
    _asama = GuncellemeAsamasi.indiriliyor;
    _hata = null;
    _toplamBayt = b.boyut;
    _inenBayt = 0;
    _ilerleme = null;
    notifyListeners();

    IOSink? akis;
    int baslangic = 0;
    try {
      if (!klasor.existsSync()) klasor.createSync(recursive: true);
      // Buraya gelindiyse hedef ya yok ya da ozeti tutmuyor.
      await _sessizSil(hedef);

      if (yarim.existsSync()) {
        final int uzunluk = await yarim.length();
        if (uzunluk > 0 && uzunluk < b.boyut) {
          baslangic = uzunluk;
        } else {
          // Ya bos ya beklenenden buyuk: guvenilmez, bastan.
          await _sessizSil(yarim);
        }
      }

      _istemci = HttpClient()
        ..connectionTimeout = const Duration(seconds: 30)
        ..idleTimeout = const Duration(seconds: 30);

      final HttpClientRequest istek =
          await _istemci!.getUrl(Uri.parse(b.apkUrl));
      if (baslangic > 0) {
        istek.headers.set(HttpHeaders.rangeHeader, 'bytes=$baslangic-');
      }
      final HttpClientResponse yanit = await istek.close();

      if (baslangic > 0 && yanit.statusCode == HttpStatus.ok) {
        // Sunucu Range'i yok saydi: bastan geliyor, dosyayi sifirla.
        baslangic = 0;
        await _sessizSil(yarim);
      } else if (yanit.statusCode != HttpStatus.ok &&
          yanit.statusCode != HttpStatus.partialContent) {
        throw HttpException('Sunucu ${yanit.statusCode} yanıtı verdi');
      }

      _inenBayt = baslangic;
      akis = yarim.openWrite(
        mode: baslangic > 0 ? FileMode.append : FileMode.write,
      );

      DateTime sonBildirim = DateTime.fromMillisecondsSinceEpoch(0);
      await for (final List<int> parca in yanit) {
        if (_indirmeIptal) throw const _Iptal();
        // Baglanti tamamen koptuysa dur. Yarim dosya diskte kalir, ag
        // gelince kaldigi yerden devam eder.
        if (!elle && !Ag.instance.durum.internetVar) throw const _Iptal();

        akis.add(parca);
        _inenBayt += parca.length;
        if (_inenBayt > b.boyut) {
          throw const FormatException('Dosya beklenenden büyük');
        }
        _ilerleme = _toplamBayt > 0 ? _inenBayt / _toplamBayt : null;

        final DateTime simdi = DateTime.now();
        if (simdi.difference(sonBildirim).inMilliseconds > 200) {
          sonBildirim = simdi;
          notifyListeners();
        }
      }
      await akis.flush();
      await akis.close();
      akis = null;

      if (await yarim.length() != b.boyut) {
        throw const HttpException('İndirme yarıda kesildi');
      }
      // Dogrulama **tasimadan once**: bozuk bir dosya asla asil adi
      // almasin, yoksa sonraki acilista "hazir" sanip kurmaya calisiriz.
      if (!await _ozetDogruMu(yarim, b.sha256)) {
        await _sessizSil(yarim);
        throw const FormatException('İnen dosya bozuk');
      }

      await yarim.rename(hedef.path);
      _hazirla(b);
    } catch (e) {
      try {
        await akis?.close();
      } catch (_) {
        // Zaten kapanmis olabilir.
      }
      if (e is _Iptal) {
        // Yarim dosya bilerek birakiliyor: sonraki denemede devam edecek.
        _bosaAl();
        return;
      }
      debugPrint('Guncelleme indirilemedi: $e');
      _hataKur(_teknikMesaj(e));
    } finally {
      _istemci?.close();
      _istemci = null;
    }
  }

  /// Bu surumun APK'si diskte ve ozeti dogru mu?
  Future<bool> _diskteHazirMi(GuncellemeBilgisi b) async {
    try {
      final File apk = File(_apkYolu(await _klasor(), b.surumKodu));
      if (!apk.existsSync()) return false;
      if (await apk.length() != b.boyut) return false;
      return await _ozetDogruMu(apk, b.sha256);
    } catch (e) {
      debugPrint('Hazir APK kontrolu basarisiz: $e');
      return false;
    }
  }

  void _hazirla(GuncellemeBilgisi b, {bool acilista = false}) {
    _acilistaHazirdi = acilista;
    _bilgi = b;
    _asama = GuncellemeAsamasi.hazir;
    _ilerleme = 1;
    _inenBayt = b.boyut;
    _toplamBayt = b.boyut;
    _hata = null;
    notifyListeners();
  }

  void _bosaAl() {
    _asama = GuncellemeAsamasi.bos;
    _ilerleme = null;
    notifyListeners();
  }

  /// Dosyayi parca parca okuyarak ozetler; 45 MB'i bellege almaz.
  Future<bool> _ozetDogruMu(File dosya, String beklenen) async {
    try {
      final Digest ozet = await sha256.bind(dosya.openRead()).first;
      return ozet.toString() == beklenen;
    } catch (e) {
      debugPrint('Ozet hesaplanamadi: $e');
      return false;
    }
  }

  // ----------------------------------------------------------------- kur

  /// Sistemin kurulum penceresini acar.
  ///
  /// `false` donerse **kullaniciya bir sey gosterilmeli**: ya izin eksik
  /// (bkz. [kurulumIzniVar]) ya da dosya kaybolmus.
  Future<bool> kur() async {
    final GuncellemeBilgisi? b = _bilgi;
    if (b == null || _asama != GuncellemeAsamasi.hazir) return false;

    try {
      _kurulumIzniVar =
          await _kanal.invokeMethod<bool>('kurulumIzniVarMi') ?? false;
      if (!_kurulumIzniVar) {
        notifyListeners();
        return false;
      }

      final String yol = _apkYolu(await _klasor(), b.surumKodu);
      if (!File(yol).existsSync()) {
        // Android depolamayi temizlemis olabilir: bastan indir.
        _bosaAl();
        unawaited(degerlendir(elle: true));
        return false;
      }

      _asama = GuncellemeAsamasi.kuruluyor;
      _imzaUyusmazligi = false;
      _iptalEdildi = false;
      _hata = null;
      notifyListeners();

      await _kanal.invokeMethod<bool>('kur', <String, String>{'apkYolu': yol});
      return true;
    } catch (e) {
      debugPrint('Kurulum baslatilamadi: $e');
      _hataKur('Kurulum başlatılamadı.');
      return false;
    }
  }

  /// Kullaniciyi "bilinmeyen kaynaklara izin ver" ekranina goturur.
  Future<bool> kurulumIzniIste() async {
    try {
      return await _kanal.invokeMethod<bool>('kurulumIzniEkraniniAc') ?? false;
    } catch (e) {
      debugPrint('Izin ekrani acilamadi: $e');
      return false;
    }
  }

  /// Izin ekranindan donuldugunde durumu tazeler.
  Future<void> izniTazele() async {
    try {
      _kurulumIzniVar =
          await _kanal.invokeMethod<bool>('kurulumIzniVarMi') ?? false;
      notifyListeners();
    } catch (e) {
      debugPrint('Kurulum izni okunamadi: $e');
    }
  }

  /// Kotlin tarafindan itilen kurulum sonuclari.
  Future<dynamic> _yerliCagri(MethodCall cagri) async {
    if (cagri.method != 'kurulumSonucu') return null;

    final Map<Object?, Object?> veri =
        (cagri.arguments as Map<Object?, Object?>?) ?? <Object?, Object?>{};
    final String sonuc = (veri['sonuc'] as String?) ?? 'hata';
    final String? mesaj = veri['mesaj'] as String?;
    debugPrint('Kurulum sonucu: $sonuc ${mesaj ?? ''}');

    switch (sonuc) {
      case 'onayBekleniyor':
        // Sistem penceresi acildi; kullanicinin kararini bekliyoruz.
        break;

      case 'tamam':
        // Genelde buraya gelinmez: kurulum basarili olunca surec
        // degistirildigi icin uygulama yeniden baslar.
        _bilgi = null;
        _bosaAl();
        await _eskiDosyalariTemizle(hepsi: true);
        break;

      case 'iptal':
        // Kullanici sistemin penceresinde "Vazgeç" dedi. Guncelleme
        // zorunlu oldugu icin ertelemiyoruz: ekran yerinde kaliyor,
        // kullanici tekrar deneyebilir.
        _asama = GuncellemeAsamasi.hazir;
        _iptalEdildi = true;
        notifyListeners();
        break;

      case 'imza':
        // Kurulu uygulama baska bir anahtarla imzalanmis.
        _imzaUyusmazligi = true;
        _hataKur('İmza uyuşmuyor: kurulu sürüm farklı bir anahtarla '
            'imzalanmış. Silmeden güncellenemez. ${mesaj ?? ''}');
        break;

      case 'yer':
        _hataKur('Telefonda yer kalmamış.');
        break;

      case 'bozuk':
      case 'dosyaYok':
        // Inen dosyaya guvenilmez; at, bir dahakine yeniden insin.
        await _eskiDosyalariTemizle(hepsi: true);
        _bosaAl();
        break;

      case 'engellendi':
        _hataKur('Kurulum telefon tarafından engellendi '
            '(Play Protect olabilir).');
        break;

      case 'uyumsuz':
        _hataKur('Bu sürüm telefonla uyumlu değil.');
        break;

      default:
        _hataKur('Kurulum tamamlanamadı. ${mesaj ?? ''}');
    }
    return null;
  }

  /// Suren indirmeyi durdurur (Ayarlar'daki elle denetim icin).
  void indirmeyiDurdur() {
    _indirmeIptal = true;
    _istemci?.close(force: true);
  }

  // ------------------------------------------------------------ yardimci

  Future<String> _klasor() async {
    if (_klasorYolu != null) return _klasorYolu!;
    final Directory kok = await getApplicationSupportDirectory();
    _klasorYolu = '${kok.path}/$_klasorAdi';
    return _klasorYolu!;
  }

  static String _apkYolu(String klasor, int surumKodu) =>
      '$klasor/hatirlaf-$surumKodu.apk';

  /// Kurulmus ya da artik beklenmeyen APK'lari siler.
  ///
  /// 45 MB'lik bir dosyanin telefonda oylece durmasi, hatiralar icin yer
  /// kalmamasi demek olabilir. [hepsi] `true` ise beklenen surum de silinir.
  Future<void> _eskiDosyalariTemizle({bool hepsi = false}) async {
    try {
      final Directory klasor = Directory(await _klasor());
      if (!klasor.existsSync()) return;

      final int? tutulacak = hepsi ? null : _bilgi?.surumKodu;
      for (final FileSystemEntity dosya in klasor.listSync()) {
        if (dosya is! File) continue;
        final int? kod = _dosyadanSurumKodu(dosya.path);
        final bool gerekli = kod != null &&
            kod == tutulacak &&
            kod > _mevcutSurumKodu;
        if (!gerekli) await _sessizSil(dosya);
      }
    } catch (e) {
      debugPrint('Eski guncelleme dosyalari silinemedi: $e');
    }
  }

  static int? _dosyadanSurumKodu(String yol) {
    final RegExpMatch? e =
        RegExp(r'hatirlaf-(\d+)\.apk(\.yarim)?$').firstMatch(yol);
    return e == null ? null : int.tryParse(e.group(1)!);
  }

  static Future<void> _sessizSil(FileSystemEntity dosya) async {
    try {
      if (dosya.existsSync()) await dosya.delete();
    } catch (e) {
      debugPrint('Silinemedi (${dosya.path}): $e');
    }
  }

  /// Hata kaydeder ama **hazir bir APK'yi gozden cikarmaz**: elle yapilan
  /// basarisiz bir denetim, dun inmis kurulabilir surumu kaybettirmemeli.
  void _hataKur(String mesaj) {
    _hata = mesaj;
    if (!hazirApkVar) {
      _asama = GuncellemeAsamasi.hata;
      _ilerleme = null;
    }
    notifyListeners();
  }

  static String _teknikMesaj(Object e) {
    if (e is SocketException) return 'İnternete bağlanılamadı.';
    if (e is TimeoutException) return 'Sunucu yanıt vermedi.';
    if (e is HttpException) return e.message;
    if (e is FormatException) return e.message;
    if (e is FileSystemException) return 'Telefonda yer kalmamış olabilir.';
    return e.toString();
  }

  @override
  void dispose() {
    _agAbonelik?.cancel();
    _istemci?.close(force: true);
    super.dispose();
  }
}

/// Indirme bilerek durduruldu (ag degisti ya da kullanici durdurdu).
class _Iptal implements Exception {
  const _Iptal();
}
