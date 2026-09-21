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

enum GuncellemeAsamasi { bos, denetleniyor, indiriliyor, hazir, kuruluyor, hata }

/// Uygulamayi kendi kendine gunceller.
///
/// Denetim ve indirme kullaniciya gorunmez; soru yalnizca APK inip ozeti
/// dogrulandiktan sonra, bir sonraki oturum basinda sorulur.
class Guncelleyici extends ChangeNotifier {
  Guncelleyici._();

  static final Guncelleyici instance = Guncelleyici._();

  static const MethodChannel _kanal = MethodChannel('hatirla/guncelleme');

  static const String _pSonDenetim = 'guncelleme_son_denetim';
  static const String _pSonBilgi = 'guncelleme_son_bilgi';

  static const String _klasorAdi = 'guncelleme';

  /// `guncelleme.json` birkac yuz bayt; otesi bir yanlislik demek.
  static const int _enBuyukBilgiBayt = 16 * 1024;

  int _mevcutSurumKodu = 0;
  int get mevcutSurumKodu => _mevcutSurumKodu;

  String _mevcutSurumAdi = '';
  String get mevcutSurumAdi => _mevcutSurumAdi;

  /// Cihazin mimarileri, tercih sirasiyla; hangi APK'nin inecegini belirler.
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

  /// Teknik hata metni; yalnizca Ayarlar'da gosterilir.
  String? _hata;
  String? get hata => _hata;

  /// Imza uyusmazligi: kullanicinin cozemeyecegi tek hata. Tek cikis yolu
  /// uygulamayi silmek, o da hatiralari siler; bu yuzden oyle denmez.
  bool _imzaUyusmazligi = false;
  bool get imzaUyusmazligi => _imzaUyusmazligi;

  /// Sistemin penceresinde "Vazgeç" dendi mi? Ekran yerinde kaldigi icin
  /// sebebi soylenmezse kullanici aciklamasiz ayni ekrana doner.
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

  /// Guncelleme uygulama acildiginda zaten hazir miydi? Kullanimin
  /// ortasinda inen bir guncelleme ekran basmasin diye.
  bool _acilistaHazirdi = false;
  bool get acilistaHazirdi => _acilistaHazirdi;

  /// On plana donus de yeni bir oturum sayilir: bekleyen guncelleme sorulabilir.
  void oturumaGirildi() {
    if (_asama == GuncellemeAsamasi.hazir && !_acilistaHazirdi) {
      _acilistaHazirdi = true;
      notifyListeners();
    }
  }

  bool get hazirApkVar =>
      _asama == GuncellemeAsamasi.hazir ||
      _asama == GuncellemeAsamasi.kuruluyor;

  /// Guncelleme zorunlu: erteleme yok, hazirsa gosterilir.
  bool get sorulabilir =>
      _asama == GuncellemeAsamasi.hazir && _bilgi != null;

  /// Acilista bir kez cagrilir; firlatmaz, acilisi bekletmez.
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
      // Diskteki son manifest: dun inmis bir guncelleme bugun internetsiz
      // de kurulabilsin.
      final String? sonBilgi = ayarlar.getString(_pSonBilgi);
      if (sonBilgi != null) {
        final GuncellemeBilgisi? b =
            GuncellemeBilgisi.cozumle(sonBilgi, _abiler);
        if (b != null && b.surumKodu > _mevcutSurumKodu) _bilgi = b;
      }

      await _eskiDosyalariTemizle();

      // Onceki oturumdan kalmis, dogrulanmis bir APK varsa sormak icin
      // dogru an.
      final GuncellemeBilgisi? b = _bilgi;
      if (b != null && await _diskteHazirMi(b)) {
        _hazirla(b, acilista: true);
      }

      Ag.instance.basla();
      _agAbonelik = Ag.instance.degisim.listen((AgDurumu d) {
        if (d.internetVar) unawaited(degerlendir());
      });

      notifyListeners();
      unawaited(degerlendir());
    } on MissingPluginException {
      // Android disi platform ya da test ortami.
      debugPrint('Guncelleme kanali yok; guncelleme kapali.');
    } catch (e, s) {
      debugPrint('Guncelleyici baslatilamadi: $e\n$s');
    }
  }

  /// "Su anda ne yapmaliyim?" Acilista, ag degisiminde ve elle denetimde
  /// cagrilir. [elle] ag ve sure kisitlarini atlar.
  Future<void> degerlendir({bool elle = false}) async {
    if (_mesgul || _mevcutSurumKodu <= 0) return;

    _mesgul = true;
    try {
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

  /// `guncelleme.json`'u okur. Basarisizsa sessizce `false` doner.
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
      // Yalnizca basarili denetim damga birakir: internetsiz gunlerin
      // ardindan ilk baglantida hemen denensin.
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

  /// APK'yi indirir, dogrular ve [GuncellemeAsamasi.hazir]'a gecer.
  /// Yarim inen dosya silinmez; `Range` ile kaldigi yerden devam eder.
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
          // Bos ya da beklenenden buyuk: guvenilmez.
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
        // Baglanti koptuysa dur; yarim dosya diskte kalir.
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
      // Dogrulama tasimadan once: bozuk dosya asil adi almasin, yoksa
      // sonraki acilista "hazir" sanilir.
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
        // Yarim dosya bilerek kaliyor: sonraki denemede devam edecek.
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

  /// Parca parca okur; 45 MB'i bellege almaz.
  Future<bool> _ozetDogruMu(File dosya, String beklenen) async {
    try {
      final Digest ozet = await sha256.bind(dosya.openRead()).first;
      return ozet.toString() == beklenen;
    } catch (e) {
      debugPrint('Ozet hesaplanamadi: $e');
      return false;
    }
  }

  /// Sistemin kurulum penceresini acar. `false` donerse kullaniciya bir sey
  /// gosterilmeli: ya izin eksik ([kurulumIzniVar]) ya da dosya kaybolmus.
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
        // Android depolamayi temizlemis olabilir.
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

  Future<bool> kurulumIzniIste() async {
    try {
      return await _kanal.invokeMethod<bool>('kurulumIzniEkraniniAc') ?? false;
    } catch (e) {
      debugPrint('Izin ekrani acilamadi: $e');
      return false;
    }
  }

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
        break;

      case 'tamam':
        // Genelde buraya gelinmez: basarili kurulumda surec degistirilir.
        _bilgi = null;
        _bosaAl();
        await _eskiDosyalariTemizle(hepsi: true);
        break;

      case 'iptal':
        // Guncelleme zorunlu: ertelemiyoruz, ekran yerinde kaliyor.
        _asama = GuncellemeAsamasi.hazir;
        _iptalEdildi = true;
        notifyListeners();
        break;

      case 'imza':
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

  /// Suren indirmeyi durdurur (elle denetim icin).
  void indirmeyiDurdur() {
    _indirmeIptal = true;
    _istemci?.close(force: true);
  }

  Future<String> _klasor() async {
    if (_klasorYolu != null) return _klasorYolu!;
    final Directory kok = await getApplicationSupportDirectory();
    _klasorYolu = '${kok.path}/$_klasorAdi';
    return _klasorYolu!;
  }

  static String _apkYolu(String klasor, int surumKodu) =>
      '$klasor/hatirlaf-$surumKodu.apk';

  /// Kurulmus ya da artik beklenmeyen APK'lari siler; 45 MB telefonda
  /// oylece durmasin. [hepsi] ise beklenen surum de silinir.
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

  /// Hata kaydeder ama hazir bir APK'yi gozden cikarmaz: basarisiz bir
  /// denetim dun inmis kurulabilir surumu kaybettirmemeli.
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

/// Indirme bilerek durduruldu.
class _Iptal implements Exception {
  const _Iptal();
}
