import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

/// Kurulu APK'dan yeni APK uretir: "fark guncellemesi".
///
/// APK bir zip dosyasi ve iki surum arasinda girdilerin neredeyse hepsi
/// bayt bayt ayni kaliyor (1.1.0 -> 1.2.0'da 560 girdinin 554'u, 19.7 MB).
/// Degismeyen kisim telefonda zaten duruyor: kurulu APK. Yama bu yuzden
/// yalnizca iki komuttan olusuyor -- "sunu kurulu APK'dan kopyala",
/// "sunu benden al" -- ve 22 MB yerine 3 MB iniyor.
///
/// Uretici taraf: `tool/yama_uret.py`, yayin oncesi dogrulama
/// `tool/yama_dogrula.dart`. Bu dosya bilerek Flutter'a bagimli degil:
/// dogrulama araci onu duz `dart run` ile calistiriyor. Bicim
/// (kucuk uclu):
///
/// ```
///  9 bayt  "HTRLYAMA1"
///  8 bayt  hedef uzunlugu
/// 32 bayt  kaynagin sha256'si
///  4 bayt  komut bolumunun uzunlugu
///  n bayt  komutlar
///  8 bayt  govdenin acilmis uzunlugu
///  …       zlib ile sikistirilmis govde (yeni baytlar)
/// ```
///
/// Komutlar: `0x01 <ofset> <uzunluk>` kaynaktan kopyalar,
/// `0x02 <uzunluk>` govdeden sirayla okur; sayilar degisken uzunlukta
/// (her baytin ust biti "devam ediyor" demek).
class Yama {
  const Yama._();

  static const String _sihir = 'HTRLYAMA1';
  static const int _kopyala = 1;
  static const int _yeni = 2;

  /// Baslik + komutlar + govde; bunun otesi ya yanlis ya kotu niyetli.
  static const int enBuyukYamaBayt = 64 * 1024 * 1024;

  static const int _parcaBayt = 1 << 20;

  /// [kaynak] + [yama] -> [hedef]. Hedef dosyanin uzerine yazar.
  ///
  /// Yama tutmazsa [FormatException] firlatir ve yarim hedefi siler;
  /// cagiran taraf tam APK'yi indirmeye donmeli. Uretilen dosyanin
  /// dogrulugu burada degil, cagirandaki sha256 karsilastirmasinda
  /// kesinlesir.
  static Future<void> uygula({
    required File kaynak,
    required File yama,
    required File hedef,
  }) async {
    final int yamaBoyut = await yama.length();
    if (yamaBoyut > enBuyukYamaBayt) {
      throw const FormatException('Yama beklenenden büyük');
    }

    final Uint8List ham = await yama.readAsBytes();
    final _Okuyucu o = _Okuyucu(ham);

    if (ham.length < _sihir.length ||
        String.fromCharCodes(ham.sublist(0, _sihir.length)) != _sihir) {
      throw const FormatException('Yama biçimi tanınmadı');
    }
    o.atla(_sihir.length);

    final int hedefUzunluk = o.uint64();
    final Uint8List kaynakOzet = o.baytlar(32);
    final Uint8List komutlar = o.baytlar(o.uint32());
    final int govdeUzunluk = o.uint64();
    final Uint8List govde = Uint8List.fromList(zlib.decode(o.kalan()));
    if (govde.length != govdeUzunluk) {
      throw const FormatException('Yama gövdesi bozuk');
    }

    // Kurulu APK gercekten yamanin beklediği dosya mi? Degilse uretilecek
    // sey copten ibaret olur; 22 MB yazmadan once anlayalim.
    final Digest bulunan = await sha256.bind(kaynak.openRead()).first;
    if (bulunan.toString() != _onaltilik(kaynakOzet)) {
      throw const FormatException('Kurulu APK yamanın beklediğinden farklı');
    }

    final int kaynakUzunluk = await kaynak.length();
    final RandomAccessFile giris = await kaynak.open();
    RandomAccessFile? cikis;
    try {
      cikis = await hedef.open(mode: FileMode.write);

      final _Okuyucu k = _Okuyucu(komutlar);
      int yazilan = 0;
      int govdeImleci = 0;

      while (!k.bitti) {
        final int komut = k.bayt();
        if (komut == _kopyala) {
          final int ofset = k.varint();
          final int uzunluk = k.varint();
          if (ofset < 0 || uzunluk < 0 || ofset + uzunluk > kaynakUzunluk) {
            throw const FormatException('Yama kaynağın dışını gösteriyor');
          }
          await giris.setPosition(ofset);
          int kalan = uzunluk;
          while (kalan > 0) {
            final int n = kalan < _parcaBayt ? kalan : _parcaBayt;
            final Uint8List parca = await giris.read(n);
            if (parca.length != n) {
              throw const FormatException('Kurulu APK beklenmedik yerde bitti');
            }
            await cikis.writeFrom(parca);
            kalan -= n;
          }
          yazilan += uzunluk;
        } else if (komut == _yeni) {
          final int uzunluk = k.varint();
          if (uzunluk < 0 || govdeImleci + uzunluk > govde.length) {
            throw const FormatException('Yama gövdesi beklenenden kısa');
          }
          await cikis.writeFrom(govde, govdeImleci, govdeImleci + uzunluk);
          govdeImleci += uzunluk;
          yazilan += uzunluk;
        } else {
          throw FormatException('Yamada bilinmeyen komut: $komut');
        }

        if (yazilan > hedefUzunluk) {
          throw const FormatException('Yama beklenenden çok bayt üretti');
        }
      }

      if (yazilan != hedefUzunluk || govdeImleci != govde.length) {
        throw const FormatException('Yama eksik kaldı');
      }
      await cikis.flush();
    } catch (_) {
      await _kapat(cikis);
      cikis = null;
      try {
        if (hedef.existsSync()) await hedef.delete();
      } catch (_) {
        // Silinemediyse de onemli degil: sonraki deneme uzerine yaziyor.
      }
      rethrow;
    } finally {
      await _kapat(cikis);
      await _kapat(giris);
    }
  }

  static Future<void> _kapat(RandomAccessFile? d) async {
    try {
      await d?.close();
    } catch (_) {
      // Zaten kapanmis olabilir.
    }
  }

  static String _onaltilik(Uint8List b) =>
      b.map((int x) => x.toRadixString(16).padLeft(2, '0')).join();
}

/// Bayt dizisini sirayla okur; sinir disina cikarsa [FormatException].
class _Okuyucu {
  _Okuyucu(this._veri);

  final Uint8List _veri;
  int _imlec = 0;

  bool get bitti => _imlec >= _veri.length;

  void atla(int n) => _imlec += n;

  void _yeter(int n) {
    if (_imlec + n > _veri.length) {
      throw const FormatException('Yama beklenenden kısa');
    }
  }

  int bayt() {
    _yeter(1);
    return _veri[_imlec++];
  }

  Uint8List baytlar(int n) {
    _yeter(n);
    final Uint8List p = _veri.sublist(_imlec, _imlec + n);
    _imlec += n;
    return p;
  }

  Uint8List kalan() => baytlar(_veri.length - _imlec);

  int uint32() =>
      ByteData.sublistView(baytlar(4)).getUint32(0, Endian.little);

  int uint64() =>
      ByteData.sublistView(baytlar(8)).getUint64(0, Endian.little);

  /// Degisken uzunluklu sayi: her baytin alt 7 biti veri, ust biti
  /// "devam ediyor". 64 bitten uzun bir sayi bozukluk demek.
  int varint() {
    int sonuc = 0;
    int kaydir = 0;
    while (true) {
      if (kaydir > 63) throw const FormatException('Yamada bozuk sayı');
      final int b = bayt();
      sonuc |= (b & 0x7F) << kaydir;
      if (b & 0x80 == 0) return sonuc;
      kaydir += 7;
    }
  }
}
