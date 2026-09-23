import 'dart:io';
import 'dart:typed_data';

/// ADTS akisindaki AAC cercevelerini sayar.
///
/// Kayit dosyasi neden ADTS: her cerceve kendi uzunlugunu tasiyor, yani
/// dosya istenen yerde kesilse bile o ana kadarki kisim gecerli kaliyor.
/// (m4a boyle degil: sureyi ve cerceve tablosunu tutan `moov` atomu en
/// sona yazildigi icin yarim kalmis bir m4a hic acilmaz.) Kayit sirasinda
/// telefon uygulamayi oldururse hatira bu sayede kurtarilabiliyor.
///
/// Sureyi de buradan cikariyoruz: ADTS'te toplam sure yazmaz, cerceve
/// saymak gerekir. AAC-LC'de her cerceve 1024 ornek.
class AdtsBilgi {
  const AdtsBilgi({
    required this.orneklemeHizi,
    required this.cerceveSayisi,
    required this.gecerliBayt,
  });

  final int orneklemeHizi;

  /// Tam olarak okunabilen cerceve sayisi.
  final int cerceveSayisi;

  /// Son tam cerceveden sonraki ofset. Dosya bundan uzunsa sonunda yarim
  /// kalmis bir cerceve var demektir.
  final int gecerliBayt;

  /// Her cerceve 1024 ornek.
  Duration get sure => Duration(
        milliseconds: orneklemeHizi <= 0
            ? 0
            : (cerceveSayisi * 1024 * 1000) ~/ orneklemeHizi,
      );

  bool get bos => cerceveSayisi == 0;

  /// ADTS basligindaki 4 bitlik dizinin karsiligi.
  static const List<int> orneklemeHizlari = <int>[
    96000, 88200, 64000, 48000, 44100, 32000,
    24000, 22050, 16000, 12000, 11025, 8000,
  ];

  static const int _blokBayt = 256 * 1024;

  /// Dosyayi tarar. ADTS gibi gorunmuyorsa `null` doner.
  ///
  /// Yarim kalmis son cerceve sayilmaz: [gecerliBayt] ondan onceyi
  /// gosterir.
  static Future<AdtsBilgi?> oku(File dosya) async {
    RandomAccessFile? f;
    try {
      final int uzunluk = await dosya.length();
      if (uzunluk < 7) return null;

      f = await dosya.open();

      int hiz = 0;
      int cerceve = 0;
      int ofset = 0;

      while (ofset < uzunluk) {
        await f.setPosition(ofset);
        final Uint8List blok = await f.read(_blokBayt);
        if (blok.length < 7) break;

        int i = 0;
        bool blogunSonu = false;
        while (i + 7 <= blok.length) {
          // Senkron sozcugu: 12 bit 1.
          if (blok[i] != 0xFF || (blok[i + 1] & 0xF0) != 0xF0) {
            // Ilk cercevede tutmuyorsa bu dosya ADTS degil.
            if (cerceve == 0) return null;
            blogunSonu = true;
            break;
          }

          final int hizDizini = (blok[i + 2] >> 2) & 0x0F;
          if (hizDizini >= orneklemeHizlari.length) {
            if (cerceve == 0) return null;
            blogunSonu = true;
            break;
          }
          if (hiz == 0) hiz = orneklemeHizlari[hizDizini];

          final int uzun = ((blok[i + 3] & 0x03) << 11) |
              (blok[i + 4] << 3) |
              (blok[i + 5] >> 5);
          if (uzun < 7) {
            if (cerceve == 0) return null;
            blogunSonu = true;
            break;
          }

          // Cerceve blogun disina tasiyorsa bir sonraki blogu onun
          // basindan okuyalim.
          if (i + uzun > blok.length) break;

          // Bir cercevede birden fazla ham veri blogu olabilir.
          final int hamBlok = (blok[i + 6] & 0x03) + 1;
          cerceve += hamBlok;
          i += uzun;
        }

        if (blogunSonu) {
          ofset += i;
          break;
        }
        if (i == 0) {
          // Tek bir cerceve bile sigmadi; ilerleyemiyoruz.
          break;
        }
        ofset += i;
      }

      if (cerceve == 0 || hiz == 0) return null;
      return AdtsBilgi(
        orneklemeHizi: hiz,
        cerceveSayisi: cerceve,
        gecerliBayt: ofset,
      );
    } catch (_) {
      // Okunamayan dosya kurtarilamaz; cagiran tarafta ele aliniyor.
      return null;
    } finally {
      try {
        await f?.close();
      } catch (_) {
        // Zaten kapanmis olabilir.
      }
    }
  }
}
