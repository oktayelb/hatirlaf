import 'dart:typed_data';

/// Testler icin ADTS cercevesi uretir.
///
/// Baslik 7 bayt (CRC yok): senkron sozcugu, AAC-LC profili, ornekleme
/// hizi dizini ve cercevenin toplam uzunlugu. Kalani dolgu.
Uint8List adtsCercevesi({
  int hizDizini = 4, // 44100
  int kanal = 1,
  int uzunluk = 200,
  int hamBlok = 1,
}) {
  if (uzunluk < 7) throw ArgumentError('cerceve en az 7 bayt');
  final Uint8List c = Uint8List(uzunluk);
  c[0] = 0xFF;
  // MPEG-4, layer 0, koruma yok (CRC yok -> baslik 7 bayt).
  c[1] = 0xF1;
  // profil(2) | hiz dizini(4) | ozel(1) | kanal ust biti(1)
  c[2] = (1 << 6) | ((hizDizini & 0x0F) << 2) | ((kanal >> 2) & 0x01);
  // kanal alt bitleri(2) | ... | uzunluk ust bitleri(2)
  c[3] = ((kanal & 0x03) << 6) | ((uzunluk >> 11) & 0x03);
  c[4] = (uzunluk >> 3) & 0xFF;
  // uzunluk alt bitleri(3) | tampon dolulugu ust bitleri(5)
  c[5] = ((uzunluk & 0x07) << 5) | 0x1F;
  // tampon dolulugu alt bitleri(6) | ham blok sayisi - 1 (2)
  c[6] = 0xFC | ((hamBlok - 1) & 0x03);
  return c;
}

/// [sayi] cerceveden olusan bir ADTS akisi.
Uint8List adtsAkisi(int sayi, {int uzunluk = 200, int hizDizini = 4}) {
  final BytesBuilder b = BytesBuilder();
  for (int i = 0; i < sayi; i++) {
    b.add(adtsCercevesi(uzunluk: uzunluk, hizDizini: hizDizini));
  }
  return b.toBytes();
}
