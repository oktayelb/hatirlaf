import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../theme.dart';
import '../widgets/common.dart';

/// "Nasil kullanilir" ekrani: her adim tek cumle, basinda renkli bir simge.
class HelpScreen extends StatelessWidget {
  const HelpScreen({super.key});

  static const List<({IconData ikon, Color renk, String baslik, String metin})>
      _adimlar = <({IconData ikon, Color renk, String baslik, String metin})>[
    (
      ikon: CupertinoIcons.mic_fill,
      renk: HatirlaColors.record,
      baslik: 'Anlatmaya başlayın',
      metin: 'Ana ekranın altındaki kırmızı düğmeye dokunun, sonra ortadaki '
          'büyük düğmeye basıp konuşmaya başlayın.',
    ),
    (
      ikon: CupertinoIcons.quote_bubble_fill,
      renk: HatirlaColors.primary,
      baslik: 'Ne anlatacağınızı bilemezseniz',
      metin: 'Ana ekranda size her gün bir soru sorulur. “Bunu Anlat”a '
          'dokunup o soruyu cevaplayabilirsiniz.',
    ),
    (
      ikon: CupertinoIcons.pause_fill,
      renk: Color(0xFF8E8E93),
      baslik: 'Ara verebilirsiniz',
      metin: 'Anlatırken yorulursanız “Ara Ver”e dokunun. Hazır olunca '
          '“Devam Et” deyip kaldığınız yerden sürdürün.',
    ),
    (
      ikon: CupertinoIcons.checkmark_alt,
      renk: HatirlaColors.confirm,
      baslik: 'Bitirince kaydedin',
      metin: 'Yeşil “Bitir ve Kaydet” düğmesine dokunun. Hatıranız '
          'kaydedilir ve yazıya çevrilmeye başlar.',
    ),
    (
      ikon: CupertinoIcons.pencil,
      renk: Color(0xFFE08600),
      baslik: 'Yazıya çevrilmesini bekleyin',
      metin: 'Telefon konuşmanızı kendi içinde yazıya döker. Uzun '
          'hatıralarda birkaç dakika sürebilir; internet gerekmez.',
    ),
    (
      ikon: CupertinoIcons.camera_fill,
      renk: Color(0xFF0A7AFF),
      baslik: 'Fotoğraf ekleyin',
      metin: 'Hatırayı açıp “Fotoğraf Ekle”ye dokunun. Eski bir fotoğrafın '
          'resmini çekebilir ya da telefondakilerden seçebilirsiniz.',
    ),
    (
      ikon: CupertinoIcons.square_arrow_up,
      renk: Color(0xFF5856D6),
      baslik: 'Ailenizle paylaşın',
      metin: 'Hatıra ekranındaki “Ailemle Paylaş” düğmesiyle sesinizi ve '
          'yazısını çocuklarınıza gönderebilirsiniz.',
    ),
    (
      ikon: CupertinoIcons.lock_fill,
      renk: Color(0xFF3A3A3C),
      baslik: 'Her şey telefonunuzda kalır',
      metin: 'Ses kayıtlarınız ve fotoğraflarınız internete yüklenmez. '
          'Siz paylaşmadıkça kimse göremez.',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const UstCubuk(baslik: 'Nasıl Kullanılır?'),
      body: SafeArea(
        top: false,
        child: ListView.separated(
          padding: const EdgeInsets.fromLTRB(
              HatirlaSizes.gutter, 4, HatirlaSizes.gutter, HatirlaSizes.gutter),
          itemCount: _adimlar.length,
          separatorBuilder: (_, _) => const SizedBox(height: 14),
          itemBuilder: (BuildContext context, int i) {
            final ({IconData ikon, Color renk, String baslik, String metin}) a =
                _adimlar[i];
            return Kart(
              padding: const EdgeInsets.all(18),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  IkonRozeti(ikon: a.ikon, renk: a.renk, boyut: 52),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Text(
                          a.baslik,
                          style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                            height: 1.3,
                            letterSpacing: -0.2,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          a.metin,
                          style: const TextStyle(
                            fontSize: 20,
                            height: 1.5,
                            color: HatirlaColors.inkSoft,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
