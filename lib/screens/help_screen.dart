import 'package:flutter/material.dart';

import '../theme.dart';

/// "Nasil kullanilir" ekrani: her adim tek cumle, basinda buyuk numara.
class HelpScreen extends StatelessWidget {
  const HelpScreen({super.key});

  static const List<({IconData ikon, String baslik, String metin})> _adimlar =
      <({IconData ikon, String baslik, String metin})>[
    (
      ikon: Icons.mic_rounded,
      baslik: 'Anlatmaya başlayın',
      metin: 'Ana ekranın altındaki kırmızı düğmeye dokunun, sonra ortadaki '
          'büyük düğmeye basıp konuşmaya başlayın.',
    ),
    (
      ikon: Icons.help_outline_rounded,
      baslik: 'Ne anlatacağınızı bilemezseniz',
      metin: 'Ana ekranda size her gün bir soru sorulur. “Bunu Anlat”a '
          'dokunup o soruyu cevaplayabilirsiniz.',
    ),
    (
      ikon: Icons.pause_rounded,
      baslik: 'Ara verebilirsiniz',
      metin: 'Anlatırken yorulursanız “Ara Ver”e dokunun. Hazır olunca '
          '“Devam Et” deyip kaldığınız yerden sürdürün.',
    ),
    (
      ikon: Icons.check_rounded,
      baslik: 'Bitirince kaydedin',
      metin: 'Yeşil “Bitir ve Kaydet” düğmesine dokunun. Hatıranız '
          'kaydedilir ve yazıya çevrilmeye başlar.',
    ),
    (
      ikon: Icons.edit_note_rounded,
      baslik: 'Yazıya çevrilmesini bekleyin',
      metin: 'Telefon konuşmanızı kendi içinde yazıya döker. Uzun '
          'hatıralarda birkaç dakika sürebilir; internet gerekmez.',
    ),
    (
      ikon: Icons.add_a_photo_rounded,
      baslik: 'Fotoğraf ekleyin',
      metin: 'Hatırayı açıp “Fotoğraf Ekle”ye dokunun. Eski bir fotoğrafın '
          'resmini çekebilir ya da telefondakilerden seçebilirsiniz.',
    ),
    (
      ikon: Icons.ios_share_rounded,
      baslik: 'Ailenizle paylaşın',
      metin: 'Hatıra ekranındaki “Ailemle Paylaş” düğmesiyle sesinizi ve '
          'yazısını çocuklarınıza gönderebilirsiniz.',
    ),
    (
      ikon: Icons.lock_rounded,
      baslik: 'Her şey telefonunuzda kalır',
      metin: 'Ses kayıtlarınız ve fotoğraflarınız internete yüklenmez. '
          'Siz paylaşmadıkça kimse göremez.',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Nasıl Kullanılır?')),
      body: SafeArea(
        child: ListView.separated(
          padding: const EdgeInsets.all(HatirlaSizes.gutter),
          itemCount: _adimlar.length,
          separatorBuilder: (_, _) => const SizedBox(height: 16),
          itemBuilder: (BuildContext context, int i) {
            final ({IconData ikon, String baslik, String metin}) a =
                _adimlar[i];
            return Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: HatirlaColors.card,
                borderRadius: BorderRadius.circular(HatirlaSizes.radius),
                border: Border.all(color: HatirlaColors.line, width: 2),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Container(
                    width: 62,
                    height: 62,
                    alignment: Alignment.center,
                    decoration: const BoxDecoration(
                      color: HatirlaColors.primarySoft,
                      shape: BoxShape.circle,
                    ),
                    child: Icon(a.ikon,
                        size: 34, color: HatirlaColors.primaryDark),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Text(
                          a.baslik,
                          style: const TextStyle(
                              fontSize: 23,
                              fontWeight: FontWeight.w700,
                              height: 1.3),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          a.metin,
                          style: const TextStyle(
                              fontSize: 20,
                              height: 1.5,
                              color: HatirlaColors.inkSoft),
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
