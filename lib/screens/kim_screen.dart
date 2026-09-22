import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../data/akrabalar.dart';
import '../services/kullanici.dart';
import '../theme.dart';
import '../widgets/common.dart';
import 'home_screen.dart';
import 'welcome_screen.dart';

/// Uygulamanin ilk ekrani: "Hangi akrabamla konuşuyorum?"
///
/// Yazi yazdirmiyoruz. Dort buyuk dugmeden birine dokunmak, titreyen bir
/// parmakla klavyeden ad yazmaktan da kolay, kuran kisi icin de daha
/// guvenilir: kimlikler her telefonda ayni yaziliyor.
///
/// Soruyu soran yuz ekranda duruyor. Telefonun sordugu bir soru degil,
/// torunun sordugu bir soru.
class KimSinizScreen extends StatelessWidget {
  const KimSinizScreen({super.key, required this.karsilamaTamam});

  /// Karsilama adimlari daha once tamamlandiysa dogrudan ana ekrana
  /// gidiyoruz: guncelleme ile gelen telefonlarda kurulum bastan
  /// tekrarlanmasin.
  final bool karsilamaTamam;

  Future<void> _sec(BuildContext context, Akraba secim) async {
    // Yanlis dokunus geri alinabilir olmali: kimlik bir kere secilip
    // bir daha sorulmuyor.
    final bool dogru = await onayIste(
      context,
      baslik: 'Siz ${secim.ad} misiniz?',
      mesaj: 'Doğruysa devam edelim. Yanlışsa geri dönüp '
          'başka birini seçebilirsiniz.',
      evetYazi: 'Evet, benim',
      hayirYazi: 'Hayır, değilim',
      evetIkon: Icons.check_rounded,
    );
    if (!dogru || !context.mounted) return;

    await Kullanici.instance.sec(secim);
    if (!context.mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute<void>(
        builder: (_) =>
            karsilamaTamam ? const HomeScreen() : const WelcomeScreen(),
      ),
      (Route<dynamic> route) => false,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: <Widget>[
                const SizedBox(height: 16),
                const Center(child: _TorunYuzu()),
                const SizedBox(height: 10),
                Text(
                  '$kTorunAdi soruyor',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w600,
                    color: HatirlaColors.inkSoft,
                  ),
                ),
                const SizedBox(height: 10),
                const _Balon(soru: 'Hangi akrabamla konuşuyorum?'),
                const SizedBox(height: 14),
                // Tek satir: dort secenegin de kaydirmadan gorunmesi,
                // uzun bir aciklamadan daha onemli.
                const Text(
                  'Aşağıdan kendinizi seçin.',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 21,
                    height: 1.3,
                    color: HatirlaColors.inkSoft,
                  ),
                ),
                const SizedBox(height: 18),
                for (final Akraba a in Akraba.values) ...<Widget>[
                  _AkrabaDugmesi(akraba: a, onSec: () => _sec(context, a)),
                  const SizedBox(height: 12),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Soruyu soran yuz. Yuklenemezse bir ikona dusuyor: eksik bir gorsel
/// yuzunden ilk ekran bos kalmasin.
class _TorunYuzu extends StatelessWidget {
  const _TorunYuzu();

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 104,
      width: 104,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: HatirlaColors.primarySoft,
        border: Border.all(color: HatirlaColors.primary, width: 3),
      ),
      child: ClipOval(
        child: Image.asset(
          kTorunFotografi,
          fit: BoxFit.cover,
          errorBuilder: (BuildContext context, Object e, StackTrace? s) =>
              const Icon(Icons.person_rounded,
                  size: 56, color: HatirlaColors.primary),
        ),
      ),
    );
  }
}

/// Soruyu konusma balonu icine aliyoruz: ekranin okudugu bir cumle
/// degil, birinin sordugu bir soru gibi dursun.
class _Balon extends StatelessWidget {
  const _Balon({required this.soru});

  final String soru;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        color: HatirlaColors.primarySoft,
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        border: Border.all(color: HatirlaColors.line, width: 2),
      ),
      child: Text(
        soru,
        textAlign: TextAlign.center,
        style: const TextStyle(
          fontSize: 27,
          height: 1.25,
          fontWeight: FontWeight.w700,
          color: HatirlaColors.primaryDark,
        ),
      ),
    );
  }
}

/// Tek bir akraba secenegi: tam genislikte, ikonlu, yazili.
class _AkrabaDugmesi extends StatelessWidget {
  const _AkrabaDugmesi({required this.akraba, required this.onSec});

  final Akraba akraba;
  final VoidCallback onSec;

  static const Map<Akraba, IconData> _ikonlar = <Akraba, IconData>{
    Akraba.anneanne: Icons.elderly_woman_rounded,
    Akraba.babaanne: Icons.elderly_woman_rounded,
    Akraba.sukruDede: Icons.elderly_rounded,
    Akraba.oktayDede: Icons.elderly_rounded,
  };

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: akraba.ad,
      child: Material(
        color: HatirlaColors.card,
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        child: InkWell(
          borderRadius: BorderRadius.circular(HatirlaSizes.radius),
          onTap: () {
            HapticFeedback.mediumImpact();
            onSec();
          },
          child: Container(
            constraints: const BoxConstraints(minHeight: 84),
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(HatirlaSizes.radius),
              border: Border.all(color: HatirlaColors.primary, width: 2.5),
            ),
            child: Row(
              children: <Widget>[
                Icon(_ikonlar[akraba], size: 40, color: HatirlaColors.primary),
                const SizedBox(width: 16),
                Expanded(
                  child: Text(
                    akraba.ad,
                    style: const TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w700,
                      color: HatirlaColors.ink,
                    ),
                  ),
                ),
                const Icon(Icons.chevron_right_rounded,
                    size: 36, color: HatirlaColors.primary),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
