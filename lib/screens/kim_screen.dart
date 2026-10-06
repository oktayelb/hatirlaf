import 'dart:math' as math;

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

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
      evetIkon: CupertinoIcons.checkmark_alt,
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
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              const Center(child: _TorunYuzu()),
              const SizedBox(height: 10),
              Text(
                '$kTorunAdi soruyor',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 19,
                  fontWeight: FontWeight.w600,
                  color: HatirlaColors.inkSoft,
                ),
              ),
              const SizedBox(height: 4),
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
              const SizedBox(height: 16),
              for (final Akraba a in Akraba.values) ...<Widget>[
                _AkrabaDugmesi(akraba: a, onSec: () => _sec(context, a)),
                const SizedBox(height: 12),
              ],
            ],
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
    return CerceveliFoto(
      genislik: 112,
      yukseklik: 100,
      yaricap: 30,
      foto: Image.asset(
        kTorunFotografi,
        fit: BoxFit.cover,
        errorBuilder: (BuildContext context, Object e, StackTrace? s) =>
            const ColoredBox(
          color: HatirlaColors.primarySoft,
          child: Icon(CupertinoIcons.person_fill,
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

  static const Color _balonRengi = Color(0xFFE9E9EB);

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.topCenter,
      clipBehavior: Clip.none,
      children: <Widget>[
        Container(
          width: double.infinity,
          margin: const EdgeInsets.only(top: 9),
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
          decoration: ShapeDecoration(
            color: _balonRengi,
            shape: yumusakKose(26),
          ),
          child: Text(
            soru,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 27,
              height: 1.25,
              fontWeight: FontWeight.w700,
              letterSpacing: -0.4,
              color: HatirlaColors.ink,
            ),
          ),
        ),
        Positioned(
          top: 0,
          child: Transform.rotate(
            angle: math.pi / 4,
            child: Container(
              width: 18,
              height: 18,
              decoration: BoxDecoration(
                color: _balonRengi,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),
        ),
      ],
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
      onTap: onSec,
      excludeSemantics: true,
      child: Kart(
        onTap: onSec,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: ConstrainedBox(
          constraints: const BoxConstraints(minHeight: 60),
          child: Row(
            children: <Widget>[
              Container(
                width: 56,
                height: 56,
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: <Color>[Color(0xFFA8AEBA), Color(0xFF858A96)],
                  ),
                ),
                child: Icon(_ikonlar[akraba], size: 36, color: Colors.white),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Text(
                  akraba.ad,
                  style: const TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.4,
                    color: HatirlaColors.ink,
                  ),
                ),
              ),
              const Icon(CupertinoIcons.chevron_forward,
                  size: 28, color: HatirlaColors.chevron),
            ],
          ),
        ),
      ),
    );
  }
}
