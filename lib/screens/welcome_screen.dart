import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../main.dart';
import '../services/permissions.dart';
import '../services/recorder.dart';
import '../services/whisper_model_manager.dart';
import '../theme.dart';
import '../utils/format.dart';
import '../widgets/common.dart';
import 'home_screen.dart';

/// Ilk acilis: her adimda tek ana buton, hicbir adim zorunlu degil, ve her
/// izin penceresi once sade bir cumleyle aciklaniyor - habersiz cikan bir
/// pencerede refleksle "Reddet"e basiliyor.
class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen> {
  final PageController _pageController = PageController();
  int _adim = 0;
  bool _mikrofonVerildi = false;
  bool _izinSoruluyor = false;

  static const int _toplamAdim = 4;

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  void _ilerle() {
    if (_adim >= _toplamAdim - 1) {
      _bitir();
      return;
    }
    setState(() => _adim++);
    _pageController.animateToPage(
      _adim,
      duration: const Duration(milliseconds: 450),
      curve: Curves.easeInOutCubic,
    );
  }

  Future<void> _bitir() async {
    try {
      final SharedPreferences prefs = await SharedPreferences.getInstance();
      await prefs.setBool(kKarsilamaAnahtari, true);
    } catch (_) {
      // Kaydedilemezse karsilama bir daha cikar.
    }
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute<void>(builder: (_) => const HomeScreen()),
      (Route<dynamic> route) => false,
    );
  }

  Future<void> _mikrofonIste() async {
    if (_izinSoruluyor) return;
    setState(() => _izinSoruluyor = true);
    bool verildi = false;
    try {
      verildi = await Recorder.instance.izinVarMi();
    } finally {
      if (mounted) {
        setState(() {
          _izinSoruluyor = false;
          _mikrofonVerildi = verildi;
        });
      }
    }
    if (!mounted) return;

    if (verildi) {
      _ilerle();
      return;
    }

    // "Bir daha sorma" dendiyse sistem penceresi acilmaz; ayarlara goturuyoruz.
    final bool kalici = await Izinler.kaliciReddedildiMi();
    if (!mounted) return;
    if (kalici) {
      final bool git = await onayIste(
        context,
        baslik: 'Mikrofon kapalı',
        mesaj: 'Sesinizi kaydedebilmemiz için mikrofon izni gerekiyor.\n\n'
            'Telefon ayarlarını açıp “İzinler” bölümünden Mikrofon’u '
            'açmamız gerek. Sizin için açayım mı?',
        evetYazi: 'Ayarları Aç',
        evetIkon: CupertinoIcons.gear_alt_fill,
      );
      if (git) await Izinler.ayarlariAc();
    } else {
      await bilgiGoster(
        context,
        baslik: 'Mikrofon gerekli',
        mesaj: 'Mikrofon izni verilmeden ses kaydı yapılamıyor. '
            'İsterseniz şimdi geçebilir, sonra tekrar deneyebilirsiniz.',
        ikon: CupertinoIcons.mic_slash_fill,
        renk: HatirlaColors.record,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: <Widget>[
            _AdimGostergesi(adim: _adim, toplam: _toplamAdim),
            Expanded(
              child: PageView(
                controller: _pageController,
                // Kaydirarak gecis yok: yanlislikla adim atlanmasin.
                physics: const NeverScrollableScrollPhysics(),
                children: <Widget>[
                  _hosGeldiniz(),
                  _mikrofonAdimi(),
                  _indirmeAdimi(),
                  _hazir(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _hosGeldiniz() {
    return _AdimGovdesi(
      ikon: CupertinoIcons.heart_fill,
      ikonRengi: HatirlaColors.record,
      baslik: 'Hoş geldiniz',
      metin: 'Bu uygulama sizin hayat hikâyenizi saklamak için yapıldı.\n\n'
          'Siz anlatırsınız, telefon hem sesinizi kaydeder hem de '
          'söylediklerinizi yazıya döker.\n\n'
          'Çocuklarınız, torunlarınız yıllar sonra da sizi '
          'kendi sesinizden dinleyebilir.',
      buton: BuyukButon(
        yazi: 'Başlayalım',
        ikon: CupertinoIcons.arrow_right,
        onPressed: _ilerle,
      ),
    );
  }

  Widget _mikrofonAdimi() {
    return _AdimGovdesi(
      ikon: _mikrofonVerildi
          ? CupertinoIcons.checkmark_alt
          : CupertinoIcons.mic_fill,
      ikonRengi:
          _mikrofonVerildi ? HatirlaColors.confirm : HatirlaColors.primary,
      baslik: 'Sesinizi duyabilmemiz için',
      metin: 'Şimdi telefon size “mikrofona izin veriyor musunuz?” diye '
          'soracak.\n\nLütfen “İzin Ver”e dokunun. Sesiniz telefondan '
          'dışarı çıkmaz, sadece bu uygulamada saklanır.',
      buton: BuyukButon(
        yazi: _izinSoruluyor ? 'Bekleyin…' : 'Mikrofonu Aç',
        ikon: CupertinoIcons.mic_fill,
        renk: HatirlaColors.confirm,
        onPressed: _izinSoruluyor ? null : _mikrofonIste,
      ),
      atla: DuzButon(
        yazi: 'Şimdi değil',
        renk: HatirlaColors.inkSoft,
        onPressed: _ilerle,
      ),
    );
  }

  Widget _indirmeAdimi() {
    return ListenableBuilder(
      listenable: WhisperModelManager.instance,
      builder: (BuildContext context, _) {
        final WhisperModelManager mm = WhisperModelManager.instance;
        final bool hazir = mm.durum == IndirmeDurumu.hazir;
        final bool iniyor = mm.durum == IndirmeDurumu.iniyor;

        String metin;
        if (hazir) {
          metin = 'Hazır! Artık konuşmalarınız telefonun kendi içinde '
              'yazıya çevrilecek. İnternet gerekmiyor.';
        } else if (iniyor) {
          metin = 'İndiriliyor, lütfen bekleyin.\n\n'
              'Bu ekranda kalmanız yeterli. İnternet bağlantınızı '
              'kesmeyin.';
        } else {
          metin = 'Söylediklerinizi yazıya çevirebilmek için bir kereliğine '
              '${WhisperModelManager.yaklasikMb} MB’lık bir paket indirmemiz gerekiyor.\n\n'
              'Mümkünse Wi-Fi’ye bağlıyken indirin. Bir daha '
              'istenmeyecek.';
        }

        return _AdimGovdesi(
          ikon: hazir
              ? CupertinoIcons.checkmark_alt
              : CupertinoIcons.cloud_download_fill,
          ikonRengi: hazir ? HatirlaColors.confirm : HatirlaColors.primary,
          baslik: hazir ? 'Yazıya çevirme hazır' : 'Yazıya çevirme paketi',
          metin: metin,
          ekstra: iniyor
              ? _IndirmeCubugu(
                  ilerleme: mm.ilerleme,
                  inen: mm.inenBayt,
                  toplam: mm.toplamBayt,
                )
              : (mm.durum == IndirmeDurumu.hata && mm.hata != null
                  ? _HataKutusu(mesaj: mm.hata!)
                  : null),
          buton: BuyukButon(
            yazi: hazir
                ? 'Devam'
                : (iniyor ? 'İndiriliyor…' : 'Paketi İndir'),
            ikon: hazir
                ? CupertinoIcons.arrow_right
                : CupertinoIcons.cloud_download_fill,
            renk: hazir ? HatirlaColors.confirm : HatirlaColors.primary,
            onPressed: iniyor
                ? null
                : () async {
                    if (hazir) {
                      _ilerle();
                      return;
                    }
                    final bool ok = await mm.download();
                    if (!mounted) return;
                    if (ok) _ilerle();
                  },
          ),
          atla: iniyor
              ? DuzButon(
                  yazi: 'İndirmeyi durdur',
                  renk: HatirlaColors.record,
                  onPressed: () => WhisperModelManager.instance.iptalEt(),
                )
              : (hazir
                  ? null
                  : DuzButon(
                      yazi: 'Sonra indireyim',
                      renk: HatirlaColors.inkSoft,
                      onPressed: _ilerle,
                    )),
        );
      },
    );
  }

  Widget _hazir() {
    return _AdimGovdesi(
      ikon: CupertinoIcons.book_fill,
      ikonRengi: HatirlaColors.primary,
      baslik: 'Her şey hazır',
      metin: 'Artık anlatmaya başlayabilirsiniz.\n\n'
          'Ne anlatacağınızı bilemezseniz merak etmeyin: uygulama size '
          'sorular soracak. Siz sadece cevaplayın.',
      buton: BuyukButon(
        yazi: 'İlk Hatırayı Anlat',
        ikon: CupertinoIcons.mic_fill,
        renk: HatirlaColors.confirm,
        onPressed: _bitir,
      ),
    );
  }
}

/// Ustteki "1 / 4" cubugu. Nokta degil cubuk: noktalar secilemiyor.
class _AdimGostergesi extends StatelessWidget {
  const _AdimGostergesi({required this.adim, required this.toplam});

  final int adim;
  final int toplam;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 16, 24, 4),
      child: Row(
        children: <Widget>[
          for (int i = 0; i < toplam; i++) ...<Widget>[
            Expanded(
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 400),
                curve: Curves.easeInOutCubic,
                height: 6,
                decoration: BoxDecoration(
                  color: i <= adim
                      ? HatirlaColors.primary
                      : HatirlaColors.paperDark,
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
            ),
            if (i != toplam - 1) const SizedBox(width: 8),
          ],
        ],
      ),
    );
  }
}

/// Tum karsilama adimlarinin ortak duzeni.
class _AdimGovdesi extends StatelessWidget {
  const _AdimGovdesi({
    required this.ikon,
    required this.baslik,
    required this.metin,
    required this.buton,
    this.ikonRengi = HatirlaColors.primary,
    this.ekstra,
    this.atla,
  });

  final IconData ikon;
  final Color ikonRengi;
  final String baslik;
  final String metin;
  final Widget buton;
  final Widget? ekstra;
  final Widget? atla;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 8, 24, 20),
      child: Column(
        children: <Widget>[
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: <Widget>[
                  const SizedBox(height: 36),
                  Center(
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      transitionBuilder: (Widget child, Animation<double> a) =>
                          ScaleTransition(
                        scale: Tween<double>(begin: 0.8, end: 1).animate(a),
                        child: FadeTransition(opacity: a, child: child),
                      ),
                      child: BuyukSimge(
                        key: ValueKey<IconData>(ikon),
                        ikon: ikon,
                        renk: ikonRengi,
                      ),
                    ),
                  ),
                  const SizedBox(height: 32),
                  Text(
                    baslik,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.headlineLarge,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    metin,
                    textAlign: TextAlign.center,
                    style: Theme.of(context)
                        .textTheme
                        .bodyLarge
                        ?.copyWith(color: HatirlaColors.inkSoft),
                  ),
                  if (ekstra != null) ...<Widget>[
                    const SizedBox(height: 24),
                    ekstra!,
                  ],
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
          buton,
          if (atla != null) ...<Widget>[
            const SizedBox(height: 4),
            atla!,
          ],
        ],
      ),
    );
  }
}

class _IndirmeCubugu extends StatelessWidget {
  const _IndirmeCubugu({
    required this.ilerleme,
    required this.inen,
    required this.toplam,
  });

  final double? ilerleme;
  final int inen;
  final int toplam;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: <Widget>[
        IlerlemeCubugu(deger: ilerleme, yukseklik: 14),
        const SizedBox(height: 14),
        Text(
          ilerleme == null
              ? '${Bicim.boyut(inen)} indirildi'
              : '%${(ilerleme! * 100).toStringAsFixed(0)}  ·  '
                  '${Bicim.boyut(inen)} / ${Bicim.boyut(toplam)}',
          style: const TextStyle(
            fontSize: 23,
            fontWeight: FontWeight.w700,
            fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
            color: HatirlaColors.primaryDark,
          ),
        ),
      ],
    );
  }
}

class _HataKutusu extends StatelessWidget {
  const _HataKutusu({required this.mesaj});

  final String mesaj;

  @override
  Widget build(BuildContext context) {
    return Kart(
      renk: HatirlaColors.recordSoft,
      golge: false,
      padding: const EdgeInsets.all(18),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          const Icon(CupertinoIcons.exclamationmark_circle_fill,
              color: HatirlaColors.record, size: 30),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              mesaj,
              style: const TextStyle(fontSize: 20, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}
