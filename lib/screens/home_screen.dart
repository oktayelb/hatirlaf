import 'dart:async';
import 'dart:io';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../data/prompts.dart';
import '../models/memory.dart';
import '../services/cover_photo.dart';
import '../services/recorder.dart';
import '../services/store.dart';
import '../services/transcriber.dart';
import '../services/updater.dart';
import '../services/uploader.dart';
import '../services/whisper_model_manager.dart';
import '../theme.dart';
import '../utils/format.dart';
import '../widgets/common.dart';
import '../widgets/memory_card.dart';
import 'memory_screen.dart';
import 'question_screen.dart';
import 'record_screen.dart';
import 'settings_screen.dart';
import 'update_screen.dart';

/// Ana ekran: hatira listesi + her zaman gorunen buyuk kayit butonu.
/// Sekme, menu, alt cubuk yok; gezinme derinligi en fazla iki.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late String _gununSorusu;
  bool _fotografYukleniyor = false;
  final ScrollController _kaydirma = ScrollController();
  bool _baslikUstte = false;

  @override
  void initState() {
    super.initState();
    _gununSorusu = Sorular.rastgeleSoru();
    _kaydirma.addListener(_kaydirmaDegisti);
  }

  @override
  void dispose() {
    _kaydirma.dispose();
    super.dispose();
  }

  void _kaydirmaDegisti() {
    final bool ustte = _kaydirma.offset > 56;
    if (ustte != _baslikUstte) setState(() => _baslikUstte = ustte);
  }

  /// Kamera mi galeri mi? Iki buyuk, yazili, ikonlu buton.
  Future<void> _kapakFotografiSec() async {
    final ImageSource? kaynak = await showModalBottomSheet<ImageSource>(
      context: context,
      builder: (BuildContext context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              Text(
                'Fotoğraf seç',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 20),
              BuyukButon(
                yazi: 'Fotoğraf Çek',
                altYazi: 'Kamerayı aç',
                ikon: CupertinoIcons.camera_fill,
                onPressed: () => Navigator.of(context).pop(ImageSource.camera),
              ),
              const SizedBox(height: 12),
              IkincilButon(
                yazi: 'Galeriden Seç',
                ikon: CupertinoIcons.photo_on_rectangle,
                onPressed: () => Navigator.of(context).pop(ImageSource.gallery),
              ),
              const SizedBox(height: 4),
              DuzButon(
                yazi: 'Vazgeç',
                renk: HatirlaColors.inkSoft,
                onPressed: () => Navigator.of(context).pop(),
              ),
            ],
          ),
        ),
      ),
    );
    if (kaynak == null || !mounted) return;

    setState(() => _fotografYukleniyor = true);
    try {
      final XFile? secilen = await ImagePicker().pickImage(
        source: kaynak,
        // 12 MP bosuna yer yiyor; kapak icin bu yeter.
        maxWidth: 1600,
        imageQuality: 85,
      );
      if (secilen == null) return;
      final bool oldu = await CoverPhoto.instance.ayarla(secilen.path);
      if (!oldu && mounted) {
        kisaMesaj(context, 'Fotoğraf kaydedilemedi. Tekrar deneyin.');
      }
    } catch (e) {
      if (!mounted) return;
      kisaMesaj(
        context,
        kaynak == ImageSource.camera
            ? 'Kamera açılamadı. İzin verilmemiş olabilir.'
            : 'Fotoğraf seçilemedi. Tekrar deneyin.',
      );
    } finally {
      if (mounted) setState(() => _fotografYukleniyor = false);
    }
  }

  Future<void> _kapakFotografiniSil() async {
    final bool emin = await onayIste(
      context,
      baslik: 'Fotoğrafı kaldıralım mı?',
      mesaj: 'Fotoğraf silinecek. Hatıralarınıza bir şey olmaz.',
      evetYazi: 'Kaldır',
      hayirYazi: 'Vazgeç',
      evetIkon: CupertinoIcons.trash,
      tehlikeli: true,
    );
    if (!emin) return;
    await CoverPhoto.instance.sil();
  }

  Future<void> _anlatmayaBasla({String? soru}) async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => RecordScreen(soru: soru)),
    );
    if (!mounted) return;
    // Anlatildiktan sonra ayni soruyu tekrar onermeyelim.
    setState(() => _gununSorusu = Sorular.rastgeleSoru(haric: _gununSorusu));
  }

  Future<void> _tumSorulariAc() async {
    final String? secilen = await Navigator.of(context).push<String>(
      MaterialPageRoute<String>(builder: (_) => const QuestionScreen()),
    );
    if (secilen != null && mounted) {
      await _anlatmayaBasla(soru: secilen);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: UstCubuk(
        baslik: 'Hatıralarım',
        baslikGorunur: _baslikUstte,
        eylemler: <Widget>[
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: _AyarlarDugmesi(
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => const SettingsScreen()),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: <Widget>[
          const _GuncellemeGozcusu(),
          Expanded(
            child: ListenableBuilder(
              listenable: MemoryStore.instance,
              builder: (BuildContext context, _) {
                final List<Memory> hatiralar = MemoryStore.instance.memories;
                return ListView(
                  controller: _kaydirma,
                  padding: const EdgeInsets.fromLTRB(
                    HatirlaSizes.gutter,
                    0,
                    HatirlaSizes.gutter,
                    28,
                  ),
                  children: <Widget>[
                    const _BuyukBaslik(),
                    const _ModelUyarisi(),
                    _KapakFotografi(
                      yukleniyor: _fotografYukleniyor,
                      onSec: _kapakFotografiSec,
                      onSil: _kapakFotografiniSil,
                    ),
                    const SizedBox(height: 24),
                    _SoruKarti(
                      soru: _gununSorusu,
                      onAnlat: () => _anlatmayaBasla(soru: _gununSorusu),
                      onBaskaSoru: () => setState(() {
                        _gununSorusu =
                            Sorular.rastgeleSoru(haric: _gununSorusu);
                      }),
                      onTumSorular: _tumSorulariAc,
                    ),
                    const SizedBox(height: 32),
                    if (hatiralar.isEmpty)
                      const _BosDurum()
                    else ...<Widget>[
                      BolumBasligi(
                        hatiralar.length == 1
                            ? '1 hatıra'
                            : '${hatiralar.length} hatıra',
                      ),
                      const SizedBox(height: 14),
                      for (final Memory m in hatiralar)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 14),
                          child: MemoryCard(
                            memory: m,
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute<void>(
                                builder: (_) => MemoryScreen(memoryId: m.id),
                              ),
                            ),
                          ),
                        ),
                    ],
                  ],
                );
              },
            ),
          ),
          _AltKayitCubugu(onBasla: () => _anlatmayaBasla()),
        ],
      ),
    );
  }
}

class _AyarlarDugmesi extends StatelessWidget {
  const _AyarlarDugmesi({required this.onPressed});

  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Ayarlar',
      onTap: onPressed,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onPressed,
        olcek: 0.9,
        child: SizedBox(
          width: 64,
          height: 64,
          child: Center(
            child: Container(
              width: 50,
              height: 50,
              decoration: const BoxDecoration(
                shape: BoxShape.circle,
                color: HatirlaColors.paperDark,
              ),
              child: const Icon(
                CupertinoIcons.gear_alt_fill,
                size: 30,
                color: HatirlaColors.inkSoft,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _BuyukBaslik extends StatelessWidget {
  const _BuyukBaslik();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 0, 4, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            Bicim.uzunTarih(DateTime.now()),
            style: const TextStyle(
              fontSize: 19,
              fontWeight: FontWeight.w600,
              color: HatirlaColors.inkSoft,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            'Hatıralarım',
            style: Theme.of(context).textTheme.headlineLarge,
          ),
        ],
      ),
    );
  }
}

/// Her zaman ekranin altinda duran ana buton: listenin icinde olsaydi
/// asagi kaydirmayi bilmeyen kullanici onu kaybederdi.
class _AltKayitCubugu extends StatelessWidget {
  const _AltKayitCubugu({required this.onBasla});

  final VoidCallback onBasla;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(
        color: Color(0xFFF9F9FB),
        border: Border(top: BorderSide(color: HatirlaColors.line)),
      ),
      child: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
            HatirlaSizes.gutter,
            12,
            HatirlaSizes.gutter,
            14,
          ),
          child: BuyukButon(
            yazi: 'Anlatmaya Başla',
            altYazi: 'Dokunun ve konuşun',
            ikon: CupertinoIcons.mic_fill,
            renk: HatirlaColors.record,
            yukseklik: 92,
            onPressed: onBasla,
          ),
        ),
      ),
    );
  }
}

/// Ana sayfanin en ustundeki tek kapak fotografi; hatiralara bagli degil.
class _KapakFotografi extends StatelessWidget {
  const _KapakFotografi({
    required this.yukleniyor,
    required this.onSec,
    required this.onSil,
  });

  final bool yukleniyor;
  final VoidCallback onSec;
  final VoidCallback onSil;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: CoverPhoto.instance,
      builder: (BuildContext context, _) {
        final CoverPhoto kapak = CoverPhoto.instance;
        return AnimatedSwitcher(
          duration: const Duration(milliseconds: 350),
          switchInCurve: Curves.easeOutCubic,
          switchOutCurve: Curves.easeInCubic,
          child: kapak.varMi
              ? _KapakVar(
                  // Dosya adi hep ayni; surum anahtari olmadan Flutter eski
                  // kareyi onbellekten gosterir.
                  key: ValueKey<int>(kapak.surum),
                  yol: kapak.yol,
                  yukleniyor: yukleniyor,
                  onSec: onSec,
                  onSil: onSil,
                )
              : _KapakYok(yukleniyor: yukleniyor, onSec: onSec),
        );
      },
    );
  }
}

class _KapakYok extends StatelessWidget {
  const _KapakYok({required this.yukleniyor, required this.onSec});

  final bool yukleniyor;
  final VoidCallback onSec;

  @override
  Widget build(BuildContext context) {
    final String yazi = yukleniyor ? 'Ekleniyor…' : 'Fotoğraf Ekle';
    return Semantics(
      button: true,
      label: yazi,
      onTap: yukleniyor ? null : onSec,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: yukleniyor ? null : onSec,
        child: CerceveliFoto(
          yukseklik: 150,
          foto: ColoredBox(
            color: const Color(0xFFF7F7FA),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Container(
                  width: 60,
                  height: 60,
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    color: HatirlaColors.primarySoft,
                  ),
                  child: yukleniyor
                      ? const CupertinoActivityIndicator(radius: 12)
                      : const Icon(CupertinoIcons.camera_fill,
                          size: 30, color: HatirlaColors.primary),
                ),
                const SizedBox(height: 12),
                Text(
                  yazi,
                  style: const TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                    color: HatirlaColors.primaryDark,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _KapakVar extends StatelessWidget {
  const _KapakVar({
    super.key,
    required this.yol,
    required this.yukleniyor,
    required this.onSec,
    required this.onSil,
  });

  final String yol;
  final bool yukleniyor;
  final VoidCallback onSec;
  final VoidCallback onSil;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        CerceveliFoto(
          yukseklik: 220,
          foto: Image.file(
            File(yol),
            fit: BoxFit.cover,
            errorBuilder: (_, _, _) => const ColoredBox(
              color: HatirlaColors.paperDark,
              child: Center(
                child: Icon(CupertinoIcons.photo,
                    size: 56, color: HatirlaColors.inkSoft),
              ),
            ),
          ),
        ),
        const SizedBox(height: 4),
        Row(
          children: <Widget>[
            Expanded(
              child: DuzButon(
                yazi: yukleniyor ? 'Ekleniyor…' : 'Değiştir',
                ikon: CupertinoIcons.photo_on_rectangle,
                onPressed: yukleniyor ? null : onSec,
              ),
            ),
            Expanded(
              child: DuzButon(
                yazi: 'Kaldır',
                ikon: CupertinoIcons.trash,
                renk: HatirlaColors.record,
                onPressed: yukleniyor ? null : onSil,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

/// "Günün sorusu" karti.
class _SoruKarti extends StatelessWidget {
  const _SoruKarti({
    required this.soru,
    required this.onAnlat,
    required this.onBaskaSoru,
    required this.onTumSorular,
  });

  final String soru;
  final VoidCallback onAnlat;
  final VoidCallback onBaskaSoru;
  final VoidCallback onTumSorular;

  @override
  Widget build(BuildContext context) {
    return Kart(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const Row(
            children: <Widget>[
              IkonRozeti(
                ikon: CupertinoIcons.quote_bubble_fill,
                renk: HatirlaColors.primary,
                boyut: 38,
              ),
              SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Bugünün sorusu',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w600,
                    color: HatirlaColors.primary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          AnimatedSize(
            duration: const Duration(milliseconds: 320),
            curve: Curves.easeInOutCubic,
            alignment: Alignment.topCenter,
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 320),
              layoutBuilder: (Widget? simdiki, List<Widget> onceki) {
                final List<Widget> hepsi = <Widget>[...onceki];
                if (simdiki != null) hepsi.add(simdiki);
                return Stack(alignment: Alignment.topLeft, children: hepsi);
              },
              transitionBuilder: (Widget child, Animation<double> a) =>
                  FadeTransition(
                opacity: a,
                child: SlideTransition(
                  position: Tween<Offset>(
                    begin: const Offset(0, 0.08),
                    end: Offset.zero,
                  ).animate(a),
                  child: child,
                ),
              ),
              child: SizedBox(
                key: ValueKey<String>(soru),
                width: double.infinity,
                child: Text(
                  soru,
                  style: const TextStyle(
                    fontSize: 27,
                    height: 1.3,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.4,
                    color: HatirlaColors.ink,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 20),
          BuyukButon(
            yazi: 'Bunu Anlat',
            ikon: CupertinoIcons.mic_fill,
            yukseklik: 76,
            onPressed: onAnlat,
          ),
          const SizedBox(height: 4),
          Row(
            children: <Widget>[
              Expanded(
                child: DuzButon(
                  yazi: 'Başka soru',
                  ikon: CupertinoIcons.arrow_clockwise,
                  onPressed: onBaskaSoru,
                ),
              ),
              Expanded(
                child: DuzButon(
                  yazi: 'Tüm sorular',
                  ikon: CupertinoIcons.list_bullet,
                  onPressed: onTumSorular,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Hic hatira yokken gosterilen sicak karsilama.
class _BosDurum extends StatelessWidget {
  const _BosDurum();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 12),
      child: Column(
        children: <Widget>[
          const BuyukSimge(
            ikon: CupertinoIcons.book_fill,
            renk: HatirlaColors.primary,
            boyut: 96,
          ),
          const SizedBox(height: 22),
          Text(
            'Henüz hatıra yok',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 10),
          Text(
            'Aşağıdaki kırmızı düğmeye dokunup ilk hatıranızı anlatın.\n'
            'Kaydettiğiniz her hatıra burada birikecek.',
            textAlign: TextAlign.center,
            style: Theme.of(context)
                .textTheme
                .bodyLarge
                ?.copyWith(color: HatirlaColors.inkSoft),
          ),
        ],
      ),
    );
  }
}

/// Model indirilmediyse ustte duran, tek dokunusla cozulen uyari serigi.
class _ModelUyarisi extends StatelessWidget {
  const _ModelUyarisi();

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: WhisperModelManager.instance,
      builder: (BuildContext context, _) {
        final WhisperModelManager mm = WhisperModelManager.instance;
        final bool iniyor = mm.durum == IndirmeDurumu.iniyor;
        return AnimatedSize(
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeInOutCubic,
          child: mm.durum == IndirmeDurumu.hazir
              ? const SizedBox(width: double.infinity)
              : Padding(
                  padding: const EdgeInsets.only(bottom: 20),
                  child: Kart(
                    renk: HatirlaColors.warningSoft,
                    golge: false,
                    padding: const EdgeInsets.fromLTRB(18, 18, 18, 14),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: <Widget>[
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: <Widget>[
                            Icon(
                              iniyor
                                  ? CupertinoIcons.cloud_download_fill
                                  : CupertinoIcons.exclamationmark_triangle_fill,
                              size: 30,
                              color: HatirlaColors.warning,
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                iniyor
                                    ? 'Yazıya çevirme paketi iniyor… '
                                        '%${((mm.ilerleme ?? 0) * 100).toStringAsFixed(0)}'
                                    : mm.hata ??
                                        'Ses kayıtları yazıya çevrilmiyor. '
                                            'Yaklaşık ${WhisperModelManager.yaklasikMb} MB’lık '
                                            'paket henüz indirilmedi.',
                                style: const TextStyle(fontSize: 20, height: 1.4),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        if (iniyor)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 6),
                            child: IlerlemeCubugu(
                              deger: mm.ilerleme,
                              renk: HatirlaColors.warning,
                              yukseklik: 10,
                            ),
                          )
                        else
                          IkincilButon(
                            yazi: 'Paketi İndir',
                            ikon: CupertinoIcons.cloud_download,
                            renk: HatirlaColors.warning,
                            onPressed: () => mm.download(),
                          ),
                      ],
                    ),
                  ),
                ),
        );
      },
    );
  }
}

/// Guncelleme ekranini dogru anda acan gorunmez gozcu. Kurulum surecin
/// kendisini degistirdigi icin o an suren her sey yarida kalir; bu yuzden
/// bes kapi var: ana ekran ustte mi, kayit suruyor mu, yaziya cevirme
/// suruyor mu, [Guncelleyici.sorulabilir] ve [Guncelleyici.acilistaHazirdi].
/// Hicbiri uygun degilse guncelleme diskte bekler.
class _GuncellemeGozcusu extends StatefulWidget {
  const _GuncellemeGozcusu();

  @override
  State<_GuncellemeGozcusu> createState() => _GuncellemeGozcusuState();
}

class _GuncellemeGozcusuState extends State<_GuncellemeGozcusu>
    with WidgetsBindingObserver {
  /// Ayni acilista tekrar tekrar acilmasin.
  bool _acildi = false;

  late final Listenable _dinlenecekler = Listenable.merge(<Listenable>[
    Guncelleyici.instance,
    Transcriber.instance,
    Recorder.instance,
  ]);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _dinlenecekler.addListener(_belkiAc);
    WidgetsBinding.instance.addPostFrameCallback((_) => _belkiAc());
  }

  @override
  void dispose() {
    _dinlenecekler.removeListener(_belkiAc);
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState durum) {
    if (durum != AppLifecycleState.resumed) return;
    // On plana donmek yeni bir oturum sayilir.
    Guncelleyici.instance.oturumaGirildi();
    // Uygulama gunlerce arka planda kalmis olabilir; donunce yeniden bak.
    unawaited(Guncelleyici.instance.degerlendir());
    // Yedekleme sessiz: kosullar uygun degilse kendi doner.
    unawaited(Yedekleyici.instance.degerlendir());
    _belkiAc();
  }

  void _belkiAc() {
    if (_acildi || !mounted) return;
    if (!Guncelleyici.instance.sorulabilir) return;
    if (!Guncelleyici.instance.acilistaHazirdi) return;
    if (Recorder.instance.durum != KayitDurumu.bos) return;
    if (Transcriber.instance.mesgul) return;

    // Cizim sirasinda gezinme yapilamaz, kareden sonraya birakiyoruz.
    // [addPostFrameCallback] yalnizca bir kare cizilirse calisir; durgun
    // ekranda Flutter kare uretmez ve geri cagirma sonsuza kadar beklerdi.
    // [ensureVisualUpdate] gerekirse bir kare planlar.
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      if (_acildi || !mounted) return;
      // Ana ekran en ustte degilse kullanicinin isini bolmeyelim.
      if (ModalRoute.of(context)?.isCurrent != true) return;
      // Beklerken kayit baslamis olabilir.
      if (Recorder.instance.durum != KayitDurumu.bos) return;
      if (Transcriber.instance.mesgul) return;

      _acildi = true;
      await Navigator.of(context).push(
        MaterialPageRoute<void>(builder: (_) => const UpdateScreen()),
      );
    });
    WidgetsBinding.instance.ensureVisualUpdate();
  }

  @override
  Widget build(BuildContext context) => const SizedBox.shrink();
}
