import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../data/prompts.dart';
import '../models/memory.dart';
import '../services/cover_photo.dart';
import '../services/recorder.dart';
import '../services/store.dart';
import '../services/transcriber.dart';
import '../services/updater.dart';
import '../services/whisper_model_manager.dart';
import '../theme.dart';
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

  @override
  void initState() {
    super.initState();
    _gununSorusu = Sorular.rastgeleSoru();
  }

  /// Kamera mi galeri mi? Iki buyuk, yazili, ikonlu buton.
  Future<void> _kapakFotografiSec() async {
    final ImageSource? kaynak = await showModalBottomSheet<ImageSource>(
      context: context,
      backgroundColor: HatirlaColors.card,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
      ),
      builder: (BuildContext context) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Text(
                'Fotoğraf seç',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 20),
              BuyukButon(
                yazi: 'Fotoğraf Çek',
                altYazi: 'Kamerayı aç',
                ikon: Icons.photo_camera_rounded,
                onPressed: () => Navigator.of(context).pop(ImageSource.camera),
              ),
              const SizedBox(height: 12),
              CerceveliButon(
                yazi: 'Galeriden Seç',
                ikon: Icons.photo_library_rounded,
                onPressed: () => Navigator.of(context).pop(ImageSource.gallery),
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
      evetIkon: Icons.delete_outline_rounded,
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hatıralarım'),
        actions: <Widget>[
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: IconButton(
              tooltip: 'Ayarlar',
              iconSize: 36,
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => const SettingsScreen()),
              ),
              icon: const Icon(Icons.settings_rounded),
            ),
          ),
        ],
      ),
      body: SafeArea(
        top: false,
        child: Column(
          children: <Widget>[
            const _GuncellemeGozcusu(),
            const _ModelUyarisi(),
            Expanded(
              child: ListenableBuilder(
                listenable: MemoryStore.instance,
                builder: (BuildContext context, _) {
                  final List<Memory> hatiralar = MemoryStore.instance.memories;
                  return ListView(
                    padding: const EdgeInsets.fromLTRB(
                      HatirlaSizes.gutter,
                      8,
                      HatirlaSizes.gutter,
                      16,
                    ),
                    children: <Widget>[
                      _KapakFotografi(
                        yukleniyor: _fotografYukleniyor,
                        onSec: _kapakFotografiSec,
                        onSil: _kapakFotografiniSil,
                      ),
                      const SizedBox(height: 26),
                      _SoruKarti(
                        soru: _gununSorusu,
                        onAnlat: () => _anlatmayaBasla(soru: _gununSorusu),
                        onBaskaSoru: () => setState(() {
                          _gununSorusu =
                              Sorular.rastgeleSoru(haric: _gununSorusu);
                        }),
                        onTumSorular: () async {
                          final String? secilen =
                              await Navigator.of(context).push<String>(
                            MaterialPageRoute<String>(
                              builder: (_) => const QuestionScreen(),
                            ),
                          );
                          if (secilen != null && mounted) {
                            await _anlatmayaBasla(soru: secilen);
                          }
                        },
                      ),
                      const SizedBox(height: 26),
                      if (hatiralar.isEmpty)
                        const _BosDurum()
                      else ...<Widget>[
                        Padding(
                          padding: const EdgeInsets.only(bottom: 14),
                          child: BolumBasligi(
                            hatiralar.length == 1
                                ? '1 hatıra'
                                : '${hatiralar.length} hatıra',
                            ikon: Icons.auto_stories_rounded,
                          ),
                        ),
                        for (final Memory m in hatiralar)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 16),
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
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: HatirlaColors.paper,
        border: Border(top: BorderSide(color: HatirlaColors.line, width: 2)),
      ),
      padding: const EdgeInsets.fromLTRB(
        HatirlaSizes.gutter,
        14,
        HatirlaSizes.gutter,
        18,
      ),
      child: BuyukButon(
        yazi: 'Anlatmaya Başla',
        altYazi: 'Dokunun ve konuşun',
        ikon: Icons.mic_rounded,
        renk: HatirlaColors.record,
        yukseklik: 96,
        onPressed: onBasla,
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
        if (!kapak.varMi) {
          return CerceveliButon(
            yazi: yukleniyor ? 'Ekleniyor…' : 'Fotoğraf Ekle',
            ikon: Icons.add_a_photo_rounded,
            onPressed: yukleniyor ? null : onSec,
          );
        }

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            ClipRRect(
              borderRadius: BorderRadius.circular(HatirlaSizes.radius),
              child: Image.file(
                File(kapak.yol),
                // Dosya adi hep ayni; surum anahtari olmadan Flutter eski
                // kareyi onbellekten gosterir.
                key: ValueKey<int>(kapak.surum),
                height: 260,
                fit: BoxFit.cover,
                errorBuilder: (_, _, _) => Container(
                  height: 260,
                  color: HatirlaColors.paperDark,
                  child: const Icon(Icons.broken_image_rounded,
                      size: 56, color: HatirlaColors.line),
                ),
              ),
            ),
            const SizedBox(height: 10),
            Row(
              children: <Widget>[
                Expanded(
                  child: TextButton.icon(
                    onPressed: yukleniyor ? null : onSec,
                    icon: const Icon(Icons.edit_rounded, size: 28),
                    label: Text(
                        yukleniyor ? 'Ekleniyor…' : 'Fotoğrafı değiştir'),
                  ),
                ),
                Expanded(
                  child: TextButton.icon(
                    onPressed: yukleniyor ? null : onSil,
                    icon: const Icon(Icons.delete_outline_rounded, size: 28),
                    label: const Text('Kaldır'),
                  ),
                ),
              ],
            ),
          ],
        );
      },
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
    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: HatirlaColors.primarySoft,
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        border: Border.all(color: HatirlaColors.primary, width: 2.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          Row(
            children: <Widget>[
              const Icon(Icons.record_voice_over_rounded,
                  size: 32, color: HatirlaColors.primaryDark),
              const SizedBox(width: 10),
              Text(
                'Bugünün sorusu',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: HatirlaColors.primaryDark,
                    ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            soru,
            style: const TextStyle(
              fontSize: 27,
              height: 1.35,
              fontWeight: FontWeight.w700,
              color: HatirlaColors.ink,
            ),
          ),
          const SizedBox(height: 20),
          BuyukButon(
            yazi: 'Bunu Anlat',
            ikon: Icons.mic_rounded,
            yukseklik: 80,
            onPressed: onAnlat,
          ),
          const SizedBox(height: 10),
          Row(
            children: <Widget>[
              Expanded(
                child: TextButton.icon(
                  onPressed: onBaskaSoru,
                  icon: const Icon(Icons.refresh_rounded, size: 28),
                  label: const Text('Başka soru'),
                ),
              ),
              Expanded(
                child: TextButton.icon(
                  onPressed: onTumSorular,
                  icon: const Icon(Icons.list_alt_rounded, size: 28),
                  label: const Text('Tüm sorular'),
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
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 36, horizontal: 20),
      alignment: Alignment.center,
      child: Column(
        children: <Widget>[
          const Icon(Icons.auto_stories_rounded,
              size: 88, color: HatirlaColors.line),
          const SizedBox(height: 20),
          Text(
            'Henüz hatıra yok',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 12),
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
        if (mm.durum == IndirmeDurumu.hazir) return const SizedBox.shrink();

        final bool iniyor = mm.durum == IndirmeDurumu.iniyor;
        return Container(
          width: double.infinity,
          margin: const EdgeInsets.fromLTRB(
              HatirlaSizes.gutter, 4, HatirlaSizes.gutter, 4),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFFFFF4DB),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: const Color(0xFFD9A400), width: 2),
          ),
          child: Row(
            children: <Widget>[
              Icon(
                iniyor ? Icons.cloud_download_rounded : Icons.info_outline_rounded,
                size: 30,
                color: const Color(0xFF8A6800),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  iniyor
                      ? 'Yazıya çevirme paketi iniyor… '
                          '%${((mm.ilerleme ?? 0) * 100).toStringAsFixed(0)}'
                      : 'Ses kayıtları yazıya çevrilmiyor. '
                          'Paket henüz indirilmedi.',
                  style: const TextStyle(fontSize: 20, height: 1.35),
                ),
              ),
              if (!iniyor) ...<Widget>[
                const SizedBox(width: 8),
                TextButton(
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const SettingsScreen(),
                    ),
                  ),
                  child: const Text('İndir'),
                ),
              ],
            ],
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
