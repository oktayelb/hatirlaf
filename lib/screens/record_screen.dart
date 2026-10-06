import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:uuid/uuid.dart';

import '../data/akrabalar.dart';
import '../data/prompts.dart';
import '../models/memory.dart';
import '../services/kurtarma.dart';
import '../services/permissions.dart';
import '../services/player.dart';
import '../services/recorder.dart';
import '../services/store.dart';
import '../services/transcriber.dart';
import '../theme.dart';
import '../utils/format.dart';
import '../widgets/common.dart';
import 'memory_screen.dart';

/// Kayit ekrani. Ayni anda en fazla iki secenek: konusurken karar vermek
/// zorunda kalinmasin.
class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key, this.soru});

  /// Ekranda gosterilecek soru (varsa).
  final String? soru;

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  final Recorder _recorder = Recorder.instance;
  String? _id;
  String? _klasor;

  /// Baslik ve tarih kayit BASLARKEN belirleniyor: yarida kesilen bir
  /// kayit da ayni adla kurtarilsin diye isaret dosyasina yazilmalari
  /// gerekiyor.
  String _baslik = '';
  DateTime _baslangic = DateTime.now();

  bool _kaydediliyor = false;

  @override
  void initState() {
    super.initState();
    // Kayit yaparken baska bir hatira calmasin.
    unawaited(Player.instance.durdur());
  }

  @override
  void dispose() {
    // Ekran bir sekilde kapanirsa yarim kayit birakma.
    if (_recorder.durum != KayitDurumu.bos) {
      final String? klasor = _klasor;
      unawaited(
        _recorder.iptal().then((_) async {
          if (klasor != null) await Kurtarma.bitti(klasor);
        }),
      );
    }
    super.dispose();
  }

  Future<void> _basla() async {
    final bool izin = await _recorder.izinVarMi();
    if (!mounted) return;
    if (!izin) {
      await _izinUyarisi();
      return;
    }

    final String id = const Uuid().v4();
    final ({String id, String klasor}) hazirlik;
    try {
      hazirlik = await MemoryStore.instance.prepareNew(id);
    } catch (e) {
      if (!mounted) return;
      await bilgiGoster(
        context,
        baslik: 'Kayıt başlatılamadı',
        mesaj: 'Telefonda yer kalmamış olabilir. Biraz yer açıp tekrar '
            'deneyin.',
        ikon: CupertinoIcons.exclamationmark_triangle_fill,
        renk: HatirlaColors.record,
      );
      return;
    }

    final bool ok = await _recorder.basla(hazirlik.klasor);
    if (!mounted) return;
    if (!ok) {
      await bilgiGoster(
        context,
        baslik: 'Kayıt başlatılamadı',
        mesaj: 'Mikrofon başka bir uygulama tarafından kullanılıyor olabilir. '
            'Telefonda açık olan diğer uygulamaları kapatıp tekrar deneyin.',
        ikon: CupertinoIcons.mic_slash_fill,
        renk: HatirlaColors.record,
      );
      return;
    }

    final DateTime simdi = DateTime.now();
    final String baslik = widget.soru != null
        ? Sorular.soruyuBasligaCevir(widget.soru!)
        : '${Bicim.gunlukTarih(simdi)} hatırası';

    // Telefon uygulamayi oldururse hatirayi bu isaretten kurtaracagiz.
    await Kurtarma.basladi(
      klasor: hazirlik.klasor,
      id: id,
      baslik: baslik,
      sesDosyasi: (_recorder.dosyaYolu ?? '').split('/').last,
      soru: widget.soru,
    );

    if (!mounted) return;
    setState(() {
      _id = id;
      _klasor = hazirlik.klasor;
      _baslik = baslik;
      _baslangic = simdi;
    });
  }

  Future<void> _izinUyarisi() async {
    final bool kalici = await Izinler.kaliciReddedildiMi();
    if (!mounted) return;
    if (kalici) {
      final bool git = await onayIste(
        context,
        baslik: 'Mikrofon kapalı',
        mesaj: 'Ses kaydı yapabilmek için mikrofon izni gerekiyor.\n\n'
            'Ayarları açıp Mikrofon’u açalım mı?',
        evetYazi: 'Ayarları Aç',
        evetIkon: CupertinoIcons.gear_alt_fill,
      );
      if (git) await Izinler.ayarlariAc();
    } else {
      await bilgiGoster(
        context,
        baslik: 'Mikrofon gerekli',
        mesaj: 'Kayıt yapabilmek için mikrofon iznine “İzin Ver” demeniz gerek.',
        ikon: CupertinoIcons.mic_slash_fill,
        renk: HatirlaColors.record,
      );
    }
  }

  Future<void> _bitir() async {
    if (_kaydediliyor) return;
    setState(() => _kaydediliyor = true);

    final String? klasor = _klasor;
    final ({Duration sure, String yol})? sonuc = await _recorder.bitir();

    // Ses dosyasi artik tamam: dizine yazilamadan kapanilsa bile kurtarilir.
    if (sonuc != null && klasor != null) {
      await Kurtarma.tamamlandi(klasor: klasor, sure: sonuc.sure);
    }
    if (!mounted) return;

    if (sonuc == null || _id == null) {
      if (klasor != null) await Kurtarma.bitti(klasor);
      if (!mounted) return;
      setState(() {
        _kaydediliyor = false;
        _id = null;
        _klasor = null;
      });
      await bilgiGoster(
        context,
        baslik: 'Kayıt alınamadı',
        mesaj: 'Ses kaydedilemedi. Lütfen tekrar deneyin.',
        ikon: CupertinoIcons.exclamationmark_circle_fill,
        renk: HatirlaColors.record,
      );
      return;
    }

    final String id = _id!;
    final Memory memory = Memory(
      id: id,
      title: _baslik,
      createdAt: _baslangic,
      audioRelPath:
          '${MemoryStore.memoriesDirName}/$id/${sonuc.yol.split('/').last}',
      durationMs: sonuc.sure.inMilliseconds,
      question: widget.soru,
    );

    await MemoryStore.instance.add(memory);
    // Hatira dizine girdi; isaret artik gereksiz.
    if (klasor != null) await Kurtarma.bitti(klasor);
    Transcriber.instance.enqueue(id);

    if (!mounted) return;
    // Detay ekraniyla degistiriyoruz ki geri tusu listeye gotursun.
    Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(
        builder: (_) => MemoryScreen(memoryId: id, yeniKaydedildi: true),
      ),
    );
  }

  Future<void> _vazgec() async {
    final bool emin = await onayIste(
      context,
      baslik: 'Kayıttan vazgeçiyor musunuz?',
      mesaj: 'Şu ana kadar anlattıklarınız silinecek ve geri getirilemeyecek.',
      evetYazi: 'Sil, Vazgeç',
      hayirYazi: 'Kayda Devam Et',
      evetIkon: CupertinoIcons.trash,
      tehlikeli: true,
    );
    if (!emin || !mounted) return;
    final String? klasor = _klasor;
    await _recorder.iptal();
    if (klasor != null) await Kurtarma.bitti(klasor);
    if (!mounted) return;
    setState(() {
      _id = null;
      _klasor = null;
    });
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _recorder,
      builder: (BuildContext context, _) {
        final KayitDurumu durum = _recorder.durum;
        final bool kayitVar = durum != KayitDurumu.bos;

        return PopScope(
          // Kayit surerken geri tusu kazara basildiginda kaydi ucurmayalim.
          canPop: !kayitVar,
          onPopInvokedWithResult: (bool didPop, Object? _) {
            if (!didPop) _vazgec();
          },
          child: Scaffold(
            appBar: UstCubuk(
              baslik: kayitVar ? 'Kaydediliyor' : 'Yeni Hatıra',
              onGeri: () {
                if (kayitVar) {
                  _vazgec();
                } else {
                  Navigator.of(context).pop();
                }
              },
            ),
            body: SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(
                  HatirlaSizes.gutter,
                  0,
                  HatirlaSizes.gutter,
                  16,
                ),
                child: Column(
                  children: <Widget>[
                    if (widget.soru != null) _SoruSeridi(soru: widget.soru!),
                    Expanded(
                      // Kayit baslamadan once fotograf da giriyor; yer
                      // yetmezse fotograf gizlenir, dugme kuculur.
                      child: LayoutBuilder(
                        builder: (BuildContext context, BoxConstraints alan) {
                          const double fotografIcinGereken = 560;
                          const double dugmeIcinGereken = 330;
                          final bool fotografSigar =
                              !kayitVar &&
                              alan.maxHeight >= fotografIcinGereken;
                          final bool dugmeSigar =
                              !kayitVar ||
                              alan.maxHeight >= dugmeIcinGereken;
                          return Column(
                            children: <Widget>[
                              AnimatedSize(
                                duration: const Duration(milliseconds: 380),
                                curve: Curves.easeInOutCubic,
                                child: AnimatedSwitcher(
                                  duration: const Duration(milliseconds: 250),
                                  child: fotografSigar
                                      ? const Padding(
                                          key: ValueKey<bool>(true),
                                          padding: EdgeInsets.only(
                                            top: 12,
                                            bottom: 8,
                                          ),
                                          child: _BirlikteFotografi(),
                                        )
                                      : const SizedBox(
                                          key: ValueKey<bool>(false),
                                          width: double.infinity,
                                        ),
                                ),
                              ),
                              Expanded(
                                child: Center(
                                  child: FittedBox(
                                    fit: BoxFit.scaleDown,
                                    child: SizedBox(
                                      width: alan.maxWidth,
                                      child: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        children: <Widget>[
                                          if (dugmeSigar) ...<Widget>[
                                            _KayitDugmesi(
                                              durum: durum,
                                              seviye: _recorder.seviye,
                                              onBasla: _basla,
                                            ),
                                            const SizedBox(height: 14),
                                          ],
                                          AnimatedSwitcher(
                                            duration: const Duration(
                                              milliseconds: 300,
                                            ),
                                            child: _Sayac(
                                              key: ValueKey<bool>(kayitVar),
                                              durum: durum,
                                              sure: _recorder.sure,
                                              // Soru varken yonerge yazisi
                                              // ekrandan dugmeyi tasiriyor;
                                              // soru zaten ne yapilacagini
                                              // soyluyor.
                                              yonergeGoster:
                                                  widget.soru == null,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                    ),
                    AnimatedSize(
                      duration: const Duration(milliseconds: 380),
                      curve: Curves.easeInOutCubic,
                      child: _AltEylemler(
                        durum: durum,
                        kaydediliyor: _kaydediliyor,
                        onDuraklat: _recorder.duraklat,
                        onDevam: _recorder.devamEt,
                        onBitir: _bitir,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Kayit sirasinda soruyu ekranda tutar.
class _SoruSeridi extends StatelessWidget {
  const _SoruSeridi({required this.soru});

  final String soru;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Kart(
        renk: HatirlaColors.primarySoft,
        golge: false,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
        child: Text(
          soru,
          textAlign: TextAlign.center,
          style: const TextStyle(
            fontSize: 29,
            height: 1.3,
            fontWeight: FontWeight.w700,
            letterSpacing: -0.5,
            color: HatirlaColors.primaryDark,
          ),
        ),
      ),
    );
  }
}

/// Kayda baslamadan once gorunen aile fotografi.
///
/// Mikrofona konusmak yabanci bir is; karsida bir yuz varken daha kolay.
/// Kayit basladigi anda kalkiyor: o andan sonra ekranda yalnizca kirmizi
/// dugme ve sayac kalmali.
class _BirlikteFotografi extends StatelessWidget {
  const _BirlikteFotografi();

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        CerceveliFoto(
          yukseklik: 140,
          foto: Image.asset(
            kBirlikteFotografi,
            fit: BoxFit.cover,
            // Gorsel bir sekilde acilmazsa ekranda cerceveli bir
            // bosluk kalmasin.
            errorBuilder: (BuildContext context, Object e, StackTrace? s) =>
                const SizedBox.shrink(),
          ),
        ),
        const SizedBox(height: 14),
        const Text(
          'Anlattıklarınızı bir gün torunlarınız dinleyecek.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 23,
            height: 1.3,
            fontWeight: FontWeight.w600,
            letterSpacing: -0.2,
            color: HatirlaColors.inkSoft,
          ),
        ),
      ],
    );
  }
}

/// Dev kayit dugmesi ve ses seviyesiyle nefes alan halka.
class _KayitDugmesi extends StatefulWidget {
  const _KayitDugmesi({
    required this.durum,
    required this.seviye,
    required this.onBasla,
  });

  final KayitDurumu durum;
  final double seviye;
  final VoidCallback onBasla;

  @override
  State<_KayitDugmesi> createState() => _KayitDugmesiState();
}

class _KayitDugmesiState extends State<_KayitDugmesi>
    with SingleTickerProviderStateMixin {
  late final AnimationController _nefes = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1500),
  );

  @override
  void initState() {
    super.initState();
    _nefesiAyarla();
  }

  @override
  void didUpdateWidget(_KayitDugmesi eski) {
    super.didUpdateWidget(eski);
    if (eski.durum != widget.durum) _nefesiAyarla();
  }

  void _nefesiAyarla() {
    if (widget.durum == KayitDurumu.bos) {
      _nefes.repeat(reverse: true);
    } else {
      _nefes.animateTo(0, duration: const Duration(milliseconds: 200));
    }
  }

  @override
  void dispose() {
    _nefes.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const double cap = 172;
    final bool bos = widget.durum == KayitDurumu.bos;
    final bool kaydediyor = widget.durum == KayitDurumu.kaydediyor;
    final bool duraklatildi = widget.durum == KayitDurumu.duraklatildi;

    return Semantics(
      button: bos,
      label: bos ? 'Kayda başla' : null,
      onTap: bos ? widget.onBasla : null,
      excludeSemantics: true,
      child: SizedBox(
        width: cap + 76,
        height: cap + 76,
        child: Stack(
          alignment: Alignment.center,
          children: <Widget>[
            // "Seni duyuyorum" demenin en anlasilir yolu.
            AnimatedContainer(
              duration: const Duration(milliseconds: 140),
              width: kaydediyor ? cap + 26 + widget.seviye * 50 : cap + 26,
              height: kaydediyor ? cap + 26 + widget.seviye * 50 : cap + 26,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: HatirlaColors.record.withValues(
                  alpha: kaydediyor ? 0.16 : 0,
                ),
              ),
            ),
            Container(
              width: cap + 24,
              height: cap + 24,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: HatirlaColors.card,
                border: Border.all(color: const Color(0xFFD1D1D6), width: 3),
                boxShadow: kKartGolgesi,
              ),
            ),
            ScaleTransition(
              scale: Tween<double>(begin: 1, end: 1.04).animate(
                CurvedAnimation(parent: _nefes, curve: Curves.easeInOut),
              ),
              child: Basilabilir(
                onTap: bos ? widget.onBasla : null,
                olcek: 0.93,
                titresim: HapticFeedback.heavyImpact,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeOut,
                  width: cap,
                  height: cap,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: duraklatildi
                        ? HatirlaColors.inkSoft
                        : HatirlaColors.record,
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      AnimatedSwitcher(
                        duration: const Duration(milliseconds: 220),
                        transitionBuilder:
                            (Widget child, Animation<double> a) =>
                                ScaleTransition(scale: a, child: child),
                        child: Icon(
                          duraklatildi
                              ? CupertinoIcons.pause_fill
                              : CupertinoIcons.mic_fill,
                          key: ValueKey<bool>(duraklatildi),
                          size: 76,
                          color: Colors.white,
                        ),
                      ),
                      if (bos)
                        const Padding(
                          padding: EdgeInsets.fromLTRB(18, 6, 18, 0),
                          child: FittedBox(
                            fit: BoxFit.scaleDown,
                            child: Text(
                              'DOKUNUN',
                              style: TextStyle(
                                fontSize: 22,
                                letterSpacing: 2,
                                fontWeight: FontWeight.w700,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Sayac extends StatelessWidget {
  const _Sayac({
    super.key,
    required this.durum,
    required this.sure,
    this.yonergeGoster = true,
  });

  final KayitDurumu durum;
  final Duration sure;

  /// Kayit baslamadan once gosterilen yonerge yazisi.
  final bool yonergeGoster;

  @override
  Widget build(BuildContext context) {
    if (durum == KayitDurumu.bos) {
      if (!yonergeGoster) return const SizedBox(width: double.infinity);
      return const Padding(
        padding: EdgeInsets.symmetric(horizontal: 12),
        child: Text(
          'Kırmızı düğmeye dokunun ve anlatın.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 25,
            height: 1.3,
            fontWeight: FontWeight.w600,
            letterSpacing: -0.3,
            color: HatirlaColors.inkSoft,
          ),
        ),
      );
    }
    final bool duraklatildi = durum == KayitDurumu.duraklatildi;
    return Column(
      children: <Widget>[
        Text(
          Bicim.sayac(sure),
          style: const TextStyle(
            fontSize: 66,
            fontWeight: FontWeight.w700,
            letterSpacing: -1.5,
            fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
            color: HatirlaColors.ink,
          ),
        ),
        const SizedBox(height: 2),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (!duraklatildi) ...<Widget>[
              const _YanipSonenNokta(),
              const SizedBox(width: 10),
            ],
            Flexible(
              child: Text(
                duraklatildi ? 'Duraklatıldı' : 'Sizi dinliyorum…',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 26,
                  fontWeight: FontWeight.w600,
                  letterSpacing: -0.3,
                  color: duraklatildi
                      ? HatirlaColors.inkSoft
                      : HatirlaColors.record,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _YanipSonenNokta extends StatefulWidget {
  const _YanipSonenNokta();

  @override
  State<_YanipSonenNokta> createState() => _YanipSonenNoktaState();
}

class _YanipSonenNoktaState extends State<_YanipSonenNokta>
    with SingleTickerProviderStateMixin {
  late final AnimationController _yanip = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _yanip.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: Tween<double>(
        begin: 1,
        end: 0.25,
      ).animate(CurvedAnimation(parent: _yanip, curve: Curves.easeInOut)),
      child: Container(
        width: 16,
        height: 16,
        decoration: const BoxDecoration(
          shape: BoxShape.circle,
          color: HatirlaColors.record,
        ),
      ),
    );
  }
}

class _AltEylemler extends StatelessWidget {
  const _AltEylemler({
    required this.durum,
    required this.kaydediliyor,
    required this.onDuraklat,
    required this.onDevam,
    required this.onBitir,
  });

  final KayitDurumu durum;
  final bool kaydediliyor;
  final Future<void> Function() onDuraklat;
  final Future<void> Function() onDevam;
  final Future<void> Function() onBitir;

  @override
  Widget build(BuildContext context) {
    if (durum == KayitDurumu.bos) {
      return const SizedBox(width: double.infinity);
    }

    final bool duraklatildi = durum == KayitDurumu.duraklatildi;
    return Column(
      children: <Widget>[
        BuyukButon(
          yazi: kaydediliyor ? 'Kaydediliyor…' : 'Bitir ve Kaydet',
          ikon: CupertinoIcons.checkmark_alt,
          renk: HatirlaColors.confirm,
          yukseklik: 90,
          onPressed: kaydediliyor ? null : () => onBitir(),
        ),
        const SizedBox(height: 12),
        IkincilButon(
          yazi: duraklatildi ? 'Devam Et' : 'Ara Ver',
          ikon: duraklatildi
              ? CupertinoIcons.play_fill
              : CupertinoIcons.pause_fill,
          renk: HatirlaColors.inkSoft,
          onPressed: kaydediliyor
              ? null
              : () => duraklatildi ? onDevam() : onDuraklat(),
        ),
      ],
    );
  }
}
