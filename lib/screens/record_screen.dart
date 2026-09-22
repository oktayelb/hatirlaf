import 'dart:async';

import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import '../data/akrabalar.dart';
import '../data/prompts.dart';
import '../models/memory.dart';
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
      unawaited(_recorder.iptal());
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
    final ({String audioPath, String audioRelPath, String id}) hazirlik;
    try {
      hazirlik = await MemoryStore.instance.prepareNew(id);
    } catch (e) {
      if (!mounted) return;
      await bilgiGoster(
        context,
        baslik: 'Kayıt başlatılamadı',
        mesaj: 'Telefonda yer kalmamış olabilir. Biraz yer açıp tekrar '
            'deneyin.',
        ikon: Icons.sd_card_alert_rounded,
        renk: HatirlaColors.record,
      );
      return;
    }

    final bool ok = await _recorder.basla(hazirlik.audioPath);
    if (!mounted) return;
    if (!ok) {
      await bilgiGoster(
        context,
        baslik: 'Kayıt başlatılamadı',
        mesaj: 'Mikrofon başka bir uygulama tarafından kullanılıyor olabilir. '
            'Telefonda açık olan diğer uygulamaları kapatıp tekrar deneyin.',
        ikon: Icons.mic_off_rounded,
        renk: HatirlaColors.record,
      );
      return;
    }
    setState(() => _id = id);
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
        evetIkon: Icons.settings_rounded,
      );
      if (git) await Izinler.ayarlariAc();
    } else {
      await bilgiGoster(
        context,
        baslik: 'Mikrofon gerekli',
        mesaj: 'Kayıt yapabilmek için mikrofon iznine “İzin Ver” demeniz gerek.',
        ikon: Icons.mic_off_rounded,
        renk: HatirlaColors.record,
      );
    }
  }

  Future<void> _bitir() async {
    if (_kaydediliyor) return;
    setState(() => _kaydediliyor = true);

    final ({Duration sure, String yol})? sonuc = await _recorder.bitir();
    if (!mounted) return;

    if (sonuc == null || _id == null) {
      setState(() {
        _kaydediliyor = false;
        _id = null;
      });
      await bilgiGoster(
        context,
        baslik: 'Kayıt alınamadı',
        mesaj: 'Ses kaydedilemedi. Lütfen tekrar deneyin.',
        ikon: Icons.error_outline_rounded,
        renk: HatirlaColors.record,
      );
      return;
    }

    final String id = _id!;
    final DateTime simdi = DateTime.now();
    final Memory memory = Memory(
      id: id,
      title: widget.soru != null
          ? Sorular.soruyuBasligaCevir(widget.soru!)
          : '${Bicim.gunlukTarih(simdi)} hatırası',
      createdAt: simdi,
      audioRelPath:
          '${MemoryStore.memoriesDirName}/$id/${sonuc.yol.split('/').last}',
      durationMs: sonuc.sure.inMilliseconds,
      question: widget.soru,
    );

    await MemoryStore.instance.add(memory);
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
      evetIkon: Icons.delete_outline_rounded,
      tehlikeli: true,
    );
    if (!emin || !mounted) return;
    await _recorder.iptal();
    if (!mounted) return;
    setState(() => _id = null);
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
            appBar: AppBar(
              title: Text(kayitVar ? 'Kaydediliyor' : 'Yeni Hatıra'),
              leading: IconButton(
                iconSize: 36,
                tooltip: 'Geri',
                icon: const Icon(Icons.arrow_back_rounded),
                onPressed: () {
                  if (kayitVar) {
                    _vazgec();
                  } else {
                    Navigator.of(context).pop();
                  }
                },
              ),
            ),
            body: SafeArea(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(
                    HatirlaSizes.gutter, 0, HatirlaSizes.gutter, 20),
                child: Column(
                  children: <Widget>[
                    if (widget.soru != null) _SoruSeridi(soru: widget.soru!),
                    Expanded(
                      // Kayit baslamadan once fotograf da giriyor;
                      // kucuk ekranda tasmasin diye kaydirilabilir.
                      child: Center(
                        child: SingleChildScrollView(
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: <Widget>[
                              if (!kayitVar) ...<Widget>[
                                const _BirlikteFotografi(),
                                const SizedBox(height: 20),
                              ],
                              _KayitDugmesi(
                                durum: durum,
                                seviye: _recorder.seviye,
                                onBasla: _basla,
                              ),
                              const SizedBox(height: 18),
                              _Sayac(
                                durum: durum,
                                sure: _recorder.sure,
                                // Soru varken yonerge yazisi ekrandan
                                // dugmeyi tasiriyor; soru zaten ne
                                // yapilacagini soyluyor.
                                yonergeGoster: widget.soru == null,
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    _AltEylemler(
                      durum: durum,
                      kaydediliyor: _kaydediliyor,
                      onDuraklat: _recorder.duraklat,
                      onDevam: _recorder.devamEt,
                      onBitir: _bitir,
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
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 18),
      decoration: BoxDecoration(
        color: HatirlaColors.primarySoft,
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
      ),
      child: Text(
        soru,
        textAlign: TextAlign.center,
        style: const TextStyle(
          fontSize: 31,
          height: 1.3,
          fontWeight: FontWeight.w700,
          color: HatirlaColors.primaryDark,
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
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: HatirlaColors.card,
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        border: Border.all(color: HatirlaColors.line, width: 2),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          ClipRRect(
            borderRadius: BorderRadius.circular(HatirlaSizes.radius - 8),
            child: Image.asset(
              kBirlikteFotografi,
              width: double.infinity,
              height: 140,
              fit: BoxFit.cover,
              // Gorsel bir sekilde acilmazsa ekranda cerceveli bir
              // bosluk kalmasin.
              errorBuilder: (BuildContext context, Object e, StackTrace? s) =>
                  const SizedBox.shrink(),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Anlattıklarınızı bir gün torunlarınız dinleyecek.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 26,
              height: 1.3,
              color: HatirlaColors.inkSoft,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

/// Dev kayit dugmesi ve ses seviyesiyle nefes alan halka.
class _KayitDugmesi extends StatelessWidget {
  const _KayitDugmesi({
    required this.durum,
    required this.seviye,
    required this.onBasla,
  });

  final KayitDurumu durum;
  final double seviye;
  final VoidCallback onBasla;

  @override
  Widget build(BuildContext context) {
    const double cap = 180;
    final bool kaydediyor = durum == KayitDurumu.kaydediyor;
    final bool duraklatildi = durum == KayitDurumu.duraklatildi;

    return SizedBox(
      width: cap + 70,
      height: cap + 70,
      child: Stack(
        alignment: Alignment.center,
        children: <Widget>[
          // "Seni duyuyorum" demenin en anlasilir yolu.
          if (kaydediyor)
            AnimatedContainer(
              duration: const Duration(milliseconds: 140),
              width: cap + 20 + seviye * 44,
              height: cap + 20 + seviye * 44,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: HatirlaColors.record.withValues(alpha: 0.16),
              ),
            ),
          SizedBox(
            width: cap,
            height: cap,
            child: Material(
              color: duraklatildi
                  ? HatirlaColors.inkSoft
                  : HatirlaColors.record,
              shape: const CircleBorder(),
              child: InkWell(
                customBorder: const CircleBorder(),
                onTap: durum == KayitDurumu.bos ? onBasla : null,
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Icon(
                        duraklatildi
                            ? Icons.pause_rounded
                            : Icons.mic_rounded,
                        size: 92,
                        color: Colors.white,
                      ),
                      if (durum == KayitDurumu.bos)
                        const Padding(
                          padding: EdgeInsets.only(top: 4),
                          child: Text(
                            'DOKUNUN',
                            style: TextStyle(
                              fontSize: 24,
                              letterSpacing: 2,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Sayac extends StatelessWidget {
  const _Sayac({
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
      if (!yonergeGoster) return const SizedBox.shrink();
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12),
        child: const Text(
          'Kırmızı düğmeye dokunun ve anlatın.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 27,
            height: 1.3,
            fontWeight: FontWeight.w600,
            color: HatirlaColors.inkSoft,
          ),
        ),
      );
    }
    return Column(
      children: <Widget>[
        Text(
          Bicim.sayac(sure),
          style: const TextStyle(
            fontSize: 64,
            fontWeight: FontWeight.w700,
            fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
            color: HatirlaColors.ink,
          ),
        ),
        const SizedBox(height: 6),
        Text(
          durum == KayitDurumu.duraklatildi
              ? 'Duraklatıldı'
              : 'Sizi dinliyorum…',
          style: TextStyle(
            fontSize: 28,
            color: durum == KayitDurumu.duraklatildi
                ? HatirlaColors.inkSoft
                : HatirlaColors.record,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
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
    if (durum == KayitDurumu.bos) return const SizedBox.shrink();

    return Column(
      children: <Widget>[
        BuyukButon(
          yazi: kaydediliyor ? 'Kaydediliyor…' : 'Bitir ve Kaydet',
          ikon: Icons.check_rounded,
          renk: HatirlaColors.confirm,
          yukseklik: 92,
          onPressed: kaydediliyor ? null : () => onBitir(),
        ),
        const SizedBox(height: 12),
        CerceveliButon(
          yazi: durum == KayitDurumu.duraklatildi ? 'Devam Et' : 'Ara Ver',
          ikon: durum == KayitDurumu.duraklatildi
              ? Icons.play_arrow_rounded
              : Icons.pause_rounded,
          onPressed: kaydediliyor
              ? null
              : () => durum == KayitDurumu.duraklatildi
                  ? onDevam()
                  : onDuraklat(),
        ),
      ],
    );
  }
}
