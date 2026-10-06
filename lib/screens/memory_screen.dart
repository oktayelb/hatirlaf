import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../models/memory.dart';
import '../services/player.dart';
import '../services/store.dart';
import '../services/transcriber.dart';
import '../theme.dart';
import '../utils/format.dart';
import '../widgets/common.dart';

/// Tek bir hatiranin ekrani: dinle ve oku.
class MemoryScreen extends StatefulWidget {
  const MemoryScreen({
    super.key,
    required this.memoryId,
    this.yeniKaydedildi = false,
  });

  final String memoryId;

  /// Kayittan hemen sonra aciliyorsa ustte tebrik seridi gosterilir.
  final bool yeniKaydedildi;

  @override
  State<MemoryScreen> createState() => _MemoryScreenState();
}

class _MemoryScreenState extends State<MemoryScreen> {
  @override
  void dispose() {
    // Ekrandan cikinca ses devam etmesin.
    if (Player.instance.aktifMi(widget.memoryId)) {
      Player.instance.durdur();
    }
    super.dispose();
  }

  Memory? get _memory => MemoryStore.instance.byId(widget.memoryId);

  Future<void> _yaziyiDuzelt(Memory memory) async {
    final TextEditingController controller =
        TextEditingController(text: memory.transcript);
    final String? yeni = await Navigator.of(context).push<String>(
      MaterialPageRoute<String>(
        builder: (_) => _YaziDuzeltEkrani(controller: controller),
      ),
    );
    controller.dispose();
    if (yeni == null) return;
    await MemoryStore.instance.update(
      memory.copyWith(transcript: yeni, status: TranscriptStatus.hazir),
    );
  }

  Future<void> _sil(Memory memory) async {
    final bool emin = await onayIste(
      context,
      baslik: 'Bu hatırayı silelim mi?',
      mesaj: 'Ses kaydı ve yazısı kalıcı olarak silinecek. '
          'Bu işlem geri alınamaz.',
      evetYazi: 'Kalıcı Olarak Sil',
      hayirYazi: 'Vazgeç',
      evetIkon: CupertinoIcons.trash_fill,
      tehlikeli: true,
    );
    if (!emin || !mounted) return;
    await Player.instance.durdur();
    await MemoryStore.instance.delete(memory.id);
    if (!mounted) return;
    Navigator.of(context).popUntil((Route<dynamic> r) => r.isFirst);
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: MemoryStore.instance,
      builder: (BuildContext context, _) {
        final Memory? memory = _memory;
        if (memory == null) {
          // Hatira silinmisse (ornegin baska ekrandan) geri don.
          return const Scaffold(
            appBar: UstCubuk(baslik: 'Hatıra'),
            body: Center(
              child: Padding(
                padding: EdgeInsets.all(28),
                child: Text(
                  'Bu hatıra artık yok.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 24),
                ),
              ),
            ),
          );
        }

        return Scaffold(
          appBar: const UstCubuk(baslik: 'Hatıra'),
          body: SafeArea(
            top: false,
            child: ListView(
              padding: const EdgeInsets.fromLTRB(
                  HatirlaSizes.gutter, 4, HatirlaSizes.gutter, 36),
              children: <Widget>[
                if (widget.yeniKaydedildi) const _KaydedildiSeridi(),
                _Baslik(memory: memory),
                const SizedBox(height: 20),
                _Oynatici(memory: memory),
                const SizedBox(height: 32),
                _YaziBolumu(
                  memory: memory,
                  onDuzelt: () => _yaziyiDuzelt(memory),
                ),
                const SizedBox(height: 32),
                IkincilButon(
                  yazi: 'Bu Hatırayı Sil',
                  ikon: CupertinoIcons.trash,
                  renk: HatirlaColors.record,
                  onPressed: () => _sil(memory),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _KaydedildiSeridi extends StatelessWidget {
  const _KaydedildiSeridi();

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween<double>(begin: 0, end: 1),
      duration: const Duration(milliseconds: 500),
      curve: Curves.easeOutBack,
      builder: (BuildContext context, double t, Widget? child) => Opacity(
        opacity: t.clamp(0.0, 1.0),
        child: Transform.translate(
          offset: Offset(0, (1 - t) * -16),
          child: child,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.only(bottom: 20),
        child: Kart(
          renk: HatirlaColors.confirmSoft,
          golge: false,
          padding: const EdgeInsets.all(18),
          child: Row(
            children: <Widget>[
              const Icon(CupertinoIcons.checkmark_circle_fill,
                  color: HatirlaColors.confirm, size: 34),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Hatıranız kaydedildi. Teşekkürler!',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: const Color(0xFF1B6B2F),
                      ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Baslik extends StatelessWidget {
  const _Baslik({required this.memory});

  final Memory memory;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            memory.title,
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          const SizedBox(height: 6),
          Text(
            '${Bicim.uzunTarih(memory.createdAt)} · ${Bicim.saat(memory.createdAt)}',
            style: const TextStyle(fontSize: 19, color: HatirlaColors.inkSoft),
          ),
        ],
      ),
    );
  }
}

/// Buyuk oynatici: cal/duraklat, 10 saniye geri/ileri, surukleme cubugu.
class _Oynatici extends StatelessWidget {
  const _Oynatici({required this.memory});

  final Memory memory;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Player.instance,
      builder: (BuildContext context, _) {
        final Player p = Player.instance;
        final bool bu = p.aktifMi(memory.id);
        final bool caliyor = bu && p.caliyor;
        final Duration uzunluk = bu && p.uzunluk > Duration.zero
            ? p.uzunluk
            : memory.duration;
        final Duration konum = bu ? p.konum : Duration.zero;
        final double enFazla = uzunluk.inMilliseconds.toDouble();
        final double deger =
            konum.inMilliseconds.clamp(0, enFazla.toInt()).toDouble();
        const TextStyle zamanYazisi = TextStyle(
          fontSize: 19,
          fontWeight: FontWeight.w600,
          fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
          color: HatirlaColors.inkSoft,
        );

        return Kart(
          padding: const EdgeInsets.fromLTRB(16, 24, 16, 14),
          child: Column(
            children: <Widget>[
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  _YuvarlakDugme(
                    ikon: CupertinoIcons.gobackward_10,
                    etiket: '10 saniye geri',
                    cap: 66,
                    onTap: bu ? () => p.geriSar() : null,
                  ),
                  const SizedBox(width: 22),
                  _YuvarlakDugme(
                    ikon: caliyor
                        ? CupertinoIcons.pause_fill
                        : CupertinoIcons.play_fill,
                    etiket: caliyor ? 'Duraklat' : 'Dinle',
                    cap: 100,
                    dolu: true,
                    onTap: () async {
                      final String? hata = await p.calDurdur(
                        memory.id,
                        MemoryStore.instance.absolute(memory.audioRelPath),
                      );
                      if (hata != null && context.mounted) {
                        kisaMesaj(context, hata);
                      }
                    },
                  ),
                  const SizedBox(width: 22),
                  _YuvarlakDugme(
                    ikon: CupertinoIcons.goforward_10,
                    etiket: '10 saniye ileri',
                    cap: 66,
                    onTap: bu ? () => p.ileriSar() : null,
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Slider(
                value: enFazla <= 0 ? 0 : deger,
                max: enFazla <= 0 ? 1 : enFazla,
                onChanged: bu && enFazla > 0
                    ? (double v) => p.sar(Duration(milliseconds: v.round()))
                    : null,
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: <Widget>[
                    Text(Bicim.sayac(konum), style: zamanYazisi),
                    Text(Bicim.sayac(uzunluk), style: zamanYazisi),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _YuvarlakDugme extends StatelessWidget {
  const _YuvarlakDugme({
    required this.ikon,
    required this.etiket,
    required this.cap,
    required this.onTap,
    this.dolu = false,
  });

  final IconData ikon;
  final String etiket;
  final double cap;
  final VoidCallback? onTap;
  final bool dolu;

  @override
  Widget build(BuildContext context) {
    final bool aktif = onTap != null;
    final Color arka = dolu
        ? HatirlaColors.primary
        : (aktif ? HatirlaColors.primarySoft : HatirlaColors.paper);
    final Color on = dolu
        ? Colors.white
        : (aktif ? HatirlaColors.primary : HatirlaColors.chevron);
    final bool oynat = ikon == CupertinoIcons.play_fill;
    return Semantics(
      button: true,
      enabled: aktif,
      label: etiket,
      onTap: onTap,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onTap,
        olcek: 0.9,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          width: cap,
          height: cap,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: arka,
            boxShadow: dolu
                ? <BoxShadow>[
                    BoxShadow(
                      color: HatirlaColors.primary.withValues(alpha: 0.3),
                      blurRadius: 20,
                      offset: const Offset(0, 8),
                    ),
                  ]
                : null,
          ),
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            transitionBuilder: (Widget child, Animation<double> a) =>
                ScaleTransition(
              scale: a,
              child: FadeTransition(opacity: a, child: child),
            ),
            child: Padding(
              key: ValueKey<IconData>(ikon),
              padding: EdgeInsets.only(left: oynat ? cap * 0.06 : 0),
              child: Icon(ikon, size: cap * 0.46, color: on),
            ),
          ),
        ),
      ),
    );
  }
}

/// Yaziya cevrilmis metin bolumu.
class _YaziBolumu extends StatelessWidget {
  const _YaziBolumu({required this.memory, required this.onDuzelt});

  final Memory memory;
  final VoidCallback onDuzelt;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const BolumBasligi('Anlattıklarınız'),
        const SizedBox(height: 12),
        Kart(
          padding: const EdgeInsets.all(20),
          child: AnimatedSize(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOutCubic,
            alignment: Alignment.topCenter,
            child: _icerik(context),
          ),
        ),
        if (memory.status == TranscriptStatus.hazir) ...<Widget>[
          const SizedBox(height: 4),
          Align(
            alignment: Alignment.centerLeft,
            child: DuzButon(
              yazi: 'Yazıyı düzelt',
              ikon: CupertinoIcons.pencil,
              onPressed: onDuzelt,
            ),
          ),
        ],
      ],
    );
  }

  Widget _icerik(BuildContext context) {
    switch (memory.status) {
      case TranscriptStatus.hazir:
        if (!memory.hasTranscript) {
          return const _BilgiSatiri(
            ikon: CupertinoIcons.speaker_slash_fill,
            baslik: 'Konuşma duyulmadı',
            aciklama: 'Kayıtta anlaşılır bir konuşma bulunamadı. '
                'Ses kaydını yine de dinleyebilirsiniz.',
          );
        }
        return SizedBox(
          width: double.infinity,
          child: SelectableText(
            memory.transcript,
            style: const TextStyle(fontSize: 22, height: 1.65),
          ),
        );

      case TranscriptStatus.cevriliyor:
        return ListenableBuilder(
          listenable: Transcriber.instance,
          builder: (BuildContext context, _) {
            final bool bu = Transcriber.instance.aktifId == memory.id;
            final int yuzde = Transcriber.instance.yuzde;
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                _BilgiSatiri(
                  ikon: CupertinoIcons.pencil,
                  baslik: bu && yuzde > 0
                      ? 'Yazıya çevriliyor… %$yuzde'
                      : 'Yazıya çevriliyor…',
                  aciklama: 'Bu işlem telefonun içinde yapılıyor, birkaç '
                      'dakika sürebilir. Uygulamayı kapatmadan bekleyin.',
                ),
                const SizedBox(height: 16),
                IlerlemeCubugu(deger: bu && yuzde > 0 ? yuzde / 100 : null),
              ],
            );
          },
        );

      case TranscriptStatus.bekliyor:
        return const _BilgiSatiri(
          ikon: CupertinoIcons.clock_fill,
          baslik: 'Sırada bekliyor',
          aciklama: 'Diğer kayıt bittiğinde bu hatıra da yazıya çevrilecek.',
        );

      case TranscriptStatus.hata:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _BilgiSatiri(
              ikon: CupertinoIcons.exclamationmark_circle_fill,
              baslik: 'Yazıya çevrilemedi',
              aciklama: memory.errorMessage ??
                  'Bir sorun oldu. Ses kaydınız güvende, tekrar deneyebilirsiniz.',
              renk: HatirlaColors.record,
            ),
            const SizedBox(height: 18),
            BuyukButon(
              yazi: 'Tekrar Dene',
              ikon: CupertinoIcons.arrow_clockwise,
              yukseklik: 76,
              onPressed: () => Transcriber.instance.retry(memory.id),
            ),
          ],
        );
    }
  }
}

class _BilgiSatiri extends StatelessWidget {
  const _BilgiSatiri({
    required this.ikon,
    required this.baslik,
    required this.aciklama,
    this.renk,
  });

  final IconData ikon;
  final String baslik;
  final String aciklama;
  final Color? renk;

  @override
  Widget build(BuildContext context) {
    final Color c = renk ?? HatirlaColors.primaryDark;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: c.withValues(alpha: 0.1),
          ),
          child: Icon(ikon, size: 26, color: c),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                baslik,
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.2,
                  color: c,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                aciklama,
                style: const TextStyle(
                    fontSize: 20, height: 1.45, color: HatirlaColors.inkSoft),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// Yaziyi duzeltmek icin tam ekran metin alani: dialogda klavye acilinca
/// metin kayboluyor.
class _YaziDuzeltEkrani extends StatelessWidget {
  const _YaziDuzeltEkrani({required this.controller});

  final TextEditingController controller;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const UstCubuk(baslik: 'Yazıyı Düzelt'),
      body: SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
              HatirlaSizes.gutter, 4, HatirlaSizes.gutter, HatirlaSizes.gutter),
          child: Column(
            children: <Widget>[
              Expanded(
                child: TextField(
                  controller: controller,
                  maxLines: null,
                  expands: true,
                  autofocus: true,
                  textAlignVertical: TextAlignVertical.top,
                  textCapitalization: TextCapitalization.sentences,
                  style: const TextStyle(fontSize: 22, height: 1.6),
                  decoration: const InputDecoration(
                    hintText: 'Anlatılanları buraya yazabilirsiniz',
                  ),
                ),
              ),
              const SizedBox(height: 16),
              BuyukButon(
                yazi: 'Kaydet',
                ikon: CupertinoIcons.checkmark_alt,
                renk: HatirlaColors.confirm,
                onPressed: () =>
                    Navigator.of(context).pop(controller.text.trim()),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
