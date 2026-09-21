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
      evetIkon: Icons.delete_forever_rounded,
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
          return Scaffold(
            appBar: AppBar(title: const Text('Hatıra')),
            body: const Center(
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
          appBar: AppBar(title: const Text('Hatıra')),
          body: SafeArea(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(
                  HatirlaSizes.gutter, 4, HatirlaSizes.gutter, 32),
              children: <Widget>[
                if (widget.yeniKaydedildi) const _KaydedildiSeridi(),
                _Tarih(memory: memory),
                const SizedBox(height: 20),
                _Oynatici(memory: memory),
                const SizedBox(height: 22),
                _YaziBolumu(
                  memory: memory,
                  onDuzelt: () => _yaziyiDuzelt(memory),
                ),
                const SizedBox(height: 28),
                CerceveliButon(
                  yazi: 'Bu Hatırayı Sil',
                  ikon: Icons.delete_outline_rounded,
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
    return Container(
      margin: const EdgeInsets.only(bottom: 18),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: const Color(0xFFE6F2E8),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: HatirlaColors.confirm, width: 2),
      ),
      child: Row(
        children: <Widget>[
          const Icon(Icons.check_circle_rounded,
              color: HatirlaColors.confirm, size: 34),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Hatıranız kaydedildi. Teşekkürler!',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    color: const Color(0xFF1E5B2E),
                  ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Hatiranin ne zaman kaydedildigi; baslik listede zaten var.
class _Tarih extends StatelessWidget {
  const _Tarih({required this.memory});

  final Memory memory;

  @override
  Widget build(BuildContext context) {
    return Text(
      '${Bicim.uzunTarih(memory.createdAt)} • ${Bicim.saat(memory.createdAt)}',
      style: const TextStyle(fontSize: 20, color: HatirlaColors.inkSoft),
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

        return Container(
          padding: const EdgeInsets.fromLTRB(18, 20, 18, 14),
          decoration: BoxDecoration(
            color: HatirlaColors.card,
            borderRadius: BorderRadius.circular(HatirlaSizes.radius),
            border: Border.all(color: HatirlaColors.line, width: 2),
          ),
          child: Column(
            children: <Widget>[
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  _YuvarlakDugme(
                    ikon: Icons.replay_10_rounded,
                    etiket: '10 saniye geri',
                    cap: 64,
                    onTap: bu ? () => p.geriSar() : null,
                  ),
                  const SizedBox(width: 20),
                  _YuvarlakDugme(
                    ikon: caliyor
                        ? Icons.pause_rounded
                        : Icons.play_arrow_rounded,
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
                  const SizedBox(width: 20),
                  _YuvarlakDugme(
                    ikon: Icons.forward_10_rounded,
                    etiket: '10 saniye ileri',
                    cap: 64,
                    onTap: bu ? () => p.ileriSar() : null,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              SliderTheme(
                data: SliderTheme.of(context).copyWith(
                  trackHeight: 12,
                  thumbShape:
                      const RoundSliderThumbShape(enabledThumbRadius: 16),
                  overlayShape:
                      const RoundSliderOverlayShape(overlayRadius: 30),
                  activeTrackColor: HatirlaColors.primary,
                  inactiveTrackColor: HatirlaColors.paperDark,
                  thumbColor: HatirlaColors.primary,
                ),
                child: Slider(
                  value: enFazla <= 0 ? 0 : deger,
                  max: enFazla <= 0 ? 1 : enFazla,
                  onChanged: bu && enFazla > 0
                      ? (double v) => p.sar(Duration(milliseconds: v.round()))
                      : null,
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 6),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: <Widget>[
                    Text(Bicim.sayac(konum),
                        style: const TextStyle(
                            fontSize: 20, color: HatirlaColors.inkSoft)),
                    Text(Bicim.sayac(uzunluk),
                        style: const TextStyle(
                            fontSize: 20, color: HatirlaColors.inkSoft)),
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
    return Semantics(
      button: true,
      label: etiket,
      child: SizedBox(
        width: cap,
        height: cap,
        child: Material(
          color: dolu
              ? HatirlaColors.primary
              : (aktif ? HatirlaColors.primarySoft : HatirlaColors.paperDark),
          shape: const CircleBorder(),
          child: InkWell(
            customBorder: const CircleBorder(),
            onTap: onTap,
            child: Icon(
              ikon,
              size: cap * 0.55,
              color: dolu
                  ? Colors.white
                  : (aktif ? HatirlaColors.primaryDark : HatirlaColors.line),
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
        const BolumBasligi('Anlattıklarınız', ikon: Icons.menu_book_rounded),
        const SizedBox(height: 14),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: HatirlaColors.card,
            borderRadius: BorderRadius.circular(HatirlaSizes.radius),
            border: Border.all(color: HatirlaColors.line, width: 2),
          ),
          child: _icerik(context),
        ),
        if (memory.status == TranscriptStatus.hazir) ...<Widget>[
          const SizedBox(height: 10),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: onDuzelt,
              icon: const Icon(Icons.edit_note_rounded, size: 28),
              label: const Text('Yazıyı düzelt'),
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
            ikon: Icons.volume_off_rounded,
            baslik: 'Konuşma duyulmadı',
            aciklama: 'Kayıtta anlaşılır bir konuşma bulunamadı. '
                'Ses kaydını yine de dinleyebilirsiniz.',
          );
        }
        return SelectableText(
          memory.transcript,
          style: const TextStyle(fontSize: 23, height: 1.65),
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
                  ikon: Icons.edit_note_rounded,
                  baslik: bu && yuzde > 0
                      ? 'Yazıya çevriliyor… %$yuzde'
                      : 'Yazıya çevriliyor…',
                  aciklama: 'Bu işlem telefonun içinde yapılıyor, birkaç '
                      'dakika sürebilir. Uygulamayı kapatmadan bekleyin.',
                ),
                const SizedBox(height: 16),
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: LinearProgressIndicator(
                    value: bu && yuzde > 0 ? yuzde / 100 : null,
                    minHeight: 16,
                    backgroundColor: HatirlaColors.paperDark,
                  ),
                ),
              ],
            );
          },
        );

      case TranscriptStatus.bekliyor:
        return const _BilgiSatiri(
          ikon: Icons.schedule_rounded,
          baslik: 'Sırada bekliyor',
          aciklama: 'Diğer kayıt bittiğinde bu hatıra da yazıya çevrilecek.',
        );

      case TranscriptStatus.hata:
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _BilgiSatiri(
              ikon: Icons.error_outline_rounded,
              baslik: 'Yazıya çevrilemedi',
              aciklama: memory.errorMessage ??
                  'Bir sorun oldu. Ses kaydınız güvende, tekrar deneyebilirsiniz.',
              renk: HatirlaColors.record,
            ),
            const SizedBox(height: 16),
            BuyukButon(
              yazi: 'Tekrar Dene',
              ikon: Icons.refresh_rounded,
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
        Icon(ikon, size: 32, color: c),
        const SizedBox(width: 14),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Text(
                baslik,
                style: TextStyle(
                    fontSize: 23, fontWeight: FontWeight.w700, color: c),
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
      appBar: AppBar(title: const Text('Yazıyı Düzelt')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(HatirlaSizes.gutter),
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
                ikon: Icons.check_rounded,
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
