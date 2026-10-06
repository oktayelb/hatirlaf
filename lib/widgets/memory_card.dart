import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../models/memory.dart';
import '../services/player.dart';
import '../services/store.dart';
import '../services/transcriber.dart';
import '../theme.dart';
import '../utils/format.dart';
import 'common.dart';

/// Listede bir hatirayi gosteren kart. Karta dokunmak hatirayi acar,
/// sagdaki dugme listeden cikmadan dinletir.
class MemoryCard extends StatelessWidget {
  const MemoryCard({super.key, required this.memory, required this.onTap});

  final Memory memory;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Kart(
      onTap: onTap,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              const _KapakSimgesi(boyut: 64),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(
                      memory.title,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        height: 1.25,
                        letterSpacing: -0.3,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      memory.durationMs > 0
                          ? '${Bicim.gunlukTarih(memory.createdAt)}  ·  '
                              '${Bicim.okunurSure(memory.duration)}'
                          : Bicim.gunlukTarih(memory.createdAt),
                      style: const TextStyle(
                        fontSize: 19,
                        height: 1.3,
                        color: HatirlaColors.inkSoft,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              _CalDugmesi(memory: memory),
            ],
          ),
          const SizedBox(height: 14),
          _DurumSatiri(memory: memory),
        ],
      ),
    );
  }
}

class _KapakSimgesi extends StatelessWidget {
  const _KapakSimgesi({required this.boyut});

  final double boyut;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: boyut,
      height: boyut,
      decoration: ShapeDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: <Color>[Color(0xFFEE8E5A), HatirlaColors.primary],
        ),
        shape: yumusakKose(boyut * 0.27),
      ),
      child: Icon(CupertinoIcons.waveform, size: boyut * 0.55, color: Colors.white),
    );
  }
}

/// Yuvarlak cal/duraklat dugmesi.
class _CalDugmesi extends StatelessWidget {
  const _CalDugmesi({required this.memory});

  final Memory memory;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Player.instance,
      builder: (BuildContext context, _) {
        final Player p = Player.instance;
        final bool caliyor = p.aktifMi(memory.id) && p.caliyor;
        Future<void> calDurdur() async {
          final String? hata = await Player.instance.calDurdur(
            memory.id,
            MemoryStore.instance.absolute(memory.audioRelPath),
          );
          if (hata != null && context.mounted) {
            kisaMesaj(context, hata);
          }
        }

        return Semantics(
          button: true,
          label: caliyor ? 'Duraklat' : 'Dinle',
          onTap: calDurdur,
          excludeSemantics: true,
          child: Basilabilir(
            onTap: calDurdur,
            olcek: 0.88,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              curve: Curves.easeOut,
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: caliyor ? HatirlaColors.primary : HatirlaColors.primarySoft,
              ),
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 220),
                transitionBuilder: (Widget child, Animation<double> a) =>
                    ScaleTransition(
                  scale: a,
                  child: FadeTransition(opacity: a, child: child),
                ),
                child: caliyor
                    ? const Icon(
                        CupertinoIcons.pause_fill,
                        key: ValueKey<bool>(true),
                        size: 32,
                        color: Colors.white,
                      )
                    : const Padding(
                        key: ValueKey<bool>(false),
                        padding: EdgeInsets.only(left: 4),
                        child: Icon(
                          CupertinoIcons.play_fill,
                          size: 32,
                          color: HatirlaColors.primary,
                        ),
                      ),
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Kartin altindaki durum/onizleme satiri.
class _DurumSatiri extends StatelessWidget {
  const _DurumSatiri({required this.memory});

  final Memory memory;

  @override
  Widget build(BuildContext context) {
    switch (memory.status) {
      case TranscriptStatus.hazir:
        if (!memory.hasTranscript) {
          return const _Etiket(
            ikon: CupertinoIcons.speaker_slash_fill,
            yazi: 'Konuşma duyulmadı',
            renk: HatirlaColors.inkSoft,
          );
        }
        return Text(
          Bicim.onizleme(memory.transcript),
          maxLines: 3,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            fontSize: 20,
            height: 1.45,
            color: HatirlaColors.inkSoft,
          ),
        );

      case TranscriptStatus.cevriliyor:
        return ListenableBuilder(
          listenable: Transcriber.instance,
          builder: (BuildContext context, _) {
            final bool bu = Transcriber.instance.aktifId == memory.id;
            final int yuzde = Transcriber.instance.yuzde;
            return _Etiket(
              ikon: CupertinoIcons.pencil,
              yazi: bu && yuzde > 0
                  ? 'Yazıya çevriliyor… %$yuzde'
                  : 'Yazıya çevriliyor…',
              renk: HatirlaColors.primaryDark,
              calisiyor: true,
            );
          },
        );

      case TranscriptStatus.bekliyor:
        return const _Etiket(
          ikon: CupertinoIcons.clock_fill,
          yazi: 'Yazıya çevrilmeyi bekliyor',
          renk: HatirlaColors.inkSoft,
        );

      case TranscriptStatus.hata:
        return _Etiket(
          ikon: CupertinoIcons.exclamationmark_circle_fill,
          yazi: memory.errorMessage ?? 'Yazıya çevrilemedi',
          renk: HatirlaColors.record,
        );
    }
  }
}

class _Etiket extends StatelessWidget {
  const _Etiket({
    required this.ikon,
    required this.yazi,
    required this.renk,
    this.calisiyor = false,
  });

  final IconData ikon;
  final String yazi;
  final Color renk;
  final bool calisiyor;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: ShapeDecoration(
        color: renk.withValues(alpha: 0.09),
        shape: yumusakKose(HatirlaSizes.radiusSmall),
      ),
      child: Row(
        children: <Widget>[
          SizedBox(
            width: 26,
            height: 26,
            child: calisiyor
                ? CupertinoActivityIndicator(radius: 11, color: renk)
                : Icon(ikon, size: 24, color: renk),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              yazi,
              style: TextStyle(
                fontSize: 19,
                height: 1.35,
                fontWeight: FontWeight.w600,
                color: renk,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
