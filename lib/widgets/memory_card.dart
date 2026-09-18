
import 'package:flutter/material.dart';

import '../models/memory.dart';
import '../services/player.dart';
import '../services/store.dart';
import '../services/transcriber.dart';
import '../theme.dart';
import '../utils/format.dart';
import 'common.dart';

/// Listede bir hatirayi gosteren kart.
///
/// Karta dokunmak hatirayi acar; sagdaki yuvarlak dugme ise listeden
/// cikmadan dinletir. "Once acayim sonra oynatayim" adimini kaldirmak
/// yaslilarda en cok fark yaratan seylerden biri.
class MemoryCard extends StatelessWidget {
  const MemoryCard({super.key, required this.memory, required this.onTap});

  final Memory memory;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: HatirlaColors.card,
      borderRadius: BorderRadius.circular(HatirlaSizes.radius),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(HatirlaSizes.radius),
            border: Border.all(color: HatirlaColors.line, width: 2),
          ),
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  const _KapakSimgesi(boyut: 88),
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
                            fontSize: 24,
                            fontWeight: FontWeight.w700,
                            height: 1.25,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          memory.durationMs > 0
                              ? '${Bicim.gunlukTarih(memory.createdAt)}  •  '
                                  '${Bicim.okunurSure(memory.duration)}'
                              : Bicim.gunlukTarih(memory.createdAt),
                          style: const TextStyle(
                            fontSize: 19,
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
              const SizedBox(height: 12),
              _DurumSatiri(memory: memory),
            ],
          ),
        ),
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
      decoration: BoxDecoration(
        color: HatirlaColors.primarySoft,
        borderRadius: BorderRadius.circular(16),
      ),
      child: const Icon(Icons.graphic_eq_rounded,
          size: 44, color: HatirlaColors.primary),
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
        final bool bu = p.aktifMi(memory.id);
        final bool caliyor = bu && p.caliyor;
        return Semantics(
          button: true,
          label: caliyor ? 'Duraklat' : 'Dinle',
          child: SizedBox(
            width: 76,
            height: 76,
            child: Material(
              color: caliyor ? HatirlaColors.primary : HatirlaColors.primarySoft,
              shape: const CircleBorder(),
              child: InkWell(
                customBorder: const CircleBorder(),
                onTap: () async {
                  final String? hata = await Player.instance.calDurdur(
                    memory.id,
                    MemoryStore.instance.absolute(memory.audioRelPath),
                  );
                  if (hata != null && context.mounted) {
                    kisaMesaj(context, hata);
                  }
                },
                child: Icon(
                  caliyor ? Icons.pause_rounded : Icons.play_arrow_rounded,
                  size: 46,
                  color: caliyor ? Colors.white : HatirlaColors.primaryDark,
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
            ikon: Icons.volume_off_rounded,
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
              ikon: Icons.edit_note_rounded,
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
          ikon: Icons.schedule_rounded,
          yazi: 'Yazıya çevrilmeyi bekliyor',
          renk: HatirlaColors.inkSoft,
        );

      case TranscriptStatus.hata:
        return _Etiket(
          ikon: Icons.error_outline_rounded,
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
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        if (calisiyor)
          SizedBox(
            width: 26,
            height: 26,
            child: CircularProgressIndicator(strokeWidth: 3.5, color: renk),
          )
        else
          Icon(ikon, size: 26, color: renk),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            yazi,
            style: TextStyle(fontSize: 20, height: 1.35, color: renk),
          ),
        ),
      ],
    );
  }
}
