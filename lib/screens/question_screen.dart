import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../data/prompts.dart';
import '../theme.dart';
import '../widgets/common.dart';

/// Soru listesi; secilen soru geriye dondurulur. Konular kapali baslar:
/// 70 soru yerine once 10 baslik.
class QuestionScreen extends StatefulWidget {
  const QuestionScreen({super.key});

  @override
  State<QuestionScreen> createState() => _QuestionScreenState();
}

class _QuestionScreenState extends State<QuestionScreen> {
  int? _acikKonu;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const UstCubuk(baslik: 'Ne Anlatalım?'),
      body: SafeArea(
        top: false,
        child: ListView.separated(
          padding: const EdgeInsets.fromLTRB(
              HatirlaSizes.gutter, 4, HatirlaSizes.gutter, HatirlaSizes.gutter),
          itemCount: Sorular.konular.length,
          separatorBuilder: (_, _) => const SizedBox(height: 14),
          itemBuilder: (BuildContext context, int i) {
            final bool acik = _acikKonu == i;
            return _KonuKarti(
              konu: Sorular.konular[i],
              acik: acik,
              onBaslik: () => setState(() => _acikKonu = acik ? null : i),
              onSoru: (String soru) => Navigator.of(context).pop(soru),
            );
          },
        ),
      ),
    );
  }
}

class _KonuKarti extends StatelessWidget {
  const _KonuKarti({
    required this.konu,
    required this.acik,
    required this.onBaslik,
    required this.onSoru,
  });

  final SoruKonusu konu;
  final bool acik;
  final VoidCallback onBaslik;
  final ValueChanged<String> onSoru;

  @override
  Widget build(BuildContext context) {
    return Kart(
      padding: EdgeInsets.zero,
      kirp: true,
      child: Column(
        children: <Widget>[
          Semantics(
            button: true,
            expanded: acik,
            label: konu.ad,
            onTap: onBaslik,
            excludeSemantics: true,
            child: Basilabilir(
              onTap: onBaslik,
              olcek: 0.985,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: <Widget>[
                    IkonRozeti(ikon: konu.ikon, renk: konu.renk, boyut: 52),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Text(
                        konu.ad,
                        style: const TextStyle(
                          fontSize: 23,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.3,
                        ),
                      ),
                    ),
                    AnimatedRotation(
                      turns: acik ? 0.25 : 0,
                      duration: const Duration(milliseconds: 280),
                      curve: Curves.easeInOutCubic,
                      child: const Icon(
                        CupertinoIcons.chevron_forward,
                        size: 28,
                        color: HatirlaColors.chevron,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          AnimatedSize(
            duration: const Duration(milliseconds: 320),
            curve: Curves.easeInOutCubic,
            alignment: Alignment.topCenter,
            child: acik
                ? Column(
                    children: <Widget>[
                      for (final String soru in konu.sorular) ...<Widget>[
                        const Divider(indent: 18),
                        _SoruSatiri(
                          soru: soru,
                          renk: konu.renk,
                          onTap: () => onSoru(soru),
                        ),
                      ],
                    ],
                  )
                : const SizedBox(width: double.infinity),
          ),
        ],
      ),
    );
  }
}

class _SoruSatiri extends StatelessWidget {
  const _SoruSatiri({
    required this.soru,
    required this.renk,
    required this.onTap,
  });

  final String soru;
  final Color renk;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: soru,
      onTap: onTap,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onTap,
        olcek: 0.985,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 16, 14, 16),
          child: Row(
            children: <Widget>[
              Expanded(
                child: Text(
                  soru,
                  style: const TextStyle(fontSize: 21, height: 1.4),
                ),
              ),
              const SizedBox(width: 12),
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: renk.withValues(alpha: 0.12),
                ),
                child: Icon(CupertinoIcons.mic_fill, size: 24, color: renk),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
