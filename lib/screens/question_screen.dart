import 'package:flutter/material.dart';

import '../data/prompts.dart';
import '../theme.dart';

/// Soru listesi. Secilen soru geriye dondurulur.
///
/// Konular kapali baslar; ekrani 70 soruyla doldurmak yerine once 10 basit
/// baslik gosteriyoruz. Secim yapmak, okumaktan kolaydir.
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
      appBar: AppBar(title: const Text('Ne Anlatalım?')),
      body: SafeArea(
        child: ListView.separated(
          padding: const EdgeInsets.all(HatirlaSizes.gutter),
          itemCount: Sorular.konular.length,
          separatorBuilder: (_, _) => const SizedBox(height: 14),
          itemBuilder: (BuildContext context, int i) {
            final SoruKonusu konu = Sorular.konular[i];
            final bool acik = _acikKonu == i;
            return Container(
              decoration: BoxDecoration(
                color: HatirlaColors.card,
                borderRadius: BorderRadius.circular(HatirlaSizes.radius),
                border: Border.all(
                  color: acik ? konu.renk : HatirlaColors.line,
                  width: acik ? 3 : 2,
                ),
              ),
              clipBehavior: Clip.antiAlias,
              child: Column(
                children: <Widget>[
                  InkWell(
                    onTap: () => setState(() => _acikKonu = acik ? null : i),
                    child: Padding(
                      padding: const EdgeInsets.all(18),
                      child: Row(
                        children: <Widget>[
                          Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              color: konu.renk.withValues(alpha: 0.14),
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Icon(konu.ikon, size: 34, color: konu.renk),
                          ),
                          const SizedBox(width: 16),
                          Expanded(
                            child: Text(
                              konu.ad,
                              style: const TextStyle(
                                fontSize: 25,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                          Icon(
                            acik
                                ? Icons.keyboard_arrow_up_rounded
                                : Icons.keyboard_arrow_down_rounded,
                            size: 40,
                            color: HatirlaColors.inkSoft,
                          ),
                        ],
                      ),
                    ),
                  ),
                  if (acik)
                    Column(
                      children: <Widget>[
                        const Divider(height: 2),
                        for (final String soru in konu.sorular)
                          InkWell(
                            onTap: () => Navigator.of(context).pop(soru),
                            child: Container(
                              width: double.infinity,
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 18, vertical: 20),
                              decoration: const BoxDecoration(
                                border: Border(
                                  bottom: BorderSide(
                                      color: HatirlaColors.paperDark, width: 2),
                                ),
                              ),
                              child: Row(
                                children: <Widget>[
                                  Expanded(
                                    child: Text(
                                      soru,
                                      style: const TextStyle(
                                          fontSize: 22, height: 1.4),
                                    ),
                                  ),
                                  const SizedBox(width: 10),
                                  Icon(Icons.mic_rounded,
                                      size: 30, color: konu.renk),
                                ],
                              ),
                            ),
                          ),
                      ],
                    ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}
