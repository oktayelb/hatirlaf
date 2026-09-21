import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme.dart';

/// Ana eylem butonu: buyuk, ikonlu ve her zaman yazili. Ikon tek basina
/// birakilmiyor.
class BuyukButon extends StatelessWidget {
  const BuyukButon({
    super.key,
    required this.yazi,
    required this.ikon,
    required this.onPressed,
    this.altYazi,
    this.renk,
    this.yaziRengi,
    this.yukseklik = 88,
  });

  final String yazi;
  final String? altYazi;
  final IconData ikon;
  final VoidCallback? onPressed;
  final Color? renk;
  final Color? yaziRengi;
  final double yukseklik;

  @override
  Widget build(BuildContext context) {
    final Color arka = renk ?? HatirlaColors.primary;
    final Color on = yaziRengi ?? Colors.white;
    return Semantics(
      button: true,
      label: altYazi == null ? yazi : '$yazi. $altYazi',
      child: FilledButton(
        onPressed: onPressed == null
            ? null
            : () {
                HapticFeedback.mediumImpact();
                onPressed!();
              },
        style: FilledButton.styleFrom(
          backgroundColor: arka,
          foregroundColor: on,
          disabledBackgroundColor: HatirlaColors.line,
          disabledForegroundColor: HatirlaColors.inkSoft,
          minimumSize: Size.fromHeight(yukseklik),
          padding: const EdgeInsets.symmetric(horizontal: 20),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(HatirlaSizes.radius),
          ),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Icon(ikon, size: 38),
            const SizedBox(width: 16),
            Flexible(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    yazi,
                    style: const TextStyle(
                      fontSize: 26,
                      fontWeight: FontWeight.w700,
                      height: 1.2,
                    ),
                  ),
                  if (altYazi != null) ...<Widget>[
                    const SizedBox(height: 4),
                    Text(
                      altYazi!,
                      style: TextStyle(
                        fontSize: 19,
                        fontWeight: FontWeight.w400,
                        height: 1.25,
                        color: on.withValues(alpha: 0.9),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Ikincil buton: cerceveli, ayni buyuklukte.
class CerceveliButon extends StatelessWidget {
  const CerceveliButon({
    super.key,
    required this.yazi,
    required this.ikon,
    required this.onPressed,
    this.renk,
  });

  final String yazi;
  final IconData ikon;
  final VoidCallback? onPressed;
  final Color? renk;

  @override
  Widget build(BuildContext context) {
    final Color c = renk ?? HatirlaColors.primaryDark;
    return OutlinedButton(
      onPressed: onPressed == null
          ? null
          : () {
              HapticFeedback.selectionClick();
              onPressed!();
            },
      style: OutlinedButton.styleFrom(
        foregroundColor: c,
        side: BorderSide(color: c, width: 2.5),
        minimumSize: const Size.fromHeight(76),
        padding: const EdgeInsets.symmetric(horizontal: 18),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          Icon(ikon, size: 32),
          const SizedBox(width: 14),
          Flexible(
            child: Text(
              yazi,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 23, fontWeight: FontWeight.w700),
            ),
          ),
        ],
      ),
    );
  }
}

/// Evet/hayir sorusu; butonlar alt alta ve buyuk. [tehlikeli] ise onay
/// butonu kirmizi ve ikinci sirada.
Future<bool> onayIste(
  BuildContext context, {
  required String baslik,
  required String mesaj,
  String evetYazi = 'Evet',
  String hayirYazi = 'Vazgeç',
  IconData evetIkon = Icons.check_rounded,
  bool tehlikeli = false,
}) async {
  final bool? sonuc = await showDialog<bool>(
    context: context,
    builder: (BuildContext context) => AlertDialog(
      icon: Icon(
        tehlikeli ? Icons.warning_amber_rounded : Icons.help_outline_rounded,
        size: 56,
        color: tehlikeli ? HatirlaColors.record : HatirlaColors.primary,
      ),
      title: Text(baslik, textAlign: TextAlign.center),
      content: Text(mesaj, textAlign: TextAlign.center),
      actionsPadding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
      actions: <Widget>[
        Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            CerceveliButon(
              yazi: hayirYazi,
              ikon: Icons.arrow_back_rounded,
              onPressed: () => Navigator.of(context).pop(false),
            ),
            const SizedBox(height: 12),
            BuyukButon(
              yazi: evetYazi,
              ikon: evetIkon,
              yukseklik: 76,
              renk: tehlikeli ? HatirlaColors.record : HatirlaColors.confirm,
              onPressed: () => Navigator.of(context).pop(true),
            ),
          ],
        ),
      ],
    ),
  );
  return sonuc ?? false;
}

/// Tek "Tamam" butonlu bilgi/hata penceresi.
Future<void> bilgiGoster(
  BuildContext context, {
  required String baslik,
  required String mesaj,
  IconData ikon = Icons.info_outline_rounded,
  Color? renk,
}) {
  return showDialog<void>(
    context: context,
    builder: (BuildContext context) => AlertDialog(
      icon: Icon(ikon, size: 56, color: renk ?? HatirlaColors.primary),
      title: Text(baslik, textAlign: TextAlign.center),
      content: Text(mesaj, textAlign: TextAlign.center),
      actionsPadding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
      actions: <Widget>[
        BuyukButon(
          yazi: 'Tamam',
          ikon: Icons.check_rounded,
          yukseklik: 76,
          onPressed: () => Navigator.of(context).pop(),
        ),
      ],
    ),
  );
}

/// Ekranin altinda kisa uyari.
void kisaMesaj(BuildContext context, String mesaj) {
  ScaffoldMessenger.of(context)
    ..clearSnackBars()
    ..showSnackBar(
      SnackBar(
        content: Text(mesaj),
        duration: const Duration(seconds: 4),
      ),
    );
}

/// Basligi ve altinda aciklamasi olan bolum basligi.
class BolumBasligi extends StatelessWidget {
  const BolumBasligi(this.yazi, {super.key, this.ikon});

  final String yazi;
  final IconData? ikon;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: <Widget>[
        if (ikon != null) ...<Widget>[
          Icon(ikon, size: 30, color: HatirlaColors.primary),
          const SizedBox(width: 10),
        ],
        Expanded(
          child: Text(
            yazi,
            style: Theme.of(context).textTheme.headlineSmall,
          ),
        ),
      ],
    );
  }
}
