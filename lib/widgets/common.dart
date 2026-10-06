import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme.dart';

class Basilabilir extends StatefulWidget {
  const Basilabilir({
    super.key,
    required this.child,
    required this.onTap,
    this.olcek = 0.97,
    this.titresim = HapticFeedback.selectionClick,
  });

  final Widget child;
  final VoidCallback? onTap;
  final double olcek;
  final Future<void> Function() titresim;

  @override
  State<Basilabilir> createState() => _BasilabilirState();
}

class _BasilabilirState extends State<Basilabilir> {
  bool _basili = false;

  void _basiliYap(bool basili) {
    if (_basili == basili) return;
    setState(() => _basili = basili);
  }

  void _dokunuldu() {
    widget.titresim();
    widget.onTap!();
  }

  @override
  Widget build(BuildContext context) {
    final bool aktif = widget.onTap != null;
    final Duration sure = _basili
        ? const Duration(milliseconds: 90)
        : const Duration(milliseconds: 280);
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTapDown: aktif ? (_) => _basiliYap(true) : null,
      onTapUp: aktif ? (_) => _basiliYap(false) : null,
      onTapCancel: aktif ? () => _basiliYap(false) : null,
      onTap: aktif ? _dokunuldu : null,
      child: AnimatedScale(
        scale: _basili ? widget.olcek : 1,
        duration: sure,
        curve: _basili ? Curves.easeOut : Curves.easeOutBack,
        child: AnimatedOpacity(
          opacity: _basili ? 0.78 : 1,
          duration: sure,
          curve: Curves.easeOut,
          child: widget.child,
        ),
      ),
    );
  }
}

class Kart extends StatelessWidget {
  const Kart({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(20),
    this.renk = HatirlaColors.card,
    this.golge = true,
    this.kirp = false,
    this.onTap,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color renk;
  final bool golge;
  final bool kirp;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    Widget icerik = Padding(padding: padding, child: child);
    if (kirp) {
      icerik = ClipRSuperellipse(
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        child: icerik,
      );
    }
    final Widget kutu = DecoratedBox(
      decoration: ShapeDecoration(
        color: renk,
        shape: yumusakKose(
          HatirlaSizes.radius,
          kenar: const BorderSide(color: HatirlaColors.hairline),
        ),
        shadows: golge ? kKartGolgesi : null,
      ),
      child: icerik,
    );
    if (onTap == null) return kutu;
    return Basilabilir(onTap: onTap, child: kutu);
  }
}

class CerceveliFoto extends StatelessWidget {
  const CerceveliFoto({
    super.key,
    required this.foto,
    this.yukseklik,
    this.genislik = double.infinity,
    this.yaricap = HatirlaSizes.radius,
    this.bosluk = 6,
  });

  final Widget foto;
  final double? yukseklik;
  final double genislik;
  final double yaricap;
  final double bosluk;

  @override
  Widget build(BuildContext context) {
    final double icYaricap = yaricap - bosluk;
    return Container(
      width: genislik,
      padding: EdgeInsets.all(bosluk),
      decoration: ShapeDecoration(
        color: HatirlaColors.card,
        shape: yumusakKose(
          yaricap,
          kenar: const BorderSide(color: Color(0x24000000)),
        ),
        shadows: kKartGolgesi,
      ),
      child: Container(
        foregroundDecoration: ShapeDecoration(
          shape: yumusakKose(
            icYaricap,
            kenar: const BorderSide(color: Color(0x14000000)),
          ),
        ),
        child: ClipRSuperellipse(
          borderRadius: BorderRadius.circular(icYaricap),
          child: SizedBox(
            width: double.infinity,
            height: yukseklik,
            child: foto,
          ),
        ),
      ),
    );
  }
}

class BuyukSimge extends StatelessWidget {
  const BuyukSimge({
    super.key,
    required this.ikon,
    this.renk = HatirlaColors.primary,
    this.boyut = 120,
  });

  final IconData ikon;
  final Color renk;
  final double boyut;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: boyut,
      height: boyut,
      decoration: ShapeDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: <Color>[Color.lerp(renk, Colors.white, 0.28)!, renk],
        ),
        shape: yumusakKose(boyut * 0.27),
        shadows: <BoxShadow>[
          BoxShadow(
            color: renk.withValues(alpha: 0.28),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Icon(ikon, size: boyut * 0.5, color: Colors.white),
    );
  }
}

class IkonRozeti extends StatelessWidget {
  const IkonRozeti({
    super.key,
    required this.ikon,
    required this.renk,
    this.boyut = 44,
  });

  final IconData ikon;
  final Color renk;
  final double boyut;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: boyut,
      height: boyut,
      decoration: ShapeDecoration(
        color: renk,
        shape: yumusakKose(boyut * 0.28),
      ),
      child: Icon(ikon, size: boyut * 0.58, color: Colors.white),
    );
  }
}

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
    final bool aktif = onPressed != null;
    final Color arka = aktif
        ? (renk ?? HatirlaColors.primary)
        : HatirlaColors.paperDark;
    final Color on = aktif
        ? (yaziRengi ?? Colors.white)
        : HatirlaColors.inkSoft;
    return Semantics(
      button: true,
      enabled: aktif,
      label: altYazi == null ? yazi : '$yazi. $altYazi',
      onTap: onPressed,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onPressed,
        titresim: HapticFeedback.mediumImpact,
        child: MediaQuery.withClampedTextScaling(
          maxScaleFactor: 1.15,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 250),
            curve: Curves.easeOut,
            constraints: BoxConstraints(minHeight: yukseklik),
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 12),
            decoration: ShapeDecoration(
              color: arka,
              shape: yumusakKose(HatirlaSizes.radius - 2),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Icon(ikon, size: 32, color: on),
                const SizedBox(width: 14),
                Flexible(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Text(
                        yazi,
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w700,
                          height: 1.2,
                          letterSpacing: -0.3,
                          color: on,
                        ),
                      ),
                      if (altYazi != null) ...<Widget>[
                        const SizedBox(height: 3),
                        Text(
                          altYazi!,
                          style: TextStyle(
                            fontSize: 19,
                            height: 1.25,
                            color: on.withValues(alpha: 0.88),
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class IkincilButon extends StatelessWidget {
  const IkincilButon({
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
    final bool aktif = onPressed != null;
    final Color c = renk ?? HatirlaColors.primary;
    final Color yaziRengi = aktif
        ? Color.lerp(c, Colors.black, 0.18)!
        : HatirlaColors.inkSoft;
    return Semantics(
      button: true,
      enabled: aktif,
      label: yazi,
      onTap: onPressed,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onPressed,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          constraints: const BoxConstraints(minHeight: 76),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
          decoration: ShapeDecoration(
            color: aktif ? c.withValues(alpha: 0.12) : HatirlaColors.paperDark,
            shape: yumusakKose(HatirlaSizes.radius - 2),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Icon(ikon, size: 28, color: yaziRengi),
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  yazi,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w600,
                    letterSpacing: -0.2,
                    color: yaziRengi,
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

class DuzButon extends StatelessWidget {
  const DuzButon({
    super.key,
    required this.yazi,
    required this.onPressed,
    this.ikon,
    this.renk = HatirlaColors.primary,
  });

  final String yazi;
  final IconData? ikon;
  final VoidCallback? onPressed;
  final Color renk;

  @override
  Widget build(BuildContext context) {
    final Color c = onPressed == null ? HatirlaColors.inkSoft : renk;
    return Semantics(
      button: true,
      enabled: onPressed != null,
      label: yazi,
      onTap: onPressed,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onPressed,
        olcek: 0.95,
        child: ConstrainedBox(
          constraints: const BoxConstraints(minHeight: 64),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                if (ikon != null) ...<Widget>[
                  Icon(ikon, size: 26, color: c),
                  const SizedBox(width: 8),
                ],
                Flexible(
                  child: Text(
                    yazi,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 21,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -0.2,
                      color: c,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class GeriDugmesi extends StatelessWidget {
  const GeriDugmesi({super.key, required this.onPressed});

  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: 'Geri',
      onTap: onPressed,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onPressed,
        olcek: 0.94,
        child: const Padding(
          padding: EdgeInsets.only(left: 8),
          child: Row(
            children: <Widget>[
              Icon(
                CupertinoIcons.chevron_back,
                size: 30,
                color: HatirlaColors.primary,
              ),
              SizedBox(width: 2),
              Flexible(
                child: Text(
                  'Geri',
                  maxLines: 1,
                  style: TextStyle(fontSize: 22, color: HatirlaColors.primary),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class UstCubuk extends StatelessWidget implements PreferredSizeWidget {
  const UstCubuk({
    super.key,
    required this.baslik,
    this.onGeri,
    this.eylemler,
    this.baslikGorunur = true,
  });

  final String baslik;
  final VoidCallback? onGeri;
  final List<Widget>? eylemler;
  final bool baslikGorunur;

  @override
  Size get preferredSize => const Size.fromHeight(HatirlaSizes.toolbar);

  @override
  Widget build(BuildContext context) {
    final bool geriVar =
        onGeri != null || (ModalRoute.of(context)?.canPop ?? false);
    return AppBar(
      automaticallyImplyLeading: false,
      leading: geriVar
          ? GeriDugmesi(
              onPressed: onGeri ?? () => Navigator.of(context).maybePop(),
            )
          : null,
      title: AnimatedOpacity(
        opacity: baslikGorunur ? 1 : 0,
        duration: const Duration(milliseconds: 200),
        child: FittedBox(fit: BoxFit.scaleDown, child: Text(baslik)),
      ),
      actions: eylemler,
    );
  }
}

class IlerlemeCubugu extends StatelessWidget {
  const IlerlemeCubugu({
    super.key,
    required this.deger,
    this.renk = HatirlaColors.primary,
    this.yukseklik = 12,
  });

  final double? deger;
  final Color renk;
  final double yukseklik;

  @override
  Widget build(BuildContext context) {
    final BorderRadius yuvarlak = BorderRadius.circular(yukseklik / 2);
    final double? oran = deger;
    return ClipRRect(
      borderRadius: yuvarlak,
      child: SizedBox(
        height: yukseklik,
        child: oran == null
            ? LinearProgressIndicator(
                minHeight: yukseklik,
                color: renk,
                backgroundColor: HatirlaColors.paperDark,
              )
            : DecoratedBox(
                decoration: const BoxDecoration(color: HatirlaColors.paperDark),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: AnimatedFractionallySizedBox(
                    duration: const Duration(milliseconds: 350),
                    curve: Curves.easeOutCubic,
                    widthFactor: oran.clamp(0.0, 1.0),
                    heightFactor: 1,
                    child: DecoratedBox(
                      decoration: BoxDecoration(
                        color: renk,
                        borderRadius: yuvarlak,
                      ),
                    ),
                  ),
                ),
              ),
      ),
    );
  }
}

class _Pencere extends StatelessWidget {
  const _Pencere({
    required this.ikon,
    required this.ikonRengi,
    required this.baslik,
    required this.mesaj,
    required this.butonlar,
  });

  final IconData ikon;
  final Color ikonRengi;
  final String baslik;
  final String mesaj;
  final List<Widget> butonlar;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 440),
            child: Material(
              color: HatirlaColors.card,
              shape: yumusakKose(32),
              clipBehavior: Clip.antiAlias,
              elevation: 0,
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(24, 30, 24, 22),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: <Widget>[
                    Center(
                      child: Container(
                        width: 72,
                        height: 72,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: ikonRengi.withValues(alpha: 0.12),
                        ),
                        child: Icon(ikon, size: 38, color: ikonRengi),
                      ),
                    ),
                    const SizedBox(height: 18),
                    Text(
                      baslik,
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    const SizedBox(height: 10),
                    Text(
                      mesaj,
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: HatirlaColors.inkSoft,
                      ),
                    ),
                    const SizedBox(height: 26),
                    for (int i = 0; i < butonlar.length; i++) ...<Widget>[
                      if (i > 0) const SizedBox(height: 10),
                      butonlar[i],
                    ],
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

Future<T?> _pencereAc<T>(BuildContext context, WidgetBuilder icerik) {
  return showGeneralDialog<T>(
    context: context,
    barrierDismissible: true,
    barrierLabel: 'Kapat',
    barrierColor: const Color(0x59000000),
    transitionDuration: const Duration(milliseconds: 280),
    pageBuilder: (BuildContext context, _, _) => icerik(context),
    transitionBuilder:
        (BuildContext context, Animation<double> animasyon, _, Widget child) {
          final CurvedAnimation egri = CurvedAnimation(
            parent: animasyon,
            curve: Curves.easeOutCubic,
            reverseCurve: Curves.easeInCubic,
          );
          return FadeTransition(
            opacity: egri,
            child: ScaleTransition(
              scale: Tween<double>(begin: 1.1, end: 1).animate(egri),
              child: child,
            ),
          );
        },
  );
}

/// Evet/hayir sorusu; butonlar alt alta ve buyuk. [tehlikeli] ise onay
/// butonu kirmizi ve ikinci sirada.
Future<bool> onayIste(
  BuildContext context, {
  required String baslik,
  required String mesaj,
  String evetYazi = 'Evet',
  String hayirYazi = 'Vazgeç',
  IconData evetIkon = CupertinoIcons.checkmark_alt,
  bool tehlikeli = false,
}) async {
  final bool? sonuc = await _pencereAc<bool>(
    context,
    (BuildContext context) => _Pencere(
      ikon: tehlikeli
          ? CupertinoIcons.exclamationmark_triangle_fill
          : CupertinoIcons.question,
      ikonRengi: tehlikeli ? HatirlaColors.record : HatirlaColors.primary,
      baslik: baslik,
      mesaj: mesaj,
      butonlar: <Widget>[
        IkincilButon(
          yazi: hayirYazi,
          ikon: CupertinoIcons.arrow_left,
          renk: HatirlaColors.inkSoft,
          onPressed: () => Navigator.of(context).pop(false),
        ),
        BuyukButon(
          yazi: evetYazi,
          ikon: evetIkon,
          yukseklik: 76,
          renk: tehlikeli ? HatirlaColors.record : HatirlaColors.confirm,
          onPressed: () => Navigator.of(context).pop(true),
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
  IconData ikon = CupertinoIcons.info,
  Color? renk,
}) {
  return _pencereAc<void>(
    context,
    (BuildContext context) => _Pencere(
      ikon: ikon,
      ikonRengi: renk ?? HatirlaColors.primary,
      baslik: baslik,
      mesaj: mesaj,
      butonlar: <Widget>[
        BuyukButon(
          yazi: 'Tamam',
          ikon: CupertinoIcons.checkmark_alt,
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
      SnackBar(content: Text(mesaj), duration: const Duration(seconds: 4)),
    );
}

class BolumBasligi extends StatelessWidget {
  const BolumBasligi(this.yazi, {super.key, this.ikon});

  final String yazi;
  final IconData? ikon;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 4),
      child: Text(yazi, style: Theme.of(context).textTheme.headlineSmall),
    );
  }
}

class Grup extends StatelessWidget {
  const Grup({super.key, required this.satirlar, this.baslik, this.dipnot});

  final List<Widget> satirlar;
  final String? baslik;
  final String? dipnot;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        if (baslik != null) ...<Widget>[
          BolumBasligi(baslik!),
          const SizedBox(height: 12),
        ],
        Kart(
          padding: EdgeInsets.zero,
          kirp: true,
          child: Column(
            children: <Widget>[
              for (int i = 0; i < satirlar.length; i++) ...<Widget>[
                if (i > 0) const Divider(indent: 20, endIndent: 0),
                satirlar[i],
              ],
            ],
          ),
        ),
        if (dipnot != null) ...<Widget>[
          const SizedBox(height: 10),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6),
            child: Text(
              dipnot!,
              style: const TextStyle(
                fontSize: 19,
                height: 1.4,
                color: HatirlaColors.inkSoft,
              ),
            ),
          ),
        ],
      ],
    );
  }
}

class GrupSatiri extends StatelessWidget {
  const GrupSatiri({
    super.key,
    required this.baslik,
    this.ikon,
    this.ikonRengi = HatirlaColors.inkSoft,
    this.deger,
    this.onTap,
  });

  final String baslik;
  final IconData? ikon;
  final Color ikonRengi;
  final String? deger;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final Widget satir = ConstrainedBox(
      constraints: const BoxConstraints(minHeight: HatirlaSizes.tapTarget),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
        child: Row(
          children: <Widget>[
            if (ikon != null) ...<Widget>[
              IkonRozeti(ikon: ikon!, renk: ikonRengi),
              const SizedBox(width: 16),
            ],
            Expanded(
              child: Wrap(
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: 12,
                runSpacing: 2,
                children: <Widget>[
                  Text(
                    baslik,
                    style: const TextStyle(fontSize: 21, height: 1.3),
                  ),
                  if (deger != null)
                    Text(
                      deger!,
                      style: const TextStyle(
                        fontSize: 21,
                        height: 1.3,
                        color: HatirlaColors.inkSoft,
                      ),
                    ),
                ],
              ),
            ),
            if (onTap != null) ...<Widget>[
              const SizedBox(width: 8),
              const Icon(
                CupertinoIcons.chevron_forward,
                size: 24,
                color: HatirlaColors.chevron,
              ),
            ],
          ],
        ),
      ),
    );
    if (onTap == null) return satir;
    return Semantics(
      button: true,
      label: baslik,
      onTap: onTap,
      excludeSemantics: true,
      child: Basilabilir(onTap: onTap, olcek: 0.985, child: satir),
    );
  }
}

class EylemSatiri extends StatelessWidget {
  const EylemSatiri({
    super.key,
    required this.yazi,
    required this.ikon,
    required this.onTap,
    this.renk = HatirlaColors.primary,
  });

  final String yazi;
  final IconData ikon;
  final VoidCallback? onTap;
  final Color renk;

  @override
  Widget build(BuildContext context) {
    final Color c = onTap == null ? HatirlaColors.inkSoft : renk;
    return Semantics(
      button: true,
      enabled: onTap != null,
      label: yazi,
      onTap: onTap,
      excludeSemantics: true,
      child: Basilabilir(
        onTap: onTap,
        olcek: 0.985,
        child: ConstrainedBox(
          constraints: const BoxConstraints(minHeight: HatirlaSizes.tapTarget),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            child: Row(
              children: <Widget>[
                Icon(ikon, size: 28, color: c),
                const SizedBox(width: 14),
                Expanded(
                  child: Text(
                    yazi,
                    style: TextStyle(
                      fontSize: 21,
                      height: 1.3,
                      fontWeight: FontWeight.w600,
                      color: c,
                    ),
                  ),
                ),
              ],
            ),
          ),
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
