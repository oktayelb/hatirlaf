import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Uygulamanin gorsel dili: 20 puntodan kucuk yazi yok, 64 pikselden
/// kucuk dokunulabilir alan yok, yuksek kontrast.
class HatirlaColors {
  const HatirlaColors._();

  static const Color paper = Color(0xFFF2F2F7);
  static const Color paperDark = Color(0xFFE5E5EA);

  /// Kartlar
  static const Color card = Color(0xFFFFFFFF);

  /// Ana vurgu: kiremit / terracotta. Sicak, "aile albumu" hissi.
  static const Color primary = Color(0xFFC2461A);
  static const Color primaryDark = Color(0xFF9A3412);
  static const Color primarySoft = Color(0xFFF9E9E3);

  /// Kayit kirmizisi
  static const Color record = Color(0xFFE0352A);
  static const Color recordSoft = Color(0xFFFCE7E5);

  /// Onay yesili
  static const Color confirm = Color(0xFF248A3D);
  static const Color confirmSoft = Color(0xFFE3F3E6);

  static const Color warning = Color(0xFF9A5200);
  static const Color warningSoft = Color(0xFFFFF2DA);

  /// Yazilar
  static const Color ink = Color(0xFF1C1C1E);
  static const Color inkSoft = Color(0xFF636366);

  /// Ayirici cizgiler
  static const Color line = Color(0xFFDCDCE1);
  static const Color hairline = Color(0x1A000000);
  static const Color chevron = Color(0xFFB4B4BA);
}

/// Her yerde ayni olculeri kullanalim.
class HatirlaSizes {
  const HatirlaSizes._();

  /// Parmak ucuyla rahat basilan en kucuk yukseklik.
  static const double tapTarget = 72;

  /// Ekran kenar boslugu.
  static const double gutter = 20;

  /// Kart kose yuvarlakligi.
  static const double radius = 24;
  static const double radiusSmall = 16;

  static const double toolbar = 68;
}

const String kYaziTipi = 'HatirlaSans';

const List<BoxShadow> kKartGolgesi = <BoxShadow>[
  BoxShadow(color: Color(0x08000000), blurRadius: 2, offset: Offset(0, 1)),
  BoxShadow(color: Color(0x0D000000), blurRadius: 18, offset: Offset(0, 6)),
];

RoundedSuperellipseBorder yumusakKose(
  double yaricap, {
  BorderSide kenar = BorderSide.none,
}) {
  return RoundedSuperellipseBorder(
    borderRadius: BorderRadius.circular(yaricap),
    side: kenar,
  );
}

class HatirlaKaydirma extends MaterialScrollBehavior {
  const HatirlaKaydirma();

  @override
  ScrollPhysics getScrollPhysics(BuildContext context) {
    return const BouncingScrollPhysics(
      parent: AlwaysScrollableScrollPhysics(),
    );
  }

  @override
  Widget buildOverscrollIndicator(
    BuildContext context,
    Widget child,
    ScrollableDetails details,
  ) {
    return child;
  }
}

ThemeData buildHatirlaTheme() {
  const ColorScheme scheme = ColorScheme(
    brightness: Brightness.light,
    primary: HatirlaColors.primary,
    onPrimary: Colors.white,
    primaryContainer: HatirlaColors.primarySoft,
    onPrimaryContainer: HatirlaColors.primaryDark,
    secondary: HatirlaColors.confirm,
    onSecondary: Colors.white,
    error: HatirlaColors.record,
    onError: Colors.white,
    surface: HatirlaColors.card,
    onSurface: HatirlaColors.ink,
    onSurfaceVariant: HatirlaColors.inkSoft,
    surfaceTint: Colors.transparent,
    surfaceContainerHighest: HatirlaColors.paperDark,
    outline: HatirlaColors.line,
    outlineVariant: HatirlaColors.line,
  );

  final TextTheme text = const TextTheme(
    displaySmall: TextStyle(
        fontSize: 40, fontWeight: FontWeight.w700, height: 1.15, letterSpacing: -0.8),
    headlineLarge: TextStyle(
        fontSize: 34, fontWeight: FontWeight.w700, height: 1.2, letterSpacing: -0.7),
    headlineMedium: TextStyle(
        fontSize: 29, fontWeight: FontWeight.w700, height: 1.25, letterSpacing: -0.5),
    headlineSmall: TextStyle(
        fontSize: 25, fontWeight: FontWeight.w700, height: 1.3, letterSpacing: -0.3),
    titleLarge: TextStyle(
        fontSize: 24, fontWeight: FontWeight.w600, height: 1.3, letterSpacing: -0.2),
    titleMedium: TextStyle(
        fontSize: 22, fontWeight: FontWeight.w600, height: 1.35, letterSpacing: -0.1),
    bodyLarge: TextStyle(fontSize: 22, fontWeight: FontWeight.w400, height: 1.5),
    bodyMedium: TextStyle(fontSize: 21, fontWeight: FontWeight.w400, height: 1.45),
    labelLarge: TextStyle(
        fontSize: 23, fontWeight: FontWeight.w600, height: 1.2, letterSpacing: -0.1),
  ).apply(
    fontFamily: kYaziTipi,
    bodyColor: HatirlaColors.ink,
    displayColor: HatirlaColors.ink,
  );

  return ThemeData(
    useMaterial3: true,
    fontFamily: kYaziTipi,
    colorScheme: scheme,
    scaffoldBackgroundColor: HatirlaColors.paper,
    canvasColor: HatirlaColors.paper,
    textTheme: text,
    splashFactory: NoSplash.splashFactory,
    highlightColor: const Color(0x0F000000),
    pageTransitionsTheme: const PageTransitionsTheme(
      builders: <TargetPlatform, PageTransitionsBuilder>{
        TargetPlatform.android: CupertinoPageTransitionsBuilder(),
        TargetPlatform.iOS: CupertinoPageTransitionsBuilder(),
      },
    ),
    cupertinoOverrideTheme: const CupertinoThemeData(
      primaryColor: HatirlaColors.primary,
      brightness: Brightness.light,
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: WidgetStateColor.resolveWith(
        (Set<WidgetState> durum) => durum.contains(WidgetState.scrolledUnder)
            ? const Color(0xFFF9F9FB)
            : HatirlaColors.paper,
      ),
      foregroundColor: HatirlaColors.ink,
      surfaceTintColor: Colors.transparent,
      shadowColor: const Color(0x66000000),
      elevation: 0,
      scrolledUnderElevation: 0.6,
      centerTitle: true,
      toolbarHeight: HatirlaSizes.toolbar,
      leadingWidth: 124,
      titleTextStyle: const TextStyle(
        fontFamily: kYaziTipi,
        fontSize: 23,
        fontWeight: FontWeight.w600,
        letterSpacing: -0.2,
        color: HatirlaColors.ink,
      ),
      iconTheme: const IconThemeData(size: 30, color: HatirlaColors.primary),
      systemOverlayStyle: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
        statusBarBrightness: Brightness.light,
      ),
    ),
    iconTheme: const IconThemeData(size: 30, color: HatirlaColors.ink),
    dividerTheme: const DividerThemeData(
      color: HatirlaColors.line,
      thickness: 1,
      space: 1,
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(HatirlaSizes.tapTarget),
        textStyle: text.labelLarge,
        shape: yumusakKose(HatirlaSizes.radius - 4),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        minimumSize: const Size(64, 64),
        textStyle: text.labelLarge,
        foregroundColor: HatirlaColors.primary,
        shape: yumusakKose(HatirlaSizes.radiusSmall),
      ),
    ),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: const Color(0xF51C1C1E),
      contentTextStyle: const TextStyle(
        fontFamily: kYaziTipi,
        fontSize: 21,
        color: Colors.white,
        height: 1.4,
      ),
      behavior: SnackBarBehavior.floating,
      elevation: 0,
      insetPadding: const EdgeInsets.all(16),
      shape: yumusakKose(20),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: HatirlaColors.card,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      shape: yumusakKose(30),
      titleTextStyle: text.headlineSmall,
      contentTextStyle: text.bodyLarge,
    ),
    bottomSheetTheme: BottomSheetThemeData(
      backgroundColor: HatirlaColors.paper,
      surfaceTintColor: Colors.transparent,
      modalBarrierColor: const Color(0x52000000),
      showDragHandle: true,
      dragHandleColor: const Color(0xFFC7C7CC),
      dragHandleSize: const Size(48, 6),
      shape: const RoundedSuperellipseBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: HatirlaColors.primary,
      linearTrackColor: HatirlaColors.paperDark,
    ),
    sliderTheme: const SliderThemeData(
      trackHeight: 8,
      activeTrackColor: HatirlaColors.primary,
      inactiveTrackColor: HatirlaColors.paperDark,
      thumbColor: Colors.white,
      disabledThumbColor: Colors.white,
      disabledActiveTrackColor: HatirlaColors.line,
      disabledInactiveTrackColor: HatirlaColors.paperDark,
      overlayColor: Colors.transparent,
      trackShape: RoundedRectSliderTrackShape(),
      thumbShape: RoundSliderThumbShape(
        enabledThumbRadius: 15,
        disabledThumbRadius: 15,
        elevation: 3,
        pressedElevation: 6,
      ),
      overlayShape: RoundSliderOverlayShape(overlayRadius: 28),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: HatirlaColors.card,
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
      hintStyle: const TextStyle(
        fontFamily: kYaziTipi,
        fontSize: 21,
        color: HatirlaColors.inkSoft,
      ),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: HatirlaColors.line),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: HatirlaColors.line),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: HatirlaColors.primary, width: 2),
      ),
    ),
    textSelectionTheme: const TextSelectionThemeData(
      cursorColor: HatirlaColors.primary,
      selectionColor: Color(0x40C2461A),
      selectionHandleColor: HatirlaColors.primary,
    ),
  );
}
