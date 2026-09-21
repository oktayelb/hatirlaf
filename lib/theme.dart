import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Uygulamanin gorsel dili: 20 puntodan kucuk yazi yok, 64 pikselden
/// kucuk dokunulabilir alan yok, yuksek kontrast, gri/ince oge yok.
class HatirlaColors {
  const HatirlaColors._();

  /// Sicak kagit rengi arka plan (beyaz parlamasi goz yormuyor).
  static const Color paper = Color(0xFFFBF3E7);
  static const Color paperDark = Color(0xFFF2E7D5);

  /// Kartlar
  static const Color card = Color(0xFFFFFFFF);

  /// Ana vurgu: kiremit / terracotta. Sicak, "aile albumu" hissi.
  static const Color primary = Color(0xFFA8461B);
  static const Color primaryDark = Color(0xFF7E3312);
  static const Color primarySoft = Color(0xFFF6E2D6);

  /// Kayit kirmizisi
  static const Color record = Color(0xFFC0392B);

  /// Onay yesili
  static const Color confirm = Color(0xFF2E6E3F);

  /// Yazilar
  static const Color ink = Color(0xFF2B231C);
  static const Color inkSoft = Color(0xFF5E5045);

  /// Ayirici cizgiler
  static const Color line = Color(0xFFE0D2BE);
}

/// Her yerde ayni olculeri kullanalim.
class HatirlaSizes {
  const HatirlaSizes._();

  /// Parmak ucuyla rahat basilan en kucuk yukseklik.
  static const double tapTarget = 72;

  /// Ekran kenar boslugu.
  static const double gutter = 20;

  /// Kart kose yuvarlakligi.
  static const double radius = 22;
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
    surfaceContainerHighest: HatirlaColors.paperDark,
    outline: HatirlaColors.line,
  );

  final TextTheme text = const TextTheme(
    displaySmall: TextStyle(fontSize: 44, fontWeight: FontWeight.w700, height: 1.15),
    headlineLarge: TextStyle(fontSize: 36, fontWeight: FontWeight.w700, height: 1.2),
    headlineMedium: TextStyle(fontSize: 30, fontWeight: FontWeight.w700, height: 1.25),
    headlineSmall: TextStyle(fontSize: 26, fontWeight: FontWeight.w700, height: 1.3),
    titleLarge: TextStyle(fontSize: 26, fontWeight: FontWeight.w600, height: 1.3),
    titleMedium: TextStyle(fontSize: 23, fontWeight: FontWeight.w600, height: 1.35),
    bodyLarge: TextStyle(fontSize: 23, fontWeight: FontWeight.w400, height: 1.55),
    bodyMedium: TextStyle(fontSize: 21, fontWeight: FontWeight.w400, height: 1.5),
    labelLarge: TextStyle(fontSize: 25, fontWeight: FontWeight.w700, height: 1.2),
  ).apply(bodyColor: HatirlaColors.ink, displayColor: HatirlaColors.ink);

  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: HatirlaColors.paper,
    textTheme: text,
    // Titreyen parmakta "basildi" dalgasi kucuk kalsin ama kaybolmasin.
    splashFactory: InkRipple.splashFactory,
    appBarTheme: const AppBarTheme(
      backgroundColor: HatirlaColors.paper,
      foregroundColor: HatirlaColors.ink,
      elevation: 0,
      scrolledUnderElevation: 0,
      centerTitle: true,
      toolbarHeight: 76,
      titleTextStyle: TextStyle(
        fontSize: 28,
        fontWeight: FontWeight.w700,
        color: HatirlaColors.ink,
      ),
      iconTheme: IconThemeData(size: 34, color: HatirlaColors.ink),
      systemOverlayStyle: SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
      ),
    ),
    iconTheme: const IconThemeData(size: 32, color: HatirlaColors.ink),
    dividerTheme: const DividerThemeData(color: HatirlaColors.line, thickness: 2),
    cardTheme: CardThemeData(
      color: HatirlaColors.card,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        side: const BorderSide(color: HatirlaColors.line, width: 2),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size.fromHeight(HatirlaSizes.tapTarget),
        textStyle: text.labelLarge,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        ),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size.fromHeight(HatirlaSizes.tapTarget),
        textStyle: text.labelLarge,
        foregroundColor: HatirlaColors.primaryDark,
        side: const BorderSide(color: HatirlaColors.primary, width: 2.5),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(HatirlaSizes.radius),
        ),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        minimumSize: const Size(64, 64),
        textStyle: text.labelLarge,
        foregroundColor: HatirlaColors.primaryDark,
      ),
    ),
    snackBarTheme: SnackBarThemeData(
      backgroundColor: HatirlaColors.ink,
      contentTextStyle: const TextStyle(fontSize: 22, color: Colors.white, height: 1.4),
      behavior: SnackBarBehavior.floating,
      insetPadding: const EdgeInsets.all(16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: HatirlaColors.card,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(26)),
      titleTextStyle: text.headlineSmall,
      contentTextStyle: text.bodyLarge,
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(
      color: HatirlaColors.primary,
      linearMinHeight: 18,
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: HatirlaColors.paperDark,
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 22),
      hintStyle: const TextStyle(fontSize: 22, color: HatirlaColors.inkSoft),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: HatirlaColors.line, width: 2),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: HatirlaColors.line, width: 2),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(18),
        borderSide: const BorderSide(color: HatirlaColors.primary, width: 3),
      ),
    ),
  );
}
