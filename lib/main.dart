import 'dart:async';

import 'package:audio_session/audio_session.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/date_symbol_data_local.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'screens/home_screen.dart';
import 'screens/kim_screen.dart';
import 'screens/welcome_screen.dart';
import 'services/cover_photo.dart';
import 'services/kullanici.dart';
import 'services/kurtarma.dart';
import 'services/store.dart';
import 'services/transcriber.dart';
import 'services/updater.dart';
import 'services/uploader.dart';
import 'services/whisper_model_manager.dart';
import 'theme.dart';

/// Karsilama ekraninin gosterilip gosterilmedigini tutan anahtar.
const String kKarsilamaAnahtari = 'karsilama_tamamlandi';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Dikey sabit.
  await SystemChrome.setPreferredOrientations(<DeviceOrientation>[
    DeviceOrientation.portraitUp,
  ]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));

  // Turkce tarih adlari ("12 Eylül 2025, Cuma").
  await initializeDateFormatting('tr_TR', null);

  bool karsilamaTamam = false;
  try {
    await MemoryStore.instance.load();
    // Telefon kayit sirasinda uygulamayi oldurduyse hatira diskte yarim
    // durur; dizine girmeden once onu geri alalim.
    await Kurtarma.tara();
    await CoverPhoto.instance.load();
    await Kullanici.instance.yukle();
    await WhisperModelManager.instance.init();
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    karsilamaTamam = prefs.getBool(kKarsilamaAnahtari) ?? false;
  } catch (e, s) {
    // Acilista bir sey patlasa bile uygulama acilsin.
    debugPrint('Acilis hatasi: $e\n$s');
  }

  try {
    final AudioSession session = await AudioSession.instance;
    await session.configure(const AudioSessionConfiguration.speech());
  } catch (e) {
    debugPrint('Ses oturumu ayarlanamadi: $e');
  }

  // Yarim kalmis yaziya cevirme islerini devral.
  Transcriber.instance.resumePending();

  // Bilerek beklenmiyor: agi olmayan telefonda acilisi geciktirmesin.
  unawaited(Guncelleyici.instance.baslat());
  unawaited(Yedekleyici.instance.baslat());

  runApp(HatirlaApp(
    karsilamaTamam: karsilamaTamam,
    akrabaSecildi: Kullanici.instance.secildi,
  ));
}

class HatirlaApp extends StatelessWidget {
  const HatirlaApp({
    super.key,
    required this.karsilamaTamam,
    required this.akrabaSecildi,
  });

  final bool karsilamaTamam;

  /// Telefonu kimin kullandigi secildi mi? Secilmediyse her sey bundan
  /// once geliyor - guncelleme ile gelen, karsilamasi coktan bitmis
  /// telefonlarda da.
  final bool akrabaSecildi;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'hatırlaf',
      debugShowCheckedModeBanner: false,
      theme: buildHatirlaTheme(),
      locale: const Locale('tr', 'TR'),
      supportedLocales: const <Locale>[Locale('tr', 'TR')],
      localizationsDelegates: const <LocalizationsDelegate<dynamic>>[
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      builder: (BuildContext context, Widget? child) {
        // Yazilar zaten buyuk; sistemden gelen olcegi sinirliyoruz ama
        // tamamen yok saymiyoruz.
        final MediaQueryData mq = MediaQuery.of(context);
        return MediaQuery(
          data: mq.copyWith(
            textScaler: TextScaler.linear(
              mq.textScaler.scale(1).clamp(1.0, 1.35),
            ),
          ),
          child: child ?? const SizedBox.shrink(),
        );
      },
      home: akrabaSecildi
          ? (karsilamaTamam ? const HomeScreen() : const WelcomeScreen())
          : KimSinizScreen(karsilamaTamam: karsilamaTamam),
    );
  }
}
