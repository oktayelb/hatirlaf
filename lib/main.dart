import 'package:audio_session/audio_session.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:intl/date_symbol_data_local.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'screens/home_screen.dart';
import 'screens/welcome_screen.dart';
import 'services/cover_photo.dart';
import 'services/store.dart';
import 'services/transcriber.dart';
import 'services/whisper_model_manager.dart';
import 'theme.dart';

/// Karsilama ekraninin gosterilip gosterilmedigini tutan anahtar.
const String kKarsilamaAnahtari = 'karsilama_tamamlandi';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Telefonu yan cevirmek yaslilarda sik kaza; dikey sabit.
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
    await CoverPhoto.instance.load();
    await WhisperModelManager.instance.init();
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    karsilamaTamam = prefs.getBool(kKarsilamaAnahtari) ?? false;
  } catch (e, s) {
    // Acilista bir sey patlarsa bile uygulama acilsin: kullanici en azindan
    // eski hatiralarini gorebilmeli.
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

  runApp(HatirlaApp(karsilamaTamam: karsilamaTamam));
}

class HatirlaApp extends StatelessWidget {
  const HatirlaApp({super.key, required this.karsilamaTamam});

  final bool karsilamaTamam;

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
        // Yazilar zaten buyuk. Sistemde "en buyuk yazi" secili bir telefonda
        // kat kat buyuyup butonlarin tasmasini engelliyoruz; ama kullanici
        // sistemden buyutmusse birazini onurlandiriyoruz.
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
      home: karsilamaTamam ? const HomeScreen() : const WelcomeScreen(),
    );
  }
}
