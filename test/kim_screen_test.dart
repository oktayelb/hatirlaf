import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hatirla/data/akrabalar.dart';
import 'package:hatirla/screens/kim_screen.dart';
import 'package:hatirla/services/kullanici.dart';
import 'package:hatirla/theme.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const MethodChannel kayitKanali =
      MethodChannel('com.llfbandit.record/messages');

  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
    Kullanici.instance.testIcinUnut();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(kayitKanali, (MethodCall c) async => null);
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(kayitKanali, null);
  });

  // Dort secenek 800 piksellik test ekranina sigmiyor; gercek telefonda
  // oldugu gibi once kaydirip sonra dokunuyoruz.
  Future<void> dokun(WidgetTester tester, String yazi) async {
    await tester.ensureVisible(find.text(yazi));
    await tester.pumpAndSettle();
    await tester.tap(find.text(yazi));
    await tester.pumpAndSettle();
  }

  Widget uygulama() => MaterialApp(
        theme: buildHatirlaTheme(),
        home: const KimSinizScreen(karsilamaTamam: false),
      );

  testWidgets('dort akraba da ekranda ve doğrudan seçilebiliyor',
      (WidgetTester tester) async {
    await tester.pumpWidget(uygulama());
    await tester.pumpAndSettle();

    expect(find.text('Hangi akrabamla konuşuyorum?'), findsOneWidget);
    for (final Akraba a in Akraba.values) {
      expect(find.text(a.ad), findsOneWidget);
    }
  });

  testWidgets('onay verilince seçim kaydediliyor', (WidgetTester tester) async {
    await tester.pumpWidget(uygulama());
    await tester.pumpAndSettle();

    await dokun(tester, 'Babannem');
    expect(find.text('Siz Babannem misiniz?'), findsOneWidget);

    await dokun(tester, 'Evet, benim');

    expect(Kullanici.instance.akraba, Akraba.babaanne);
  });

  // Yanlis dokunus kimligi kilitlememeli: onay penceresi kapanip ekran
  // oldugu gibi kalmali.
  testWidgets('vazgeçilince hiçbir şey kaydedilmiyor',
      (WidgetTester tester) async {
    await tester.pumpWidget(uygulama());
    await tester.pumpAndSettle();

    await dokun(tester, 'Oktay Dedem');
    await dokun(tester, 'Hayır, değilim');

    expect(Kullanici.instance.secildi, isFalse);
    expect(find.text('Hangi akrabamla konuşuyorum?'), findsOneWidget);
  });
}
