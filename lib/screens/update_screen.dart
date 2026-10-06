import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import '../services/updater.dart';
import '../theme.dart';
import '../widgets/common.dart';

/// Kullanicinin guncellemeyle ilgili gordugu tek ekran. Buraya yalnizca
/// her sey hazirken gelinir: bir dokunus ve sistemin onay penceresi.
///
/// Guncelleme zorunlu ("sonra" yok, geri tusu yok) ama asla kilitlemez:
/// kullanicinin cozemeyecegi bir hata varsa ekran kapatilabilir, yoksa
/// kendi hatiralarina erisemez hale gelirdi.
class UpdateScreen extends StatefulWidget {
  const UpdateScreen({super.key});

  @override
  State<UpdateScreen> createState() => _UpdateScreenState();
}

class _UpdateScreenState extends State<UpdateScreen>
    with WidgetsBindingObserver {
  bool _basiliyor = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState durum) {
    // Izin ekranindan donulmus olabilir.
    if (durum == AppLifecycleState.resumed) {
      Guncelleyici.instance.izniTazele();
    }
  }

  /// Yalnizca kullanicinin cozemeyecegi bir engel varsa cikilabilir.
  static bool _kacisVar(Guncelleyici g) =>
      g.imzaUyusmazligi ||
      !g.kurulumIzniVar ||
      g.asama == GuncellemeAsamasi.hata;

  Future<void> _guncelle() async {
    setState(() => _basiliyor = true);
    try {
      await Guncelleyici.instance.kur();
    } finally {
      if (mounted) setState(() => _basiliyor = false);
    }
  }

  Future<void> _tekrarDene() async {
    setState(() => _basiliyor = true);
    try {
      await Guncelleyici.instance.degerlendir(elle: true);
      if (Guncelleyici.instance.asama == GuncellemeAsamasi.hazir) {
        await Guncelleyici.instance.kur();
      }
    } finally {
      if (mounted) setState(() => _basiliyor = false);
    }
  }

  void _kapat() => Navigator.of(context).pop();

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Guncelleyici.instance,
      builder: (BuildContext context, _) {
        final Guncelleyici g = Guncelleyici.instance;
        return PopScope(
          // canPop false iken geri cagirmadan maybePop() CAGIRMAYIN:
          // geri cagirmayi yeniden tetikler, sonsuz donguye girer.
          canPop: _kacisVar(g),
          child: Scaffold(
            body: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(HatirlaSizes.gutter),
                child: Column(
                  children: <Widget>[
                    Expanded(
                      child: SingleChildScrollView(
                        child: AnimatedSwitcher(
                          duration: const Duration(milliseconds: 300),
                          child: _govde(g),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _butonlar(g),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _govde(Guncelleyici g) {
    if (g.imzaUyusmazligi) return const _AileyeDanis();
    if (!g.kurulumIzniVar) return const _IzinAnlatimi();
    if (g.asama == GuncellemeAsamasi.hata) return const _Aksadi();
    return _Hazir(g: g);
  }

  Widget _butonlar(Guncelleyici g) {
    // Imza uyusmazliginda kurma butonu gosterilmez: bir daha denemek
    // ayni duvara carpar.
    if (g.imzaUyusmazligi) {
      return IkincilButon(
        yazi: 'Kapat',
        ikon: CupertinoIcons.xmark,
        renk: HatirlaColors.inkSoft,
        onPressed: _kapat,
      );
    }

    final bool bekleniyor =
        _basiliyor || g.asama == GuncellemeAsamasi.kuruluyor;

    if (!g.kurulumIzniVar) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          BuyukButon(
            yazi: 'İzin Ver',
            altYazi: 'Telefon ayarları açılacak',
            ikon: CupertinoIcons.lock_open_fill,
            onPressed: () => Guncelleyici.instance.kurulumIzniIste(),
          ),
          const SizedBox(height: 12),
          IkincilButon(
            yazi: 'Kapat',
            ikon: CupertinoIcons.xmark,
            renk: HatirlaColors.inkSoft,
            onPressed: _kapat,
          ),
        ],
      );
    }

    if (g.asama == GuncellemeAsamasi.hata) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          BuyukButon(
            yazi: bekleniyor ? 'Deneniyor…' : 'Tekrar Dene',
            ikon: CupertinoIcons.arrow_clockwise,
            onPressed: bekleniyor ? null : _tekrarDene,
          ),
          const SizedBox(height: 12),
          IkincilButon(
            yazi: 'Kapat',
            ikon: CupertinoIcons.xmark,
            renk: HatirlaColors.inkSoft,
            onPressed: bekleniyor ? null : _kapat,
          ),
        ],
      );
    }

    // Kurulabilir guncelleme: tek buton, cikis yok.
    return BuyukButon(
      yazi: bekleniyor ? 'Kuruluyor…' : 'Güncelle',
      altYazi: bekleniyor ? null : 'Birkaç saniye sürer',
      ikon: CupertinoIcons.arrow_down_circle_fill,
      renk: HatirlaColors.confirm,
      yukseklik: 96,
      onPressed: bekleniyor ? null : _guncelle,
    );
  }
}

class _Govde extends StatelessWidget {
  const _Govde({
    required this.ikon,
    required this.renk,
    required this.baslik,
    required this.metin,
    this.ekler = const <Widget>[],
  });

  final IconData ikon;
  final Color renk;
  final String baslik;
  final String metin;
  final List<Widget> ekler;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const SizedBox(height: 36),
        Center(child: BuyukSimge(ikon: ikon, renk: renk)),
        const SizedBox(height: 30),
        Text(
          baslik,
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 14),
        Text(
          metin,
          textAlign: TextAlign.center,
          style: Theme.of(context)
              .textTheme
              .bodyLarge
              ?.copyWith(color: HatirlaColors.inkSoft),
        ),
        ...ekler,
      ],
    );
  }
}

/// Normal durum: her sey hazir, kurulmasi bekleniyor.
class _Hazir extends StatelessWidget {
  const _Hazir({required this.g});

  final Guncelleyici g;

  @override
  Widget build(BuildContext context) {
    final String? notlar = g.bilgi?.notlar.trim();
    return _Govde(
      ikon: CupertinoIcons.sparkles,
      renk: HatirlaColors.primary,
      baslik: 'Uygulamanın yeni hâli hazır',
      metin: 'İndirildi, kurulmayı bekliyor.\n'
          'Hatıralarınıza hiçbir şey olmaz.',
      ekler: <Widget>[
        // "Vazgeç" denmisse sebebini soyleyelim.
        if (g.iptalEdildi) ...<Widget>[
          const SizedBox(height: 22),
          const Kart(
            renk: HatirlaColors.warningSoft,
            golge: false,
            padding: EdgeInsets.all(18),
            child: Text(
              'Kurulum tamamlanmadı. Devam edebilmek için '
              '“Güncelle”ye dokunup açılan pencerede onay verin.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 20, height: 1.4),
            ),
          ),
        ],
        if (notlar != null && notlar.isNotEmpty) ...<Widget>[
          const SizedBox(height: 26),
          Kart(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  'Neler değişti',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: HatirlaColors.primary,
                      ),
                ),
                const SizedBox(height: 10),
                Text(
                  notlar,
                  style: const TextStyle(fontSize: 21, height: 1.45),
                ),
              ],
            ),
          ),
        ],
        const SizedBox(height: 20),
        Text(
          'Şimdiki sürüm ${g.mevcutSurumAdi} → yeni sürüm '
          '${g.bilgi?.surumAdi ?? ''}',
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 19, color: HatirlaColors.inkSoft),
        ),
      ],
    );
  }
}

/// "Bilinmeyen kaynak" izni verilmemis. Normalde telefonu teslim ederken
/// bir kez acilir.
class _IzinAnlatimi extends StatelessWidget {
  const _IzinAnlatimi();

  @override
  Widget build(BuildContext context) {
    return const _Govde(
      ikon: CupertinoIcons.lock_fill,
      renk: HatirlaColors.primary,
      baslik: 'Telefonun izni gerekiyor',
      metin: 'Yeni sürümü kurabilmek için telefonun bir kez izin vermesi '
          'gerekiyor.\n\n'
          '“İzin Ver”e dokunun, açılan ekrandaki düğmeyi açın ve geri '
          'dönün. Bunu yalnızca bir kere yapacaksınız.',
    );
  }
}

/// Kurulum bir hatayla bitti. Cikis kapisi acik: cozemeyecegi bir hata
/// kullaniciyi kendi hatiralarindan etmemeli.
class _Aksadi extends StatelessWidget {
  const _Aksadi();

  @override
  Widget build(BuildContext context) {
    return const _Govde(
      ikon: CupertinoIcons.info,
      renk: Color(0xFF8E8E93),
      baslik: 'Şimdi olmadı',
      metin: 'Güncelleme kurulamadı. Hatıralarınıza bir şey olmadı; '
          'uygulamayı eskisi gibi kullanmaya devam edebilirsiniz.\n\n'
          'Birazdan kendiliğinden tekrar denenecek.',
    );
  }
}

/// Imza uyusmazligi. Burada asla "silip yeniden kurun" denmez: dogru
/// tavsiye o olsa bile hatiralari siler.
class _AileyeDanis extends StatelessWidget {
  const _AileyeDanis();

  @override
  Widget build(BuildContext context) {
    return const _Govde(
      ikon: CupertinoIcons.person_2_fill,
      renk: HatirlaColors.primary,
      baslik: 'Bu güncelleme kurulamıyor',
      metin: 'Uygulamayı size kuran kişinin yardım etmesi gerekiyor.\n\n'
          'Hatıralarınız yerli yerinde duruyor ve uygulama eskisi gibi '
          'çalışmaya devam ediyor. Acele edilecek bir şey yok.',
    );
  }
}
