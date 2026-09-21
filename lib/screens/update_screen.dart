import 'package:flutter/material.dart';

import '../services/updater.dart';
import '../theme.dart';
import '../widgets/common.dart';

/// Yasli kullanicinin guncellemeyle ilgili gordugu **tek** ekran.
///
/// Buraya yalnizca her sey hazirken gelinir: yeni surum inmis, ozeti
/// dogrulanmis, kurulmayi bekliyor. Yani burada beklenecek, iptal
/// edilecek, yarida kalacak bir sey yok - bir dokunus, bir de sistemin
/// onay penceresi.
///
/// ## Guncelleme zorunludur
///
/// "Sonra" yok, geri tusu yok: kurulabilir bir guncelleme varken
/// uygulamaya devam edilemez. Sebep, herkesin ayni surumde olmasinin
/// destegi mumkun kilmasi - 20-30 telefonun farkli surumlere dagilmasi,
/// her sorunda "sende ne yaziyor?" diye telefonda konusmak demek.
///
/// ## Ama asla kilitlemez
///
/// Zorunluluk yalnizca **kurulumun mumkun oldugu** durumda gecerli.
/// Imza uyusmazligi, yer yoklugu, Play Protect engeli gibi kullanicinin
/// cozemeyecegi bir hata varsa ekran kapatilabilir. Aksi halde yasli
/// kullanici, cozemeyecegi bir hata yuzunden **kendi hatiralarina
/// erisemez** hale gelirdi; bu, eski surumde kalmaktan cok daha kotu.
/// Ekran her acilista yeniden gelir, yani israr surer - ama kapi acik.
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
    // Kullanici izin ekranindan donmus olabilir; izni yeniden soralim ki
    // "İzin Ver" butonu bos yere ekranda kalmasin.
    if (durum == AppLifecycleState.resumed) {
      Guncelleyici.instance.izniTazele();
    }
  }

  /// Kullanici bu ekrandan cikabilir mi?
  ///
  /// Yalnizca kurulumun onunde kullanicinin cozemeyecegi bir engel varsa.
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
          // Kurulabilir bir guncelleme varken geri tusu calismaz.
          //
          // canPop false iken geri cagirmadan maybePop() CAGIRMAYIN:
          // geri cagirmayi yeniden tetikler ve sonsuz donguye girer.
          canPop: _kacisVar(g),
          child: Scaffold(
            body: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(HatirlaSizes.gutter),
                child: Column(
                  children: <Widget>[
                    Expanded(child: SingleChildScrollView(child: _govde(g))),
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
    // Imza uyusmazliginda kurma butonu hic gosterilmez: bir daha denemek
    // ayni duvara carpar, kullaniciyi bosuna ugrastirir.
    if (g.imzaUyusmazligi) {
      return CerceveliButon(
        yazi: 'Kapat',
        ikon: Icons.arrow_back_rounded,
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
            ikon: Icons.lock_open_rounded,
            onPressed: () => Guncelleyici.instance.kurulumIzniIste(),
          ),
          const SizedBox(height: 12),
          CerceveliButon(
            yazi: 'Kapat',
            ikon: Icons.arrow_back_rounded,
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
            ikon: Icons.refresh_rounded,
            onPressed: bekleniyor ? null : _tekrarDene,
          ),
          const SizedBox(height: 12),
          CerceveliButon(
            yazi: 'Kapat',
            ikon: Icons.arrow_back_rounded,
            onPressed: bekleniyor ? null : _kapat,
          ),
        ],
      );
    }

    // Asil durum: kurulabilir bir guncelleme var. Tek buton, cikis yok.
    return BuyukButon(
      yazi: bekleniyor ? 'Kuruluyor…' : 'Güncelle',
      altYazi: bekleniyor ? null : 'Birkaç saniye sürer',
      ikon: Icons.download_done_rounded,
      renk: HatirlaColors.confirm,
      yukseklik: 96,
      onPressed: bekleniyor ? null : _guncelle,
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
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const SizedBox(height: 24),
        const Icon(Icons.auto_awesome_rounded,
            size: 96, color: HatirlaColors.primary),
        const SizedBox(height: 24),
        Text(
          'Uygulamanın yeni hâli hazır',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 18),
        Text(
          'İndirildi, kurulmayı bekliyor.\n'
          'Hatıralarınıza hiçbir şey olmaz.',
          textAlign: TextAlign.center,
          style: Theme.of(context)
              .textTheme
              .bodyLarge
              ?.copyWith(color: HatirlaColors.inkSoft),
        ),

        // Sistemin penceresinde "Vazgeç" denmisse sebebini soyleyelim;
        // yoksa kullanici ayni ekrana hicbir aciklama olmadan doner.
        if (g.iptalEdildi) ...<Widget>[
          const SizedBox(height: 22),
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: const Color(0xFFFFF4DB),
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: const Color(0xFFD9A400), width: 2),
            ),
            child: const Text(
              'Kurulum tamamlanmadı. Devam edebilmek için '
              '“Güncelle”ye dokunup açılan pencerede onay verin.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 20, height: 1.4),
            ),
          ),
        ],

        if (notlar != null && notlar.isNotEmpty) ...<Widget>[
          const SizedBox(height: 26),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: HatirlaColors.primarySoft,
              borderRadius: BorderRadius.circular(HatirlaSizes.radius),
              border: Border.all(color: HatirlaColors.primary, width: 2),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  'Neler değişti',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: HatirlaColors.primaryDark,
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
          style: const TextStyle(fontSize: 18, color: HatirlaColors.inkSoft),
        ),
      ],
    );
  }
}

/// "Bilinmeyen kaynak" izni verilmemis.
///
/// Bu izin normalde telefonu teslim ederken bir kez acilir. Yine de
/// kapaliysa kullaniciyi sucla yuzlestirmeden, tek cumleyle anlatiyoruz.
class _IzinAnlatimi extends StatelessWidget {
  const _IzinAnlatimi();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const SizedBox(height: 24),
        const Icon(Icons.lock_outline_rounded,
            size: 96, color: HatirlaColors.primary),
        const SizedBox(height: 24),
        Text(
          'Telefonun izni gerekiyor',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 18),
        Text(
          'Yeni sürümü kurabilmek için telefonun bir kez izin vermesi '
          'gerekiyor.\n\n'
          '“İzin Ver”e dokunun, açılan ekrandaki düğmeyi açın ve geri '
          'dönün. Bunu yalnızca bir kere yapacaksınız.',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyLarge,
        ),
      ],
    );
  }
}

/// Kurulum bir hatayla bitti.
///
/// Burada cikis kapisi aciktir: kullanicinin cozemeyecegi bir hata
/// yuzunden kendi hatiralarina erisemez hale gelmesi, eski surumde
/// kalmasindan cok daha kotu.
class _Aksadi extends StatelessWidget {
  const _Aksadi();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const SizedBox(height: 24),
        const Icon(Icons.info_outline_rounded,
            size: 96, color: HatirlaColors.inkSoft),
        const SizedBox(height: 24),
        Text(
          'Şimdi olmadı',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 18),
        Text(
          'Güncelleme kurulamadı. Hatıralarınıza bir şey olmadı; '
          'uygulamayı eskisi gibi kullanmaya devam edebilirsiniz.\n\n'
          'Birazdan kendiliğinden tekrar denenecek.',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyLarge,
        ),
      ],
    );
  }
}

/// Imza uyusmazligi: kullanicinin cozemeyecegi tek hata.
///
/// Burada **asla** "uygulamayi silip yeniden kurun" denmez. Dogru olan
/// tavsiye bu olsa bile, o islem hatiralari siler ve yasli kullanici
/// uyarıyı okumadan ilerleyebilir.
class _AileyeDanis extends StatelessWidget {
  const _AileyeDanis();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        const SizedBox(height: 24),
        const Icon(Icons.support_agent_rounded,
            size: 96, color: HatirlaColors.primary),
        const SizedBox(height: 24),
        Text(
          'Bu güncelleme kurulamıyor',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        const SizedBox(height: 18),
        Text(
          'Uygulamayı size kuran kişinin yardım etmesi gerekiyor.\n\n'
          'Hatıralarınız yerli yerinde duruyor ve uygulama eskisi gibi '
          'çalışmaya devam ediyor. Acele edilecek bir şey yok.',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodyLarge,
        ),
      ],
    );
  }
}
