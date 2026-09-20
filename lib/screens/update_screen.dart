import 'dart:async';

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
/// Ekranin tonu bilerek sakin: "yeni bir sey geldi", "hatiralariniza bir
/// sey olmaz". Guncelleme kelimesi bile cogu yasli kullanici icin
/// "bir seyler bozulacak" demek.
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

  Future<void> _guncelle() async {
    setState(() => _basiliyor = true);
    try {
      await Guncelleyici.instance.kur();
    } finally {
      if (mounted) setState(() => _basiliyor = false);
    }
  }

  /// Ekrandan cikildiginda ertelemeyi kaydeder.
  ///
  /// Tek yol var: ekran kapanir, [PopScope] geri cagirmasi ertelemeyi
  /// yazar. Once erteleyip sonra kapatmak (ya da tersi) iki kod yolu
  /// demek olurdu ve ikisi birbirini tetikleyebilirdi.
  void _sonra() => Navigator.of(context).pop();

  @override
  Widget build(BuildContext context) {
    return PopScope(
      // Geri tusu "Sonra" demekle ayni sey: ekran kullaniciyi hapsetmez,
      // ama nasil cikilirsa cikilsin erteleme kaydedilir.
      //
      // canPop **true** olmali: false olsaydi maybePop() bu geri cagirmayi
      // yeniden tetikler, o da tekrar kapatmayi denerdi - sonsuz dongu.
      canPop: true,
      onPopInvokedWithResult: (bool ciktiMi, Object? _) {
        if (ciktiMi) unawaited(Guncelleyici.instance.ertele());
      },
      child: Scaffold(
        body: SafeArea(
          child: ListenableBuilder(
            listenable: Guncelleyici.instance,
            builder: (BuildContext context, _) {
              final Guncelleyici g = Guncelleyici.instance;
              return Padding(
                padding: const EdgeInsets.all(HatirlaSizes.gutter),
                child: Column(
                  children: <Widget>[
                    Expanded(child: SingleChildScrollView(child: _govde(g))),
                    const SizedBox(height: 16),
                    _butonlar(g),
                  ],
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _govde(Guncelleyici g) {
    if (g.imzaUyusmazligi) return const _AileyeDanis();
    if (!g.kurulumIzniVar) return const _IzinAnlatimi();
    if (g.asama == GuncellemeAsamasi.hata) return _Aksadi(hata: g.hata);
    return _Hazir(g: g);
  }

  Widget _butonlar(Guncelleyici g) {
    // Imza uyusmazliginda kurma butonu hic gosterilmez: bir daha denemek
    // ayni duvara carpar, kullaniciyi bosuna ugrastirir.
    if (g.imzaUyusmazligi) {
      return CerceveliButon(
        yazi: 'Kapat',
        ikon: Icons.arrow_back_rounded,
        onPressed: _sonra,
      );
    }

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
            yazi: 'Sonra',
            ikon: Icons.schedule_rounded,
            onPressed: _sonra,
          ),
        ],
      );
    }

    final bool bekleniyor =
        _basiliyor || g.asama == GuncellemeAsamasi.kuruluyor;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        BuyukButon(
          yazi: bekleniyor ? 'Kuruluyor…' : 'Güncelle',
          altYazi: bekleniyor ? null : 'Birkaç saniye sürer',
          ikon: Icons.download_done_rounded,
          renk: HatirlaColors.confirm,
          yukseklik: 96,
          onPressed: bekleniyor ? null : _guncelle,
        ),
        const SizedBox(height: 12),
        CerceveliButon(
          yazi: 'Sonra',
          ikon: Icons.schedule_rounded,
          onPressed: bekleniyor ? null : _sonra,
        ),
      ],
    );
  }
}

/// Normal durum: her sey hazir.
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
class _Aksadi extends StatelessWidget {
  const _Aksadi({required this.hata});

  final String? hata;

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
