import 'dart:io';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';

import '../models/memory.dart';
import '../services/permissions.dart';
import '../services/store.dart';
import '../services/updater.dart';
import '../services/uploader.dart';
import '../services/whisper_model_manager.dart';
import '../theme.dart';
import '../utils/format.dart';
import '../widgets/common.dart';
import 'help_screen.dart';
import 'update_screen.dart';

/// Ayarlar. Bilerek kisa: kullanicinin buraya girmek zorunda kalmamasi
/// hedef, ekran daha cok kuran kisi icin.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  int? _kullanilanYer;
  bool _paylasiliyor = false;

  @override
  void initState() {
    super.initState();
    _yerHesapla();
  }

  Future<void> _yerHesapla() async {
    int toplam = 0;
    try {
      final Directory dir = MemoryStore.instance.memoriesDir;
      if (dir.existsSync()) {
        await for (final FileSystemEntity e
            in dir.list(recursive: true, followLinks: false)) {
          if (e is File) toplam += await e.length();
        }
      }
    } catch (_) {
      // Yer hesabi bilgi amacli; hata olursa gostermeyiz.
    }
    if (mounted) setState(() => _kullanilanYer = toplam);
  }

  Future<void> _tumYazilariPaylas() async {
    final List<Memory> hatiralar = MemoryStore.instance.memories;
    if (hatiralar.isEmpty) {
      await bilgiGoster(
        context,
        baslik: 'Henüz hatıra yok',
        mesaj: 'Paylaşılacak bir hatıra bulunmuyor.',
      );
      return;
    }

    setState(() => _paylasiliyor = true);
    try {
      final StringBuffer sb = StringBuffer()
        ..writeln('HATIRA DEFTERİ')
        ..writeln('${hatiralar.length} hatıra')
        ..writeln('');

      // Eskiden yeniye: defter gibi okunsun.
      for (final Memory m in hatiralar.reversed) {
        sb
          ..writeln('────────────────────')
          ..writeln(m.title)
          ..writeln(Bicim.uzunTarih(m.createdAt));
        if (m.question != null) sb.writeln('Soru: ${m.question}');
        sb.writeln('');
        sb.writeln(
          m.hasTranscript ? m.transcript : '(Bu hatıranın yazısı yok.)',
        );
        sb.writeln('');
      }
      sb.writeln('— hatırlaf uygulamasıyla hazırlandı');

      final Directory gecici = Directory.systemTemp;
      final File dosya = File('${gecici.path}/hatira_defteri.txt');
      await dosya.writeAsString(sb.toString());

      await SharePlus.instance.share(
        ShareParams(
          files: <XFile>[XFile(dosya.path)],
          subject: 'Hatıra Defteri',
          text: 'Anlatılan hatıraların yazıya dökülmüş hâli.',
        ),
      );
    } catch (e) {
      if (!mounted) return;
      kisaMesaj(context, 'Paylaşılamadı. Tekrar deneyin.');
    } finally {
      if (mounted) setState(() => _paylasiliyor = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const UstCubuk(baslik: 'Ayarlar'),
      body: SafeArea(
        top: false,
        child: ListView(
          padding: const EdgeInsets.fromLTRB(
              HatirlaSizes.gutter, 8, HatirlaSizes.gutter, 32),
          children: <Widget>[
            Grup(
              satirlar: <Widget>[
                GrupSatiri(
                  baslik: 'Nasıl Kullanılır?',
                  ikon: CupertinoIcons.book_fill,
                  ikonRengi: const Color(0xFF0A7AFF),
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(builder: (_) => const HelpScreen()),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 32),

            const BolumBasligi('Yazıya Çevirme'),
            const SizedBox(height: 12),
            const _ModelBolumu(),
            const SizedBox(height: 32),

            ListenableBuilder(
              listenable: MemoryStore.instance,
              builder: (BuildContext context, _) => Grup(
                baslik: 'Hatıra Defteri',
                satirlar: <Widget>[
                  GrupSatiri(
                    baslik: 'Hatıra sayısı',
                    ikon: CupertinoIcons.book_fill,
                    ikonRengi: HatirlaColors.primary,
                    deger: '${MemoryStore.instance.memories.length}',
                  ),
                  GrupSatiri(
                    baslik: 'Kapladığı yer',
                    ikon: CupertinoIcons.archivebox_fill,
                    ikonRengi: const Color(0xFF8E8E93),
                    deger: _kullanilanYer == null
                        ? 'hesaplanıyor…'
                        : Bicim.boyut(_kullanilanYer!),
                  ),
                  EylemSatiri(
                    yazi: _paylasiliyor ? 'Hazırlanıyor…' : 'Tüm Yazıları Paylaş',
                    ikon: CupertinoIcons.square_arrow_up,
                    onTap: _paylasiliyor ? null : _tumYazilariPaylas,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),

            Grup(
              baslik: 'İzinler',
              dipnot: 'Mikrofon ve kamera izinlerini buradan açıp '
                  'kapatabilirsiniz.',
              satirlar: <Widget>[
                GrupSatiri(
                  baslik: 'Telefon Ayarlarını Aç',
                  ikon: CupertinoIcons.gear_alt_fill,
                  ikonRengi: const Color(0xFF8E8E93),
                  onTap: () => Izinler.ayarlariAc(),
                ),
              ],
            ),
            const SizedBox(height: 32),

            const _GuncellemeBolumu(),
            const SizedBox(height: 32),

            if (Yedekleyici.instance.acikMi) ...<Widget>[
              const _YedekBolumu(),
              const SizedBox(height: 32),
            ],

            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Text(
                // Yedekleme acikken "disari cikmaz" demek dogru degil.
                // Bu ekran dogruyu soylemek zorunda: kayitlar gercekten
                // telefondan cikiyor.
                Yedekleyici.instance.acikMi
                    ? 'Kayıtlarınız şifrelenerek yalnızca uygulamayı '
                        'kuran aile üyenize ulaşır. Başka kimse açamaz.'
                    : 'Sesiniz telefonunuzdan dışarı çıkmaz.',
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 19, height: 1.5, color: HatirlaColors.inkSoft),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Yaziya cevirme paketinin indirilmesi. Tek model var, secenek yok.
class _ModelBolumu extends StatelessWidget {
  const _ModelBolumu();

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: WhisperModelManager.instance,
      builder: (BuildContext context, _) {
        final WhisperModelManager mm = WhisperModelManager.instance;
        final bool hazir = mm.durum == IndirmeDurumu.hazir;
        final bool iniyor = mm.durum == IndirmeDurumu.iniyor;
        final Color renk = hazir ? HatirlaColors.confirm : HatirlaColors.warning;

        return Kart(
          padding: const EdgeInsets.all(18),
          child: AnimatedSize(
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOutCubic,
            alignment: Alignment.topCenter,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: <Widget>[
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    IkonRozeti(
                      ikon: hazir
                          ? CupertinoIcons.checkmark_alt
                          : CupertinoIcons.cloud_download_fill,
                      renk: renk,
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Text(
                        hazir
                            ? 'Paket telefonda. Konuşmalar internetsiz olarak '
                                'yazıya çevriliyor.'
                            : 'Paket indirilmedi. İndirilene kadar ses kayıtları '
                                'yazıya çevrilmez.',
                        style: const TextStyle(fontSize: 20, height: 1.4),
                      ),
                    ),
                  ],
                ),
                if (iniyor) ...<Widget>[
                  const SizedBox(height: 18),
                  IlerlemeCubugu(deger: mm.ilerleme),
                  const SizedBox(height: 10),
                  Text(
                    '%${((mm.ilerleme ?? 0) * 100).toStringAsFixed(0)}  ·  '
                    '${Bicim.boyut(mm.inenBayt)} / ${Bicim.boyut(mm.toplamBayt)}',
                    style: const TextStyle(
                      fontSize: 20,
                      height: 1.4,
                      fontWeight: FontWeight.w600,
                      fontFeatures: <FontFeature>[FontFeature.tabularFigures()],
                    ),
                  ),
                  const SizedBox(height: 14),
                  IkincilButon(
                    yazi: 'İndirmeyi Durdur',
                    ikon: CupertinoIcons.stop_fill,
                    renk: HatirlaColors.record,
                    onPressed: mm.iptalEt,
                  ),
                ] else if (!hazir) ...<Widget>[
                  if (mm.hata != null) ...<Widget>[
                    const SizedBox(height: 14),
                    Text(
                      mm.hata!,
                      style: const TextStyle(
                          fontSize: 20, height: 1.4, color: HatirlaColors.record),
                    ),
                  ],
                  const SizedBox(height: 16),
                  BuyukButon(
                    yazi: 'Paketi İndir',
                    altYazi: 'Yaklaşık ${WhisperModelManager.yaklasikMb} MB',
                    ikon: CupertinoIcons.cloud_download_fill,
                    onPressed: () => mm.download(),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

/// Surum ve guncelleme durumu; kuran kisi icin. "Neden guncellenmedi?"
/// sorusunun cevabi telefonla degil buradan okunur.
class _GuncellemeBolumu extends StatefulWidget {
  const _GuncellemeBolumu();

  @override
  State<_GuncellemeBolumu> createState() => _GuncellemeBolumuState();
}

class _GuncellemeBolumuState extends State<_GuncellemeBolumu> {
  bool _denetleniyor = false;

  Future<void> _denetle() async {
    setState(() => _denetleniyor = true);
    try {
      await Guncelleyici.instance.degerlendir(elle: true);
      if (!mounted) return;
      final Guncelleyici g = Guncelleyici.instance;
      if (g.asama == GuncellemeAsamasi.hazir) {
        await Navigator.of(context).push(
          MaterialPageRoute<void>(builder: (_) => const UpdateScreen()),
        );
      } else if (g.asama == GuncellemeAsamasi.bos && g.hata == null) {
        if (!mounted) return;
        kisaMesaj(context, 'En son sürümü kullanıyorsunuz.');
      }
    } finally {
      if (mounted) setState(() => _denetleniyor = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Guncelleyici.instance,
      builder: (BuildContext context, _) {
        final Guncelleyici g = Guncelleyici.instance;
        final bool iniyor = g.asama == GuncellemeAsamasi.indiriliyor;
        final bool hazir = g.asama == GuncellemeAsamasi.hazir;
        final bool yeniVar =
            g.bilgi != null && g.bilgi!.surumKodu > g.mevcutSurumKodu;

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Grup(
              baslik: 'Uygulama Sürümü',
              dipnot: 'Güncellemeler kendiliğinden iner (yaklaşık 22 MB). '
                  'Hazır olduğunda bir kez sorulur.',
              satirlar: <Widget>[
                GrupSatiri(
                  baslik: 'Kurulu sürüm',
                  deger: g.mevcutSurumAdi.isEmpty
                      ? '—'
                      : '${g.mevcutSurumAdi} (${g.mevcutSurumKodu})',
                ),
                GrupSatiri(
                  baslik: 'Son denetim',
                  deger: _denetimZamani(g.sonDenetim),
                ),
                if (yeniVar)
                  GrupSatiri(baslik: 'Yeni sürüm', deger: g.bilgi!.surumAdi),
                if (yeniVar)
                  GrupSatiri(
                    baslik: 'İndirilecek',
                    deger: '${Bicim.boyut(g.bilgi!.boyut)} (${g.bilgi!.abi})',
                  ),
                if (iniyor)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(18, 16, 18, 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: <Widget>[
                        IlerlemeCubugu(deger: g.ilerleme),
                        const SizedBox(height: 10),
                        Text(
                          'İniyor: ${Bicim.boyut(g.inenBayt)} / '
                          '${Bicim.boyut(g.toplamBayt)}',
                          style: const TextStyle(fontSize: 20, height: 1.4),
                        ),
                      ],
                    ),
                  ),
                if (iniyor)
                  EylemSatiri(
                    yazi: 'İndirmeyi Durdur',
                    ikon: CupertinoIcons.stop_fill,
                    renk: HatirlaColors.record,
                    onTap: Guncelleyici.instance.indirmeyiDurdur,
                  )
                else if (!hazir)
                  EylemSatiri(
                    yazi: _denetleniyor ? 'Bakılıyor…' : 'Güncelleme Var mı?',
                    ikon: CupertinoIcons.arrow_clockwise,
                    onTap: _denetleniyor ? null : _denetle,
                  ),
              ],
            ),
            if (hazir) ...<Widget>[
              const SizedBox(height: 14),
              BuyukButon(
                yazi: 'Şimdi Güncelle',
                altYazi: 'Yeni sürüm indirildi, kurulmayı bekliyor',
                ikon: CupertinoIcons.arrow_down_circle_fill,
                renk: HatirlaColors.confirm,
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const UpdateScreen()),
                ),
              ),
            ],
            if (g.hata != null) ...<Widget>[
              const SizedBox(height: 14),
              Kart(
                renk: HatirlaColors.warningSoft,
                golge: false,
                padding: const EdgeInsets.all(16),
                child: Text(
                  g.hata!,
                  style: const TextStyle(fontSize: 19, height: 1.4),
                ),
              ),
            ],
          ],
        );
      },
    );
  }

  static String _denetimZamani(DateTime? t) {
    if (t == null) return 'hiç';
    final Duration gecen = DateTime.now().difference(t);
    if (gecen.isNegative) return Bicim.gunlukTarih(t);
    if (gecen.inMinutes < 1) return 'az önce';
    if (gecen.inHours < 1) return '${gecen.inMinutes} dakika önce';
    if (gecen.inHours < 24) return '${gecen.inHours} saat önce';
    return '${Bicim.gunlukTarih(t)}, ${Bicim.saat(t)}';
  }
}

/// Yedekleme durumu; kuran kisi icin. Kullaniciya bir sey sorulmaz,
/// burasi yalnizca "neden yuklenmedi?" sorusunun cevabi.
class _YedekBolumu extends StatelessWidget {
  const _YedekBolumu();

  static String _asamaMetni(Yedekleyici y) {
    switch (y.asama) {
      case YedekAsamasi.bos:
        return y.bekleyenSayisi == 0 ? 'Hepsi gönderildi' : 'Sırada';
      case YedekAsamasi.sifreliyor:
        return 'Şifreleniyor';
      case YedekAsamasi.yukleniyor:
        final double? i = y.ilerleme;
        return i == null ? 'Gönderiliyor' : 'Gönderiliyor %${(i * 100).round()}';
      case YedekAsamasi.bekliyor:
        return 'Wi-Fi bekleniyor';
      case YedekAsamasi.hata:
        return 'Aksadı';
    }
  }

  /// Kuran kisi telefonu teslim ederken bir kez doldurur. Kayitlar
  /// kovaya bu adla gider; yoksa elinizde 30 tane rastgele kimlik olur
  /// ve hangisinin kim oldugunu bilemezsiniz.
  static Future<void> _adiSor(BuildContext context, String mevcut) async {
    final TextEditingController kutu = TextEditingController(text: mevcut);
    final String? sonuc = await showDialog<String>(
      context: context,
      builder: (BuildContext c) => AlertDialog(
        title: const Text('Bu telefon kimin?'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            const Text(
              'Kayıtların hangi telefondan geldiğini ayırt etmek için. '
              'Örnek: Dedem Ahmet',
              style: TextStyle(fontSize: 17, color: HatirlaColors.inkSoft),
            ),
            const SizedBox(height: 14),
            TextField(
              controller: kutu,
              autofocus: true,
              textCapitalization: TextCapitalization.words,
              style: const TextStyle(fontSize: 21),
              decoration: const InputDecoration(fillColor: HatirlaColors.paper),
              onSubmitted: (String v) => Navigator.of(c).pop(v),
            ),
          ],
        ),
        actions: <Widget>[
          TextButton(
            onPressed: () => Navigator.of(c).pop(),
            child: const Text('Vazgeç', style: TextStyle(fontSize: 19)),
          ),
          TextButton(
            onPressed: () => Navigator.of(c).pop(kutu.text),
            child: const Text('Kaydet', style: TextStyle(fontSize: 19)),
          ),
        ],
      ),
    );
    kutu.dispose();
    if (sonuc != null) await Yedekleyici.instance.sahibiKaydet(sonuc);
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Yedekleyici.instance,
      builder: (BuildContext context, _) {
        final Yedekleyici y = Yedekleyici.instance;
        return Grup(
          baslik: 'Aile Yedeği',
          dipnot: y.hata,
          satirlar: <Widget>[
            GrupSatiri(baslik: 'Durum', deger: _asamaMetni(y)),
            GrupSatiri(baslik: 'Gönderilen', deger: '${y.yuklenenSayisi}'),
            GrupSatiri(baslik: 'Bekleyen', deger: '${y.bekleyenSayisi}'),
            GrupSatiri(
              baslik: 'Telefon',
              deger: y.sahip.isEmpty ? '— (adsız)' : y.sahip,
            ),
            GrupSatiri(baslik: 'Cihaz', deger: y.cihaz.isEmpty ? '—' : y.cihaz),
            EylemSatiri(
              yazi: y.sahip.isEmpty ? 'Telefonu Adlandır' : 'Adı Değiştir',
              ikon: CupertinoIcons.person_crop_circle,
              onTap: () => _adiSor(context, y.sahip),
            ),
            EylemSatiri(
              yazi: 'Şimdi Gönder',
              ikon: CupertinoIcons.cloud_upload,
              onTap: () => Yedekleyici.instance.tekrarDene(),
            ),
          ],
        );
      },
    );
  }
}
