import 'dart:io';

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
      appBar: AppBar(title: const Text('Ayarlar')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(HatirlaSizes.gutter),
          children: <Widget>[
            const BolumBasligi('Yardım', ikon: Icons.help_outline_rounded),
            const SizedBox(height: 14),
            CerceveliButon(
              yazi: 'Nasıl Kullanılır?',
              ikon: Icons.menu_book_rounded,
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => const HelpScreen()),
              ),
            ),
            const SizedBox(height: 30),

            const BolumBasligi('Yazıya Çevirme', ikon: Icons.edit_note_rounded),
            const SizedBox(height: 14),
            const _ModelBolumu(),
            const SizedBox(height: 30),

            const BolumBasligi('Hatıra Defteri',
                ikon: Icons.auto_stories_rounded),
            const SizedBox(height: 14),
            ListenableBuilder(
              listenable: MemoryStore.instance,
              builder: (BuildContext context, _) => _BilgiKutusu(
                satirlar: <(String, String)>[
                  ('Hatıra sayısı', '${MemoryStore.instance.memories.length}'),
                  (
                    'Kapladığı yer',
                    _kullanilanYer == null
                        ? 'hesaplanıyor…'
                        : Bicim.boyut(_kullanilanYer!)
                  ),
                ],
              ),
            ),
            const SizedBox(height: 14),
            CerceveliButon(
              yazi: _paylasiliyor ? 'Hazırlanıyor…' : 'Tüm Yazıları Paylaş',
              ikon: Icons.ios_share_rounded,
              onPressed: _paylasiliyor ? null : _tumYazilariPaylas,
            ),
            const SizedBox(height: 30),

            const BolumBasligi('İzinler', ikon: Icons.lock_open_rounded),
            const SizedBox(height: 14),
            CerceveliButon(
              yazi: 'Telefon Ayarlarını Aç',
              ikon: Icons.settings_rounded,
              onPressed: () => Izinler.ayarlariAc(),
            ),
            const SizedBox(height: 10),
            const Text(
              'Mikrofon ve kamera izinlerini buradan açıp kapatabilirsiniz.',
              style: TextStyle(fontSize: 19, color: HatirlaColors.inkSoft),
            ),
            const SizedBox(height: 30),

            const BolumBasligi('Uygulama Sürümü',
                ikon: Icons.system_update_rounded),
            const SizedBox(height: 14),
            const _GuncellemeBolumu(),
            const SizedBox(height: 34),

            if (Yedekleyici.instance.acikMi) ...<Widget>[
              const BolumBasligi('Aile Yedeği',
                  ikon: Icons.lock_outline_rounded),
              const SizedBox(height: 14),
              const _YedekBolumu(),
              const SizedBox(height: 34),
            ],

            Center(
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
                    fontSize: 18, height: 1.5, color: HatirlaColors.inkSoft),
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

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: hazir ? const Color(0xFFE6F2E8) : const Color(0xFFFFF4DB),
                borderRadius: BorderRadius.circular(18),
                border: Border.all(
                  color: hazir
                      ? HatirlaColors.confirm
                      : const Color(0xFFD9A400),
                  width: 2,
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Icon(
                    hazir
                        ? Icons.check_circle_rounded
                        : Icons.cloud_download_rounded,
                    size: 32,
                    color: hazir
                        ? HatirlaColors.confirm
                        : const Color(0xFF8A6800),
                  ),
                  const SizedBox(width: 12),
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
            ),
            if (iniyor) ...<Widget>[
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: LinearProgressIndicator(
                  value: mm.ilerleme,
                  minHeight: 18,
                  backgroundColor: HatirlaColors.paperDark,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                '%${((mm.ilerleme ?? 0) * 100).toStringAsFixed(0)}  •  '
                '${Bicim.boyut(mm.inenBayt)} / ${Bicim.boyut(mm.toplamBayt)}',
                style: const TextStyle(fontSize: 21, height: 1.4),
              ),
              const SizedBox(height: 12),
              CerceveliButon(
                yazi: 'İndirmeyi Durdur',
                ikon: Icons.stop_rounded,
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
              const SizedBox(height: 14),
              BuyukButon(
                yazi: 'Paketi İndir',
                altYazi: 'Yaklaşık ${WhisperModelManager.yaklasikMb} MB',
                ikon: Icons.cloud_download_rounded,
                onPressed: () => mm.download(),
              ),
            ],
          ],
        );
      },
    );
  }
}

class _BilgiKutusu extends StatelessWidget {
  const _BilgiKutusu({required this.satirlar});

  final List<(String, String)> satirlar;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 6),
      decoration: BoxDecoration(
        color: HatirlaColors.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: HatirlaColors.line, width: 2),
      ),
      child: Column(
        children: <Widget>[
          for (final (String ad, String deger) satir in satirlar)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 14),
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: Text(satir.$1,
                        style: const TextStyle(fontSize: 21)),
                  ),
                  Text(
                    satir.$2,
                    style: const TextStyle(
                        fontSize: 21, fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ),
        ],
      ),
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

        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _BilgiKutusu(
              satirlar: <(String, String)>[
                (
                  'Kurulu sürüm',
                  g.mevcutSurumAdi.isEmpty
                      ? '—'
                      : '${g.mevcutSurumAdi} (${g.mevcutSurumKodu})'
                ),
                ('Son denetim', _denetimZamani(g.sonDenetim)),
                if (g.bilgi != null && g.bilgi!.surumKodu > g.mevcutSurumKodu)
                  ('Yeni sürüm', g.bilgi!.surumAdi),
                if (g.bilgi != null && g.bilgi!.surumKodu > g.mevcutSurumKodu)
                  ('İndirilecek', '${Bicim.boyut(g.bilgi!.boyut)}'
                      ' (${g.bilgi!.abi})'),
              ],
            ),
            if (iniyor) ...<Widget>[
              const SizedBox(height: 16),
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: LinearProgressIndicator(
                  value: g.ilerleme,
                  minHeight: 18,
                  backgroundColor: HatirlaColors.paperDark,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'İniyor: ${Bicim.boyut(g.inenBayt)} / '
                '${Bicim.boyut(g.toplamBayt)}',
                style: const TextStyle(fontSize: 20, height: 1.4),
              ),
              const SizedBox(height: 12),
              CerceveliButon(
                yazi: 'İndirmeyi Durdur',
                ikon: Icons.stop_rounded,
                renk: HatirlaColors.record,
                onPressed: Guncelleyici.instance.indirmeyiDurdur,
              ),
            ] else if (hazir) ...<Widget>[
              const SizedBox(height: 14),
              BuyukButon(
                yazi: 'Şimdi Güncelle',
                altYazi: 'Yeni sürüm indirildi, kurulmayı bekliyor',
                ikon: Icons.download_done_rounded,
                renk: HatirlaColors.confirm,
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(builder: (_) => const UpdateScreen()),
                ),
              ),
            ] else ...<Widget>[
              const SizedBox(height: 14),
              CerceveliButon(
                yazi: _denetleniyor ? 'Bakılıyor…' : 'Güncelleme Var mı?',
                ikon: Icons.refresh_rounded,
                onPressed: _denetleniyor ? null : _denetle,
              ),
            ],
            if (g.hata != null) ...<Widget>[
              const SizedBox(height: 14),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFF4DB),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFFD9A400), width: 2),
                ),
                child: Text(
                  g.hata!,
                  style: const TextStyle(fontSize: 19, height: 1.4),
                ),
              ),
            ],
            const SizedBox(height: 12),
            const Text(
              'Güncellemeler kendiliğinden iner (yaklaşık 22 MB). '
              'Hazır olduğunda bir kez sorulur.',
              style: TextStyle(fontSize: 19, color: HatirlaColors.inkSoft),
            ),
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
              decoration: const InputDecoration(border: OutlineInputBorder()),
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
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            _BilgiKutusu(
              satirlar: <(String, String)>[
                ('Durum', _asamaMetni(y)),
                ('Gönderilen', '${y.yuklenenSayisi}'),
                ('Bekleyen', '${y.bekleyenSayisi}'),
                ('Telefon', y.sahip.isEmpty ? '— (adsız)' : y.sahip),
                ('Cihaz', y.cihaz.isEmpty ? '—' : y.cihaz),
              ],
            ),
            const SizedBox(height: 14),
            CerceveliButon(
              yazi: y.sahip.isEmpty ? 'Telefonu Adlandır' : 'Adı Değiştir',
              ikon: Icons.badge_outlined,
              onPressed: () => _adiSor(context, y.sahip),
            ),
            if (y.hata != null) ...<Widget>[
              const SizedBox(height: 14),
              Text(
                y.hata!,
                style: const TextStyle(
                    fontSize: 17, color: HatirlaColors.inkSoft),
              ),
            ],
            const SizedBox(height: 14),
            CerceveliButon(
              yazi: 'Şimdi Gönder',
              ikon: Icons.cloud_upload_outlined,
              onPressed: () => Yedekleyici.instance.tekrarDene(),
            ),
          ],
        );
      },
    );
  }
}
