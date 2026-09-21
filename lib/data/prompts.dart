import 'dart:math';

import 'package:flutter/material.dart';

/// Hatira konusu (soru demeti).
class SoruKonusu {
  const SoruKonusu({
    required this.ad,
    required this.ikon,
    required this.renk,
    required this.sorular,
  });

  final String ad;
  final IconData ikon;
  final Color renk;
  final List<String> sorular;
}

/// Bos bir kayit ekrani karsisinda ne anlatacagini bilmek zor; uygulama
/// soruyu kendisi soruyor.
class Sorular {
  const Sorular._();

  static final List<SoruKonusu> konular = <SoruKonusu>[
    const SoruKonusu(
      ad: 'Çocukluğum',
      ikon: Icons.child_care_rounded,
      renk: Color(0xFFB4651E),
      sorular: <String>[
        'Çocukluğunuz nerede geçti? O yeri bize anlatır mısınız?',
        'Çocukken en çok hangi oyunu oynardınız? Kimlerle oynardınız?',
        'Çocukluğunuzda evinizin kokusunu hatırlıyor musunuz? Neye benzerdi?',
        'İlk okulunuzu ve öğretmeninizi anlatır mısınız?',
        'Çocukken en çok neyden korkardınız?',
        'Size alınan ilk oyuncak neydi?',
        'Yaz tatillerinde neler yapardınız?',
        'Çocukken en büyük hayaliniz neydi?',
        'Hiç yaramazlık yapıp azar işittiniz mi? Ne olmuştu?',
        'Çocukken en sevdiğiniz yemek hangisiydi, kim yapardı?',
      ],
    ),
    const SoruKonusu(
      ad: 'Annem ve Babam',
      ikon: Icons.family_restroom_rounded,
      renk: Color(0xFF8E4A2B),
      sorular: <String>[
        'Annenizi bize anlatır mısınız? Nasıl biriydi?',
        'Babanız ne iş yapardı? Onu çalışırken hatırlıyor musunuz?',
        'Anne babanızdan öğrendiğiniz en önemli şey neydi?',
        'Babaannenizi, dedenizi hatırlıyor musunuz? Onları anlatın.',
        'Evinizde akşamları neler yapardınız, sofra nasıl kurulurdu?',
        'Anne babanızın size en çok söylediği söz neydi?',
        'Kardeşlerinizle aranız nasıldı? En çok kiminle anlaşırdınız?',
        'Ailenizde herkesin bildiği bir hikâye var mı?',
      ],
    ),
    const SoruKonusu(
      ad: 'Gençliğim',
      ikon: Icons.auto_awesome_rounded,
      renk: Color(0xFF9C5A16),
      sorular: <String>[
        'Gençliğinizde nasıl giyinirdiniz, modaya uyar mıydınız?',
        'İlk kez şehir dışına çıktığınız yolculuğu anlatır mısınız?',
        'Gençken en sevdiğiniz şarkı hangisiydi?',
        'Askerlik yaptınız mı? Nerede, nasıl geçti?',
        'Gençliğinizde arkadaşlarınızla nereye giderdiniz?',
        'Hayatınızda ilk kez para kazandığınız günü hatırlıyor musunuz?',
        'O yıllarda dünya nasıldı? Bugünden en büyük farkı neydi?',
        'Gençken kendinizi nerede görüyordunuz?',
      ],
    ),
    const SoruKonusu(
      ad: 'Sevgi ve Yuva',
      ikon: Icons.favorite_rounded,
      renk: Color(0xFFA83246),
      sorular: <String>[
        'Eşinizle nasıl tanıştınız? O günü anlatır mısınız?',
        'Onu ilk gördüğünüzde ne düşünmüştünüz?',
        'Düğününüz nasıl oldu? Kimler vardı?',
        'İlk evinizi anlatır mısınız? Nasıl bir yerdi?',
        'Çocuğunuz doğduğunda neler hissettiniz?',
        'Evliliğin sırrı nedir sizce?',
        'Birlikte en çok güldüğünüz an hangisiydi?',
        'Torunlarınız doğduğunda ne hissettiniz?',
      ],
    ),
    const SoruKonusu(
      ad: 'Emek ve Ekmek',
      ikon: Icons.handyman_rounded,
      renk: Color(0xFF5E6B2A),
      sorular: <String>[
        'Hayatınız boyunca ne iş yaptınız? Nasıl başladınız?',
        'İşinizde en gurur duyduğunuz iş neydi?',
        'Çalışırken en zor gününüz hangisiydi?',
        'Zanaatınızı kimden öğrendiniz?',
        'İlk maaşınızla ne aldınız?',
        'Emekli olduğunuz günü hatırlıyor musunuz?',
        'Gençlere işle ilgili ne tavsiye edersiniz?',
      ],
    ),
    const SoruKonusu(
      ad: 'Memleket',
      ikon: Icons.home_work_rounded,
      renk: Color(0xFF3F6B5E),
      sorular: <String>[
        'Memleketinizi anlatır mısınız? Nesi meşhurdur?',
        'Doğduğunuz evi hatırlıyor musunuz? Kaç odası vardı?',
        'Mahallenizde kimler otururdu? Komşularınızı anlatın.',
        'Memleketten göç ettiniz mi? Nasıl bir yolculuktu?',
        'Köyde ya da mahallede bayramlar nasıl geçerdi?',
        'Çocukken gittiğiniz, artık olmayan bir yer var mı?',
        'Memleketinizin bir türküsü, deyişi var mı?',
      ],
    ),
    const SoruKonusu(
      ad: 'Sofra ve Bayram',
      ikon: Icons.local_dining_rounded,
      renk: Color(0xFFB3701A),
      sorular: <String>[
        'En iyi yaptığınız yemeği tarif eder misiniz?',
        'Bayram sabahları eviniz nasıl olurdu?',
        'Kışa hazırlık nasıl yapılırdı? Neler kurutulur, kaynatılırdı?',
        'Annenizin en sevdiğiniz yemeği hangisiydi, nasıl yapardı?',
        'Misafir geldiğinde sofraya ne konurdu?',
        'Ramazanları ya da özel günleri anlatır mısınız?',
        'Bir tarifi torunlarınıza bırakacak olsanız hangisi olurdu?',
      ],
    ),
    const SoruKonusu(
      ad: 'Dostluk',
      ikon: Icons.groups_rounded,
      renk: Color(0xFF4A5B8C),
      sorular: <String>[
        'En yakın arkadaşınız kimdi? Onu anlatır mısınız?',
        'Arkadaşlarınızla yaşadığınız komik bir olay var mı?',
        'Uzun yıllar görüşemediğiniz, hâlâ aklınızda olan biri var mı?',
        'Size iyilik yapmış, unutamadığınız bir insan var mı?',
        'Bir dostunuzdan öğrendiğiniz bir şey var mı?',
      ],
    ),
    const SoruKonusu(
      ad: 'Hayat Dersleri',
      ikon: Icons.menu_book_rounded,
      renk: Color(0xFF6B4A7C),
      sorular: <String>[
        'Hayatta öğrendiğiniz en önemli ders neydi?',
        'Zor bir dönemi nasıl atlattınız?',
        'Gençliğinize dönebilseniz kendinize ne söylerdiniz?',
        'Sizi en çok ne mutlu eder?',
        'Torunlarınızın sizi nasıl hatırlamasını istersiniz?',
        'Hayatınızda en gurur duyduğunuz şey nedir?',
        'Sizce iyi bir insan olmak ne demektir?',
        'Ailenize bırakmak istediğiniz bir öğüdünüz var mı?',
      ],
    ),
    const SoruKonusu(
      ad: 'Bugün',
      ikon: Icons.wb_sunny_rounded,
      renk: Color(0xFF9A7B12),
      sorular: <String>[
        'Bugün nasıl geçti? Neler yaptınız?',
        'Bugün aklınıza gelen bir anı var mı?',
        'Şu an en çok kimi özlüyorsunuz?',
        'Bugünlerde en çok neye seviniyorsunuz?',
        'Yarın için bir planınız var mı?',
        'Bugün torunlarınıza ne söylemek istersiniz?',
      ],
    ),
  ];

  /// Tum sorular tek listede.
  static List<String> get tumSorular => <String>[
        for (final SoruKonusu k in konular) ...k.sorular,
      ];

  static final Random _rastgele = Random();

  /// Rastgele bir soru. [haric] verilirse ayni soruyu tekrar vermemeye calisir.
  static String rastgeleSoru({String? haric}) {
    final List<String> hepsi = tumSorular;
    if (hepsi.length < 2) return hepsi.first;
    String secilen;
    int deneme = 0;
    do {
      secilen = hepsi[_rastgele.nextInt(hepsi.length)];
      deneme++;
    } while (secilen == haric && deneme < 8);
    return secilen;
  }

  /// Sorudan kisa bir baslik uretir: "Çocukluğunuz nerede geçti? ..." ->
  /// "Çocukluğunuz nerede geçti".
  static String soruyuBasligaCevir(String soru) {
    String s = soru.split('?').first.trim();
    if (s.isEmpty) s = soru.trim();
    if (s.length > 48) s = '${s.substring(0, 45).trimRight()}…';
    return s;
  }
}
