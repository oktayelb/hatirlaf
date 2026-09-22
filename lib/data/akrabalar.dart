/// Telefonu kullanan akraba.
///
/// Uygulamanin tek kullanici kimligi budur: ilk acilista bir kez
/// secilir, sonra hem ekranda hem de yedek klasorunun adinda gorunur.
/// Serbest metin yerine kapali bir liste: yasli bir kullanicinin klavye
/// ile ad yazmasini beklemek yerine dort buyuk dugmeden birine
/// dokunmasini istiyoruz, ve boylece kimlikler her telefonda ayni
/// yaziliyor.
enum Akraba {
  anneanne('anneanne', 'Anneannem'),
  babaanne('babaanne', 'Babannem'),
  sukruDede('sukru-dede', 'Şükrü Dedem'),
  oktayDede('oktay-dede', 'Oktay Dedem');

  const Akraba(this.kimlik, this.ad);

  /// Kaliciya yazilan degismez kimlik. Ekrandaki ad degisse bile bu
  /// sabit kalmali: eski telefonlardaki secim bozulmasin.
  final String kimlik;

  /// Ekranda gorunen ve yedege giden ad.
  final String ad;

  /// Kaydedilmis kimligi geri cevirir. Taninmayan deger `null`:
  /// listeden bir ad kalkarsa uygulama cokmek yerine yeniden sorar.
  static Akraba? kimlikten(String? kimlik) {
    if (kimlik == null || kimlik.isEmpty) return null;
    for (final Akraba a in Akraba.values) {
      if (a.kimlik == kimlik) return a;
    }
    return null;
  }
}

/// Soruyu soran torun. Metinlerde tek yerden degissin.
const String kTorunAdi = 'Oktay';

/// Soruyu soran torunun fotografi. Ekranda tanidik bir yuz olsun diye:
/// "telefon soruyor" ile "torunum soruyor" arasindaki fark buyuk.
const String kTorunFotografi = 'assets/oktay.jpg';

/// Torunla birlikte cekilmis fotograf.
const String kBirlikteFotografi = 'assets/birlikte.jpg';
