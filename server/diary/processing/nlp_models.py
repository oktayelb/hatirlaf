"""Shared NLP dataclasses and lexicons."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Token:
    surface: str
    lemma: str
    char_start: int
    char_end: int
    pos: str = ""
    is_proper: bool = False


@dataclass
class EntityMention:
    surface: str
    lemma: str
    char_start: int
    char_end: int
    mention_type: str  # PERSON | LOCATION | TIME | EVENT | ORG | PRONOUN
    source: str = ""  # "hf_ner" | "rules" | "gazetteer" | "pronoun" | "time"
    score: float = 1.0
    hint: str = ""


@dataclass
class ParseResult:
    tokens: list[Token] = field(default_factory=list)
    mentions: list[EntityMention] = field(default_factory=list)
    lemma_text: str = ""
    nlp_backend: str = "rules"
    ner_backend: str = "rules"


AMBIGUOUS_PRONOUNS = {
    "o", "onu", "ona", "onun", "onda", "ondan", "onunla", "onlar", "onları",
    "onlara", "onların", "onlarda", "onlardan",
    "bu", "bunu", "buna", "bunun", "bunda", "bundan", "bununla",
    "şu", "şunu", "şuna", "şunun", "şunda", "şundan", "şununla",
    "bunlar", "bunları", "bunlara", "bunların",
    "şunlar", "şunları", "şunlara", "şunların",
    "kendisi", "kendileri",
    "oradaki", "oradakini", "oradakine", "oradakinin", "oradakiler",
    "buradaki", "buradakini", "buradakine", "buradakinin", "buradakiler",
    "şuradaki", "şuradakini", "şuradakine", "şuradakinin", "şuradakiler",
}

RELATIVE_TIME_WORDS = {
    "dün", "bugün", "yarın", "geçen", "önceki", "gelecek", "gelecekteki",
    "önce", "sonra", "biraz", "az", "şimdi", "az önce", "biraz önce",
    "demin", "az sonra", "öbür", "evvelki", "ertesi",
    "dünkü", "dünkünü", "dünküne", "dünkünün", "bugünkü", "bugünkünü",
    "bugünküne", "bugünkünün", "yarınki", "yarınkini", "yarınkine",
    "yarınkinin",
    "hafta", "ay", "yıl", "gün", "akşam", "sabah", "gece", "öğle", "öğleden",
}

RELATIVE_LOCATION_WORDS = {
    "burası", "orası", "şurası", "burada", "orada", "şurada",
    "buraya", "oraya", "şuraya", "buradan", "oradan", "şuradan",
    "buradaki", "oradaki", "şuradaki",
}

TR_CITIES = {
    "istanbul", "ankara", "izmir", "bursa", "antalya", "konya", "adana",
    "gaziantep", "şanlıurfa", "mersin", "diyarbakır", "kayseri", "eskişehir",
    "samsun", "denizli", "trabzon", "malatya", "erzurum", "kocaeli",
    "sakarya", "manisa", "hatay", "balıkesir", "tekirdağ", "van", "aydın",
    "muğla", "tokat", "elazığ", "sivas", "rize", "ordu", "giresun", "artvin",
}

TR_COUNTRIES = {
    "türkiye", "almanya", "fransa", "italya", "ispanya", "yunanistan",
    "bulgaristan", "rusya", "ukrayna", "amerika", "abd", "ingiltere",
    "japonya", "çin", "kore", "brezilya", "kanada", "hollanda", "belçika",
}

TR_MONTHS = {
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
}

TR_WEEKDAYS = {
    "pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar",
}
