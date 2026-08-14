"""Common Turkish given names.

Used by the extractor to catch person mentions that the rule-based NER
misses because the user wrote them lowercase ("emre alp geldi"). This
list is intentionally broad — a few false positives are acceptable
because the LLM pass + conflict review will prune them, while a false
negative loses an event entirely.
"""

from __future__ import annotations

TR_GIVEN_NAMES: frozenset[str] = frozenset({
    # Male
    "ahmet", "ali", "ayhan", "bahadır", "barış", "batuhan", "berat", "berke",
    "burak", "can", "cem", "cemil", "cenk", "deniz", "doruk", "efe", "emir",
    "emre", "eren", "ertan", "ertuğrul", "fatih", "ferit", "furkan", "gökhan",
    "görkem", "halil", "hakan", "hasan", "hüseyin", "ibrahim", "ilhan",
    "ilker", "ilyas", "iskender", "ismail", "kaan", "kadir", "kamil",
    "kemal", "kerem", "koray", "kutay", "levent", "mahmut", "mehmet", "melih",
    "mert", "metin", "murat", "mustafa", "nazım", "necati", "nuri", "oğuz",
    "oğuzhan", "okan", "oktay", "olcay", "onur", "orhan", "osman", "ozan",
    "öner", "ömer", "özcan", "özgür", "polat", "rıza", "sabri", "sadık",
    "said", "salih", "samet", "sarp", "selim", "serdar", "serhan", "serkan",
    "seyit", "sinan", "soner", "suat", "süleyman", "şahin", "şeref", "tayfun",
    "tolga", "tuğrul", "turan", "ufuk", "uğur", "umut", "ümit", "vedat",
    "volkan", "yalçın", "yasin", "yavuz", "yiğit", "yılmaz", "yunus",
    "yusuf", "zafer",
    # Female
    "ahu", "ayla", "aylin", "ayşe", "banu", "başak", "belgin", "berna",
    "beyza", "buse", "ceren", "ceylan", "damla", "defne", "derya", "didem",
    "dilara", "dilek", "duygu", "ebru", "ecem", "eda", "elif", "emel", "esma",
    "esra", "eylül", "fatma", "feray", "feride", "figen", "filiz", "funda",
    "gamze", "gizem", "gönül", "gül", "gülnur", "gülsün", "handan", "hande",
    "hatice", "hilal", "hülya", "ipek", "irem", "işıl", "kadriye", "kezban",
    "lale", "leyla", "melek", "meltem", "merve", "meryem", "miray", "müge",
    "naz", "nazan", "nazlı", "nergis", "neslihan", "neva", "nihal", "nilay",
    "nilgün", "nilüfer", "nisa", "nur", "nuray", "oya", "özden", "özge",
    "özlem", "pelin", "pınar", "rabia", "sanem", "seda", "sedef", "selin",
    "semra", "sevda", "sevgi", "sevilay", "sevim", "sıla", "simge", "songül",
    "sude", "şule", "tuğba", "tülay", "tülin", "yasemin", "yeliz", "yonca",
    "zehra", "zeynep", "zuhal",
    # Unisex / common
    "alper", "asya", "ata", "ayberk", "berk", "bora", "çağla", "çınar",
    "doğa", "doğan", "ege", "eylem", "işık", "kıvanç", "poyraz", "sevgi",
})
