"""Local llama.cpp wrapper that turns NLP hints into an event log.

Two-pass design (see chat.py for the standalone prototype):

  Pass 1 (free-form): model reads the paragraph + structured NLP hints
          and produces a rich natural-language breakdown. Giving the
          model room to "think" beats asking it to emit JSON up front.

  Pass 2 (JSON-constrained): model reformats Pass 1 into strict JSON
          validated against ``OLAY_LOG_SCHEMA``. The schema is enforced
          by llama.cpp's GBNF grammar, so we don't need post-hoc repair.

The model stays loaded for the life of the process. Loading is
**lazy + thread-safe** so unit tests and the REST server don't pay the
cold-start cost unless a session actually triggers LLM inference.

If the model file is absent (dev machines without the weights) the
wrapper falls back to ``fallback_events_from_hints``, which packages the
NLP extractor's clause list into the same JSON shape. The rest of the
backend therefore always sees a consistent event log, with or without an
LLM.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from typing import Any

from django.conf import settings

from .extractor import ExtractionResult, unmask

logger = logging.getLogger(__name__)


OLAY_LOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "olay_loglari": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zaman_dilimi": {
                        "type": "string",
                        "enum": ["Geçmiş", "Şu An", "Gelecek"],
                    },
                    "tarih": {"type": "string"},
                    "saat": {"type": "string"},
                    "lokasyon": {"type": "string"},
                    "olay": {"type": "string"},
                    "kisiler": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "zaman_dilimi",
                    "tarih",
                    "lokasyon",
                    "olay",
                    "kisiler",
                ],
            },
        }
    },
    "required": ["olay_loglari"],
}


_lock = threading.Lock()
_cached_llm = None
_cached_path: str | None = None
_load_failed = False


def _model_path() -> str:
    return getattr(
        settings,
        "HATIRLAF_LLM_MODEL_PATH",
        os.environ.get("HATIRLAF_LLM_MODEL_PATH", "./Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    )


def _load_llm():
    """Return a cached ``llama_cpp.Llama`` or ``None`` on failure."""
    global _cached_llm, _cached_path, _load_failed
    if _load_failed:
        return None
    with _lock:
        if _cached_llm is not None:
            return _cached_llm
        path = _model_path()
        if not path or not os.path.exists(path):
            logger.warning("LLM model not found at %s — NLP-only fallback enabled.", path)
            _load_failed = True
            return None
        try:
            from llama_cpp import Llama  # type: ignore

            logger.info("Loading llama.cpp model from %s", path)
            _cached_llm = Llama(
                model_path=path,
                n_ctx=int(getattr(settings, "HATIRLAF_LLM_N_CTX", 4096)),
                n_gpu_layers=int(getattr(settings, "HATIRLAF_LLM_N_GPU_LAYERS", -1)),
                verbose=False,
            )
            _cached_path = path
        except Exception as exc:
            logger.exception("Failed to load llama.cpp model: %s", exc)
            _load_failed = True
            _cached_llm = None
        return _cached_llm


def is_available() -> bool:
    return _load_llm() is not None


def preload() -> None:
    """Best-effort warm-load of the LLM weights.

    Called from ``DiaryConfig.ready`` in a background thread so the server
    starts answering health checks immediately while the multi-GB model
    streams into memory. Idempotent — safe to call repeatedly.
    """
    try:
        if _load_llm() is not None:
            logger.info("LLM preload complete (path=%s).", _cached_path)
        else:
            logger.info("LLM preload skipped — model unavailable.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM preload failed: %s", exc)


def _hints_prompt(extraction: ExtractionResult) -> str:
    lines = ["### Önişleme Verisi (NLP'den gelen kesin ipuçları)"]
    anchor = extraction.recorded_at.isoformat() if extraction.recorded_at else "bilinmiyor"
    lines.append(f"- Kayıt anı: {anchor}")
    if extraction.persons:
        lines.append(f"- Kişi adayları: {', '.join(extraction.persons)}")
    if extraction.locations:
        lines.append(f"- Yer adayları: {', '.join(extraction.locations)}")
    if extraction.orgs:
        lines.append(f"- Kurum adayları: {', '.join(extraction.orgs)}")
    if extraction.references:
        lines.append(
            "- Belirsiz gönderge/zaman adayları: "
            f"{', '.join(extraction.references)}"
        )

    lines.append("\n### Cümle bazlı ipuçları (tarih ve zaman dilimi çoktan çözüldü)")
    for c in extraction.clauses:
        parts = [f"[{c.clause_index}] \"{c.text.strip()}\""]
        if c.date_iso:
            parts.append(f"tarih={c.date_iso}")
        if c.time_hm:
            parts.append(f"saat={c.time_hm}")
        if c.zaman_dilimi:
            parts.append(f"zaman={c.zaman_dilimi}")
        if c.persons:
            parts.append(f"kisiler={','.join(c.persons)}")
        if c.locations:
            parts.append(f"lokasyon={','.join(c.locations)}")
        if c.time_phrases:
            parts.append(f"ifade={'/'.join(c.time_phrases)}")
        if c.references:
            parts.append(f"gondergeler={','.join(c.references)}")
        if c.subject_person:
            parts.append(
                "ozne="
                f"{c.subject_person}/{c.subject_pronoun}"
                f"({c.subject_verb}:{c.subject_tense})"
            )
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _baseline_prompt(baseline: dict) -> str:
    return (
        "### Deterministik NLP taslağı\n"
        "Bu JSON, modelden önce üretilen güvenli taslaktır. Eksik olabilir ama "
        "tarih/saat/özne/kişi/yer alanları için ana referanstır. Model bunu "
        "iyileştirebilir; açık kanıt yoksa taslakla çelişmemelidir.\n"
        f"{json.dumps(baseline.get('olay_loglari', []), ensure_ascii=False, indent=2)}"
    )


PASS_1_SYSTEM = (
    "Sen Hatırlaf'ın Türkçe olay çıkarım motorusun. Görevin, kullanıcının "
    "serbest konuşma/metin günlüğünü takvimde gösterilecek olay kayıtlarına "
    "dönüştürmektir. Model yaratıcılığı değil, kanıta dayalı yapılandırma "
    "isteniyor.\n\n"
    "GİRDİ SÖZLEŞMESİ:\n"
    "- 'Önişleme Verisi' kural tabanlı NLP katmanından gelir ve en güvenilir "
    "kaynak kabul edilir.\n"
    "- 'Deterministik NLP taslağı' güvenli başlangıç çıktısıdır. Onu yalnızca "
    "orijinal metin ve ipuçları açıkça daha iyi bir birleşim gerektiriyorsa "
    "değiştir.\n"
    "- Orijinal metin yalnızca kanıt kontrolü ve olayları birleştirme için "
    "kullanılır; metinde/ipucunda olmayan kişi, yer, tarih veya saat üretme.\n\n"
    "ALAN KURALLARI:\n"
    "1) tarih: 'tarih=YYYY-MM-DD' ipucu varsa aynen kullan. Yoksa kayıt anına "
    "göre çıkar; emin değilsen kayıt gününü kullan.\n"
    "2) saat: yalnız açık saat ifadesi varsa HH:MM yaz. Yoksa boş bırak.\n"
    "3) zaman_dilimi: tarih kayıt gününden önceyse Geçmiş, aynı günse Şu An, "
    "sonraysa Gelecek. Bu sınıflandırmayı metin hissine göre bozma.\n"
    "4) kisiler: açık kişi adlarını, 'Ben'i ve 'ozne=' ipucundan gelen düşmüş "
    "özneyi kullan. 'ozne=1sg/Ben' -> Ben, '1pl/Biz' -> Biz, '2sg/Sen' -> Sen, "
    "'2pl/Siz' -> Siz, '3pl/Onlar' -> Onlar. '3sg/O' tek başına kişi kanıtı "
    "değildir; yalnız açık 'o/o da' kişi göndergesi veya yakın kişi adayı varsa "
    "kişiye bağla.\n"
    "5) lokasyon: yalnız açık yer/kurum adaylarından seç. 'orada/burada/"
    "oradaki/buradaki' gibi göndergeleri açık yakın yerle eşleştiremiyorsan "
    "lokasyonu boş bırak veya Bilinmeyen Lokasyon yaz.\n"
    "6) olay: kısa, doğal Türkçe cümle yaz. Belirsiz gönderge çözülmediyse "
    "bunu olay metninde koru: 'oradaki konuşma', 'bu plan' gibi.\n\n"
    "BELİRSİZ GÖNDERGE ÇÖZME:\n"
    "- Önce aynı cümledeki açık kişi/yer/olaya bak.\n"
    "- Sonra bir önceki cümledeki en yakın uyumlu adaya bak.\n"
    "- 'o/onun' kişi veya olay olabilir; açık kanıt yoksa kişi adı üretme.\n"
    "- 'bu/şu/bunun/şunun' çoğunlukla önceki olay/konu/planı gösterir; kişi "
    "olarak yorumlama.\n"
    "- 'orada/oradaki' en yakın önceki lokasyona bağlanabilir; yoksa "
    "Bilinmeyen Lokasyon.\n"
    "- 'dünkü/bugünkü/yarınki' tarih ipucudur; kişi/yer değildir.\n\n"
    "OLAY BİRLEŞTİRME:\n"
    "- Aynı tarih, aynı zaman bağlamı ve devam cümleleri tek olay olmalı.\n"
    "- 'sonra', 'orada', 'bunun üzerine', 'o da' gibi devam ifadeleri genelde "
    "önceki olayı zenginleştirir.\n"
    "- Farklı tarih, farklı ana fiil ve farklı katılımcı/yer varsa ayrı olay "
    "aç.\n\n"
    "ÇIKIŞ BİÇİMİ: Numaralı liste halinde, her olay için ayrı satır grubu.\n"
    "Olay 1:\n"
    "  - zaman_dilimi: ...\n"
    "  - tarih: YYYY-MM-DD\n"
    "  - saat: HH:MM veya (yok)\n"
    "  - lokasyon: ...\n"
    "  - olay: kısa Türkçe cümle\n"
    "  - kisiler: [Ben, Ad1, Ad2]\n"
)


PASS_CRITIQUE_SYSTEM = (
    "Sen Hatırlaf'ın olay çıkarımı denetçisisin. Sana orijinal metin, NLP "
    "ipuçları, deterministik taslak ve analist çıktısı verilecek.\n\n"
    "GÖREVİN: Analist çıktısını güvenilirlik açısından düzeltmek. Yeni bilgi "
    "icat etme; yalnızca aşağıdaki hataları ara ve onar:\n"
    "- Tarih ipucuyla çelişiyor mu? (örn. ipucu 2026-04-24 ama olay 2026-04-23)\n"
    "- zaman_dilimi tarih ile tutarlı mı? (tarih bugünse Şu An, gelecekse "
    "Gelecek, geçmişse Geçmiş)\n"
    "- 'ozne=' ipucundan gelen düşmüş özne kişi listesinde kaybolmuş mu?\n"
    "- 'gondergeler=' alanındaki belirsiz ifade kişi/yer uydurularak yanlış "
    "çözülmüş mü?\n"
    "- Uydurulmuş kişi veya yer var mı? Metinde veya NLP ipuçlarında "
    "geçmiyorsa silmelisin.\n"
    "- Küçük harfle yazılmış özel ad düzeltilmiş mi? (emre → Emre)\n"
    "- Aynı olay iki kere listelenmiş mi? Birleştir.\n"
    "- Bir cümle önceki olayın devamıysa ayrı olay olarak kalmış mı? Birleştir.\n\n"
    "Düzeltilmiş listeyi aynı biçimde tekrar yaz. Hiçbir hata yoksa listeyi "
    "değiştirmeden kopyala ve sonunda '(Kontrol edildi, sorun yok)' yaz."
)


PASS_2_SYSTEM = (
    "Sen katı bir JSON dönüştürücüsüsün. Verilen olay listesini kesin bir JSON "
    "şemasına dönüştür. Kurallar:\n"
    "- Yalnızca JSON ver, başka metin yazma.\n"
    "- tarih: YYYY-MM-DD biçiminde.\n"
    "- saat: HH:MM biçiminde veya boş string.\n"
    "- zaman_dilimi: 'Geçmiş', 'Şu An' veya 'Gelecek'.\n"
    "- kisiler: metindeki/ipuçlarındaki adlar ve çekimden gelen zamirler. "
    "'Ben' her zaman dahil.\n"
    "- lokasyon bilinmiyorsa boş string veya 'Bilinmeyen Lokasyon'.\n"
    "- Eksik alanlar için boş string kullan, null kullanma.\n"
)


def run(extraction: ExtractionResult) -> dict:
    """Return a canonical ``{"olay_loglari": [...]}``.

    Uses the llama.cpp model when available (three-pass: analyse, self-critique,
    JSON format), otherwise synthesizes the same structure from the
    deterministic NLP hints. The NLP fallback always runs first so we have a
    known-good baseline to merge with whatever the LLM produces.
    """
    baseline = _fallback_from_hints(extraction)

    llm = _load_llm()
    if llm is None:
        return baseline

    try:
        hints = _hints_prompt(extraction)
        baseline_text = _baseline_prompt(baseline)
        paragraph = extraction.paragraph

        # Pass 1 — free-form analysis of events.
        pass_1 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": PASS_1_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{hints}\n\n"
                        f"{baseline_text}\n\n"
                        f"### Orijinal metin\n{paragraph}"
                    ),
                },
            ],
            temperature=0.1,
        )
        draft = pass_1["choices"][0]["message"]["content"]

        # Pass 2 — self-critique + repair.
        pass_critique = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": PASS_CRITIQUE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"### Orijinal metin\n{paragraph}\n\n"
                        f"{hints}\n\n"
                        f"{baseline_text}\n\n"
                        f"### Analist Çıktısı\n{draft}"
                    ),
                },
            ],
            temperature=0.0,
        )
        verified = pass_critique["choices"][0]["message"]["content"]

        # Pass 3 — strict JSON.
        pass_2 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": PASS_2_SYSTEM},
                {"role": "user", "content": verified},
            ],
            response_format={"type": "json_object", "schema": OLAY_LOG_SCHEMA},
            temperature=0.0,
        )
        raw = pass_2["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        events = parsed.get("olay_loglari") or []
        normalised = [_normalize(e, extraction) for e in events]

        # Guardrail: if the model returned nothing coherent, fall back to the
        # deterministic baseline. Otherwise post-process to keep fields sane.
        cleaned = [ev for ev in normalised if ev.get("olay")]
        if not cleaned:
            logger.warning("LLM produced no usable events; using NLP fallback.")
            return baseline

        cleaned = [_sanitize_event(ev, extraction) for ev in cleaned]
        return {"olay_loglari": cleaned, "backend": "llama.cpp"}
    except Exception as exc:
        logger.exception("LLM inference failed, using NLP-only fallback: %s", exc)
        return baseline


def _normalize(event: dict, extraction: ExtractionResult) -> dict:
    """Enforce schema + translate any mask tokens the LLM echoed back."""
    mask = extraction.mask_map or {}
    out = {
        "zaman_dilimi": event.get("zaman_dilimi", ""),
        "tarih": event.get("tarih", ""),
        "saat": event.get("saat", ""),
        "lokasyon": unmask(event.get("lokasyon", "") or "", mask),
        "olay": unmask(event.get("olay", "") or "", mask),
        "kisiler": [unmask(p, mask) for p in event.get("kisiler", []) if p],
    }
    out["olay"] = (out["olay"] or "").strip()
    return out


_ALLOWED_BUCKETS = {"Geçmiş", "Şu An", "Gelecek"}
_TIME_RE = __import__("re").compile(r"^(\d{1,2}):(\d{2})$")


def _sanitize_event(event: dict, extraction: ExtractionResult) -> dict:
    """Final guardrails on LLM output. Keeps tarih/zaman_dilimi consistent,
    normalises time format, ensures 'Ben' appears, and drops any made-up
    persons that don't appear in the NLP hints.
    """
    tarih = (event.get("tarih") or "").strip()
    try:
        if tarih:
            dt.date.fromisoformat(tarih)
    except ValueError:
        tarih = ""
    if not tarih:
        tarih = (
            extraction.recorded_at.date().isoformat()
            if extraction.recorded_at
            else dt.date.today().isoformat()
        )

    # zaman_dilimi must match tarih vs recorded_at.
    inferred_bucket = _default_bucket(tarih, extraction.recorded_at)
    bucket = (event.get("zaman_dilimi") or "").strip()
    if bucket not in _ALLOWED_BUCKETS:
        bucket = inferred_bucket or "Geçmiş"
    elif inferred_bucket and bucket != inferred_bucket:
        bucket = inferred_bucket

    saat = (event.get("saat") or "").strip()
    if saat:
        m = _TIME_RE.match(saat)
        if m:
            h = int(m.group(1))
            mnt = int(m.group(2))
            if 0 <= h <= 23 and 0 <= mnt <= 59:
                saat = f"{h:02d}:{mnt:02d}"
            else:
                saat = ""
        else:
            saat = ""

    # Restrict kisiler to names the NLP hints actually saw, plus grammatical
    # pronouns recovered from dropped Turkish subjects.
    known_people = {p.lower(): p for p in extraction.persons}
    grammatical_people = {"ben": "Ben"}
    for c in extraction.clauses:
        if c.subject_pronoun and c.subject_person != "3sg":
            grammatical_people[c.subject_pronoun.lower()] = c.subject_pronoun
    kisiler_in = event.get("kisiler") or []
    cleaned_people: list[str] = []
    seen: set[str] = set()
    for raw in kisiler_in:
        p = (raw or "").strip()
        if not p:
            continue
        key = p.lower()
        if key in grammatical_people:
            display = grammatical_people[key]
        elif key in known_people:
            display = known_people[key]
        elif any(p in extraction.paragraph for p in [p]):
            display = p
        else:
            # Unknown name that the LLM hallucinated — drop it.
            continue
        if display.lower() not in seen:
            seen.add(display.lower())
            cleaned_people.append(display)
    if "ben" not in seen:
        cleaned_people.insert(0, "Ben")

    lokasyon = (event.get("lokasyon") or "").strip()
    if (
        lokasyon
        and lokasyon != "Bilinmeyen Lokasyon"
        and lokasyon.lower() not in extraction.paragraph.lower()
        and lokasyon not in extraction.locations
        and lokasyon not in extraction.orgs
    ):
        # Location wasn't in the paragraph or hints — probably hallucinated.
        lokasyon = ""

    return {
        "zaman_dilimi": bucket,
        "tarih": tarih,
        "saat": saat,
        "lokasyon": lokasyon,
        "olay": (event.get("olay") or "").strip(),
        "kisiler": cleaned_people,
    }


def _fallback_from_hints(extraction: ExtractionResult) -> dict:
    """Package the NLP extractor's clauses into the same JSON shape.

    Contiguous clauses that share a date and have no hard contradictions
    are coalesced into one event so "gittik. orada vardı" stays a single
    meetup rather than two half-described ones.
    """
    events: list[dict] = []
    for c in extraction.clauses:
        if not (c.date_iso or c.persons or c.locations or c.event_phrase):
            continue

        # Try to merge into the previous event when the clauses clearly
        # describe the same situation.
        if events and _should_merge(events[-1], c, extraction):
            prev = events[-1]
            prev["kisiler"] = _unique_merge(prev["kisiler"], _clause_people(c))
            if c.locations and (prev["lokasyon"] in ("", "Bilinmeyen Lokasyon")):
                prev["lokasyon"] = c.locations[0]
            if c.time_hm and not prev["saat"]:
                prev["saat"] = c.time_hm
            if c.event_phrase and c.event_phrase not in prev["olay"]:
                prev["olay"] = f"{prev['olay']} {c.event_phrase}".strip()
            continue

        events.append(
            {
                "zaman_dilimi": c.zaman_dilimi or _default_bucket(c.date_iso, extraction.recorded_at),
                "tarih": c.date_iso or _recorded_iso(extraction),
                "saat": c.time_hm,
                "lokasyon": (c.locations[0] if c.locations else (c.orgs[0] if c.orgs else "")),
                "olay": c.event_phrase or c.text[:120],
                "kisiler": _clause_people(c),
            }
        )

    # Guarantee: every processed session contributes at least one calendar
    # entry. If nothing was extractable (empty transcript, noise-only text)
    # we still anchor a minimal event on the recording day so the calendar
    # surfaces that the session happened.
    if not events:
        fallback_text = (extraction.paragraph or "").strip()
        events.append(
            {
                "zaman_dilimi": _default_bucket(_recorded_iso(extraction), extraction.recorded_at) or "Geçmiş",
                "tarih": _recorded_iso(extraction),
                "saat": _recorded_time(extraction),
                "lokasyon": "",
                "olay": fallback_text[:160] if fallback_text else "Ses kaydı",
                "kisiler": ["Ben"],
            }
        )
    return {"olay_loglari": events, "backend": "nlp-only"}


def _clause_people(clause) -> list[str]:
    people = list(clause.persons or [])
    if (
        clause.subject_pronoun
        and clause.subject_person != "3sg"
        and clause.subject_pronoun not in people
    ):
        people.append(clause.subject_pronoun)
    return _unique_merge(people, ["Ben"])


def _recorded_iso(extraction: ExtractionResult) -> str:
    if extraction.recorded_at is None:
        return dt.date.today().isoformat()
    return extraction.recorded_at.date().isoformat()


def _recorded_time(extraction: ExtractionResult) -> str:
    if extraction.recorded_at is None:
        return ""
    return extraction.recorded_at.strftime("%H:%M")


def _should_merge(prev: dict, clause, extraction: ExtractionResult) -> bool:
    if prev["tarih"] and clause.date_iso and prev["tarih"] != clause.date_iso:
        return False
    prev_bucket = prev["zaman_dilimi"]
    cur_bucket = clause.zaman_dilimi or _default_bucket(clause.date_iso, extraction.recorded_at)
    if prev_bucket and cur_bucket and prev_bucket != cur_bucket:
        return False
    # Merge when the clause introduces *extra* info about the prior event
    # (more people or a location) but no contradictory new date.
    if not clause.date_iso and (clause.persons or clause.locations):
        return True
    return False


def _unique_merge(*lists: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for it in lst or []:
            k = it.lower()
            if k and k not in seen:
                seen.add(k)
                out.append(it)
    return out


def _default_bucket(
    date_iso: str, recorded_at: dt.datetime | None
) -> str:
    if not date_iso or recorded_at is None:
        return ""
    try:
        d = dt.date.fromisoformat(date_iso)
    except ValueError:
        return ""
    today = recorded_at.date()
    if d < today:
        return "Geçmiş"
    if d > today:
        return "Gelecek"
    return "Şu An"
