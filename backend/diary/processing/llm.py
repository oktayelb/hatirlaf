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
        lines.append(" | ".join(parts))
    return "\n".join(lines)


PASS_1_SYSTEM = (
    "Sen Contextual Voice Diary için çalışan bir analist LLM'sin. "
    "Sana kullanıcının serbest metni ve bu metin üzerinde çalışmış kural tabanlı "
    "bir NLP motorunun çıkardığı CÜMLE BAZLI KESİN ipuçları verilecek.\n\n"
    "GÖREVİN: Metni yeniden yorumlama. İpuçlarındaki tarih, zaman dilimi, kişi ve "
    "lokasyon bilgilerini OLDUĞU GİBİ kullan. Yalnızca hangi cümlelerin aynı olayı "
    "anlattığını birleştir, kişi ve yerleri doğru olayla eşleştir.\n\n"
    "Kurallar:\n"
    "- İpuçlarında 'tarih=YYYY-MM-DD' varsa o tarihi kesin kullan.\n"
    "- İpuçlarında zaman dilimi varsa onu kullan (Geçmiş/Şu An/Gelecek).\n"
    "- Söyleyen kişiyi 'Ben' olarak ekle.\n"
    "- Metinde geçmeyen yer veya kişi uydurma, 'Bilinmeyen Lokasyon' / 'Bilinmeyen Kişi' yaz.\n"
    "- Her olay için: zaman_dilimi, tarih, saat (varsa), lokasyon, olay açıklaması, kisiler listesi.\n"
)


PASS_2_SYSTEM = (
    "Sen bir JSON dönüştürücüsüsün. Sana verilen olay listesini kesin bir JSON "
    "şemasına dönüştür. Başka açıklama yazma. Tarih değerleri YYYY-MM-DD, "
    "saat değerleri HH:MM biçiminde olmalıdır. Eksik alan için boş string kullan."
)


def run(extraction: ExtractionResult) -> dict:
    """Return a canonical ``{"olay_loglari": [...]}``.

    Uses the llama.cpp model when available, otherwise synthesizes the
    same structure from the deterministic NLP hints.
    """
    llm = _load_llm()
    if llm is None:
        return _fallback_from_hints(extraction)

    try:
        hints = _hints_prompt(extraction)
        paragraph = extraction.paragraph

        # Pass 1 — free-form reasoning.
        pass_1 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": PASS_1_SYSTEM},
                {
                    "role": "user",
                    "content": f"{hints}\n\n### Orijinal metin\n{paragraph}",
                },
            ],
            temperature=0.1,
        )
        structured_text = pass_1["choices"][0]["message"]["content"]

        # Pass 2 — strict JSON.
        pass_2 = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": PASS_2_SYSTEM},
                {"role": "user", "content": structured_text},
            ],
            response_format={"type": "json_object", "schema": OLAY_LOG_SCHEMA},
            temperature=0.0,
        )
        raw = pass_2["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        events = parsed.get("olay_loglari") or []
        return {
            "olay_loglari": [_normalize(e, extraction) for e in events],
            "backend": "llama.cpp",
        }
    except Exception as exc:
        logger.exception("LLM inference failed, using NLP-only fallback: %s", exc)
        return _fallback_from_hints(extraction)


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
    # Clean trailing punctuation / whitespace from the event text.
    out["olay"] = (out["olay"] or "").strip()
    return out


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
            prev["kisiler"] = _unique_merge(prev["kisiler"], c.persons)
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
                "kisiler": _unique_merge(c.persons, ["Ben"]),
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
