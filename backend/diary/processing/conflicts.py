"""Conflict detection + default fallback routing.

A conflict is an ``EntityMention`` that the parser can't confidently ground to
a canonical Node. Typical examples per spec:

  * Pronouns ("o", "bu", "şu", "onlar") with no nearby antecedent.
  * Relative times ("dün", "geçen hafta") when the session's
    ``recorded_at`` is ambiguous or missing.
  * Relative locations ("burası", "orası") when no anchor is set.
  * Unknown proper nouns that don't already exist in the graph.

This module walks the parser output, tags each ambiguous mention with a
``conflict_reason`` + ``conflict_hint``, and returns a structured plan the API
layer can serialize for the Review Screen.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .nlp import AMBIGUOUS_PRONOUNS, EntityMention, RELATIVE_LOCATION_WORDS, RELATIVE_TIME_WORDS


@dataclass
class FlaggedMention:
    mention: EntityMention
    is_conflict: bool
    conflict_reason: str = ""
    conflict_hint: str = ""
    suggested_kind: str = ""
    suggested_resolution: dict = field(default_factory=dict)


def detect_conflicts(
    mentions: list[EntityMention],
    text: str,
    recorded_at: dt.datetime | None = None,
) -> list[FlaggedMention]:
    """Return a FlaggedMention for every input mention, marking conflicts."""
    flagged: list[FlaggedMention] = []
    for m in mentions:
        reason = ""
        hint = ""
        suggested_kind = _mention_to_kind(m.mention_type)
        suggested_resolution: dict = {}
        clean_surface = " ".join(m.surface.lower().split())

        if m.mention_type == "PRONOUN":
            reason = "pronoun"
            hint = m.hint or "Belirsiz zamir — kime atıfta bulunuyor?"
            antecedent = _find_antecedent(m, mentions)
            if antecedent is not None:
                hint = (
                    f"Yakındaki aday: '{antecedent.surface}'. "
                    "Onaylamak için seçebilirsin."
                )
                suggested_resolution = {
                    "action": "ASSIGNED",
                    "mention_type": antecedent.mention_type,
                    "label": antecedent.surface,
                }
        elif (
            m.mention_type == "LOCATION"
            and clean_surface in RELATIVE_LOCATION_WORDS
        ):
            reason = "unknown_referent"
            hint = m.hint or "Belirsiz yer — neresi kastediliyor?"
        elif (
            m.mention_type == "TIME"
            and m.source == "time"
            and (
                clean_surface in RELATIVE_TIME_WORDS
                or any(w in clean_surface.split() for w in RELATIVE_TIME_WORDS)
            )
        ):
            absolute = _ground_relative_time(clean_surface, recorded_at)
            if absolute is None:
                reason = "relative_time"
                hint = m.hint or "Göreceli zaman — tam tarih gerekli mi?"
            else:
                # We can auto-ground with recorded_at, but still surface it for
                # user confirmation on the review screen.
                hint = f"Otomatik: {absolute.isoformat()}"
                suggested_resolution = {
                    "action": "ASSIGNED",
                    "mention_type": "TIME",
                    "label": absolute.strftime("%d %B %Y"),
                    "time_value": absolute.isoformat(),
                }

        is_conflict = bool(reason) or (
            m.source == "rules"
            and m.mention_type in ("PERSON", "LOCATION", "ORG")
            and not suggested_resolution
        )
        if (
            is_conflict
            and not reason
            and m.mention_type in ("PERSON", "LOCATION", "ORG")
        ):
            reason = "unknown_referent"
            hint = "Bu kişi/yeri daha önce kaydetmedin — yeni düğüm olarak ekleyelim mi?"

        flagged.append(
            FlaggedMention(
                mention=m,
                is_conflict=is_conflict,
                conflict_reason=reason,
                conflict_hint=hint,
                suggested_kind=suggested_kind,
                suggested_resolution=suggested_resolution,
            )
        )
    return flagged


def _mention_to_kind(mention_type: str) -> str:
    return {
        "PERSON": "PERSON",
        "LOCATION": "LOCATION",
        "TIME": "TIME",
        "EVENT": "EVENT",
        "ORG": "ORG",
        "PRONOUN": "PERSON",
    }.get(mention_type, "OTHER")


def _find_antecedent(
        pronoun: EntityMention, mentions: list[EntityMention]
) -> EntityMention | None:
    """Walk backwards to the nearest NAMED PERSON/ORG/LOCATION."""
    candidates = [
        m for m in mentions
        if m.char_end <= pronoun.char_start
        and m.mention_type in ("PERSON", "ORG", "LOCATION")
        and m.source in ("berturk", "rules")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.char_end)


def _ground_relative_time(
        surface: str, recorded_at: dt.datetime | None
) -> dt.date | None:
    if recorded_at is None:
        return None
    today = recorded_at.date()

    mapping = {
        "bugün": today,
        "bugünkü": today,
        "dün": today - dt.timedelta(days=1),
        "dünkü": today - dt.timedelta(days=1),
        "yarın": today + dt.timedelta(days=1),
        "yarınki": today + dt.timedelta(days=1),
        "evvelsi gün": today - dt.timedelta(days=2),
        "evvelki gün": today - dt.timedelta(days=2),
        "dünden önceki gün": today - dt.timedelta(days=2),
        "yarından sonraki gün": today + dt.timedelta(days=2),
        "geçen hafta": today - dt.timedelta(days=7),
        "geçtiğimiz hafta": today - dt.timedelta(days=7),
        "haftaya": today + dt.timedelta(days=7),
        "gelecek hafta": today + dt.timedelta(days=7),
        "önümüzdeki hafta": today + dt.timedelta(days=7),
    }
    
    # Note: "geçen gün" is intentionally omitted here as it typically means 
    # "a few days ago" idiomatically. Returning None ensures it correctly 
    # flags as a conflict for the user to resolve manually.

    return mapping.get(surface)