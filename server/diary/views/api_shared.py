from __future__ import annotations

import datetime as dt
import logging
from collections import Counter, defaultdict

from django.db.models import Count
from django.utils import timezone

from ..models import Edge, Mention, MentionType, Node, NodeKind, Session, SessionStatus
from ..processing import nlp as nlp_mod

logger = logging.getLogger(__name__)


def _normalise_text(value) -> str:
    return str(value or "").strip().casefold()


def _entity_display_label(value) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    display = nlp_mod.normalize_entity_label(raw)
    return display or raw


def _default_kind_for(mention: Mention) -> str:
    mapping = {
        "PERSON": NodeKind.PERSON,
        "LOCATION": NodeKind.LOCATION,
        "TIME": NodeKind.TIME,
        "EVENT": NodeKind.EVENT,
        "ORG": NodeKind.ORG,
        "PRONOUN": NodeKind.PERSON,
    }
    return mapping.get(mention.mention_type, NodeKind.OTHER)


def _rebuild_edges_for_session(session: Session) -> None:
    """Recompute co-mention edges for the session."""
    mentions = (
        session.mentions.filter(node__isnull=False, resolved=True)
        .select_related("node")
    )
    node_ids = list({m.node_id for m in mentions})
    nodes = {n.id: n for n in Node.objects.filter(id__in=node_ids)}

    Edge.objects.filter(session=session).delete()

    persons = [nid for nid in node_ids if nodes[nid].kind == NodeKind.PERSON]
    locations = [nid for nid in node_ids if nodes[nid].kind == NodeKind.LOCATION]
    times = [nid for nid in node_ids if nodes[nid].kind == NodeKind.TIME]

    created: set[tuple[int, int, str]] = set()

    def _make(source_id, target_id, relation):
        if source_id == target_id:
            return
        key = (source_id, target_id, relation)
        if key in created:
            return
        Edge.objects.update_or_create(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            session=session,
            defaults={"weight": 1.0},
        )
        created.add(key)

    for p in persons:
        for loc in locations:
            _make(p, loc, Edge.Relation.HAPPENED_AT)
        for t in times:
            _make(p, t, Edge.Relation.HAPPENED_ON)

    for i, a in enumerate(node_ids):
        for b in node_ids[i + 1 :]:
            _make(a, b, Edge.Relation.MENTIONED_WITH)
            _make(b, a, Edge.Relation.MENTIONED_WITH)


def _tr_month_name(value: dt.date) -> str:
    names = [
        "Ocak",
        "Şubat",
        "Mart",
        "Nisan",
        "Mayıs",
        "Haziran",
        "Temmuz",
        "Ağustos",
        "Eylül",
        "Ekim",
        "Kasım",
        "Aralık",
    ]
    return names[value.month - 1]


def _calendar_events_for_session(session: Session) -> list[dict]:
    """Return structured events, then NLP hints, then a recording fallback."""
    events = session.structured_events or []
    if events:
        expanded = _expand_structured_event_text(events, session)
        return [event for event in expanded if isinstance(event, dict)]

    hint_events = _events_from_nlp_hints(session)
    if hint_events:
        return hint_events

    recorded_at = timezone.localtime(session.recorded_at) if session.recorded_at else None
    return [
        {
            "zaman_dilimi": "Geçmiş",
            "tarih": recorded_at.date().isoformat() if recorded_at else "",
            "saat": recorded_at.strftime("%H:%M") if recorded_at else "",
            "lokasyon": "",
            "olay": (session.transcript or "").strip()[:140] or "Kayıt tamamlandı.",
            "kisiler": [],
        }
    ]


def _expand_structured_event_text(events: list[dict], session: Session) -> list[dict]:
    clauses = (session.nlp_hints or {}).get("clauses") or []

    expanded = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if not isinstance(clauses, list):
            expanded.append(event)
            continue
        full_text = _matching_clause_text(event, clauses)
        if not full_text:
            expanded.append(event)
            continue
        enriched = dict(event)
        enriched["olay"] = full_text
        expanded.append(enriched)
    return expanded


def _matching_clause_text(event: dict, clauses: list) -> str:
    event_text = (event.get("olay") or "").strip()
    if not event_text:
        return ""
    event_date = (event.get("tarih") or "").strip()
    for clause in clauses:
        if not isinstance(clause, dict):
            continue
        clause_text = (clause.get("text") or "").strip()
        if not clause_text or len(clause_text) <= len(event_text):
            continue
        clause_date = (clause.get("date_iso") or "").strip()
        if event_date and clause_date and event_date != clause_date:
            continue
        phrase = (clause.get("event_phrase") or "").strip()
        if event_text in clause_text or event_text == phrase or event_text in phrase:
            return clause_text
    return ""


def _events_from_nlp_hints(session: Session) -> list[dict]:
    hints = session.nlp_hints or {}
    clauses = hints.get("clauses") or []
    if not isinstance(clauses, list):
        return []

    events: list[dict] = []
    for clause in clauses:
        if not isinstance(clause, dict):
            continue
        text = (clause.get("text") or "").strip()
        date_iso = (clause.get("date_iso") or "").strip()
        people = _hint_people(clause)
        locations = clause.get("locations") or []
        orgs = clause.get("orgs") or []
        location = ""
        if isinstance(locations, list) and locations:
            location = str(locations[0] or "").strip()
        elif isinstance(orgs, list) and orgs:
            location = str(orgs[0] or "").strip()

        if not (date_iso or text or people or location):
            continue

        events.append(
            {
                "zaman_dilimi": _bucket_for_date(date_iso, session.recorded_at)
                or (clause.get("zaman_dilimi") or ""),
                "tarih": date_iso or _recorded_date_iso(session),
                "saat": clause.get("time_hm") or "",
                "lokasyon": location,
                "olay": text or (clause.get("event_phrase") or "").strip(),
                "kisiler": people,
            }
        )
    return events


def _hint_people(clause: dict) -> list[str]:
    people = [str(p).strip() for p in (clause.get("persons") or []) if str(p).strip()]
    subject = (clause.get("subject_pronoun") or "").strip()
    subject_person = (clause.get("subject_person") or "").strip()
    if subject and subject_person != "3sg" and subject not in people:
        people.append(subject)
    return people


def _recorded_date_iso(session: Session) -> str:
    if not session.recorded_at:
        return ""
    return timezone.localtime(session.recorded_at).date().isoformat()


def _bucket_for_date(date_iso: str, recorded_at) -> str:
    if not date_iso or recorded_at is None:
        return ""
    try:
        event_date = dt.date.fromisoformat(date_iso)
    except ValueError:
        return ""
    recorded_date = timezone.localtime(recorded_at).date()
    if event_date < recorded_date:
        return "Geçmiş"
    if event_date > recorded_date:
        return "Gelecek"
    return "Şu An"


def _node_memories_for(*, kind: str, label: str, node: Node | None) -> list[dict]:
    label_norm = _normalise_text(_entity_display_label(label))
    sessions = (
        Session.objects.exclude(status=SessionStatus.FAILED)
        .prefetch_related("mentions__node")
        .order_by("-recorded_at")
    )

    rows: list[dict] = []
    for session in sessions:
        matched_mentions = [
            mention
            for mention in session.mentions.all()
            if _mention_matches_entity(mention, kind=kind, label_norm=label_norm, node=node)
        ]
        event_matches = [
            event
            for event in _calendar_events_for_session(session)
            if _event_matches_entity(event, kind=kind, label_norm=label_norm, label=label)
        ]
        if not matched_mentions and not event_matches:
            continue

        rows.append(
            {
                "session_id": session.id,
                "recorded_at": session.recorded_at,
                "status": session.status,
                "eventification_status": session.eventification_status,
                "matched_label": label,
                "transcript_excerpt": _memory_excerpt(session, label, matched_mentions, event_matches),
                "matched_mentions": [
                    {
                        "id": mention.id,
                        "surface": mention.surface,
                        "display_label": _entity_display_label(mention.node.label if mention.node else mention.surface),
                        "mention_type": mention.mention_type,
                        "resolved": mention.resolved,
                        "node_id": mention.node_id,
                    }
                    for mention in matched_mentions
                ],
                "event_matches": [
                    {
                        "date": event.get("tarih", ""),
                        "time": event.get("saat", ""),
                        "bucket": event.get("zaman_dilimi", ""),
                        "text": event.get("olay", ""),
                        "place": _entity_display_label(event.get("lokasyon", "")),
                        "people": [_entity_display_label(person) for person in event.get("kisiler", [])],
                    }
                    for event in event_matches
                ],
            }
        )
    return rows


def _mention_matches_entity(mention: Mention, *, kind: str, label_norm: str, node: Node | None) -> bool:
    if node and mention.node_id == node.id:
        return True
    if not label_norm:
        return False
    if mention.mention_type != kind:
        return False
    candidate = mention.node.label if mention.node else mention.surface
    return _normalise_text(_entity_display_label(candidate)) == label_norm


def _event_matches_entity(event: dict, *, kind: str, label_norm: str, label: str) -> bool:
    if kind == NodeKind.PERSON:
        people = event.get("kisiler") or []
        for person in people:
            person_norm = _normalise_text(_entity_display_label(person))
            if person_norm and person_norm != "ben" and person_norm == label_norm:
                return True
        return False
    if kind == NodeKind.LOCATION:
        return _normalise_text(_entity_display_label(event.get("lokasyon"))) == label_norm
    if kind == NodeKind.ORG:
        return _normalise_text(_entity_display_label(event.get("lokasyon"))) == label_norm or any(
            _normalise_text(_entity_display_label(person)) == label_norm for person in (event.get("kisiler") or [])
        )
    return label_norm in _normalise_text(event.get("olay"))


def _memory_excerpt(
    session: Session,
    label: str,
    mentions: list[Mention],
    event_matches: list[dict],
) -> str:
    text = (session.transcript or "").strip()
    if text:
        for mention in mentions:
            if mention.char_start is None or mention.char_end is None:
                continue
            if mention.char_start < 0 or mention.char_end <= mention.char_start:
                continue
            start = max(0, mention.char_start - 52)
            end = min(len(text), mention.char_end + 90)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet = snippet + "…"
            if snippet:
                return snippet
        label_norm = _normalise_text(label)
        idx = _normalise_text(text).find(label_norm)
        if idx >= 0:
            start = max(0, idx - 52)
            end = min(len(text), idx + len(label) + 90)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet = snippet + "…"
            if snippet:
                return snippet
    if event_matches:
        first = event_matches[0]
        bits = [str(first.get("text") or "").strip()]
        place = str(first.get("place") or "").strip()
        if place:
            bits.append(place)
        return " · ".join(bit for bit in bits if bit)
    return text[:180] + ("…" if len(text) > 180 else "")


def _counter_items(counter: Counter[str], limit: int = 6) -> list[dict]:
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def _recap_title(
    month_start: dt.date,
    session_count: int,
    events: list[dict],
    people: Counter[str],
    places: Counter[str],
) -> str:
    month_name = _tr_month_name(month_start)
    if people:
        return f"{month_name}: en çok {people.most_common(1)[0][0]} akılda kaldı"
    if places:
        return f"{month_name}: {places.most_common(1)[0][0]} öne çıktı"
    if events:
        return f"{month_name}: {len(events)} anı yakalandı"
    if session_count:
        return f"{month_name}: {session_count} kayıt saklandı"
    return f"{month_name}: henüz sessiz"


def _recap_summary(
    month_start: dt.date,
    sessions: list[Session],
    events: list[dict],
    people: Counter[str],
    places: Counter[str],
) -> str:
    if not sessions:
        return "Bu ay için henüz günlük kaydı yok. İlk kaydı eklediğinde burası kendiliğinden bir hafıza özetine dönüşür."
    month_name = _tr_month_name(month_start)
    bits = [f"{month_name} ayında {len(sessions)} kayıt ve {len(events)} takvim olayı oluştu."]
    if people:
        names = ", ".join(label for label, _ in people.most_common(3))
        bits.append(f"En çok adı geçen kişiler: {names}.")
    if places:
        labels = ", ".join(label for label, _ in places.most_common(2))
        bits.append(f"Öne çıkan yerler: {labels}.")
    if not people and not places:
        bits.append("Kayıtlar daha çok kişisel notlar ve olay akışı etrafında toplanıyor.")
    return " ".join(bits)


def _recap_highlights(events: list[dict], sessions: list[Session]) -> list[dict]:
    if events:
        ranked = sorted(
            events,
            key=lambda ev: (
                len(str(ev.get("people") or [])),
                len(str(ev.get("text") or "")),
            ),
            reverse=True,
        )
        return ranked[:5]
    fallback = []
    for session in sessions[:5]:
        text = (session.transcript or "").strip()
        if not text:
            continue
        fallback.append(
            {
                "session_id": session.id,
                "recorded_at": session.recorded_at,
                "date": _recorded_date_iso(session),
                "time": timezone.localtime(session.recorded_at).strftime("%H:%M"),
                "bucket": "Geçmiş",
                "text": text[:180],
                "people": [],
                "place": "",
            }
        )
    return fallback

