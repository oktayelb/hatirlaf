from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict

from django.db.models import Count, Q
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ..models import Edge, EventificationStatus, Mention, MentionType, Node, NodeKind, Session, SessionStatus
from ..processing import startup
from .api_shared import (
    _calendar_events_for_session,
    _counter_items,
    _entity_display_label,
    _recap_highlights,
    _recap_summary,
    _recap_title,
    _recorded_date_iso,
)


@api_view(["GET"])
def timeline_view(request):
    """Return sessions + key mentions in chronological order for the timeline."""
    sessions = (
        Session.objects.exclude(status=SessionStatus.FAILED)
        .prefetch_related("mentions__node")
        .order_by("-recorded_at")
    )
    data = []
    for s in sessions:
        mentions = [
            {
                "id": m.id,
                "surface": m.surface,
                "mention_type": m.mention_type,
                "node_id": m.node_id,
                "node_label": m.node.label if m.node else None,
                "node_is_unknown": m.node.is_unknown if m.node else None,
                "is_conflict": m.is_conflict,
                "resolved": m.resolved,
            }
            for m in s.mentions.all()
        ]
        data.append(
            {
                "id": s.id,
                "client_uuid": s.client_uuid,
                "recorded_at": s.recorded_at,
                "duration_seconds": s.duration_seconds,
                "status": s.status,
                "transcript": s.transcript,
                "conflict_count": sum(1 for m in mentions if m["is_conflict"] and not m["resolved"]),
                "mentions": mentions,
            }
        )
    return Response({"results": data})


@api_view(["GET"])
def calendar_view(request):
    """Event-per-day rollup for the Personal History calendar."""
    month = (request.query_params.get("month") or "").strip()
    month_start: dt.date | None = None
    month_end: dt.date | None = None
    if month:
        try:
            y, m = month.split("-")
            month_start = dt.date(int(y), int(m), 1)
            if int(m) == 12:
                month_end = dt.date(int(y) + 1, 1, 1)
            else:
                month_end = dt.date(int(y), int(m) + 1, 1)
        except ValueError:
            return Response(
                {"detail": "month must be YYYY-MM"},
                status=400,
            )

    qs = Session.objects.exclude(status=SessionStatus.FAILED).filter(
        Q(status=SessionStatus.COMPLETED)
        | Q(eventification_status=EventificationStatus.COMPLETED)
    )
    days: dict[str, list[dict]] = defaultdict(list)

    for s in qs.only("id", "recorded_at", "structured_events", "transcript", "nlp_hints"):
        events = _calendar_events_for_session(s)
        for ev in events:
            iso = (ev.get("tarih") or "").strip()
            if not iso:
                continue
            try:
                d = dt.date.fromisoformat(iso)
            except ValueError:
                continue
            if month_start and (d < month_start or d >= month_end):
                continue
            days[iso].append(
                {
                    "session_id": s.id,
                    "zaman_dilimi": ev.get("zaman_dilimi", ""),
                    "tarih": iso,
                    "saat": ev.get("saat", ""),
                    "lokasyon": _entity_display_label(ev.get("lokasyon", "")),
                    "olay": ev.get("olay", ""),
                    "kisiler": [_entity_display_label(person) for person in ev.get("kisiler", [])],
                }
            )

    for iso in days:
        days[iso].sort(key=lambda e: (e.get("saat") or "99:99"))

    return Response({"days": days})


@api_view(["GET"])
def recap_view(request):
    """Monthly memory digest built from sessions, events, and graph mentions."""
    month = (request.query_params.get("month") or "").strip()
    if month:
        try:
            month_start, month_end = _month_bounds(month)
        except ValueError:
            return Response(
                {"detail": "month must be YYYY-MM"},
                status=400,
            )
    else:
        today = timezone.localdate()
        month_start, month_end = _month_bounds(f"{today.year}-{today.month:02d}")

    sessions = list(
        Session.objects.exclude(status=SessionStatus.FAILED)
        .filter(recorded_at__date__gte=month_start, recorded_at__date__lt=month_end)
        .prefetch_related("mentions__node")
        .order_by("recorded_at")
    )
    event_rows: list[dict] = []
    people: Counter[str] = Counter()
    places: Counter[str] = Counter()
    days: Counter[str] = Counter()
    future_count = 0
    past_count = 0

    for session in sessions:
        for mention in session.mentions.all():
            label = _entity_display_label(mention.node.label if mention.node else mention.surface)
            if not label or (mention.node and mention.node.is_unknown):
                continue
            if mention.mention_type == MentionType.PERSON:
                people[label] += 1
            elif mention.mention_type in {MentionType.LOCATION, MentionType.ORG}:
                places[label] += 1

        for event in _calendar_events_for_session(session):
            if not isinstance(event, dict):
                continue
            iso = (event.get("tarih") or _recorded_date_iso(session)).strip()
            try:
                event_date = dt.date.fromisoformat(iso)
            except ValueError:
                continue
            if event_date < month_start or event_date >= month_end:
                continue
            row = {
                "session_id": session.id,
                "recorded_at": session.recorded_at,
                "date": iso,
                "time": event.get("saat", ""),
                "bucket": event.get("zaman_dilimi", ""),
                "text": event.get("olay", ""),
                "people": [_entity_display_label(person) for person in event.get("kisiler", [])],
                "place": _entity_display_label(event.get("lokasyon", "")),
            }
            event_rows.append(row)
            days[iso] += 1
            if row["bucket"] == "Gelecek":
                future_count += 1
            elif row["bucket"] == "Geçmiş":
                past_count += 1
            for person in row["people"] if isinstance(row["people"], list) else []:
                person = str(person).strip()
                if person and person.lower() != "ben":
                    people[person] += 1
            place = str(row["place"] or "").strip()
            if place and "bilinmeyen" not in place.lower():
                places[place] += 1

    unresolved_count = Mention.objects.filter(
        session__in=sessions,
        is_conflict=True,
        resolved=False,
    ).count()
    highlights = _recap_highlights(event_rows, sessions)
    busiest = [
        {"date": iso, "count": count}
        for iso, count in sorted(days.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]

    return Response(
        {
            "month": month_start.strftime("%Y-%m"),
            "title": _recap_title(month_start, len(sessions), event_rows, people, places),
            "summary": _recap_summary(month_start, sessions, event_rows, people, places),
            "stats": {
                "sessions": len(sessions),
                "events": len(event_rows),
                "people": len(people),
                "places": len(places),
                "future_events": future_count,
                "past_events": past_count,
                "unresolved_conflicts": unresolved_count,
                "active_days": len(days),
            },
            "top_people": _counter_items(people),
            "top_places": _counter_items(places),
            "busiest_days": busiest,
            "highlights": highlights,
        }
    )


@api_view(["GET"])
def graph_view(request):
    nodes = list(
        Node.objects.annotate(_mention_count=Count("mentions")).values(
            "id", "kind", "label", "is_unknown", "_mention_count"
        )
    )
    edges = list(Edge.objects.values("id", "source_id", "target_id", "relation", "weight"))
    return Response({"nodes": nodes, "edges": edges})


@api_view(["GET"])
def health_view(request):
    return Response({"ok": True, "startup": startup.snapshot()})


def _month_bounds(month: str) -> tuple[dt.date, dt.date]:
    y, m = month.split("-")
    year = int(y)
    month_num = int(m)
    start = dt.date(year, month_num, 1)
    if month_num == 12:
        return start, dt.date(year + 1, 1, 1)
    return start, dt.date(year, month_num + 1, 1)
