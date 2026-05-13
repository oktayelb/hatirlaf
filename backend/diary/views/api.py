"""REST API endpoints.

Routes (all prefixed with ``/api/``):

  POST   sessions/                   Upload a new diary session (audio + meta).
  GET    sessions/                   List sessions (for timeline).
  GET    sessions/<pk>/              Session detail incl. mentions.
  POST   sessions/<pk>/process/      Re-trigger processing (idempotent).
  GET    sessions/<pk>/audio/        Stream the raw audio.

  GET    mentions/?session=<id>      List mentions.
  POST   mentions/<pk>/resolve/      Resolve a conflict (assign / unknown / ignore).

  GET    nodes/                      List nodes (for assignment UI).
  POST   nodes/                      Create node.
  GET    timeline/                   Chronological feed.
  GET    graph/                      Lightweight node+edge dump.
"""

from __future__ import annotations

import logging
import datetime as dt
from collections import Counter

from django.db.models import Count, Q
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import generics, status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from ..models import (
    Edge,
    EventificationStatus,
    Mention,
    MentionType,
    Node,
    NodeKind,
    Session,
    SessionStatus,
)
from ..processing.pipeline import (
    is_eventification_active,
    is_processing_active,
    kickoff,
)
from ..serializers import (
    EdgeSerializer,
    MentionSerializer,
    NodeSerializer,
    ResolveMentionSerializer,
    SessionDetailSerializer,
    SessionSerializer,
    SessionUploadSerializer,
)

logger = logging.getLogger(__name__)


class SessionViewSet(viewsets.ModelViewSet):
    queryset = Session.objects.all().prefetch_related("mentions__node")
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_serializer_class(self):
        if self.action == "create":
            return SessionUploadSerializer
        if self.action == "retrieve":
            return SessionDetailSerializer
        return SessionSerializer

    def create(self, request, *args, **kwargs):
        """Accept a new recording from the client sync queue.

        If a Session with the same client_uuid already exists we short-circuit
        (idempotent upload — the client's background queue retries safely).
        """
        client_uuid = request.data.get("client_uuid")
        if client_uuid:
            existing = Session.objects.filter(client_uuid=client_uuid).first()
            if existing is not None:
                return Response(
                    SessionDetailSerializer(existing, context={"request": request}).data,
                    status=status.HTTP_200_OK,
                )

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        session = serializer.save()
        kickoff(session.id)
        return Response(
            SessionDetailSerializer(session, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )

    @action(detail=True, methods=["post"], url_path="process")
    def process(self, request, pk=None):
        session = self.get_object()
        processing_in_flight = session.status in {
            SessionStatus.QUEUED,
            SessionStatus.TRANSCRIBING,
            SessionStatus.PARSING,
        }
        eventification_in_flight = session.eventification_status in {
            EventificationStatus.QUEUED,
            EventificationStatus.RUNNING,
        }
        worker_is_active = is_processing_active(session.id) or is_eventification_active(session.id)
        if (processing_in_flight or eventification_in_flight) and worker_is_active:
            return Response(
                SessionDetailSerializer(session, context={"request": request}).data,
                status=status.HTTP_202_ACCEPTED,
            )
        session.status = SessionStatus.QUEUED
        session.status_detail = ""
        session.structured_events = []
        session.eventification_status = EventificationStatus.QUEUED
        session.eventification_detail = "Yeniden olaylaştırma sıraya alındı."
        session.save(
            update_fields=[
                "status",
                "status_detail",
                "structured_events",
                "eventification_status",
                "eventification_detail",
                "updated_at",
            ]
        )
        kickoff(session.id)
        return Response(
            SessionDetailSerializer(session, context={"request": request}).data
        )

    @action(detail=True, methods=["get"], url_path="audio")
    def audio(self, request, pk=None):
        session = self.get_object()
        if not session.audio_file:
            raise Http404("Ses dosyası bulunamadı.")
        return FileResponse(session.audio_file.open("rb"), filename=session.audio_file.name)

    def destroy(self, request, *args, **kwargs):
        """Delete a session and everything that hangs off it.

        Mentions and Edges are removed automatically by the FK cascade.
        Structured events live on the Session row itself, so the calendar
        endpoint stops returning them as soon as the row is gone. The audio
        file is **not** deleted by Django's default storage, so we wipe it
        from disk explicitly.
        """
        session = self.get_object()
        audio = session.audio_file
        audio_storage = audio.storage if audio else None
        audio_name = audio.name if audio else ""

        response = super().destroy(request, *args, **kwargs)

        if audio_storage and audio_name:
            try:
                audio_storage.delete(audio_name)
            except Exception:
                logger.warning(
                    "Could not delete audio file %s for session %s",
                    audio_name,
                    session.pk,
                )
        return response


class MentionViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = MentionSerializer
    queryset = Mention.objects.select_related("node", "session").all()

    def get_queryset(self):
        qs = super().get_queryset()
        session_id = self.request.query_params.get("session")
        only_conflicts = self.request.query_params.get("conflicts")
        if session_id:
            qs = qs.filter(session_id=session_id)
        if only_conflicts in ("1", "true", "True"):
            qs = qs.filter(is_conflict=True, resolved=False)
        return qs

    @action(detail=True, methods=["post"], url_path="resolve")
    def resolve(self, request, pk=None):
        mention = self.get_object()
        serializer = ResolveMentionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        action_kind = serializer.validated_data["action"]

        node: Node | None = None
        resolution_action = "ASSIGNED"

        if action_kind == "ASSIGN":
            node_id = serializer.validated_data.get("node_id")
            if not node_id:
                return Response(
                    {"detail": "node_id gerekli."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            node = get_object_or_404(Node, pk=node_id)

        elif action_kind == "NEW":
            label = (serializer.validated_data.get("label") or mention.surface).strip()
            kind = serializer.validated_data.get("kind") or _default_kind_for(mention)
            if kind not in {c[0] for c in NodeKind.choices}:
                kind = NodeKind.OTHER
            node, _ = Node.objects.get_or_create(
                kind=kind,
                label=label,
                defaults={
                    "time_value": serializer.validated_data.get("time_value", ""),
                    "notes": serializer.validated_data.get("notes", ""),
                },
            )

        elif action_kind == "UNKNOWN":
            kind = _default_kind_for(mention)
            node = Node.unknown_for(kind)
            resolution_action = "UNKNOWN"

        elif action_kind == "IGNORE":
            # User dismissed the flag entirely — still keep the mention but
            # unattached. Per spec we preserve graph integrity by routing to
            # the Unknown fallback so relationships can still be computed.
            kind = _default_kind_for(mention)
            node = Node.unknown_for(kind)
            resolution_action = "IGNORED"

        mention.node = node
        mention.resolved = True
        mention.is_conflict = False
        mention.resolution_action = resolution_action
        mention.save(update_fields=["node", "resolved", "is_conflict", "resolution_action"])

        _rebuild_edges_for_session(mention.session)

        return Response(MentionSerializer(mention, context={"request": request}).data)


class NodeViewSet(viewsets.ModelViewSet):
    serializer_class = NodeSerializer
    queryset = Node.objects.annotate(_mention_count=Count("mentions")).all()

    def get_queryset(self):
        qs = super().get_queryset()
        kind = self.request.query_params.get("kind")
        q = self.request.query_params.get("q")
        if kind:
            qs = qs.filter(kind=kind)
        if q:
            qs = qs.filter(Q(label__icontains=q) | Q(aliases__icontains=q))
        return qs


class EdgeListView(generics.ListAPIView):
    serializer_class = EdgeSerializer
    queryset = Edge.objects.select_related("source", "target", "session").all()


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
    """Event-per-day rollup for the Personal History calendar.

    Query params:
      * ``month`` — ``YYYY-MM``. Restricts returned days to that month.
                    Defaults to all events.

    Returns::
        {
          "days": {
            "2026-04-14": [ { "session_id", "zaman_dilimi", "tarih", "saat",
                              "lokasyon", "olay", "kisiler" }, ... ],
            ...
          }
        }
    """
    from collections import defaultdict

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
                status=status.HTTP_400_BAD_REQUEST,
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
                    "lokasyon": ev.get("lokasyon", ""),
                    "olay": ev.get("olay", ""),
                    "kisiler": ev.get("kisiler", []),
                }
            )

    # Sort events within each day by saat if present.
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
                status=status.HTTP_400_BAD_REQUEST,
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
            label = mention.node.label if mention.node else mention.surface
            label = (label or "").strip()
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
                "people": event.get("kisiler", []),
                "place": event.get("lokasyon", ""),
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


def _month_bounds(month: str) -> tuple[dt.date, dt.date]:
    y, m = month.split("-")
    year = int(y)
    month_num = int(m)
    start = dt.date(year, month_num, 1)
    if month_num == 12:
        return start, dt.date(year + 1, 1, 1)
    return start, dt.date(year, month_num + 1, 1)


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
    from ..processing import startup

    return Response({"ok": True, "startup": startup.snapshot()})


# --- helpers ----------------------------------------------------------------


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
    """Recompute co-mention edges for the session.

    Strategy: within a session, every resolved Node becomes connected to
    every other resolved Node via MENTIONED_WITH. PERSON→LOCATION links are
    upgraded to HAPPENED_AT, PERSON→TIME to HAPPENED_ON.
    """
    mentions = (
        session.mentions.filter(node__isnull=False, resolved=True)
        .select_related("node")
    )
    node_ids = list({m.node_id for m in mentions})
    nodes = {n.id: n for n in Node.objects.filter(id__in=node_ids)}

    # Clear existing edges for this session before rebuilding.
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
