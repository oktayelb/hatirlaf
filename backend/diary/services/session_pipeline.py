"""Persistence helpers for the transcript/event pipeline."""

from __future__ import annotations

from django.db import transaction
from django.utils import timezone

from ..models import (
    ConflictReason,
    EventificationStatus,
    Mention,
    MentionType,
    Node,
    NodeKind,
    Session,
    SessionStatus,
)
from ..processing import entity_registry as entity_registry_mod


def persist_parsing_result(session: Session, extraction, parsed, flagged) -> int:
    """Persist mentions, session NLP hints, and completion state."""
    session.nlp_hints = extraction.to_json()
    session.structured_events = []
    session.eventification_status = EventificationStatus.QUEUED
    session.eventification_detail = "Olaylaştırma sıraya alındı."

    with transaction.atomic():
        session.mentions.all().delete()

        conflict_count = 0
        for fm in flagged:
            m = fm.mention
            audio_start, audio_end = _span_to_audio(session.word_timings, m.char_start, m.char_end)
            mention = Mention.objects.create(
                session=session,
                surface=m.surface,
                lemma=m.lemma,
                char_start=m.char_start,
                char_end=m.char_end,
                mention_type=_safe_mention_type(m.mention_type),
                audio_start=audio_start,
                audio_end=audio_end,
                is_conflict=fm.is_conflict,
                conflict_reason=_safe_reason(fm.conflict_reason),
                conflict_hint=fm.conflict_hint,
            )
            if fm.is_conflict:
                conflict_count += 1
            else:
                _auto_resolve(mention, fm)

        session.processed_text = parsed.lemma_text
        session.status = SessionStatus.COMPLETED
        session.status_detail = (
            f"NLP: {parsed.nlp_backend}; NER: {parsed.ner_backend}; "
            f"olaylaştırma: sırada; "
            f"{conflict_count} çatışma."
        )
        session.save(
            update_fields=[
                "processed_text",
                "status",
                "status_detail",
                "structured_events",
                "eventification_status",
                "eventification_detail",
                "nlp_hints",
                "updated_at",
            ]
        )
        transaction.on_commit(lambda: entity_registry_mod.record_session(session.id))

    return conflict_count


def mark_session_failed(session_id: int, detail: str) -> None:
    Session.objects.filter(pk=session_id).update(
        status=SessionStatus.FAILED,
        status_detail=detail[:2000],
        updated_at=timezone.now(),
    )


def mark_eventification_empty(session_id: int) -> None:
    Session.objects.filter(pk=session_id).update(
        structured_events=[],
        eventification_status=EventificationStatus.COMPLETED,
        eventification_detail="Transkript boş; takvim olayı üretilmedi.",
        updated_at=timezone.now(),
    )


def mark_eventification_running(session_id: int) -> None:
    Session.objects.filter(pk=session_id).update(
        eventification_status=EventificationStatus.RUNNING,
        eventification_detail="LLM olaylaştırması çalışıyor.",
        updated_at=timezone.now(),
    )


def persist_eventification_result(session: Session, extraction, llm_result: dict) -> None:
    events = llm_result.get("olay_loglari", [])
    Session.objects.filter(pk=session.pk).update(
        structured_events=events,
        nlp_hints=extraction.to_json(),
        eventification_status=EventificationStatus.COMPLETED,
        eventification_detail=(
            f"{llm_result.get('backend', 'nlp-only')} ile "
            f"{len(events)} olay üretildi."
        ),
        updated_at=timezone.now(),
    )


def mark_eventification_failed(session_id: int, detail: str) -> None:
    Session.objects.filter(pk=session_id).update(
        structured_events=[],
        eventification_status=EventificationStatus.FAILED,
        eventification_detail=detail[:2000],
        updated_at=timezone.now(),
    )


def _auto_resolve(mention: Mention, fm) -> None:
    """Create/find a Node for a non-conflict mention and wire it up."""
    kind = _suggested_kind_to_node_kind(fm.suggested_kind or mention.mention_type)
    label = mention.surface.strip()
    if not label:
        return
    node, _ = Node.objects.get_or_create(kind=kind, label=label)
    if kind == NodeKind.TIME and fm.suggested_resolution.get("time_value"):
        node.time_value = fm.suggested_resolution["time_value"]
        node.save(update_fields=["time_value", "updated_at"])
    mention.node = node
    mention.resolved = True
    mention.resolution_action = "AUTO"
    mention.save(update_fields=["node", "resolved", "resolution_action"])


def _span_to_audio(word_timings: list[dict], char_start: int, char_end: int):
    if not word_timings:
        return None, None
    hits = [
        w for w in word_timings
        if w.get("char_start", -1) >= 0
        and w["char_start"] < char_end
        and w["char_end"] > char_start
    ]
    if not hits:
        return None, None
    return float(hits[0]["start"]), float(hits[-1]["end"])


def _safe_mention_type(value: str) -> str:
    valid = {c[0] for c in MentionType.choices}
    return value if value in valid else MentionType.PERSON


def _safe_reason(value: str) -> str:
    if not value:
        return ""
    valid = {c[0] for c in ConflictReason.choices}
    return value if value in valid else ""


def _suggested_kind_to_node_kind(kind: str) -> str:
    valid = {c[0] for c in NodeKind.choices}
    return kind if kind in valid else NodeKind.OTHER
