"""Local persistent registry of encountered people and places.

This registry lives in the app's SQLite database so it is local to each clone
of the repo and survives restarts without needing a full rebuild. We only
upsert new bare-root entities as they are encountered.
"""

from __future__ import annotations

import threading

from django.utils import timezone

from ..models import EncounteredEntity, Mention, MentionType, Node, NodeKind
from . import nlp as nlp_mod

_REGISTRY_LOCK = threading.Lock()
_VALID_KINDS = {NodeKind.PERSON, NodeKind.LOCATION}


def normalize_label(value: str) -> str:
    """Normalize an entity surface to a bare-root lowercase label."""
    return nlp_mod.normalize_entity_lemma(str(value or "")).strip().lower()


def record_entity(kind: str, label: str) -> EncounteredEntity | None:
    """Insert or refresh a bare-root entity in the local registry."""
    kind = _normalize_kind(kind)
    normalized = normalize_label(label)
    if not kind or not normalized:
        return None

    with _REGISTRY_LOCK:
        entity, created = EncounteredEntity.objects.get_or_create(kind=kind, label=normalized)
        if not created:
            EncounteredEntity.objects.filter(pk=entity.pk).update(last_seen_at=timezone.now())
        return entity


def record_node(node: Node) -> EncounteredEntity | None:
    if node.kind not in _VALID_KINDS or node.is_unknown:
        return None
    return record_entity(node.kind, node.label)


def record_mention(mention: Mention) -> EncounteredEntity | None:
    if mention.mention_type not in _VALID_KINDS:
        return None

    if mention.node and not mention.node.is_unknown and mention.node.kind in _VALID_KINDS:
        return record_entity(mention.node.kind, mention.node.label)
    return record_entity(mention.mention_type, mention.surface)


def record_session(session_id: int) -> dict:
    """Record all PERSON/LOCATION mentions from a session into the registry."""
    mentions = (
        Mention.objects.filter(session_id=session_id, mention_type__in=_VALID_KINDS)
        .select_related("node")
        .only("surface", "mention_type", "node__kind", "node__label", "node__is_unknown")
    )
    for mention in mentions:
        record_mention(mention)
    return snapshot()


def snapshot() -> dict:
    """Return the current registry as a JSON-serializable payload."""
    people = list(
        EncounteredEntity.objects.filter(kind=NodeKind.PERSON)
        .order_by("label")
        .values_list("label", flat=True)
    )
    places = list(
        EncounteredEntity.objects.filter(kind=NodeKind.LOCATION)
        .order_by("label")
        .values_list("label", flat=True)
    )
    return {
        "version": 1,
        "updated_at": timezone.now().isoformat(),
        "people": people,
        "places": places,
    }


def _normalize_kind(kind: str) -> str:
    if kind in _VALID_KINDS:
        return kind
    if kind == MentionType.PERSON:
        return NodeKind.PERSON
    if kind == MentionType.LOCATION:
        return NodeKind.LOCATION
    return ""
