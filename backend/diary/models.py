"""
Relational adjacency-list models for the Contextual Voice Diary.

Design notes (per spec):
  * Node / Edge replace a graph database. Edges are typed; both endpoints
    point into Node. This keeps every recall query a plain JOIN.
  * Unknown entities are real rows with ``is_unknown=True``. When the user
    leaves a conflict unresolved, the mention is wired to one of the
    canonical "Unknown Person / Location / Time / Event" singleton nodes,
    preserving structural integrity.
  * Mention is the bridge between raw transcript character spans and the
    Node graph. Its ``is_conflict`` flag plus ``conflict_reason`` drives
    the Review Screen.
"""

from __future__ import annotations

from django.db import models


class NodeKind(models.TextChoices):
    PERSON = "PERSON", "Kişi"
    LOCATION = "LOCATION", "Yer"
    TIME = "TIME", "Zaman"
    EVENT = "EVENT", "Olay"
    ORG = "ORG", "Kurum"
    OTHER = "OTHER", "Diğer"


class SessionStatus(models.TextChoices):
    QUEUED = "queued", "Yüklendi, sırada"
    TRANSCRIBING = "transcribing", "Ses yazıya çevriliyor"
    PARSING = "parsing", "Metin analiz ediliyor"
    REVIEW = "review", "İnceleme bekliyor"
    COMPLETED = "completed", "Tamamlandı"
    FAILED = "failed", "Hata"


class ConflictReason(models.TextChoices):
    PRONOUN = "pronoun", "Belirsiz zamir"
    RELATIVE_TIME = "relative_time", "Belirsiz zaman ifadesi"
    AMBIGUOUS_VERB = "ambiguous_verb", "Belirsiz eylem öznesi"
    UNKNOWN_REFERENT = "unknown_referent", "Tanımlanamayan gönderge"


class MentionType(models.TextChoices):
    """Broad surface categories emitted by the parser before resolution."""

    PERSON = "PERSON", "Kişi"
    LOCATION = "LOCATION", "Yer"
    TIME = "TIME", "Zaman"
    EVENT = "EVENT", "Olay"
    ORG = "ORG", "Kurum"
    PRONOUN = "PRONOUN", "Zamir"


class Node(models.Model):
    """A canonical entity in the user's personal knowledge graph."""

    kind = models.CharField(max_length=16, choices=NodeKind.choices)
    label = models.CharField(max_length=200)
    aliases = models.JSONField(default=list, blank=True)
    is_unknown = models.BooleanField(default=False)
    # ISO 8601 string for TIME nodes, else empty.
    time_value = models.CharField(max_length=40, blank=True, default="")
    # Free-form user notes.
    notes = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["kind", "label"]
        indexes = [models.Index(fields=["kind", "label"])]
        constraints = [
            models.UniqueConstraint(
                fields=["kind", "label"],
                name="unique_kind_label",
            )
        ]

    def __str__(self) -> str:
        tag = " (Bilinmiyor)" if self.is_unknown else ""
        return f"[{self.get_kind_display()}] {self.label}{tag}"

    @classmethod
    def unknown_for(cls, kind: str) -> "Node":
        """Return the singleton Unknown-Entity node for ``kind``, creating it."""
        labels = {
            NodeKind.PERSON: "Bilinmeyen Kişi",
            NodeKind.LOCATION: "Bilinmeyen Yer",
            NodeKind.TIME: "Bilinmeyen Zaman",
            NodeKind.EVENT: "Bilinmeyen Olay",
            NodeKind.ORG: "Bilinmeyen Kurum",
            NodeKind.OTHER: "Bilinmeyen",
        }
        label = labels.get(kind, "Bilinmeyen")
        node, _ = cls.objects.get_or_create(
            kind=kind,
            label=label,
            defaults={"is_unknown": True},
        )
        return node


class Session(models.Model):
    """One recorded diary entry."""

    # Stable identifier chosen by the client before upload. Lets the mobile
    # client dedupe retries in its sync queue.
    client_uuid = models.CharField(max_length=64, unique=True)

    audio_file = models.FileField(upload_to="sessions/%Y/%m/%d/", blank=True, null=True)
    duration_seconds = models.FloatField(default=0.0)
    recorded_at = models.DateTimeField()
    language = models.CharField(max_length=8, default="tr")

    status = models.CharField(
        max_length=20, choices=SessionStatus.choices, default=SessionStatus.QUEUED
    )
    status_detail = models.TextField(blank=True, default="")

    transcript = models.TextField(blank=True, default="")
    processed_text = models.TextField(blank=True, default="")

    # Whisper word-level timings, stored as a list of {"word","start","end"}.
    word_timings = models.JSONField(default=list, blank=True)

    # LLM + NLP output: list of events with resolved ISO dates, zaman_dilimi,
    # lokasyon, olay, kisiler. Drives the calendar view.
    structured_events = models.JSONField(default=list, blank=True)
    # Clause-level hints from the deterministic NLP pre-pass. Kept so we
    # can re-run only the LLM step without re-parsing.
    nlp_hints = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-recorded_at"]
        indexes = [
            models.Index(fields=["-recorded_at"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self) -> str:
        return f"Session {self.client_uuid} @ {self.recorded_at:%Y-%m-%d %H:%M}"


class Mention(models.Model):
    """A span in the transcript that references an entity."""

    session = models.ForeignKey(Session, related_name="mentions", on_delete=models.CASCADE)

    surface = models.CharField(max_length=200)
    lemma = models.CharField(max_length=200, blank=True, default="")

    char_start = models.IntegerField()
    char_end = models.IntegerField()

    mention_type = models.CharField(max_length=16, choices=MentionType.choices)

    # Approximate time window in the audio (seconds). Filled when word
    # timings are available so the UI can scrub directly to the mention.
    audio_start = models.FloatField(null=True, blank=True)
    audio_end = models.FloatField(null=True, blank=True)

    # Resolved target node (nullable until resolved).
    node = models.ForeignKey(
        Node, related_name="mentions", on_delete=models.SET_NULL, null=True, blank=True
    )

    is_conflict = models.BooleanField(default=False)
    conflict_reason = models.CharField(
        max_length=32, choices=ConflictReason.choices, blank=True, default=""
    )
    conflict_hint = models.CharField(max_length=300, blank=True, default="")

    resolved = models.BooleanField(default=False)
    # How the mention was resolved: ASSIGNED to an existing/new node,
    # routed to the UNKNOWN fallback, or IGNORED (user dismissed it).
    resolution_action = models.CharField(
        max_length=16,
        choices=[
            ("ASSIGNED", "Assigned"),
            ("UNKNOWN", "Unknown fallback"),
            ("IGNORED", "Ignored"),
            ("AUTO", "Auto-resolved"),
        ],
        blank=True,
        default="",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["session", "char_start"]
        indexes = [
            models.Index(fields=["session", "char_start"]),
            models.Index(fields=["is_conflict", "resolved"]),
        ]

    def __str__(self) -> str:
        return f"{self.mention_type}:{self.surface} ({self.session_id})"


class Edge(models.Model):
    """A typed relationship between two Nodes, sourced from a Session."""

    class Relation(models.TextChoices):
        MENTIONED_WITH = "MENTIONED_WITH", "İle birlikte anıldı"
        HAPPENED_AT = "HAPPENED_AT", "Şurada gerçekleşti"
        HAPPENED_ON = "HAPPENED_ON", "Şu tarihte"
        ASSOCIATED_WITH = "ASSOCIATED_WITH", "İle ilişkili"

    source = models.ForeignKey(Node, related_name="outgoing_edges", on_delete=models.CASCADE)
    target = models.ForeignKey(Node, related_name="incoming_edges", on_delete=models.CASCADE)
    relation = models.CharField(max_length=32, choices=Relation.choices)

    session = models.ForeignKey(
        Session, related_name="edges", on_delete=models.CASCADE, null=True, blank=True
    )

    weight = models.FloatField(default=1.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["source", "relation"]),
            models.Index(fields=["target", "relation"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["source", "target", "relation", "session"],
                name="unique_edge_per_session",
            )
        ]

    def __str__(self) -> str:
        return f"{self.source} -[{self.relation}]-> {self.target}"
