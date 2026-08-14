from __future__ import annotations

from django.utils import timezone
from django.urls import reverse
from rest_framework import serializers

from .pipeline import flags
from .processing import nlp as nlp_mod
from .models import Edge, EventificationStatus, Mention, Node, Session, SessionStatus


class NodeSerializer(serializers.ModelSerializer):
    kind_display = serializers.CharField(source="get_kind_display", read_only=True)
    mention_count = serializers.IntegerField(source="mentions.count", read_only=True)
    display_label = serializers.SerializerMethodField()

    class Meta:
        model = Node
        fields = [
            "id",
            "kind",
            "kind_display",
            "label",
            "display_label",
            "aliases",
            "is_unknown",
            "time_value",
            "notes",
            "mention_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "kind_display",
            "display_label",
            "mention_count",
            "created_at",
            "updated_at",
        ]

    def get_display_label(self, obj):
        return nlp_mod.normalize_entity_label(obj.label)


class EdgeSerializer(serializers.ModelSerializer):
    source_label = serializers.CharField(source="source.label", read_only=True)
    source_kind = serializers.CharField(source="source.kind", read_only=True)
    target_label = serializers.CharField(source="target.label", read_only=True)
    target_kind = serializers.CharField(source="target.kind", read_only=True)
    relation_display = serializers.CharField(source="get_relation_display", read_only=True)

    class Meta:
        model = Edge
        fields = [
            "id",
            "source",
            "source_label",
            "source_kind",
            "target",
            "target_label",
            "target_kind",
            "relation",
            "relation_display",
            "session",
            "weight",
            "created_at",
        ]


class MentionSerializer(serializers.ModelSerializer):
    node = NodeSerializer(read_only=True)
    node_id = serializers.PrimaryKeyRelatedField(
        source="node",
        queryset=Node.objects.all(),
        required=False,
        allow_null=True,
        write_only=True,
    )
    mention_type_display = serializers.CharField(source="get_mention_type_display", read_only=True)
    conflict_reason_display = serializers.CharField(
        source="get_conflict_reason_display", read_only=True
    )
    display_label = serializers.SerializerMethodField()

    class Meta:
        model = Mention
        fields = [
            "id",
            "session",
            "surface",
            "lemma",
            "char_start",
            "char_end",
            "audio_start",
            "audio_end",
            "mention_type",
            "mention_type_display",
            "node",
            "node_id",
            "display_label",
            "is_conflict",
            "conflict_reason",
            "conflict_reason_display",
            "conflict_hint",
            "resolved",
            "resolution_action",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "session",
            "surface",
            "lemma",
            "char_start",
            "char_end",
            "audio_start",
            "audio_end",
            "mention_type",
            "mention_type_display",
            "is_conflict",
            "conflict_reason",
            "conflict_reason_display",
            "conflict_hint",
            "node",
            "display_label",
            "created_at",
        ]

    def get_display_label(self, obj):
        source = obj.node.label if obj.node and obj.node.label else obj.surface
        return nlp_mod.normalize_entity_label(source)


class SessionSerializer(serializers.ModelSerializer):
    #: Everything the understanding pipeline produces. Removed from the
    #: payload entirely while the NLP flag is off, so a client cannot show
    #: analysis output that does not exist.
    NLP_FIELDS = (
        "processed_text",
        "word_timings",
        "structured_events",
        "eventification_status",
        "eventification_detail",
        "conflict_count",
        "mention_count",
        "mentions",
        "mood",
        "mood_source",
        "tags",
        "tags_source",
    )

    status_display = serializers.CharField(source="get_status_display", read_only=True)
    audio_url = serializers.SerializerMethodField()
    conflict_count = serializers.SerializerMethodField()
    mention_count = serializers.SerializerMethodField()
    processing_progress = serializers.SerializerMethodField()
    processing_detail = serializers.SerializerMethodField()
    mood = serializers.CharField(required=False, allow_blank=True)
    tags = serializers.ListField(
        child=serializers.CharField(allow_blank=False, max_length=40),
        required=False,
    )
    word_timings = serializers.JSONField(read_only=True)
    structured_events = serializers.JSONField(read_only=True)

    class Meta:
        model = Session
        fields = [
            "id",
            "client_uuid",
            "audio_url",
            "duration_seconds",
            "recorded_at",
            "language",
            "status",
            "status_display",
            "status_detail",
            "transcript",
            "processed_text",
            "mood",
            "mood_source",
            "tags",
            "tags_source",
            "word_timings",
            "structured_events",
            "eventification_status",
            "eventification_detail",
            "processing_progress",
            "processing_detail",
            "conflict_count",
            "mention_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "status_display",
            "status_detail",
            "processed_text",
            "word_timings",
            "structured_events",
            "eventification_status",
            "eventification_detail",
            "processing_progress",
            "processing_detail",
            "conflict_count",
            "mention_count",
            "audio_url",
            "mood_source",
            "tags_source",
            "created_at",
            "updated_at",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not flags.nlp_enabled():
            for name in self.NLP_FIELDS:
                self.fields.pop(name, None)

    def validate_tags(self, value):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise serializers.ValidationError("tags must be a list of strings.")
        cleaned = []
        seen = set()
        for item in value[:20]:
            tag = str(item or "").strip().lower()
            if not tag or tag in seen:
                continue
            seen.add(tag)
            cleaned.append(tag[:40])
        return cleaned

    def validate_mood(self, value):
        return str(value or "").strip().lower()[:40]

    def update(self, instance, validated_data):
        if "mood" in validated_data:
            validated_data["mood_source"] = "manual"
        if "tags" in validated_data:
            validated_data["tags_source"] = "manual"
        return super().update(instance, validated_data)

    def get_audio_url(self, obj):
        if obj.audio_file:
            request = self.context.get("request")
            url = reverse("session-audio", kwargs={"pk": obj.pk})
            return request.build_absolute_uri(url) if request else url
        return None

    def get_conflict_count(self, obj):
        return obj.mentions.filter(is_conflict=True, resolved=False).count()

    def get_mention_count(self, obj):
        return obj.mentions.count()

    def get_processing_progress(self, obj):
        if not flags.nlp_enabled():
            return self._capture_only_progress(obj)
        if obj.status == SessionStatus.FAILED or obj.eventification_status == EventificationStatus.FAILED:
            return 100
        if obj.status == SessionStatus.QUEUED:
            return 0
        if obj.status == SessionStatus.TRANSCRIBING:
            return self._elapsed_progress(obj, start=12, end=42, seconds=90)
        if obj.status == SessionStatus.PARSING:
            return self._elapsed_progress(obj, start=45, end=64, seconds=45)
        if obj.status in {SessionStatus.COMPLETED, SessionStatus.REVIEW}:
            if obj.eventification_status == EventificationStatus.QUEUED:
                return 68
            if obj.eventification_status == EventificationStatus.RUNNING:
                return self._elapsed_progress(obj, start=72, end=96, seconds=180)
            if obj.eventification_status == EventificationStatus.COMPLETED:
                return 100
            return 62
        return 0

    def _capture_only_progress(self, obj) -> int:
        """Progress when the diary is just recording: upload, transcribe, done."""
        if obj.status == SessionStatus.FAILED:
            return 100
        if obj.status == SessionStatus.QUEUED:
            return 5
        if obj.status in {SessionStatus.TRANSCRIBING, SessionStatus.PARSING}:
            return self._elapsed_progress(obj, start=15, end=95, seconds=90)
        return 100

    def _elapsed_progress(self, obj, *, start: int, end: int, seconds: int) -> int:
        elapsed = max((timezone.now() - obj.updated_at).total_seconds(), 0)
        ratio = min(elapsed / max(seconds, 1), 1.0)
        return round(start + ((end - start) * ratio))

    def get_processing_detail(self, obj):
        if not flags.nlp_enabled():
            if obj.status == SessionStatus.FAILED:
                return "Bu kayıt yazıya çevrilemedi."
            if obj.status == SessionStatus.QUEUED:
                return "Kaydın sırada bekliyor."
            if obj.status in {SessionStatus.TRANSCRIBING, SessionStatus.PARSING}:
                return "Sesin yazıya çevriliyor."
            return "Hazır."
        if obj.status == SessionStatus.FAILED:
            return obj.status_detail or obj.get_status_display()
        if obj.status in {
            SessionStatus.QUEUED,
            SessionStatus.TRANSCRIBING,
            SessionStatus.PARSING,
        }:
            return obj.status_detail or obj.get_status_display()
        if obj.eventification_status in {
            EventificationStatus.QUEUED,
            EventificationStatus.RUNNING,
            EventificationStatus.FAILED,
            EventificationStatus.COMPLETED,
        }:
            return obj.eventification_detail or obj.get_eventification_status_display()
        return obj.status_detail or obj.get_status_display()


class SessionDetailSerializer(SessionSerializer):
    mentions = MentionSerializer(many=True, read_only=True)

    class Meta(SessionSerializer.Meta):
        fields = SessionSerializer.Meta.fields + ["mentions"]


class SessionUploadSerializer(serializers.ModelSerializer):
    audio = serializers.FileField(required=False, allow_null=True, write_only=True)
    transcript = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = Session
        fields = [
            "client_uuid",
            "audio",
            "transcript",
            "duration_seconds",
            "recorded_at",
            "language",
        ]

    def create(self, validated_data):
        audio = validated_data.pop("audio", None)
        session = Session.objects.create(**validated_data)
        if audio:
            session.audio_file = audio
            session.save(update_fields=["audio_file", "updated_at"])
        return session


class ResolveMentionSerializer(serializers.Serializer):
    action = serializers.ChoiceField(choices=["ASSIGN", "NEW", "UNKNOWN", "IGNORE"])
    # For ASSIGN: existing node id
    node_id = serializers.IntegerField(required=False, allow_null=True)
    # For NEW: create a new node
    label = serializers.CharField(required=False, allow_blank=True)
    kind = serializers.CharField(required=False, allow_blank=True)
    time_value = serializers.CharField(required=False, allow_blank=True)
    notes = serializers.CharField(required=False, allow_blank=True)
