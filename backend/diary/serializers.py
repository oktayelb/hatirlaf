from __future__ import annotations

from rest_framework import serializers

from .models import Edge, Mention, Node, Session


class NodeSerializer(serializers.ModelSerializer):
    kind_display = serializers.CharField(source="get_kind_display", read_only=True)
    mention_count = serializers.IntegerField(source="mentions.count", read_only=True)

    class Meta:
        model = Node
        fields = [
            "id",
            "kind",
            "kind_display",
            "label",
            "aliases",
            "is_unknown",
            "time_value",
            "notes",
            "mention_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "kind_display", "mention_count", "created_at", "updated_at"]


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
            "created_at",
        ]


class SessionSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    audio_url = serializers.SerializerMethodField()
    conflict_count = serializers.SerializerMethodField()
    mention_count = serializers.SerializerMethodField()

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
            "word_timings",
            "structured_events",
            "eventification_status",
            "eventification_detail",
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
            "conflict_count",
            "mention_count",
            "audio_url",
            "created_at",
            "updated_at",
        ]

    def get_audio_url(self, obj):
        if obj.audio_file:
            request = self.context.get("request")
            url = obj.audio_file.url
            return request.build_absolute_uri(url) if request else url
        return None

    def get_conflict_count(self, obj):
        return obj.mentions.filter(is_conflict=True, resolved=False).count()

    def get_mention_count(self, obj):
        return obj.mentions.count()


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
