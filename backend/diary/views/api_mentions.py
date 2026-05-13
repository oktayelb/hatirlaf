from __future__ import annotations

from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..models import Mention, Node, NodeKind
from ..processing import entity_registry as entity_registry_mod
from ..serializers import MentionSerializer, ResolveMentionSerializer
from .api_shared import _default_kind_for, _rebuild_edges_for_session


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
            kind = _default_kind_for(mention)
            node = Node.unknown_for(kind)
            resolution_action = "IGNORED"

        mention.node = node
        mention.resolved = True
        mention.is_conflict = False
        mention.resolution_action = resolution_action
        mention.save(update_fields=["node", "resolved", "is_conflict", "resolution_action"])

        _rebuild_edges_for_session(mention.session)
        transaction.on_commit(lambda: entity_registry_mod.record_mention(mention))

        return Response(MentionSerializer(mention, context={"request": request}).data)
