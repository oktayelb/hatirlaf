from __future__ import annotations

from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from rest_framework import generics, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..models import Edge, Node, NodeKind
from ..processing import entity_registry as entity_registry_mod
from ..serializers import EdgeSerializer, NodeSerializer
from .api_shared import _entity_display_label, _node_memories_for


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

    def perform_create(self, serializer):
        node = serializer.save()
        transaction.on_commit(lambda: entity_registry_mod.record_node(node))

    def perform_update(self, serializer):
        node = serializer.save()
        transaction.on_commit(lambda: entity_registry_mod.record_node(node))

    @action(detail=False, methods=["get"], url_path="memories")
    def memories(self, request):
        kind = (request.query_params.get("kind") or "").strip().upper()
        label = (request.query_params.get("label") or "").strip()
        node_id = (request.query_params.get("node_id") or "").strip()

        valid_kinds = {choice[0] for choice in NodeKind.choices}
        if kind not in valid_kinds:
            return Response(
                {"detail": "kind must be a valid node kind"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not label and not node_id:
            return Response(
                {"detail": "label or node_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        node = None
        if node_id:
            node = get_object_or_404(Node, pk=node_id)
            kind = node.kind
            label = _entity_display_label(node.label)
        else:
            label = _entity_display_label(label)
            node = next(
                (
                    candidate
                    for candidate in Node.objects.filter(kind=kind)
                    if _entity_display_label(candidate.label).casefold() == label.casefold()
                ),
                None,
            )

        memories = _node_memories_for(kind=kind, label=label, node=node)
        total_mentions = sum(len(item["matched_mentions"]) for item in memories)
        total_events = sum(len(item["event_matches"]) for item in memories)

        node_payload = (
            NodeSerializer(node, context={"request": request}).data
            if node
            else {
                "id": None,
                "kind": kind,
                "kind_display": dict(NodeKind.choices).get(kind, kind),
                "label": label,
                "display_label": label,
                "aliases": [],
                "is_unknown": False,
                "time_value": "",
                "notes": "",
                "mention_count": total_mentions,
                "created_at": None,
                "updated_at": None,
            }
        )

        return Response(
            {
                "node": node_payload,
                "resolved": bool(node),
                "stats": {
                    "sessions": len(memories),
                    "mentions": total_mentions,
                    "events": total_events,
                },
                "memories": memories,
            }
        )


class EdgeListView(generics.ListAPIView):
    serializer_class = EdgeSerializer
    queryset = Edge.objects.select_related("source", "target", "session").all()
