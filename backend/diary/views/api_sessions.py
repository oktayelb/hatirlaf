from __future__ import annotations

import logging

from django.http import FileResponse, Http404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from ..models import EventificationStatus, Session, SessionStatus
from ..processing.pipeline import (
    is_eventification_active,
    is_processing_active,
    kickoff,
)
from ..serializers import SessionDetailSerializer, SessionSerializer, SessionUploadSerializer

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
        """Accept a new recording from the client sync queue."""
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
        """Delete a session and everything that hangs off it."""
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
