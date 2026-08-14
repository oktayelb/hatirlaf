"""The NLP switch: what disappears when it is off, what returns when it is on."""

from __future__ import annotations

import datetime as dt

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from diary.models import EventificationStatus, Session, SessionStatus
from diary.pipeline import stages


@override_settings(HATIRLAF_PRELOAD_MODELS=False, HATIRLAF_NLP_ENABLED=False)
class NlpDisabledTests(TestCase):
    def test_config_endpoint_reports_the_switch(self):
        response = self.client.get(reverse("config"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"features": {"nlp": False}, "nlp_enabled": False})

    def test_understanding_endpoints_are_not_served(self):
        for name in ("timeline", "calendar", "recap", "graph", "edge-list"):
            with self.subTest(endpoint=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 404)
        for name in ("node-list", "mention-list"):
            with self.subTest(endpoint=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 404)

    def test_capture_endpoints_still_work(self):
        self.assertEqual(self.client.get(reverse("session-list")).status_code, 200)
        self.assertEqual(self.client.get(reverse("health")).status_code, 200)

    def test_session_payload_carries_no_analysis_fields(self):
        session = Session.objects.create(
            client_uuid="flag-off-1",
            recorded_at=timezone.make_aware(dt.datetime(2026, 4, 30, 9, 0)),
            transcript="Bugün Ayşe ile Kadıköyde yürüdük.",
            status=SessionStatus.COMPLETED,
        )

        payload = self.client.get(
            reverse("session-detail", kwargs={"pk": session.pk})
        ).json()

        self.assertEqual(payload["transcript"], "Bugün Ayşe ile Kadıköyde yürüdük.")
        for field in (
            "structured_events",
            "mentions",
            "mention_count",
            "conflict_count",
            "eventification_status",
            "mood",
            "tags",
            "processed_text",
            "word_timings",
        ):
            with self.subTest(field=field):
                self.assertNotIn(field, payload)

    def test_pipeline_drops_its_understanding_stages(self):
        keys = [s.key for s in stages.active_stages(deferred=False)]
        self.assertEqual(keys, ["transcribe", "archive"])
        self.assertEqual(stages.active_stages(deferred=True), [])

    def test_text_entry_is_archived_without_analysis(self):
        with self.settings(HATIRLAF_SYNC_PROCESSING=True):
            response = self.client.post(
                reverse("session-list"),
                data={
                    "client_uuid": "flag-off-upload",
                    "recorded_at": "2026-04-30T09:00:00Z",
                    "transcript": "Yarın Ahmet ile buluşacağım.",
                    "duration_seconds": 0,
                    "language": "tr",
                },
            )
        self.assertEqual(response.status_code, 201)

        session = Session.objects.get(client_uuid="flag-off-upload")
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.status_detail, "Günlüğe kaydedildi.")
        self.assertEqual(session.eventification_status, EventificationStatus.NOT_STARTED)
        self.assertFalse(session.mentions.exists())
        self.assertEqual(session.structured_events, [])

    def test_bulk_reextract_declines_while_the_switch_is_off(self):
        response = self.client.post(reverse("session-reextract-all"))
        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.json()["started"])


@override_settings(HATIRLAF_PRELOAD_MODELS=False, HATIRLAF_NLP_ENABLED=True)
class NlpEnabledTests(TestCase):
    def test_config_endpoint_reports_the_switch(self):
        self.assertTrue(self.client.get(reverse("config")).json()["nlp_enabled"])

    def test_understanding_endpoints_come_back(self):
        for name in ("timeline", "calendar", "recap", "graph", "edge-list", "node-list"):
            with self.subTest(endpoint=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_pipeline_regains_its_understanding_stages(self):
        keys = [s.key for s in stages.active_stages(deferred=False)]
        self.assertEqual(keys, ["transcribe", "understand"])
        self.assertEqual([s.key for s in stages.active_stages(deferred=True)], ["eventify"])

    def test_session_payload_carries_analysis_fields(self):
        session = Session.objects.create(
            client_uuid="flag-on-1",
            recorded_at=timezone.make_aware(dt.datetime(2026, 4, 30, 9, 0)),
            transcript="Bugün Ayşe ile Kadıköyde yürüdük.",
            status=SessionStatus.COMPLETED,
        )
        payload = self.client.get(
            reverse("session-detail", kwargs={"pk": session.pk})
        ).json()
        self.assertIn("structured_events", payload)
        self.assertIn("mentions", payload)
