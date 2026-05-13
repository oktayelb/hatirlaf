from __future__ import annotations

import datetime as dt

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from diary.models import EventificationStatus, Mention, MentionType, Session, SessionStatus


@override_settings(HATIRLAF_PRELOAD_MODELS=False)
class NodeMemoriesApiTests(TestCase):
    def test_memories_page_includes_unresolved_person_mentions(self):
        session = Session.objects.create(
            client_uuid="memories-unresolved-person",
            recorded_at=timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15)),
            status=SessionStatus.COMPLETED,
            transcript="Ayşe ile yürüdüm.",
            eventification_status=EventificationStatus.COMPLETED,
        )
        Mention.objects.create(
            session=session,
            surface="Ayşe",
            char_start=0,
            char_end=4,
            mention_type=MentionType.PERSON,
            resolved=False,
        )

        response = self.client.get(reverse("node-memories"), {"kind": "PERSON", "label": "Ayşe"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["resolved"])
        self.assertEqual(payload["stats"]["sessions"], 1)
        self.assertEqual(payload["stats"]["mentions"], 1)
        self.assertEqual(payload["memories"][0]["matched_mentions"][0]["surface"], "Ayşe")

    def test_memories_page_includes_structured_event_matches_for_places(self):
        Session.objects.create(
            client_uuid="memories-structured-place",
            recorded_at=timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15)),
            status=SessionStatus.COMPLETED,
            transcript="Kadıköy'de buluştuk.",
            eventification_status=EventificationStatus.COMPLETED,
            structured_events=[
                {
                    "zaman_dilimi": "Şu An",
                    "tarih": "2026-04-30",
                    "saat": "08:15",
                    "lokasyon": "Kadıköy",
                    "olay": "Kadıköy'de buluştuk.",
                    "kisiler": ["Ben", "Ayşe"],
                }
            ],
        )

        response = self.client.get(reverse("node-memories"), {"kind": "LOCATION", "label": "Kadıköy"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["resolved"], False)
        self.assertEqual(payload["stats"]["sessions"], 1)
        self.assertEqual(payload["stats"]["events"], 1)
        self.assertEqual(payload["memories"][0]["event_matches"][0]["place"], "Kadıköy")
