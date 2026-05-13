from __future__ import annotations

import datetime as dt

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from diary.models import EventificationStatus, Mention, MentionType, Node, NodeKind, Session, SessionStatus


@override_settings(HATIRLAF_PRELOAD_MODELS=False)
class RecapApiTests(TestCase):
    def test_recap_rolls_up_monthly_memory_digest(self):
        recorded_at = timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15))
        session = Session.objects.create(
            client_uuid="recap-monthly-digest",
            recorded_at=recorded_at,
            status=SessionStatus.COMPLETED,
            transcript="Ayşe ile Kadıköy'de yürüdüm. Yarın Ahmet ile okula gideceğim.",
            eventification_status=EventificationStatus.COMPLETED,
            structured_events=[
                {
                    "zaman_dilimi": "Şu An",
                    "tarih": "2026-04-30",
                    "saat": "08:15",
                    "lokasyon": "Kadıköy",
                    "olay": "Ayşe ile yürüdüm.",
                    "kisiler": ["Ben", "Ayşe"],
                }
            ],
        )
        ayse = Node.objects.create(kind=NodeKind.PERSON, label="Ayşe")
        kadikoy = Node.objects.create(kind=NodeKind.LOCATION, label="Kadıköy")
        Mention.objects.create(
            session=session,
            surface="Ayşe",
            char_start=0,
            char_end=4,
            mention_type=MentionType.PERSON,
            node=ayse,
            resolved=True,
        )
        Mention.objects.create(
            session=session,
            surface="Kadıköy",
            char_start=10,
            char_end=17,
            mention_type=MentionType.LOCATION,
            node=kadikoy,
            resolved=True,
        )

        response = self.client.get(reverse("recap"), {"month": "2026-04"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["month"], "2026-04")
        self.assertEqual(payload["stats"]["sessions"], 1)
        self.assertEqual(payload["stats"]["events"], 1)
        self.assertEqual(payload["top_people"][0]["label"], "Ayşe")
        self.assertEqual(payload["top_places"][0]["label"], "Kadıköy")
        self.assertEqual(payload["highlights"][0]["text"], "Ayşe ile yürüdüm.")
