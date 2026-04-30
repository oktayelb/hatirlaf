from __future__ import annotations

import datetime as dt

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from diary.models import EventificationStatus, Session, SessionStatus
from diary.processing import extractor, llm


@override_settings(HATIRLAF_PRELOAD_MODELS=False)
class CalendarApiTests(TestCase):
    def test_completed_session_appears_while_eventification_is_still_running(self):
        Session.objects.create(
            client_uuid="calendar-running-eventification",
            recorded_at=timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15)),
            status=SessionStatus.COMPLETED,
            transcript="Bugün okula gittim.",
            eventification_status=EventificationStatus.RUNNING,
            eventification_detail="LLM olaylaştırması çalışıyor.",
            structured_events=[],
        )

        response = self.client.get(reverse("calendar"), {"month": "2026-04"})

        self.assertEqual(response.status_code, 200)
        day_events = response.json()["days"]["2026-04-30"]
        self.assertEqual(len(day_events), 1)
        self.assertEqual(day_events[0]["saat"], "08:15")
        self.assertEqual(day_events[0]["olay"], "Bugün okula gittim.")

    def test_running_eventification_uses_nlp_hint_dates(self):
        recorded_at = timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15))
        text = "Dün çalıştım"
        Session.objects.create(
            client_uuid="calendar-running-relative-date",
            recorded_at=recorded_at,
            status=SessionStatus.COMPLETED,
            transcript=text,
            eventification_status=EventificationStatus.RUNNING,
            structured_events=[],
            nlp_hints=extractor.extract(text, recorded_at).to_json(),
        )

        response = self.client.get(reverse("calendar"), {"month": "2026-04"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("2026-04-29", response.json()["days"])
        self.assertNotIn("2026-04-30", response.json()["days"])
        day_events = response.json()["days"]["2026-04-29"]
        self.assertEqual(day_events[0]["olay"], "Dün çalıştım")
        self.assertEqual(day_events[0]["zaman_dilimi"], "Geçmiş")

    def test_running_eventification_uses_full_clause_text_for_nlp_hint_events(self):
        recorded_at = timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15))
        text = "Dün işteydim, bugün ise erken kalktım ve okula gitmeyi düşünüyorum."
        Session.objects.create(
            client_uuid="calendar-running-full-clause",
            recorded_at=recorded_at,
            status=SessionStatus.COMPLETED,
            transcript=text,
            eventification_status=EventificationStatus.RUNNING,
            structured_events=[],
            nlp_hints=extractor.extract(text, recorded_at).to_json(),
        )

        response = self.client.get(reverse("calendar"), {"month": "2026-04"})

        self.assertEqual(response.status_code, 200)
        day_events = response.json()["days"]["2026-04-29"]
        self.assertEqual(day_events[0]["olay"], text)

    def test_nlp_only_eventification_uses_full_clause_text(self):
        recorded_at = timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15))
        text = "Dün işteydim, bugün ise erken kalktım ve okula gitmeyi düşünüyorum."

        events = llm._fallback_from_hints(extractor.extract(text, recorded_at))["olay_loglari"]

        self.assertEqual(events[0]["tarih"], "2026-04-29")
        self.assertEqual(events[0]["olay"], text)

    def test_calendar_expands_old_short_structured_event_text_from_hints(self):
        recorded_at = timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15))
        text = "Dün işteydim, bugün ise erken kalktım ve okula gitmeyi düşünüyorum."
        Session.objects.create(
            client_uuid="calendar-old-short-structured",
            recorded_at=recorded_at,
            status=SessionStatus.COMPLETED,
            transcript=text,
            eventification_status=EventificationStatus.COMPLETED,
            nlp_hints=extractor.extract(text, recorded_at).to_json(),
            structured_events=[
                {
                    "zaman_dilimi": "Geçmiş",
                    "tarih": "2026-04-29",
                    "saat": "",
                    "lokasyon": "",
                    "olay": "erken kalktım ve okula gitmeyi düşünüyorum",
                    "kisiler": ["Ben"],
                }
            ],
        )

        response = self.client.get(reverse("calendar"), {"month": "2026-04"})

        self.assertEqual(response.status_code, 200)
        day_events = response.json()["days"]["2026-04-29"]
        self.assertEqual(day_events[0]["olay"], text)

    def test_structured_events_are_used_when_eventification_completed(self):
        Session.objects.create(
            client_uuid="calendar-completed-eventification",
            recorded_at=timezone.make_aware(dt.datetime(2026, 4, 30, 8, 15)),
            status=SessionStatus.COMPLETED,
            transcript="Bugün okula gittim.",
            eventification_status=EventificationStatus.COMPLETED,
            structured_events=[
                {
                    "zaman_dilimi": "Gelecek",
                    "tarih": "2026-05-01",
                    "saat": "10:30",
                    "lokasyon": "Okul",
                    "olay": "Toplantıya gideceğim.",
                    "kisiler": ["Ben"],
                }
            ],
        )

        response = self.client.get(reverse("calendar"), {"month": "2026-05"})

        self.assertEqual(response.status_code, 200)
        day_events = response.json()["days"]["2026-05-01"]
        self.assertEqual(len(day_events), 1)
        self.assertEqual(day_events[0]["saat"], "10:30")
        self.assertEqual(day_events[0]["olay"], "Toplantıya gideceğim.")
