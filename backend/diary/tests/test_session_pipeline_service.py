from __future__ import annotations

import datetime as dt

from django.test import TestCase

from diary.models import Session, SessionStatus
from diary.processing import extractor, nlp
from diary.processing.conflicts import detect_conflicts
from diary.services import session_pipeline


class SessionPipelineServiceTests(TestCase):
    def test_persist_parsing_result_creates_mentions_and_completes_session(self):
        recorded_at = dt.datetime(2026, 4, 30, 12, 0)
        text = "Ayşe Kadıköyde buluştu."
        session = Session.objects.create(
            client_uuid="svc-1",
            recorded_at=recorded_at,
            transcript=text,
            status=SessionStatus.PARSING,
        )

        extraction = extractor.extract(text, recorded_at)
        parsed = extraction.parse or nlp.analyze(text)
        flagged = detect_conflicts(parsed.mentions, text, recorded_at)

        conflict_count = session_pipeline.persist_parsing_result(session, extraction, parsed, flagged)

        session.refresh_from_db()
        self.assertEqual(conflict_count, sum(1 for fm in flagged if fm.is_conflict))
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.processed_text, parsed.lemma_text)
        self.assertEqual(session.mentions.count(), len(flagged))
        self.assertEqual(session.eventification_status, "queued")

    def test_persist_eventification_result_updates_structured_events(self):
        recorded_at = dt.datetime(2026, 4, 30, 12, 0)
        text = "Ayşe Kadıköyde buluştu."
        session = Session.objects.create(
            client_uuid="svc-2",
            recorded_at=recorded_at,
            transcript=text,
            status=SessionStatus.COMPLETED,
        )

        extraction = extractor.extract(text, recorded_at)
        session_pipeline.persist_eventification_result(
            session,
            extraction,
            {
                "backend": "llm",
                "olay_loglari": [{"baslik": "Buluşma", "tarih": "2026-04-30"}],
            },
        )

        session.refresh_from_db()
        self.assertEqual(session.eventification_status, "completed")
        self.assertEqual(session.structured_events, [{"baslik": "Buluşma", "tarih": "2026-04-30"}])
        self.assertIn("llm ile 1 olay üretildi", session.eventification_detail)
