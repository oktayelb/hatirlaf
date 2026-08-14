from __future__ import annotations

import datetime as dt
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from diary.models import Edge, EncounteredEntity, Mention, MentionType, Node, NodeKind, Session, SessionStatus
from diary.processing import extractor, llm, nlp
from diary.processing.conflicts import detect_conflicts
from diary.services import session_pipeline


@override_settings(HATIRLAF_NLP_ENABLED=True)
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
            {
                "mood": "mutlu",
                "tags": ["arkadaşlar", "okul"],
            },
        )

        session.refresh_from_db()
        self.assertEqual(session.eventification_status, "completed")
        self.assertEqual(session.structured_events, [{"baslik": "Buluşma", "tarih": "2026-04-30"}])
        self.assertEqual(session.mood, "mutlu")
        self.assertEqual(session.tags, ["arkadaşlar", "okul"])
        self.assertEqual(session.mood_source, "ai")
        self.assertEqual(session.tags_source, "ai")
        self.assertIn("llm ile 1 olay üretildi", session.eventification_detail)

    def test_persist_eventification_result_preserves_manual_mood_and_tags(self):
        recorded_at = dt.datetime(2026, 4, 30, 12, 0)
        session = Session.objects.create(
            client_uuid="svc-manual-labels",
            recorded_at=recorded_at,
            transcript="Okulda sınavım vardı.",
            status=SessionStatus.COMPLETED,
            mood="sakin",
            mood_source="manual",
            tags=["kişisel"],
            tags_source="manual",
        )
        extraction = extractor.extract(session.transcript, recorded_at)

        session_pipeline.persist_eventification_result(
            session,
            extraction,
            {"backend": "llm", "olay_loglari": []},
            {"mood": "stresli", "tags": ["okul"]},
        )

        session.refresh_from_db()
        self.assertEqual(session.mood, "sakin")
        self.assertEqual(session.tags, ["kişisel"])
        self.assertEqual(session.mood_source, "manual")
        self.assertEqual(session.tags_source, "manual")

    def test_reextract_all_endpoint_rebuilds_mentions_from_transcripts(self):
        recorded_at = dt.datetime(2026, 4, 30, 12, 0, tzinfo=dt.timezone.utc)
        session = Session.objects.create(
            client_uuid="bulk-1",
            recorded_at=recorded_at,
            transcript="Fatihle buluştum.",
            status=SessionStatus.COMPLETED,
        )
        empty = Session.objects.create(
            client_uuid="bulk-empty",
            recorded_at=recorded_at,
            transcript="",
            status=SessionStatus.COMPLETED,
        )
        stale_node = Node.objects.create(kind=NodeKind.PERSON, label="Fatihle")
        fresh_node = Node.objects.create(kind=NodeKind.PERSON, label="Fatih")
        Mention.objects.create(
            session=session,
            surface="Fatihle",
            lemma="fatihle",
            char_start=0,
            char_end=7,
            mention_type=MentionType.PERSON,
            node=stale_node,
            resolved=True,
        )
        Edge.objects.create(
            session=session,
            source=stale_node,
            target=fresh_node,
            relation=Edge.Relation.MENTIONED_WITH,
        )
        EncounteredEntity.objects.create(kind=NodeKind.PERSON, label="fatihle")

        with (
            patch("diary.pipeline.runner.kickoff_eventification"),
            self.captureOnCommitCallbacks(execute=True),
        ):
            response = self.client.post(reverse("session-reextract-all"))

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["processed"], 1)
        self.assertEqual(response.json()["skipped"], 1)
        self.assertFalse(empty.mentions.exists())

        session.refresh_from_db()
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.eventification_status, "completed")
        self.assertFalse(session.edges.exists())
        mentions = list(session.mentions.values_list("surface", "lemma", "mention_type"))
        self.assertEqual(mentions, [("Fatih", "fatih", MentionType.PERSON)])
        labels = set(EncounteredEntity.objects.values_list("kind", "label"))
        self.assertIn((NodeKind.PERSON, "fatih"), labels)
        self.assertNotIn((NodeKind.PERSON, "fatihle"), labels)

    @patch("diary.pipeline.runner.kickoff_eventification")
    @patch("diary.pipeline.stages.tx_mod.transcribe")
    def test_reprocess_endpoint_uses_saved_transcript_instead_of_retranscribing_audio(
        self,
        transcribe,
        kickoff_eventification,
    ):
        recorded_at = dt.datetime(2026, 4, 30, 12, 0, tzinfo=dt.timezone.utc)
        session = Session.objects.create(
            client_uuid="manual-stt-edit",
            recorded_at=recorded_at,
            audio_file="sessions/manual-stt-edit.m4a",
            transcript="Eski transkript.",
            status=SessionStatus.COMPLETED,
        )
        response = self.client.patch(
            reverse("session-detail", kwargs={"pk": session.pk}),
            data={"transcript": "Yarın Ahmet ile buluşacağım."},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        with self.settings(HATIRLAF_SYNC_PROCESSING=True):
            response = self.client.post(reverse("session-process", kwargs={"pk": session.pk}))

        self.assertEqual(response.status_code, 200)
        transcribe.assert_not_called()
        kickoff_eventification.assert_called_once_with(session.pk)

        session.refresh_from_db()
        self.assertEqual(session.transcript, "Yarın Ahmet ile buluşacağım.")
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertEqual(session.eventification_status, "queued")
        mention_surfaces = set(session.mentions.values_list("surface", flat=True))
        self.assertIn("Ahmet", mention_surfaces)
        self.assertNotIn("Eski", mention_surfaces)

    @patch("diary.processing.llm._load_llm", return_value=None)
    def test_mood_tag_fallback_classifies_school_and_stress(self, _load_llm):
        recorded_at = dt.datetime(2026, 4, 30, 12, 0, tzinfo=dt.timezone.utc)
        extraction = extractor.extract("Yarın okulda sınavım var, biraz stresliyim.", recorded_at)
        events = [
            {
                "zaman_dilimi": "Gelecek",
                "tarih": "2026-05-01",
                "olay": "Okulda sınavım var.",
                "kisiler": ["Ben"],
            }
        ]

        enrichment = llm.run_mood_tags(extraction, events)
        enriched_events = llm.merge_mood_tags(events, enrichment)

        self.assertEqual(enrichment["mood"], "stresli")
        self.assertIn("okul", enrichment["tags"])
        self.assertEqual(enriched_events[0]["kategori"], "okul")
        self.assertIn("okul", enriched_events[0]["etiketler"])
