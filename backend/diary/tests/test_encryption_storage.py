from __future__ import annotations

import datetime as dt
import os
import tempfile

from django.core.files.base import ContentFile
from django.db import connection
from django.test import TestCase, override_settings
from django.utils import timezone

from diary.encryption import FILE_PREFIX, TEXT_PREFIX
from diary.models import EncounteredEntity, Mention, MentionType, NodeKind, Session
from diary.processing import entity_registry


@override_settings(HATIRLAF_PRELOAD_MODELS=False)
class EncryptionStorageTests(TestCase):
    def test_session_sensitive_fields_are_encrypted_in_database(self):
        session = Session.objects.create(
            client_uuid="encrypted-session-fields",
            recorded_at=timezone.make_aware(dt.datetime(2026, 5, 26, 10, 0)),
            transcript="Bugün gizli bir günlük yazdım.",
            processed_text="bugün gizli bir günlük yazdım",
            word_timings=[{"word": "Bugün", "start": 0, "end": 1}],
            structured_events=[{"olay": "Gizli günlük", "tarih": "2026-05-26"}],
            nlp_hints={"clauses": [{"text": "Bugün gizli bir günlük yazdım."}]},
        )

        refreshed = Session.objects.get(pk=session.pk)
        self.assertEqual(refreshed.transcript, "Bugün gizli bir günlük yazdım.")
        self.assertEqual(refreshed.word_timings[0]["word"], "Bugün")
        self.assertEqual(refreshed.structured_events[0]["olay"], "Gizli günlük")

        with connection.cursor() as cursor:
            cursor.execute(
                "select transcript, word_timings, structured_events, nlp_hints "
                "from diary_session where id = %s",
                [session.pk],
            )
            row = cursor.fetchone()

        self.assertTrue(row[0].startswith(TEXT_PREFIX))
        self.assertTrue(row[1].startswith(TEXT_PREFIX))
        self.assertTrue(row[2].startswith(TEXT_PREFIX))
        self.assertTrue(row[3].startswith(TEXT_PREFIX))
        self.assertNotIn("gizli", row[0])

    def test_mentions_and_encountered_registry_are_encrypted_in_database(self):
        session = Session.objects.create(
            client_uuid="encrypted-mention-fields",
            recorded_at=timezone.make_aware(dt.datetime(2026, 5, 26, 10, 30)),
        )
        mention = Mention.objects.create(
            session=session,
            surface="Ayşe",
            lemma="ayşe",
            char_start=0,
            char_end=4,
            mention_type=MentionType.PERSON,
            conflict_hint="Belirsiz kişi",
        )
        entity = EncounteredEntity.objects.create(kind=NodeKind.PERSON, label="ayşe")

        self.assertEqual(Mention.objects.get(pk=mention.pk).surface, "Ayşe")
        self.assertEqual(EncounteredEntity.objects.get(pk=entity.pk).label, "ayşe")

        with connection.cursor() as cursor:
            cursor.execute("select surface, conflict_hint from diary_mention where id = %s", [mention.pk])
            mention_row = cursor.fetchone()
            cursor.execute("select label from diary_encounteredentity where id = %s", [entity.pk])
            entity_row = cursor.fetchone()

        self.assertTrue(mention_row[0].startswith(TEXT_PREFIX))
        self.assertTrue(mention_row[1].startswith(TEXT_PREFIX))
        self.assertTrue(entity_row[0].startswith(TEXT_PREFIX))

    def test_entity_registry_dedupes_encrypted_labels(self):
        first = entity_registry.record_entity(NodeKind.PERSON, "Ayşe")
        second = entity_registry.record_entity(NodeKind.PERSON, "Ayşe")

        self.assertEqual(first.pk, second.pk)
        self.assertEqual(EncounteredEntity.objects.count(), 1)
        self.assertEqual(entity_registry.snapshot()["people"], ["ayşe"])

    def test_uploaded_media_is_encrypted_on_disk_and_decrypted_on_open(self):
        with tempfile.TemporaryDirectory() as media_root:
            with override_settings(MEDIA_ROOT=media_root):
                session = Session.objects.create(
                    client_uuid="encrypted-audio-file",
                    recorded_at=timezone.make_aware(dt.datetime(2026, 5, 26, 11, 0)),
                )
                session.audio_file.save("sample.webm", ContentFile(b"plain audio bytes"), save=True)

                with open(session.audio_file.path, "rb") as raw_file:
                    raw = raw_file.read()

                self.assertTrue(raw.startswith(FILE_PREFIX))
                self.assertNotIn(b"plain audio bytes", raw)

                with session.audio_file.open("rb") as decrypted:
                    self.assertEqual(decrypted.read(), b"plain audio bytes")

                self.assertFalse(os.path.exists(session.audio_file.path + ".tmp"))
