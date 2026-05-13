from __future__ import annotations

import datetime as dt

from django.test import TestCase, override_settings

from diary.models import EncounteredEntity, Mention, MentionType, Node, NodeKind, Session
from diary.processing import entity_registry


@override_settings(HATIRLAF_USE_BERTURK=False, HATIRLAF_USE_SAVYAR=True)
class EntityRegistryTests(TestCase):
    def test_registry_persists_bare_root_people_and_places(self):
        session = Session.objects.create(
            client_uuid="registry-test",
            recorded_at=dt.datetime(2026, 4, 30, 9, 15, tzinfo=dt.timezone(dt.timedelta(hours=3))),
            transcript="Yiğitle Kadıköy'de buluştum.",
        )
        ayse = Node.objects.create(kind=NodeKind.PERSON, label="Ayşe")
        kadikoy = Node.objects.create(kind=NodeKind.LOCATION, label="Kadıköy")
        Mention.objects.create(
            session=session,
            surface="Yiğitle",
            lemma="yiğit",
            char_start=0,
            char_end=7,
            mention_type=MentionType.PERSON,
        )
        Mention.objects.create(
            session=session,
            surface="Kadıköy'de",
            lemma="kadıköy",
            char_start=8,
            char_end=18,
            mention_type=MentionType.LOCATION,
        )

        entity_registry.record_node(ayse)
        entity_registry.record_node(kadikoy)
        entity_registry.record_session(session.id)

        payload = entity_registry.snapshot()
        labels = {(row.kind, row.label) for row in EncounteredEntity.objects.all()}

        self.assertIn((NodeKind.PERSON, "ayşe"), labels)
        self.assertIn((NodeKind.PERSON, "yiğit"), labels)
        self.assertIn((NodeKind.LOCATION, "kadıköy"), labels)
        self.assertEqual(payload["people"], ["ayşe", "yiğit"])
        self.assertEqual(payload["places"], ["kadıköy"])
