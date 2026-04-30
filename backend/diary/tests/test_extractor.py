from __future__ import annotations

import datetime as dt

from django.test import SimpleTestCase, override_settings

from diary.processing import extractor, nlp
from diary.processing import llm as llm_mod


class LlmLifecycleTests(SimpleTestCase):
    def test_close_cached_llm_releases_cached_model(self):
        class FakeLlama:
            closed = False

            def close(self):
                self.closed = True

        fake = FakeLlama()
        old_llm = llm_mod._cached_llm
        old_path = llm_mod._cached_path
        try:
            llm_mod._cached_llm = fake
            llm_mod._cached_path = "/tmp/model.gguf"

            llm_mod.close_cached_llm()

            self.assertTrue(fake.closed)
            self.assertIsNone(llm_mod._cached_llm)
            self.assertIsNone(llm_mod._cached_path)
        finally:
            llm_mod._cached_llm = old_llm
            llm_mod._cached_path = old_path


@override_settings(HATIRLAF_USE_BERTURK=False)
class TurkishNlpPrepassTests(SimpleTestCase):
    def test_referential_pronouns_are_preserved_as_references(self):
        text = "Ayşe ile dün Kadıköy'de buluştuk. Oradaki konuşma önemliydi, bu yarınki planı etkiledi."
        anchor = dt.datetime(2026, 4, 30, 12, 0)

        result = extractor.extract(text, anchor)

        self.assertIn("Ayşe", result.persons)
        self.assertIn("Kadıköy", result.locations)
        self.assertIn("dün", result.references)
        self.assertIn("Oradaki", result.references)
        self.assertIn("bu", result.references)
        self.assertIn("yarınki", result.references)

        second_clause = result.clauses[1]
        self.assertEqual(second_clause.date_iso, "2026-05-01")
        self.assertIn("Oradaki", second_clause.references)
        self.assertIn("yarınki", second_clause.references)

    def test_relative_time_adjectives_are_grounded(self):
        anchor = dt.datetime(2026, 4, 30, 9, 15)

        yesterday = extractor.extract("Dünkü toplantıda Emre ile konuştum.", anchor)
        tomorrow = extractor.extract("Yarınki randevu saat 14:30'da.", anchor)

        self.assertEqual(yesterday.clauses[0].date_iso, "2026-04-29")
        self.assertEqual(yesterday.clauses[0].zaman_dilimi, "Geçmiş")
        self.assertEqual(tomorrow.clauses[0].date_iso, "2026-05-01")
        self.assertEqual(tomorrow.clauses[0].time_hm, "14:30")
        self.assertEqual(tomorrow.clauses[0].zaman_dilimi, "Gelecek")

    def test_nlp_layer_recognizes_requested_pronoun_forms(self):
        parsed = nlp.analyze("O bunun şunun oradaki buradaki dünkü yarınki etkisini anlattı.")
        mentions = {(m.surface.lower(), m.mention_type) for m in parsed.mentions}

        self.assertIn(("o", "PRONOUN"), mentions)
        self.assertIn(("bunun", "PRONOUN"), mentions)
        self.assertIn(("şunun", "PRONOUN"), mentions)
        self.assertIn(("oradaki", "PRONOUN"), mentions)
        self.assertIn(("buradaki", "PRONOUN"), mentions)
        self.assertIn(("dünkü", "TIME"), mentions)
        self.assertIn(("yarınki", "TIME"), mentions)

    def test_clause_subject_is_inferred_from_verb_conjugation(self):
        text = "Dün gittim. Bugün konuşuyoruz. Yarın gelecekler. Gitmelisin."
        anchor = dt.datetime(2026, 4, 30, 9, 15)

        result = extractor.extract(text, anchor)

        self.assertEqual(result.clauses[0].subject_person, "1sg")
        self.assertEqual(result.clauses[0].subject_pronoun, "Ben")
        self.assertEqual(result.clauses[0].subject_verb, "gittim")
        self.assertEqual(result.clauses[0].subject_tense, "definite_past")
        self.assertEqual(result.clauses[1].subject_person, "1pl")
        self.assertEqual(result.clauses[1].subject_pronoun, "Biz")
        self.assertEqual(result.clauses[1].subject_tense, "present_continuous")
        self.assertEqual(result.clauses[2].subject_person, "3pl")
        self.assertEqual(result.clauses[2].subject_pronoun, "Onlar")
        self.assertEqual(result.clauses[2].subject_tense, "future")
        self.assertEqual(result.clauses[3].subject_person, "2sg")
        self.assertEqual(result.clauses[3].subject_pronoun, "Sen")
        self.assertEqual(result.clauses[3].subject_tense, "necessitative")
