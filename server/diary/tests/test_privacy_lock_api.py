from __future__ import annotations

import datetime as dt

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from diary.models import PrivacySettings, Session


@override_settings(HATIRLAF_PRELOAD_MODELS=False)
class PrivacyLockApiTests(TestCase):
    def test_api_data_is_blocked_until_app_password_is_unlocked(self):
        settings = PrivacySettings.load()
        settings.set_password("secret123")

        response = self.client.get(reverse("session-list"))
        self.assertEqual(response.status_code, 423)
        self.assertTrue(response.json()["locked"])

        response = self.client.post(
            reverse("privacy-unlock"),
            {"password": "wrong"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            reverse("privacy-unlock"),
            {"password": "secret123"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["unlocked"])

        response = self.client.get(reverse("session-list"))
        self.assertEqual(response.status_code, 200)

    def test_media_paths_are_blocked_when_locked(self):
        settings = PrivacySettings.load()
        settings.set_password("secret123")

        response = self.client.get("/media/sessions/private.webm")

        self.assertEqual(response.status_code, 423)
        self.assertTrue(response.json()["locked"])

    def test_session_audio_urls_use_locked_api_endpoint(self):
        session = Session.objects.create(
            client_uuid="privacy-audio-url",
            recorded_at=timezone.make_aware(dt.datetime(2026, 5, 26, 9, 30)),
            audio_file="sessions/private.webm",
        )

        response = self.client.get(reverse("session-list"))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.json()[0]["audio_url"].endswith(
                reverse("session-audio", kwargs={"pk": session.pk})
            )
        )

    def test_password_can_be_added_and_removed(self):
        response = self.client.get(reverse("privacy-status"))
        self.assertEqual(response.json(), {"password_enabled": False, "unlocked": True})

        response = self.client.post(
            reverse("privacy-set-password"),
            {"new_password": "secret123"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["password_enabled"])

        response = self.client.post(reverse("privacy-lock"), {}, content_type="application/json")
        self.assertEqual(response.status_code, 200)

        response = self.client.get(reverse("session-list"))
        self.assertEqual(response.status_code, 423)

        self.client.post(
            reverse("privacy-unlock"),
            {"password": "secret123"},
            content_type="application/json",
        )
        response = self.client.post(
            reverse("privacy-clear-password"),
            {"current_password": "secret123"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["password_enabled"])

        response = self.client.get(reverse("session-list"))
        self.assertEqual(response.status_code, 200)
