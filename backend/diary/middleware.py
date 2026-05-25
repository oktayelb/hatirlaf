from __future__ import annotations

from django.db import DatabaseError, OperationalError, ProgrammingError
from django.http import JsonResponse

from .models import PrivacySettings


class PrivacyLockMiddleware:
    """Gate local diary API data behind the optional app password."""

    EXEMPT_PATHS = {
        "/api/health/",
        "/api/privacy/status/",
        "/api/privacy/unlock/",
    }

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if self._should_block(request):
            return JsonResponse(
                {"detail": "Uygulama kilitli.", "locked": True},
                status=423,
            )
        return self.get_response(request)

    def _should_block(self, request) -> bool:
        path = request.path_info
        if path in self.EXEMPT_PATHS:
            return False
        if not (path.startswith("/api/") or path.startswith("/media/")):
            return False
        try:
            settings = PrivacySettings.load()
        except (DatabaseError, OperationalError, ProgrammingError):
            return False
        if not settings.password_enabled:
            return False
        return not bool(request.session.get("privacy_unlocked"))
