from __future__ import annotations

from rest_framework.decorators import api_view
from rest_framework.response import Response

from ..models import PrivacySettings


MIN_PASSWORD_LENGTH = 6


@api_view(["GET"])
def privacy_status_view(request):
    settings = PrivacySettings.load()
    return Response(
        {
            "password_enabled": settings.password_enabled,
            "unlocked": not settings.password_enabled or bool(request.session.get("privacy_unlocked")),
        }
    )


@api_view(["POST"])
def privacy_unlock_view(request):
    settings = PrivacySettings.load()
    if not settings.password_enabled:
        request.session["privacy_unlocked"] = True
        return Response({"unlocked": True, "password_enabled": False})

    password = str(request.data.get("password") or "")
    if not settings.password_matches(password):
        return Response({"detail": "Parola yanlış."}, status=400)

    request.session.cycle_key()
    request.session["privacy_unlocked"] = True
    return Response({"unlocked": True, "password_enabled": True})


@api_view(["POST"])
def privacy_lock_view(request):
    request.session.pop("privacy_unlocked", None)
    return Response({"unlocked": False})


@api_view(["POST"])
def privacy_set_password_view(request):
    settings = PrivacySettings.load()
    new_password = str(request.data.get("new_password") or "")
    current_password = str(request.data.get("current_password") or "")

    if len(new_password) < MIN_PASSWORD_LENGTH:
        return Response(
            {"detail": f"Parola en az {MIN_PASSWORD_LENGTH} karakter olmalı."},
            status=400,
        )

    if settings.password_enabled and not settings.password_matches(current_password):
        return Response({"detail": "Mevcut parola yanlış."}, status=400)

    settings.set_password(new_password)
    request.session["privacy_unlocked"] = True
    return Response({"password_enabled": True, "unlocked": True})


@api_view(["POST"])
def privacy_clear_password_view(request):
    settings = PrivacySettings.load()
    if not settings.password_enabled:
        request.session.pop("privacy_unlocked", None)
        return Response({"password_enabled": False, "unlocked": True})

    current_password = str(request.data.get("current_password") or "")
    if not settings.password_matches(current_password):
        return Response({"detail": "Mevcut parola yanlış."}, status=400)

    settings.clear_password()
    request.session.pop("privacy_unlocked", None)
    return Response({"password_enabled": False, "unlocked": True})
