"""Feature configuration endpoint plus the gates that enforce it.

The clients ask ``/api/config/`` once at boot and build their navigation
from the answer, so a screen that depends on a disabled feature is never
rendered. The gates below make that more than a courtesy: with the flag
off, the endpoints behind it are simply not there.
"""

from __future__ import annotations

from functools import wraps

from django.http import Http404
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ..pipeline import flags


@api_view(["GET"])
def config_view(request):
    """Report which parts of the app are switched on."""
    features = flags.snapshot()
    return Response(
        {
            "features": features,
            # Convenience alias — the clients branch on this constantly.
            "nlp_enabled": features["nlp"],
        }
    )


def nlp_only(view):
    """Hide a function-based view while the NLP pipeline is disabled."""

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        _require_nlp()
        return view(request, *args, **kwargs)

    return wrapper


class NlpOnlyMixin:
    """Hide a DRF view/viewset while the NLP pipeline is disabled."""

    def initial(self, request, *args, **kwargs):
        _require_nlp()
        return super().initial(request, *args, **kwargs)


def _require_nlp() -> None:
    if not flags.nlp_enabled():
        raise Http404("NLP hattı kapalı.")
