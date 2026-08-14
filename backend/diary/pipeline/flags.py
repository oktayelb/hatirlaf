"""The switch.

Everything that is *understanding* rather than *capture* lives behind
``nlp_enabled()``. Capture — record audio, transcribe it, store the text —
always runs, because that is the diary itself. Understanding — mentions,
entity resolution, conflicts, LLM eventification, the calendar and the
memory pages — only runs when the flag is on.

To flip the whole thing, set one value:

    # backend/diary_backend/settings.py
    HATIRLAF_NLP_ENABLED = True

or from the environment, without touching code:

    HATIRLAF_NLP_ENABLED=1 scripts/run.sh

Nothing else needs to change. The pipeline drops its understanding stages,
the API stops serving the NLP endpoints, the web and mobile clients read
``/api/config/`` and hide the screens that depend on them.
"""

from __future__ import annotations

from django.conf import settings

#: Keys the rest of the app may ask about. Kept explicit so a typo in a
#: feature name fails loudly instead of silently reading as "off".
FEATURES = ("nlp",)


def nlp_enabled() -> bool:
    """Return True when the natural-language understanding pipeline is live."""
    return bool(getattr(settings, "HATIRLAF_NLP_ENABLED", False))


def is_enabled(feature: str) -> bool:
    """Return the state of a named feature. Raises on unknown names."""
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature flag: {feature!r}")
    return {"nlp": nlp_enabled}[feature]()


def snapshot() -> dict:
    """Serialisable view of every flag, for ``/api/config/``."""
    return {feature: is_enabled(feature) for feature in FEATURES}
