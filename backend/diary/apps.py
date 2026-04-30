import logging
import os
import sys
import threading

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


# Management commands that should never trigger model preloading. The list
# covers the obvious offenders (migrations, schema introspection, tests) plus
# anything where blocking on a multi-GB load would be obviously wrong.
_PRELOAD_SKIP_COMMANDS = {
    "migrate",
    "makemigrations",
    "showmigrations",
    "sqlmigrate",
    "collectstatic",
    "createsuperuser",
    "shell",
    "dbshell",
    "test",
    "check",
    "diffsettings",
    "loaddata",
    "dumpdata",
    "compilemessages",
    "makemessages",
}


class DiaryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "diary"
    verbose_name = "Contextual Voice Diary"

    def ready(self) -> None:  # noqa: D401 — Django hook
        """Kick off background warm-load of LLM + Whisper.

        Both models are heavy enough that loading them lazily on the first
        request adds visible latency to the user. Warming them at app
        startup hides that cost behind Django's normal boot, while the
        background thread keeps the dev server responsive immediately.

        Skipped under common one-shot management commands (migrate, test,
        etc.) and when ``HATIRLAF_PRELOAD_MODELS`` is explicitly disabled.
        """
        if not _should_preload():
            from .processing import startup

            startup.mark_preload_disabled()
            return

        # Avoid double-loading under Django's autoreload. ``runserver``
        # forks: the parent watches files, the child (with RUN_MAIN=true)
        # actually serves. Skip the parent so we don't burn RAM twice.
        # Production servers (gunicorn etc.) don't set RUN_MAIN at all, so
        # those still preload normally.
        is_runserver = len(sys.argv) > 1 and sys.argv[1] == "runserver"
        autoreload_enabled = "--noreload" not in sys.argv
        if is_runserver and autoreload_enabled and os.environ.get("RUN_MAIN") != "true":
            return

        logger.info("Warm-loading STT + LLM in background…")
        from .processing import startup

        startup.begin()

        thread = threading.Thread(
            target=_preload_models,
            daemon=True,
            name="hatirlaf-preload",
        )
        thread.start()


def _should_preload() -> bool:
    if not bool(getattr(settings, "HATIRLAF_PRELOAD_MODELS", True)):
        return False
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in _PRELOAD_SKIP_COMMANDS:
        return False
    return True


def _preload_models() -> None:
    # Imports are deferred so this module stays importable even when the
    # heavy ML deps aren't installed (e.g., during ``manage.py check``).
    from .processing import startup

    try:
        from .processing import llm as llm_mod
        from .processing import savyar_adapter as savyar_mod
        from .processing import transcription as tx_mod
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Preload import failed: %s", exc)
        startup.mark_done("stt", "failed", "STT modülü yüklenemedi.")
        startup.mark_done("llm", "failed", "LLM modülü yüklenemedi.")
        startup.mark_done("savyar", "failed", "Savyar modülü yüklenemedi.")
        return

    # STT first: cheaper to load and runs on every voice entry, so prioritise
    # warming it. The LLM follows; if it can't fit, the pipeline degrades to
    # the deterministic NLP fallback automatically.
    startup.mark_loading("stt", "Whisper STT modeli yükleniyor.")
    try:
        tx_mod.preload()
        backend = getattr(tx_mod, "_cached_backend", None) or "bilinmiyor"
        if backend == "placeholder":
            startup.mark_done("stt", "failed", "Whisper bulunamadı; metin girişi kullanılacak.")
        else:
            startup.mark_done("stt", "ready", f"Whisper hazır ({backend}).")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("STT preload failed: %s", exc)
        startup.mark_done("stt", "failed", "Whisper yüklenemedi.")

    startup.mark_loading("llm", "Yerel LLM ağırlıkları belleğe alınıyor.")
    try:
        llm_mod.preload()
        if getattr(llm_mod, "_cached_llm", None) is not None:
            startup.mark_done("llm", "ready", "Yerel LLM hazır.")
        else:
            startup.mark_done("llm", "failed", "LLM yok; deterministik NLP kullanılacak.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("LLM preload failed: %s", exc)
        startup.mark_done("llm", "failed", "LLM yüklenemedi; NLP yedeği kullanılacak.")

    startup.mark_loading("savyar", "Savyar morfoloji köprüsü kontrol ediliyor.")
    try:
        if savyar_mod.preload():
            startup.mark_done("savyar", "ready", "Savyar hazır.")
        elif savyar_mod.enabled():
            startup.mark_done("savyar", "failed", "Savyar etkin ama köprü hazır değil.")
        else:
            startup.mark_done("savyar", "skipped", "Savyar devre dışı.")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Savyar preload failed: %s", exc)
        startup.mark_done("savyar", "failed", "Savyar kontrolü tamamlanamadı.")
