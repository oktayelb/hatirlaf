"""
Django settings for Hatırlaf — Contextual Voice Diary.

Privacy-first defaults:
  * SQLite by default (simple, local, file-based). PostgreSQL is a drop-in
    replacement — see README for the DATABASE_URL convention.
  * DEBUG off unless HATIRLAF_DEBUG=1. A reasonable dev key is generated
    on first run (not safe for production).
  * CORS wide-open in DEBUG so the web client and potential RN client can
    connect from any localhost port.
"""

from pathlib import Path
import os
import secrets

BASE_DIR = Path(__file__).resolve().parent.parent

# Dev-friendly secret handling: pull from env if provided, else ephemeral.
SECRET_KEY = os.environ.get("HATIRLAF_SECRET_KEY") or secrets.token_urlsafe(48)

DEBUG = os.environ.get("HATIRLAF_DEBUG", "1") == "1"

ALLOWED_HOSTS = ["*"] if DEBUG else os.environ.get("HATIRLAF_ALLOWED_HOSTS", "").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",
    "diary",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "diary_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "diary_backend.wsgi.application"

# --- Database ----------------------------------------------------------------
# The PDF spec calls for PostgreSQL with a relational adjacency list. For
# frictionless Linux testing we default to SQLite, which supports every
# query the MVP needs. Point HATIRLAF_DATABASE_URL at a Postgres instance to
# switch in production:
#   HATIRLAF_DATABASE_URL=postgres://user:pass@host:5432/dbname
_database_url = os.environ.get("HATIRLAF_DATABASE_URL")
if _database_url and _database_url.startswith("postgres"):
    from urllib.parse import urlparse

    parsed = urlparse(_database_url)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": parsed.path.lstrip("/"),
            "USER": parsed.username or "",
            "PASSWORD": parsed.password or "",
            "HOST": parsed.hostname or "localhost",
            "PORT": str(parsed.port or 5432),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "tr"
TIME_ZONE = "Europe/Istanbul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- DRF ---------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.AllowAny"],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
}

# --- CORS --------------------------------------------------------------------
# Single-user local MVP: permissive CORS so RN / Expo web / dev tooling can
# reach the API from any origin during development.
CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = os.environ.get("HATIRLAF_CORS_ORIGINS", "").split(",") if not DEBUG else []

# --- App config --------------------------------------------------------------
# Whisper model size. ``large-v3-turbo`` is the recommended Turkish default —
# nearly identical accuracy to ``large-v3`` at roughly 4-5× the speed on CPU.
# Override with HATIRLAF_WHISPER_MODEL=small (or tiny/base) for low-RAM
# machines, or large-v3 if you want maximum accuracy at the cost of speed.
HATIRLAF_WHISPER_MODEL = os.environ.get("HATIRLAF_WHISPER_MODEL", "large-v3-turbo")
HATIRLAF_WHISPER_LANG = os.environ.get("HATIRLAF_WHISPER_LANG", "tr")
# CTranslate2 compute type. ``int8_float32`` keeps fp32 accuracy on hot paths
# while quantising weights to int8. Falls back to plain ``int8`` automatically
# on older runtimes.
HATIRLAF_WHISPER_COMPUTE_TYPE = os.environ.get(
    "HATIRLAF_WHISPER_COMPUTE_TYPE", "int8_float32"
)
HATIRLAF_WHISPER_DEVICE = os.environ.get("HATIRLAF_WHISPER_DEVICE", "cpu")
# Beam search width. 5 is a quality/speed sweet spot; lower for faster decode.
HATIRLAF_WHISPER_BEAM_SIZE = int(os.environ.get("HATIRLAF_WHISPER_BEAM_SIZE", "5"))
# Voice activity detection skips silence before decoding — typically a
# 30-60% speedup on real diary recordings.
HATIRLAF_WHISPER_VAD = os.environ.get("HATIRLAF_WHISPER_VAD", "1") == "1"

# Warm-load STT + LLM models in a background thread when Django boots, so
# the first user request doesn't pay the cold-start cost. Disable for
# headless tooling that doesn't need them.
HATIRLAF_PRELOAD_MODELS = os.environ.get("HATIRLAF_PRELOAD_MODELS", "1") == "1"

# Run NLP synchronously instead of in a background thread (useful for tests).
HATIRLAF_SYNC_PROCESSING = os.environ.get("HATIRLAF_SYNC_PROCESSING", "0") == "1"

# Enable Hugging Face BERTurk NER. Defaults off (large download).
HATIRLAF_USE_BERTURK = os.environ.get("HATIRLAF_USE_BERTURK", "0") == "1"

# Local llama.cpp model used for structured event extraction. Leave the
# path at a non-existent file to run NLP-only (the pipeline degrades
# gracefully and still produces a calendar of events).
HATIRLAF_LLM_MODEL_PATH = os.environ.get(
    "HATIRLAF_LLM_MODEL_PATH",
    str(BASE_DIR.parent / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
)
HATIRLAF_LLM_N_CTX = int(os.environ.get("HATIRLAF_LLM_N_CTX", "4096"))
HATIRLAF_LLM_N_GPU_LAYERS = int(os.environ.get("HATIRLAF_LLM_N_GPU_LAYERS", "-1"))

# Uploaded audio limit (50 MB is generous for a single diary session).
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[{levelname}] {name}: {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "diary": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
