# Hatırlaf

Hatırlaf is a local-first Turkish voice diary. You record yourself or write, and the backend transcribes the audio so you can read your entries back later.

Behind a single feature flag there is a second, larger app: a natural-language pipeline that extracts people, places, times and events from what you said, asks for clarification when a reference is ambiguous, and lays the result out on a calendar. **That pipeline ships switched off.** See [The NLP Switch](#the-nlp-switch).

The repository holds two things that no longer share an architecture:

- **`server/` plus `clients/web/`** — a Django + DRF backend and the browser client that speaks its REST API. This is where the NLP and LLM pipeline lives, and where the research side of the project happens.
- **`clients/mobile/`** — an Expo/React Native app that keeps the entire diary on the phone. It has no server, no account, and no network calls: Turkish speech-to-text runs on-device, entries live in a local SQLite database, and audio never leaves the handset. See [its README](clients/mobile/README.md).

The split is deliberate. The pipeline needs multi-gigabyte models and a machine to run them on; a diary you carry needs neither, and hosting one would mean holding someone's private recordings on a server. The two halves share a palette and a set of ideas, not a data path.

## Quick Start

```bash
make setup      # virtualenv, dependencies, migrations
make run        # http://127.0.0.1:8000
```

Run `make` on its own for the full list of targets. If `make setup` is too
heavy for your machine — it pulls several GB of model weights — use
`make setup-minimal` instead and add the models later.

New to the codebase? [Repository Layout](#repository-layout) is the map, and
`docs/structure.md` is the file-by-file tour.

## Current Status

This is an MVP/reference implementation, not a production-ready consumer deployment.

It is ready for:

- Local single-user use on a trusted machine
- Demoing the full diary pipeline end to end
- Testing Turkish speech-to-text, NLP extraction, eventification, conflict resolution, and calendar rollups
- Serving as the research half of the project, alongside the standalone mobile app

It is not yet ready for:

- Public internet deployment without authentication and authorization
- Multi-user accounts
- Production-grade background processing
- Encrypted at-rest storage
- App Store / Play Store release
- Operational monitoring, backups, and privacy/compliance review

The most important production gap is security: the API currently allows any caller to read/write data when they can reach the server. Treat the current app as local/private only.

## What The App Does

With the NLP switch off — the shipped default — the app does three things:

- Record a voice diary entry in Turkish.
- Type a diary entry instead, when speaking is not an option.
- Read your entries back, play the audio, and correct the transcribed text.

The UI is two screens, plus a settings page behind the gear in the header:

- **Ana**: one or two photos to look at while you talk, a large record button, and a box to write in.
- **Günlüğüm**: every entry, newest first, each with its audio and its transcribed text.

Turn the switch on and two more screens appear:

- **Takvim**: month calendar showing extracted events on their resolved dates.
- **Özet**: monthly rollup of people, places and moods.

Example:

```text
Recorded on 2026-04-30:
"Dün işteydim, bugün ise erken kalktım ve okula gitmeyi düşünüyorum."
```

The calendar should show that event on `2026-04-29`, not merely on the recording day, because `Dün` is resolved relative to the recording timestamp.

## The NLP Switch

Everything that *understands* an entry, as opposed to merely *capturing* it, sits behind one flag:

```python
# server/config/settings.py
HATIRLAF_NLP_ENABLED = os.environ.get("HATIRLAF_NLP_ENABLED", "0") == "1"
```

Change the default, or set the variable and leave the code alone:

```bash
make run NLP=1
```

That single value moves all of the following at once.

**Off (the default)**

- The pipeline runs `transcribe → archive`. Audio becomes text; the entry is saved.
- `/api/timeline/`, `/api/calendar/`, `/api/recap/`, `/api/graph/`, `/api/mentions/`, `/api/nodes/` and `/api/edges/` return **404**. They are not merely hidden — they are not served.
- Session payloads carry no `structured_events`, `mentions`, `mention_count`, `conflict_count`, `eventification_*`, `mood`, `tags`, `processed_text` or `word_timings`. A client cannot display analysis output, because it never receives any.
- The LLM and the SAVYAR morphology bridge are not preloaded, so several GB of weights stay unloaded.
- The web client reads `/api/config/` at boot and builds its navigation from it: two tabs, no calendar, no reminders. (The mobile app is unaffected — it never had these features, because it has no server.)

**On**

- The pipeline runs `transcribe → understand → (eventify)`.
- The endpoints, the payload fields, the model preloading, and the Takvim/Özet screens all come back.

### Where the switch lives

```text
server/diary/pipeline/
├── flags.py    the switch itself, plus /api/config/'s payload
├── stages.py   the ordered list of steps, each declaring when it applies
└── runner.py   threads, re-entrancy, failure bookkeeping
```

`runner.py` knows nothing about what a stage does. It walks `stages.PIPELINE`, skips the stages whose flag is off, and runs the rest. Adding a step means adding an entry to that list:

```python
Stage(
    key="understand",
    label="Metin analiz ediliyor",
    run=understand,
    requires=NLP_ON,   # NLP_ANY | NLP_ON | NLP_OFF
)
```

`deferred=True` hands a stage to its own worker after the synchronous chain returns, so a slow model never delays the entry from appearing.

## Architecture

This describes the server side. The mobile client is a separate stack that
touches none of it — see [Mobile App](#mobile-app).

```text
Browser client
  - MediaRecorder
  - IndexedDB queue
  - REST API client
        |
        v
Django + Django REST Framework
  - Session upload and idempotency
  - Background processing thread
  - Audio transcription
  - Turkish NLP extraction
  - Conflict detection
  - Local LLM eventification
  - Calendar API
        |
        v
SQLite today / PostgreSQL later
  - Session
  - Mention
  - Node
  - Edge
  - structured_events JSON
```

The backend is deliberately client-agnostic, and the browser client is replaceable.

## End-To-End Pipeline

Every diary entry becomes a `Session` row.

With the NLP switch **off**:

```text
audio upload -> Whisper transcription -> word timing alignment -> archive
manual text  -> archive
```

With the NLP switch **on**:

```text
audio upload
  -> Whisper transcription
  -> word timing alignment
  -> Turkish NLP extraction
  -> conflict detection
  -> local LLM eventification   (deferred, own worker)
  -> calendar rollup

manual text
  -> Turkish NLP extraction
  -> conflict detection
  -> local LLM eventification   (deferred, own worker)
  -> calendar rollup
```

Processing is started by `diary/pipeline/runner.py`. The HTTP upload returns quickly and the work runs in a daemon thread.

## Backend Data Model

The core models live in `server/diary/models.py`.

- `Session`: one diary entry. Stores audio metadata, transcript, status, `structured_events`, and eventification status.
- `Mention`: a span in the transcript that references a person, place, time, event, organization, or pronoun.
- `Node`: canonical graph entity, such as a person or location.
- `Edge`: relationship between nodes, attached to a session.

`structured_events` is a JSON list stored on `Session`. It drives the calendar. Deleting a session automatically removes its calendar contribution because the calendar is computed from sessions.

## Calendar Behavior

The calendar API is implemented in `calendar_view` in `server/diary/views/api_analytics.py`. It returns 404 while the NLP switch is off.

Priority order for event display:

1. Use completed `structured_events` if present.
2. If eventification is still queued/running, use saved NLP clause hints so the entry still appears on the correct resolved date.
3. If no hints exist, fall back to the recording date with transcript text.

This matters because the full LLM eventification step can be slow or unavailable. The calendar should still show useful entries as soon as the basic NLP pass has completed.

## Models And Tools Used

### Web Framework

- **Django 5.x**: backend framework, routing, settings, ORM.
- **Django REST Framework**: REST API, serializers, viewsets.
- **django-cors-headers**: permissive local development CORS.

### Database

- **SQLite by default**: simple local database at `var/db.sqlite3`.
- **PostgreSQL supported by env var**: set `HATIRLAF_DATABASE_URL=postgres://...`.

SQLite is fine for local single-user use. PostgreSQL should be used for real deployment.

### Speech-To-Text

Implemented in `server/diary/processing/transcription.py`.

Supported backends:

- **faster-whisper** with CTranslate2, preferred
- **openai-whisper**, fallback
- **placeholder**, if no STT backend is installed

Default model:

- `large-v3-turbo`

Important settings:

- `HATIRLAF_WHISPER_MODEL`
- `HATIRLAF_WHISPER_LANG`
- `HATIRLAF_WHISPER_DEVICE`
- `HATIRLAF_WHISPER_COMPUTE_TYPE`
- `HATIRLAF_WHISPER_BEAM_SIZE`
- `HATIRLAF_WHISPER_VAD`

The code uses Turkish-specific prompt text, VAD silence skipping, beam search, and word-level timing where supported.

### Turkish NLP

Implemented across:

- `server/diary/processing/nlp.py`
- `server/diary/processing/extractor.py`
- `server/diary/processing/conflicts.py`

Tools and techniques:

- **Zeyrek** for Turkish morphology
- Hugging Face Turkish NER for person/place/organization extraction when enabled
- Rule-based named entity and time extraction as the no-download fallback
- `dateparser` for absolute and relative date grounding
- Optional `transformers` + `torch` runtime for the Turkish NER model

The extractor creates clause-level hints:

- resolved date, such as `2026-04-29`
- time of day, such as `14:30`
- time bucket: `Geçmiş`, `Şu An`, `Gelecek`
- people
- places
- organizations
- pronoun/reference candidates
- inferred subject from Turkish verb conjugation
- full clause text

Turkish named entities:

- Default model: `savasy/bert-base-turkish-ner-cased`
- Labels consumed by Hatırlaf: `PER` -> person, `LOC` -> place, `ORG` -> organization
- Alternative researched model: `akdeniz27/xlm-roberta-base-turkish-ner`
- Enable with `HATIRLAF_USE_TURKISH_NER=1`
- Override with `HATIRLAF_TURKISH_NER_MODEL=<huggingface-model-id>`

The LLM prompt now treats NER people, places, and organizations as the
authoritative entity candidate list. It may use grammatical subjects such as
`Ben`, but it should not invent person or place names outside the transcript or
NER hints.

### Local LLM

Implemented in `server/diary/processing/llm.py`.

Default model path:

```text
Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

Runtime:

- **llama-cpp-python**
- local GGUF weights
- no cloud API

The LLM converts NLP hints into structured event JSON. If the model is missing or fails, Hatırlaf falls back to deterministic NLP-only events.

The LLM path uses:

- free-form analysis
- critique/repair pass
- JSON-constrained output schema
- post-processing to sanitize dates, people, locations, and event fields

### Frontend

No Node build step is required.

Always-on files:

- `clients/web/templates/diary/index.html`
- `clients/web/static/js/app.js` — router; builds navigation from `/api/config/`
- `clients/web/static/js/config.js` — feature flags
- `clients/web/static/js/screens/home.js` — photos, recorder, composer
- `clients/web/static/js/screens/entries.js` — the entry log
- `clients/web/static/js/screens/settings.js` — text size and the app password
- `clients/web/static/js/photos.js` — the photo board
- `clients/web/static/js/textsize.js` — reader-controlled type scale
- `clients/web/static/js/db.js`, `sync.js`, `audio.js`, `icons.js`
- `clients/web/static/css/app.css` and `clients/web/static/css/modules/`

Loaded but only routable while the NLP switch is on:

- `screens/timeline.js`, `screens/recap.js`, `screens/memories.js`, `screens/review.js`

Browser APIs:

- MediaRecorder for audio capture
- IndexedDB for the offline upload queue **and** the home-screen photos
- Fetch API for REST calls

### Design

The palette is warm paper with muted sage and clay accents — no dark mode and no saturated colour, chosen to stay readable for people over 40 on a phone in poor light. Base type is 19px, buttons have a 56px minimum touch target, and every control carries a full-sentence explanation in body-sized text rather than a caption.

Readers can scale every font in the app from **Ayarlar → Yazı Boyutu**. It works because each size token is a multiple of a single `--text-scale` custom property, so nothing in the layout has to know about it.

## Setup

Prerequisites:

- Linux or compatible environment
- Python 3.10+
- `ffmpeg` for audio decoding
- A modern browser
- Enough RAM for selected models

Two commands get you a running app:

```bash
make setup     # create .venv, install dependencies, run migrations
make run       # serve on http://127.0.0.1:8000
```

`make` on its own lists every target. The common ones:

| Command | What it does |
|---|---|
| `make setup` | Full install, including the local ML stack. Safe to re-run. |
| `make setup-minimal` | Same, minus the multi-GB models — no STT, NER or LLM. |
| `make run` | Migrate, then serve. `make run NLP=1` turns the understanding pipeline on; `make run PORT=9000` moves the port. |
| `make test` | Backend test suite. |
| `make seed` | Fill the local database with demo entries. |
| `make mobile` | Start the Expo dev server for the mobile client. Needs a development build — Expo Go cannot run it. |
| `make reset` | Delete the local database and recorded audio, after confirming. |

Each target is a thin wrapper over `scripts/*.sh` or `manage.py`, so you can
always drop down a level and run the underlying command directly.

By default, `make setup` also installs the local ML stack used by the app:

- `faster-whisper`
- `openai-whisper`
- `transformers`
- `torch`
- `llama-cpp-python`

It also falls back to the main project virtualenv if a dedicated `vendor/savyar/.venv`
is not present, so SAVYAR does not need a separate manual bootstrap step.

If you want a lighter install on a constrained machine, skip the ML stack:

```bash
make setup-minimal
```

Optional helper commands are still available if you want to reinstall or swap
one backend later:

```bash
./scripts/install_whisper.sh faster
./scripts/install_whisper.sh openai
./scripts/install_whisper.sh ner
./scripts/install_whisper.sh all
```

Run locally:

```bash
make run
```

Open:

```text
http://127.0.0.1:8000/
```

Everything the app writes at runtime — the SQLite database, uploaded audio, the
encryption key, collected static files — lands in `var/`. That directory is
gitignored and disposable: delete it and `make run` builds it again.

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `HATIRLAF_NLP_ENABLED` | `0` | **The switch.** Turns the entire understanding pipeline, its endpoints and its screens on or off |
| `HATIRLAF_DEBUG` | `1` | Enables Django debug mode and permissive dev settings |
| `HATIRLAF_SECRET_KEY` | generated | Django secret key; must be set in production |
| `HATIRLAF_ALLOWED_HOSTS` | empty | Required when debug is off |
| `HATIRLAF_DATABASE_URL` | SQLite | Optional PostgreSQL URL |
| `HATIRLAF_HOST` | `127.0.0.1` | Dev server host used by `make run` |
| `HATIRLAF_PORT` | `8000` | Dev server port used by `make run` |
| `HATIRLAF_VAR_DIR` | `var/` | Where the database, media and keys are written |
| `HATIRLAF_WHISPER_MODEL` | `large-v3-turbo` | Whisper model size/name |
| `HATIRLAF_WHISPER_LANG` | `tr` | Transcription language |
| `HATIRLAF_WHISPER_COMPUTE_TYPE` | `int8_float32` | faster-whisper compute type |
| `HATIRLAF_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `HATIRLAF_WHISPER_BEAM_SIZE` | `5` | Beam search width |
| `HATIRLAF_WHISPER_VAD` | `1` | Enables VAD silence skipping |
| `HATIRLAF_USE_TURKISH_NER` | `0` | Enables optional Hugging Face Turkish NER |
| `HATIRLAF_TURKISH_NER_MODEL` | `savasy/bert-base-turkish-ner-cased` | Hugging Face token-classification model id |
| `HATIRLAF_USE_BERTURK` | `0` | Backwards-compatible alias for `HATIRLAF_USE_TURKISH_NER` |
| `HATIRLAF_LLM_MODEL_PATH` | `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Local Qwen GGUF file |
| `HATIRLAF_LLM_N_CTX` | `4096` | LLM context window |
| `HATIRLAF_LLM_N_GPU_LAYERS` | `-1` | GPU offload layers for llama.cpp |
| `HATIRLAF_SETUP_MINIMAL` | `0` | Skip ML installs during `make setup` |
| `HATIRLAF_PRELOAD_MODELS` | `1` | Warm-load STT and LLM at startup |
| `HATIRLAF_SYNC_PROCESSING` | `0` | Run processing inline, mainly for tests |

## API Overview

All API routes are under `/api/`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/` | Liveness check, model warm-up progress, feature flags |
| `GET` | `/config/` | Feature flags, used by both clients to build navigation |
| `POST` | `/sessions/` | Upload audio/text session |
| `GET` | `/sessions/` | List sessions |
| `GET` | `/sessions/<id>/` | Session detail |
| `PATCH` | `/sessions/<id>/` | Edit transcript |
| `DELETE` | `/sessions/<id>/` | Delete session and audio file |
| `POST` | `/sessions/<id>/process/` | Re-run processing |
| `GET` | `/sessions/<id>/audio/` | Stream audio |

The routes below are served **only while `HATIRLAF_NLP_ENABLED=1`**. With the switch off they return 404.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/mentions/?session=<id>` | List mentions |
| `POST` | `/mentions/<id>/resolve/` | Resolve mention conflict |
| `GET` | `/nodes/` | List/search graph nodes |
| `POST` | `/nodes/` | Create graph node |
| `GET` | `/edges/` | List graph edges |
| `GET` | `/timeline/` | Timeline feed |
| `GET` | `/calendar/?month=YYYY-MM` | Calendar event buckets |
| `GET` | `/recap/?month=YYYY-MM` | Monthly memory rollup |
| `GET` | `/graph/` | Compact graph dump |

Manual transcript upload:

```bash
curl -F "client_uuid=$(uuidgen)" \
  -F "recorded_at=$(date -Iseconds)" \
  -F "language=tr" \
  -F "transcript=Dün Ahmet ile İstanbul'da buluştuk." \
  http://127.0.0.1:8000/api/sessions/
```

Edit and reprocess:

```bash
curl -X PATCH \
  -H "Content-Type: application/json" \
  -d '{"transcript":"Bugün Ayşe ile sahilde yürüdük."}' \
  http://127.0.0.1:8000/api/sessions/42/

curl -X POST http://127.0.0.1:8000/api/sessions/42/process/
```

## Deployment Readiness

### What Is Already In Good Shape

- Backend API is separated from the browser client and could serve other callers.
- SQLite and PostgreSQL paths already exist.
- Audio upload, transcript editing, reprocessing, delete cascade, and calendar APIs exist.
- STT and LLM can run locally without cloud calls.
- The app degrades when heavy ML dependencies are missing.
- Processing status is persisted on the session row.
- Calendar can show NLP-derived events before LLM eventification finishes.
- Tests cover key NLP/calendar behavior.

### What Must Change Before Production

- Add authentication.
- Add per-user ownership to every session, node, mention, edge, and audio file.
- Replace permissive DRF permissions with user-scoped access checks.
- Set `HATIRLAF_DEBUG=0`.
- Set a stable `HATIRLAF_SECRET_KEY`.
- Configure `HATIRLAF_ALLOWED_HOSTS`.
- Restrict CORS.
- Use PostgreSQL instead of SQLite.
- Serve media securely.
- Add HTTPS.
- Add backup and restore workflows.
- Move background processing to a durable queue such as Celery/RQ/Django-Q.
- Add retry handling for failed eventification.
- Add observability: logs, metrics, error reporting, health checks.
- Add rate limits and upload validation.
- Add privacy controls: export, delete account, data retention, consent text.
- Consider encryption at rest for transcripts/audio.

### Background Job Risk

The current background processing uses daemon threads. That is acceptable for local MVP use, but not durable. If the process restarts during transcription or eventification, the work can be interrupted.

For deployment, use:

- Celery + Redis/RabbitMQ
- RQ + Redis
- Django-Q
- a managed task queue

Each task should be idempotent and restartable from `Session.status` and `eventification_status`.

### Model Hosting Risk

Running Whisper and Qwen locally is private but resource-heavy.

Deployment options:

- Run models on the same backend host for simplicity.
- Put STT and LLM behind internal worker services.
- Use GPU acceleration for better latency.
- Use smaller Whisper models for mobile-ish responsiveness.
- Keep deterministic NLP fallback as a reliability path.

### Security Risk

The current app has no user model integration and no permissions. A deployed version must assume diary data is highly sensitive.

Minimum production security baseline:

- Authenticated users
- User-scoped querysets
- Private media storage
- CSRF/session strategy or token strategy
- HTTPS only
- Encrypted backups
- Secrets managed outside git
- Explicit privacy policy

## Pros And Cons

### Pros

- Local-first and privacy-oriented.
- No cloud LLM or cloud STT is required.
- Works with both voice and typed entries.
- Turkish-specific relative date handling.
- Calendar remains useful even before the LLM finishes.
- REST backend is reusable by other clients.
- Deterministic fallback keeps the app functional without large model files.
- Simple deployment story for local demos.
- Data model can grow into a personal knowledge graph.

### Cons

- Heavy local models need RAM, disk, and CPU/GPU capacity.
- Daemon-thread jobs are not production-durable.
- No multi-user/auth layer yet.
- Turkish NLP is heuristic in places and will need real-world evaluation.
- Local LLM output can still be imperfect and needs guardrails.
- SQLite is not appropriate for multi-user production.
- No encrypted storage yet.
- The mobile app is not packaged for either store yet.

## Mobile App

`clients/mobile/` is a standalone Expo app. It does not talk to the backend in
this repository, and it has no network code at all.

Stack:

- Expo / React Native
- `expo-speech-recognition` — Turkish speech-to-text on the device
- `expo-sqlite` — the diary itself
- `expo-file-system` — audio files
- `expo-audio` — playback
- `expo-secure-store` + `expo-crypto` — the app password
- `expo-sharing` — export

### Why it has no server

Uploading the audio would mean hosting someone's diary, which brings
authentication, per-user ownership, HTTPS, private media serving and a privacy
policy with it — and would make the app useless off the home network. Keeping
everything on the phone removes all of that, at the cost of the NLP features,
which cannot run on a handset.

### The privacy rule

`src/services/speech.js` always passes `requiresOnDeviceRecognition: true`.
Both platforms will otherwise fall back to network recognition, which on
Android means sending diary audio to Google. When on-device Turkish is
unavailable, the app records without a transcript rather than going online.

Consequences:

| Situation | Behaviour |
|---|---|
| iOS, modern device | Live transcript plus recording |
| Android 13+ with Turkish model | Live transcript plus recording |
| Android 13+ without the model | Offers to install it; records meanwhile |
| Android 12 and below | Recording only — the OS cannot persist recogniser audio |

### Known constraints

- Recordings are 16 kHz 16-bit mono WAV, ~1.9 MB/min. The recogniser will not
  accept compressed AAC, so there is no transcode-free way to shrink them.
- The phone is the only copy. Export writes all entry text to one file; bulk
  audio export is not implemented yet.
- Native modules mean Expo Go cannot run it — a development build is required.

## Privacy Model

Current privacy posture:

- Audio and transcripts stay on the server you run.
- The LLM uses local GGUF weights.
- No cloud API is required by the app code.
- SQLite database and media files are local files.

Current privacy gaps:

- No app-level encryption.
- No user isolation.
- No audited delete/export workflow.
- No production privacy policy.
- Optional ML dependencies may download model weights during installation.

For a shipped mobile app, privacy should be a product feature, not just an implementation detail.

## Testing

Run the Django test suite:

```bash
make test
```

Tests that exercise the understanding pipeline declare it explicitly, because
it is off by default:

```python
@override_settings(HATIRLAF_NLP_ENABLED=True)
class CalendarApiTests(TestCase):
    ...
```

Current tests cover:

- both sides of the NLP switch: which endpoints are served, which session
  fields are serialised, and which stages the pipeline runs
  (`diary/tests/test_feature_flags.py`)
- Turkish relative date extraction
- pronoun/reference detection
- subject inference from Turkish verb conjugation
- calendar fallback behavior while eventification is running
- NLP-only eventification text preservation
- LLM cache lifecycle cleanup
- encrypted storage and the privacy lock

Recommended next tests:

- API auth and permissions, after auth is added
- audio upload validation
- reprocessing idempotency
- background job retry behavior
- mobile sync conflict cases
- calendar edge cases across time zones and month boundaries

## Repository Layout

Five top-level directories, each with one job.

```text
hatırlaf/
├── Makefile              the entrypoint — `make` lists everything
├── README.md
├── docs/                 written docs and the original project brief
│   ├── structure.md      file-by-file tour of the tree
│   └── project-kickoff.pdf
├── server/               the Django project — the only thing that owns data
│   ├── manage.py
│   ├── requirements.txt
│   ├── config/           project settings, root URLs, WSGI/ASGI
│   └── diary/            the single Django app
│       ├── models.py     Session, Mention, Node, Edge
│       ├── serializers.py
│       ├── urls.py
│       ├── views/        one module per API surface
│       │   ├── api_sessions.py
│       │   ├── api_analytics.py    NLP-only, gated
│       │   ├── api_config.py       feature flags + the gates
│       │   └── web.py              serves the web client's shell
│       ├── pipeline/     what happens to an entry, and when
│       │   ├── flags.py            the NLP switch
│       │   ├── stages.py           ordered steps, each with its flag
│       │   └── runner.py           threads and failure handling
│       ├── processing/   the work each stage does
│       │   ├── transcription.py    Whisper
│       │   ├── nlp*.py             Turkish morphology, NER, mentions
│       │   ├── extractor.py        deterministic event pre-pass
│       │   ├── conflicts.py        ambiguity detection
│       │   ├── llm.py              local llama.cpp eventification
│       │   └── savyar_adapter.py   bridge to vendor/savyar
│       ├── services/     orchestration above the ORM
│       ├── management/   custom manage.py commands (seed_demo)
│       ├── migrations/
│       └── tests/
├── clients/              two front ends, one REST API
│   ├── web/              the browser SPA, served by Django
│   │   ├── templates/    the HTML shell
│   │   └── static/       css/ and js/, no build step
│   └── mobile/           the standalone Expo app (no server; see its README)
│       ├── App.js
│       └── src/          screens/, services/, ui/
├── vendor/               third-party source checked in, not our code
│   └── savyar/           Turkish morphological analyser
├── scripts/              setup.sh, run.sh, install_whisper.sh, and the
│                         savyar bridge the backend shells out to
├── models/               large model weights (gitignored)
│   └── Qwen2.5-7B-Instruct-Q4_K_M.gguf
└── var/                  everything written at runtime (gitignored)
    ├── db.sqlite3
    ├── media/            uploaded audio
    ├── staticfiles/      collectstatic output
    └── encryption.key
```

The two rules that keep it navigable:

- **`server/config/` is configuration; `server/diary/` is the application.**
  Nothing about the diary belongs in `config/`, and no Django wiring belongs
  in `diary/`.
- **`var/` and `models/` hold no source.** Deleting either loses only data or
  downloads, never work. That is why neither is committed.

## Suggested Roadmap

Near term:

- Add authentication and user ownership.
- Convert daemon-thread processing to a real queue.
- Add pagination and sync cursors.
- Harden upload validation.
- Add production settings.
- Add API tests around all session lifecycle endpoints.

Mobile (`clients/mobile/`, now server-free — see [Mobile App](#mobile-app)):

- Test on-device Turkish recognition against real speech, on both platforms.
- Add bulk audio export, so a lost phone is not a lost diary.
- Replace the generated placeholder icons.
- Decide whether the diary should be encrypted at rest, not just locked.
- Ship a build: `eas build --profile preview` for Android, TestFlight for iOS.

Production:

- PostgreSQL
- private media storage
- HTTPS
- background workers
- monitoring
- backups
- privacy/export/delete flows
- mobile packaging and store release process
