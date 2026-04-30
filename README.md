# Hatırlaf

Hatırlaf is a local-first Turkish voice diary. You record or type diary entries, the backend transcribes audio, extracts people/places/times/events, asks for clarification when references are ambiguous, and shows the resulting life events on a calendar.

The current repository is a working Django + DRF backend with a mobile-shaped browser client. It is intentionally built close to the future mobile architecture: the frontend talks to the backend through REST APIs, records audio locally, queues uploads offline, and can later be replaced by an Expo/React Native app without rewriting the backend.

## Current Status

This is an MVP/reference implementation, not a production-ready consumer deployment.

It is ready for:

- Local single-user use on a trusted machine
- Demoing the full diary pipeline end to end
- Testing Turkish speech-to-text, NLP extraction, eventification, conflict resolution, and calendar rollups
- Serving as the backend foundation for a future mobile app

It is not yet ready for:

- Public internet deployment without authentication and authorization
- Multi-user accounts
- Production-grade background processing
- Encrypted at-rest storage
- App Store / Play Store release
- Operational monitoring, backups, and privacy/compliance review

The most important production gap is security: the API currently allows any caller to read/write data when they can reach the server. Treat the current app as local/private only.

## What The App Does

Hatırlaf supports three user workflows:

- Record a voice diary entry in Turkish.
- Type a diary entry manually when audio is not available.
- Review entries, edit transcripts, re-run processing, delete entries, and inspect extracted calendar events.

The browser UI has three main screens:

- **Ana**: microphone recorder and text composer.
- **Girişler**: diary entries with transcript editing, audio playback, reprocess, and delete actions.
- **Takvim**: month calendar showing extracted events on their resolved dates.

Example:

```text
Recorded on 2026-04-30:
"Dün işteydim, bugün ise erken kalktım ve okula gitmeyi düşünüyorum."
```

The calendar should show that event on `2026-04-29`, not merely on the recording day, because `Dün` is resolved relative to the recording timestamp.

## Architecture

```text
Browser / future mobile client
  - MediaRecorder today, expo-av later
  - IndexedDB queue today, expo-sqlite later
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

The app is deliberately backend-centered. The client is replaceable. The future mobile app should keep the same API contract and replace only the browser-specific pieces.

## End-To-End Pipeline

Every diary entry becomes a `Session` row.

For audio entries:

```text
audio upload
  -> Whisper transcription
  -> word timing alignment
  -> Turkish NLP extraction
  -> conflict detection
  -> local LLM eventification
  -> calendar rollup
```

For text entries:

```text
manual transcript
  -> Turkish NLP extraction
  -> conflict detection
  -> local LLM eventification
  -> calendar rollup
```

Processing is started by `diary/processing/pipeline.py`. The HTTP upload returns quickly, and the actual work runs in a daemon thread.

## Backend Data Model

The core models live in `backend/diary/models.py`.

- `Session`: one diary entry. Stores audio metadata, transcript, status, `structured_events`, and eventification status.
- `Mention`: a span in the transcript that references a person, place, time, event, organization, or pronoun.
- `Node`: canonical graph entity, such as a person or location.
- `Edge`: relationship between nodes, attached to a session.

`structured_events` is a JSON list stored on `Session`. It drives the calendar. Deleting a session automatically removes its calendar contribution because the calendar is computed from sessions.

## Calendar Behavior

The calendar API is implemented in `calendar_view` in `backend/diary/views/api.py`.

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

- **SQLite by default**: simple local database at `backend/db.sqlite3`.
- **PostgreSQL supported by env var**: set `HATIRLAF_DATABASE_URL=postgres://...`.

SQLite is fine for local single-user use. PostgreSQL should be used for real deployment.

### Speech-To-Text

Implemented in `backend/diary/processing/transcription.py`.

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

- `backend/diary/processing/nlp.py`
- `backend/diary/processing/extractor.py`
- `backend/diary/processing/conflicts.py`

Tools and techniques:

- **Zeyrek** for Turkish morphology
- Rule-based named entity and time extraction
- `dateparser` for absolute and relative date grounding
- Optional **BERTurk** through `transformers` + `torch`

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

### Local LLM

Implemented in `backend/diary/processing/llm.py`.

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

Files:

- `backend/templates/diary/index.html`
- `backend/static/js/app.js`
- `backend/static/js/screens/home.js`
- `backend/static/js/screens/record.js`
- `backend/static/js/screens/review.js`
- `backend/static/js/screens/timeline.js`
- `backend/static/js/db.js`
- `backend/static/js/sync.js`
- `backend/static/css/app.css`

Browser APIs:

- MediaRecorder for audio capture
- IndexedDB for offline upload queue
- Fetch API for REST calls

This frontend is a development and MVP shell. The intended long-term product should be a mobile app.

## Setup

Prerequisites:

- Linux or compatible environment
- Python 3.10+
- `ffmpeg` for audio decoding
- A modern browser
- Enough RAM for selected models

Create the virtual environment, install Python dependencies, and run migrations:

```bash
./scripts/setup.sh
```

Install optional ML dependencies:

```bash
./scripts/install_whisper.sh faster
```

Other modes:

```bash
./scripts/install_whisper.sh openai
./scripts/install_whisper.sh berturk
./scripts/install_whisper.sh all
```

Run locally:

```bash
./scripts/run.sh
```

Open:

```text
http://127.0.0.1:8000/
```

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `HATIRLAF_DEBUG` | `1` | Enables Django debug mode and permissive dev settings |
| `HATIRLAF_SECRET_KEY` | generated | Django secret key; must be set in production |
| `HATIRLAF_ALLOWED_HOSTS` | empty | Required when debug is off |
| `HATIRLAF_DATABASE_URL` | SQLite | Optional PostgreSQL URL |
| `HATIRLAF_HOST` | `127.0.0.1` | Dev server host used by `scripts/run.sh` |
| `HATIRLAF_PORT` | `8000` | Dev server port used by `scripts/run.sh` |
| `HATIRLAF_WHISPER_MODEL` | `large-v3-turbo` | Whisper model size/name |
| `HATIRLAF_WHISPER_LANG` | `tr` | Transcription language |
| `HATIRLAF_WHISPER_COMPUTE_TYPE` | `int8_float32` | faster-whisper compute type |
| `HATIRLAF_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `HATIRLAF_WHISPER_BEAM_SIZE` | `5` | Beam search width |
| `HATIRLAF_WHISPER_VAD` | `1` | Enables VAD silence skipping |
| `HATIRLAF_USE_BERTURK` | `0` | Enables optional BERTurk NER |
| `HATIRLAF_LLM_MODEL_PATH` | repo GGUF path | Local Qwen GGUF file |
| `HATIRLAF_LLM_N_CTX` | `4096` | LLM context window |
| `HATIRLAF_LLM_N_GPU_LAYERS` | `-1` | GPU offload layers for llama.cpp |
| `HATIRLAF_PRELOAD_MODELS` | `1` | Warm-load STT and LLM at startup |
| `HATIRLAF_SYNC_PROCESSING` | `0` | Run processing inline, mainly for tests |

## API Overview

All API routes are under `/api/`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/` | Liveness check |
| `POST` | `/sessions/` | Upload audio/text session |
| `GET` | `/sessions/` | List sessions |
| `GET` | `/sessions/<id>/` | Session detail |
| `PATCH` | `/sessions/<id>/` | Edit transcript |
| `DELETE` | `/sessions/<id>/` | Delete session and audio file |
| `POST` | `/sessions/<id>/process/` | Re-run processing |
| `GET` | `/sessions/<id>/audio/` | Stream audio |
| `GET` | `/mentions/?session=<id>` | List mentions |
| `POST` | `/mentions/<id>/resolve/` | Resolve mention conflict |
| `GET` | `/nodes/` | List/search graph nodes |
| `POST` | `/nodes/` | Create graph node |
| `GET` | `/edges/` | List graph edges |
| `GET` | `/timeline/` | Timeline feed |
| `GET` | `/calendar/?month=YYYY-MM` | Calendar event buckets |
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

- Backend API is separated from the client and can serve a future mobile app.
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
- REST backend is reusable by a future mobile app.
- Deterministic fallback keeps the app functional without large model files.
- Simple deployment story for local demos.
- Data model can grow into a personal knowledge graph.

### Cons

- Heavy local models need RAM, disk, and CPU/GPU capacity.
- Daemon-thread jobs are not production-durable.
- No multi-user/auth layer yet.
- Turkish NLP is heuristic in places and will need real-world evaluation.
- Local LLM output can still be imperfect and needs guardrails.
- Browser MediaRecorder is not the final mobile recording stack.
- SQLite is not appropriate for multi-user production.
- No encrypted storage yet.
- No packaged mobile app yet.

## Future Mobile App Plan

The future mobile app should be treated as a first-class client of the same backend, not a rewrite of the backend.

Recommended stack:

- Expo / React Native
- `expo-av` or the modern Expo audio APIs for recording
- `expo-file-system` for local audio files
- `expo-sqlite` for offline queue and cached sessions
- React Query or a small custom sync layer
- SecureStore for auth tokens
- Push notifications later for reminders, not needed for MVP

Mobile client replacements:

| Current browser piece | Mobile replacement |
|---|---|
| `MediaRecorder` | `expo-av` / Expo audio recording |
| IndexedDB queue | `expo-sqlite` |
| Hash router | React Navigation |
| Static CSS UI | React Native components |
| Browser fetch | fetch/Axios with auth token |
| Browser audio player | Expo audio playback |

Backend APIs to keep stable:

- `POST /api/sessions/`
- `GET /api/sessions/`
- `PATCH /api/sessions/<id>/`
- `POST /api/sessions/<id>/process/`
- `GET /api/calendar/?month=YYYY-MM`
- `GET /api/mentions/?session=<id>`
- `POST /api/mentions/<id>/resolve/`

Mobile-specific backend work needed:

- Authentication tokens
- Per-device idempotent upload handling
- Better upload progress handling
- File cleanup for abandoned uploads
- User-specific sync cursors
- Pagination
- Conflict resolution UX optimized for touch
- Offline-first reconciliation rules

Mobile product questions to settle:

- Is all processing self-hosted on the user's computer, or on a private server?
- Will mobile upload audio to a home server, a cloud VM, or an on-device model?
- Should transcripts/audio be encrypted before upload?
- Should the app support multiple devices per user?
- Should event extraction happen immediately or when the phone is charging/on Wi-Fi?

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
./.venv/bin/python backend/manage.py test diary
```

Current tests cover:

- Turkish relative date extraction
- pronoun/reference detection
- subject inference from Turkish verb conjugation
- calendar fallback behavior while eventification is running
- NLP-only eventification text preservation
- LLM cache lifecycle cleanup

Recommended next tests:

- API auth and permissions, after auth is added
- audio upload validation
- reprocessing idempotency
- background job retry behavior
- mobile sync conflict cases
- calendar edge cases across time zones and month boundaries

## Repository Layout

```text
hatırlaf/
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── db.sqlite3
│   ├── diary_backend/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── asgi.py
│   │   └── wsgi.py
│   ├── diary/
│   │   ├── models.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   ├── apps.py
│   │   ├── views/
│   │   │   ├── api.py
│   │   │   └── web.py
│   │   ├── processing/
│   │   │   ├── transcription.py
│   │   │   ├── nlp.py
│   │   │   ├── extractor.py
│   │   │   ├── conflicts.py
│   │   │   ├── llm.py
│   │   │   └── pipeline.py
│   │   └── tests/
│   ├── static/
│   │   ├── css/app.css
│   │   └── js/
│   └── templates/
├── scripts/
│   ├── setup.sh
│   ├── run.sh
│   └── install_whisper.sh
├── Qwen2.5-7B-Instruct-Q4_K_M.gguf
└── README.md
```

## Suggested Roadmap

Near term:

- Add authentication and user ownership.
- Convert daemon-thread processing to a real queue.
- Add pagination and sync cursors.
- Harden upload validation.
- Add production settings.
- Add API tests around all session lifecycle endpoints.

Mobile MVP:

- Build Expo client against current REST API.
- Implement local SQLite queue.
- Add authenticated session upload.
- Add calendar and entries views.
- Add conflict review UI.
- Test offline upload and retry flows.

Production:

- PostgreSQL
- private media storage
- HTTPS
- background workers
- monitoring
- backups
- privacy/export/delete flows
- mobile packaging and store release process

