# Hatırlaf — Contextual Voice Diary

> Sesli günlük tutun. Yerel sunucu yazıya döker, kim/ne/nerede/ne zaman
> bilgilerini çıkarır, takvime yerleştirir ve belirsiz yerleri size sorar.
> Hiçbir veri üçüncü taraf bulutuna gitmez.

Hatırlaf is a **local-first Turkish voice diary**. You speak (or type), the
backend transcribes the audio with Whisper, runs Turkish NLP + a local LLM to
extract a structured event log (date, time, place, people, what happened),
and surfaces everything in a calendar plus an editable entries list. The
mobile-shaped web UI lets you record, review, edit and delete diary entries
without ever leaving your machine.

This repo is the reference implementation of the
*Contextual Voice Diary* MVP described in
`Project_Kickoff_Contextual_Voice_Diary-Final.pdf`. The spec targets React
Native; this repo ships the full Django backend **and** a responsive PWA
web client so you can exercise the entire pipeline end-to-end on a Linux
desktop without installing Node/Expo.

---

## 1. What the program does (end-user view)

Three screens, one bottom-tab nav:

1. **Ana (Home)** — Big red microphone for voice recording, plus a text
   composer right below it for typing entries when a mic isn't an option.
   Both submit through the same backend pipeline.
2. **Girişler (Entries)** — Every diary entry the user has captured, in
   reverse chronological order. Voice entries show an `<audio>` player **plus**
   the editable Whisper transcript. Text entries show only the editable text.
   Each card has **Kaydet** (save edits), **Yeniden işle** (re-run the NLP +
   LLM pipeline against the edited text) and a destructive **Sil** that
   wipes the row, the audio file, the mentions, the edges, and the calendar
   events that came from it.
3. **Takvim (Calendar)** — Monthly grid. Each day is a tile showing the
   structured events the LLM extracted for that date. Tapping a tile opens a
   drawer with full event details — what happened, who was there, where.

Behind the scenes, every entry triggers a five-stage pipeline:

```
audio?  →  Whisper STT  →  Zeyrek + NER  →  conflict detection  →  LLM event log
                                            (mention bridge)        (calendar feed)
```

If the LLM weights aren't on disk, the pipeline degrades gracefully to a
deterministic NLP-only fallback so the calendar always gets at least one
event per session.

---

## 2. Architecture at a glance

```
┌─────────────────────────┐      HTTP/local      ┌─────────────────────────────┐
│  Mobile-style Web App   │ ─── audio upload ──▶ │  Django + DRF Backend       │
│  (PWA, no build step)   │ ◀── JSON / audio ─── │  • faster-whisper STT       │
│  • MediaRecorder        │                      │  • Zeyrek morphology + NER  │
│  • IndexedDB sync queue │                      │  • Conflict detector        │
│  • Hash router (3 tabs) │                      │  • llama.cpp (Qwen2.5-7B)   │
│  • Per-card audio player│                      │  • SQLite / PostgreSQL      │
└─────────────────────────┘                      └─────────────────────────────┘
```

Both ML models — Whisper and the Qwen2.5 GGUF — are **warm-loaded into
memory in a background thread when Django boots**, so the first user request
doesn't pay the cold-start cost.

The mapping between the PDF spec and this repo:

| Spec component                 | This repo                                              |
|--------------------------------|--------------------------------------------------------|
| React Native (Expo) + SQLite   | `templates/diary/index.html` + `static/js/*` + IndexedDB sync queue (`db.js`) |
| Django + DRF                   | `backend/` (same stack, 1:1)                           |
| Whisper self-hosted            | `diary/processing/transcription.py` (faster-whisper → openai-whisper → placeholder) |
| Zeyrek + BERTurk               | `diary/processing/nlp.py` (Zeyrek + optional BERTurk + rule-based fallback) |
| Local LLM event extraction     | `diary/processing/llm.py` (llama.cpp + Qwen2.5-7B, three-pass with self-critique) |
| Node/Edge relational graph     | `diary/models.py` (`Node`, `Edge`, `Mention`, `Session`) |
| Unknown-Entity fallback        | `Node.unknown_for(kind)` + `IGNORE`/`UNKNOWN` resolve actions |
| Conflict detection + Review UI | `diary/processing/conflicts.py` + `static/js/screens/review.js` |
| Calendar view                  | `diary/views/api.py` (`calendar_view`) + `static/js/screens/timeline.js` |
| Background processing          | `diary/processing/pipeline.py` (`threading.Thread`)    |

No Celery, no Redis, no Node.js.

---

## 3. How the code works (end-to-end)

### 3.1 Frontend submission

`static/js/screens/home.js` owns the home screen. The recorder uses
`audio.js` (a thin MediaRecorder wrapper) and writes the resulting Blob to
the IndexedDB-backed queue in `db.js`. The text composer skips the audio
step and pushes a `{transcript: "..."}` row into the same queue.

`sync.js` drains that queue. Every queued item becomes a `multipart/form-data`
POST to `/api/sessions/`. The `client_uuid` makes the upload idempotent — the
queue retries safely on flaky networks.

### 3.2 Backend kickoff

`SessionViewSet.create` in `diary/views/api.py`:

1. Looks up the session by `client_uuid`. If it already exists, returns it
   with HTTP 200 (idempotent retry).
2. Otherwise saves a new `Session` row in `queued` status.
3. Calls `pipeline.kickoff(session.id)`, which spawns a daemon thread
   (`hatirlaf-<id>`) so the HTTP response returns instantly.

### 3.3 Background pipeline

`diary/processing/pipeline.py::_process` walks each session through:

1. **Transcribe** (`transcription.py`) — `faster-whisper` `large-v3-turbo`
   by default, with VAD pre-filtering, beam search width 5,
   `condition_on_previous_text=False`, and a Turkish `initial_prompt` that
   nudges the decoder toward proper-noun casing and standard punctuation.
   Word-level timings come back too, so a mention can scrub the audio
   player to its exact range later.
2. **Skip transcription for text-only entries.** If the user typed the
   entry, the transcript is already there and the audio step is skipped.
3. **NLP extraction** (`extractor.py` + `nlp.py`) — Zeyrek lemmatization
   plus rule-based + optional BERTurk NER. Produces clauses with concrete
   facts (`tarih=2026-04-25`, `zaman=Şu An`, `kisiler=...`,
   `lokasyon=...`).
4. **Conflict detection** (`conflicts.py`) — flags ambiguous pronouns,
   relative time expressions and unknown referents that the user will need
   to confirm in the review screen.
5. **LLM event log** (`llm.py`) — three passes through llama.cpp:
   * *Pass 1:* free-form analysis of the paragraph, given the NLP hints
     as ground truth.
   * *Pass 2:* a self-critique pass that re-checks dates, time bucketing
     ("Geçmiş/Şu An/Gelecek"), proper-noun casing and merges duplicate
     events.
   * *Pass 3:* JSON-constrained output validated against a strict schema
     (`OLAY_LOG_SCHEMA`).
   * If the LLM file isn't on disk, `_fallback_from_hints` packages the
     deterministic clause list into the same JSON shape so downstream code
     never has to special-case the no-LLM path.
6. **Persist mentions + edges** — `Mention` rows bridge transcript
   character spans to canonical `Node`s; non-conflict mentions are
   auto-resolved into a Node. The session's `structured_events` field
   stores the LLM (or fallback) calendar feed.

The `Session.status` field is the user-facing progress tracker
(`queued → transcribing → parsing → completed`), polled by the review
screen until it settles.

### 3.4 Calendar

`calendar_view` in `diary/views/api.py` rolls every Session's
`structured_events` into a `{ "YYYY-MM-DD": [event, …] }` shape, optionally
filtered by `?month=YYYY-MM`. The frontend `screens/timeline.js` paints a
month grid; tapping a day opens a modal with the full event list.

Because the calendar reads directly from `Session.structured_events`,
deleting a session removes its events from the calendar automatically — no
separate cache to invalidate.

### 3.5 Editing and deleting

* **Edit transcript.** `screens/record.js` lets the user edit any entry's
  transcript. Saving sends `PATCH /api/sessions/<id>/` with the new text.
  *Yeniden işle* (re-run pipeline) hits `/process/`, which moves the
  session back to `queued` and kicks off a fresh pass.
* **Delete entry.** Sil triggers `DELETE /api/sessions/<id>/`.
  `SessionViewSet.destroy` runs the row delete (which cascades to
  `Mention` and `Edge` via FK `on_delete=CASCADE`), then explicitly
  deletes the audio file from `MEDIA_ROOT` (Django's FileField doesn't
  remove files on row delete by default).

### 3.6 Warm-loading models

`diary/apps.py::DiaryConfig.ready` spawns a `hatirlaf-preload` daemon
thread that calls `transcription.preload()` and then `llm.preload()`. Both
helpers are idempotent and thread-safe — the existing lazy loaders just
get warmed up. Skipped under `migrate`/`test`/`check`/`shell`/etc., the
Django autoreload parent process, and when
`HATIRLAF_PRELOAD_MODELS=0` is set.

---

## 4. Project layout

```
hatırlaf/
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── diary_backend/                 # Django project (settings, urls, wsgi)
│   │   └── settings.py
│   ├── diary/                         # Main app
│   │   ├── apps.py                    # Warm-loads STT + LLM at startup
│   │   ├── models.py                  # Session, Node, Edge, Mention
│   │   ├── serializers.py             # DRF serializers (transcript editable)
│   │   ├── admin.py
│   │   ├── urls.py
│   │   ├── views/
│   │   │   ├── api.py                 # ViewSets, calendar, delete cascade
│   │   │   └── web.py                 # Serves the SPA shell
│   │   ├── processing/
│   │   │   ├── transcription.py       # faster-whisper large-v3-turbo + VAD
│   │   │   ├── nlp.py                 # Zeyrek + BERTurk + rules
│   │   │   ├── extractor.py           # Clause-level fact extraction
│   │   │   ├── conflicts.py           # Pronoun / time / unknown ambiguity
│   │   │   ├── llm.py                 # llama.cpp + Qwen2.5-7B (3-pass)
│   │   │   ├── pipeline.py            # Orchestration + daemon thread
│   │   │   └── name_gazetteer.py
│   │   └── management/commands/
│   │       └── seed_demo.py           # python manage.py seed_demo --reset
│   ├── templates/diary/index.html     # SPA shell (3 tabs: Ana / Girişler / Takvim)
│   └── static/
│       ├── css/app.css
│       └── js/
│           ├── app.js                 # Hash router + online indicator
│           ├── api.js                 # REST client (incl. update/delete)
│           ├── db.js                  # IndexedDB offline queue
│           ├── sync.js                # Background upload worker
│           ├── audio.js               # MediaRecorder wrapper
│           ├── events.js              # Tiny pub/sub + toast
│           └── screens/
│               ├── home.js            # Mic + text composer
│               ├── record.js          # Entries list (edit / delete / reprocess)
│               ├── review.js          # Per-session conflict resolution UI
│               ├── timeline.js        # Monthly calendar
│               └── utils.js           # el(), modal(), fmtRelative()
├── scripts/
│   ├── setup.sh                       # venv + pip install + migrate
│   ├── run.sh                         # python manage.py runserver
│   └── install_whisper.sh             # Optional STT/NER deps
├── Qwen2.5-7B-Instruct-Q4_K_M.gguf    # Local LLM weights (4.4 GB; optional)
└── Project_Kickoff_Contextual_Voice_Diary-Final.pdf
```

---

## 5. Setup (Linux: Fedora / Ubuntu)

Prerequisites:

- Python ≥ 3.10 (tested on 3.14)
- `ffmpeg` (Whisper needs it for decoding)
- A modern browser (Firefox / Chromium / Chrome)
- ~6 GB free RAM if you want both faster-whisper `large-v3-turbo` (~1.5 GB)
  and the Qwen2.5-7B Q4 GGUF (~5 GB) loaded simultaneously

```bash
cd "$(dirname "$0")"           # the hatırlaf/ folder
./scripts/setup.sh
./scripts/install_whisper.sh   # installs faster-whisper (recommended)
```

`setup.sh` creates `.venv/`, installs the Python deps, runs migrations on
SQLite (`backend/db.sqlite3`) and is safe to re-run.

`install_whisper.sh` modes:

```bash
./scripts/install_whisper.sh faster      # faster-whisper (recommended on CPU)
./scripts/install_whisper.sh openai      # reference openai-whisper (heavier)
./scripts/install_whisper.sh berturk     # adds Hugging Face BERTurk NER
./scripts/install_whisper.sh all         # faster-whisper + BERTurk
```

If you skip the Whisper install entirely, the app still works — voice
entries land with empty transcripts and the user types them in via the
home-screen composer instead.

---

## 6. Running

```bash
./scripts/run.sh
```

Open **http://127.0.0.1:8000/** in your browser. On first boot the
preloader will spend ~30-60 s pulling the Whisper weights and a few seconds
mapping the LLM into memory; the server answers HTTP requests during this
window because the load runs in a background thread.

The sample data in `seed_demo` gives you three demo Turkish entries to
explore the calendar and conflict-resolution UIs immediately.

---

## 7. Configuration (env vars)

| Variable                          | Default              | Meaning                                          |
|-----------------------------------|----------------------|--------------------------------------------------|
| `HATIRLAF_DEBUG`                  | `1`                  | Toggle Django debug + permissive CORS            |
| `HATIRLAF_SECRET_KEY`             | *auto*               | Django secret (set in prod)                      |
| `HATIRLAF_DATABASE_URL`           | —                    | `postgres://…` to swap SQLite for Postgres       |
| `HATIRLAF_HOST`                   | `127.0.0.1`          | Bind host for `run.sh`                           |
| `HATIRLAF_PORT`                   | `8000`               | Bind port for `run.sh`                           |
| `HATIRLAF_WHISPER_MODEL`          | `large-v3-turbo`     | `tiny` / `base` / `small` / `medium` / `large-v3` / `large-v3-turbo` |
| `HATIRLAF_WHISPER_LANG`           | `tr`                 | Whisper language code                            |
| `HATIRLAF_WHISPER_COMPUTE_TYPE`   | `int8_float32`       | CTranslate2 compute type (auto-falls-back to `int8`) |
| `HATIRLAF_WHISPER_DEVICE`         | `cpu`                | `cpu` or `cuda`                                  |
| `HATIRLAF_WHISPER_BEAM_SIZE`      | `5`                  | Beam search width (lower = faster, less accurate)|
| `HATIRLAF_WHISPER_VAD`            | `1`                  | Voice-activity-detection silence skipping        |
| `HATIRLAF_USE_BERTURK`            | `0`                  | Enable Hugging Face BERTurk NER                  |
| `HATIRLAF_LLM_MODEL_PATH`         | `./Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Path to the GGUF weights          |
| `HATIRLAF_LLM_N_CTX`              | `4096`               | LLM context window                               |
| `HATIRLAF_LLM_N_GPU_LAYERS`       | `-1`                 | Layers to offload to GPU (`-1` = all)            |
| `HATIRLAF_PRELOAD_MODELS`         | `1`                  | Warm-load STT + LLM at app boot                  |
| `HATIRLAF_SYNC_PROCESSING`        | `0`                  | Run pipeline inline instead of a daemon thread (useful in tests) |

### Why `large-v3-turbo` for Turkish?

OpenAI released `large-v3-turbo` in late 2024. It keeps almost the same
word-error-rate as `large-v3` on Turkish but runs **4-5× faster on CPU**.
Combined with VAD pre-filtering (cuts silence before decoding) and beam
search 5, it's currently the strongest *open-source* Turkish STT setup
that still runs on a single laptop. If you need maximum accuracy and don't
mind the wait, set `HATIRLAF_WHISPER_MODEL=large-v3`. If you're tight on
RAM, `small` (~500 MB) keeps the rest of the optimisations and still
handles clear Turkish prose well.

---

## 8. API reference

All endpoints live under `/api/`.

| Method | Path                          | Purpose                                |
|--------|-------------------------------|----------------------------------------|
| GET    | `/health/`                    | `{ok: true}` liveness probe            |
| POST   | `/sessions/`                  | Upload audio + metadata (idempotent on `client_uuid`) |
| GET    | `/sessions/`                  | List sessions                          |
| GET    | `/sessions/<id>/`             | Session detail incl. mentions          |
| PATCH  | `/sessions/<id>/`             | Edit the transcript                    |
| DELETE | `/sessions/<id>/`             | Delete the session, mentions, edges, calendar events, and audio file |
| POST   | `/sessions/<id>/process/`     | Re-run the pipeline                    |
| GET    | `/sessions/<id>/audio/`       | Stream the raw audio                   |
| GET    | `/mentions/?session=<id>`     | List mentions                          |
| POST   | `/mentions/<id>/resolve/`     | Resolve a conflict (`ASSIGN`/`NEW`/`UNKNOWN`/`IGNORE`) |
| GET    | `/nodes/`                     | Search/list canonical entities         |
| POST   | `/nodes/`                     | Create a node                          |
| GET    | `/edges/`                     | List relationship edges                |
| GET    | `/timeline/`                  | Combined timeline feed                 |
| GET    | `/calendar/?month=YYYY-MM`    | Day-bucketed structured events         |
| GET    | `/graph/`                     | Compact `{nodes, edges}` dump          |

Example — upload a manual transcript (no audio):

```bash
curl -F "client_uuid=$(uuidgen)" \
     -F "recorded_at=$(date -Iseconds)" \
     -F "language=tr" \
     -F "transcript=Dün Ahmet ile İstanbul'da buluştuk." \
     http://127.0.0.1:8000/api/sessions/
```

Example — edit a transcript and re-run extraction:

```bash
curl -X PATCH -H "Content-Type: application/json" \
     -d '{"transcript":"Bugün Ayşe ile sahilde yürüdük."}' \
     http://127.0.0.1:8000/api/sessions/42/

curl -X POST http://127.0.0.1:8000/api/sessions/42/process/
```

Example — delete an entry (cascades to mentions, edges, audio file, calendar):

```bash
curl -X DELETE http://127.0.0.1:8000/api/sessions/42/
```

---

## 9. Cascade delete: what gets removed

Deleting a session removes:

- the `Session` row itself
- every `Mention` row whose FK points at it (`on_delete=CASCADE`)
- every `Edge` row whose FK points at it (`on_delete=CASCADE`)
- the audio file under `backend/media/sessions/YYYY/MM/DD/…` (explicit
  `storage.delete()` in `SessionViewSet.destroy`)
- every event the session contributed to the calendar — because
  `structured_events` lives on the session row, and `calendar_view`
  reads from sessions, so the day's tile shrinks (or disappears)
  automatically on the next calendar load

Nodes are *intentionally* preserved. A `Node` is the canonical record for
a person, place, or time and may be referenced by other sessions. If a
node ends up with zero mentions it just sits in the autocomplete pool for
future entries.

---

## 10. Privacy

- No outbound network calls beyond the browser ↔ your own Django server.
- No telemetry, no crash reporting, no CDN fonts.
- Audio files live under `backend/media/sessions/YYYY/MM/DD/` on your disk.
- The SQLite DB lives at `backend/db.sqlite3`. Back this file up and
  you've backed up every diary entry, mention, node, and edge.
- The Qwen2.5 GGUF weights are loaded from local disk; the LLM never
  reaches out to the network.

---

## 11. Migrating to React Native later

The backend is already the RN-compatible one in the PDF. To swap the
`templates/` + `static/` client for an Expo app:

1. Replace the IndexedDB layer (`db.js`) with `expo-sqlite`.
2. Replace MediaRecorder (`audio.js`) with `expo-av`.
3. Keep the same REST endpoints — `fetch`/Axios calls in `api.js` translate
   directly.

---

## 12. Mapping to the PDF roadmap

- **Phase 1 — Offline-First Client & Local Infrastructure.** ✓ Django +
  relational Node/Edge tables, Whisper-ready backend, IndexedDB-backed
  client cache + background sync queue.
- **Phase 2 — Local Turkish NLP Engine.** ✓ Zeyrek lemmatization, optional
  BERTurk NER, rule-based fallback. Pronoun / relative-time / unknown-referent
  conflict engine with the Unknown-Entity fallback. Local LLM (Qwen2.5-7B
  via llama.cpp) for structured event extraction with a deterministic
  fallback when the model is missing.
- **Phase 3 — Resolution UI & Timeline View.** ✓ Editable entries list with
  per-card audio playback and cascading delete. Review screen with
  highlighted mentions and word-aligned audio scrubbing. Monthly calendar
  rollup driven by the LLM event log.
