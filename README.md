# Hatırlaf — Contextual Voice Diary

> Sesli günlük kaydedin, yerel sunucunuz yazıya döksün, belirsiz yerleri siz
> onaylayın. Hiçbir veri üçüncü taraf bulutlarına gitmez.

Reference implementation of the **Contextual Voice Diary** (MVP) described in
`Project_Kickoff_Contextual_Voice_Diary-Final.pdf`. The spec targets React Native
on mobile; this repo ships the full Django backend **plus** a mobile-shaped
responsive web client so you can exercise the whole pipeline end-to-end on a
Linux desktop without installing Node/Expo.

Everything is local-first: Whisper runs on *your* machine, Turkish NLP runs
against local models, and audio lives under `backend/media/`.

---

## 1. Architecture at a glance

```
┌─────────────────────────┐      HTTPS/local     ┌──────────────────────────┐
│  Mobile-style Web App   │ ─── audio upload ──▶ │  Django + DRF Backend    │
│  (PWA: static/ + tmpl)  │ ◀── JSON / audio ─── │  - Whisper (self-hosted) │
│  - Record (MediaRecorder│                      │  - Zeyrek (morphology)   │
│  - Offline IndexedDB q. │                      │  - BERTurk NER (optional)│
│  - Review / Timeline UI │                      │  - Conflict engine       │
└─────────────────────────┘                      │  - SQLite / PostgreSQL   │
                                                 └──────────────────────────┘
```

The React Native stack from the PDF maps onto this repo as follows:

| Spec component                 | This repo                                              |
|--------------------------------|--------------------------------------------------------|
| React Native (Expo) + SQLite   | `backend/templates/diary/index.html` + `static/js/*` + **IndexedDB** sync queue (`db.js`) |
| Django + DRF                   | `backend/` (same stack, 1:1)                           |
| Whisper self-hosted            | `diary/processing/transcription.py` (faster-whisper → openai-whisper → placeholder) |
| Zeyrek + BERTurk               | `diary/processing/nlp.py` (Zeyrek + optional BERTurk + rule-based fallback) |
| Node/Edge relational graph     | `diary/models.py` (`Node`, `Edge`, `Mention`, `Session`) |
| Unknown-Entity fallback        | `Node.unknown_for(kind)` + `IGNORE`/`UNKNOWN` resolve actions |
| Conflict Detection + Review UI | `diary/processing/conflicts.py` + `static/js/screens/review.js` |
| Timeline View                  | `static/js/screens/timeline.js`                        |
| Background thread for tasks    | `diary/processing/pipeline.py` (`threading.Thread`)     |

No Celery, no Redis, no Node.js.

---

## 2. First-time setup (Linux, Fedora/Ubuntu)

Prerequisites:

- Python ≥ 3.10 (tested on 3.14)
- `ffmpeg` (for Whisper)
- A modern browser (Firefox, Chromium, Chrome) for the web client

```bash
cd "$(dirname "$0")"   # your hatırlaf/ folder
./scripts/setup.sh
```

This:

1. Creates `.venv/` in the project root.
2. Installs Django, DRF, CORS headers, Zeyrek.
3. Runs migrations on SQLite (`backend/db.sqlite3`).
4. Seeds three sample Turkish diary entries so the UI has content on first
   launch.

**Optional — enable real speech-to-text:**

```bash
./scripts/install_whisper.sh faster      # faster-whisper (recommended on CPU)
# or
./scripts/install_whisper.sh openai      # reference openai-whisper
# or
./scripts/install_whisper.sh berturk     # adds Hugging Face BERTurk NER
# or
./scripts/install_whisper.sh all         # faster-whisper + BERTurk
```

Then set the BERTurk flag when running the server:

```bash
export HATIRLAF_USE_BERTURK=1
```

If you skip Whisper install, the app still works: records are stored and
the pipeline will prompt the user to paste the transcript via the
**"Yazarak Ekle (mikrofonsuz)"** panel in the Record screen.

---

## 3. Running

```bash
./scripts/run.sh
```

Then open **http://127.0.0.1:8000/** in your browser.

You'll see three demo sessions on the timeline. Tap one — the highlighted
yellow spans are detected conflicts (ambiguous pronouns like *"o"*, relative
time expressions like *"dün"*, unknown persons, etc.). Click a span or the
matching card under **ÇATIŞMALAR** to resolve it:

- **Bağla** — attach to an existing node (autocompleted list).
- **Yeni Düğüm** — create a new Person / Location / Time / Event.
- **Bilinmeyen** — route to the *Unknown Entity* fallback (spec behaviour).
- **Yoksay** — dismiss (also falls back to *Unknown*, to preserve graph integrity).

Any time you resolve a mention, the edge table is re-built so the Timeline +
Node list stay consistent.

### Testing the recording flow

1. Tap **Kaydet** in the bottom tab bar.
2. Grant microphone permission when the browser asks.
3. Say a Turkish sentence (e.g. *"Dün Ahmet ile Ankara'da buluştuk"*). Tap
   the stop button when done.
4. Tap **Yüklemeye Ekle** — the audio is queued in IndexedDB and flushed to
   the Django backend by the sync queue.
5. You'll be redirected to the timeline; when transcription finishes the
   Review screen for that session lights up.

### Testing offline mode

1. Open DevTools → Network tab → throttle to "Offline".
2. Record a session — it's stored locally, the header dot turns orange.
3. Flip throttling back to "Online" — sync fires automatically and you'll
   see a toast notification.

---

## 4. Project layout

```
hatırlaf/
├── backend/
│   ├── manage.py
│   ├── requirements.txt
│   ├── diary_backend/          # Django project settings / urls
│   ├── diary/                  # Main app
│   │   ├── models.py           # Session, Node, Edge, Mention
│   │   ├── serializers.py      # DRF serializers
│   │   ├── admin.py            # /admin/ registration
│   │   ├── urls.py             # API routes
│   │   ├── views/              # api.py + web.py
│   │   ├── processing/         # NLP pipeline
│   │   │   ├── transcription.py  # Whisper
│   │   │   ├── nlp.py            # Zeyrek + BERTurk + rules
│   │   │   ├── conflicts.py      # Pronoun / time ambiguity
│   │   │   └── pipeline.py       # Orchestration + threading
│   │   └── management/commands/
│   │       └── seed_demo.py    # `python manage.py seed_demo --reset`
│   ├── templates/diary/index.html
│   └── static/
│       ├── css/app.css
│       └── js/
│           ├── app.js          # Router + online indicator
│           ├── api.js          # REST client
│           ├── db.js           # IndexedDB offline cache
│           ├── sync.js         # Background upload queue
│           ├── audio.js        # MediaRecorder wrapper
│           ├── events.js       # Event bus + toast
│           └── screens/        # home, record, review, timeline, nodes
├── scripts/
│   ├── setup.sh
│   ├── run.sh
│   └── install_whisper.sh
└── Project_Kickoff_Contextual_Voice_Diary-Final.pdf
```

---

## 5. API reference

All endpoints live under `/api/`.

| Method | Path                          | Purpose                                |
|--------|-------------------------------|----------------------------------------|
| POST   | `/sessions/`                  | Upload audio + metadata (idempotent on `client_uuid`) |
| GET    | `/sessions/`                  | List sessions (timeline feed)          |
| GET    | `/sessions/<id>/`             | Session detail incl. mentions          |
| POST   | `/sessions/<id>/process/`     | Re-run the pipeline                    |
| GET    | `/sessions/<id>/audio/`       | Stream the raw audio                   |
| GET    | `/mentions/?session=<id>`     | List mentions                          |
| POST   | `/mentions/<id>/resolve/`     | Resolve a conflict (`ASSIGN`/`NEW`/`UNKNOWN`/`IGNORE`) |
| GET    | `/nodes/`                     | Search/list canonical entities         |
| POST   | `/nodes/`                     | Create a node                          |
| GET    | `/edges/`                     | List relationship edges                |
| GET    | `/timeline/`                  | Combined timeline feed                 |
| GET    | `/graph/`                     | Compact {nodes, edges} dump            |
| GET    | `/health/`                    | `{ok: true}` liveness probe            |

Example — upload a manual transcript (no audio):

```bash
curl -F "client_uuid=$(uuidgen)" \
     -F "recorded_at=$(date -Iseconds)" \
     -F "language=tr" \
     -F "transcript=Dün Ahmet ile İstanbul'da buluştuk." \
     http://127.0.0.1:8000/api/sessions/
```

---

## 6. Configuration (env vars)

| Variable                     | Default  | Meaning                                          |
|------------------------------|----------|--------------------------------------------------|
| `HATIRLAF_DEBUG`             | `1`      | Toggle Django debug + permissive CORS            |
| `HATIRLAF_SECRET_KEY`        | *auto*   | Django secret (set in prod)                      |
| `HATIRLAF_DATABASE_URL`      | —        | `postgres://…` to swap SQLite for Postgres       |
| `HATIRLAF_WHISPER_MODEL`     | `base`   | `tiny` / `base` / `small` / `medium` / `large`   |
| `HATIRLAF_WHISPER_LANG`      | `tr`     | Default language code for Whisper                |
| `HATIRLAF_USE_BERTURK`       | `0`      | Enable Hugging Face BERTurk NER                  |
| `HATIRLAF_SYNC_PROCESSING`   | `0`      | Run pipeline inline instead of a daemon thread   |
| `HATIRLAF_HOST`              | `127.0.0.1` | Bind host for `run.sh`                        |
| `HATIRLAF_PORT`              | `8000`   | Bind port for `run.sh`                           |

---

## 7. Mapping to the PDF roadmap

- **Phase 1 — Offline-First Client & Local Infrastructure.** ✓ Django +
  relational Node/Edge tables, Whisper-ready backend, IndexedDB-based
  client cache + background sync queue.
- **Phase 2 — Local Turkish NLP Engine.** ✓ Zeyrek lemmatization, optional
  BERTurk NER, rule-based fallback. Pronoun/relative-time/unknown-referent
  conflict engine with the Unknown-Entity fallback.
- **Phase 3 — Resolution UI & Timeline View.** ✓ Review screen with
  highlighted mentions, targeted audio playback (click a mention → audio
  jumps to its word range), chronological grouped timeline, node detail
  with edges. Force-directed graph is deferred (per spec).

---

## 8. Migrating to React Native later

The backend is already the RN-compatible one in the PDF. Swap the
`templates/` + `static/` client for an Expo app by:

1. Replacing the IndexedDB layer (`db.js`) with `expo-sqlite`.
2. Replacing MediaRecorder (`audio.js`) with `expo-av`.
3. Replacing `fetch` with Axios (if you prefer) — the endpoints are stable.

---

## 9. Privacy

- No outbound network calls beyond the browser ↔ your own Django server.
- No telemetry, no crash reporting, no CDN fonts.
- Audio files live under `backend/media/sessions/YYYY/MM/DD/` on your disk.
- The SQLite DB lives at `backend/db.sqlite3`. Back this up and you've backed
  up every diary entry, mention, node, and edge.
