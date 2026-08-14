# Project Structure

A file-by-file tour of the Hatırlaf tree. The [README](../README.md#repository-layout)
has the short version; this is the one to read when you need to know where a
particular thing lives.

Generated and vendored contents are described but not enumerated: `.git/`,
`.venv/`, `__pycache__/`, `var/media/`, and the treebank corpora under
`vendor/savyar/data/`.

## Top level

| Path | What it is |
|---|---|
| `Makefile` | The entrypoint. `make` lists every target; each one wraps a script or a `manage.py` command. |
| `README.md` | Project overview, setup, API reference, roadmap. |
| `docs/` | Written documentation and the original brief. |
| `server/` | The Django project. The only component that owns data. |
| `clients/` | The two front ends: browser SPA and Expo app. |
| `vendor/` | Third-party source checked into the repo. |
| `scripts/` | Shell entrypoints and the SAVYAR subprocess bridge. |
| `models/` | Large model weights. Gitignored — see the README for what to download. |
| `var/` | Everything written at runtime. Gitignored and disposable. |

`var/` and `models/` are deliberately outside `server/`. Django's defaults
would scatter the database, uploads and collected static across the project
package; keeping them in one gitignored directory makes "what is disposable"
answerable at a glance, and `make reset` a safe command.

## `server/`

The Django project root — where `manage.py` lives and where every command runs.

- `manage.py` — Django command entry point. Points at `config.settings`.
- `requirements.txt` — Python dependencies for the app and the ML integrations.

### `server/config/`

Django project configuration. Named `config` rather than after the app, so
there is never a question of which of two similarly-named packages you want.

- `settings.py` — settings, installed apps, the path constants (`REPO_ROOT`,
  `VAR_DIR`, `WEB_CLIENT_DIR`, `MODELS_DIR`, `VENDOR_DIR`), and every
  `HATIRLAF_*` flag.
- `urls.py` — project-level routing; mounts the app and the API.
- `wsgi.py` / `asgi.py` — deployment entry points.

### `server/diary/`

The single Django app. Everything about the diary itself.

- `models.py` — `Session`, `Mention`, `Node`, `Edge`, plus status and conflict
  metadata.
- `serializers.py` — DRF serializers. The NLP switch decides which fields are
  emitted at all.
- `urls.py` — API route registration.
- `apps.py` — app config; kicks off background model warm-loading at startup.
- `admin.py` — Django admin registration.
- `middleware.py` — the privacy lock, which gates every request when the app
  is locked.
- `encryption.py` / `storage.py` / `media.py` — encrypted-at-rest file storage
  for uploaded audio, and the view that serves it back.

#### `server/diary/views/`

One module per API surface, so a route is easy to find from its URL.

- `api.py` — the aggregate module the URLconf imports from.
- `api_shared.py` — helpers shared by the API modules.
- `api_sessions.py` — session CRUD, upload, reprocessing.
- `api_config.py` — `/api/config/`: feature flags, and the gate decorators the
  other modules use.
- `api_analytics.py` — timeline, calendar, recap, graph. NLP-only; 404 when the
  switch is off.
- `api_mentions.py` / `api_nodes.py` — mention review and the entity graph.
  NLP-only.
- `api_privacy.py` — lock/unlock and privacy settings.
- `web.py` — serves the web client's HTML shell.

#### `server/diary/pipeline/`

What happens to an entry, and when. Import from the package, not the modules.

- `flags.py` — the NLP switch itself, plus the payload behind `/api/config/`.
- `stages.py` — the ordered list of steps. Each declares whether it applies
  with NLP on, off, or always, and whether it is deferred to its own worker.
- `runner.py` — walks the applicable stages; owns threading, re-entrancy and
  failure bookkeeping. Knows nothing about what any stage does.

#### `server/diary/processing/`

The work the stages actually do.

- `transcription.py` — speech-to-text via `faster-whisper` or
  `openai-whisper`, with a safe placeholder fallback.
- `nlp.py` — the Turkish analysis layer: tokens, lemmas, mentions.
- `nlp_models.py` / `nlp_morph.py` / `nlp_ner.py` — model loading, morphology,
  and named-entity recognition behind it.
- `extractor.py` — deterministic Turkish event extraction pre-pass; produces
  the structured hints the LLM refines.
- `conflicts.py` — flags ambiguous mentions, bare pronouns and relative time
  expressions for review.
- `llm.py` — local `llama.cpp` wrapper turning hints into calendar events,
  degrading gracefully when the model is absent.
- `entity_registry.py` / `name_gazetteer.py` — entity resolution and
  name/place lookup.
- `savyar_adapter.py` — the client side of the SAVYAR bridge; shells out to
  `scripts/savyar_ml_bridge.py`.
- `startup.py` — readiness/progress tracking for STT, LLM and SAVYAR, which is
  what the startup splash screen renders.

#### `server/diary/services/`

- `session_pipeline.py` — orchestration that sits above the ORM and below the
  views.

#### `server/diary/management/commands/`

- `seed_demo.py` — demo data for local testing and UI work (`make seed`).

#### `server/diary/migrations/` and `server/diary/tests/`

Schema history, and the test suite (`make test`). The tests cover both sides of
the NLP switch, Turkish relative-date extraction, pronoun detection, subject
inference from verb conjugation, calendar fallback, encrypted storage and the
privacy lock.

## `clients/`

Two front ends against one REST API. Neither owns data; both read
`/api/config/` at boot and build their navigation from the answer.

### `clients/web/`

The browser SPA. Django serves it — `templates/` is on the template path and
`static/` is on `STATICFILES_DIRS` — but no Django code lives here, and there
is no build step.

- `templates/diary/index.html` — the HTML shell, including the startup splash.
- `static/css/app.css` — the only stylesheet the browser loads. It contains
  ordered `@import`s and nothing else.
- `static/css/modules/` — one file per feature area. Order matters: `base.css`
  defines the design tokens, `responsive.css` overrides last.
- `static/js/api.js` — thin REST client for every `/api/` call.
- `static/js/app.js` — hash router, startup gate, online indicator, status
  watcher.
- `static/js/config.js` — reads `/api/config/` once at boot; assumes the
  smallest app if it cannot.
- `static/js/db.js` / `sync.js` — IndexedDB queue and the offline upload loop.
- `static/js/audio.js` / `photos.js` / `textsize.js` / `icons.js` / `events.js`
  — recording, the device-local photo board, the reader type scale, inline SVG
  icons, and a small event bus with toasts.
- `static/js/screens/home.js` — Ana: the photo board, record button, composer.
- `static/js/screens/entries.js` — Günlüğüm: every entry with audio and text.
- `static/js/screens/settings.js` — Ayarlar: text size and the app password.
- `static/js/screens/utils.js` — shared DOM, modal and formatting helpers.
- `static/js/screens/review.js`, `timeline.js`, `recap.js`, `memories.js` —
  routable only while the NLP switch is on.

### `clients/mobile/`

The Expo / React Native client, same API and palette as the web client.
`make mobile` starts it.

- `App.js` — root component and navigation.
- `src/screens/` — Home, Entries, Calendar, Recap, Settings, Lock.
- `src/services/` — `api.js`, `config.js`, `features.js`, `queue.js`,
  `background.js`, `photos.js`, `reminders.js`.
- `src/ui/` — `Primitives.js` and `PhotoBoard.js`.
- `src/theme.js` — the shared palette.

## `vendor/savyar/`

SAVYAR, a Turkish morphological analyser, checked in with its own README,
licence, requirements and layout. It is treated as a dependency: Hatırlaf never
edits it and never imports its internals directly. The only contact point is
`scripts/savyar_ml_bridge.py`, which runs in SAVYAR's own environment and
returns ranked candidates as JSON.

Disabled by default (`HATIRLAF_USE_SAVYAR=0`) — the NLP layer must work without
it.

## `scripts/`

- `setup.sh` — virtualenv, dependencies, migrations. Behind `make setup`.
- `run.sh` — migrate and serve. Behind `make run`.
- `install_whisper.sh` — install or swap a speech-to-text backend after the
  fact.
- `savyar_ml_bridge.py` — not a developer command. The backend spawns it as a
  subprocess so SAVYAR can run under its own interpreter.
- `evaluate_savyar_boun.py` — measures subject/tense accuracy against the BOUN
  treebank with SAVYAR off and on.

## `docs/`

- `structure.md` — this file.
- `project-kickoff.pdf` — the original project brief.
