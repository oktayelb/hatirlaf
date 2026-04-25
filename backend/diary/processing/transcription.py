"""Self-hosted speech-to-text, tuned for Turkish.

Backend selection (best to worst):
  1. ``faster-whisper`` (CTranslate2). Fast on CPU, supports word timings,
     handles Turkish very well at ``large-v3-turbo`` and above. This is the
     default and the recommended setup.
  2. ``openai-whisper`` (reference PyTorch implementation). Works but is much
     heavier; used only when faster-whisper is not installed.
  3. No-op placeholder. Returns an empty transcript so the UI can fall back
     to the manual text composer.

Why ``large-v3-turbo``?
  Whisper ``large-v3-turbo`` (released by OpenAI in late 2024) is roughly
  4–5× faster than ``large-v3`` on CPU while keeping nearly identical word
  error rates on Turkish. For a personal diary on commodity hardware this is
  the sweet spot. Override with ``HATIRLAF_WHISPER_MODEL`` if you need a
  smaller footprint (``base``, ``small``) or higher accuracy at the cost of
  speed (``large-v3``).

Performance notes:
  * The model is **warm-loaded once** at Django startup (see ``apps.py``),
    so the first transcription request does not pay the multi-second cold
    start cost.
  * VAD pre-filtering trims silence before decoding — typically a 30-60%
    speedup on real diary recordings, which contain lots of pauses.
  * ``beam_size=5`` is a quality/speed sweet spot — noticeably better than
    greedy decoding (the default) with only a small wall-clock penalty.
  * ``int8_float32`` (faster-whisper's hybrid quantisation) gives close to
    fp32 quality with int8 throughput; falls back to ``int8`` when the
    runtime doesn't support it.
  * A short Turkish ``initial_prompt`` nudges the decoder to emit standard
    Turkish punctuation and proper-noun casing instead of all-lowercase.

No audio ever leaves the server.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from django.conf import settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_cached_model = None
_cached_backend: str | None = None

# Turkish prompt: hints to the decoder that the audio is Turkish prose with
# punctuation and proper nouns. Keeps the transcript readable.
_TR_INITIAL_PROMPT = (
    "Bugün notlarımı tutuyorum. Türkçe konuşuyorum, isimleri büyük harfle "
    "yazıyorum ve cümlelerimi noktalama işaretleriyle bitiriyorum."
)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float

    def as_dict(self) -> dict:
        return {"word": self.word, "start": self.start, "end": self.end}


@dataclass
class TranscriptionResult:
    text: str
    words: list[WordTiming] = field(default_factory=list)
    language: str = "tr"
    backend: str = "placeholder"
    duration: float = 0.0

    def words_as_dicts(self) -> list[dict]:
        return [w.as_dict() for w in self.words]


def _model_size() -> str:
    return getattr(settings, "HATIRLAF_WHISPER_MODEL", "large-v3-turbo")


def _compute_type() -> str:
    return getattr(settings, "HATIRLAF_WHISPER_COMPUTE_TYPE", "int8_float32")


def _device() -> str:
    return getattr(settings, "HATIRLAF_WHISPER_DEVICE", "cpu")


def _beam_size() -> int:
    return int(getattr(settings, "HATIRLAF_WHISPER_BEAM_SIZE", 5))


def _vad_filter() -> bool:
    return bool(getattr(settings, "HATIRLAF_WHISPER_VAD", True))


def _load_model():
    """Load whichever Whisper backend is available, once.

    Thread-safe so concurrent requests during startup don't race. Returns
    ``(model, backend_name)``; ``model`` is ``None`` only when no backend is
    installed.
    """
    global _cached_model, _cached_backend

    with _model_lock:
        if _cached_backend is not None:
            return _cached_model, _cached_backend

        size = _model_size()

        # Preferred: faster-whisper (CTranslate2). Much lighter on CPU.
        try:
            from faster_whisper import WhisperModel  # type: ignore

            compute_type = _compute_type()
            device = _device()
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute=%s)",
                size,
                device,
                compute_type,
            )
            try:
                _cached_model = WhisperModel(
                    size, device=device, compute_type=compute_type
                )
            except ValueError:
                # Older CTranslate2 builds may not know int8_float32 — degrade
                # to plain int8 instead of failing the whole load.
                logger.warning(
                    "compute_type=%s not supported, falling back to int8.",
                    compute_type,
                )
                _cached_model = WhisperModel(size, device=device, compute_type="int8")
            _cached_backend = "faster-whisper"
            return _cached_model, _cached_backend
        except Exception as exc:  # pragma: no cover - depends on local env
            logger.debug("faster-whisper unavailable: %s", exc)

        # Fallback: reference openai-whisper.
        try:
            import whisper  # type: ignore

            logger.info("Loading openai-whisper model '%s'", size)
            _cached_model = whisper.load_model(size)
            _cached_backend = "openai-whisper"
            return _cached_model, _cached_backend
        except Exception as exc:
            logger.debug("openai-whisper unavailable: %s", exc)

        _cached_backend = "placeholder"
        return None, _cached_backend


def preload() -> None:
    """Best-effort warm-load of the STT model.

    Called from ``DiaryConfig.ready`` in a background thread so the server
    starts answering health checks immediately while the model loads.
    """
    try:
        _model, backend = _load_model()
        logger.info("STT preload complete (backend=%s).", backend)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("STT preload failed: %s", exc)


def is_available() -> bool:
    _model, backend = _load_model()
    return backend in ("faster-whisper", "openai-whisper")


def transcribe(audio_path: str | Path, language: str = "tr") -> TranscriptionResult:
    """Transcribe an audio file to Turkish text with word-level timings.

    Never raises for missing backends — always returns a TranscriptionResult.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.error("Audio file not found: %s", audio_path)
        return TranscriptionResult(text="", backend="placeholder")

    model, backend = _load_model()

    if backend == "faster-whisper" and model is not None:
        return _run_faster_whisper(model, audio_path, language)
    if backend == "openai-whisper" and model is not None:
        return _run_openai_whisper(model, audio_path, language)

    logger.warning(
        "No Whisper backend available — returning empty transcript. "
        "Install faster-whisper or openai-whisper for real transcription."
    )
    return TranscriptionResult(
        text="",
        language=language,
        backend="placeholder",
    )


def _run_faster_whisper(model, audio_path: Path, language: str) -> TranscriptionResult:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        beam_size=_beam_size(),
        vad_filter=_vad_filter(),
        # Short clips: don't let prior context bias the decode (a common
        # source of repeated/hallucinated phrases in diary recordings).
        condition_on_previous_text=False,
        initial_prompt=_TR_INITIAL_PROMPT if language == "tr" else None,
    )
    words: list[WordTiming] = []
    text_parts: list[str] = []
    for seg in segments:
        text_parts.append(seg.text)
        for w in getattr(seg, "words", None) or []:
            # faster-whisper sometimes prefixes words with a leading space.
            token = (w.word or "").strip()
            if not token:
                continue
            words.append(WordTiming(word=token, start=float(w.start), end=float(w.end)))
    text = "".join(text_parts).strip()
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        backend="faster-whisper",
        duration=duration,
    )


def _run_openai_whisper(model, audio_path: Path, language: str) -> TranscriptionResult:
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        beam_size=_beam_size(),
        condition_on_previous_text=False,
        initial_prompt=_TR_INITIAL_PROMPT if language == "tr" else None,
        fp16=False,
    )
    words: list[WordTiming] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            token = (w.get("word") or "").strip()
            if not token:
                continue
            words.append(
                WordTiming(
                    word=token,
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                )
            )
    text = (result.get("text") or "").strip()
    return TranscriptionResult(
        text=text,
        words=words,
        language=language,
        backend="openai-whisper",
        duration=float(result.get("duration", 0.0) or 0.0),
    )


def assign_word_timings(words: Iterable[WordTiming], text: str) -> list[dict]:
    """Align timed words back onto the original transcript character offsets.

    Whisper returns word tokens but not character positions — this walks the
    transcript and finds the next occurrence of each token, returning a list of
    ``{word, start, end, char_start, char_end}`` dicts. Approximate but good
    enough to jump the audio player to a mention.
    """
    aligned: list[dict] = []
    cursor = 0
    for w in words:
        token = w.word.strip()
        if not token:
            continue
        idx = text.find(token, cursor)
        if idx == -1:
            idx = text.lower().find(token.lower(), cursor)
        if idx == -1:
            aligned.append(
                {
                    "word": token,
                    "start": w.start,
                    "end": w.end,
                    "char_start": -1,
                    "char_end": -1,
                }
            )
            continue
        aligned.append(
            {
                "word": token,
                "start": w.start,
                "end": w.end,
                "char_start": idx,
                "char_end": idx + len(token),
            }
        )
        cursor = idx + len(token)
    return aligned
