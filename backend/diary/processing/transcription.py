"""Self-hosted speech-to-text.

Tries, in order:
  1. ``faster-whisper`` (CTranslate2 backend — fastest CPU option)
  2. ``openai-whisper`` (reference implementation, heavier)
  3. No-op placeholder that returns an empty transcript and a structured hint
     so the client can prompt the user to type the entry manually.

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


def _load_model():
    """Load whichever Whisper backend is available, once, lazily."""
    global _cached_model, _cached_backend

    with _model_lock:
        if _cached_model is not None:
            return _cached_model, _cached_backend

        model_size = getattr(settings, "HATIRLAF_WHISPER_MODEL", "base")

        # Preferred: faster-whisper (CTranslate2). Much lighter on CPU.
        try:
            from faster_whisper import WhisperModel  # type: ignore

            logger.info("Loading faster-whisper model '%s' (CPU)", model_size)
            _cached_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            _cached_backend = "faster-whisper"
            return _cached_model, _cached_backend
        except Exception as exc:  # pragma: no cover - depends on local env
            logger.debug("faster-whisper unavailable: %s", exc)

        # Fallback: reference openai-whisper.
        try:
            import whisper  # type: ignore

            logger.info("Loading openai-whisper model '%s'", model_size)
            _cached_model = whisper.load_model(model_size)
            _cached_backend = "openai-whisper"
            return _cached_model, _cached_backend
        except Exception as exc:
            logger.debug("openai-whisper unavailable: %s", exc)

        _cached_backend = "placeholder"
        return None, _cached_backend


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
        vad_filter=False,
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
