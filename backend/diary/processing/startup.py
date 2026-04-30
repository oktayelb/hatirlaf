"""Process-local startup readiness and coarse model loading progress."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Literal

ComponentKey = Literal["stt", "llm", "savyar"]
ComponentStatus = Literal["pending", "loading", "ready", "skipped", "failed"]


_lock = threading.Lock()
_started = False
_components: dict[ComponentKey, dict] = {
    "stt": {
        "key": "stt",
        "label": "STT",
        "status": "pending",
        "detail": "Whisper bekleniyor.",
        "weight": 35,
        "progress": 0,
        "expected_seconds": 75,
        "started_at": None,
    },
    "llm": {
        "key": "llm",
        "label": "LLM",
        "status": "pending",
        "detail": "Yerel LLM bekleniyor.",
        "weight": 45,
        "progress": 0,
        "expected_seconds": 180,
        "started_at": None,
    },
    "savyar": {
        "key": "savyar",
        "label": "Savyar",
        "status": "pending",
        "detail": "Morfoloji köprüsü bekleniyor.",
        "weight": 20,
        "progress": 0,
        "expected_seconds": 45,
        "started_at": None,
    },
}


def begin() -> None:
    global _started
    with _lock:
        _started = True


def mark_loading(key: ComponentKey, detail: str = "") -> None:
    with _lock:
        component = _components[key]
        component["status"] = "loading"
        component["progress"] = max(int(component.get("progress") or 0), 2)
        component["started_at"] = time.monotonic()
        if detail:
            component["detail"] = detail


def mark_done(key: ComponentKey, status: ComponentStatus, detail: str = "") -> None:
    if status not in {"ready", "skipped", "failed"}:
        raise ValueError("done components must be ready, skipped, or failed")
    with _lock:
        component = _components[key]
        component["status"] = status
        component["progress"] = 100
        component["started_at"] = None
        if detail:
            component["detail"] = detail


def mark_preload_disabled() -> None:
    global _started
    with _lock:
        _started = True
        for component in _components.values():
            component["status"] = "skipped"
            component["detail"] = "Model ön yükleme kapalı."
            component["progress"] = 100
            component["started_at"] = None


def snapshot() -> dict:
    with _lock:
        components = [deepcopy(c) for c in _components.values()]
        started = _started

    total = sum(c["weight"] for c in components)
    loaded = 0.0
    for component in components:
        status = component["status"]
        progress = int(component.get("progress") or 0)
        if status == "loading":
            progress = _estimated_loading_progress(component)
            component["progress"] = progress
        if status in {"ready", "skipped", "failed"}:
            loaded += component["weight"]
        elif status == "loading":
            loaded += component["weight"] * (progress / 100)

    ready = started and all(
        component["status"] in {"ready", "skipped", "failed"}
        for component in components
    )
    current = next(
        (component for component in components if component["status"] == "loading"),
        None,
    )
    pending = sum(1 for component in components if component["status"] == "pending")
    for component in components:
        component.pop("started_at", None)
        component.pop("expected_seconds", None)
    if current is not None:
        current = deepcopy(current)
        current.pop("started_at", None)
        current.pop("expected_seconds", None)
    return {
        "ready": ready,
        "started": started,
        "progress": round((loaded / total) * 100) if total else 100,
        "current": current,
        "pending": pending,
        "components": components,
    }


def _estimated_loading_progress(component: dict) -> int:
    started_at = component.get("started_at")
    if not started_at:
        return max(int(component.get("progress") or 0), 2)
    elapsed = max(time.monotonic() - float(started_at), 0.0)
    expected = max(float(component.get("expected_seconds") or 60), 1.0)
    ratio = elapsed / expected
    # Ease toward 94% while the loader is still active. Actual completion is
    # still reported only by mark_done after the model load returns.
    eased = 1 - ((1 - min(ratio, 1.0)) ** 2)
    return max(2, min(94, round(eased * 94)))
