// MediaRecorder wrapper with level metering. Works in Chromium/Firefox on
// Linux as well as iOS/Android Safari/Chrome.

export class Recorder {
  constructor({ onLevel } = {}) {
    this.onLevel = onLevel;
    this.stream = null;
    this.mediaRecorder = null;
    this.chunks = [];
    this.audioCtx = null;
    this.rafId = null;
    this.startedAt = 0;
  }

  async start() {
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const mime = pickMime();
    this.mediaRecorder = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined);
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) this.chunks.push(e.data);
    };
    this.mediaRecorder.start(500);
    this.startedAt = performance.now();
    this._startMetering();
  }

  stop() {
    if (!this.mediaRecorder) return Promise.resolve(null);
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = () => {
        this._stopMetering();
        const blob = new Blob(this.chunks, { type: this.mediaRecorder.mimeType || "audio/webm" });
        const duration = (performance.now() - this.startedAt) / 1000;
        if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null;
        resolve({ blob, duration, mime: blob.type });
      };
      if (this.mediaRecorder.state !== "inactive") this.mediaRecorder.stop();
    });
  }

  cancel() {
    this._stopMetering();
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  _startMetering() {
    if (!this.stream || !this.onLevel) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioCtx();
      const source = this.audioCtx.createMediaStreamSource(this.stream);
      const analyser = this.audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const data = new Uint8Array(analyser.frequencyBinCount);
      const loop = () => {
        analyser.getByteFrequencyData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i];
        const avg = sum / data.length / 255; // 0..1
        this.onLevel(Math.min(1, avg * 2));
        this.rafId = requestAnimationFrame(loop);
      };
      loop();
    } catch (e) {
      console.warn("metering failed", e);
    }
  }

  _stopMetering() {
    if (this.rafId) cancelAnimationFrame(this.rafId);
    if (this.audioCtx) this.audioCtx.close().catch(() => {});
    this.audioCtx = null;
    this.rafId = null;
  }
}

function pickMime() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/mpeg",
  ];
  for (const m of candidates) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
  }
  return null;
}

export function fileExtFor(mime) {
  if (!mime) return "webm";
  if (mime.includes("webm")) return "webm";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mp4")) return "m4a";
  if (mime.includes("mpeg")) return "mp3";
  return "webm";
}

export function fmtDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
