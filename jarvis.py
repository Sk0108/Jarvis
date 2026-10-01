#!/usr/bin/env python3
"""
Desktop clap listener: reads the default microphone and logs when two loud transients
(a double clap) are detected within a short time window.

Run:
  python -m pip install -r requirements.txt
  python clap_listen.py

Tuning (constants below):
  SAMPLE_RATE   — usually 44100 or 48000; match your device if needed.
  BLOCK_MS      — analysis window size; smaller = snappier, noisier.
  SPIKE_RATIO   — how many times louder than the noise floor counts as a clap;
                    raise if false triggers; lower if claps are missed.
  COOLDOWN_S    — minimum seconds between double-clap logs (debounce).
  MIN_DOUBLE_GAP_S / MAX_DOUBLE_GAP_S — allowed time between the two claps.
  RETRIGGER_RATIO — audio must fall below threshold * this before another hit counts.
  NOISE_FLOOR_ALPHA — closer to 1 = slower baseline adaptation to room noise.
  MIN_RMS       — ignore spikes below this absolute level (float audio ~ [-1, 1]).
  SONG_URI      — Spotify or YouTube URL/URI to open on each double clap (empty = log only).
  FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP — if True, launch VS Code without -n (reuse / focus existing instance).
  OPEN_NEW_CURSOR_ON_DOUBLE_CLAP — if True, also launch VS Code with -n (extra new window; runs after focus launch if both).
  CURSOR_OPEN_FULLSCREEN — Windows: after focus/launch, send F11 to enter VS Code fullscreen (toggle off with F11).
  OPEN_CLAUDE_CODE_IN_CHROME — Claude desktop app if installed, else Claude in Brave (falls back to Chrome).
  OPEN_GMAIL_IN_BRAVE — Gmail inbox in Brave after Spotify (GMAIL_URL, falls back to Chrome).
  CLAUDE_CHROME_MONITOR / GMAIL_CHROME_MONITOR — 1-based display index (Windows: sorted left-to-top).
  CHROME_SEPARATE_SITE_PROFILES — Windows: if True, uses temp --user-data-dir per site (not your normal profile).
    Default False so Claude/Gmail use your usual Chrome profile and logins; enable only if both windows keep
    opening on the same monitor and you accept a separate profile for automation.
  OPEN_CHROME_FULLSCREEN — Fullscreen on the chosen monitor (Windows: new window is detected and snapped with SetWindowPos).
  JARVIS_WELCOME_* — TTS after the song (ElevenLabs). Configure via environment or a `.env`
    file next to this script (ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, etc.).
    With JARVIS_WELCOME_CACHE_ENABLED, audio is saved under `.cache/jarvis_welcome/` (WAV) and
    replayed when phrase + voice + model + format match—no repeat API call. Delete that folder
    or set JARVIS_WELCOME_CACHE_ENABLED=False to force a fresh fetch.
  The welcome sequence runs only once per process. The assistant speaks in the background so Cursor
    opens without waiting for playback to finish (restart the script to run again).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import sounddevice as sd

# --- tuning knobs -----------------------------------------------------------
SAMPLE_RATE = 44100
BLOCK_MS = 40
CHANNELS = 1

SPIKE_RATIO = 7.0
COOLDOWN_S = 0.45
MIN_DOUBLE_GAP_S = 0.05
MAX_DOUBLE_GAP_S = 0.35
RETRIGGER_RATIO = 0.55
NOISE_FLOOR_ALPHA = 0.992
MIN_RMS = 0.012
# Hard ceiling on the clap threshold, regardless of how loud the room's ambient noise gets.
# Without this, threshold = noise_floor * SPIKE_RATIO grows unbounded — in a genuinely loud
# room (noise_floor ~0.2+), that demands a threshold above 1.0, which no real sound can ever
# reach (1.0 is the loudest a mic can report), silently making detection impossible.
SPIKE_THRESHOLD_MAX = 0.35
QUIET_GATE_MULT = 2.2  # update noise floor only when below floor * this
# Startup mic probe: if default input RMS stays below this, scan for a louder device.
INPUT_PROBE_S = 0.5
INPUT_SILENT_RMS = 0.001

# Spotify: "spotify:track:TRACK_ID" or https://open.spotify.com/track/...
# YouTube: https://www.youtube.com/watch?v=...
SONG_URI = "https://open.spotify.com/track/0frKt739Ov9vvKS3JRu5Vi"

# Cursor: focus existing instance (no -n). Set OPEN_NEW_CURSOR_ON_DOUBLE_CLAP for a new window as well.
FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP = True
OPEN_NEW_CURSOR_ON_DOUBLE_CLAP = False
CURSOR_OPEN_FULLSCREEN = True

# Google Chrome (fallback: default browser). URLs overridable in .env.
OPEN_CLAUDE_CODE_IN_CHROME = True
OPEN_GMAIL_IN_BRAVE = True
OPEN_CHROME_FULLSCREEN = True
# False = default Chrome profile (your normal user, extensions, cookies). True = temp dirs under %TEMP% per site.
CHROME_SEPARATE_SITE_PROFILES = False
# Which physical screen (1 = leftmost/top-first after sorting). Windows only; ignored elsewhere.
# Single-monitor setup: both windows target monitor 1.
CLAUDE_CHROME_MONITOR = 1
GMAIL_CHROME_MONITOR = 1

JARVIS_WELCOME_ENABLED = True
JARVIS_WELCOME_PHRASE = (
    "Welcome home sir. "
    "Congratulations on the new client for your SaaS app—make sure to follow up. "
    "If it helps: a short, specific note while the deal is still fresh usually "
    "anchors trust better than a polished deck sent cold a few days later."
)
# Name JARVIS uses to address the user.
JARVIS_ADDRESS_NAME = "Miss Khan"
# If True, ignore JARVIS_WELCOME_PHRASE and generate a fresh, personality-driven greeting each
# run via the Anthropic API — optionally grounded in the latest unread Gmail message. Requires
# ANTHROPIC_API_KEY in the environment/.env. Falls back to a plain time-of-day greeting if the
# API key, Gmail credentials, or either API call is unavailable.
JARVIS_DYNAMIC_GREETING_ENABLED = True
# Which LLM powers the dynamic greeting and "Jarvis" conversation: "anthropic", "gemini", or
# "groq". Groq (console.groq.com) has a generous free daily tier, no card required, and is
# very fast — good fallback if Gemini's free tier runs dry. Needs: pip install groq
LLM_PROVIDER = "groq"
JARVIS_GREETING_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-3.6-flash"
# openai/gpt-oss-120b — Groq deprecated its older llama-3.3-70b-versatile / llama-3.1-8b-instant
# chat models; this is their current recommended general-purpose model with tool-use support.
GROQ_MODEL = "openai/gpt-oss-120b"
# Gmail read-only access for the "latest unread mail" context. One-time setup required:
#   1. In Google Cloud Console, create a project, enable the "Gmail API".
#   2. Create OAuth client credentials of type "Desktop app".
#   3. Download the JSON and save it next to this script as the filename below.
#   4. First run opens a browser once for consent; after that a token is cached and reused.
GMAIL_CREDENTIALS_PATH = "credentials.json"
GMAIL_TOKEN_PATH = ".cache/gmail_token.json"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# --- voice conversation ("Jarvis" wake word) ---
# One-time installs: pip install openwakeword onnxruntime scipy
# Uses the same ELEVENLABS_API_KEY (speech-to-text + text-to-speech) and ANTHROPIC_API_KEY
# (conversation + Gmail tool use) already configured above.
JARVIS_VOICE_ENABLED = True
WAKE_WORD_MODEL = "hey_jarvis"  # openwakeword's built-in "Hey Jarvis" model
WAKE_WORD_THRESHOLD = 0.5
WAKE_WORD_COOLDOWN_S = 2.0
VOICE_SILENCE_RMS_MULT = 1.6  # audio below noise_floor * this counts as "quiet"
VOICE_UTTERANCE_SILENCE_S = 0.6  # pause length that ends an utterance
VOICE_UTTERANCE_MAX_S = 8.0
VOICE_MIN_UTTERANCE_S = 0.35  # ignore accidental blips shorter than this
VOICE_FOLLOWUP_WINDOW_S = 8.0  # after a reply, how long to wait for a follow-up before
# requiring the wake word again
# A clap is a single sharp transient; speech is sustained. Require this many consecutive
# loud blocks before treating follow-up-window audio as "you're talking" rather than a clap.
FOLLOWUP_SUSTAIN_BLOCKS = 3
# If ambient noise stays loud (not a brief clap/spike, but sustained) for this long, treat it
# as the room's new baseline and let the noise floor catch up — otherwise a noisy environment
# permanently freezes the floor at a stale, quieter estimate, and detection locks up entirely
# after the first sound because it can never register the room as "quiet" again.
NOISE_FLOOR_CATCHUP_S = 2.0
NOISE_FLOOR_CATCHUP_ALPHA = 0.98
# After JARVIS finishes speaking, ignore audio for this long — the tail end of its own voice
# through your speakers otherwise bleeds back into the mic and gets mistaken for you talking
# (or worse, gets transcribed as part of your next sentence). Not a substitute for real echo
# cancellation, but a cheap guard against the worst of it.
ECHO_GUARD_S = 0.6
ELEVENLABS_STT_MODEL_ID = "scribe_v1"
# Skip offering the Gmail tool at all when the question clearly isn't about email — saves an
# API call on ordinary chit-chat ("tell me a joke", "how's your day") since the model doesn't
# have to spend a turn deciding not to search. Errs toward keeping the tool available when in
# doubt, since a false negative just means it can't check mail, while a false positive only
# costs one extra call.
GMAIL_INTENT_KEYWORDS = (
    "email", "e-mail", "mail", "gmail", "inbox", "unread", "message", "messages",
    "subject", "from ", "reply", "replied", "sent", "attachment", "interview",
    "assignment", "deadline", "due", "application", "applied", "recruiter",
    "offer", "invite", "invitation", "schedule", "meeting",
)


def _looks_like_gmail_question(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in GMAIL_INTENT_KEYWORDS)

_assistant_speaking = threading.Event()
_assistant_spoke_until_ts = 0.0
_voice_followup_deadline_ts = 0.0
_voice_history: list = []
_hotkey_jarvis_event = threading.Event()

# Manual hotkeys — 100% reliable regardless of room noise, as a backup for clap/wake-word
# detection in loud environments. Needs: pip install keyboard
# On Windows, global hotkeys may require running the terminal "as Administrator" to register.
HOTKEY_DOUBLE_CLAP = "ctrl+alt+j"  # same actions as a double clap: Spotify, Claude, Gmail, etc.
HOTKEY_JARVIS = "ctrl+alt+k"  # starts a Jarvis conversation turn, same as saying "Jarvis"


def _register_hotkeys() -> None:
    try:
        import keyboard
    except ImportError:
        log.warning(
            'Hotkeys disabled (pip install keyboard to enable %s / %s as manual backups).',
            HOTKEY_DOUBLE_CLAP,
            HOTKEY_JARVIS,
        )
        return
    try:
        keyboard.add_hotkey(
            HOTKEY_DOUBLE_CLAP,
            lambda: threading.Thread(target=run_double_clap_actions, daemon=True).start(),
        )
        keyboard.add_hotkey(HOTKEY_JARVIS, _hotkey_jarvis_event.set)
        log.info(
            'Hotkeys ready: %s = double-clap actions, %s = start a Jarvis conversation.',
            HOTKEY_DOUBLE_CLAP,
            HOTKEY_JARVIS,
        )
    except Exception as e:
        log.warning(
            "Could not register hotkeys (%s). On Windows, try running the terminal "
            "as Administrator.",
            e,
        )

GMAIL_TOOL_SPEC = {
    "name": "search_gmail",
    "description": (
        "Search the user's Gmail inbox using Gmail search syntax, e.g. 'is:unread', "
        "'from:someone@example.com', 'subject:invoice'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Gmail search query"},
            "max_results": {
                "type": "integer",
                "description": "Max emails to return (default 5)",
            },
        },
        "required": ["query"],
    },
}
# Same tool, OpenAI/Groq function-calling shape ("parameters" instead of "input_schema").
GMAIL_TOOL_SPEC_OPENAI = {
    "name": GMAIL_TOOL_SPEC["name"],
    "description": GMAIL_TOOL_SPEC["description"],
    "parameters": GMAIL_TOOL_SPEC["input_schema"],
}
# Seconds after launching SONG_URI before speaking (gives Spotify/browser time to start).
JARVIS_AFTER_SONG_DELAY_S = 1.0
# Save ElevenLabs PCM as WAV under .cache/jarvis_welcome/; replay skips the API when the key matches.
JARVIS_WELCOME_CACHE_ENABLED = True

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("clap_listen")


def block_samples() -> int:
    n = int(SAMPLE_RATE * BLOCK_MS / 1000)
    return max(n, 1)


def rms_mono(block: np.ndarray) -> float:
    if block.ndim > 1:
        block = np.mean(block.astype(np.float64), axis=1)
    else:
        block = block.astype(np.float64)
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block**2)))


def _input_devices() -> list[tuple[int, dict]]:
    return [
        (i, dev)
        for i, dev in enumerate(sd.query_devices())
        if dev["max_input_channels"] >= 1
    ]


def _resolve_input_device_index(spec: str) -> int:
    spec = spec.strip()
    if spec.isdigit():
        idx = int(spec)
        sd.query_devices(idx)
        return idx
    needle = spec.lower()
    for idx, dev in _input_devices():
        if needle in dev["name"].lower():
            return idx
    raise ValueError(f"No input device matches {spec!r}")


def _probe_input_max_rms(device: int, blocksize: int) -> float | None:
    dev_info = sd.query_devices(device)
    attempts = [
        (SAMPLE_RATE, CHANNELS),
        (int(dev_info["default_samplerate"]), CHANNELS),
        (int(dev_info["default_samplerate"]), min(dev_info["max_input_channels"], CHANNELS) or 1),
    ]
    last_err: Exception | None = None
    for samplerate, channels in attempts:
        try:
            with sd.InputStream(
                device=device,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
            ) as stream:
                peak = 0.0
                deadline = time.monotonic() + INPUT_PROBE_S
                while time.monotonic() < deadline:
                    data, _ = stream.read(blocksize)
                    peak = max(peak, rms_mono(data))
                return peak
        except (sd.PortAudioError, ValueError) as e:
            last_err = e
            continue
        log.warning("Probe failed for device %d (%s): %s", device, dev_info["name"], last_err)
    return None


def _choose_input_device(blocksize: int) -> int:
    log.info("Audio devices:\n%s", sd.query_devices())

    override = (os.environ.get("JARVIS_INPUT_DEVICE") or "").strip()
    if override:
        try:
            idx = _resolve_input_device_index(override)
        except ValueError as e:
            log.error("%s", e)
            log.error("Set JARVIS_INPUT_DEVICE to a device index or name substring.")
            raise SystemExit(1) from e
        name = sd.query_devices(idx)["name"]
        peak = _probe_input_max_rms(idx, blocksize)
        log.info("Using JARVIS_INPUT_DEVICE [%d]: %s", idx, name)
        if peak is None:
            log.warning("Could not open configured mic; trying anyway.")
        elif peak < INPUT_SILENT_RMS:
            log.warning(
                "Configured mic looks silent (probe rms=%.5f). "
                "Check Windows input level or try another JARVIS_INPUT_DEVICE.",
                peak,
            )
        else:
            log.info("Mic probe OK (rms=%.5f).", peak)
        return idx

    default = sd.default.device[0]
    if default is not None and default >= 0:
        default_name = sd.query_devices(default)["name"]
        peak = _probe_input_max_rms(default, blocksize)
        if peak is not None and peak >= INPUT_SILENT_RMS:
            log.info(
                "Using default microphone [%d]: %s (probe rms=%.5f)",
                default,
                default_name,
                peak,
            )
            return default
        log.warning(
            "Default mic [%d] %s is silent or unavailable (probe rms=%s); "
            "scanning other inputs...",
            default,
            default_name,
            f"{peak:.5f}" if peak is not None else "unopenable",
        )

    best_idx: int | None = None
    best_peak = -1.0
    for idx, dev in _input_devices():
        if default is not None and idx == default:
            continue
        peak = _probe_input_max_rms(idx, blocksize)
        if peak is not None and peak > best_peak:
            best_peak = peak
            best_idx = idx

    if best_idx is not None and best_peak >= INPUT_SILENT_RMS:
        log.info(
            "Auto-selected microphone [%d]: %s (probe rms=%.5f)",
            best_idx,
            sd.query_devices(best_idx)["name"],
            best_peak,
        )
        return best_idx

    if default is not None and default >= 0:
        log.warning("No active mic found; falling back to default [%d].", default)
        return default
    inputs = _input_devices()
    if not inputs:
        log.error("No input devices found.")
        raise SystemExit(1)
    idx, dev = inputs[0]
    log.warning("No active mic found; falling back to [%d] %s.", idx, dev["name"])
    return idx


def _elevenlabs_pcm_sample_rate(output_format: str) -> int:
    override = (os.environ.get("ELEVENLABS_PCM_SAMPLE_RATE") or "").strip()
    if override.isdigit():
        return int(override)
    if output_format.startswith("pcm_"):
        try:
            return int(output_format.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            pass
    return 24000


def elevenlabs_env_config() -> tuple[str, str, str, int]:
    """voice_id, model_id, output_format, pcm_sample_rate."""
    voice = (os.environ.get("ELEVENLABS_VOICE_ID") or "").strip()
    model = (os.environ.get("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2").strip()
    fmt = (os.environ.get("ELEVENLABS_OUTPUT_FORMAT") or "pcm_24000").strip()
    rate = _elevenlabs_pcm_sample_rate(fmt)
    return voice, model, fmt, rate


def _jarvis_welcome_cache_dir() -> Path:
    base = Path(__file__).resolve().parent
    override = (os.environ.get("JARVIS_WELCOME_CACHE_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return base / ".cache" / "jarvis_welcome"


def _jarvis_welcome_cache_path(
    text: str, voice_id: str, model_id: str, output_format: str
) -> Path:
    key = f"{text}|{voice_id}|{model_id}|{output_format}".encode()
    digest = hashlib.sha256(key).hexdigest()[:24]
    return _jarvis_welcome_cache_dir() / f"{digest}.wav"


def _play_pcm_wav_file(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            rate = wf.getframerate()
            if ch != 1 or sw != 2:
                log.warning("Unsupported cached WAV (channels=%s, width=%s).", ch, sw)
                return False
            raw = wf.readframes(wf.getnframes())
    except (OSError, wave.Error) as e:
        log.warning("Could not read cached welcome audio: %s", e)
        return False
    if not raw:
        return False
    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    pcm_f = pcm_i16.astype(np.float32) / 32768.0
    try:
        sd.play(pcm_f, rate)
        sd.wait()
    except Exception as e:
        log.warning("Could not play cached welcome audio: %s", e)
        return False
    return True


def _save_pcm_wav_file(path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        tmp.replace(path)
    except OSError:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def _gmail_service():
    """Builds an authorized Gmail API client, running one-time OAuth consent if needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        log.warning(
            "Gmail summary needs: pip install google-auth google-auth-oauthlib "
            "google-api-python-client"
        )
        return None

    base = Path(__file__).resolve().parent
    creds_path = base / GMAIL_CREDENTIALS_PATH
    token_path = base / GMAIL_TOKEN_PATH

    creds = None
    if token_path.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
        except (OSError, ValueError) as e:
            log.warning("Could not read cached Gmail token: %s", e)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                log.warning("Gmail token refresh failed: %s", e)
                creds = None
        if not creds or not creds.valid:
            if not creds_path.is_file():
                log.warning(
                    "Gmail: %s not found next to jarvis.py; skipping inbox context. "
                    "See GMAIL_CREDENTIALS_PATH setup notes in the script header.",
                    GMAIL_CREDENTIALS_PATH,
                )
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), GMAIL_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as e:
                log.warning("Gmail OAuth consent failed: %s", e)
                return None
        try:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())
        except OSError as e:
            log.warning("Could not cache Gmail token: %s", e)

    try:
        return build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        log.warning("Could not start Gmail client: %s", e)
        return None


def _latest_unread_email_context() -> str | None:
    """Returns a short 'From X, subject Y: snippet' string for the newest unread inbox mail."""
    service = _gmail_service()
    if service is None:
        return None
    try:
        resp = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["INBOX", "UNREAD"], maxResults=1)
            .execute()
        )
        msgs = resp.get("messages", [])
        if not msgs:
            return None
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=msgs[0]["id"],
                format="metadata",
                metadataHeaders=["From", "Subject"],
            )
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        sender = headers.get("From", "an unknown sender")
        subject = headers.get("Subject", "(no subject)")
        snippet = (msg.get("snippet") or "").strip()
        return f"From {sender}, subject \"{subject}\": {snippet}"
    except Exception as e:
        log.warning("Could not fetch Gmail summary: %s", e)
        return None


def _resample_ratio() -> tuple[int, int]:
    from fractions import Fraction

    frac = Fraction(16000, SAMPLE_RATE).limit_denominator(1000)
    return frac.numerator, frac.denominator


def _resample_block_to_16k_int16(block_f32: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    up, down = _resample_ratio()
    mono = block_f32.reshape(-1)
    resampled = resample_poly(mono, up, down)
    clipped = np.clip(resampled, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


class _WakeWordDetector:
    """Feed float32 mic blocks at SAMPLE_RATE; returns True on a "Jarvis" wake trigger."""

    def __init__(self) -> None:
        self._ok = False
        self._model = None
        self._buf = np.zeros(0, dtype=np.int16)
        self._chunk = 1280  # ~80ms at 16kHz, openwakeword's expected frame size
        self.last_score = 0.0
        self._last_trigger = 0.0
        try:
            from openwakeword.model import Model
            from openwakeword import utils as owwutils

            try:
                owwutils.download_models(model_names=[WAKE_WORD_MODEL])
            except Exception as e:
                log.warning("Could not auto-download wake word model files: %s", e)

            self._model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
            self._ok = True
        except Exception as e:
            log.warning(
                "Wake word disabled (%s). Install with: pip install openwakeword onnxruntime scipy",
                e,
            )

    @property
    def available(self) -> bool:
        return self._ok

    def feed(self, block_f32: np.ndarray) -> bool:
        if not self._ok:
            return False
        try:
            pcm16 = _resample_block_to_16k_int16(block_f32)
            self._buf = np.concatenate([self._buf, pcm16])
            triggered = False
            while len(self._buf) >= self._chunk:
                chunk = self._buf[: self._chunk]
                self._buf = self._buf[self._chunk :]
                scores = self._model.predict(chunk)
                score = float(scores.get(WAKE_WORD_MODEL, 0.0))
                self.last_score = max(self.last_score * 0.7, score)  # decay for display only
                now = time.monotonic()
                if (
                    score >= WAKE_WORD_THRESHOLD
                    and (now - self._last_trigger) >= WAKE_WORD_COOLDOWN_S
                ):
                    self._last_trigger = now
                    triggered = True
            return triggered
        except Exception as e:
            # Transient per-block error (e.g. a resampling hiccup) — skip this block only.
            # Never permanently disable wake word detection over a one-off glitch.
            log.warning("Wake word: skipped one audio block after error: %s", e)
            self._buf = np.zeros(0, dtype=np.int16)
            return False


def _record_utterance_blocking(
    stream: "sd.InputStream", blocksize: int, noise_floor: float
) -> np.ndarray | None:
    """Blocks the calling thread reading from `stream` until the speaker pauses.

    Recalibrates the silence floor fresh from the current room right before recording,
    instead of trusting the slow-moving background estimate from the clap detector — that
    estimate barely updates and goes stale the moment ambient sound changes (music, fans),
    which otherwise makes "you've gone quiet" nearly impossible to detect. The calibrated
    floor is capped so a loud burst during the calibration window (e.g. music still playing)
    can't push the threshold so high that normal speech pauses look like silence.
    """
    block_dur = blocksize / SAMPLE_RATE
    calib_blocks = max(1, int(0.25 / block_dur))
    calib_levels = []
    for _ in range(calib_blocks):
        data, _overflowed = stream.read(blocksize)
        calib_levels.append(rms_mono(data))
    ambient_raw = max(float(np.median(calib_levels)), noise_floor, 1e-6)
    ambient = min(ambient_raw, MIN_RMS * 3)  # cap: don't let loud background inflate this
    silence_thresh = max(ambient * VOICE_SILENCE_RMS_MULT, MIN_RMS * 0.5)

    frames: list[np.ndarray] = []
    quiet_for = 0.0
    started = time.monotonic()
    while True:
        data, _overflowed = stream.read(blocksize)
        frames.append(data.copy())
        level = rms_mono(data)
        quiet_for = quiet_for + block_dur if level < silence_thresh else 0.0
        elapsed = time.monotonic() - started
        if quiet_for >= VOICE_UTTERANCE_SILENCE_S or elapsed >= VOICE_UTTERANCE_MAX_S:
            break
    audio = np.concatenate(frames).reshape(-1) if frames else np.zeros(0, dtype=np.float32)
    trim_samples = int(VOICE_UTTERANCE_SILENCE_S * SAMPLE_RATE * 0.8)
    if len(audio) > trim_samples:
        audio = audio[:-trim_samples]
    duration_s = len(audio) / SAMPLE_RATE
    if duration_s < VOICE_MIN_UTTERANCE_S:
        log.info(
            "Heard %.2fs — too short to process (background noise may be masking your "
            "voice; try again, or lower Spotify/speaker volume while talking to Jarvis).",
            duration_s,
        )
        return None
    return audio


def _transcribe_utterance(pcm_f32: np.ndarray) -> str | None:
    api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key or pcm_f32.size == 0:
        return None
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        log.warning("Install dependencies: pip install elevenlabs")
        return None
    pcm_clipped = np.clip(pcm_f32, -1.0, 1.0)
    pcm_i16 = (pcm_clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_i16.tobytes())
    buf.seek(0)
    buf.name = "utterance.wav"
    try:
        client = ElevenLabs(api_key=api_key)
        result = client.speech_to_text.convert(file=buf, model_id=ELEVENLABS_STT_MODEL_ID)
        return (getattr(result, "text", "") or "").strip() or None
    except Exception as e:
        log.warning("ElevenLabs speech-to-text failed: %s", e)
        return None


def _gmail_search(query: str, max_results: int = 5) -> str:
    service = _gmail_service()
    if service is None:
        return "Gmail is not connected."
    try:
        resp = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max(1, min(int(max_results or 5), 10)))
            .execute()
        )
        msgs = resp.get("messages", [])
        if not msgs:
            return "No matching emails found."
        lines = []
        for m in msgs:
            full = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=m["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
            h = {x["name"]: x["value"] for x in full.get("payload", {}).get("headers", [])}
            snippet = (full.get("snippet") or "").strip()
            lines.append(
                f"From {h.get('From', '?')} | {h.get('Subject', '(no subject)')} "
                f"| {h.get('Date', '')}: {snippet}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Gmail search failed: {e}"


def _run_claude_with_gmail_tool(history: list) -> str | None:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return "I can't reach Claude right now — the API key isn't set."
    try:
        import anthropic
    except ImportError:
        return "The anthropic package isn't installed."
    client = anthropic.Anthropic(api_key=api_key)
    system = (
        f"You are JARVIS, Tony Stark's AI assistant, speaking with {JARVIS_ADDRESS_NAME}. "
        "Composed, dry-witted, concise, unfailingly loyal. Keep replies to 1-3 sentences "
        "since they are read aloud. Use the search_gmail tool whenever the question needs "
        "current inbox information; otherwise answer directly. No emoji or markdown."
    )
    latest_user_text = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
    )
    offer_tool = isinstance(latest_user_text, str) and _looks_like_gmail_question(latest_user_text)
    messages = list(history)
    for _ in range(3):
        try:
            create_kwargs = dict(
                model=JARVIS_GREETING_MODEL, max_tokens=300, system=system, messages=messages
            )
            if offer_tool:
                create_kwargs["tools"] = [GMAIL_TOOL_SPEC]
            resp = client.messages.create(**create_kwargs)
        except Exception as e:
            return f"I ran into a problem reaching Claude: {e}"
        if resp.stop_reason != "tool_use":
            return "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "search_gmail":
                result_text = _gmail_search(
                    block.input.get("query", ""), block.input.get("max_results", 5)
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )
        messages.append({"role": "user", "content": tool_results})
    return "I got a bit tangled looking through your mail — try asking again."


_gemini_chat_session = None


def _get_gemini_chat_session():
    global _gemini_chat_session
    if _gemini_chat_session is not None:
        return _gemini_chat_session
    import google.generativeai as genai

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    genai.configure(api_key=api_key)
    system = (
        f"You are JARVIS, Tony Stark's AI assistant, speaking with {JARVIS_ADDRESS_NAME}. "
        "Composed, dry-witted, concise, unfailingly loyal. Keep replies to 1-3 sentences "
        "since they are read aloud. Call search_gmail whenever the question needs current "
        "inbox information; otherwise answer directly. No emoji or markdown."
    )

    def search_gmail(query: str, max_results: int = 5) -> str:
        """Search the user's Gmail inbox using Gmail search syntax, e.g. 'is:unread',
        'from:someone@example.com', 'subject:invoice'."""
        return _gmail_search(query, max_results)

    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system, tools=[search_gmail])
    _gemini_chat_session = model.start_chat(enable_automatic_function_calling=True)
    return _gemini_chat_session


def _run_gemini_turn(user_text: str) -> str | None:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return "I can't reach Gemini right now — the API key isn't set."
    try:
        import google.generativeai as genai  # noqa: F401  (surfaces ImportError here)
    except ImportError:
        return "The google-generativeai package isn't installed."
    try:
        chat = _get_gemini_chat_session()
        resp = chat.send_message(user_text)
        return (resp.text or "").strip() or None
    except Exception as e:
        return f"I ran into a problem reaching Gemini: {e}"


_groq_messages: list = []


def _run_groq_turn(user_text: str) -> str | None:
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        return "I can't reach Groq right now — the API key isn't set."
    try:
        from groq import Groq
    except ImportError:
        return "The groq package isn't installed."

    if not _groq_messages:
        _groq_messages.append(
            {
                "role": "system",
                "content": (
                    f"You are JARVIS, Tony Stark's AI assistant, speaking with "
                    f"{JARVIS_ADDRESS_NAME}. Composed, dry-witted, concise, unfailingly "
                    "loyal. Keep replies to 1-3 sentences since they are read aloud. Use "
                    "the search_gmail tool whenever the question needs current inbox "
                    "information; otherwise answer directly. No emoji or markdown."
                ),
            }
        )
    _groq_messages.append({"role": "user", "content": user_text})

    # Always declare the tool (Groq/OpenAI-style APIs get confused if a tool call shows up in
    # history from an earlier turn but the tool definition is missing this time) — instead,
    # explicitly forbid calling it via tool_choice="none" when the heuristic says it's not
    # needed. This still avoids most of the extra round-trip cost on obvious non-email
    # questions, without risking the "tool choice is none, but model called a tool" 400 error
    # that omitting the tool definition entirely can trigger.
    tools = [{"type": "function", "function": GMAIL_TOOL_SPEC_OPENAI}]
    tool_choice = "auto" if _looks_like_gmail_question(user_text) else "none"
    client = Groq(api_key=api_key)
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=_groq_messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=300,
            )
        except Exception as e:
            return f"I ran into a problem reaching Groq: {e}"
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            content = (msg.content or "").strip()
            _groq_messages.append({"role": "assistant", "content": content})
            return content or None
        _groq_messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            if tc.function.name == "search_gmail":
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result_text = _gmail_search(args.get("query", ""), args.get("max_results", 5))
            else:
                result_text = "Unknown tool."
            _groq_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_text}
            )
    return "I got a bit tangled looking through your mail — try asking again."


def _handle_voice_turn(pcm_f32: np.ndarray) -> None:
    global _voice_followup_deadline_ts
    text = _transcribe_utterance(pcm_f32)
    if not text:
        return
    log.info("You said: %s", text)

    if LLM_PROVIDER == "gemini":
        reply = _run_gemini_turn(text)
    elif LLM_PROVIDER == "groq":
        reply = _run_groq_turn(text)
    else:
        _voice_history.append({"role": "user", "content": text})
        reply = _run_claude_with_gmail_tool(_voice_history)
        if reply:
            _voice_history.append({"role": "assistant", "content": reply})

    if reply:
        log.info("JARVIS: %s", reply)
        _assistant_speaking.set()
        try:
            say_jarvis_welcome(reply)
        finally:
            _assistant_speaking.clear()
            global _assistant_spoke_until_ts
            _assistant_spoke_until_ts = time.monotonic() + ECHO_GUARD_S
    _voice_followup_deadline_ts = time.monotonic() + VOICE_FOLLOWUP_WINDOW_S


def _time_of_day_greeting() -> str:
    hour = time.localtime().tm_hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def _build_greeting_text() -> str:
    """Builds the spoken greeting: a JARVIS-styled line, grounded in the latest unread email
    when available, falling back to a plain greeting if any step is unavailable."""
    fallback = f"{_time_of_day_greeting()}, {JARVIS_ADDRESS_NAME}."
    if not JARVIS_DYNAMIC_GREETING_ENABLED:
        return JARVIS_WELCOME_PHRASE.strip() or fallback

    email_context = _latest_unread_email_context()
    user_context = (
        f"Latest unread email — {email_context}"
        if email_context
        else "No unread email context is available right now."
    )
    system = (
        "You are JARVIS: Tony Stark's AI assistant. Composed, dry-witted, unfailingly "
        f"loyal, and precise. You address the user only as '{JARVIS_ADDRESS_NAME}'. "
        f"Open with '{_time_of_day_greeting()}, {JARVIS_ADDRESS_NAME}.' — then in 2-3 "
        "more short sentences, naturally weave in the email context you're given if any, "
        "and offer one genuinely useful, specific observation or nudge (not generic "
        "filler). No emoji, no markdown, no stage directions — this is read aloud as-is."
    )

    if LLM_PROVIDER == "gemini":
        text = _generate_greeting_gemini(system, user_context)
    elif LLM_PROVIDER == "groq":
        text = _generate_greeting_groq(system, user_context)
    else:
        text = _generate_greeting_anthropic(system, user_context)
    return text or fallback


def _generate_greeting_anthropic(system: str, user_context: str) -> str | None:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set; using a plain greeting instead of a generated one.")
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("Dynamic greeting needs: pip install anthropic")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=JARVIS_GREETING_MODEL,
            max_tokens=180,
            system=system,
            messages=[{"role": "user", "content": user_context}],
        )
        return "".join(
            b.text for b in resp.content if getattr(b, "type", "") == "text"
        ).strip() or None
    except Exception as e:
        log.warning("Dynamic greeting generation (Anthropic) failed: %s", e)
        return None


def _generate_greeting_gemini(system: str, user_context: str) -> str | None:
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        log.warning("GEMINI_API_KEY not set; using a plain greeting instead of a generated one.")
        return None
    try:
        import google.generativeai as genai
    except ImportError:
        log.warning("Dynamic greeting needs: pip install google-generativeai")
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)
        resp = model.generate_content(user_context)
        return (resp.text or "").strip() or None
    except Exception as e:
        log.warning("Dynamic greeting generation (Gemini) failed: %s", e)
        return None


def _generate_greeting_groq(system: str, user_context: str) -> str | None:
    api_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not api_key:
        log.warning("GROQ_API_KEY not set; using a plain greeting instead of a generated one.")
        return None
    try:
        from groq import Groq
    except ImportError:
        log.warning("Dynamic greeting needs: pip install groq")
        return None
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=180,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_context},
            ],
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as e:
        log.warning("Dynamic greeting generation (Groq) failed: %s", e)
        return None


def say_jarvis_welcome(text: str | None = None) -> None:
    if not JARVIS_WELCOME_ENABLED:
        return
    text = (text if text is not None else JARVIS_WELCOME_PHRASE).strip()
    if not text:
        return
    vid, model_id, output_format, pcm_rate = elevenlabs_env_config()
    if not vid:
        log.warning("Set ELEVENLABS_VOICE_ID in the environment for ElevenLabs TTS.")
        return

    cache_path = _jarvis_welcome_cache_path(text, vid, model_id, output_format)
    if JARVIS_WELCOME_CACHE_ENABLED and cache_path.is_file():
        log.info("Playing welcome from cache: %s", cache_path)
        if _play_pcm_wav_file(cache_path):
            return
        log.warning("Cache miss after read failure; fetching from ElevenLabs.")

    api_key = (os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        log.warning("Set ELEVENLABS_API_KEY in the environment for ElevenLabs TTS.")
        return
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        log.warning("Install dependencies: pip install -r requirements.txt")
        return
    try:
        client = ElevenLabs(api_key=api_key)
        chunks = client.text_to_speech.convert(
            voice_id=vid,
            text=text,
            model_id=model_id,
            output_format=output_format,
        )
        raw = b"".join(chunks)
    except Exception as e:
        log.warning("ElevenLabs TTS failed: %s", e)
        return
    if not raw:
        log.warning("ElevenLabs returned empty audio.")
        return
    if JARVIS_WELCOME_CACHE_ENABLED:
        try:
            _save_pcm_wav_file(cache_path, raw, pcm_rate)
            log.info("Saved welcome audio to cache: %s", cache_path)
        except OSError as e:
            log.warning("Could not save welcome cache: %s", e)
    pcm_i16 = np.frombuffer(raw, dtype=np.int16)
    pcm_f = pcm_i16.astype(np.float32) / 32768.0
    try:
        sd.play(pcm_f, pcm_rate)
        sd.wait()
    except Exception as e:
        log.warning("Could not play ElevenLabs audio: %s", e)


def play_song(uri: str) -> None:
    u = uri.strip()
    if not u:
        return
    try:
        if sys.platform == "win32":
            os.startfile(u)
        else:
            webbrowser.open(u)
    except OSError as e:
        log.warning("Could not open SONG_URI: %s", e)


def _chrome_executable() -> str | None:
    if sys.platform == "win32":
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if not base:
                continue
            p = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
            if os.path.isfile(p):
                return p
    return shutil.which("google-chrome") or shutil.which("chrome")


def _brave_executable() -> str | None:
    if sys.platform == "win32":
        for base in (
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ):
            if not base:
                continue
            p = os.path.join(base, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
            if os.path.isfile(p):
                return p
    return shutil.which("brave") or shutil.which("brave-browser")


def _claude_desktop_executable() -> str | None:
    """Locates the Claude desktop app (not Claude Code). Returns None if not installed."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = []
        if local:
            candidates.append(os.path.join(local, "AnthropicClaude", "claude.exe"))
            candidates.append(os.path.join(local, "Programs", "Claude", "Claude.exe"))
            candidates.append(os.path.join(local, "Programs", "AnthropicClaude", "Claude.exe"))
        for p in candidates:
            if os.path.isfile(p):
                return p
    elif sys.platform == "darwin":
        p = "/Applications/Claude.app"
        if os.path.isdir(p):
            return p
    return None


def _win32_sorted_monitor_rects() -> list[tuple[int, int, int, int]]:
    """Each monitor as (left, top, right, bottom), sorted left-to-right then top-to-bottom."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    collected: list[tuple[int, int, int, int]] = []

    @ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )
    def _cb(_hm, _hdc, lprc, _lp):
        r = lprc.contents
        collected.append((int(r.left), int(r.top), int(r.right), int(r.bottom)))
        return True

    ctypes.windll.user32.EnumDisplayMonitors(None, None, _cb, 0)
    collected.sort(key=lambda t: (t[0], t[1]))
    return collected


def _chrome_monitor_top_left(one_based_index: int) -> tuple[int, int]:
    """Top-left corner on virtual desktop for monitor N (1-based)."""
    l, t, _, _ = _chrome_monitor_bounds(one_based_index)
    return (l, t)


def _chrome_monitor_bounds(one_based_index: int) -> tuple[int, int, int, int]:
    """Monitor N as (left, top, right, bottom), 1-based index (sorted like other Chrome helpers)."""
    rects = _win32_sorted_monitor_rects()
    if not rects:
        return (0, 0, 1920, 1080)
    idx = one_based_index - 1
    if idx < 0:
        idx = 0
    if idx >= len(rects):
        log.warning(
            "Monitor %d requested but only %d found; using last monitor.",
            one_based_index,
            len(rects),
        )
        idx = len(rects) - 1
    return rects[idx]


def _chrome_monitor_pixel_size(one_based_index: int) -> tuple[int, int]:
    l, t, r, b = _chrome_monitor_bounds(one_based_index)
    return (max(320, r - l), max(240, b - t))


def _chrome_window_size() -> tuple[int, int]:
    w = (os.environ.get("CHROME_WINDOW_WIDTH") or "1400").strip()
    h = (os.environ.get("CHROME_WINDOW_HEIGHT") or "900").strip()
    try:
        return (max(400, int(w)), max(300, int(h)))
    except ValueError:
        return (1400, 900)


def _chrome_site_user_data_dir(site_key: str) -> str:
    p = Path(tempfile.gettempdir()) / "clap-trigger-chrome" / site_key
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _chrome_new_window_wait_timeout_s() -> float:
    try:
        return max(2.0, float((os.environ.get("CHROME_NEW_WINDOW_WAIT_S") or "6").strip()))
    except ValueError:
        return 6.0


def _chrome_top_level_browser_hwnds_win32(exe_basename: str = "chrome.exe") -> set[int]:
    """HWND ints for visible-or-minimized top-level browser windows matching exe_basename."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    GW_OWNER = 4
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    found: set[int] = set()
    target = exe_basename.lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: wintypes.HWND, _lp: wintypes.LPARAM) -> bool:
        if user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        if not user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return True
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return True
        try:
            buf = ctypes.create_unicode_buffer(4096)
            sz = wintypes.DWORD(len(buf))
            if not kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(sz)):
                return True
            exe_path = buf.value
        finally:
            kernel32.CloseHandle(hproc)
        if os.path.basename(exe_path).lower() != target:
            return True
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < 80 or h < 80:
            return True
        found.add(int(hwnd))
        return True

    user32.EnumWindows(_enum, 0)
    return found


def _wait_new_chrome_hwnd_win32(
    before: set[int], timeout: float, exe_basename: str = "chrome.exe"
) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.12)
        now = _chrome_top_level_browser_hwnds_win32(exe_basename)
        new = now - before
        if not new:
            continue
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        best: int | None = None
        best_area = 0
        for h in new:
            r = wintypes.RECT()
            if user32.GetWindowRect(h, ctypes.byref(r)):
                a = max(0, r.right - r.left) * max(0, r.bottom - r.top)
                if a > best_area:
                    best_area = a
                    best = h
        if best is not None:
            return best
    return None


def _chrome_snap_window_to_monitor_win32(
    hwnd: int,
    one_based_monitor: int,
    *,
    fullscreen: bool,
    windowed_size: tuple[int, int] | None,
) -> None:
    import ctypes
    from ctypes import wintypes

    ml, mt, mr, mb = _chrome_monitor_bounds(one_based_monitor)
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    SW_SHOWMAXIMIZED = 3
    HWND_TOP = 0
    SWP_SHOWWINDOW = 0x0040
    SWP_FRAMECHANGED = 0x0020
    flags = SWP_SHOWWINDOW | SWP_FRAMECHANGED

    user32.ShowWindow(hwnd, SW_RESTORE)
    if fullscreen:
        w, h = mr - ml, mb - mt
        x, y = ml, mt
    else:
        ww, wh = windowed_size or _chrome_window_size()
        w, h = ww, wh
        x = ml + max(0, (mr - ml - w) // 2)
        y = mt + max(0, (mb - mt - h) // 2)
    user32.SetWindowPos(hwnd, HWND_TOP, x, y, w, h, flags)

    if fullscreen:
        user32.ShowWindow(hwnd, SW_SHOWMAXIMIZED)
        KEYEVENTF_KEYUP = 0x0002
        VK_F11 = 0x7A
        fg = user32.GetForegroundWindow()
        tid_tgt = user32.GetWindowThreadProcessId(hwnd, None)
        tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        if tid_fg and tid_tgt:
            user32.AttachThreadInput(tid_fg, tid_tgt, True)
        user32.SetForegroundWindow(hwnd)
        if tid_fg and tid_tgt:
            user32.AttachThreadInput(tid_fg, tid_tgt, False)
        user32.keybd_event(VK_F11, 0, 0, 0)
        user32.keybd_event(VK_F11, 0, KEYEVENTF_KEYUP, 0)


def _open_url_in_chrome(
    url: str,
    *,
    new_window: bool = True,
    label: str = "URL",
    window_position: tuple[int, int] | None = None,
    window_size: tuple[int, int] | None = None,
    fullscreen: bool = False,
    win32_post_fullscreen_monitor: int | None = None,
    user_data_dir: str | None = None,
    browser_exe: str | None = None,
    browser_exe_basename: str = "chrome.exe",
    browser_name: str = "Chrome",
) -> None:
    u = url.strip()
    if not u:
        return
    chrome = browser_exe if browser_exe is not None else _chrome_executable()
    try:
        if chrome:
            args = [chrome]
            if user_data_dir:
                args.append(f"--user-data-dir={user_data_dir}")
                args.append("--no-first-run")
            if new_window:
                args.append("--new-window")
            if window_position is not None:
                x, y = window_position
                args.append(f"--window-position={x},{y}")
            if window_size:
                args.append(f"--window-size={window_size[0]},{window_size[1]}")
            if fullscreen and not (
                sys.platform == "win32" and win32_post_fullscreen_monitor is not None
            ):
                args.append("--start-fullscreen")
            args.append(u)
            popen_kw: dict = {
                "args": args,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            before: set[int] | None = None
            if sys.platform == "win32" and win32_post_fullscreen_monitor is not None:
                before = _chrome_top_level_browser_hwnds_win32(browser_exe_basename)
            subprocess.Popen(**popen_kw)
            if sys.platform == "win32" and win32_post_fullscreen_monitor is not None:
                mon = win32_post_fullscreen_monitor
                hwnd = _wait_new_chrome_hwnd_win32(
                    before, _chrome_new_window_wait_timeout_s(), browser_exe_basename
                )
                if hwnd is not None:
                    _chrome_snap_window_to_monitor_win32(
                        hwnd,
                        mon,
                        fullscreen=fullscreen,
                        windowed_size=window_size if not fullscreen else None,
                    )
                else:
                    log.warning(
                        "%s: timed out waiting for new window (%s); check "
                        "CHROME_NEW_WINDOW_WAIT_S or close extra %s instances.",
                        browser_name,
                        label,
                        browser_name,
                    )
        else:
            log.warning("%s not found; opening %s in default browser.", browser_name, label)
            webbrowser.open(u)
    except OSError as e:
        log.warning("Could not open %s in %s: %s", label, browser_name, e)


def open_claude_in_chrome() -> None:
    if not OPEN_CLAUDE_CODE_IN_CHROME:
        return
    url = (os.environ.get("CLAUDE_CODE_URL") or "https://claude.ai/new").strip()

    desktop_exe = _claude_desktop_executable()
    if desktop_exe:
        try:
            if sys.platform == "darwin" and desktop_exe.endswith(".app"):
                subprocess.Popen(["open", "-a", desktop_exe])
            else:
                subprocess.Popen(
                    [desktop_exe],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            log.info("Launched Claude desktop app.")
            return
        except OSError as e:
            log.warning("Could not launch Claude desktop app (%s); falling back to browser.", e)

    pos: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    fs = OPEN_CHROME_FULLSCREEN
    post_mon: int | None = None
    user_data: str | None = None
    if sys.platform == "win32":
        post_mon = CLAUDE_CHROME_MONITOR
        pos = _chrome_monitor_top_left(CLAUDE_CHROME_MONITOR)
        if fs:
            size = _chrome_monitor_pixel_size(CLAUDE_CHROME_MONITOR)
        else:
            size = _chrome_window_size()
        if CHROME_SEPARATE_SITE_PROFILES:
            user_data = _chrome_site_user_data_dir("claude")
    elif not fs:
        size = _chrome_window_size()
    else:
        size = None

    brave_exe = _brave_executable()
    if brave_exe:
        browser_exe, exe_basename, browser_name = brave_exe, "brave.exe", "Brave"
    else:
        log.warning("Brave not found; falling back to Chrome for Claude.")
        browser_exe, exe_basename, browser_name = _chrome_executable(), "chrome.exe", "Chrome"

    _open_url_in_chrome(
        url,
        new_window=True,
        label="Claude",
        window_position=pos,
        window_size=size,
        fullscreen=fs,
        win32_post_fullscreen_monitor=post_mon,
        user_data_dir=user_data,
        browser_exe=browser_exe,
        browser_exe_basename=exe_basename,
        browser_name=browser_name,
    )


def open_gmail_in_brave() -> None:
    if not OPEN_GMAIL_IN_BRAVE:
        return
    url = (os.environ.get("GMAIL_URL") or "https://mail.google.com/mail/u/0/#inbox").strip()
    pos: tuple[int, int] | None = None
    size: tuple[int, int] | None = None
    fs = OPEN_CHROME_FULLSCREEN
    post_mon: int | None = None
    user_data: str | None = None
    if sys.platform == "win32":
        post_mon = GMAIL_CHROME_MONITOR
        pos = _chrome_monitor_top_left(GMAIL_CHROME_MONITOR)
        if fs:
            size = _chrome_monitor_pixel_size(GMAIL_CHROME_MONITOR)
        else:
            size = _chrome_window_size()
        if CHROME_SEPARATE_SITE_PROFILES:
            user_data = _chrome_site_user_data_dir("gmail")
    elif not fs:
        size = _chrome_window_size()
    else:
        size = None

    brave_exe = _brave_executable()
    if brave_exe:
        browser_exe, exe_basename, browser_name = brave_exe, "brave.exe", "Brave"
    else:
        log.warning("Brave not found; falling back to Chrome for Gmail.")
        browser_exe, exe_basename, browser_name = _chrome_executable(), "chrome.exe", "Chrome"

    _open_url_in_chrome(
        url,
        new_window=True,
        label="Gmail",
        window_position=pos,
        window_size=size,
        fullscreen=fs,
        win32_post_fullscreen_monitor=post_mon,
        user_data_dir=user_data,
        browser_exe=browser_exe,
        browser_exe_basename=exe_basename,
        browser_name=browser_name,
    )


def _cursor_executable() -> str | None:
    """Locates the VS Code executable (name kept for minimal diff elsewhere)."""
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidates = []
        if local:
            candidates.append(os.path.join(local, "Programs", "Microsoft VS Code", "Code.exe"))
        if program_files:
            candidates.append(os.path.join(program_files, "Microsoft VS Code", "Code.exe"))
        for p in candidates:
            if os.path.isfile(p):
                return p
    return shutil.which("code") or shutil.which("code.cmd")


def _cursor_largest_main_hwnd_win32() -> int | None:
    """Largest top-level Cursor.exe window (visible or minimized)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    GW_OWNER = 4
    GWL_EXSTYLE = -20
    WS_EX_TOOLWINDOW = 0x00000080
    candidates: list[tuple[int, wintypes.HWND]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd: wintypes.HWND, _lp: wintypes.LPARAM) -> bool:
        if user32.GetWindow(hwnd, GW_OWNER):
            return True
        if user32.GetWindowLongW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        if not user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == 0:
            return True
        hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not hproc:
            return True
        try:
            buf = ctypes.create_unicode_buffer(4096)
            sz = wintypes.DWORD(len(buf))
            if not kernel32.QueryFullProcessImageNameW(hproc, 0, buf, ctypes.byref(sz)):
                return True
            exe_path = buf.value
        finally:
            kernel32.CloseHandle(hproc)
        if os.path.basename(exe_path).lower() != "code.exe":
            return True
        r = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return True
        w, h = r.right - r.left, r.bottom - r.top
        if w < 200 or h < 200:
            return True
        candidates.append((w * h, hwnd))
        return True

    user32.EnumWindows(_enum, 0)
    if not candidates:
        return None
    return int(max(candidates, key=lambda t: t[0])[1])


def _cursor_foreground_hwnd_win32(hwnd: int) -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(hwnd, SW_RESTORE)
    fg = user32.GetForegroundWindow()
    tid_tgt = user32.GetWindowThreadProcessId(hwnd, None)
    tid_fg = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    if tid_fg and tid_tgt:
        user32.AttachThreadInput(tid_fg, tid_tgt, True)
    user32.SetForegroundWindow(hwnd)
    if tid_fg and tid_tgt:
        user32.AttachThreadInput(tid_fg, tid_tgt, False)


def _cursor_send_f11_fullscreen_win32(hwnd: int) -> None:
    """F11 toggles Zen/fullscreen in Cursor (Electron)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    KEYEVENTF_KEYUP = 0x0002
    VK_F11 = 0x7A
    _cursor_foreground_hwnd_win32(hwnd)
    user32.keybd_event(VK_F11, 0, 0, 0)
    user32.keybd_event(VK_F11, 0, KEYEVENTF_KEYUP, 0)


def _focus_existing_cursor_window_win32() -> bool:
    """Bring an existing Cursor.exe main window to the foreground (no new process)."""
    if sys.platform != "win32":
        return False
    hwnd = _cursor_largest_main_hwnd_win32()
    if hwnd is None:
        return False
    _cursor_foreground_hwnd_win32(hwnd)
    return True


def run_double_clap_actions() -> None:
    """Run outside the mic loop so sleeps do not stall capture. Each action runs in its own
    thread so a slow one (e.g. a browser window taking a moment to appear) never delays the
    others — Spotify, Claude, Gmail, VS Code, and the spoken greeting all fire in parallel."""
    threading.Thread(target=play_song, args=(SONG_URI,), daemon=True).start()
    threading.Thread(target=open_claude_in_chrome, daemon=True).start()
    threading.Thread(target=open_gmail_in_brave, daemon=True).start()
    threading.Thread(target=open_cursor_window, daemon=True).start()
    if JARVIS_WELCOME_ENABLED:
        delay = max(0.0, JARVIS_AFTER_SONG_DELAY_S)

        def _speak():
            global _assistant_spoke_until_ts
            if delay:
                time.sleep(delay)
            text = _build_greeting_text()
            _assistant_speaking.set()
            try:
                say_jarvis_welcome(text)
            finally:
                _assistant_speaking.clear()
                _assistant_spoke_until_ts = time.monotonic() + ECHO_GUARD_S

        threading.Thread(target=_speak, daemon=True).start()


def open_cursor_window() -> None:
    if not FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP and not OPEN_NEW_CURSOR_ON_DOUBLE_CLAP:
        return
    exe = _cursor_executable()
    if not exe:
        log.warning(
            "Could not find VS Code (install app or add the `code` command to PATH "
            "via VS Code's 'Shell Command: Install code command in PATH')."
        )
        return
    popen_kw: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        if FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP:
            focused = (
                sys.platform == "win32" and _focus_existing_cursor_window_win32()
            )
            if not focused:
                subprocess.Popen([exe], **popen_kw)
        if OPEN_NEW_CURSOR_ON_DOUBLE_CLAP:
            subprocess.Popen([exe, "-n"], **popen_kw)
    except OSError as e:
        log.warning("Could not start or focus VS Code: %s", e)
        return
    if sys.platform == "win32" and CURSOR_OPEN_FULLSCREEN:
        time.sleep(0.5)
        hwnd = _cursor_largest_main_hwnd_win32()
        if hwnd is not None:
            _cursor_send_f11_fullscreen_win32(hwnd)
        else:
            log.warning("VS Code fullscreen: no VS Code window found to send F11.")


def main() -> int:
    blocksize = block_samples()
    noise_floor = 1e-4
    last_logged_double = 0.0
    first_clap_time: float | None = None
    spike_armed = True
    followup_run = 0
    loud_streak_s = 0.0

    log.info(
        "Listening (double clap: %.2f–%.2fs apart, rate=%d, block=%d ms, "
        "spike_ratio=%.1f, cooldown=%.2fs). Ctrl+C to stop.",
        MIN_DOUBLE_GAP_S,
        MAX_DOUBLE_GAP_S,
        SAMPLE_RATE,
        BLOCK_MS,
        SPIKE_RATIO,
        COOLDOWN_S,
    )
    if SONG_URI.strip():
        log.info("Double clap opens this track: %s", SONG_URI.strip())
    else:
        log.info("SONG_URI is empty — set it to play one song on each double clap.")
    if FOCUS_EXISTING_CURSOR_ON_DOUBLE_CLAP:
        log.info(
            "Double clap will foreground an existing VS Code window (Windows API); "
            "falls back to launching VS Code if none is running."
        )
    if OPEN_NEW_CURSOR_ON_DOUBLE_CLAP:
        log.info("Double clap will also open a new VS Code window (-n).")
    if CURSOR_OPEN_FULLSCREEN and sys.platform == "win32":
        log.info("VS Code will be sent F11 for fullscreen after focus/launch.")
    if OPEN_CLAUDE_CODE_IN_CHROME:
        cu = (os.environ.get("CLAUDE_CODE_URL") or "https://claude.ai/new").strip()
        log.info(
            "After Spotify, open Claude desktop app if installed, else Brave%s on monitor %d: %s",
            " fullscreen" if OPEN_CHROME_FULLSCREEN else "",
            CLAUDE_CHROME_MONITOR,
            cu,
        )
    if OPEN_GMAIL_IN_BRAVE:
        gu = (
            os.environ.get("GMAIL_URL") or "https://mail.google.com/mail/u/0/#inbox"
        ).strip()
        log.info(
            "After Spotify, open Gmail in Brave%s on monitor %d: %s",
            " fullscreen" if OPEN_CHROME_FULLSCREEN else "",
            GMAIL_CHROME_MONITOR,
            gu,
        )
    if JARVIS_WELCOME_ENABLED:
        ev, em, ef, er = elevenlabs_env_config()
        if JARVIS_DYNAMIC_GREETING_ENABLED:
            log.info(
                "After song + %.2fs: JARVIS will generate a fresh greeting for %s "
                "(Gmail context if available; ElevenLabs voice=%s, model=%s, format=%s, pcm_rate=%d)",
                JARVIS_AFTER_SONG_DELAY_S,
                JARVIS_ADDRESS_NAME,
                ev or "(unset)",
                em,
                ef,
                er,
            )
        else:
            log.info(
                "After song + %.2fs: %r (ElevenLabs voice=%s, model=%s, format=%s, pcm_rate=%d)",
                JARVIS_AFTER_SONG_DELAY_S,
                JARVIS_WELCOME_PHRASE.strip(),
                ev or "(unset)",
                em,
                ef,
                er,
            )

    input_idx = _choose_input_device(blocksize)
    debug_levels = (os.environ.get("JARVIS_DEBUG") or "").strip().lower() in ("1", "true", "yes")
    if debug_levels:
        log.info("JARVIS_DEBUG on: logging live level/threshold and every single spike.")
    last_debug_log = 0.0

    global _voice_followup_deadline_ts
    wake_detector = _WakeWordDetector() if JARVIS_VOICE_ENABLED else None
    if JARVIS_VOICE_ENABLED and wake_detector is not None and wake_detector.available:
        log.info(
            'Say "Jarvis" to start a conversation '
            "(needs ELEVENLABS_API_KEY + ANTHROPIC_API_KEY)."
        )
    _register_hotkeys()

    try:
        with sd.InputStream(
            device=input_idx,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=blocksize,
        ) as stream:
            while True:
                data, overflowed = stream.read(blocksize)
                if overflowed:
                    log.warning("Input overflow; try a larger BLOCK_MS")

                level = rms_mono(data)

                quiet_gate = noise_floor * QUIET_GATE_MULT
                if level < quiet_gate:
                    noise_floor = NOISE_FLOOR_ALPHA * noise_floor + (
                        1.0 - NOISE_FLOOR_ALPHA
                    ) * level
                    noise_floor = max(noise_floor, 1e-7)
                    loud_streak_s = 0.0
                else:
                    loud_streak_s += blocksize / SAMPLE_RATE
                    if loud_streak_s >= NOISE_FLOOR_CATCHUP_S:
                        # Sustained loud ambient, not a brief clap/spike — the room's baseline
                        # has likely shifted (fan, crowd, traffic). Let the floor catch up
                        # instead of staying stuck at a stale, quieter estimate forever.
                        noise_floor = NOISE_FLOOR_CATCHUP_ALPHA * noise_floor + (
                            1.0 - NOISE_FLOOR_CATCHUP_ALPHA
                        ) * level
                        noise_floor = max(noise_floor, 1e-7)

                threshold = min(max(noise_floor * SPIKE_RATIO, MIN_RMS), SPIKE_THRESHOLD_MAX)
                now = time.monotonic()
                retrigger_level = threshold * RETRIGGER_RATIO

                if debug_levels and (now - last_debug_log) >= 0.5:
                    last_debug_log = now
                    wake_score = wake_detector.last_score if wake_detector is not None else -1.0
                    log.info(
                        "level=%.5f noise_floor=%.5f threshold=%.5f wake_score=%.3f "
                        "(threshold=%.2f)",
                        level,
                        noise_floor,
                        threshold,
                        wake_score,
                        WAKE_WORD_THRESHOLD,
                    )

                hotkey_triggered = _hotkey_jarvis_event.is_set()
                if hotkey_triggered:
                    _hotkey_jarvis_event.clear()

                if (
                    wake_detector is not None
                    and wake_detector.available
                    and not _assistant_speaking.is_set()
                    and now >= _assistant_spoke_until_ts
                ):
                    wake_triggered = wake_detector.feed(data.reshape(-1))
                    in_followup = 0 < _voice_followup_deadline_ts and now <= _voice_followup_deadline_ts
                    if not in_followup:
                        followup_run = 0
                    elif level >= max(noise_floor * VOICE_SILENCE_RMS_MULT, MIN_RMS):
                        followup_run += 1
                    else:
                        followup_run = 0
                    followup_speech_starting = in_followup and followup_run >= FOLLOWUP_SUSTAIN_BLOCKS
                else:
                    wake_triggered = False
                    followup_speech_starting = False

                if wake_triggered or followup_speech_starting or hotkey_triggered:
                    followup_run = 0
                    _voice_followup_deadline_ts = 0.0
                    if wake_triggered:
                        log.info('Wake word "Jarvis" detected — listening...')
                    elif hotkey_triggered:
                        log.info('Hotkey %s pressed — listening...', HOTKEY_JARVIS)
                    utterance = _record_utterance_blocking(stream, blocksize, noise_floor)
                    if utterance is not None:
                        threading.Thread(
                            target=_handle_voice_turn, args=(utterance,), daemon=True
                        ).start()
                    continue

                if level < retrigger_level:
                    spike_armed = True

                if (
                    spike_armed
                    and level >= threshold
                    and (now - last_logged_double) >= COOLDOWN_S
                ):
                    spike_armed = False
                    if debug_levels:
                        log.info(
                            "Spike detected (rms=%.5f >= threshold=%.5f)", level, threshold
                        )
                    if first_clap_time is None:
                        first_clap_time = now
                    else:
                        gap = now - first_clap_time
                        if gap < MIN_DOUBLE_GAP_S:
                            pass
                        elif gap <= MAX_DOUBLE_GAP_S:
                            first_clap_time = None
                            last_logged_double = now
                            log.info(
                                "Double clap detected (gap=%.3fs, rms=%.5f, "
                                "noise_floor=%.5f, threshold=%.5f) — running actions",
                                gap,
                                level,
                                noise_floor,
                                threshold,
                            )
                            threading.Thread(
                                target=run_double_clap_actions, daemon=True
                            ).start()
                        else:
                            first_clap_time = now

    except KeyboardInterrupt:
        log.info("Stopped.")
        return 0
    except sd.PortAudioError as e:
        log.error("Audio error: %s", e)
        log.error("If PortAudio fails, install/repair drivers or try another SAMPLE_RATE.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())