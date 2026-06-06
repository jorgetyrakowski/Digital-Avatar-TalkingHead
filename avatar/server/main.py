import asyncio
import base64
import io
import json
import os
import re
import struct
import time
import wave
import httpx
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Vision (SmolVLM2-256M) — optional; the server still runs if it can't import.
try:
    from avatar.server import vision
except Exception:
    try:
        import vision  # when launched from the server directory
    except Exception as _ve:
        vision = None
        print(f"[VISION] module unavailable, vision mode disabled: {_ve}")

try:
    from avatar.server import visemes_zh
except ImportError:
    try:
        import visemes_zh  # when launched from the server directory
    except Exception as _vze:
        visemes_zh = None
        print(f"[TTS] visemes_zh unavailable, Chinese lip-sync disabled: {_vze}")

FISH_API_KEY     = os.environ.get("FISH_API_KEY", "")
FISH_REF_ID      = os.environ.get("FISH_REF_ID", "933563129e564b19a115bedd57b7406a")  # Sarah voice
FISH_API_URL     = "https://api.fish.audio/v1/tts/stream/with-timestamp"
LLM_API_URL      = os.environ.get("LLM_API_URL", "http://localhost:5003")
ROBOT_CONTAINER  = os.environ.get("ROBOT_CONTAINER", "robotic_agent_system")  # docker container with ROS2
ROBOT_ENABLED    = os.environ.get("ROBOT_ENABLED", "true").lower() == "true"
USE_RAG          = os.environ.get("USE_RAG", "true").lower() == "true"
# TTS backend per language: "kokoro" (local) | "edge" (Microsoft cloud, free,
# Taiwan-accented Mandarin) | "fish" (cloud, needs FISH_API_KEY).
# TTS_BACKEND sets the default for both; the per-language vars override it.
TTS_BACKEND      = os.environ.get("TTS_BACKEND", "kokoro")
TTS_BACKEND_EN   = os.environ.get("TTS_BACKEND_EN", TTS_BACKEND)
TTS_BACKEND_ZH   = os.environ.get("TTS_BACKEND_ZH", TTS_BACKEND)

# Speaking speed per language (1.0 = normal, 0.85 = 15% slower). Applied
# natively in kokoro and mapped to edge-tts' rate ("-15%"). Lip-sync stays
# correct at any speed: viseme timing is always measured from the real audio.
TTS_SPEED_EN     = float(os.environ.get("TTS_SPEED_EN", "1.0"))
TTS_SPEED_ZH     = float(os.environ.get("TTS_SPEED_ZH", "1.0"))
VISION_ENABLED   = os.environ.get("VISION_ENABLED", "true").lower() == "true"  # allow disabling the feature

SENTENCE_END = set("。！？…!?.")
MIN_SENTENCE = 8
MAX_SENTENCE = 120
# The FIRST chunk may also break at a Chinese comma: a short opening fragment
# reaches the TTS ~3-4x sooner, so the avatar starts speaking almost
# immediately while the rest of the (full-length) answer keeps streaming.
FIRST_SPLIT  = SENTENCE_END | set("，、；")

app = FastAPI()
_server_dir = os.path.dirname(os.path.abspath(__file__))
_client_dir = os.path.join(_server_dir, "..", "client")
app.mount("/client", StaticFiles(directory=_client_dir), name="client")

whisper_model: WhisperModel = None
active_connections: set = set()
pending_object_query: dict = None  # {request_id, name, option_count} — set while robot awaits user choice

# --- Vision mode runtime state (single POC session) ---
vision_on          = False  # user toggled vision mode on
vision_loading     = False  # model load in progress
vision_description  = ""     # latest VLM reading of the user


@app.on_event("startup")
async def startup():
    global whisper_model
    loop = asyncio.get_event_loop()
    whisper_model = await loop.run_in_executor(
        None, lambda: WhisperModel("tiny", device="cpu", compute_type="int8")
    )
    # Kokoro is always preloaded: it's either the configured backend or the
    # local fallback when a cloud backend (edge, fish) fails / has no internet.
    print("[TTS] Pre-loading Kokoro model...")
    await loop.run_in_executor(None, _get_kokoro)
    if visemes_zh:
        await loop.run_in_executor(None, lambda: _get_kokoro("z"))  # Mandarin
    print(f"[TTS] Kokoro ready. Backends: EN={TTS_BACKEND_EN}, ZH={TTS_BACKEND_ZH}")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_client_dir, "index.html"))


# Known whisper-tiny mishearings of domain terms (what it writes → what was said).
# Deterministic post-STT fixes — safer than prompt biasing, which loops on tiny.
STT_FIX_LATIN = {          # matched as whole words (case-insensitive)
    "E3": "ITRI",
}
STT_FIX_CJK = {            # plain substring replace (no word boundaries in CJK)
    "公演院": "工研院",
}


def _fix_transcript(text: str) -> str:
    for wrong, right in STT_FIX_LATIN.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    for wrong, right in STT_FIX_CJK.items():
        text = text.replace(wrong, right)
    return text


async def transcribe_audio(audio_bytes: bytes) -> str:
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    loop = asyncio.get_event_loop()
    segments, _ = await loop.run_in_executor(
        None, lambda: whisper_model.transcribe(audio_array, beam_size=1)
    )
    return _fix_transcript("".join(seg.text for seg in segments).strip())


async def stream_llm_sentences(text: str, user_description: str = ""):
    """Stream LLM response and yield complete sentences as they arrive."""
    buffer = ""
    first = True
    robot_cmd_checked = False

    payload = {"text_user_msg": text, "session_id": "talkinghead-poc", "use_rag": USE_RAG}
    if user_description:
        payload["user_description"] = user_description  # VLM view of the user
        payload["convert_tone"] = True                  # adapt tone to detected age

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST", f"{LLM_API_URL}/api/rag-llm/query",
            json=payload
        ) as r:
            async for chunk in r.aiter_text():
                if "END_FLAG" in chunk:
                    buffer += chunk.replace("END_FLAG", "")
                    break
                buffer += chunk

                # Extract [ROBOT: ...] tag from the start of the stream
                if ROBOT_ENABLED and not robot_cmd_checked:
                    if buffer.startswith("[ROBOT:"):
                        close = buffer.find("]")
                        if close == -1:
                            continue  # keep buffering until tag is complete
                        robot_command = buffer[7:close].strip()
                        buffer = buffer[close + 1:].lstrip()
                        robot_cmd_checked = True
                        yield {"robot_command": robot_command}
                    elif len(buffer) > 12:
                        robot_cmd_checked = True  # no tag, proceed normally
                elif not ROBOT_ENABLED:
                    robot_cmd_checked = True

                while True:
                    max_s = 60 if first else MAX_SENTENCE
                    ends = FIRST_SPLIT if first else SENTENCE_END
                    idx = next(
                        (i for i, ch in enumerate(buffer)
                         if ch in ends and i >= MIN_SENTENCE - 1),
                        -1
                    )
                    if idx == -1 and len(buffer) >= max_s:
                        idx = max_s - 1
                    if idx == -1:
                        break
                    sentence = buffer[:idx + 1].strip()
                    buffer = buffer[idx + 1:]
                    if sentence:
                        first = False
                        yield sentence

    if buffer.strip():
        yield buffer.strip()


async def publish_robot_command(command: str):
    """Publish a robot command to /manual_command via docker exec into the ROS2 container."""
    # Escape single quotes in command to prevent shell injection
    safe_cmd = command.replace("'", r"'\''")
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"ros2 topic pub --once /manual_command std_msgs/msg/String "
        f"\"{{data: '{safe_cmd}'}}\""
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", ROBOT_CONTAINER, "bash", "-c", bash_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        if proc.returncode != 0:
            print(f"[ROBOT] docker exec error: {stderr.decode().strip()}")
        else:
            print(f"[ROBOT] Published to /manual_command: '{command}'")
    except asyncio.TimeoutError:
        print(f"[ROBOT] docker exec timed out for command: '{command}'")
    except Exception as e:
        print(f"[ROBOT] Failed to publish command: {e}")


async def get_tts_with_timestamps(text: str):
    """Dispatch to the TTS backend configured for the text's language.

    Returns (audio_bytes, segments, visemes): visemes is None for English
    (the client's lipsync-en module handles it) or a precomputed
    (visemes, vtimes, vdurations) tuple for Chinese — TalkingHead has no
    zh module, so we generate Oculus visemes server-side (see visemes_zh).

    Cloud backends (edge, fish) fall back to Kokoro (local) on any failure,
    so the avatar keeps speaking without internet.
    """
    zh = bool(visemes_zh and visemes_zh.is_chinese(text))
    backend = TTS_BACKEND_ZH if zh else TTS_BACKEND_EN
    fn = _TTS_BACKENDS.get(backend, _tts_kokoro)
    try:
        return await fn(text)
    except Exception as e:
        if fn is _tts_kokoro:
            raise
        print(f"[TTS] {backend} failed ({e}), falling back to kokoro")
        return await _tts_kokoro(text)


_fish_client = None

def _get_fish_client():
    """Persistent client: keep-alive skips the ~2s TLS handshake per sentence."""
    global _fish_client
    if _fish_client is None:
        _fish_client = httpx.AsyncClient(timeout=30)
    return _fish_client


async def _tts_fish(text: str):
    audio_chunks = []
    all_segments = []

    zh = bool(visemes_zh and visemes_zh.is_chinese(text))
    speed = TTS_SPEED_ZH if zh else TTS_SPEED_EN

    client = _get_fish_client()
    async with client.stream(
        "POST", FISH_API_URL,
        headers={
            "Authorization": f"Bearer {FISH_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "reference_id": FISH_REF_ID,
            "format": "mp3",
            "latency": "low",
            "prosody": {"speed": speed},
        }
    ) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
                if data.get("audio_base64"):
                    audio_chunks.append(base64.b64decode(data["audio_base64"]))
                alignment = data.get("alignment")
                if alignment and alignment.get("segments"):
                    offset = data.get("chunk_audio_offset_sec", 0.0)
                    for seg in alignment["segments"]:
                        all_segments.append({
                            "text":  seg["text"],
                            "start": seg["start"] + offset,
                            "end":   seg["end"]   + offset,
                        })
            except Exception:
                continue

    # Fish re-sends (and sometimes refines) cumulative alignment across
    # chunks, occasionally with zero-duration entries — sort by time, prefer
    # the longer reading on ties, drop stale overlapping re-sends.
    all_segments.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    segments, prev_end = [], -1.0
    for s in all_segments:
        if s["start"] < prev_end - 0.05:
            continue  # stale duplicate from a cumulative re-send
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.04
        segments.append(s)
        prev_end = s["end"]

    # Chinese: Fish aligns per character — exact timing for server-side visemes.
    vis = None
    if visemes_zh and visemes_zh.is_chinese(text):
        vis = visemes_zh.visemes_from_segments(segments)

    return b"".join(audio_chunks), segments, vis


_kokoro_pipelines = {}
KOKORO_VOICE_EN = os.environ.get("KOKORO_VOICE_EN", "af_heart")
KOKORO_VOICE_ZH = os.environ.get("KOKORO_VOICE_ZH", "zf_xiaobei")

def _get_kokoro(lang_code: str = "a"):
    """Lazy per-language Kokoro pipeline ("a" American English, "z" Mandarin)."""
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline
        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]


def _numpy_to_wav(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


_whisper_align = None  # dedicated instance: the STT one may be busy on another thread

def _whisper_words(audio: np.ndarray, sample_rate: int, language: str):
    """Word-level timestamps of synthesized audio via Whisper-tiny (CPU int8,
    ~0.1×RT) — for TTS backends that provide none."""
    global _whisper_align
    if _whisper_align is None:
        _whisper_align = WhisperModel("tiny", device="cpu", compute_type="int8")
    idx = np.arange(0, len(audio), sample_rate / 16000.0)  # resample → 16 kHz
    a16 = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    segs, _ = _whisper_align.transcribe(a16, language=language, beam_size=1,
                                        word_timestamps=True)
    return [w for s in segs for w in s.words or []]


def _align_zh_chars(audio: np.ndarray, text: str, sample_rate: int = 24000):
    """Forced-align synthesized Mandarin audio → per-hanzi segments.

    Whisper's transcription ERRORS don't matter — we only keep the clock and
    map the KNOWN text onto it. Returns [{text, start, end}] (seconds) or
    None when alignment fails.
    """
    anchors = []  # (start, end) of each transcribed hanzi
    for w in _whisper_words(audio, sample_rate, "zh"):
        chars = [c for c in w.word if visemes_zh.is_chinese(c)]
        if not chars:
            continue
        per = (w.end - w.start) / len(chars)
        for k in range(len(chars)):
            anchors.append((w.start + k * per, w.start + (k + 1) * per))
    hanzi = [c for c in text if visemes_zh.is_chinese(c)]
    if not anchors or not hanzi:
        return None
    if len(anchors) == len(hanzi):  # usual case: 1:1 by position
        return [{"text": ch, "start": a[0], "end": a[1]}
                for ch, a in zip(hanzi, anchors)]
    # Count mismatch (mis-transcription) — map proportionally onto anchors.
    n, m = len(hanzi), len(anchors)
    out = []
    for i, ch in enumerate(hanzi):
        a0 = anchors[min(int(i * m / n), m - 1)]
        a1 = anchors[min(int((i + 1) * m / n), m - 1)]
        out.append({"text": ch, "start": a0[0], "end": max(a1[0], a0[1])})
    return out


async def _tts_kokoro(text: str):
    loop = asyncio.get_event_loop()
    zh = bool(visemes_zh and visemes_zh.is_chinese(text))

    def _generate():
        pipeline = _get_kokoro("z" if zh else "a")
        voice = KOKORO_VOICE_ZH if zh else KOKORO_VOICE_EN
        speed = TTS_SPEED_ZH if zh else TTS_SPEED_EN
        audio_chunks = []
        segments = []
        for result in pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                audio_chunks.append(result.audio)
            for token in (result.tokens or []):
                if token.text and token.text not in ".,!?;:":
                    segments.append({
                        "text":  token.text,
                        "start": token.start_ts,
                        "end":   token.end_ts,
                    })
        audio = np.concatenate(audio_chunks) if audio_chunks else np.zeros(0, dtype=np.float32)

        # Mandarin: Kokoro's zh pipeline returns no token timestamps — recover
        # real per-hanzi timing with Whisper forced alignment (~0.1×RT, CPU),
        # falling back to uniform per-syllable distribution.
        if zh and len(audio):
            aligned = None
            try:
                aligned = _align_zh_chars(audio, text)
            except Exception as e:
                print(f"[TTS] zh alignment failed, using uniform timing: {e}")
            segments = aligned or visemes_zh.char_segments(text, len(audio) / 24000 * 1000)
        return audio, segments

    audio, segments = await loop.run_in_executor(None, _generate)
    wav_bytes = _numpy_to_wav(audio)

    vis = None
    if zh and segments:
        vis = visemes_zh.visemes_from_segments(segments)

    return wav_bytes, segments, vis


EDGE_VOICE_EN = os.environ.get("EDGE_VOICE_EN", "en-US-AvaNeural")
EDGE_VOICE_ZH = os.environ.get("EDGE_VOICE_ZH", "zh-TW-HsiaoChenNeural")  # Taiwan accent


async def _tts_edge(text: str):
    """Microsoft Edge TTS (cloud, free, no API key).

    Why it's here: the only zero-cost source of Taiwan-accented Mandarin
    (zh-TW-* voices) — Kokoro's zh voices are all mainland. It also returns
    native word timestamps, so no Whisper alignment is needed (saves ~0.8s
    per Chinese sentence vs the Kokoro path).
    """
    import edge_tts

    zh = bool(visemes_zh and visemes_zh.is_chinese(text))
    voice = EDGE_VOICE_ZH if zh else EDGE_VOICE_EN
    speed = TTS_SPEED_ZH if zh else TTS_SPEED_EN
    rate = f"{round((speed - 1) * 100):+d}%"  # 0.85 → "-15%"
    tts = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")

    mp3 = b""
    words = []  # [{text, start, end}] in seconds (offsets come in 100ns ticks)
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            words.append({
                "text":  chunk["text"],
                "start": chunk["offset"] / 1e7,
                "end":   (chunk["offset"] + chunk["duration"]) / 1e7,
            })
    if not mp3:
        raise RuntimeError("edge-tts returned no audio")

    audio, sr = sf.read(io.BytesIO(mp3), dtype="float32")
    wav_bytes = _numpy_to_wav(audio, sr)

    segments, vis = words, None
    if zh:
        # Edge aligns per WORD (often multi-hanzi, e.g. 工研院) — split each
        # word's span evenly across its hanzi for per-syllable visemes.
        segments = []
        for w in words:
            chars = [c for c in w["text"] if visemes_zh.is_chinese(c)]
            if not chars:
                continue
            per = (w["end"] - w["start"]) / len(chars)
            for k, ch in enumerate(chars):
                segments.append({
                    "text":  ch,
                    "start": w["start"] + k * per,
                    "end":   w["start"] + (k + 1) * per,
                })
        if segments:
            vis = visemes_zh.visemes_from_segments(segments)

    return wav_bytes, segments, vis


# Backend registry — to add a new TTS: write _tts_<name>() returning
# (wav_bytes, segments, visemes) and register it here; select it in .env
# via TTS_BACKEND_EN / TTS_BACKEND_ZH.
_TTS_BACKENDS = {
    "kokoro": _tts_kokoro,
    "edge":   _tts_edge,
    "fish":   _tts_fish,
}


# ---------------------------------------------------------------------------
# Robot integration helpers (only active when ROBOT_ENABLED=true)
# ---------------------------------------------------------------------------

def parse_option_choice(text: str, option_count: int):
    """Parse spoken choice ('option one', 'the first', '2') → 0-based index or None."""
    text = text.lower()
    ordinals = ["first", "second", "third", "fourth"]
    words    = ["one",   "two",    "three",  "four"]
    for i in range(option_count):
        if str(i + 1) in text:                              return i
        if i < len(words)    and words[i]    in text:       return i
        if i < len(ordinals) and ordinals[i] in text:       return i
    return None


_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_actual_index(text: str, indices: list):
    """Parse a spoken MAP index against the indexes overlaid on the preview
    image (they are the robot's real, often non-consecutive indices: 0,2,5,9…).
    Returns the actual index or None."""
    text = text.lower()
    candidates = [int(n) for n in re.findall(r"\d+", text)]
    candidates += [v for w, v in _NUM_WORDS.items() if w in text]
    for c in candidates:
        if c in indices:
            return c
    return None


async def publish_object_query_reply(request_id: str, index: int):
    """Send the user's object choice to /object_query_reply via docker exec."""
    reply_json = json.dumps({"request_id": request_id, "index": index})
    safe_reply = reply_json.replace("'", r"'\''")
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"ros2 topic pub --once /object_query_reply std_msgs/msg/String "
        f"\"{{data: '{safe_reply}'}}\""
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", ROBOT_CONTAINER, "bash", "-c", bash_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=8.0)
        if proc.returncode != 0:
            print(f"[ROBOT] object_query_reply error: {stderr.decode().strip()}")
        else:
            print(f"[ROBOT] Published object_query_reply: request_id={request_id} index={index}")
    except asyncio.TimeoutError:
        print(f"[ROBOT] object_query_reply docker exec timed out")
    except Exception as e:
        print(f"[ROBOT] Failed to publish object_query_reply: {e}")


async def _push_speak(text: str):
    """TTS a text and broadcast speak + done to all active WebSocket connections."""
    if visemes_zh and visemes_zh.is_chinese(text):
        text = visemes_zh.sanitize_for_speech(text)
    audio_bytes, segments, vis = await get_tts_with_timestamps(text)
    speak_msg = {
        "type":       "speak",
        "text":       text,
        "audio_b64":  base64.b64encode(audio_bytes).decode(),
        "words":      [s["text"] for s in segments],
        "wtimes":     [round(s["start"] * 1000) for s in segments],
        "wdurations": [round((s["end"] - s["start"]) * 1000) for s in segments],
        "clear":      True,
    }
    if vis:
        speak_msg["visemes"], speak_msg["vtimes"], speak_msg["vdurations"] = vis
    for ws in list(active_connections):
        try:
            await ws.send_json(speak_msg)
            await ws.send_json({"type": "done"})
        except Exception:
            active_connections.discard(ws)


async def _broadcast_json(msg: dict):
    """Send a JSON message to all connected clients."""
    for ws in list(active_connections):
        try:
            await ws.send_json(msg)
        except Exception:
            active_connections.discard(ws)


async def _handle_object_query_response(user_text: str):
    """Process user's reply to a pending object query."""
    global pending_object_query
    query = pending_object_query

    # With a preview image the ONLY valid answers are the indexes drawn on the
    # map (often non-consecutive: 0, 2, 5, 9…) — never fall back to positional
    # interpretation, or "1" would silently become index 0.
    if query.get("has_image"):
        actual_index = parse_actual_index(user_text, query["indices"])
        if actual_index is None:
            nums = ", ".join(str(i) for i in query["indices"])
            await _push_speak(f"That number isn't on the map. Choose one of: {nums}.")
            return
    else:
        # No image: answer by position ("the first one", "option 2").
        position = parse_option_choice(user_text, query["option_count"])
        if position is None:
            n = query["option_count"]
            if n == 2:
                await _push_speak("Sorry, I didn't catch that. Say the first one or the second one.")
            else:
                await _push_speak(f"Sorry, I didn't catch that. Say a number from 1 to {n}.")
            return
        actual_index = query["indices"][position]

    pending_object_query = None
    await _broadcast_json({"type": "object_query_image_clear"})
    print(f"[ROBOT] User chose index {actual_index} for '{query['name']}'")
    asyncio.create_task(publish_object_query_reply(query["request_id"], actual_index))
    await _push_speak(f"Got it — going with number {actual_index}."
                      if query.get("has_image") else "Got it!")


async def handle_object_query_choice(data: dict):
    """Robot found multiple objects — ask the user to choose."""
    global pending_object_query
    if data.get("event") == "clear":
        pending_object_query = None
        await _broadcast_json({"type": "object_query_image_clear"})
        return
    name       = data.get("name", "object")
    options    = data.get("options", [])
    request_id = data.get("request_id", "")
    n = len(options)
    # Store actual index values from each option so we reply with the right index
    indices = [opt.get("index", i) for i, opt in enumerate(options)]

    # BEV preview image: the robot overlays the candidate indexes on a map and
    # writes the PNG to a volume shared with the host — read it directly.
    image_b64 = None
    preview = data.get("preview_image") or {}
    if preview.get("path"):
        try:
            with open(preview["path"], "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
        except OSError as e:
            print(f"[ROBOT] preview image unreadable ({preview['path']}): {e}")

    pending_object_query = {"request_id": request_id, "name": name,
                            "option_count": n, "indices": indices,
                            "has_image": bool(image_b64)}

    if image_b64:
        await _broadcast_json({"type": "object_query_image",
                               "image_b64": image_b64, "name": name})
        question = (f"I found {n} {name}s. Take a look at the map and "
                    f"tell me the number of the one you mean.")
    elif n == 2:
        question = f"I found 2 {name}s. Which one — the first one or the second one?"
    else:
        question = f"I found {n} {name}s. Say a number from 1 to {n} to choose."
    print(f"[ROBOT] Object query active: {question} (indices: {indices}, image: {bool(image_b64)})")
    await _push_speak(question)


async def handle_task_reply(data: dict):
    """Robot finished (success or failure) — tell the user."""
    success = data.get("success", False)
    message = data.get("message", "")
    if success:
        reply_text = "Done! The task was completed successfully."
    else:
        reason = f" {message}" if message else ""
        reply_text = f"The task couldn't be completed.{reason}"
    print(f"[ROBOT] Task reply: success={success}")
    await _push_speak(reply_text)


@app.post("/robot_event")
async def robot_event(request: Request):
    """Receive events from the robot bridge script (robot_bridge.py inside robotic_agent_system)."""
    if not ROBOT_ENABLED:
        return JSONResponse({"status": "robot_disabled"})
    payload = await request.json()
    topic   = payload.get("topic", "")
    data    = payload.get("data", {})
    if topic == "/object_query_choice":
        asyncio.create_task(handle_object_query_choice(data))
    elif topic == "/task_reply":
        asyncio.create_task(handle_task_reply(data))
    else:
        print(f"[ROBOT] Unknown topic from bridge: {topic}")
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Vision mode (SmolVLM2-256M): observe the user via webcam frames over the WS.
# Inference runs in a thread executor so the conversation hot-path never blocks.
# ---------------------------------------------------------------------------
async def _load_vision(ws: WebSocket):
    global vision_loading, vision_on
    loop = asyncio.get_event_loop()
    try:
        print("[VISION] Loading SmolVLM2-256M (one-time, ~1 min)...")
        await loop.run_in_executor(None, vision.load)
        print("[VISION] Model ready.")
        await ws.send_json({"type": "vision_status", "enabled": True, "message": "Vision ready 👁"})
    except Exception as e:
        vision_on = False
        print(f"[VISION] Load failed: {e}")
        await ws.send_json({"type": "vision_status", "enabled": False, "message": f"Vision load failed: {e}"})
    finally:
        vision_loading = False


async def _run_vision_inference(image_bytes: bytes, ws: WebSocket):
    """Describe one frame and cache the result. Drops the frame if busy."""
    global vision_description
    loop = asyncio.get_event_loop()
    try:
        desc = await loop.run_in_executor(None, vision.describe, image_bytes)
    except Exception as e:
        print(f"[VISION] Inference error: {e}")
        return
    if desc:
        vision_description = desc
        try:
            await ws.send_json({"type": "vision_result", "description": desc})
        except Exception:
            pass


async def handle_vision_message(msg: dict, ws: WebSocket):
    global vision_on, vision_loading, vision_description
    if vision is None or not VISION_ENABLED:
        await ws.send_json({"type": "vision_status", "enabled": False,
                            "message": "Vision not available on this server"})
        return

    if msg.get("type") == "vision_toggle":
        vision_on = bool(msg.get("enabled"))
        if not vision_on:
            vision_description = ""
            await ws.send_json({"type": "vision_status", "enabled": False, "message": "Vision off"})
            return
        if vision.is_loaded():
            await ws.send_json({"type": "vision_status", "enabled": True, "message": "Vision on 👁"})
        elif not vision_loading:
            vision_loading = True
            await ws.send_json({"type": "vision_status", "enabled": True,
                                "message": "Loading vision model (~1 min)…"})
            asyncio.create_task(_load_vision(ws))
        return

    if msg.get("type") == "vision_frame":
        if not vision_on or not vision.is_loaded():
            return
        image_b64 = msg.get("image", "")
        if not image_b64:
            return
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return
        asyncio.create_task(_run_vision_inference(image_bytes, ws))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.add(ws)
    try:
        while True:
            data = await ws.receive()
            if data.get("type") == "websocket.disconnect":
                break  # client closed/reloaded the page — exit cleanly

            if "bytes" in data:
                await ws.send_json({"type": "status", "message": "Transcribing..."})
                user_text = await transcribe_audio(data["bytes"])
                if not user_text:
                    await ws.send_json({"type": "done"})
                    continue
                await ws.send_json({"type": "transcript", "text": user_text})
            else:
                raw = data.get("text")
                if raw is None:
                    continue  # control frame (e.g. disconnect) — ignore
                # Vision control/data messages arrive as JSON; plain text is a user message.
                if raw.lstrip().startswith("{"):
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict) and parsed.get("type") in ("vision_toggle", "vision_frame"):
                        await handle_vision_message(parsed, ws)
                        continue
                user_text = raw

            # If the robot is waiting for a choice, parse it instead of calling LLM
            if ROBOT_ENABLED and pending_object_query:
                await _handle_object_query_response(user_text)
                continue

            await ws.send_json({"type": "status", "message": "Thinking..."})

            user_desc = vision_description if (vision_on and vision_description) else ""

            t_start = time.perf_counter()
            first = True
            async for item in stream_llm_sentences(user_text, user_description=user_desc):
                if isinstance(item, dict) and "robot_command" in item:
                    if ROBOT_ENABLED:
                        cmd = item["robot_command"]
                        print(f"[ROBOT COMMAND DETECTED] >>> {cmd}")
                        await ws.send_json({"type": "robot_command", "command": cmd})
                        asyncio.create_task(publish_robot_command(cmd))
                    continue

                sentence = item
                if visemes_zh and visemes_zh.is_chinese(sentence):
                    sentence = visemes_zh.sanitize_for_speech(sentence)
                else:
                    # Kokoro mangles the acronym — speak the full name instead.
                    sentence = re.sub(r"\bITRI\b", "Industrial Technology Research Institute", sentence)
                t_llm = round(time.perf_counter() - t_start, 3)

                t_tts_start = time.perf_counter()
                audio_bytes, segments, vis = await get_tts_with_timestamps(sentence)
                t_tts = round(time.perf_counter() - t_tts_start, 3)

                msg = {
                    "type":       "speak",
                    "text":       sentence,
                    "audio_b64":  base64.b64encode(audio_bytes).decode(),
                    "words":      [s["text"] for s in segments],
                    "wtimes":     [round(s["start"] * 1000) for s in segments],
                    "wdurations": [round((s["end"] - s["start"]) * 1000) for s in segments],
                }
                if vis:
                    # Chinese: precomputed Oculus visemes (TalkingHead has no
                    # zh lip-sync module; speakAudio uses these directly).
                    msg["visemes"], msg["vtimes"], msg["vdurations"] = vis

                if first:
                    msg["timing"] = {
                        "t_llm":   t_llm,
                        "t_tts":   t_tts,
                        "t_total": round(t_llm + t_tts, 3),
                    }
                    await ws.send_json({"type": "status", "message": "Speaking..."})
                    first = False

                await ws.send_json(msg)
                t_start = time.perf_counter()

            await ws.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    finally:
        active_connections.discard(ws)
