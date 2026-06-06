# Digital Avatar TalkingHead

**A real-time, bilingual (English / Traditional Chinese) 3D digital avatar that answers questions about your organization — running entirely on a single NVIDIA Jetson AGX Thor.**

This project builds on the excellent [**TalkingHead**](https://github.com/met4citizen/TalkingHead) library by Mika Suominen (MIT), which provides the 3D avatar rendering, animation and English lip-sync in the browser. On top of it, we built a complete edge-AI conversational system:

- a **RAG + LLM backend** (ChromaDB + Gemma4-26B on vLLM) so the avatar answers from a real knowledge base — here, Taiwan's Industrial Technology Research Institute (ITRI)
- **Mandarin speech and lip-sync** — TalkingHead has no Chinese module, so we generate visemes server-side (hanzi → pinyin → Oculus visemes) with exact per-character timing
- **speech-to-text, three pluggable TTS engines, webcam vision, and optional ROS2 robot control**
- everything tuned so the avatar **starts speaking ~1.5 s** after you finish talking

## 🎥 Demos

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://youtu.be/tHQHYcrDoNQ">
        <img src="https://img.youtube.com/vi/tHQHYcrDoNQ/maxresdefault.jpg" alt="English conversation demo" />
      </a>
      <br/><b>▶️ English conversation</b>
    </td>
    <td align="center" width="50%">
      <a href="https://youtu.be/Ehv7Wde5eDo">
        <img src="https://img.youtube.com/vi/Ehv7Wde5eDo/maxresdefault.jpg" alt="Chinese conversation demo" />
      </a>
      <br/><b>▶️ 中文對話 · Chinese conversation</b>
    </td>
  </tr>
</table>

🤖 *Robot integration demo: coming soon*

## Features

- **Bilingual**: ask by voice or text in English or Traditional Chinese — the system detects the language and answers in kind
- **Mandarin lip-sync** (our extension to TalkingHead): server-side viseme generation with per-hanzi timing; numbers are converted to hanzi (1973 → 一九七三) so they are spoken *and* lip-synced
- **Pluggable TTS, selectable per language**: `kokoro` (local/offline), `edge` (Microsoft cloud, free, Taiwan-accented Mandarin) or `fish` (Fish Audio cloud, custom voices) — cloud backends fall back to local automatically
- **RAG**: hybrid search (vector + keyword) over a ChromaDB knowledge base; embeddings run in-process via fastembed — no extra server
- **Vision mode** 👁: the avatar sees the user through the webcam (SmolVLM2-256M) and adapts its tone to the detected audience
- **Robot integration** (optional): natural-language commands forwarded to a ROS2 robot, with visual object disambiguation — see [Robot Integration](#-robot-integration)
- **Streaming pipeline**: the avatar speaks each sentence while the next ones are still being generated

## Pipeline

```
 🎤 voice ──► VAD (browser, AudioWorklet RMS) ──► WebSocket ──► Whisper tiny (STT, CPU int8)
                                                                     │ text
                                                                     ▼
                              RAG: multilingual-e5-large embeddings (fastembed, in-process)
                                   → ChromaDB hybrid search (vector + keyword, top-6)
                                                                     │ context + question
                                                                     ▼
                              vLLM · Gemma4-26B-A4B NVFP4 — streamed token by token
                                                                     │ split into sentences
                                                                     │ as they arrive
                                                                     ▼
                    TTS per language ──► EN: Kokoro (local) · ZH: Edge / Fish / Kokoro
                    ZH also gets server-side visemes (hanzi → pinyin → Oculus visemes)
                                                                     │ WAV + word timings
                                                                     │ (+ visemes for zh)
                                                                     ▼
                    Browser: TalkingHead 3D avatar speaks each sentence with lip-sync
                    while the next ones are still being generated — first words ~1.5 s
```

Services layout:

```
┌────────────────────────────────────────────────────────┐
│  Browser                                               │
│  avatar/client/   ← 3D WebGL avatar (TalkingHead/Three)│
│       │  WebSocket (HTTPS :8010)                       │
│  avatar/server/   ← FastAPI · Whisper STT · TTS · visemes
│       │  HTTP :5003                                    │
│  llm/api/         ← RAG + LLM API (Flask)              │
│       │  HTTP :8000 (OpenAI-compatible)                │
│  [Docker: vLLM]   ← Gemma4-26B / Qwen3-30B (NVFP4)     │
└────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline including the robot path, and [DIAGRAM.md](DIAGRAM.md) for component diagrams.

## Models & Inference

| Role | Model | Runtime | Notes |
|------|-------|---------|-------|
| **LLM** | [Gemma4-26B-A4B](https://huggingface.co/bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4) (MoE, NVFP4) | **vLLM** — NVIDIA Jetson Thor container, `--quantization modelopt --moe-backend marlin` | **~38 tok/s** decode measured on the Thor; 4096 ctx |
| LLM (alt) | [Qwen3-30B-A3B](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4) NVFP4 | same | `./start.sh qwen3` |
| Embeddings | intfloat/multilingual-e5-large | fastembed (ONNX, in-process) | no embedding server needed |
| STT | Whisper tiny (int8) | faster-whisper, CPU | bilingual EN/ZH, ~0.5 s per utterance |
| Vision | SmolVLM2-256M-Video-Instruct | transformers, in-process | one webcam frame every 5 s |
| TTS (EN) | Kokoro-82M `af_heart` | local | offline-capable |
| TTS (ZH) | Edge TTS `zh-TW` / Fish Audio / Kokoro `zf_xiaoxiao` | cloud / local | native word timestamps; auto-fallback to Kokoro |

The NVFP4 MoE quantization is what makes a 26B-parameter model interactive on the Jetson: only ~4B parameters are active per token, decoded at ~38 tok/s — faster than the avatar speaks.

## Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| NVIDIA Jetson AGX Thor | 128 GB unified memory, JetPack with Docker + NVIDIA runtime |
| [Miniforge / Conda](https://github.com/conda-forge/miniforge) | for the two Python environments |
| `openssl`, `curl`, `tmux` | usually preinstalled |
| *(optional)* Fish Audio API key | only for the `fish` TTS backend |

### 1. Clone

```bash
git clone <repo-url>
cd Digital-Avatar-TalkingHead
```

### 2. Create the Python environments

The project uses **two conda environments** — one for the LLM/RAG backend, one for the avatar server (they have conflicting dependency trees, e.g. onnxruntime vs torch):

```bash
bash setup.sh
```

`setup.sh` does four things; if you prefer to run them manually:

```bash
# 2a. LLM + RAG environment
conda create -n itri-llm python=3.12 -y
conda activate itri-llm
pip install -r llm/requirements.txt

# 2b. Avatar server environment (STT, TTS, lip-sync, vision)
conda create -n itri-talkinghead python=3.12 -y
conda activate itri-talkinghead
pip install -r avatar/requirements.txt

# 2c. Self-signed HTTPS certificate (browsers require HTTPS for the microphone)
mkdir -p ssl
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
    -keyout ssl/key.pem -out ssl/cert.pem -subj "/CN=digital-avatar"

# 2d. cloudflared — optional, only for the public tunnel URL
mkdir -p bin
curl -fL -o bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
chmod +x bin/cloudflared
```

### 3. Download the LLM weights and vLLM image

```bash
# Model weights → ~/.cache/huggingface (≈15 GB)
pip install -U "huggingface_hub[cli]"
huggingface-cli download bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4

# vLLM container for Jetson Thor
docker pull ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor
```

`scripts/start_vllm.sh` finds the weights automatically wherever the HF cache puts them.

### 4. Configure

```bash
cp .env.example .env
```

The defaults work without any API key (local Kokoro TTS for English, free Edge TTS for Chinese). See [Configuration](#configuration-env) below for all options.

### 5. Build the knowledge base

```bash
conda activate itri-llm
cd llm && python -m rag.RAG_LLM_realtime --RAG_RELOAD && cd ..
```

This embeds the JSON files in `llm/knowledge/` into `chroma_db/`. To use your own knowledge, replace the JSONs and re-run.

### 6. Start

```bash
./start.sh           # full stack with Gemma4 (default)
./start.sh qwen3     # ...or Qwen3
./stop.sh            # stop everything
```

First start takes ~2 min while vLLM loads the model. The script launches everything in tmux and prints the URLs when ready:

| Service | Port | tmux session |
|---------|------|--------------|
| vLLM inference (Docker) | 8000 | `itri-llm` — window 0 (`vllm`) |
| RAG + LLM API (Flask) | 5003 | `itri-llm` — window 1 (`api`) |
| Avatar WebSocket + frontend (HTTPS) | 8010 | `itri-avatar` |
| Cloudflare tunnel (optional) | — | `cf-tunnel` |

Open **`https://localhost:8010`** (same machine), **`https://<jetson-ip>:8010`** (LAN), or the **public `https://….trycloudflare.com` URL** printed at the end (any network).

> HTTPS uses the self-signed certificate — accept the browser warning once per device, then the microphone works.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| Mic button does nothing | You must use **https** (not http) and accept the certificate warning |
| "Model weights not found" | Run the `huggingface-cli download` from step 3 |
| First answer is slow | Warm-up: models load lazily — ask a throwaway question first |
| No public URL printed | Check `tmux attach -t cf-tunnel`; or set `ENABLE_TUNNEL=false` |
| Logs | `tmux attach -t itri-llm` (Ctrl+B 0/1) · `tmux attach -t itri-avatar` · `Ctrl+B D` to detach |

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_BACKEND_EN` / `TTS_BACKEND_ZH` | `kokoro` / `edge` | TTS engine per language: `kokoro` \| `edge` \| `fish`. Cloud backends fall back to kokoro automatically if offline |
| `KOKORO_VOICE_EN` / `KOKORO_VOICE_ZH` | `af_heart` / `zf_xiaoxiao` | Kokoro voices |
| `EDGE_VOICE_ZH` | `zh-TW-HsiaoChenNeural` | Edge voice (Taiwan-accented Mandarin; also HsiaoYu, YunJhe) |
| `FISH_API_KEY` / `FISH_REF_ID` | — | Fish Audio credentials + voice reference (fish backend only) |
| `TTS_SPEED_EN` / `TTS_SPEED_ZH` | `1.0` | Speaking speed (e.g. `0.85` = 15 % slower) — works on all backends |
| `USE_RAG` | `true` | Ground answers in the knowledge base |
| `VISION_ENABLED` | `true` | Enable the 👁 webcam vision mode |
| `BRIEF_ANSWERS` | `false` | Force one-sentence answers (live-demo mode) |
| `ENABLE_TUNNEL` | `true` | Publish a free public URL via Cloudflare tunnel |
| `ROBOT_ENABLED` | `false` | Forward `[ROBOT: …]` commands to the ROS2 container |

## Project Structure

```
Digital-Avatar-TalkingHead/
├── llm/
│   ├── api/            ← Flask API + tone system prompts
│   ├── rag/            ← RAG pipeline (ChromaDB + fastembed)
│   ├── knowledge/      ← knowledge base (12 JSON files)
│   └── config.py
├── avatar/
│   ├── server/         ← FastAPI WebSocket · Whisper STT · TTS dispatch
│   │   ├── main.py     ←   + sentence streaming, timing, robot events
│   │   ├── visemes_zh.py ← Mandarin viseme generation (lip-sync)
│   │   └── vision.py   ← SmolVLM2 vision mode
│   └── client/         ← 3D WebGL frontend (vendored TalkingHead + Three.js)
├── scripts/
│   ├── start_vllm.sh   ← vLLM Docker container
│   └── robot_bridge.py ← ROS2 ↔ avatar bridge (auto-deployed by start.sh)
├── start.sh / stop.sh / setup.sh
└── .env.example        ← copy to .env
```

## Performance (Jetson AGX Thor)

| Stage | English | Chinese |
|-------|---------|---------|
| STT (Whisper tiny, CPU) | ~0.5 s | ~0.6 s |
| LLM first sentence (Gemma4-26B NVFP4) | ~0.5 s | ~0.5 s |
| TTS first fragment | ~0.6 s (kokoro) | ~0.8 s (fish) |
| **Time to first spoken word** | **~1.5 s** | **~1.5 s** |

The first Chinese fragment is cut at the first comma so speech starts while the rest of the answer is still streaming.

## 🤖 Robot Integration

The avatar can drive a physical robot: when the user asks for something physical ("bring me the water bottle", "go to the kitchen"), the LLM emits a `[ROBOT: …]` tag that the avatar server converts into a ROS2 message.

🎥 **Demo**: *coming soon*

```
User ──► Avatar LLM ──► [ROBOT: bring water bottle] detected
                              │ docker exec → ros2 topic pub /manual_command
                              ▼
                    robotic_agent_system (ROS2 container)
                              │ /task_reply, /object_query_choice
                              ▼
              scripts/robot_bridge.py ──► avatar speaks the outcome
```

- The avatar **answers and speaks immediately** while the command is dispatched in parallel — the robot never blocks the conversation
- Robot → avatar events (task done, object disambiguation questions) flow back through `scripts/robot_bridge.py`, which `start.sh` automatically copies into the robot container and (re)starts — no manual steps
- **Visual object disambiguation**: when the robot finds several objects with the same name, it sends a bird's-eye-view map with the candidates' indexes overlaid. The avatar shows it next to the 3D character (click to enlarge), asks *"tell me the number of the one you mean"*, and the user answers **by voice** — the spoken index is validated against the map and published back to `/object_query_reply`
- Enable with `ROBOT_ENABLED=true` in `.env` (requires the `robotic_agent_system` ROS2 container running)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full message flow.

## License

This project is released under the [MIT License](LICENSE).

It vendors and depends on third-party components under their own licenses (TalkingHead — MIT, Three.js — MIT, Ready Player Me sample avatar, and others) — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the complete list.

## Acknowledgements

- [met4citizen/TalkingHead](https://github.com/met4citizen/TalkingHead) by **Mika Suominen** — the 3D avatar engine this project is built on
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — local TTS
- [Fish Audio](https://fish.audio) / [edge-tts](https://github.com/rany2/edge-tts) — cloud TTS backends
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — STT
- NVIDIA Jetson AI Lab — vLLM container images for Jetson Thor
- [ITRI](https://www.itri.org.tw) — Industrial Technology Research Institute, Taiwan, where this project was developed
