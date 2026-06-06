# Digital Avatar TalkingHead

A real-time, bilingual (English / Traditional Chinese) 3D talking avatar powered by a local RAG + LLM backend. Runs fully on an NVIDIA Jetson AGX Thor — speech-to-text, retrieval, inference, TTS and lip-sync — with answers spoken in under ~2 seconds. Optionally connects to a physical robot via ROS2.

## 🎥 Demos

| Demo | Link |
|------|------|
| English conversation | [YouTube](TODO_LINK_EN) |
| Chinese conversation (中文對話) | [YouTube](TODO_LINK_ZH) |
| Robot integration | [YouTube](TODO_LINK_ROBOT) |

## Features

- **Bilingual**: ask in English or Traditional Chinese — the avatar detects the language and answers in kind
- **Chinese lip-sync**: TalkingHead has no Mandarin module, so visemes are generated server-side (hanzi → pinyin → Oculus visemes) with exact per-character timing
- **Pluggable TTS, selectable per language**: `kokoro` (local, offline), `edge` (Microsoft cloud, free, Taiwan-accented Mandarin) or `fish` (Fish Audio cloud, custom voices)
- **RAG**: answers are grounded in a ChromaDB knowledge base (ITRI corporate knowledge, 12 JSON files) with hybrid search — embeddings run in-process via fastembed
- **Vision mode** 👁: the avatar can see the user through the webcam (SmolVLM2-256M, in-process) and adapts tone to the detected audience
- **Robot integration** (optional): natural-language commands are forwarded to a ROS2 robot — see [Robot Integration](#-robot-integration)
- **Fast**: first spoken words in ~1.5 s after the question (streamed sentence-by-sentence TTS)

## Architecture

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

## Requirements

- NVIDIA Jetson AGX Thor (128 GB unified memory)
- Docker with the NVIDIA runtime
- Conda (miniforge recommended)
- vLLM Docker image from NVIDIA (`ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor`)
- The model weights available locally in the HuggingFace cache (see `scripts/start_vllm.sh`)
- *(optional)* Fish Audio API key — only if you select the `fish` TTS backend

## Setup (first time only)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd Digital-Avatar-TalkingHead

# 2. Conda envs + HTTPS certificate + cloudflared (one script)
bash setup.sh

# 3. Configure
cp .env.example .env
# defaults work out of the box; set FISH_API_KEY only for the fish TTS backend

# 4. Build the knowledge base
conda activate itri-llm
cd llm && python -m rag.RAG_LLM_realtime --RAG_RELOAD && cd ..
```

## Start / Stop

```bash
./start.sh           # full stack with Gemma4 (default)
./start.sh qwen3     # ...or Qwen3
./stop.sh            # stop everything
```

`./start.sh` launches the services in tmux and prints the URLs when ready:

| Service | Port | tmux session |
|---------|------|--------------|
| vLLM inference (Docker) | 8000 | `itri-llm` — window 0 (`vllm`) |
| RAG + LLM API (Flask) | 5003 | `itri-llm` — window 1 (`api`) |
| Avatar WebSocket + frontend (HTTPS) | 8010 | `itri-avatar` |
| Cloudflare tunnel (optional) | — | `cf-tunnel` |

Open **`https://localhost:8010`** (same machine), **`https://<jetson-ip>:8010`** (LAN), or the **public `https://….trycloudflare.com` URL** printed at the end (any network — requires `ENABLE_TUNNEL=true`).

> HTTPS is required for microphone access; accept the self-signed certificate warning once per device.

```bash
tmux attach -t itri-llm      # Ctrl+B 0 = vLLM logs, Ctrl+B 1 = API logs
tmux attach -t itri-avatar   # avatar server logs    (Ctrl+B D to detach)
```

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

🎥 **Demo**: [YouTube](TODO_LINK_ROBOT)

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
- Object disambiguation is conversational: the robot asks "which one?", the user answers by voice ("the first one"), and the choice is published back
- Enable with `ROBOT_ENABLED=true` in `.env` (requires the `robotic_agent_system` ROS2 container running)

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full message flow.

## Notes

- `chroma_db/`, `ssl/` and `bin/` are generated locally (gitignored) — `setup.sh` creates them
- Rebuild ChromaDB after changing the knowledge base: `cd llm && python -m rag.RAG_LLM_realtime --RAG_RELOAD`
- vLLM runs in NVIDIA's Docker container — it is not installed via pip
- The first request after startup is slower (model warm-up); warm up with a throwaway question before demos

## Acknowledgements

- [met4citizen/TalkingHead](https://github.com/met4citizen/talkinghead) — 3D avatar rendering and lip-sync engine
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — local TTS
- [Fish Audio](https://fish.audio) / [edge-tts](https://github.com/rany2/edge-tts) — cloud TTS backends
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — STT
- NVIDIA Jetson AI Lab — vLLM container images for Jetson Thor
