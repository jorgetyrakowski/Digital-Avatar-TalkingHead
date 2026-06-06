# Third-Party Notices

This project builds on the work of several open-source projects. Vendored
code keeps its original license headers; pip dependencies are installed by
the user and are not redistributed in this repository.

## Vendored in this repository

| Component | Where | License | Notes |
|-----------|-------|---------|-------|
| [TalkingHead](https://github.com/met4citizen/TalkingHead) by Mika Suominen | `avatar/client/vendor/talkinghead/` | MIT | 3D avatar rendering, animation and lip-sync engine — the foundation of the client. License header retained in the source |
| [Three.js](https://github.com/mrdoob/three.js) | `avatar/client/vendor/three/` | MIT | WebGL renderer (r170) |
| [es-module-shims](https://github.com/guybedford/es-module-shims) | `avatar/client/vendor/es-module-shims.js` | MIT | import-maps polyfill |
| Sample avatar `brunette.glb` | `avatar/client/avatars/` | © [Ready Player Me](https://readyplayer.me) | Sample avatar distributed with the upstream TalkingHead project. You can replace it with your own Ready Player Me avatar (see TalkingHead's docs for the required morph targets) |

## Python dependencies (installed via pip, not redistributed)

| Package | License | Used for |
|---------|---------|----------|
| kokoro / misaki | Apache-2.0 | local TTS (EN + ZH) |
| faster-whisper | MIT | speech-to-text |
| edge-tts | LGPL-3.0 | Microsoft Edge TTS backend (optional; used as an unmodified library) |
| fastembed | Apache-2.0 | RAG embeddings |
| chromadb | Apache-2.0 | vector store |
| FastAPI / uvicorn / Flask | MIT / BSD-3 | servers |
| transformers | Apache-2.0 | vision model runtime |
| pypinyin / jieba / cn2an | MIT | Mandarin G2P for lip-sync |

## Models (downloaded at runtime by the user)

| Model | License / Terms |
|-------|-----------------|
| Gemma4-26B-A4B NVFP4 | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| Qwen3-30B-A3B NVFP4 | Apache-2.0 |
| Kokoro-82M | Apache-2.0 |
| Whisper (tiny) | MIT |
| SmolVLM2-256M | Apache-2.0 |
| multilingual-e5-large | MIT |

## Cloud services (optional)

- **Microsoft Edge TTS** — accessed through the `edge-tts` library (unofficial API); subject to Microsoft's terms
- **Fish Audio** — commercial TTS API, requires your own API key and is subject to Fish Audio's terms
- **Cloudflare Quick Tunnels** — optional public URL, subject to Cloudflare's terms
