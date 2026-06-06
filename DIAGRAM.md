# Full Pipeline Diagram

> Updated 2026-06-02 — reflects current implementation

---

## Main Flow — User Interaction

```
                        User speaks or types
                               │
                    [VAD — AudioWorklet RMS]
                    [Whisper tiny — CPU int8]
                               │
                           user_text
                               │
               ┌───────────────┴───────────────┐
               │ pending_object_query active?   │ NO
               │YES                             │
               ▼                                ▼
     parse_option_choice()        Avatar LLM — Gemma4-26B (vLLM)
               │                        Is this a robot command?
       ┌───────┴───────┐                ┌──────────┴──────────┐
       │ found         │ not found     YES                     NO
       ▼               ▼               │                       │
 docker exec      Aria asks      Extract                Normal conversation
 /object_query_  again (TTS)    [ROBOT: ...]            (RAG optional)
 reply                               │                        │
       │                       ┌─────┴──────┐                 │
       │                       │            │          Kokoro TTS (local)
       │                 docker exec     Kokoro TTS     ~0.08s latency
       │                 /manual_       (Avatar speaks  word timestamps
       │                 command        simultaneously)        │
       │                       │                               │
       │               agent_decision_                         │
       │               maker_node                              │
       │                       │                               │
       │                Robot (Kachaka)                        │
       └───────────────────────┴───────────────────────────────┘
                                                               │
                                                               ▼
                                              Avatar speaks — 3D WebGL
                                                  (TalkingHead)
```

---

## Robot → Avatar Flow (new topics)

The `robot_bridge.py` script runs inside `robotic_agent_system` and bridges
ROS2 topics to the avatar server via HTTP POST.

```
  robotic_agent_system (Docker container)
  ─────────────────────────────────────────────────────────────────
  │                                                               │
  │   object_query_server                agent_decision_maker_node│
  │          │                                      │             │
  │  /object_query_choice          /task_reply (success/fail)     │
  │          │                                      │             │
  │          └──────────────┬───────────────────────┘             │
  │                         │                                     │
  │                   robot_bridge.py                             │
  │                  (ROS2 subscriber)                            │
  ─────────────────────────────────────┼───────────────────────────
                                       │
                              HTTPS POST /robot_event
                                       │
                             Avatar Server (port 8010)
                                       │
                    ┌──────────────────┴──────────────────┐
                    │ /object_query_choice                 │ /task_reply
                    │                                      │
                    │  Aria asks: "I found 2 tables.       │  Aria says: "Done!"
                    │  Option 1 or option 2?"              │  or "The task failed."
                    │                                      │
                    │  sets pending_object_query           │  (no state change)
                    └──────────────────┬───────────────────┘
                                       │
                                  Kokoro TTS
                                       │
                           Avatar speaks — TalkingHead
```

---

## Topic Summary

| Topic | Direction | Via | Purpose |
|---|---|---|---|
| `/manual_command` | Avatar → Robot | `docker exec` | Send robot command (natural language) |
| `/object_query_choice` | Robot → Avatar | `robot_bridge` → HTTP | Robot found multiple objects, asks user |
| `/object_query_reply` | Avatar → Robot | `docker exec` | User's choice (index + request_id) |
| `/task_reply` | Robot → Avatar | `robot_bridge` → HTTP | Task result (success / failure) |

---

## ROBOT_ENABLED Flag

```
ROBOT_ENABLED=true   → Full pipeline: all topics active, LLM knows about robot
ROBOT_ENABLED=false  → Avatar-only mode: no robot logic, no topics, pure Aria
```

When `false`: `/robot_event` endpoint exists but returns immediately without action.
No bridge script needed — just don't run it.

---

## Services

| Service | Port | Conda env | tmux |
|---|---|---|---|
| vLLM — Gemma4-26B NVFP4 (Docker) | 8000 | — | `itri-llm` win 0 |
| RAG + LLM API (Flask) | 5003 | `itri-llm` | `itri-llm` win 1 |
| Avatar server + frontend (FastAPI, HTTPS) | 8010 | `itri-talkinghead` | `itri-avatar` |
| Cloudflare tunnel (public URL) | — | — | `cf-tunnel` |
| robot_bridge.py (inside container) | — | ROS2 Humble | manual |
