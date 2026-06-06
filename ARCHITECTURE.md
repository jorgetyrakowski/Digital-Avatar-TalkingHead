# System Architecture — Digital Avatar + Robot Integration

## Full Pipeline

```
                        ┌─────────────────────┐
                        │   User interacts     │  (voice or text)
                        └────────┬────────────┘
                                 │ text
                                 ▼
                        ┌─────────────────────┐
                        │   Avatar LLM         │  Gemma4-26B (vLLM, Jetson)
                        │                      │
                        │  Detects robot intent│
                        │  via [ROBOT: ...] tag│
                        └────────┬────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │ Robot command detected?              │
               │                                     │
              YES                                    NO
               │                                     │
               ▼                                     ▼
  ┌────────────────────────┐          ┌──────────────────────────┐
  │  Extract command from  │          │   Normal conversation    │
  │  [ROBOT: go to kitchen]│          │                          │
  └────────────┬───────────┘          │   (RAG optional)         │
               │                      │   Answer → TTS           │
               │  publish             └──────────────┬───────────┘
               ▼                                     │
  ┌────────────────────────┐                         │
  │   docker exec          │                         │
  │   robotic_agent_system │                         │
  │                        │                         │
  │   ros2 topic pub       │                         │
  │   /manual_command      │                         │
  └────────────┬───────────┘                         │
               │  (fire and forget,                  │
               │   non-blocking)                     │
               │                                     │
               ▼                                     │
  ┌────────────────────────┐                         │
  │ agent_decision_maker   │  ◄── also needs:        │
  │ _node                  │                         │
  │                        │  object_query_server    │
  │ (AI agent, plans       │  (knows where objects   │
  │  action primitives)    │   are in the lab)       │
  └────────────┬───────────┘                         │
               │                                     │
               │ ["goto:kitchen",                    │
               │  "grasp:bottle", ...]               │
               ▼                                     │
  ┌────────────────────────┐                         │
  │   Robot (Kachaka)      │                         │
  │   navigates & acts     │                         │
  └────────────────────────┘                         │
                                                     ▼
                                        ┌────────────────────────┐
                                        │   TTS (per language:    │
                                        │   kokoro / edge / fish) │
                                        │   + word timestamps     │
                                        └────────────┬───────────┘
                                                     │ audio + lip sync
                                                     ▼
                                        ┌────────────────────────┐
                                        │   Avatar speaks         │
                                        │   (3D WebGL, Three.js)  │
                                        └────────────────────────┘
```

> ⚡ The robot command path and the TTS/speech path run **simultaneously** —
> the avatar starts speaking while the robot command is being published.

---

## Components

| Component | What it does | Technology |
|---|---|---|
| **Avatar frontend** | 3D talking avatar rendered in the browser, shows lip sync | Three.js / WebGL |
| **Avatar server** | Receives user input via WebSocket, orchestrates STT → LLM → TTS | FastAPI (Python) |
| **Avatar LLM** | Understands the user, generates a reply, detects robot commands via `[ROBOT: ...]` tag | Gemma4-26B-A4B NVFP4 (vLLM) |
| **TTS** | Converts the LLM reply to speech with word timestamps for lip sync; backend selectable per language | Kokoro (local) / Edge TTS / Fish Audio |
| **ROS2 bridge** | When a robot command is detected, publishes it to `/manual_command` inside the robot container | `docker exec` + `ros2 topic pub` |
| **object_query_server** | Knows the position of objects in the lab — answers "where is the water bottle?" | ROS2 node (robot side) |
| **agent_decision_maker_node** | Receives the natural language command from `/manual_command`, uses an AI agent to plan and execute actions | ROS2 node + LLM agent (robot side) |
| **Robot (Kachaka)** | Physical robot that navigates, grasps, and delivers objects | Hardware |

