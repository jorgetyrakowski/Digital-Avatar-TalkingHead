# Demo Guide — Digital Avatar + Robot Integration

## Overview

The avatar listens to the user, understands if the request is a conversation or a robot
command, speaks a natural reply, and simultaneously publishes the command to `/manual_command`.
The robot can also push questions and task results back to the avatar at any time.

```
User speaks/types
  → Whisper ASR + VAD
  → Gemma4-26B (vLLM)
      ├── [ROBOT: cmd] detected → docker exec → /manual_command → decision_maker_node → Robot
      │                           Avatar speaks reply simultaneously (Kokoro TTS)
      └── Normal conversation  → Kokoro TTS → Avatar speaks (TalkingHead 3D WebGL)

Robot → avatar (via robot_bridge.py inside container):
  /object_query_choice → Aria asks user to choose between objects
  /task_reply          → Aria announces task success or failure
```

> Full diagram: see `DIAGRAM.md`

---

## 1. Start the Full Stack

```bash
cd /home/acm/llm_teams/Digital-Avatar-TalkingHead
./start.sh
```

Services launched:

| Service | Port | tmux |
|---|---|---|
| vLLM — Gemma4-26B NVFP4 (Docker) | 8000 | `itri-llm` win 0 |
| RAG + LLM API (Flask) | 5003 | `itri-llm` win 1 |
| Avatar server + frontend (FastAPI, **HTTPS**) | 8010 | `itri-avatar` |

> First start takes ~2 min while vLLM loads the model.

Open browser (note **https** — required for the microphone):
- Same machine: `https://localhost:8010` — same LAN: `https://<jetson-ip>:8010`
- From the internet (any network): the public `https://….trycloudflare.com` URL printed by `./start.sh` (requires `ENABLE_TUNNEL=true`; the URL changes on each restart)

> The certificate is self-signed, so the first time each device shows a security
> warning → **Advanced → Proceed**. Accept it once and the mic will work.

---

## 2. Enable / Disable Robot Integration

Edit `.env` before starting:

```bash
ROBOT_ENABLED=true    # full pipeline — robot commands active
ROBOT_ENABLED=false   # avatar-only mode — pure Aria, no robot logic
```

Then restart: `./stop.sh && ./start.sh`

When `ROBOT_ENABLED=false`, the LLM system prompt has no robot instructions and
the avatar never outputs `[ROBOT: ...]` tags — zero overhead.

---

## 3. Open Monitoring Terminals

**Terminal A — Avatar logs:**
```bash
tmux attach -t itri-avatar
```
Shows: LLM output, robot command detection, TTS timing, robot event POSTs.

**Terminal B — ROS2 monitor (outgoing commands):**
```bash
docker exec robotic_agent_system bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /manual_command"
```

**Terminal C — ROS2 monitor (incoming from robot):**
```bash
docker exec robotic_agent_system bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /object_query_choice" &
docker exec robotic_agent_system bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /task_reply"
```

---

## 4. Demo Flow

### Step 1 — Normal conversation

Say:
> *"What is ITRI?"*
> *"Tell me something interesting about robotics."*

- Avatar responds naturally, speaks via Kokoro TTS
- Terminal B stays silent — nothing published to `/manual_command`
- Proves LLM distinguishes conversation from robot commands

---

### Step 2 — Robot command

Say:
> *"Can you bring me the water bottle?"*
> *"Go to the kitchen."*
> *"Clean the table."*

Terminal A shows:
```
[ROBOT COMMAND DETECTED] >>> bring me the water bottle
[ROBOT] Published to /manual_command: 'bring me the water bottle'
```

Terminal B shows:
```
data: bring me the water bottle
---
```

Avatar speaks the reply **at the same time** as the command is published (non-blocking).

---

### Step 3 — Object query (robot asks user to choose)

**Simulate with curl** (or wait for robot_bridge to trigger it):
```bash
curl -sk -X POST https://localhost:8010/robot_event \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "/object_query_choice",
    "data": {
      "event": "active",
      "name": "table",
      "request_id": "table-1",
      "options": [{"index": 0, "instance_id": "12"}, {"index": 1, "instance_id": "18"}]
    }
  }'
```

Avatar says: *"I found 2 tables. Which one do you mean — option 1 or option 2?"*

User replies (voice): *"option one"* / *"the first one"* / *"1"*

Avatar confirms: *"Got it — going with option 1."*
Then publishes to `/object_query_reply`:
```json
{"request_id": "table-1", "index": 0}
```

---

### Step 4 — Task result (robot reports back)

**Simulate success:**
```bash
curl -sk -X POST https://localhost:8010/robot_event \
  -H "Content-Type: application/json" \
  -d '{"topic": "/task_reply", "data": {"success": true, "message": "all subtasks completed", "task": "bring water bottle"}}'
```
Avatar says: *"Done! The task was completed successfully."*

**Simulate failure:**
```bash
curl -sk -X POST https://localhost:8010/robot_event \
  -H "Content-Type: application/json" \
  -d '{"topic": "/task_reply", "data": {"success": false, "message": "Object not found in the environment", "task": "bring water bottle"}}'
```
Avatar says: *"The task couldn't be completed. Object not found in the environment."*

---

## 5. robot_bridge (when running with physical robot)

**Automatic since `start.sh` handles it**: when `ROBOT_ENABLED=true` in `.env` and the
`robotic_agent_system` container is running, `./start.sh` copies the latest
`scripts/robot_bridge.py` into the container and (re)starts it. `./stop.sh` stops it.

Check that it's running / view its log:
```bash
docker exec robotic_agent_system bash -c "pgrep -af robot_bridge; tail /tmp/robot_bridge.log"
```

Manual start (only if you need a non-default avatar URL, e.g. IP differs from
`172.17.0.1`, the default Docker bridge on Linux):
```bash
docker exec -it robotic_agent_system bash
source /opt/ros/humble/setup.bash
AVATAR_SERVER_URL=https://<host_ip>:8010 python3 /robot_bridge.py
```
> The bridge talks to the avatar over **HTTPS** and skips cert verification automatically.

---

## 6. Example Commands

| Say this | Published to `/manual_command` |
|---|---|
| "Bring me the water bottle" | `bring me the water bottle` |
| "Go to the kitchen" | `go to the kitchen` |
| "Clean the table" | `clean table` |
| "Go back to your charging spot" | `go home` |
| "Fetch me a drink" | `fetch drink` |
| "Move the box to the desk" | `bring box to desk` |

---

## 7. Stop Everything

```bash
./stop.sh
```

Stops all tmux sessions, Docker container, frees GPU/RAM.

---

## Key Highlights

| What | How |
|---|---|
| **On-device LLM** | Gemma4-26B-A4B NVFP4 via vLLM on Jetson AGX Thor |
| **Voice input** | VAD (AudioWorklet RMS) + Whisper tiny (CPU, int8) |
| **TTS** | Kokoro local (~0.08s), word-level timestamps for lip sync |
| **Robot detection** | LLM outputs `[ROBOT: cmd]` tag — no separate classifier |
| **Avatar → Robot** | `docker exec robotic_agent_system ros2 topic pub /manual_command` |
| **Robot → Avatar** | `robot_bridge.py` subscribes ROS2 topics, POSTs to `/robot_event` |
| **Non-blocking** | Avatar speaks while command is published simultaneously |
| **Toggle** | `ROBOT_ENABLED=false` disables all robot logic with one env var |
