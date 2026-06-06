#!/bin/bash
# Start the full Digital Avatar stack
# Usage: ./start.sh [qwen3|gemma4]
#
# Services started:
#   itri-llm  window 0 (vllm)  — vLLM inference engine (Docker, port 8000)
#   itri-llm  window 1 (api)   — RAG + LLM API (Flask, port 5003)
#   itri-avatar                — TalkingHead WebSocket + frontend (port 8010, HTTPS)
#   robot_bridge.py            — inside $ROBOT_CONTAINER, if ROBOT_ENABLED=true

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-gemma4}"

# Load .env if present
if [ -f "$REPO_DIR/.env" ]; then
    export $(grep -v '^#' "$REPO_DIR/.env" | xargs)
fi

# --- LLM stack ---
SESSION_LLM="itri-llm"
tmux kill-session -t "$SESSION_LLM" 2>/dev/null || true
sleep 1
tmux new-session -d -s "$SESSION_LLM" -n "vllm"
tmux send-keys -t "$SESSION_LLM:vllm" "cd $REPO_DIR && bash scripts/start_vllm.sh $MODEL" Enter

echo "Waiting for vLLM to be ready (may take ~2 min)..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 5; done
echo "vLLM ready."

tmux new-window -t "$SESSION_LLM" -n "api"
tmux send-keys -t "$SESSION_LLM:api" \
    "conda activate itri-llm && cd $REPO_DIR/llm && set -a && [ -f ../.env ] && source ../.env; set +a && python -m api.rag_llm_api --auto-init --port 5003 --llm-backend vllm" Enter

# --- Avatar ---
SESSION_AVATAR="itri-avatar"
tmux kill-session -t "$SESSION_AVATAR" 2>/dev/null || true
tmux new-session -d -s "$SESSION_AVATAR" -n "avatar"
tmux send-keys -t "$SESSION_AVATAR:avatar" \
    "conda activate itri-talkinghead && cd $REPO_DIR && set -a && [ -f .env ] && source .env; set +a && uvicorn avatar.server.main:app --host 0.0.0.0 --port 8010 --ssl-keyfile $REPO_DIR/ssl/key.pem --ssl-certfile $REPO_DIR/ssl/cert.pem" Enter

# --- Robot bridge (inside the robot container) ---
# Copies the latest scripts/robot_bridge.py into the robot container and
# (re)starts it, so robot → avatar events (/task_reply, /object_query_choice)
# reach the avatar. The copy step keeps the container from running a stale
# version after the script changes in this repo.
ROBOT_ENABLED="${ROBOT_ENABLED:-false}"
ROBOT_CONTAINER="${ROBOT_CONTAINER:-robotic_agent_system}"
if [ "$ROBOT_ENABLED" = "true" ]; then
    if docker ps --format '{{.Names}}' | grep -qx "$ROBOT_CONTAINER"; then
        docker exec "$ROBOT_CONTAINER" pkill -f robot_bridge.py 2>/dev/null || true
        docker cp "$REPO_DIR/scripts/robot_bridge.py" "$ROBOT_CONTAINER:/robot_bridge.py"
        docker exec -d "$ROBOT_CONTAINER" bash -c \
            "source /opt/ros/humble/setup.bash && python3 /robot_bridge.py > /tmp/robot_bridge.log 2>&1"
        echo "Robot bridge running in '$ROBOT_CONTAINER' (log: /tmp/robot_bridge.log inside the container)."
    else
        echo "⚠️  ROBOT_ENABLED=true but container '$ROBOT_CONTAINER' is not running — robot bridge NOT started."
    fi
fi

# --- Public tunnel (Cloudflare) ---
# Gives a public https URL reachable from ANY network, no router config needed.
# Disable by setting ENABLE_TUNNEL=false in .env (e.g. main-project-only / offline use).
ENABLE_TUNNEL="${ENABLE_TUNNEL:-true}"
PUBLIC_URL=""
if [ "$ENABLE_TUNNEL" = "true" ] && [ -x "$REPO_DIR/bin/cloudflared" ]; then
    echo "Waiting for avatar (port 8010) to be ready..."
    for i in $(seq 1 30); do
        if curl -sk -o /dev/null --max-time 3 https://localhost:8010/ 2>/dev/null; then break; fi
        sleep 2
    done

    SESSION_TUNNEL="cf-tunnel"
    TUNNEL_LOG="$REPO_DIR/cf_tunnel.log"
    : > "$TUNNEL_LOG"
    tmux kill-session -t "$SESSION_TUNNEL" 2>/dev/null || true
    tmux new-session -d -s "$SESSION_TUNNEL" -n "tunnel"
    tmux send-keys -t "$SESSION_TUNNEL:tunnel" \
        "$REPO_DIR/bin/cloudflared tunnel --url https://localhost:8010 --no-tls-verify 2>&1 | tee $TUNNEL_LOG" Enter

    echo "Starting public tunnel..."
    for i in $(seq 1 20); do
        PUBLIC_URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$TUNNEL_LOG" 2>/dev/null | head -1)
        if [ -n "$PUBLIC_URL" ]; then break; fi
        sleep 2
    done
fi

echo ""
echo "Stack started."
echo "  LLM:    tmux attach -t $SESSION_LLM  (Ctrl+B 0=vllm, Ctrl+B 1=api)"
echo "  Avatar: tmux attach -t $SESSION_AVATAR"
echo "  Web (this device):  https://localhost:8010"
echo "  Web (same Wi-Fi):   https://$(hostname -I | awk '{print $1}'):8010"
if [ -n "$PUBLIC_URL" ]; then
    echo ""
    echo "  🌐 PUBLIC URL (any network, share this):"
    echo "       $PUBLIC_URL"
    echo "     (tunnel runs in tmux 'cf-tunnel'; URL changes on each restart)"
elif [ "$ENABLE_TUNNEL" = "true" ]; then
    echo ""
    echo "  ⚠️  Tunnel did not start. Check: tmux attach -t cf-tunnel"
fi
