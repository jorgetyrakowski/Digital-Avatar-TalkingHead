#!/bin/bash
# Stop the full Digital Avatar stack

# Stop the robot bridge inside the robot container (started by start.sh)
ROBOT_CONTAINER="${ROBOT_CONTAINER:-robotic_agent_system}"
docker exec "$ROBOT_CONTAINER" pkill -f robot_bridge.py 2>/dev/null || true

tmux send-keys -t "cf-tunnel:tunnel" C-c 2>/dev/null || true
tmux kill-session -t "cf-tunnel" 2>/dev/null || true

tmux send-keys -t "itri-avatar:avatar" C-c 2>/dev/null || true
tmux kill-session -t "itri-avatar" 2>/dev/null || true

tmux send-keys -t "itri-llm:api" C-c 2>/dev/null || true
sleep 2
# Stop ONLY our vLLM container by name — other teams run containers from the
# same jetson-thor image (e.g. decision-vllm), never stop those.
docker stop itri-vllm 2>/dev/null || true
sleep 2
tmux kill-session -t "itri-llm" 2>/dev/null || true

# Release page cache so vLLM model weights don't stay cached in RAM
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

echo "Stack stopped."
