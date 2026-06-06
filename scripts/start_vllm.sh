#!/bin/bash
# Start vLLM with NVFP4 model on Jetson Thor
# Usage: ./start_vllm_nvfp4.sh [qwen3|gemma4]
#
# Key flags explained:
#   --quantization modelopt       : Required for ModelOpt/NVFP4 format
#   --moe-backend marlin          : Required for MoE FP4 expert layers
#   VLLM_USE_FLASHINFER_MOE_FP4=0 : FlashInfer FP4 kernel broken on Jetson Thor
#   --gpu-memory-utilization 0.50 : 0.72 was max before OOM; 0.50 sufficient for single-user POC

MODEL="${1:-qwen3}"

HF_CACHE=/home/acm/.cache/huggingface/hub

if [ "$MODEL" = "gemma4" ]; then
    MODEL_ID="/data/models/huggingface/hub/models--bg-digitalservices--Gemma-4-26B-A4B-it-NVFP4/snapshots/a15dd6f161881b62db952303a5bfb7be118ed15e"
    CONTAINER="ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-thor"
else
    MODEL_ID="/data/models/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3"
    CONTAINER="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor"
fi

echo "Starting vLLM with model: $MODEL_ID"
echo "Container: $CONTAINER"

docker run --rm --name itri-vllm --runtime=nvidia \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HUB_OFFLINE=1 \
  -e HF_HOME=/data/models/huggingface \
  -v /home/acm/.cache/huggingface:/data/models/huggingface \
  -v /home/acm/thor-vllm-cache:/root/.cache/vllm \
  -p 8000:8000 \
  "$CONTAINER" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$MODEL" \
    --quantization modelopt \
    --moe-backend marlin \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.37 \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --trust-remote-code
