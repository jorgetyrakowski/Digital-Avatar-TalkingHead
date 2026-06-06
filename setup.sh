#!/bin/bash
# One-time setup after cloning: conda environments, self-signed HTTPS
# certificate (required for browser microphone access) and the cloudflared
# binary (optional public tunnel).

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up itri-llm environment ==="
conda create -n itri-llm python=3.12 -y
conda run -n itri-llm pip install -r "$REPO_DIR/llm/requirements.txt"

echo "=== Setting up itri-talkinghead environment ==="
conda create -n itri-talkinghead python=3.12 -y
conda run -n itri-talkinghead pip install -r "$REPO_DIR/avatar/requirements.txt"

echo "=== Generating self-signed HTTPS certificate ==="
# Browsers only allow microphone access over HTTPS — a self-signed cert is
# enough (accept the warning once per device).
mkdir -p "$REPO_DIR/ssl"
if [ ! -f "$REPO_DIR/ssl/key.pem" ]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
        -keyout "$REPO_DIR/ssl/key.pem" -out "$REPO_DIR/ssl/cert.pem" \
        -subj "/CN=digital-avatar"
    echo "ssl/key.pem + ssl/cert.pem created."
else
    echo "ssl/key.pem already exists — skipping."
fi

echo "=== Downloading cloudflared (optional public tunnel) ==="
mkdir -p "$REPO_DIR/bin"
if [ ! -x "$REPO_DIR/bin/cloudflared" ]; then
    case "$(uname -m)" in
        aarch64) CF_ARCH=arm64 ;;
        x86_64)  CF_ARCH=amd64 ;;
        *)       CF_ARCH="$(uname -m)" ;;
    esac
    if curl -fL -o "$REPO_DIR/bin/cloudflared" \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF_ARCH"; then
        chmod +x "$REPO_DIR/bin/cloudflared"
        echo "bin/cloudflared downloaded."
    else
        echo "⚠️  cloudflared download failed — public tunnel disabled."
        echo "    Set ENABLE_TUNNEL=false in .env, or download it manually into bin/."
    fi
else
    echo "bin/cloudflared already exists — skipping."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. cp .env.example .env    # fill FISH_API_KEY only if you use the fish TTS backend"
echo "  2. Build the knowledge base:"
echo "       conda activate itri-llm && cd llm && python -m rag.RAG_LLM_realtime --RAG_RELOAD"
echo "  3. ./start.sh"
