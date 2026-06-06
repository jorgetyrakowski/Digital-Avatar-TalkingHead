import os as _os
LLM_MODEL_NAME = "qwen3:30b-a3b-instruct-2507-q4_K_M"
VLLM_MODEL_NAME = "gemma4"
CHROMA_DB_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "chroma_db")