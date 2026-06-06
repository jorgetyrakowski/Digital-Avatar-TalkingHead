"""SmolVLM2-256M vision module — observe the user via webcam frames.

Runs in-process inside the avatar server (env: itri-talkinghead). One frame is
analysed at a time; the latest structured description is cached by main.py and
forwarded to the LLM as `user_description`. Designed to stay off the
conversation hot-path: the model loads lazily (~1 min, once) and each inference
is ~0.3s on the Jetson, run in a thread executor so the WebSocket never blocks.
"""
import io
import threading

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.bfloat16 if _DEVICE == "cuda" else torch.float32
_MAX_NEW = 80
_MAX_WIDTH = 512  # downscale incoming webcam frames for speed/footprint

# Free-form one-sentence description. A 256M VLM does NOT reliably follow a rigid
# multi-line tag format (it collapses to a single word), but a guided natural
# sentence works well and is keyword-rich — which is exactly what the LLM-side
# rule-based tone selector matches on (boy/elderly/suit/...). See Phase 2.
VISION_PROMPT = (
    "Describe the person in one sentence: approximate age (child, young, adult, "
    "or elderly), gender, clothing, and facial expression."
)

_model = None
_processor = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # serialise inference: one frame at a time


def _load_processor():
    """transformers 5.x rejects this repo's legacy `Idefics3ImageProcessor`
    name via the Auto* lookup, so fall back to assembling the processor from the
    dedicated SmolVLM classes (video sub-processor included to satisfy __init__)."""
    try:
        return AutoProcessor.from_pretrained(MODEL_ID)
    except ValueError:
        from transformers import (SmolVLMProcessor, SmolVLMImageProcessor,
                                  SmolVLMVideoProcessor, AutoTokenizer)
        image_proc = SmolVLMImageProcessor.from_pretrained(MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        return SmolVLMProcessor(
            image_processor=image_proc, tokenizer=tokenizer,
            video_processor=SmolVLMVideoProcessor(),
            chat_template=getattr(tokenizer, "chat_template", None))


def load():
    """Load model + processor. Blocking (~1 min) — call once from an executor."""
    global _model, _processor
    with _load_lock:
        if _model is not None:
            return
        proc = _load_processor()
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=_DTYPE, attn_implementation="sdpa").to(_DEVICE).eval()
        _processor, _model = proc, model


def is_loaded() -> bool:
    return _model is not None


def describe(image_bytes: bytes) -> str:
    """Run the VLM on a JPEG/PNG frame; return the raw structured description.

    Returns "" if the model isn't loaded yet or another inference is in flight
    (the frame is simply dropped — we only ever need the latest reading)."""
    if _model is None:
        return ""
    if not _infer_lock.acquire(blocking=False):
        return ""  # busy: drop this frame
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        if w > _MAX_WIDTH:
            image = image.resize((_MAX_WIDTH, int(h * _MAX_WIDTH / w)), Image.LANCZOS)

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": VISION_PROMPT}]}]
        text = _processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = _processor(text=text, images=[image], return_tensors="pt").to(_DEVICE)
        if _DEVICE == "cuda":
            inputs = {k: (v.to(_DTYPE) if v.is_floating_point() else v)
                      for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=_MAX_NEW,
                                 do_sample=False, use_cache=True)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return _processor.decode(gen, skip_special_tokens=True).strip()
    finally:
        _infer_lock.release()
