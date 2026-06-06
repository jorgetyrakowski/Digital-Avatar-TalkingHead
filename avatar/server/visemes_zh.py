"""
Mandarin viseme generation for TalkingHead lip-sync.

TalkingHead has no lipsync-zh module (hanzi never match the English letter
rules, so the mouth barely moves on Chinese sentences). Instead of writing a
client-side module we precompute Oculus visemes server-side — speakAudio()
uses `visemes`/`vtimes`/`vdurations` directly when they are given, bypassing
the language modules entirely.

Pipeline: hanzi → pinyin (pypinyin, handles Traditional Chinese) →
initial/final → 1–4 Oculus visemes per syllable. Timing comes either from
exact per-character TTS timestamps (Fish Audio, future CosyVoice 2) or, when
the TTS gives none (Kokoro zh), from distributing the audio duration over the
syllables — Mandarin is close to isochronous, so uniform slots look right.
"""
import re

from pypinyin import lazy_pinyin, Style

_CJK_RE   = re.compile(r"[㐀-䶿一-鿿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_PAUSE_CHARS = set("。，！？…、；：,!?;: ")

# Parenthetical containing Latin — 工業技術研究院（ITRI） — full/half-width parens.
_PAREN_LATIN_RE = re.compile(r"[（(][^（）()]*[A-Za-z][^（）()]*[)）]")


def sanitize_for_speech(text: str) -> str:
    """Prepare a Chinese sentence for TTS + lip-sync (and as the subtitle).

    - Drops parenthetical Latin asides like 工業技術研究院（ITRI）: Kokoro's zh
      pipeline can't pronounce them and they read oddly as subtitles.
    - Converts Arabic numerals to hanzi (1973 → 一九七三, 50% → 百分之五十) so
      numbers are both pronounced AND lip-synced — the viseme pipeline maps
      hanzi only, so digits would otherwise move no lips.
    """
    text = _PAREN_LATIN_RE.sub("", text)
    try:
        import cn2an  # ships with misaki[zh]
        text = cn2an.transform(text, "an2cn")
    except Exception:
        pass  # cn2an missing/failed — speak the original digits
    return text.strip()

# Pinyin initial → Oculus viseme (mouth shape of the consonant onset).
_INITIAL_VISEME = {
    "b": "PP", "p": "PP", "m": "PP",
    "f": "FF",
    "d": "DD", "t": "DD",
    "n": "nn", "l": "nn",
    "g": "kk", "k": "kk", "h": "kk",
    "j": "CH", "q": "CH", "x": "CH",
    "zh": "CH", "ch": "CH", "sh": "CH",
    "r": "RR",
    "z": "SS", "c": "SS", "s": "SS",
    "y": "I", "w": "U",  # semivowel onsets (研 yán, 灣 wān) — strict=False keeps them
}

_VOWEL_VISEME = {"a": "aa", "o": "O", "e": "E", "i": "I", "u": "U", "ü": "U", "v": "U"}


def is_chinese(text: str) -> bool:
    """True when a sentence should take the Chinese TTS/lip-sync path."""
    return len(_CJK_RE.findall(text)) > len(_LATIN_RE.findall(text))


def _syllable_visemes(ch: str):
    """Oculus viseme sequence for one hanzi (e.g. 灣/wan → [U, aa, nn])."""
    initial = lazy_pinyin(ch, style=Style.INITIALS, strict=False)[0]
    final   = lazy_pinyin(ch, style=Style.FINALS,   strict=False)[0]
    vis = []
    v = _INITIAL_VISEME.get(initial)
    if v:
        vis.append(v)
    i = 0
    while i < len(final):
        c = final[i]
        if c in _VOWEL_VISEME:
            vis.append(_VOWEL_VISEME[c])
        elif c == "n":
            vis.append("nn")
            if i + 1 < len(final) and final[i + 1] == "g":
                i += 1  # -ng closes the mouth the same way as -n
        elif c == "r":
            vis.append("RR")
        i += 1
    out = []  # collapse repeats (e.g. yi → I,I)
    for v in vis:
        if not out or out[-1] != v:
            out.append(v)
    return out[:4]


def char_segments(text: str, duration_ms: float):
    """Approximate per-hanzi segments when the TTS gives no timestamps.

    Distributes duration_ms over the sentence: each hanzi gets a full slot,
    pause punctuation a fractional silent slot. Returns the same
    [{text, start, end}] shape (seconds) as the Fish Audio alignment.
    """
    slots = []  # (char, weight, is_hanzi)
    for ch in text:
        if _CJK_RE.match(ch):
            slots.append((ch, 1.0, True))
        elif ch in _PAUSE_CHARS:
            slots.append((ch, 0.6, False))
    total_w = sum(w for _, w, _ in slots)
    if not total_w:
        return []
    segments, t = [], 0.0
    per_w = duration_ms / 1000.0 / total_w
    for ch, w, is_hanzi in slots:
        if is_hanzi:
            segments.append({"text": ch, "start": t, "end": t + w * per_w})
        t += w * per_w
    return segments


def visemes_from_segments(segments):
    """Per-character segments (seconds) → (visemes, vtimes, vdurations) in ms.

    Works with exact timestamps (Fish Audio / CosyVoice 2) and with the
    approximated ones from char_segments() alike.
    """
    visemes, vtimes, vdurations = [], [], []
    for seg in segments:
        ch = seg["text"].strip()
        if len(ch) != 1 or not _CJK_RE.match(ch):
            continue
        syl = _syllable_visemes(ch)
        if not syl:
            continue
        t0   = seg["start"] * 1000.0
        slot = max(seg["end"] - seg["start"], 0.05) * 1000.0
        per  = slot / len(syl)
        for j, v in enumerate(syl):
            visemes.append(v)
            vtimes.append(round(t0 + j * per))
            vdurations.append(round(per))
    return visemes, vtimes, vdurations
