"""
utils/language.py
-----------------
Language detection using langdetect.
Translation using Ollama (local Llama3) — free, no API key required.
"""

import re
import time
from functools import lru_cache

import requests
from langdetect import detect, LangDetectException

# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

@lru_cache(maxsize=1)
def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

GEMINI_AVAILABLE = False   # kept for backward compat with app.py imports
GEMINI_API_KEY   = ""      # kept for backward compat with app.py imports

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_INDIC_LANGS = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
}

_MAX_TRANSLATION_CHARS = 600
_TRANSLATION_RETRIES = 2

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def is_english(text: str) -> bool:
    return detect_language(text) == "en"

# ---------------------------------------------------------------------------
# Translation using Ollama / Llama3
# ---------------------------------------------------------------------------

def _split_text_for_translation(text: str, max_chars: int = _MAX_TRANSLATION_CHARS) -> list[str]:
    """
    Break long text into paragraph-aware chunks so local LLM translation
    finishes faster and avoids request timeouts.
    """
    normalized = text.strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
    if not paragraphs:
        return [normalized]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= max_chars:
            current = para
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        sentence_chunk = ""
        for sentence in sentences:
            candidate = f"{sentence_chunk} {sentence}".strip() if sentence_chunk else sentence
            if len(candidate) <= max_chars:
                sentence_chunk = candidate
            else:
                if sentence_chunk:
                    chunks.append(sentence_chunk)
                sentence_chunk = sentence
        if sentence_chunk:
            current = sentence_chunk

    if current:
        chunks.append(current)

    return chunks


@lru_cache(maxsize=512)
def _translate_chunk(text: str, target_lang_code: str) -> str:
    """
    Translate one chunk of text to an Indian language using local Ollama.

    Args:
        text:             Text chunk to translate (any language).
        target_lang_code: ISO code like 'te', 'hi', 'ta', etc.

    Returns:
        Translated text string, or original text on failure.
    """
    if not text or not text.strip():
        return text

    if target_lang_code == "en":
        return text

    lang_name = SUPPORTED_INDIC_LANGS.get(target_lang_code, "Telugu")

    prompt = f"""Translate the following text to {lang_name}.
Rules:
- Translate EVERYTHING including headings, bullet points, and labels
- Keep numbers, dates, names, and proper nouns as they are
- Preserve formatting (bullet points, line breaks, bold markers)
- Return ONLY the translated text, nothing else
- Do not add any explanation or preamble

Text to translate:
{text}"""

    for attempt in range(_TRANSLATION_RETRIES + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": max(len(text) * 2, 256),
                    },
                    "keep_alive": "10m",
                },
                timeout=(5, 120),
            )
            response.raise_for_status()
            translated = response.json().get("response", "").strip()
            return translated if translated else text
        except requests.exceptions.ConnectionError:
            return text
        except requests.exceptions.Timeout:
            if attempt < _TRANSLATION_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            return text
        except Exception:
            return text


def translate_to_indic(text: str, target_lang_code: str) -> str:
    """
    Cached public translation helper.

    Repeated Streamlit reruns often request the same translation multiple times,
    so this wrapper keeps identical text+language requests fast.
    """
    if not text or not text.strip() or target_lang_code == "en":
        return text

    chunks = _split_text_for_translation(text)
    if len(chunks) <= 1:
        return _translate_chunk(text, target_lang_code)

    translated_chunks = [
        _translate_chunk(chunk, target_lang_code)
        for chunk in chunks
    ]
    return "\n\n".join(translated_chunks)


def translate_all(texts: dict, target_lang_code: str) -> dict:
    """
    Translate multiple text fields at once.

    Args:
        texts:            Dict of {key: text_to_translate}
        target_lang_code: Target language code

    Returns:
        Dict of {key: translated_text}
    """
    if target_lang_code == "en":
        return texts

    return {key: translate_to_indic(text, target_lang_code) for key, text in texts.items()}
