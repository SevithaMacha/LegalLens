"""
utils/ocr.py
------------
Handles text extraction from images and PDF files.

Strategy:
  1. Try Tesseract OCR first (fast, works for printed text).
  2. If Tesseract output looks poor (too short / garbled),
     fall back to Google Gemini Vision (handles handwriting + Indian languages).
  3. For PDFs: native text extraction first, then page-by-page OCR if scanned.
"""

import io
import re
import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Gemini setup (optional — only used if API key is available)
# ---------------------------------------------------------------------------

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Supported Tesseract languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGS = [
    "eng", "hin", "tel", "tam", "kan", "mal", "ben", "mar", "guj", "pan", "ori",
]


def _get_tesseract_lang_string() -> str:
    try:
        available = pytesseract.get_languages(config="")
        langs_to_use = [l for l in SUPPORTED_LANGS if l in available]
        return "+".join(langs_to_use) if langs_to_use else "eng"
    except Exception:
        return "eng+hin+tel+tam+kan+mal"


def _is_poor_ocr(text: str) -> bool:
    if len(text.strip()) < 30:
        return True
    non_standard = re.findall(r"[^\x00-\x7F\u0900-\u0DFF\u0A00-\u0A7F]", text)
    ratio = len(non_standard) / max(len(text), 1)
    return ratio > 0.4


def _tesseract_extract(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    lang_string = _get_tesseract_lang_string()
    try:
        return pytesseract.image_to_string(
            image, config=f"--oem 3 --psm 6 -l {lang_string}"
        ).strip()
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(
            image, config="--oem 3 --psm 6 -l eng"
        ).strip()


def _gemini_extract(image: Image.Image) -> str:
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return ""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """You are an expert OCR system specialized in legal documents.
Extract ALL text from this image exactly as written.
- Preserve the original language (Telugu, Hindi, Tamil, English, etc.)
- Keep paragraph structure and line breaks
- Include all words, numbers, dates, and signatures
- If the text is handwritten, transcribe it carefully
- Do not translate or summarize — extract the raw text only"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        response = model.generate_content(
            [prompt, {"mime_type": "image/png", "data": img_byte_arr.read()}]
        )
        return response.text.strip()
    except Exception:
        return ""


def extract_text_from_image(image: Image.Image, force_gemini: bool = False) -> str:
    if force_gemini:
        gemini_text = _gemini_extract(image)
        return gemini_text if gemini_text else _tesseract_extract(image)
    tesseract_text = _tesseract_extract(image)
    if _is_poor_ocr(tesseract_text) and GEMINI_AVAILABLE and GEMINI_API_KEY:
        gemini_text = _gemini_extract(image)
        if gemini_text and len(gemini_text) > len(tesseract_text):
            return gemini_text
    return tesseract_text


def extract_text_from_pdf(pdf_bytes: bytes, force_gemini: bool = False) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip() and not force_gemini:
            all_text.append(text.strip())
        else:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            all_text.append(extract_text_from_image(image, force_gemini=force_gemini))
    doc.close()
    return "\n\n".join(all_text)


def extract_text(uploaded_file, force_gemini: bool = False) -> str:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes, force_gemini=force_gemini)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        return extract_text_from_image(image, force_gemini=force_gemini)


def get_ocr_status() -> dict:
    return {
        "tesseract": True,
        "gemini": GEMINI_AVAILABLE and bool(GEMINI_API_KEY),
        "gemini_package": GEMINI_AVAILABLE,
        "gemini_key_set": bool(GEMINI_API_KEY),
    }