"""
app.py
------
LegalLens – Multilingual AI Legal Document Assistant
"""

import streamlit as st

from utils.ocr import extract_text, get_ocr_status
from utils.language import (
    detect_language,
    translate_to_indic,
    SUPPORTED_INDIC_LANGS,
    _ollama_available,
)
from utils.rag_engine import build_index_from_text, segment_clauses
from utils.llm import (
    simplify_clause,
    answer_question,
    detect_risk_alerts,
    form_filling_guide,
    summarize_document,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LegalLens",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(15,52,96,0.4);
}
.main-header h1 { color: #e2c97e; font-size: 3rem; letter-spacing: 2px; margin: 0; }
.main-header p { color: #a8b2d8; font-size: 1.1rem; margin-top: 0.5rem; }
.step-card {
    background: #f8f9ff; border-left: 4px solid #0f3460;
    border-radius: 8px; padding: 1rem 1.5rem; margin: 0.8rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.alert-box {
    background: #fff3cd; border-left: 4px solid #e67e22;
    border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem 0;
}
.success-box {
    background: #d4edda; border-left: 4px solid #27ae60;
    border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem 0;
}
.clause-card {
    background: white; border: 1px solid #e0e0e0; border-radius: 10px;
    padding: 1rem; margin: 0.5rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.stButton > button {
    background: linear-gradient(135deg, #0f3460, #16213e);
    color: #e2c97e; border: none; border-radius: 8px;
    padding: 0.6rem 1.8rem; font-weight: 600; font-size: 0.95rem;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(15,52,96,0.4); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>⚖️ LegalLens</h1>
    <p>Multilingual AI Legal Document Assistant · Extract · Analyse · Simplify</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

CACHE_VERSION = 2

if st.session_state.get("cache_version") != CACHE_VERSION:
    for key in [
        "translation_cache",
        "answer_cache",
        "simplified_cache",
        "form_guide_cache",
        "translated_risk_alerts",
    ]:
        st.session_state.pop(key, None)
    st.session_state["cache_version"] = CACHE_VERSION

for key in ["extracted_text", "summary", "risk_alerts", "clauses", "faiss_index"]:
    if key not in st.session_state:
        if key in ["extracted_text", "summary"]:
            st.session_state[key] = ""
        elif key == "faiss_index":
            st.session_state[key] = None
        else:
            st.session_state[key] = []

for key in [
    "translation_cache",
    "answer_cache",
    "simplified_cache",
    "form_guide_cache",
    "translated_risk_alerts",
]:
    if key not in st.session_state:
        st.session_state[key] = {}

# ---------------------------------------------------------------------------
# Helper — translate with spinner
# ---------------------------------------------------------------------------

def translate(text: str, lang_code: str, label: str = "") -> str:
    """Translate text only once per language/content pair."""
    if lang_code == "en" or not text:
        return text

    cache_key = (lang_code, text)
    cached = st.session_state.translation_cache.get(cache_key)
    if cached is not None:
        return cached

    spinner_label = label or "text"
    with st.spinner(f"Translating {spinner_label} to {output_lang}…"):
        translated = translate_to_indic(text, lang_code)
    st.session_state.translation_cache[cache_key] = translated
    return translated


def translate_many(texts: list[str], lang_code: str, label: str = "content") -> list[str]:
    """Translate a list of texts while reusing cached results."""
    if lang_code == "en":
        return texts
    return [translate(text, lang_code, label) for text in texts]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    output_lang = st.selectbox(
        "Output Language",
        options=["English"] + list(SUPPORTED_INDIC_LANGS.values()),
        index=0,
        help="All answers, summaries, and explanations will be shown in this language.",
    )

    lang_code_map = {v: k for k, v in SUPPORTED_INDIC_LANGS.items()}
    selected_lang_code = lang_code_map.get(output_lang, "en")

    # Translation status — now uses Ollama instead of Gemini
    if selected_lang_code != "en":
        if _ollama_available():
            st.success(f"✅ Translation to {output_lang} enabled (Ollama/Llama3)")
        else:
            st.error(
                f"❌ Ollama is not running.\n\n"
                "Start it with:\n"
                "`ollama serve`"
            )

    st.divider()
    st.markdown("## 📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a legal document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Supports scanned images, handwritten docs, and PDFs.",
    )

    # OCR status
    ocr_status = get_ocr_status()
    if ocr_status["gemini"]:
        st.success("✅ Gemini Vision ready\n(handwriting supported)")
    else:
        st.info("ℹ️ Tesseract OCR active\n(printed text only)")

    # Handwriting toggle
    handwriting_mode = st.toggle(
        "✍️ Handwritten Document",
        value=False,
        help="Enable for handwritten documents.",
        disabled=not ocr_status["gemini"],
    )

    if uploaded_file and st.button("🔍 Extract & Analyse", use_container_width=True):

        # Reset
        for key in ["extracted_text", "summary", "risk_alerts", "clauses", "faiss_index"]:
            if key in ["extracted_text", "summary"]:
                st.session_state[key] = ""
            elif key == "faiss_index":
                st.session_state[key] = None
            else:
                st.session_state[key] = []

        for key in [
            "answer_cache",
            "simplified_cache",
            "form_guide_cache",
            "translated_risk_alerts",
        ]:
            st.session_state[key] = {}

        # 1. Extract text
        with st.spinner("Extracting text via OCR…"):
            try:
                text = extract_text(uploaded_file, force_gemini=handwriting_mode)
                st.session_state.extracted_text = text
            except Exception as e:
                st.error(f"OCR failed: {e}")
                st.stop()

        if not st.session_state.extracted_text.strip():
            st.warning("No text could be extracted. Please try a clearer image.")
            st.stop()

        # 2. Build FAISS index
        with st.spinner("Building semantic search index…"):
            clauses = segment_clauses(st.session_state.extracted_text)
            st.session_state.clauses = clauses
            st.session_state.faiss_index = build_index_from_text(
                st.session_state.extracted_text
            )

        # 3. Generate summary in English first
        with st.spinner("Generating document summary…"):
            summary_en = summarize_document(st.session_state.extracted_text)
            st.session_state.summary = summary_en

        # 4. Risk alerts in English first
        with st.spinner("Scanning for risk alerts…"):
            st.session_state.risk_alerts = detect_risk_alerts(clauses)

        st.success(f"✅ Done! Found **{len(clauses)}** clauses.")

    st.divider()
    st.markdown("<small>BVRIT Hyderabad · Dept. of CSE · 2026</small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Summary", "🔎 Ask a Question",
    "📑 Clause Browser", "⚠️ Risk Alerts", "📝 Form Help"
])

# ── TAB 1: Summary ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Document Summary")

    if not st.session_state.extracted_text:
        st.info("Upload and analyse a document to see the summary here.")
    else:
        # Detected language
        lang = detect_language(st.session_state.extracted_text)
        lang_label = (
            translate("📌 Detected Language:", selected_lang_code, "label")
            if selected_lang_code != "en"
            else "📌 Detected Language:"
        )
        st.markdown(
            f'<div class="step-card">{lang_label} <b>{lang.upper()}</b></div>',
            unsafe_allow_html=True,
        )

        # Translate summary
        summary = st.session_state.summary
        if selected_lang_code != "en":
            if not _ollama_available():
                st.error("❌ Ollama is not running. Start it with: `ollama serve`")
            else:
                translated = translate(summary, selected_lang_code, "summary")
                if translated and translated != summary:
                    summary = translated
                else:
                    st.warning("⚠️ Translation did not return a result. Showing English.")

        st.markdown(
            f'<div class="step-card">{summary}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📄 View Raw Extracted Text"):
            st.text_area("Extracted Text", value=st.session_state.extracted_text, height=300, disabled=True)

# ── TAB 2: Q&A ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Ask About Your Document")

    if not st.session_state.faiss_index:
        st.info("Upload and analyse a document first.")
    else:
        placeholder = "e.g. What are my termination rights? What fees apply?"
        if selected_lang_code != "en":
            hint_translated = translate(
                "Type your question in English or Telugu — both work!",
                selected_lang_code, "hint"
            )
            st.info(hint_translated)

        query = st.text_input("Your question", placeholder=placeholder)

        if st.button("💬 Get Answer") and query.strip():
            cache_key = (query.strip(), selected_lang_code)
            cached_payload = st.session_state.answer_cache.get(cache_key)

            if cached_payload is None:
                with st.spinner("Searching relevant clauses…"):
                    results = st.session_state.faiss_index.search(query, top_k=5)
                    context_clauses = [r[0] for r in results]

                with st.spinner("Generating answer…"):
                    answer_en = answer_question(query, context_clauses)

                answer = (
                    translate(answer_en, selected_lang_code, "answer")
                    if selected_lang_code != "en"
                    else answer_en
                )
                cached_payload = {"answer": answer, "results": results}
                st.session_state.answer_cache[cache_key] = cached_payload
            else:
                answer = cached_payload["answer"]
                results = cached_payload["results"]

            st.markdown(f"#### ✅ {'సమాధానం' if selected_lang_code == 'te' else 'Answer'}")
            st.markdown(f'<div class="success-box">{answer}</div>', unsafe_allow_html=True)

            st.markdown("#### 🔗 Retrieved Context Clauses")
            for i, (clause, dist) in enumerate(results, 1):
                with st.expander(f"Clause {i}  (relevance: {1/(1+dist):.2f})"):
                    st.write(clause)

# ── TAB 3: Clause Browser ───────────────────────────────────────────────────
with tab3:
    st.markdown("### Browse & Simplify Clauses")

    if not st.session_state.clauses:
        st.info("Upload and analyse a document first.")
    else:
        total_label = translate(
            f"Total clauses extracted: {len(st.session_state.clauses)}",
            selected_lang_code, "label"
        )
        st.markdown(f"**{total_label}**")

        clause_labels = [
            f"Clause {i+1}: {c[:80]}…" if len(c) > 80 else f"Clause {i+1}: {c}"
            for i, c in enumerate(st.session_state.clauses)
        ]

        selected_idx = st.selectbox(
            "Select a clause",
            options=range(len(clause_labels)),
            format_func=lambda i: clause_labels[i],
        )

        selected_clause = st.session_state.clauses[selected_idx]

        original_label = translate("Original Clause:", selected_lang_code, "label")
        st.markdown(f"**{original_label}**")
        st.markdown(f'<div class="clause-card">{selected_clause}</div>', unsafe_allow_html=True)

        btn_label = translate("✨ Simplify This Clause", selected_lang_code, "button")
        if st.button(btn_label):
            cache_key = (selected_clause, selected_lang_code)
            simplified = st.session_state.simplified_cache.get(cache_key)

            if simplified is None:
                with st.spinner("Simplifying…"):
                    simplified_en = simplify_clause(selected_clause)

                simplified = (
                    translate(simplified_en, selected_lang_code, "simplified clause")
                    if selected_lang_code != "en"
                    else simplified_en
                )
                st.session_state.simplified_cache[cache_key] = simplified

            simplified_label = translate(f"Simplified ({output_lang}):", selected_lang_code, "label")
            st.markdown(f"**{simplified_label}**")
            st.markdown(f'<div class="success-box">{simplified}</div>', unsafe_allow_html=True)

# ── TAB 4: Risk Alerts ──────────────────────────────────────────────────────
with tab4:
    title = translate("### ⚠️ Risk Alerts", selected_lang_code, "title")
    st.markdown(title)

    desc = translate(
        "Automatically detected clauses that may contain risks, hidden penalties, or unfair provisions.",
        selected_lang_code, "description"
    )
    st.markdown(desc)

    if not st.session_state.extracted_text:
        st.info(translate("Upload and analyse a document first.", selected_lang_code))
    elif not st.session_state.risk_alerts:
        st.success(translate("✅ No significant risks detected in this document.", selected_lang_code))
    else:
        count_msg = translate(
            f"{len(st.session_state.risk_alerts)} potential risk(s) found:",
            selected_lang_code, "count"
        )
        st.warning(f"**{count_msg}**")

        if selected_lang_code != "en":
            translated_alerts = st.session_state.translated_risk_alerts.get(selected_lang_code)
            if translated_alerts is None:
                translated_alerts = translate_many(
                    st.session_state.risk_alerts,
                    selected_lang_code,
                    "risk alert",
                )
                st.session_state.translated_risk_alerts[selected_lang_code] = translated_alerts
        else:
            translated_alerts = st.session_state.risk_alerts

        for translated_alert in translated_alerts:
            st.markdown(
                f'<div class="alert-box">⚠️ {translated_alert}</div>',
                unsafe_allow_html=True,
            )

# ── TAB 5: Form Help ────────────────────────────────────────────────────────
with tab5:
    form_title = translate("### 📝 Form Filling Assistant", selected_lang_code, "title")
    st.markdown(form_title)

    form_desc = translate(
        "Paste a form field label or instruction below and get step-by-step guidance on how to fill it correctly.",
        selected_lang_code, "description"
    )
    st.markdown(form_desc)

    field_text = st.text_area(
        translate("Form field / instruction text", selected_lang_code, "label"),
        placeholder="e.g. 'Affiant's full legal name as per Aadhaar'",
        height=120,
    )

    btn_text = translate("📋 Get Filling Guide", selected_lang_code, "button")
    if st.button(btn_text) and field_text.strip():
        cache_key = (field_text.strip(), selected_lang_code)
        guide = st.session_state.form_guide_cache.get(cache_key)

        if guide is None:
            with st.spinner("Generating guidance…"):
                guide_en = form_filling_guide(field_text)

            guide = (
                translate(guide_en, selected_lang_code, "guide")
                if selected_lang_code != "en"
                else guide_en
            )
            st.session_state.form_guide_cache[cache_key] = guide

        answer_label = translate("✅ How to Fill This Field", selected_lang_code, "label")
        st.markdown(f"#### {answer_label}")
        st.markdown(f'<div class="success-box">{guide}</div>', unsafe_allow_html=True)
