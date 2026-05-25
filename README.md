# ⚖️ LegalLens — Multilingual AI Legal Document Assistant

> BVRIT Hyderabad · Department of CSE · 2026
> Team: P. Mounika · M. Bhavishya · M. Sevitha  
> Guide: Ms. G. Rashmitha

---

## 📌 Overview

LegalLens is an intelligent, multilingual system that:

1. **Extracts** text from scanned/printed legal documents (PDF or image) via **Tesseract OCR**
2. **Segments** documents into logical clauses
3. **Indexes** clauses using **FAISS** vector similarity search with **SentenceTransformer** embeddings
4. **Simplifies** complex legal language with a local **LLM (Llama 3 via Ollama)**
5. **Detects risk alerts** — unusual penalties, rights restrictions, hidden fees
6. **Assists form filling** with step-by-step guidance
7. **Supports multilingual output** in English, Telugu, and Hindi

---

## 🗂️ Project Structure

```
legallens/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
└── utils/
    ├── __init__.py
    ├── ocr.py              # OCR extraction (Tesseract + PyMuPDF)
    ├── language.py         # Language detection & IndicTrans2 translation
    ├── rag_engine.py       # Clause segmentation, FAISS index, semantic search
    └── llm.py              # Ollama/Llama3 LLM interactions
```

---

## ⚙️ Setup Instructions

### Step 1 — Install System Dependencies

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel

# macOS
brew install tesseract
```

### Step 2 — Install Ollama & Pull Llama 3

```bash
# Install Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the Llama 3 model
ollama pull llama3

# Start Ollama server (keep this running in a terminal)
ollama serve
```

### Step 3 — Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Optional — IndicTrans2 (for Indian language output):**
> ```bash
> pip install IndicTransToolkit transformers sentencepiece
> ```

### Step 5 — Run the Application

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔄 System Workflow

```
User Uploads Document (Image / PDF)
           │
           ▼
    OCR Processing (Tesseract)
           │
           ▼
     Extracted Text
           │
    ┌──────┴──────┐
    │             │
Is English?      No → Translate to English (IndicTrans2)
    │Yes          │
    └──────┬──────┘
           ▼
   Clause Segmentation
           │
           ▼
  SentenceTransformer Embeddings
           │
           ▼
    FAISS Vector Database
           │
           ▼
  Retrieve Relevant Clauses
           │
           ▼
   LLM Simplification (Llama 3)
           │
           ▼
  Translate to User Language
           │
     ┌─────┴──────┐
     ▼            ▼
Simplified    Risk Alerts
Explanation
```

---

## 📦 Technology Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| OCR | Tesseract OCR + PyTesseract |
| PDF Handling | PyMuPDF (fitz) |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | FAISS (faiss-cpu) |
| LLM | Ollama + Llama 3 |
| Language Detection | langdetect |
| Indian Translation | IndicTrans2 (optional) |
| Image Handling | Pillow (PIL) |
| Numerics | NumPy |

---

## 🌐 Supported Languages

English, Telugu, Hindi

---

## 📚 References

- [arxiv.org/pdf/2512.18004](https://www.arxiv.org/pdf/2512.18004)
- [FAISS](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
- [Ollama](https://ollama.com/)
