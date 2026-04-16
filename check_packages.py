packages = {
    "streamlit": "streamlit",
    "pytesseract": "pytesseract",
    "Pillow": "PIL",
    "faiss-cpu": "faiss",
    "sentence-transformers": "sentence_transformers",
    "numpy": "numpy",
    "langdetect": "langdetect",
    "ollama": "ollama",
    "PyMuPDF": "fitz",
}

print("Checking packages...\n")
all_ok = True
for pkg_name, import_name in packages.items():
    try:
        __import__(import_name)
        print(f"  ✅  {pkg_name}")
    except ImportError:
        print(f"  ❌  {pkg_name}  → run: pip install {pkg_name}")
        all_ok = False

print("\n✅ All good!" if all_ok else "\n⚠️ Install missing packages above.")
