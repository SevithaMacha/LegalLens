"""
utils/llm.py
------------
Handles all interactions with the local LLM via Ollama (llama3 model).

Provides:
  - simplify_clause()    – Simplify a legal clause in plain language.
  - answer_question()    – Answer a user query given retrieved context clauses.
  - detect_risk_alerts() – Flag potentially risky clauses.
  - form_filling_guide() – Provide guidance on filling a detected form field.
"""

import ollama
from typing import List

MODEL = "llama3"  # Change to "llama3:8b" or "mistral" if needed


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _chat(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """
    Send a chat message to Ollama and return the assistant reply.

    Args:
        system_prompt:  Instructions for the LLM role.
        user_message:   The actual user input / task.
        temperature:    Sampling temperature (lower = more deterministic).

    Returns:
        LLM response as a plain string.
    """
    try:
        response = ollama.chat(
            model=MODEL,
            options={"temperature": temperature},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM Error] {str(e)}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simplify_clause(clause: str) -> str:
    """
    Convert a complex legal clause into plain, easy-to-understand language.

    Args:
        clause: Raw legal clause text.

    Returns:
        Simplified explanation string.
    """
    system = (
        "You are a legal simplification expert. "
        "Your task is to rewrite legal clauses in simple, clear language "
        "that a non-lawyer can easily understand. "
        "Avoid jargon. Use short sentences. "
        "Preserve all important meaning. "
        "Respond ONLY with the simplified explanation – no preamble."
    )
    return _chat(system, f"Simplify this legal clause:\n\n{clause}")


def answer_question(query: str, context_clauses: List[str]) -> str:
    """
    Answer a user's question using the retrieved context clauses.

    Args:
        query:           The user's question about the document.
        context_clauses: Top-k relevant clauses from FAISS retrieval.

    Returns:
        LLM-generated answer string.
    """
    context = "\n\n---\n\n".join(context_clauses)
    system = (
        "You are a knowledgeable legal assistant. "
        "Answer the user's question based ONLY on the provided document context. "
        "If the answer is not in the context, say so clearly. "
        "Be concise, accurate, and use plain language."
    )
    user_msg = (
        f"Document context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Please answer based on the context above."
    )
    return _chat(system, user_msg)


def detect_risk_alerts(clauses: List[str]) -> List[str]:
    """
    Scan clauses for potentially risky, unfair, or important provisions.

    Args:
        clauses: List of document clauses.

    Returns:
        List of risk alert strings (may be empty if none found).
    """
    system = (
        "You are a legal risk analyst. "
        "Review the following clauses and identify any that: "
        "(1) impose unusual liabilities or penalties, "
        "(2) restrict important rights, "
        "(3) contain hidden fees or auto-renewals, "
        "(4) include one-sided arbitration clauses, "
        "(5) waive important legal protections. "
        "For each risky clause, output a concise one-sentence risk alert. "
        "If no risks are found, output: NO_RISKS_FOUND"
    )
    combined = "\n\n".join(clauses[:20])  # limit to first 20 clauses
    response = _chat(system, f"Clauses to review:\n\n{combined}", temperature=0.2)

    if "NO_RISKS_FOUND" in response:
        return []

    # Split response into individual alerts
    alerts = [line.strip() for line in response.split("\n") if line.strip()]
    # Filter out empty lines and numbering artifacts
    alerts = [a.lstrip("0123456789.-) ") for a in alerts if len(a) > 10]
    return alerts


def form_filling_guide(field_text: str) -> str:
    """
    Provide step-by-step guidance on filling a legal form field.

    Args:
        field_text: The form field label or instruction text.

    Returns:
        Plain-language guidance string.
    """
    system = (
        "You are a helpful legal form assistant. "
        "Explain clearly how to fill in the given form field. "
        "Mention what information is required, acceptable formats, "
        "and any common mistakes to avoid. "
        "Use simple language and bullet points."
    )
    return _chat(system, f"How should I fill in this form field?\n\n{field_text}")


def summarize_document(text: str) -> str:
    """
    Generate a concise summary of the entire legal document.

    Args:
        text: Full document text (will be truncated if very long).

    Returns:
        Summary string.
    """
    system = (
        "You are a legal document summarizer. "
        "Provide a clear, structured summary of the document. "
        "Include: document type, main parties involved (if any), "
        "key obligations, important dates or deadlines, and main terms. "
        "Use plain language. Format with clear sections."
    )
    # Truncate to avoid token limits
    truncated = text[:6000] if len(text) > 6000 else text
    return _chat(system, f"Summarize this legal document:\n\n{truncated}")
