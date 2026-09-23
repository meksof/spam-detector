#!/usr/bin/env python3
"""
Local spam classifier powered by a GGUF model served via Ollama.

Provides classify_local(email) which:
- Sends the email content to the local model.
- Parses the free-text response to extract a verdict (spam/ham).
- Prints the verdict and explanation to stdout.
- Returns {"verdict": ..., "explanation": ...}, or None on connection error.
"""

import re
import warnings
from typing import Optional

import ollama

MODEL_NAME = "safe-space-spam-detector"


def _build_prompt(email: dict) -> str:
    """Build a structured user prompt from an email dict."""
    msg = email.get("message", {})
    subject = msg.get("subject", "(no subject)")
    sender_name = msg.get("sender", {}).get("display_name", "")
    sender_email = msg.get("sender", {}).get("email", "")
    body = msg.get("body", "").strip()

    links = msg.get("links", [])
    links_text = ""
    if links:
        link_lines = [
            f"  - [{lk.get('text', '')}]({lk.get('url', '')})" for lk in links[:10]
        ]
        links_text = "\nLinks:\n" + "\n".join(link_lines)

    return (
        f"Subject: {subject}\n"
        f"From: {sender_name} <{sender_email}>\n"
        f"\n{body}"
        f"{links_text}"
    )


def _extract_verdict(text: str) -> str:
    """
    Scan the response text for 'Verdict: spam' or 'Verdict: ham'.
    Falls back to scanning for the bare words if the prefix is absent.
    Returns 'spam', 'ham', or 'unknown'.
    """
    lower = text.lower()

    # Prefer the explicit "Verdict: <label>" pattern
    match = re.search(r"verdict\s*:\s*(spam|ham)", lower)
    if match:
        return match.group(1)

    # Fallback: first occurrence of either word
    spam_pos = lower.find("spam")
    ham_pos = lower.find("ham")

    if spam_pos == -1 and ham_pos == -1:
        return "unknown"
    if spam_pos == -1:
        return "ham"
    if ham_pos == -1:
        return "spam"
    return "spam" if spam_pos < ham_pos else "ham"


def classify_local(email: dict) -> Optional[dict]:
    """
    Classify an email using the local Ollama model.

    Prints the verdict and explanation to stdout.
    Returns {"verdict": str, "explanation": str}, or None if Ollama is unavailable.
    """
    prompt = _build_prompt(email)

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # covers ConnectionError, ResponseError, etc.
        warnings.warn(f"[LOCAL MODEL] Ollama unavailable: {exc}")
        return None

    explanation = response["message"]["content"].strip()
    verdict = _extract_verdict(explanation)

    print(f"       [LOCAL MODEL] Verdict: {verdict.upper()}")
    print(f"       {explanation}\n")

    return {"verdict": verdict, "explanation": explanation}
