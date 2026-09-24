#!/usr/bin/env python3
"""
Spam classifier powered by TypeSafe (Jev) with encoder second-opinion support.

Reads every JSON file from emails-json/, sends the email state to the
TypeSafe System One API using the questions defined in questions.json,
and writes a "classification" key back into each JSON file.

All questions run in a single parallel request per email.
The spam verdict is derived from the combination of Noul answers.

On a borderline case (spam_signals_fired == SPAM_SIGNAL_THRESHOLD), a
fine-tuned ONNX encoder is also invoked. Its verdict and confidence are
printed to stdout and appended to metrics.csv for future training.
The encoder never modifies the JSON classification block.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient

from encoder_classifier import classify_encoder
from metrics import log_metric

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ.get("TYPESAFE_API_KEY", "")

# An email is considered spam when at least this many signals fire (noul >= 0.5)
SPAM_SIGNAL_THRESHOLD = 2


def load_questions(path: Path) -> dict:
    """Load questions.json and convert to typesafe_sdk Noul objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = {}
    for qid, qdef in raw.items():
        if qdef["type"] == "noul":
            questions[qid] = Noul(instructions=qdef["instructions"])
        # extend here if Choice / Score types are ever added to questions.json
    return questions


def classify(client: TypeSafeClient, questions: dict, email: dict) -> dict:
    """Send one request with all questions and return a structured result."""
    # State mirrors the paths referenced in questions.json (message.body, etc.)
    state = {"message": email["message"]}

    response = client.system_one(state=state, questions=questions)

    signals = {}
    fired = 0
    for qid in questions:
        prob = round(response.answers[qid].noul, 4)
        signals[qid] = prob
        if prob >= 0.5:
            fired += 1

    verdict = "spam" if fired >= SPAM_SIGNAL_THRESHOLD else "ham"

    return {
        "verdict": verdict,
        "spam_signals_fired": fired,
        "signals": signals,
    }


def main():
    root = Path(__file__).parent
    questions_path = root / "questions.json"
    emails_json_dir = root / "emails-json"

    questions = load_questions(questions_path)
    json_files = sorted(emails_json_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found in emails-json/")
        return

    print(f"Classifying {len(json_files)} email(s) with {len(questions)} signal(s)...\n")

    with TypeSafeClient(api_key=API_KEY) as client:
        for path in json_files:
            try:
                # If email already has a classification, skip it
                email = json.loads(path.read_text(encoding="utf-8"))
                if "classification" in email:
                    print(f"✓ {path.name}: Already classified")
                    continue

                result = classify(client, questions, email)
                email["classification"] = result
                path.write_text(
                    json.dumps(email, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                icon = "🚨" if result["verdict"] == "spam" else "✅"
                fired = result["spam_signals_fired"]
                total = len(questions)
                print(f"{icon} [{result['verdict'].upper():4s}]  {fired}/{total} signals  {path.name}")
                for sig, prob in result["signals"].items():
                    marker = "●" if prob >= 0.5 else "○"
                    print(f"       {marker} {sig}: {prob:.2f}")
                print()

                # Borderline case: fire encoder for a second opinion + log metrics
                if fired == SPAM_SIGNAL_THRESHOLD:
                    print(f"       ⚠️  Borderline case ({fired}/{total} signals) — consulting encoder...")
                    local_result = classify_encoder(email)
                    if local_result:
                        log_metric(path.name, fired, local_result)

            except Exception as exc:
                print(f"✗  {path.name}: {exc}\n")

    print(f"Done — results written to {emails_json_dir}/")


if __name__ == "__main__":
    main()
