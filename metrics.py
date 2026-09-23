#!/usr/bin/env python3
"""
Metrics logger for borderline spam classification cases.

A "borderline case" is an email where TypeSafe fired exactly
SPAM_SIGNAL_THRESHOLD signals — right on the spam/ham boundary.
The local model's second opinion on these cases is appended to
metrics.csv for future model evaluation and training.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

METRICS_PATH = Path(__file__).parent / "metrics.csv"
CSV_COLUMNS = ["timestamp", "filename", "verdict", "explanation"]


def log_metric(filename: str, classification: dict) -> None:
    """
    Append one row to metrics.csv for a borderline case.

    Args:
        filename:       The JSON filename of the email (e.g. "welcome.json").
        classification: Dict with at least "verdict" and "explanation" keys,
                        as returned by classify_local().
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "verdict": classification.get("verdict", "unknown"),
        "explanation": classification.get("explanation", ""),
    }

    file_exists = METRICS_PATH.exists()

    with METRICS_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
