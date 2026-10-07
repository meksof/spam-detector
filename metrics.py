#!/usr/bin/env python3
"""
Metrics logger for borderline spam classification cases.

A "borderline case" is an email where TypeSafe fired exactly
SPAM_SIGNAL_THRESHOLD signals — right on the spam/ham boundary.
The encoder's second opinion on these cases is appended to
metrics.csv for future model evaluation and training.

CSV schema (V4):
    timestamp               ISO 8601 UTC time of classification
    filename                JSON filename of the email
    typesafe_signals_fired  int — TypeSafe signal count that triggered borderline path
    encoder_verdict         spam or ham
    encoder_confidence      float 0.0–1.0 — spam probability from the encoder
    encoder_latency_ms      float — inference time in milliseconds
    encoder_used_model      short string (e.g. "MiniLM ONNX INT8")

Schema change from V3: 'encoder_latency_ms' column added.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

METRICS_PATH = Path(__file__).parent / "metrics.csv"
CSV_COLUMNS = [
    "timestamp",
    "filename",
    "typesafe_signals_fired",
    "encoder_verdict",
    "encoder_confidence",
    "encoder_latency_ms",
    "encoder_used_model",
]


def log_metric(filename: str, signals_fired: int, classification: dict) -> None:
    """
    Append one row to metrics.csv for a borderline case.

    Args:
        filename:        The JSON filename of the email (e.g. "welcome.json").
        signals_fired:   The TypeSafe spam_signals_fired count (== SPAM_SIGNAL_THRESHOLD).
        classification:  Dict with "verdict", "confidence", and "used_model" keys,
                         as returned by classify_encoder().
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "typesafe_signals_fired": signals_fired,
        "encoder_verdict": classification.get("verdict", "unknown"),
        "encoder_confidence": classification.get("confidence", ""),
        "encoder_latency_ms": classification.get("latency_ms", ""),
        "encoder_used_model": classification.get("used_model", ""),
    }

    file_exists = METRICS_PATH.exists()

    with METRICS_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
