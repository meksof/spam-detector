#!/usr/bin/env python3
"""
Encoder-based spam classifier for borderline cases.

Drop-in replacement for local_classifier.py. Uses a fine-tuned MiniLM/DistilBERT
encoder exported to ONNX INT8 for fast (~1-5ms) CPU inference.

The model is loaded once at import time (singleton). Subsequent calls to
classify_encoder() reuse the same pipeline with no reload overhead.

Model path resolution (in order of priority):
  1. ENCODER_MODEL_DIR environment variable
  2. Default: ./spam-classifier/model-onnx/ (relative to this file)
"""

import os
import warnings
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy singleton — model loaded once on first import
# ---------------------------------------------------------------------------

_pipeline = None
_model_dir: Optional[Path] = None


def _get_pipeline():
    global _pipeline, _model_dir

    if _pipeline is not None:
        return _pipeline

    # Resolve model directory
    env_path = os.environ.get("ENCODER_MODEL_DIR")
    if env_path:
        _model_dir = Path(env_path)
    else:
        _model_dir = Path(__file__).parent / "spam-classifier" / "model-onnx"

    if not _model_dir.exists():
        warnings.warn(
            f"[ENCODER] Model directory not found: {_model_dir}\n"
            "Run spam-classifier/train_spam_classifier.py then export_and_quantize.py first,\n"
            "or set ENCODER_MODEL_DIR to the correct path."
        )
        return None

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
    except ImportError:
        warnings.warn(
            "[ENCODER] Missing dependencies. Install with:\n"
            '  pip install "optimum[onnxruntime]>=1.20.0" "onnxruntime>=1.18.0" "transformers>=4.44.0"'
        )
        return None

    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            str(_model_dir),
            file_name="model_quantized.onnx",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(_model_dir))
        _pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,          # return scores for all labels
        )
        return _pipeline
    except Exception as exc:
        warnings.warn(f"[ENCODER] Failed to load model from {_model_dir}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_encoder(email: dict) -> Optional[dict]:
    """
    Classify a borderline email using the fine-tuned ONNX encoder.

    Prints the verdict and confidence to stdout.
    Returns {"verdict": str, "confidence": float, "explanation": str},
    or None if the model is unavailable.
    """
    pipe = _get_pipeline()
    if pipe is None:
        return None

    msg = email.get("message", {})
    subject = msg.get("subject", "")
    body = msg.get("body", "")
    text = (subject + " " + body).strip()[:512]

    try:
        results = pipe(text, truncation=True, max_length=256)
    except Exception as exc:
        warnings.warn(f"[ENCODER] Inference error: {exc}")
        return None

    # results is a list of lists: [[{"label": "LABEL_0", "score": ...}, ...]]
    scores = {r["label"]: r["score"] for r in results[0]}

    # LABEL_1 = spam, LABEL_0 = ham (HuggingFace default for binary classifiers)
    spam_score = scores.get("LABEL_1", scores.get("spam", 0.0))
    verdict = "spam" if spam_score > 0.5 else "ham"
    confidence = round(spam_score, 4)
    explanation = f"MiniLM ONNX INT8, confidence={confidence:.4f}"

    print(f"       [ENCODER] Verdict: {verdict.upper()}  (confidence: {confidence:.2f})")

    return {
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
    }
