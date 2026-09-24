# Implementation Plan V3 — Fine-Tuned Encoder for Borderline Case Classification

## Context and Motivation

### What V2 did

V2 introduced a two-stage pipeline in `spam-detector/`:

1. **TypeSafe** (always runs) — primary classifier, writes verdict to the JSON file.
2. **Local Ollama LLM** (runs on borderline cases only) — provided a second opinion when `spam_signals_fired == SPAM_SIGNAL_THRESHOLD`.

### Why V2 is being replaced

From `observations.md`:

> "It seems that LLM models are better at chat than at classification. And they are not able to provide a good explanation of their classification."

Concretely, the LLM approach had three failure modes on borderline cases:

- **Verdict inconsistency**: generative models tend to produce verbose, ambiguous outputs that require fragile keyword parsing (`"spam"` / `"ham"` scan), making the verdict unreliable.
- **Weak explanations**: the model could not produce structured, signal-level reasoning — it reasoned in natural language, which is hard to act on programmatically or log meaningfully.
- **Latency**: even `tinyllama:latest` took 400ms–13s per email, which is unacceptable for a second-opinion path that should be fast.

### Why the encoder approach is better for this task

A fine-tuned BERT/MiniLM encoder is purpose-built for classification:

| Property | LLM (Ollama) | Fine-tuned encoder (ONNX) |
|---|---|---|
| Output | Free text, requires parsing | Direct logits → probability score |
| Latency | 400ms–13s | 1–5ms (INT8 ONNX, CPU) |
| Footprint | 600MB–10GB | 25–70MB |
| Confidence score | No | Yes — calibrated probability |
| Explanation | Hallucinated prose | Interpretable via confidence + signal correlation |
| Borderline accuracy | Poor (observations.md) | High (fine-tuned on same-domain data) |

The encoder gives a **calibrated spam probability** (0.0–1.0), which is exactly what is needed for borderline cases: not just a verdict, but a confidence level that can be used to decide how aggressively to flag.

---

## Architecture V3

```
classify_emails.py
        │
        ▼
TypeSafe API (always called)
        │
        ▼
spam_signals_fired == SPAM_SIGNAL_THRESHOLD?
        │
   Yes  │              No
   ▼                    ▼
encoder_classifier.py  (nothing extra)
(ONNX, stdout only)
        │
        ▼
   metrics.py
   └─► metrics.csv  (appended — borderline cases only)
```

Key change from V2: `local_classifier.py` (Ollama LLM) is replaced by `encoder_classifier.py` (fine-tuned ONNX encoder).

The rest of the architecture — TypeSafe as primary, borderline trigger, metrics logging — is unchanged.

---

## Goals

1. **Replace** the Ollama LLM second-opinion path in `spam-detector/` with a call to the fine-tuned ONNX encoder from `spam-classifier/`.
2. **Train and export** a MiniLM or DistilBERT encoder fine-tuned on a public spam dataset.
3. **Expose** the encoder as a reusable module (`encoder_classifier.py`) that can be dropped into `spam-detector/` in place of `local_classifier.py`.
4. **Log** the encoder's probability score (not just a binary verdict) to `metrics.csv` for richer training data in future iterations.
5. **Keep** the TypeSafe pipeline and the rest of `classify_emails.py` unchanged.

---

## Task Breakdown

### Task 1 — Acquire and prepare training data

**Objective:** Produce `data/train.csv` and `data/eval.csv` with `text` (str) and `label` (int: 0=ham, 1=spam) columns.

**Recommended sources (public, permissive):**
- [Enron-Spam](http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/) — ~33,000 emails, well-balanced
- [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) — ~6,000 emails
- [Ling-Spam](https://www.kaggle.com/datasets/mandygu/lingspam-dataset) — ~2,400 emails

**Preprocessing rules:**
- Concatenate `subject + " " + body` into the `text` field.
- Strip HTML tags; keep plain text only.
- Truncate to 512 characters (encoder max_length is 256 tokens ≈ ~512 chars).
- 80/20 train/eval split, stratified by label.
- Target: ≥ 5,000 train examples, ≥ 1,000 eval examples.

**Output:**
```
spam-classifier/data/train.csv   (columns: text, label)
spam-classifier/data/eval.csv    (columns: text, label)
```

**Script to create:** `prepare_data.py` — downloads and preprocesses the chosen dataset into the required CSV format.

**Acceptance:** Both CSVs load without error; label distribution printed (target: 40–60% spam in train).

---

### Task 2 — Fine-tune the encoder

**Objective:** Fine-tune a MiniLM-L12 (primary) or DistilBERT (fallback) encoder on the prepared data and save the PyTorch checkpoint.

**Command:**
```bash
python train_spam_classifier.py \
    --train_csv data/train.csv \
    --eval_csv data/eval.csv \
    --model_name microsoft/MiniLM-L12-H384-uncased \
    --output_dir ./model \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5
```

**Expected metrics (on eval set):**
- F1 ≥ 0.95
- Precision ≥ 0.93 (minimise false positives — flagging ham as spam is a worse error)
- Recall ≥ 0.95

**Notes:**
- If the dataset is imbalanced (spam < 30%), add class weighting via a custom `Trainer` subclass that overrides `compute_loss` to apply `torch.nn.CrossEntropyLoss(weight=...)`.
- `train_spam_classifier.py` already exists and is ready to use without modification.

**Output:** `spam-classifier/model/` (PyTorch checkpoint + tokenizer)

**Acceptance:** Final eval metrics printed; F1 ≥ 0.95.

---

### Task 3 — Export and quantize to ONNX INT8

**Objective:** Convert the fine-tuned model to a quantized ONNX file for CPU inference.

**Command:**
```bash
python export_and_quantize.py \
    --model_dir ./model \
    --output_dir ./model-onnx
```

**Output:** `spam-classifier/model-onnx/model_quantized.onnx` + tokenizer files.

**Expected:**
- Model file size ≤ 40MB (MiniLM-L12 INT8)
- Single-email inference ≤ 5ms on CPU

**Notes:** `export_and_quantize.py` already exists and is ready to use without modification.

**Acceptance:** `inference.py --model_dir ./model-onnx --text "test"` runs and returns a result within 5ms.

---

### Task 4 — Create `encoder_classifier.py` in `spam-detector/`

**Objective:** A drop-in replacement for `local_classifier.py` that uses the ONNX encoder instead of the Ollama LLM. Must have the same external interface so `classify_emails.py` requires minimal changes.

**File:** `spam-detector/encoder_classifier.py`

**Interface:**

```python
def classify_encoder(email: dict) -> dict | None:
    """
    Classify a borderline email using the fine-tuned ONNX encoder.

    Returns:
        {
            "verdict": "spam" | "ham",
            "confidence": float,   # spam probability 0.0–1.0
            "explanation": str,    # human-readable summary
        }
        or None on error.
    """
```

**Implementation details:**
- Load the model once at module import time (singleton pattern) to avoid reload overhead on repeated calls.
- Model path: configurable via `ENCODER_MODEL_DIR` environment variable; default: `../fine-tuning/spam-classifier/model-onnx` (relative to `spam-detector/`).
- Build the input text as `subject + " " + body` (same as training preprocessing).
- Run inference with `ORTModelForSequenceClassification` + `AutoTokenizer` + HuggingFace `pipeline`.
- Return `verdict = "spam"` if `confidence > 0.5`, else `"ham"`.
- Print to stdout in the same format as `local_classifier.py` so the terminal output is consistent:
  ```
         [ENCODER] Verdict: SPAM  (confidence: 0.87)
  ```
- Handle import errors (`optimum`, `onnxruntime` not installed) gracefully — warn and return `None`.

**Dependencies to add to `spam-detector/` venv:**
```
optimum[onnxruntime]>=1.20.0
onnxruntime>=1.18.0
transformers>=4.44.0
```

**Acceptance:** `classify_encoder(email_fixture)` returns a dict with `verdict`, `confidence`, and `explanation` keys; prints to stdout; runs in < 10ms.

---

### Task 5 — Update `classify_emails.py` in `spam-detector/`

**Objective:** Replace the `local_classifier` import with `encoder_classifier`. Minimal diff.

**Change:**

```python
# Before (V2)
from local_classifier import classify_local
...
local_result = classify_local(email)

# After (V3)
from encoder_classifier import classify_encoder
...
local_result = classify_encoder(email)
```

No other changes to the classification logic or the TypeSafe path.

**Acceptance:** Running `classify_emails.py` on a borderline email calls `classify_encoder` instead of `classify_local`, prints `[ENCODER]` output to stdout, and logs to `metrics.csv`.

---

### Task 6 — Update `metrics.py` to log confidence score

**Objective:** Capture the encoder's probability score in `metrics.csv`. This is richer than a binary verdict and enables threshold tuning in future iterations.

**Current CSV columns:** `timestamp`, `filename`, `verdict`, `explanation`

**New columns:**

| Column | Value |
|---|---|
| `timestamp` | ISO 8601 UTC timestamp |
| `filename` | JSON filename of the email |
| `typesafe_signals_fired` | int — the TypeSafe signal count that triggered the borderline path |
| `encoder_verdict` | `spam` or `ham` |
| `encoder_confidence` | float 0.0–1.0 — spam probability from the encoder |
| `encoder_explanation` | short string summary (e.g. `"MiniLM-L12 ONNX INT8, confidence=0.87"`) |

**Rename** the existing `verdict` and `explanation` columns to `encoder_verdict` and `encoder_explanation` respectively (breaking change on the CSV schema — document this).

**Acceptance:** After running on a borderline email, `metrics.csv` contains the `encoder_confidence` column with a float value.

---

### Task 7 — Update `README.md` in both projects

**`spam-detector/README.md`:**
- Replace the "Local model setup" section (Ollama, Modelfile) with "Encoder setup" (point to `spam-classifier/` directory, ONNX model path, `ENCODER_MODEL_DIR` env var).
- Update the classification path description to reference the encoder.
- Update the `metrics.csv` column table.
- Remove the `ollama` dependency; add `optimum[onnxruntime]`, `onnxruntime`, `transformers`.

**`spam-classifier/README.md`:**
- Add a section "Integration with spam-detector" explaining that the ONNX model produced by this pipeline is used by `spam-detector/encoder_classifier.py`.

**Acceptance:** Both READMEs accurately describe V3 architecture.

---

## File Changes Summary

| File | Project | Action | Description |
|---|---|---|---|
| `prepare_data.py` | `spam-classifier` | Create | Download + preprocess public spam dataset into train/eval CSVs |
| `data/train.csv` | `spam-classifier` | Create | Training data (≥5000 rows) |
| `data/eval.csv` | `spam-classifier` | Create | Evaluation data (≥1000 rows) |
| `model/` | `spam-classifier` | Create | Fine-tuned PyTorch checkpoint |
| `model-onnx/` | `spam-classifier` | Create | Quantized ONNX model + tokenizer |
| `encoder_classifier.py` | `spam-detector` | Create | Drop-in replacement for `local_classifier.py` using ONNX encoder |
| `classify_emails.py` | `spam-detector` | Modify | Swap `classify_local` import for `classify_encoder` (2-line change) |
| `metrics.py` | `spam-detector` | Modify | Add `typesafe_signals_fired` and `encoder_confidence` columns |
| `README.md` | `spam-detector` | Modify | Document encoder setup, remove Ollama section |
| `README.md` | `spam-classifier` | Modify | Add integration section |

**Files left unchanged:**
- `train_spam_classifier.py` — already correct
- `export_and_quantize.py` — already correct
- `inference.py` — already correct
- `inference_ollama.py` — kept for ad-hoc testing, not part of the main pipeline
- `local_classifier.py` — kept for reference / rollback, not imported anymore
- `questions.json`, `parse_emails.py`, `.env` — unchanged

---

## Dependencies

### `spam-classifier/` (already in `requirements.txt`, no changes needed)
```
transformers>=4.44.0
datasets>=2.20.0
torch>=2.2.0
scikit-learn>=1.4.0
accelerate>=0.30.0
optimum[onnxruntime]>=1.20.0
onnxruntime>=1.18.0
```

### `spam-detector/` (new additions to the venv)
```bash
pip install "optimum[onnxruntime]>=1.20.0" "onnxruntime>=1.18.0" "transformers>=4.44.0"
```

The `ollama` package can be removed from the venv once V3 is validated.

---

## Execution Order

```
Task 1 (prepare data)
    └─► Task 2 (fine-tune)
            └─► Task 3 (export + quantize)
                    └─► Task 4 (encoder_classifier.py)
                            ├─► Task 5 (update classify_emails.py)
                            └─► Task 6 (update metrics.py)
                                    └─► Task 7 (update READMEs)
```

Tasks 5, 6, and 7 can be done in parallel once Task 4 is complete.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Training data too small / unbalanced | Use stratified split; add class weighting if spam < 30% of data |
| F1 below 0.95 target | Try DistilBERT instead of MiniLM-L12; increase epochs to 5 |
| ONNX export fails for the chosen base model | Fall back to raw PyTorch inference (remove ONNX step); latency increases to ~20ms but still far below the LLM baseline |
| `optimum` version conflict with `spam-detector` venv | Pin versions; use a fresh venv if needed |
| Encoder model path wrong at runtime | Make path configurable via `ENCODER_MODEL_DIR` env var with a clear error message |
| Borderline cases in production differ from training distribution | Log all borderline cases to `metrics.csv`; retrain quarterly on accumulated data |

---

## Success Criteria

1. `train_spam_classifier.py` completes with eval **F1 ≥ 0.95** on the held-out eval set.
2. `inference.py` returns a result in **≤ 5ms** on CPU (ONNX INT8 model).
3. `classify_encoder(email)` returns `{"verdict": ..., "confidence": ..., "explanation": ...}` in **< 10ms**.
4. Running `classify_emails.py` on a borderline email prints `[ENCODER]` output to stdout.
5. `metrics.csv` contains `encoder_confidence` (float) after a borderline classification.
6. No regression on non-borderline emails — TypeSafe path behaviour is unchanged.
