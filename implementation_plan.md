# Implementation Plan — Local Model Fallback + Metrics Logging

## Problem Statement

When `SPAM_SIGNAL_THRESHOLD == 2` (the default), the TypeSafe API should be bypassed in favour of a local GGUF model served via Ollama. The local model must produce a verdict, a per-signal breakdown with probabilities, and a human-readable explanation. All threshold-2 classifications are appended to a CSV metrics file for future model evaluation.

---

## Requirements

- Invoke TypeSafe client only when `SPAM_SIGNAL_THRESHOLD != 2`
- When `SPAM_SIGNAL_THRESHOLD == 2`, classify using `model/safe-space-spam-detector.Q2_K.gguf` served via Ollama
- Local model output must include: verdict (spam/ham), per-signal probabilities (same 6 signals as questions.json), and a free-text explanation
- Metrics for threshold-2 cases appended to `metrics.csv`, capturing: filename, timestamp, verdict, explanation, and all 6 signal scores
- The GGUF model is loaded from `./model/` and served by a locally running Ollama server

---

## Background

- The existing `classify()` function in `classify_emails.py` returns `{verdict, spam_signals_fired, signals}` — the local path must produce the same shape, with an added `explanation` field
- The 6 signal IDs and their descriptions already live in `questions.json` — the local classifier should reuse them as prompt context
- Ollama exposes an OpenAI-compatible REST API (`/api/chat` or `/v1/chat/completions`), making structured JSON responses straightforward via the `ollama` Python SDK or `requests`
- The GGUF model needs to be registered with Ollama via a `Modelfile` before it can be invoked by name
- The `classification` block written back to JSON should remain backward-compatible (verdict, spam_signals_fired, signals), with `explanation` added only when the local path is used

---

## Architecture

```
classify_emails.py
        │
        ▼
SPAM_SIGNAL_THRESHOLD == 2?
        │
   Yes  │  No
   ▼         ▼
local_classifier.py    TypeSafe API (existing path)
(Ollama GGUF model)
        │
        ├──► Write classification + explanation to JSON
        │
        └──► metrics.py ──► metrics.csv (appended)
```

The local classifier prompts the model with a structured system prompt (built from the signal descriptions in `questions.json`) and requests a JSON response. A `metrics.py` module handles CSV appending.

---

## Task Breakdown

### Task 1 — Register the GGUF model with Ollama

**Objective:** Make the local model invocable by name through Ollama.

**Implementation:**
- Create a `Modelfile` in `./model/` that references `safe-space-spam-detector.Q2_K.gguf` with a system prompt establishing it as a spam-detection expert
- Document the one-time `ollama create` command in the README

**Test:** Run `ollama run safe-space-spam-detector` and verify it responds to a basic prompt.

**Demo:** The model responds to a manual chat prompt via the Ollama CLI.

---

### Task 2 — Add `ollama` SDK dependency and create `local_classifier.py`

**Objective:** Implement a standalone module that calls the local model and returns the same classification shape as the TypeSafe path, plus an `explanation` field.

**Implementation:**
- Add `ollama` to dependencies
- Create `local_classifier.py` with a `classify_local(questions: dict, email: dict) -> dict` function
- Build a structured system prompt from the signal descriptions in `questions.json`
- Request a JSON response from the model with this schema:
  ```json
  {
    "verdict": "spam" | "ham",
    "explanation": "<free-text reasoning>",
    "signals": {
      "requests_credentials": 0.0,
      "offers_unexpected_reward": 0.0,
      "creates_time_pressure": 0.0,
      "sender_identity_mismatch": 0.0,
      "link_domain_mismatch": 0.0,
      "disguises_link_destination": 0.0
    }
  }
  ```
- Parse and validate the response; fall back gracefully if the model returns malformed JSON
- Compute `spam_signals_fired` from the returned probabilities using the `>= 0.5` rule

**Test:** Write `test_local_classifier.py` using a fixture email (e.g. one of the existing JSON files). Assert the returned dict has all required keys and valid value types.

**Demo:** Running `python local_classifier.py` on a sample email prints a valid classification block.

---

### Task 3 — Create `metrics.py` — CSV metrics logger

**Objective:** Append a row to `metrics.csv` for every email classified by the local model.

**Implementation:**
- Create `metrics.py` with a `log_metric(filename: str, classification: dict) -> None` function
- CSV columns:
  - `timestamp`
  - `filename`
  - `verdict`
  - `spam_signals_fired`
  - `explanation`
  - `requests_credentials`
  - `offers_unexpected_reward`
  - `creates_time_pressure`
  - `sender_identity_mismatch`
  - `link_domain_mismatch`
  - `disguises_link_destination`
- Create the file with headers on first write; append on subsequent runs

**Test:** Write `test_metrics.py` — call `log_metric` twice and assert the CSV has a header row and two data rows with correct values.

**Demo:** Running the logger twice produces a `metrics.csv` with 2 appended rows and correct headers.

---

### Task 4 — Wire the local classifier and metrics logger into `classify_emails.py`

**Objective:** Branch on `SPAM_SIGNAL_THRESHOLD` in the main classification loop — use local model at default (2), TypeSafe otherwise.

**Implementation:**
- In `main()`, check `SPAM_SIGNAL_THRESHOLD == 2` before creating the `TypeSafeClient`
- When `== 2`: call `classify_local()`, then `log_metric()`, then write result to JSON (include `explanation` in the `classification` block)
- When `!= 2`: use the existing `TypeSafeClient` path unchanged
- Ensure the `TypeSafeClient` context manager is only entered when the TypeSafe path is active
- Print the explanation line to stdout under the signal breakdown when the local path is used

**Test:** Update/add an integration test that mocks both the Ollama client and the TypeSafe client, asserting each is called (or not) based on the threshold value.

**Demo:** Running `python classify_emails.py` with default threshold classifies emails via the local model, writes `explanation` into JSON, and appends rows to `metrics.csv`. Changing `SPAM_SIGNAL_THRESHOLD = 3` switches back to TypeSafe.

---

### Task 5 — Update README

**Objective:** Document the two classification paths, the Ollama setup step, and the metrics CSV.

**Implementation:**
- Add a "Local model setup" section with the `ollama create` command
- Update the threshold configuration section to describe both paths
- Document the `metrics.csv` output format and columns

**Demo:** A new developer can follow the README end-to-end to set up and run both paths.

---

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `model/Modelfile` | Create | Ollama model registration for the GGUF file |
| `local_classifier.py` | Create | Local Ollama-based classification logic |
| `metrics.py` | Create | CSV metrics logger for threshold-2 cases |
| `classify_emails.py` | Modify | Branch on threshold; wire local classifier + metrics |
| `test_local_classifier.py` | Create | Unit tests for local classifier |
| `test_metrics.py` | Create | Unit tests for metrics logger |
| `README.md` | Modify | Document local model setup, both paths, metrics CSV |
