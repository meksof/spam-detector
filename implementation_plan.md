# Implementation Plan — Local Model Pipeline + Metrics Logging

## Problem Statement

Extend the spam classifier with two behaviours:

1. **TypeSafe always runs first**: the existing TypeSafe classification path is unchanged and is always executed.
2. **Sequential local model invocation**: on a **borderline case**, the local GGUF model is also invoked after TypeSafe and prints its verdict and explanation to stdout. The local model never writes to the JSON file.
3. **Metrics logging for borderline cases**: every borderline case is appended to `metrics.csv` so the data can later be used to improve the local model.

---

## Architecture

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
local_classifier.py   (nothing extra)
(Ollama, stdout only)
        │
        ▼
   metrics.py
   └─► metrics.csv  (appended — borderline cases only)
```

### Key design decisions

- **TypeSafe always runs.** It is the authoritative classifier and always writes the `classification` block to the JSON file. This is unchanged from the existing behaviour.
- **Local model triggers on borderline cases.** The result is right on the boundary between spam and ham, making it the most valuable case for a second opinion and for training data.
- **The local model is stdout-only.** It never writes to the JSON file. Its verdict and explanation are printed to the terminal and captured in `metrics.csv` only.
- **The local model is already expert** on spam detection. No system-prompt engineering is needed beyond providing the email content.
- The GGUF model (`model/safe-space-spam-detector.Q2_K.gguf`) is served by a locally running Ollama instance. It must be registered once via a `Modelfile` before use.

---

## Task Breakdown

### Task 1 — Register the GGUF model with Ollama

**Objective:** Make the local model invocable by name through Ollama.

**Files:**
- Create `model/Modelfile`

**Implementation:**
- Write a minimal `Modelfile` that points `FROM` at the local GGUF path and sets a concise `SYSTEM` prompt establishing the role ("You are a spam detection expert. …").
- Document the one-time setup command in the README:
  ```bash
  ollama create safe-space-spam-detector -f model/Modelfile
  ```

**Acceptance:** `ollama run safe-space-spam-detector "Is this spam?"` returns a response without error.

---

### Task 2 — Create `local_classifier.py`

**Objective:** Implement a module that calls the local Ollama model and returns a structured result suitable for writing to JSON and/or printing to stdout.

**Files:**
- Create `local_classifier.py`

**Implementation:**
- Add `ollama` to the project dependencies (`pip install ollama`).
- Define `classify_local(email: dict) -> dict` that:
  1. Builds a user prompt from the email's subject, sender, body, and links.
  2. Calls `ollama.chat(model="safe-space-spam-detector", messages=[...])`.
  3. Parses the model's free-text response to extract:
     - `verdict`: `"spam"` or `"ham"` (scan response for these keywords; default to `"unknown"` if absent).
     - `explanation`: the full response text.
  4. Returns `{"verdict": verdict, "explanation": explanation}`.
- Print the result to stdout in a readable format (used by both the sequential and local-only paths).
- Handle Ollama connection errors gracefully (log warning, return `None` so the caller can skip).

**Acceptance:** Calling `classify_local(email)` with a fixture email dict returns a dict with `verdict` and `explanation` keys and prints to stdout.

---

### Task 3 — Create `metrics.py`

**Objective:** Append a row to `metrics.csv` for every borderline case, capturing the local model's second opinion for future training.

**Files:**
- Create `metrics.py`

**Implementation:**
- Define `log_metric(filename: str, classification: dict) -> None` that writes to `metrics.csv`.
- CSV columns:
  | Column | Value |
  |---|---|
  | `timestamp` | ISO 8601 UTC timestamp of classification |
  | `filename` | JSON filename of the email |
  | `verdict` | `spam` or `ham` from the local model |
  | `explanation` | Full free-text explanation from the local model |
- Create the file with a header row on first write; append on all subsequent runs.

**Acceptance:** Calling `log_metric` twice produces a `metrics.csv` with one header row and two data rows.

---

### Task 4 — Update `classify_emails.py`

**Objective:** Wire the two new modules into the main classification loop. TypeSafe always runs; the local model fires only on borderline results.

**Files:**
- Modify `classify_emails.py`

**Implementation:**

```
for each email JSON file:

    result = classify(client, questions, email)   # TypeSafe (always)
    write result to JSON classification block     # unchanged

    if result["spam_signals_fired"] == SPAM_SIGNAL_THRESHOLD:
        local_result = classify_local(email)      # local model, stdout only
        log_metric(filename, local_result)        # append to metrics.csv
```

- The `TypeSafeClient` context manager wraps all emails as before.
- The local model is invoked *after* the JSON file has already been written, so a failure in the local path does not affect the TypeSafe output.
- The local model's output is printed to stdout under the existing TypeSafe signal breakdown, clearly labelled (e.g. `[LOCAL MODEL]`). Nothing is added to the JSON file.

**Acceptance:**
- TypeSafe is called for every email regardless of threshold value.
- On a **borderline case**: local model runs, prints to stdout, `metrics.csv` appended.
- On a non-borderline case: local model is not called, `metrics.csv` not touched.

---

### Task 5 — Update `README.md`

**Objective:** Document the new classification paths, Ollama setup, and metrics output.

**Files:**
- Modify `README.md`

**Sections to add/update:**
- **Local model setup**: one-time `ollama create` command and requirement for a running Ollama server.
- **Classification paths**: table or description of which path runs under which threshold value.
- **Metrics logging**: describe `metrics.csv`, its columns, and its purpose (future local model improvement).
- **Dependencies**: add `ollama` to the dependencies table.

---

## File Changes Summary

| File | Action | Description |
|---|---|---|
| `model/Modelfile` | Create | Ollama model registration pointing at the local GGUF file |
| `local_classifier.py` | Create | Ollama-based classification; returns verdict + explanation; prints to stdout only |
| `metrics.py` | Create | CSV logger for borderline cases |
| `classify_emails.py` | Modify | After TypeSafe result, fire local model + metrics when signals hit threshold exactly |
| `README.md` | Modify | Document Ollama setup, local model trigger condition, metrics CSV |

---

## Dependencies

| Package | Purpose | Install |
|---|---|---|
| `ollama` | Python client for the local Ollama server | `pip install ollama` |

Existing dependencies (`typesafe-sdk`, `python-dotenv`) are unchanged.

---

## Open Questions / Assumptions

- **Ollama server** must already be running locally (`ollama serve`). The plan does not auto-start it.
- **Verdict extraction** from the local model's free-text response uses keyword scanning (`"spam"` / `"ham"` present in the response). If the model's output format is inconsistent, a more robust parser (regex, JSON mode) can be added in a follow-up.
- **`metrics.csv` location**: project root, git-ignored. Add `metrics.csv` to `.gitignore`.
- **Local model stdout format**: the verdict and explanation are printed after the TypeSafe signal breakdown for each borderline email, clearly labelled (e.g. `[LOCAL MODEL]`).

## Acronyms

SST: SPAM_SIGNAL_THRESHOLD

## Definitions

**Borderline case**: an email whose TypeSafe classification produced exactly `spam_signals_fired == SPAM_SIGNAL_THRESHOLD` fired signals — i.e. the result sits right on the decision boundary between spam and ham. These cases are the most ambiguous and therefore the most valuable for local model evaluation and future training.
