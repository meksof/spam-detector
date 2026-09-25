# Spam Detector

A two-step pipeline that converts raw `.eml` email files into structured JSON and classifies each one as **spam** or **ham** using [TypeSafe](https://typesafe.ai)'s Jev model — a System One AI that returns typed, probabilistic judgments instead of free-form text.

---

## How it works

```
emails/*.eml
      │
      ▼  parse_emails.py
emails-json/*.json   (subject, sender, body, links)
      │
      ▼  classify_emails.py  ←  questions.json + TypeSafe API
emails-json/*.json   (+ classification block)
```

### Step 1 — Parse (`parse_emails.py`)

Reads every `.eml` file in `emails/` and extracts:

| Field | Description |
|---|---|
| `message.subject` | Decoded email subject |
| `message.sender.display_name` | Sender's display name |
| `message.sender.email` | Sender's email address |
| `message.body` | Plain-text body (HTML-stripped fallback) |
| `message.links[i].url` | Hyperlink URL extracted from the body |
| `message.links[i].text` | Anchor text of that link |

Output is one `.json` file per email, written to `emails-json/`.

### Step 2 — Classify (`classify_emails.py`)

Reads each JSON file from `emails-json/` and sends the email state to the TypeSafe API. The questions are loaded from `questions.json` — each one is a **Noul** (yes/no probability from 0 to 1). All questions run in a **single parallel request** per email.

| Question ID | What it detects |
|---|---|
| `requests_credentials` | Asks the recipient for a password or login |
| `offers_unexpected_reward` | Claims the recipient won a prize or payment |
| `creates_time_pressure` | Pressures the recipient to act quickly |
| `sender_identity_mismatch` | Display name conflicts with sender domain |
| `link_domain_mismatch` | Link domain conflicts with the claimed sender |
| `disguises_link_destination` | Anchor text hides the real link destination |

A signal **fires** when its probability is ≥ 0.5. An email is classified as **spam** when at least 2 signals fire (configurable via `SPAM_SIGNAL_THRESHOLD` in the script).

The result is written back into the same JSON file under a `"classification"` key:

```json
"classification": {
  "verdict": "spam",
  "spam_signals_fired": 5,
  "signals": {
    "requests_credentials": 0.08,
    "offers_unexpected_reward": 0.96,
    "creates_time_pressure": 0.51,
    "sender_identity_mismatch": 0.92,
    "link_domain_mismatch": 0.90,
    "disguises_link_destination": 0.79
  }
}
```

On a **borderline case** — where `spam_signals_fired` equals `SPAM_SIGNAL_THRESHOLD` exactly — a fine-tuned ONNX encoder is also consulted. Its verdict and confidence score are printed to stdout and appended to `metrics.csv`. The encoder never modifies the JSON output.

---

## Project structure

```
spam-detector/
├── emails-eml/                  # Input: raw .eml files
├── emails-json/                 # Output: parsed + classified JSON files
├── spam-classifier/             # Fine-tuning workspace (encoder training)
│   ├── data/                    # train.csv + eval.csv (generated)
│   ├── spam-model/              # Fine-tuned PyTorch checkpoint (generated)
│   ├── spam-model-onnx/         # Quantized ONNX model + tokenizer (generated)
│   ├── prepare_data.py          # Download + preprocess training data
│   ├── train_spam_classifier.py # Fine-tune MiniLM/DistilBERT
│   ├── export_and_quantize.py   # Export to ONNX INT8
│   ├── requirements.txt         # Fine-tuning dependencies
│   └── README.md
├── params                       # Field spec used for EML parsing
├── questions.json               # TypeSafe Noul questions for classification
├── parse_emails.py              # Step 1: EML → JSON
├── classify_emails.py           # Step 2: JSON → classification via TypeSafe
├── encoder_classifier.py        # ONNX encoder classifier (borderline cases)
├── local_classifier.py          # Ollama classifier (legacy, kept for rollback)
├── metrics.py                   # CSV logger for borderline case metrics
├── metrics.csv                  # Appended at runtime (git-ignored)
├── requirements.txt             # Runtime dependencies
├── .env                         # API key (git-ignored)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file at the project root (or edit the existing one):

```bash
TYPESAFE_API_KEY=your_api_key_here
```

Get your key at [console.typesafe.ai](https://console.typesafe.ai/).

### 4. Set up the encoder (one-time)

The encoder is a fine-tuned MiniLM model exported to ONNX INT8. It lives in `spam-classifier/spam-model-onnx/` and is built by the fine-tuning pipeline in `spam-classifier/`. See `spam-classifier/README.md` for the full training steps.

Once trained and exported, the encoder is loaded automatically from `./spam-classifier/spam-model-onnx/` at runtime. To use a different path:

```bash
export ENCODER_MODEL_DIR=/path/to/spam-model-onnx
```

---

## Usage

### Parse emails

Place `.eml` files in the `emails/` folder, then run:

```bash
python parse_emails.py
```

Each email is converted to a JSON file in `emails-json/`.

### Classify emails

```bash
python classify_emails.py
```

This reads every JSON file in `emails-json/`, calls the TypeSafe API, and writes the `"classification"` result back into each file.

You can run both steps in sequence:

```bash
python parse_emails.py && python classify_emails.py
```

---

## Customising the classifier

### Change the spam threshold

In `classify_emails.py`, adjust how many signals must fire for an email to be considered spam:

```python
SPAM_SIGNAL_THRESHOLD = 2   # default: spam if ≥ 2 signals fire
```

A **borderline case** is any email where `spam_signals_fired` equals `SPAM_SIGNAL_THRESHOLD` exactly — right on the decision boundary. For these emails, the encoder is automatically consulted and its second opinion is printed to stdout and logged to `metrics.csv`.

### Add or modify questions

Edit `questions.json`. Each entry must have a `type` (`"noul"`) and `instructions`. Reference email fields using backtick paths that match the JSON structure (`message.body`, `message.sender.email`, etc.):

```json
{
  "impersonates_bank": {
    "type": "noul",
    "instructions": "Does `message.sender.display_name` or `message.body` impersonate a bank or financial institution?"
  }
}
```

No code changes needed — the classifier loads questions dynamically from the file.

---

## Dependencies

Runtime dependencies for this project are declared in `requirements.txt`:

| Package | Purpose |
|---|---|
| [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/) | TypeSafe System One API client |
| [`python-dotenv`](https://pypi.org/project/python-dotenv/) | Load `TYPESAFE_API_KEY` from `.env` |
| [`optimum[onnxruntime]`](https://pypi.org/project/optimum/) | Load and run the ONNX encoder model |
| [`onnxruntime`](https://pypi.org/project/onnxruntime/) | ONNX runtime for CPU inference |
| [`transformers`](https://pypi.org/project/transformers/) | Tokenizer and pipeline for the encoder |

The standard library covers everything else (`email`, `html.parser`, `json`, `re`, `csv`).

### Fine-tuning workspace (`spam-classifier/`)

The encoder training, evaluation, and ONNX export dependencies are declared separately in `spam-classifier/requirements.txt` (`datasets`, `torch`, `scikit-learn`, `accelerate`, plus the `optimum[onnxruntime]`, `onnxruntime`, and `transformers` trio). See `spam-classifier/README.md` for the setup.

---

## Metrics

Borderline cases are logged to `metrics.csv` at the project root (git-ignored). Each row captures the encoder's second opinion for a case where TypeSafe fired exactly `SPAM_SIGNAL_THRESHOLD` signals.

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 UTC time of classification |
| `filename` | JSON filename of the email |
| `typesafe_signals_fired` | Number of TypeSafe signals that fired (always == `SPAM_SIGNAL_THRESHOLD`) |
| `encoder_verdict` | Encoder verdict: `spam` or `ham` |
| `encoder_confidence` | Spam probability from the encoder: float 0.0–1.0 |
| `encoder_explanation` | Short summary (e.g. `MiniLM ONNX INT8, confidence=0.87`) |

This data can be used to evaluate and improve the encoder over time.
