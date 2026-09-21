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

---

## Project structure

```
spam-detector/
├── emails-eml/                  # Input: raw .eml files
├── emails-json/             # Output: parsed + classified JSON files
├── params                   # Field spec used for EML parsing
├── questions.json           # TypeSafe Noul questions for classification
├── parse_emails.py          # Step 1: EML → JSON
├── classify_emails.py       # Step 2: JSON → classification via TypeSafe
├── .env                     # API key (git-ignored)
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
pip install typesafe-sdk python-dotenv
```

### 3. Configure your API key

Create a `.env` file at the project root (or edit the existing one):

```bash
TYPESAFE_API_KEY=your_api_key_here
```

Get your key at [console.typesafe.ai](https://console.typesafe.ai/).

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

| Package | Purpose |
|---|---|
| [`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/) | TypeSafe System One API client |
| [`python-dotenv`](https://pypi.org/project/python-dotenv/) | Load `TYPESAFE_API_KEY` from `.env` |

Both are standard library-free additions; everything else (`email`, `html.parser`, `json`, `re`) is built into Python 3.
